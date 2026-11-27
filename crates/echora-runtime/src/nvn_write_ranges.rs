use std::ops::Range;

/// A Vulkan device address interval backed by a continuous range in a
/// guest memory file. Different Vulkan imports may name the same bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedSpan {
    pub device: Range<u64>,
    pub file: i32,
    pub offset: u64,
}

/// One continuous stretch of GPU virtual address space backed by a
/// continuous range in a guest memory file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedPages {
    gpu: Range<u64>,
    max_gpu_end: u64,
    pub file: i32,
    pub offset: u64,
}

/// The file bytes a GPU write may touch through the current page table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRange {
    pub file: i32,
    pub bytes: Range<u64>,
}

/// Length of the first run of holes or consecutive device pages in each
/// block. A page-table patch only invalidates the blocks it touches.
pub(super) struct PageRunHints(Vec<u16>);

impl PageRunHints {
    const BLOCK: usize = 256;

    pub(super) fn new(entries: &[u64], page_size: u64) -> Self {
        Self(
            entries
                .chunks(Self::BLOCK)
                .map(|chunk| Self::prefix(chunk, page_size))
                .collect(),
        )
    }

    fn prefix(chunk: &[u64], page_size: u64) -> u16 {
        let step = if chunk[0] == 0 { 0 } else { page_size };
        (1 + chunk
            .windows(2)
            .take_while(|pair| pair[0].checked_add(step) == Some(pair[1]))
            .count()) as u16
    }

    /// Resize after the table's existing prefix was preserved and any new
    /// pages were zero-filled. Only the old last block can mix old and new pages.
    pub(super) fn resize(&mut self, previous: usize, entries: &[u64], page_size: u64) {
        if previous == entries.len() {
            return;
        }
        self.0
            .resize(entries.len().div_ceil(Self::BLOCK), Self::BLOCK as u16);
        if entries.len() > previous && !previous.is_multiple_of(Self::BLOCK) {
            let block = previous / Self::BLOCK;
            let start = block * Self::BLOCK;
            self.0[block] = Self::prefix(
                &entries[start..(start + Self::BLOCK).min(entries.len())],
                page_size,
            );
        }
        let tail = entries.len() % Self::BLOCK;
        if tail != 0 {
            let last = &mut self.0[entries.len() / Self::BLOCK];
            *last = (*last).min(tail as u16);
        }
    }

    pub(super) fn update(&mut self, entries: &[u64], page_size: u64, patches: &[Range<u64>]) {
        let mut next = 0;
        for patch in patches.iter().filter(|p| !p.is_empty()) {
            let start = (patch.start as usize / Self::BLOCK).max(next);
            let end = (patch.end as usize).div_ceil(Self::BLOCK);
            for index in start..end {
                let at = index * Self::BLOCK;
                self.0[index] = Self::prefix(
                    &entries[at..(at + Self::BLOCK).min(entries.len())],
                    page_size,
                );
            }
            next = end;
        }
    }

    #[inline]
    fn advance(&self, page: usize) -> usize {
        if page.is_multiple_of(Self::BLOCK) {
            usize::from(self.0[page / Self::BLOCK])
        } else {
            1
        }
    }
}

#[cfg(test)]
pub(super) fn project_table(
    first_gpu: u64,
    page_size: u64,
    entries: &[u64],
    imported: Vec<ImportedSpan>,
) -> Option<Vec<SharedPages>> {
    project_table_with_hints(
        first_gpu,
        page_size,
        entries,
        &PageRunHints::new(entries, page_size),
        imported,
    )
}

pub(super) fn project_table_with_hints(
    first_gpu: u64,
    page_size: u64,
    entries: &[u64],
    hints: &PageRunHints,
    mut imported: Vec<ImportedSpan>,
) -> Option<Vec<SharedPages>> {
    imported.sort_unstable_by_key(|span| span.device.start);
    let mut furthest = 0;
    let maximum_ends: Vec<_> = imported
        .iter()
        .map(|span| {
            furthest = furthest.max(span.device.end);
            furthest
        })
        .collect();
    let mut mapped = Vec::new();
    let mut max_gpu_end = 0;
    let mut page = 0;
    while let Some(&address) = entries.get(page) {
        let first = page;
        let mut end = address;
        while entries.get(page) == Some(&end) {
            let count = hints.advance(page);
            page += count;
            if address != 0 {
                end = end.checked_add((count as u64).checked_mul(page_size)?)?;
            }
        }
        if address == 0 {
            continue;
        }
        let gpu = first_gpu.checked_add((first as u64).checked_mul(page_size)?)?;
        let since = maximum_ends.partition_point(|&end| end <= address);
        let until = imported.partition_point(|span| span.device.start < end);
        for span in &imported[since..until] {
            let start = span.device.start.max(address);
            let stop = span.device.end.min(end);
            if start < stop {
                let gpu = gpu.checked_add(start - address)?..gpu.checked_add(stop - address)?;
                max_gpu_end = max_gpu_end.max(gpu.end);
                mapped.push(SharedPages {
                    gpu,
                    max_gpu_end,
                    file: span.file,
                    offset: span.offset.checked_add(start - span.device.start)?,
                });
            }
        }
    }
    Some(mapped)
}

/// Count the calls, input writes, file ranges before merging and mapping
/// spans walked for this backend's current profiling interval.
pub(super) fn project_writes(
    mapping: &[SharedPages],
    writes: &[Range<u64>],
    counters: &std::cell::Cell<[u64; 4]>,
) -> Option<Vec<FileRange>> {
    let mut physical: Vec<FileRange> = Vec::new();
    let mut walked = 0u64;
    for write in writes {
        // Page-table runs and the imports within each run both start in
        // ascending order. The prefix maximum also accounts for nested
        // imports whose end precedes an earlier import's end.
        let first = mapping.partition_point(|span| span.max_gpu_end <= write.start);
        for span in mapping[first..]
            .iter()
            .take_while(|span| span.gpu.start < write.end)
        {
            walked += 1;
            let lo = span.gpu.start.max(write.start);
            let hi = span.gpu.end.min(write.end);
            if lo < hi {
                let start = span.offset.checked_add(lo - span.gpu.start)?;
                let end = span.offset.checked_add(hi - span.gpu.start)?;
                // Contiguous imports often project adjacent bytes of the
                // same file even when their Vulkan address ranges differ.
                if let Some(last) = physical.last_mut()
                    && last.file == span.file
                    && start <= last.bytes.end
                    && end >= last.bytes.start
                {
                    last.bytes.start = last.bytes.start.min(start);
                    last.bytes.end = last.bytes.end.max(end);
                } else {
                    physical.push(FileRange {
                        file: span.file,
                        bytes: start..end,
                    });
                }
            }
        }
    }
    let [calls, inputs, outputs, spans] = counters.get();
    counters.set([
        calls + 1,
        inputs + writes.len() as u64,
        outputs + physical.len() as u64,
        spans + walked,
    ]);
    physical.sort_unstable_by_key(|range| (range.file, range.bytes.start));
    physical.dedup_by(|next, previous| {
        if previous.file == next.file && next.bytes.start <= previous.bytes.end {
            previous.bytes.end = previous.bytes.end.max(next.bytes.end);
            true
        } else {
            false
        }
    });
    Some(physical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resized_page_runs_match_a_full_scan_across_partial_blocks() {
        let page = 4096;
        let mut entries = Vec::new();
        let mut hints = PageRunHints::new(&entries, page);
        for size in [257, 258, 300, 900, 511, 256, 255, 256, 1024, 1, 0, 517] {
            let previous = entries.len();
            entries.resize(size, 0);
            hints.resize(previous, &entries, page);
            assert_eq!(hints.0, PageRunHints::new(&entries, page).0);
            // Alternate holes and consecutive device pages before the next resize.
            for (i, value) in entries.iter_mut().enumerate() {
                *value = if (i / PageRunHints::BLOCK + size).is_multiple_of(3) {
                    0
                } else {
                    (1 + i as u64) * page
                };
            }
            hints.update(&entries, page, std::slice::from_ref(&(0..size as u64)));
        }
    }

    #[test]
    fn page_run_hints_follow_holes_remaps_and_short_final_blocks() {
        let page = 16;
        let mut entries: Vec<_> = (0..803).map(|index| 0x2000 + index * page).collect();
        let mut hints = PageRunHints::new(&entries, page);
        let imports = vec![
            ImportedSpan {
                device: 0x2000..0x8000,
                file: 1,
                offset: 12,
            },
            ImportedSpan {
                device: 0x2ff0..0x3800,
                file: 2,
                offset: 0,
            },
        ];
        let scalar = PageRunHints(vec![1; hints.0.len()]);
        let check = |entries: &[u64], hints: &PageRunHints| {
            assert_eq!(
                project_table_with_hints(0x1500, page, entries, hints, imports.clone()),
                project_table_with_hints(0x1500, page, entries, &scalar, imports.clone()),
            );
        };
        check(&entries, &hints);
        for (start, end, base) in [
            (16, 255, 0),
            (255, 513, 0),
            (400, 770, 0x3000),
            (766, 803, 0),
            (0, 256, 0),
            (0, 17, 0x4000),
        ] {
            for (i, entry) in entries[start..end].iter_mut().enumerate() {
                *entry = if base == 0 {
                    0
                } else {
                    base + (i as u64) * page
                };
            }
            // Separate modified intervals can fall into the same hint block.
            let mid = (start + end) / 2;
            hints.update(
                &entries,
                page,
                &[start as u64..mid as u64, mid as u64..end as u64],
            );
            assert_eq!(hints.0, PageRunHints::new(&entries, page).0);
            check(&entries, &hints);
        }
        entries[800] = u64::MAX;
        hints.update(&entries, page, &[800..801]);
        assert!(project_table_with_hints(0, page, &entries, &hints, imports).is_none());
    }

    #[test]
    fn projects_aliases_holes_and_noncontiguous_pages() {
        let imports = [
            ImportedSpan {
                device: 400..500,
                file: 1,
                offset: 1000,
            },
            ImportedSpan {
                device: 800..920,
                file: 1,
                offset: 1000,
            },
            ImportedSpan {
                device: 920..1000,
                file: 1,
                offset: 5000,
            },
        ];
        let map = project_table(
            100,
            20,
            &[800, 820, 0, 400, 420, 0, 500, 900, 920],
            imports.to_vec(),
        )
        .unwrap();
        let bounds = [105..111, 170..194, 220..270, 700..10000];
        let writes = project_writes(&map, &bounds, &Default::default()).unwrap();
        assert_eq!(
            writes,
            [
                FileRange {
                    file: 1,
                    bytes: 1005..1034
                },
                FileRange {
                    file: 1,
                    bytes: 1100..1120
                },
                FileRange {
                    file: 1,
                    bytes: 5000..5010
                },
            ]
        );
        assert_eq!(
            project_writes(&map, &[140..159], &Default::default()).unwrap(),
            []
        );
        // A different Vulkan buffer and GPU virtual address can write
        // the same guest file bytes.
        let aliased = project_writes(&map, &[165..175], &Default::default()).unwrap();
        assert_eq!(
            aliased,
            [FileRange {
                file: 1,
                bytes: 1005..1015
            }]
        );
    }

    #[test]
    fn projects_writes_past_the_end_of_a_nested_import() {
        let mapping = project_table(
            100,
            20,
            &[800, 820, 840, 860],
            vec![
                ImportedSpan {
                    device: 800..880,
                    file: 2,
                    offset: 1000,
                },
                ImportedSpan {
                    device: 820..840,
                    file: 1,
                    offset: 2000,
                },
            ],
        )
        .unwrap();
        let writes = project_writes(&mapping, &[150..170, 110..160], &Default::default()).unwrap();
        assert_eq!(
            writes,
            [
                FileRange {
                    file: 1,
                    bytes: 2000..2020,
                },
                FileRange {
                    file: 2,
                    bytes: 1010..1070,
                },
            ]
        );
    }

    #[test]
    fn projected_writes_match_a_bytewise_reference() {
        use std::collections::BTreeSet;
        let imports = vec![
            ImportedSpan {
                device: 256..320,
                file: 1,
                offset: 1000,
            },
            ImportedSpan {
                device: 512..576,
                file: 1,
                offset: 1016,
            },
            ImportedSpan {
                device: 768..784,
                file: 1,
                offset: 3000,
            },
            ImportedSpan {
                device: 784..832,
                file: 1,
                offset: 6000,
            },
        ];
        let mut state = 1u32;
        let mut next = |limit| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 11) as u64 % limit
        };
        let page = 16;
        let first = 1024;
        for _ in 0..80 {
            let entries: Vec<_> = (0..29)
                .map(|_| {
                    let base = [0, 256, 512, 768, 1024][next(5) as usize];
                    if base == 0 { 0 } else { base + next(4) * page }
                })
                .collect();
            let writes: Vec<_> = (0..12)
                .map(|_| {
                    let start = 1000 + next(551);
                    start..start + next(95)
                })
                .collect();
            let mapping = project_table(first, page, &entries, imports.clone()).unwrap();
            let actual: BTreeSet<_> = project_writes(&mapping, &writes, &Default::default())
                .unwrap()
                .into_iter()
                .flat_map(|range| range.bytes.map(move |offset| (range.file, offset)))
                .collect();
            let mut expected = BTreeSet::new();
            for va in writes.into_iter().flatten() {
                let Some(relative) = va.checked_sub(first) else {
                    continue;
                };
                let Some(&bda) = entries.get((relative / page) as usize) else {
                    continue;
                };
                if bda != 0 {
                    let byte = bda + relative % page;
                    for span in imports.iter().filter(|span| span.device.contains(&byte)) {
                        expected.insert((span.file, span.offset + byte - span.device.start));
                    }
                }
            }
            assert_eq!(actual, expected);
        }
    }
}
