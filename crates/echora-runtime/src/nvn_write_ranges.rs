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
    pub gpu: Range<u64>,
    pub file: i32,
    pub offset: u64,
}

/// The file bytes a GPU write may touch through the current page table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRange {
    pub file: i32,
    pub bytes: Range<u64>,
}

pub fn project_table(
    first_gpu: u64,
    page_size: u64,
    entries: &[u64],
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
    let mut pages = entries.iter().copied().enumerate().peekable();
    while let Some((first, address)) = pages.next() {
        if address == 0 {
            continue;
        }
        let mut last = first + 1;
        let mut end = address.checked_add(page_size)?;
        while pages.next_if(|&(_, next)| next == end).is_some() {
            end = end.checked_add(page_size)?;
            last += 1;
        }
        let gpu = first_gpu.checked_add((first as u64).checked_mul(page_size)?)?;
        debug_assert_eq!(end - address, (last - first) as u64 * page_size);
        let since = maximum_ends.partition_point(|&end| end <= address);
        let until = imported.partition_point(|span| span.device.start < end);
        for span in &imported[since..until] {
            let start = span.device.start.max(address);
            let stop = span.device.end.min(end);
            if start < stop {
                mapped.push(SharedPages {
                    gpu: gpu.checked_add(start - address)?..gpu.checked_add(stop - address)?,
                    file: span.file,
                    offset: span.offset.checked_add(start - span.device.start)?,
                });
            }
        }
    }
    Some(mapped)
}

pub fn project_writes(mapping: &[SharedPages], writes: &[Range<u64>]) -> Option<Vec<FileRange>> {
    let mut physical = Vec::new();
    for write in writes {
        for span in mapping.iter().take_while(|span| span.gpu.start < write.end) {
            let lo = span.gpu.start.max(write.start);
            let hi = span.gpu.end.min(write.end);
            if lo < hi {
                physical.push(FileRange {
                    file: span.file,
                    bytes: span.offset.checked_add(lo - span.gpu.start)?
                        ..span.offset.checked_add(hi - span.gpu.start)?,
                });
            }
        }
    }
    physical.sort_unstable_by_key(|range| (range.file, range.bytes.start));
    let mut merged: Vec<FileRange> = Vec::with_capacity(physical.len());
    for range in physical {
        match merged.last_mut() {
            Some(last) if range.file == last.file && range.bytes.start <= last.bytes.end => {
                last.bytes.end = last.bytes.end.max(range.bytes.end);
            }
            _ => merged.push(range),
        }
    }
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let writes = project_writes(&map, &bounds).unwrap();
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
        assert_eq!(project_writes(&map, &[140..159]).unwrap(), []);
        // A different Vulkan buffer and GPU virtual address can write
        // the same guest file bytes.
        let aliased = project_writes(&map, &[165..175]).unwrap();
        assert_eq!(
            aliased,
            [FileRange {
                file: 1,
                bytes: 1005..1015
            }]
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
            let actual: BTreeSet<_> = project_writes(&mapping, &writes)
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
