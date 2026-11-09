//! Bound every structured loop of a SPIR-V module.
//!
//! A guest shader's loop runs until data the GPU reads ends it (a list
//! walked until an entry's w is not positive, say). When that data is not
//! what the console holds, the loop never ends, the GPU job runs past the
//! kernel's limit and amdgpu resets the whole GPU (the user's desktop with
//! it). Every loop header here counts its executions in one per-invocation
//! (Private) variable shared by all the module's loops, and each loop
//! leaves through its merge block once that count passes `LIMIT`: a wrong
//! result for that invocation instead of a reset. One shared budget, so
//! nested loops cannot multiply it.
//!
//! The bound acts at the loop's existing conditional exits (branches with
//! one edge leaving the loop, directly or through a block that only breaks
//! out), so the control flow keeps its shape and dominance holds. Loops
//! without such an exit are left as they are.

/// Loop header executions an invocation may make, all loops together,
/// before every loop exits: low enough that loops walking wrong data
/// across hundreds of workgroups end well inside the kernel's ten-second
/// job limit (four million per loop, multiplied by nesting, did not, and
/// reset the GPU).
pub const LIMIT: u32 = 1 << 16;

const OP_FUNCTION: u32 = 54;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_IADD: u32 = 128;
const OP_LOGICAL_OR: u32 = 166;
const OP_LOGICAL_AND: u32 = 167;
const OP_LOGICAL_NOT: u32 = 168;
const OP_UGREATER_THAN: u32 = 172;
const OP_LOOP_MERGE: u32 = 246;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const STORAGE_PRIVATE: u32 = 6;
const OP_ENTRY_POINT: u32 = 15;

fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
    let mut words = Vec::with_capacity(operands.len() + 1);
    words.push(((operands.len() as u32 + 1) << 16) | opcode);
    words.extend_from_slice(operands);
    words
}

/// `spirv` with every guardable loop bounded; unchanged if it does not
/// parse or has no loops.
pub fn guard_loops(spirv: &[u32]) -> Vec<u32> {
    guard(spirv).unwrap_or_else(|| spirv.to_vec())
}

fn guard(spirv: &[u32]) -> Option<Vec<u32>> {
    if spirv.len() < 5 || spirv[0] != 0x0723_0203 {
        return None;
    }
    // Split into instructions.
    let mut instructions: Vec<Vec<u32>> = Vec::new();
    let mut at = 5;
    while at < spirv.len() {
        let count = (spirv[at] >> 16) as usize;
        if count == 0 || at + count > spirv.len() {
            return None;
        }
        instructions.push(spirv[at..at + count].to_vec());
        at += count;
    }
    let opcode = |words: &Vec<u32>| words[0] & 0xffff;
    if !instructions
        .iter()
        .any(|words| opcode(words) == OP_LOOP_MERGE)
    {
        return None;
    }
    let mut bound = spirv[3];
    let mut fresh = || {
        let id = bound;
        bound += 1;
        id
    };
    let first_function = instructions
        .iter()
        .position(|words| opcode(words) == OP_FUNCTION)?;
    // Types and constants, reused when declared (types must be unique).
    let mut globals: Vec<Vec<u32>> = Vec::new();
    let find = |predicate: &dyn Fn(&Vec<u32>) -> bool| -> Option<u32> {
        instructions[..first_function]
            .iter()
            .find(|words| predicate(words))
            .map(|words| words[1])
    };
    let uint = match find(&|words| {
        opcode(words) == OP_TYPE_INT && words.len() == 4 && words[2] == 32 && words[3] == 0
    }) {
        Some(id) => id,
        None => {
            let id = fresh();
            globals.push(instruction(OP_TYPE_INT, &[id, 32, 0]));
            id
        }
    };
    let boolean = match find(&|words| opcode(words) == OP_TYPE_BOOL) {
        Some(id) => id,
        None => {
            let id = fresh();
            globals.push(instruction(OP_TYPE_BOOL, &[id]));
            id
        }
    };
    let pointer = match find(&|words| {
        opcode(words) == OP_TYPE_POINTER
            && words.len() == 4
            && words[2] == STORAGE_PRIVATE
            && words[3] == uint
    }) {
        Some(id) => id,
        None => {
            let id = fresh();
            globals.push(instruction(OP_TYPE_POINTER, &[id, STORAGE_PRIVATE, uint]));
            id
        }
    };
    let mut constant = |value: u32, globals: &mut Vec<Vec<u32>>| {
        let id = fresh();
        globals.push(instruction(OP_CONSTANT, &[uint, id, value]));
        id
    };
    let zero = constant(0, &mut globals);
    let one = constant(1, &mut globals);
    let limit = constant(LIMIT, &mut globals);
    // The budget every loop header draws on, per invocation.
    let counter = fresh();
    globals.push(instruction(
        OP_VARIABLE,
        &[pointer, counter, STORAGE_PRIVATE, zero],
    ));

    // Block boundaries: label index -> (start, end exclusive) in
    // `instructions`, and the label id of each block.
    let mut blocks: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    let mut start = None;
    for (index, words) in instructions.iter().enumerate() {
        match opcode(words) {
            OP_LABEL => start = Some((words[1], index)),
            OP_BRANCH | OP_BRANCH_CONDITIONAL | 251 /* switch */ | 252 /* kill */
            | 253 /* return */ | 254 /* return value */ | 255 /* unreachable */
            | 4416 /* terminate invocation */ => {
                if let Some((label, begin)) = start.take() {
                    blocks.insert(label, (begin, index + 1));
                }
            }
            _ => {}
        }
    }

    // Edits: instructions inserted before an index and replaced
    // terminators.
    let mut insert_before: std::collections::BTreeMap<usize, Vec<Vec<u32>>> = Default::default();
    let mut replace: std::collections::HashMap<usize, Vec<u32>> = Default::default();

    // Blocks ending in an unconditional branch to `target`.
    let breaks_to = |label: u32, target: u32| {
        blocks.get(&label).is_some_and(|&(_, end)| {
            let last = &instructions[end - 1];
            opcode(last) == OP_BRANCH && last[1] == target
        })
    };
    for (index, words) in instructions.iter().enumerate() {
        if opcode(words) != OP_LOOP_MERGE {
            continue;
        }
        let merge = words[1];
        let Some(&(merge_start, _)) = blocks.get(&merge) else {
            continue;
        };
        // The loop's conditional exits: branches between the header and
        // the merge block with an edge to it, or to a block that only
        // breaks to it. Each also exits once the count is exceeded; the
        // control flow keeps its shape, so dominance holds.
        let mut exits = Vec::new();
        for (branch, words) in instructions
            .iter()
            .enumerate()
            .take(merge_start)
            .skip(index + 1)
        {
            if opcode(words) != OP_BRANCH_CONDITIONAL {
                continue;
            }
            let leaves = |target: u32| target == merge || breaks_to(target, merge);
            if leaves(words[2]) && !leaves(words[3]) {
                exits.push((branch, true));
            } else if leaves(words[3]) && !leaves(words[2]) {
                exits.push((branch, false));
            }
        }
        if exits.is_empty() {
            continue;
        }
        let loaded = fresh();
        let added = fresh();
        insert_before.entry(index).or_default().extend([
            instruction(OP_LOAD, &[uint, loaded, counter]),
            instruction(OP_IADD, &[uint, added, loaded, one]),
            instruction(OP_STORE, &[counter, added]),
        ]);
        for (branch, true_exits) in exits {
            let words = &instructions[branch];
            let count = fresh();
            let exceeded = fresh();
            let mut before = vec![
                instruction(OP_LOAD, &[uint, count, counter]),
                instruction(OP_UGREATER_THAN, &[boolean, exceeded, count, limit]),
            ];
            let condition = fresh();
            if true_exits {
                before.push(instruction(
                    OP_LOGICAL_OR,
                    &[boolean, condition, words[1], exceeded],
                ));
            } else {
                let not = fresh();
                before.push(instruction(OP_LOGICAL_NOT, &[boolean, not, exceeded]));
                before.push(instruction(
                    OP_LOGICAL_AND,
                    &[boolean, condition, words[1], not],
                ));
            }
            // The header's own exit sits after its OpLoopMerge, which must
            // stay right before the branch: insert before the merge
            // instruction there.
            let at = if branch == index + 1 { index } else { branch };
            insert_before.entry(at).or_default().extend(before);
            replace.insert(
                branch,
                instruction(OP_BRANCH_CONDITIONAL, &[condition, words[2], words[3]]),
            );
        }
    }
    if replace.is_empty() {
        return None;
    }
    // From SPIR-V 1.4, an entry point lists every global variable it uses.
    if spirv[1] >= 0x0001_0400 {
        for words in instructions.iter_mut() {
            if opcode(words) == OP_ENTRY_POINT {
                words.push(counter);
                words[0] = ((words.len() as u32) << 16) | OP_ENTRY_POINT;
            }
        }
    }

    let mut out = Vec::with_capacity(spirv.len() + 64 * replace.len());
    out.extend_from_slice(&spirv[..5]);
    for (index, words) in instructions.iter().enumerate() {
        if index == first_function {
            for global in &globals {
                out.extend_from_slice(global);
            }
        }
        if let Some(inserted) = insert_before.get(&index) {
            for words in inserted {
                out.extend_from_slice(words);
            }
        }
        if let Some(replacement) = replace.get(&index) {
            out.extend_from_slice(replacement);
        } else {
            out.extend_from_slice(words);
        }
    }
    out[3] = bound;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modules_without_loops_or_unparsable_input_are_unchanged() {
        // Header only (no instructions): nothing to guard.
        let module = vec![0x0723_0203, 0x0001_0000, 0, 8, 0];
        assert_eq!(guard_loops(&module), module);
        let garbage = vec![1, 2, 3];
        assert_eq!(guard_loops(&garbage), garbage);
    }
}
