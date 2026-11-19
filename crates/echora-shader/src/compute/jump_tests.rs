use super::{Layout, blocks, blocks_for_targets, emit_blocks, structure};
use crate::decode::{Instruction, Op, Operand, Pred, Reg, Src};
use crate::jump_tables::JumpTables;
use crate::jump_tables::JumpTargets;
use crate::sim::{Inputs, Machine};

fn immediate(n: u32) -> Operand {
    Operand {
        src: Src::Imm(n),
        abs: false,
        neg: false,
    }
}

fn register(n: u8) -> Operand {
    Operand {
        src: Src::Reg(Reg(n)),
        abs: false,
        neg: false,
    }
}

fn word(op: Op) -> Instruction {
    Instruction {
        raw: 0,
        guard: Pred::TRUE,
        op,
        uniform: false,
    }
}

fn branch(from: i64, to: i64) -> Op {
    Op::Bra {
        offset: (to - from - 1) * 16,
        mode: 0,
        condition: Pred::TRUE,
    }
}

fn program() -> Vec<Instruction> {
    [
        Op::S2r {
            d: Reg(0),
            special: 0x21,
        },
        Op::Imnmx {
            d: Reg(1),
            a: register(0),
            b: immediate(2),
            min: Pred::TRUE,
            signed: false,
        },
        Op::Shf {
            d: Reg(1),
            low: Reg(1),
            shift: immediate(2),
            high: register(255),
            kind: 3,
            wrap: false,
            right: false,
            high_word: false,
        },
        Op::Ldc {
            d: Reg(2),
            index: Reg(1),
            bank: 1,
            offset: 0,
            size: 4,
            mode: 0,
        },
        Op::Brx {
            target: Src::Reg(Reg(2)),
        },
        Op::Mov {
            d: Reg(3),
            src: Src::Imm(11),
            lanes: 15,
        },
        branch(6, 11),
        Op::Mov {
            d: Reg(3),
            src: Src::Imm(22),
            lanes: 15,
        },
        branch(8, 11),
        Op::Mov {
            d: Reg(3),
            src: Src::Imm(33),
            lanes: 15,
        },
        branch(10, 11),
        Op::Exit {
            condition: Pred::TRUE,
        },
    ]
    .into_iter()
    .map(word)
    .collect()
}

#[test]
fn structured_and_dispatched_switches_agree_for_divergent_lanes() {
    let code = program();
    let constants: Vec<_> = [80u32, 112, 144, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let tables = JumpTables::analyze(&code).unwrap();
    let possible = tables.resolve(&constants).unwrap();
    assert_eq!(possible.0[&4], [5, 7, 9]);
    let run = |known: Option<&JumpTargets>, constants: &[u8]| {
        let layout = Layout::workgroup([32, 1, 1]);
        let blocks = if let Some(known) = known {
            blocks_for_targets(&code, Some(known)).unwrap()
        } else {
            blocks(&code).unwrap()
        };
        let graph = structure::Graph::new(&blocks, known);
        assert_eq!(graph.is_some(), known.is_some());
        let emitter = emit_blocks(&code, &layout, &blocks, graph.as_ref()).unwrap();
        let result = emitter.registers[&3];
        let module = emitter.finish();
        if let Some(problem) = crate::spirv::tests::validate(&module.spirv) {
            panic!("{problem}");
        }
        let binding = module
            .descriptors
            .iter()
            .find(|desc| desc.index == 1)
            .unwrap()
            .binding;
        let inputs = Inputs {
            uniforms: [((0, binding), constants.to_vec())].into(),
            ..Inputs::default()
        };
        let output = Machine::new(&module.spirv).simulate(&inputs);
        assert!(output.problems.is_empty(), "{:?}", output.problems);
        output
            .invocations
            .iter()
            .map(|lane| lane[&result][0])
            .collect::<Vec<_>>()
    };
    let interpreted = run(None, &constants);
    let structured = run(Some(&possible), &constants);
    assert_eq!(structured, interpreted);
    assert_eq!(structured[..3], [11, 22, 33]);
    assert!(structured[3..].iter().all(|&value| value == 33));
    // Replacing an entry may change the set of possible destinations.
    // Unrelated words do not affect the same specialization.
    let mut changed = constants;
    changed[12] = 0xff;
    assert_eq!(tables.resolve(&changed), Some(possible.clone()));
    changed[..4].copy_from_slice(&112u32.to_le_bytes());
    let changed_targets = tables.resolve(&changed).unwrap();
    assert_ne!(possible, changed_targets);
    assert_eq!(run(None, &changed), run(Some(&changed_targets), &changed));
}

#[test]
fn a_switch_requires_all_entries_and_an_uninterrupted_index_definition() {
    let mut code = program();
    let constants: Vec<_> = [80u32, 112, 144]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let tables = JumpTables::analyze(&code).unwrap();
    assert!(tables.resolve(&constants[..8]).is_none());
    let mut invalid = constants.clone();
    invalid[4] = 113;
    assert!(tables.resolve(&invalid).is_none());
    code[6] = word(branch(6, 2));
    assert!(JumpTables::analyze(&code).is_none());
}
