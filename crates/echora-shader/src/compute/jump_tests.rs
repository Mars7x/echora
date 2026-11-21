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
#[test]
fn fragment_jump_tables_preserve_pixels_in_a_divergent_quad() {
    use crate::header::{Header, Interpolation};
    use crate::sim::QuadPixel;
    use echora_shader_abi::Stage;

    let mut code = program();
    // IPA R0, a[0x80] reads the first flat input as a register word.
    code[0] = crate::decode(
        0x326 | (7 << 12) | (255 << 24) | (255 << 32) | (32 << 64) | (1 << 76) | (7 << 81),
    );
    for (index, value) in [(5, 0.25f32), (7, 0.5), (9, 0.75)] {
        code[index] = word(Op::Mov {
            d: Reg(0),
            src: Src::Imm(value.to_bits()),
            lanes: 15,
        });
    }
    let mut imap = [Interpolation::Unused; 128];
    imap[0] = Interpolation::Constant;
    let layout = Layout {
        stage: Stage::Fragment,
        header: Some(Header {
            stage: Stage::Fragment,
            imap,
            targets: 1,
            writes_depth: false,
            writes_sample_mask: false,
            local_bytes: Some(0),
            outputs: [0; 5],
            outputs_beyond: false,
        }),
        ..Layout::workgroup([1; 3])
    };
    let table = JumpTables::analyze(&code).unwrap();
    let control = super::translate(&code, &layout).unwrap();
    for destinations in [[80u32, 112, 144], [144, 80, 112]] {
        let constants: Vec<_> = destinations
            .into_iter()
            .chain([0])
            .flat_map(u32::to_le_bytes)
            .collect();
        let targets = table.resolve(&constants).unwrap();
        let (optimized, _) = super::translate_with_jump_targets(&code, &layout, &targets).unwrap();
        assert_eq!(control.descriptors, optimized.descriptors);
        let binding = control
            .descriptors
            .iter()
            .find(|descriptor| descriptor.index == 1)
            .unwrap()
            .binding;
        let colors = |program: &echora_shader_abi::Translated| {
            if let Some(problem) = crate::spirv::tests::validate(&program.spirv) {
                panic!("{problem}");
            }
            let machine = Machine::new(&program.spirv);
            let result = machine.simulate(&Inputs {
                uniforms: [((0, binding), constants.clone())].into(),
                quad: (0..4)
                    .map(|value| QuadPixel {
                        frag_coord: [0.0; 4],
                        helper: false,
                        locations: [(0, vec![value, 0, 0, 0])].into(),
                    })
                    .collect(),
                ..Inputs::default()
            });
            assert!(result.problems.is_empty(), "{:?}", result.problems);
            result
                .invocations
                .iter()
                .map(|pixel| machine.located_outputs(pixel)[&(0, 0)][0] as u32)
                .collect::<Vec<_>>()
        };
        let observed = colors(&control);
        assert_eq!(colors(&optimized), observed);
        let expected = [0, 1, 2, 2].map(|index| match destinations[index] {
            80 => 0.25f32.to_bits(),
            112 => 0.5f32.to_bits(),
            _ => 0.75f32.to_bits(),
        });
        assert_eq!(observed, expected);
    }
}
