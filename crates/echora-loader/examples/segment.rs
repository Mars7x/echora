//! Writes a module's decompressed text, rodata or data segment to a file
//! (for disassembling guest code): `segment EXEFS main text OUT`.
#![allow(clippy::unwrap_used)]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, exefs, module, segment, out] = &args[..] else {
        panic!("usage: segment EXEFS MODULE text|rodata|data OUT");
    };
    let modules = echora_loader::load_exefs(std::path::Path::new(exefs)).unwrap();
    let m = modules.iter().find(|m| &m.name == module).unwrap();
    let bytes = match segment.as_str() {
        "text" => &m.text.bytes,
        "rodata" => &m.rodata.bytes,
        "data" => &m.data.bytes,
        other => panic!("no segment {other}"),
    };
    std::fs::write(out, bytes).unwrap();
}
