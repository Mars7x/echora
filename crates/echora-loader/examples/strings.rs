//! Prints NUL-terminated identifier strings in a module's rodata and data
//! that start with a prefix, e.g. NVN entry points: `strings EXEFS main nvn`.
#![allow(clippy::unwrap_used)]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, exefs, module, prefix] = &args[..] else {
        panic!("usage: strings EXEFS MODULE PREFIX");
    };
    let modules = echora_loader::load_exefs(std::path::Path::new(exefs)).unwrap();
    let m = modules.iter().find(|m| &m.name == module).unwrap();
    for segment in [&m.rodata, &m.data] {
        for s in segment.bytes.split(|&b| b == 0) {
            if s.starts_with(prefix.as_bytes())
                && s.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_')
            {
                println!("{}", String::from_utf8_lossy(s));
            }
        }
    }
}
