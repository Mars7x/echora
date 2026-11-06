//! Print the merged gamepad state whenever it changes (for checking a
//! controller's mapping without the game). `watch rumble` also plays a
//! short rumble on each pad once one connects.
fn main() {
    let gamepads = echora_input::gamepads();
    let rumble = std::env::args()
        .nth(1)
        .is_some_and(|argument| argument == "rumble");
    let mut last = None;
    let mut rumbled = false;
    loop {
        let state = gamepads.state();
        if last != Some(state) {
            println!(
                "buttons {:#08x} left {:?} right {:?}",
                state.buttons, state.left, state.right
            );
            last = Some(state);
        }
        if rumble && !rumbled && gamepads.connected() {
            rumbled = true;
            println!("rumble strong 0.8 weak 0.3");
            echora_input::rumble(0, 0.8, 0.3);
            std::thread::sleep(std::time::Duration::from_millis(500));
            println!("rumble off");
            echora_input::rumble(0, 0.0, 0.0);
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
