#![no_main]

use libfuzzer_sys::fuzz_target;
use pykep_core::time::epoch::Epoch;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = core::str::from_utf8(data) {
        let _ = Epoch::from_iso(text);
    }
});
