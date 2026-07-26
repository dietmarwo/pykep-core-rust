#![no_main]

use libfuzzer_sys::fuzz_target;
use pykep_core::astro::propagation::state_transition_matrix_reynolds;

fn value(data: &[u8], index: usize) -> f64 {
    let start = index * 8;
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&data[start..start + 8]);
    f64::from_bits(u64::from_le_bytes(bytes))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 112 {
        return;
    }
    let initial = core::array::from_fn(|index| value(data, index));
    let final_state = core::array::from_fn(|index| value(data, index + 6));
    let _ =
        state_transition_matrix_reynolds(&initial, &final_state, value(data, 12), value(data, 13));
});
