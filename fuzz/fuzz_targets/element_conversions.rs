#![no_main]

use libfuzzer_sys::fuzz_target;
use pykep_core::astro::elements::{
    ClassicalElements, cartesian_to_classical, classical_to_cartesian,
};

fn word(data: &[u8], index: usize) -> u64 {
    let mut bytes = [0_u8; 8];
    let start = index * 8;
    bytes.copy_from_slice(&data[start..start + 8]);
    u64::from_le_bytes(bytes)
}

fn unit(word: u64) -> f64 {
    word as f64 / u64::MAX as f64
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 56 {
        return;
    }

    let arbitrary = ClassicalElements::new(
        f64::from_bits(word(data, 0)),
        f64::from_bits(word(data, 1)),
        f64::from_bits(word(data, 2)),
        f64::from_bits(word(data, 3)),
        f64::from_bits(word(data, 4)),
        f64::from_bits(word(data, 5)),
    );
    let arbitrary_mu = f64::from_bits(word(data, 6));
    let _ = classical_to_cartesian(arbitrary, arbitrary_mu);

    let bounded = ClassicalElements::new(
        0.1 + 20.0 * unit(word(data, 0)),
        0.95 * unit(word(data, 1)),
        core::f64::consts::PI * unit(word(data, 2)),
        core::f64::consts::TAU * unit(word(data, 3)),
        core::f64::consts::TAU * unit(word(data, 4)),
        core::f64::consts::TAU * unit(word(data, 5)),
    );
    if let Ok(state) = classical_to_cartesian(bounded, 0.1 + unit(word(data, 6))) {
        let _ = cartesian_to_classical(&state, 0.1 + unit(word(data, 6)));
    }
});
