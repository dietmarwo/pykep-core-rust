#![no_main]

use libfuzzer_sys::fuzz_target;
use pykep_core::astro::lambert::LambertProblem;

fn component(byte: u8) -> f64 {
    4.0 * (f64::from(byte) / 255.0 - 0.5)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 11 {
        return;
    }
    let initial = [
        component(data[0]) + 0.01,
        component(data[1]),
        component(data[2]),
    ];
    let final_position = [
        component(data[3]),
        component(data[4]) + 0.01,
        component(data[5]),
    ];
    let time = 0.01 + 50.0 * f64::from(data[6]) / 255.0;
    let mu = 0.01 + 10.0 * f64::from(data[7]) / 255.0;
    let clockwise = data[8] & 1 == 1;
    let revolutions = usize::from(data[9] % 5);
    let _ = LambertProblem::new(initial, final_position, time, mu, clockwise, revolutions);
});
