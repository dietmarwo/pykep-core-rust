// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Fixed-protocol release benchmark and coarse CI regression guard.

use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint};

const FULL_SAMPLES: usize = 100;
const FULL_WARMUP: Duration = Duration::from_secs(3);
const MISMATCH_CALLS: usize = 10_000;
const GRADIENT_CALLS: usize = 1_000;
const MISMATCH_CI_LIMIT_NS: f64 = 10_000.0;
const GRADIENT_CI_LIMIT_NS: f64 = 45_000.0;

fn make_leg() -> pykep_core::Result<SimsFlanaganLeg> {
    use pykep_core::constants::{
        ASTRONOMICAL_UNIT, DAY_TO_SECONDS, EARTH_ORBITAL_VELOCITY, MU_SUN, STANDARD_GRAVITY,
    };

    SimsFlanaganLeg::new(
        SpacecraftEndpoint::new(
            [
                ASTRONOMICAL_UNIT,
                0.1 * ASTRONOMICAL_UNIT,
                -0.1 * ASTRONOMICAL_UNIT,
                0.2 * EARTH_ORBITAL_VELOCITY,
                EARTH_ORBITAL_VELOCITY,
                -0.2 * EARTH_ORBITAL_VELOCITY,
            ],
            1_500.0,
        )?,
        vec![
            [0.10, 0.11, 0.12],
            [0.13, 0.14, 0.15],
            [0.16, 0.17, 0.18],
            [0.19, 0.20, 0.21],
            [0.22, 0.23, 0.24],
        ],
        SpacecraftEndpoint::new(
            [
                1.2 * ASTRONOMICAL_UNIT,
                -0.1 * ASTRONOMICAL_UNIT,
                0.1 * ASTRONOMICAL_UNIT,
                -0.2 * EARTH_ORBITAL_VELOCITY,
                1.023 * EARTH_ORBITAL_VELOCITY,
                -0.44 * EARTH_ORBITAL_VELOCITY,
            ],
            1_300.0,
        )?,
        SimsFlanaganSettings::new(
            324.0 * DAY_TO_SECONDS,
            0.12,
            100.0 * STANDARD_GRAVITY,
            MU_SUN,
            0.6,
        )?,
    )
}

fn measure<F>(
    name: &str,
    samples: usize,
    calls_per_sample: usize,
    warmup: Duration,
    mut operation: F,
) -> Vec<f64>
where
    F: FnMut() -> f64,
{
    let warmup_started = Instant::now();
    let mut checksum = 0.0;
    while warmup_started.elapsed() < warmup {
        checksum += black_box(operation());
    }

    let mut timings = Vec::with_capacity(samples);
    for sample in 0..samples {
        let started = Instant::now();
        for _ in 0..calls_per_sample {
            checksum += black_box(operation());
        }
        let elapsed = started.elapsed().as_secs_f64();
        let nanoseconds = elapsed * 1e9 / calls_per_sample as f64;
        println!("{name},{sample},{nanoseconds:.6}");
        timings.push(nanoseconds);
    }
    eprintln!("{name} checksum={checksum:.17e}");
    timings
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.iter().any(|argument| {
        argument != "--quick"
            && argument != "--check"
            && argument != "--mismatch-only"
            && argument != "--gradient-only"
    }) || (arguments
        .iter()
        .any(|argument| argument == "--mismatch-only")
        && arguments
            .iter()
            .any(|argument| argument == "--gradient-only"))
    {
        eprintln!(
            "usage: pykep-release-benchmark [--quick] [--check] \
             [--mismatch-only|--gradient-only]"
        );
        return ExitCode::from(2);
    }
    let quick = arguments.iter().any(|argument| argument == "--quick");
    let check = arguments.iter().any(|argument| argument == "--check");
    let run_mismatch = !arguments
        .iter()
        .any(|argument| argument == "--gradient-only");
    let run_gradient = !arguments
        .iter()
        .any(|argument| argument == "--mismatch-only");
    let samples = if quick { 11 } else { FULL_SAMPLES };
    let warmup = if quick {
        Duration::from_millis(100)
    } else {
        FULL_WARMUP
    };
    let mismatch_calls = if quick { 1_000 } else { MISMATCH_CALLS };
    let gradient_calls = if quick { 100 } else { GRADIENT_CALLS };

    let leg = match make_leg() {
        Ok(leg) => leg,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("workload,sample,ns_per_call");
    let mut mismatch = if run_mismatch {
        measure(
            "sims_flanagan_mismatch",
            samples,
            mismatch_calls,
            warmup,
            || {
                leg.mismatch_constraints()
                    .expect("validated benchmark mismatch")[0]
            },
        )
    } else {
        Vec::new()
    };
    let mut gradient = if run_gradient {
        measure(
            "sims_flanagan_gradient",
            samples,
            gradient_calls,
            warmup,
            || {
                leg.mismatch_jacobian()
                    .expect("validated benchmark gradient")
                    .departure[0][0]
            },
        )
    } else {
        Vec::new()
    };

    if check {
        if run_mismatch {
            let mismatch_median = median(&mut mismatch);
            eprintln!("regression median: mismatch={mismatch_median:.1} ns");
            if mismatch_median > MISMATCH_CI_LIMIT_NS {
                eprintln!(
                    "performance regression: mismatch limit is \
                     {MISMATCH_CI_LIMIT_NS:.0} ns"
                );
                return ExitCode::FAILURE;
            }
        }
        if run_gradient {
            let gradient_median = median(&mut gradient);
            eprintln!("regression median: gradient={gradient_median:.1} ns");
            if gradient_median > GRADIENT_CI_LIMIT_NS {
                eprintln!(
                    "performance regression: gradient limit is \
                     {GRADIENT_CI_LIMIT_NS:.0} ns"
                );
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
