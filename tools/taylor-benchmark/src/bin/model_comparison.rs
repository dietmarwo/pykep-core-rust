// SPDX-License-Identifier: MPL-2.0
//! Warm native Taylor timings with a separate tight DOP853 endpoint reference.

use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics, pontryagin::*, zoh::*};
use pykep_core::integration::{
    Dop853, InitialValueProblem, IntegratorOptions, Propagation, SensitivityProblem, Taylor,
    TaylorDynamicsModel,
};
use std::{hint::black_box, time::Instant};

#[path = "model_comparison/corpus.rs"]
mod corpus;

struct Timing {
    samples: [f64; 7],
}

impl std::fmt::Display for Timing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut sorted = self.samples;
        sorted.sort_by(f64::total_cmp);
        write!(f, "{:.6},{:.6},{:.6}", sorted[0], sorted[3], sorted[6])
    }
}

impl Timing {
    fn raw(&self) -> String {
        self.samples.map(|sample| format!("{sample:.6}")).join("|")
    }
}

fn argument(prefix: &str) -> Option<String> {
    std::env::args().find_map(|arg| arg.strip_prefix(prefix).map(str::to_owned))
}

fn selected(name: &str) -> bool {
    argument("--only=")
        .is_none_or(|patterns| patterns.split(',').any(|pattern| name.starts_with(pattern)))
}

fn measure(mut run: impl FnMut(), repeats: usize) -> Timing {
    run();
    // Scale the number of calls per sample without changing the fixtures or warmup.
    let multiplier = argument("--repeat-scale=").map_or(1, |value| {
        value.parse::<usize>().expect("positive repeat scale")
    });
    assert!((1..=32).contains(&multiplier));
    let repeats = repeats * multiplier;
    let mut samples = [0.0; 7];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..repeats {
            run();
        }
        *sample = start.elapsed().as_secs_f64() * 1e6 / repeats as f64;
    }
    Timing { samples }
}

fn error(state: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(state.len(), reference.len());
    state
        .iter()
        .zip(reference)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

fn emit<const N: usize>(
    name: &str,
    method: &str,
    tolerance: f64,
    run: impl Fn() -> Propagation<N>,
    reference: &[f64; N],
) {
    let result = run();
    let timing = measure(
        || {
            black_box(run());
        },
        256,
    );
    println!(
        "{name},{method},{tolerance},{timing},{:.17e},{},{},{:.17e},{}",
        error(&result.state, reference),
        result.stats.accepted_steps,
        result.stats.rhs_evaluations,
        result.state.iter().sum::<f64>(),
        timing.raw()
    );
}

fn model<M, const N: usize, const P: usize>(
    name: &str,
    model: M,
    initial: [f64; N],
    parameters: [f64; P],
    duration: f64,
) where
    M: TaylorDynamicsModel<N, P>,
{
    if !selected(name) {
        return;
    }
    if fixtures() {
        println!(
            r#"{{"model":"{name}","state":{initial:?},"parameters":{parameters:?},"final_time":{duration:?}}}"#
        );
        return;
    }
    if corpus::enabled() {
        corpus::check(name, &model, initial, parameters, duration);
        return;
    }
    let problem = InitialValueProblem::new(0.0, initial, duration, parameters);
    let reference = Dop853
        .propagate(
            &model,
            problem,
            IntegratorOptions {
                relative_tolerance: 3e-14,
                absolute_tolerance: 3e-14,
                maximum_step: Some(duration / 128.0),
                ..IntegratorOptions::default()
            },
        )
        .unwrap()
        .state;
    if reference_fixtures() {
        println!(
            r#"{{"model":"{name}","state":{initial:?},"parameters":{parameters:?},"final_time":{duration:?},"reference":{reference:?}}}"#
        );
        return;
    }
    let timing_tolerance =
        argument("--run-tolerance=").map(|tol| tol.parse::<f64>().expect("valid tolerance"));
    for tolerance in [1e-9, 1e-12, 1e-14] {
        if timing_tolerance.is_some_and(|requested| requested != tolerance) {
            continue;
        }
        let options = IntegratorOptions {
            relative_tolerance: tolerance,
            absolute_tolerance: tolerance,
            ..IntegratorOptions::default()
        };
        emit(
            name,
            "taylor",
            tolerance,
            || {
                Taylor
                    .propagate(&model, black_box(problem), options)
                    .unwrap()
            },
            &reference,
        );
        if std::env::args().any(|arg| arg == "--taylor-only") {
            continue;
        }
        emit(
            name,
            "dop853",
            tolerance,
            || {
                Dop853
                    .propagate(&model, black_box(problem), options)
                    .unwrap()
            },
            &reference,
        );
    }
}

fn fixtures() -> bool {
    std::env::args().any(|arg| arg == "--fixtures")
}

fn reference_fixtures() -> bool {
    std::env::args().any(|arg| arg == "--reference-fixtures")
}

fn main() {
    let kepler = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
    let rotating = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let mu = 0.012_150_585_609_624_04;
    let cartesian = [1., 0., 0., 0., 1., 0., 10., 1., 1., 1., 1., 1., 1., 1.];
    let equinoctial = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ];
    if corpus::enabled() {
        println!("{}", corpus::HEADER);
    } else if !fixtures() && !reference_fixtures() {
        println!(
            "model,method,tolerance,min_us,median_us,max_us,scaled_endpoint_error,steps,work_units,checksum,samples_us"
        );
    }
    model("kepler", KeplerDynamics, kepler, [1.0], 0.5);
    model("cr3bp", Cr3bpDynamics, rotating, [mu], 0.75);
    model(
        "bcp",
        BcpDynamics,
        rotating,
        [mu, 328_900.56, 389.172, -0.925_195_985_520_347],
        0.25,
    );
    model(
        "zoh_kepler",
        ZohKeplerDynamics,
        [0.8, -0.2, 0.1, 0.03, 1.0, 0.02, 1.1],
        [0.02, 0.3, -0.4, 0.5, 0.01],
        0.5,
    );
    model(
        "zoh_cr3bp",
        ZohCr3bpDynamics,
        [0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 1.1],
        [0.02, 0.3, -0.4, 0.5, 0.01, mu],
        0.5,
    );
    model(
        "zoh_equinoctial",
        ZohEquinoctialDynamics,
        [1.1, 0.1, -0.05, 0.02, -0.03, 0.4, 1.1],
        [0.02, 0.3, -0.4, 0.5, 0.01],
        0.5,
    );
    model(
        "solar_sail",
        ZohSolarSailDynamics,
        [0.8, -0.4, 0.3, 0.2, 0.9, -0.1],
        [0.25, -1.1, 0.04],
        0.5,
    );
    model(
        "cartesian_mass",
        CartesianMassOptimal,
        cartesian,
        [1.0, 0.01, 1.0, 0.5, 1.0],
        0.1,
    );
    model(
        "cartesian_time",
        CartesianTimeOptimal,
        cartesian,
        [1.0, 0.01, 1.0],
        0.1,
    );
    model(
        "equinoctial_mass",
        EquinoctialMassOptimal,
        equinoctial,
        [1.0, 1e-4, 1.0, 1.0, 1e-4],
        0.1,
    );
    model(
        "equinoctial_time",
        EquinoctialTimeOptimal,
        equinoctial,
        [1.0, 1e-4, 1.0],
        0.1,
    );
    if corpus::enabled() {
        corpus::extra();
        return;
    }
    if fixtures() || reference_fixtures() {
        return;
    }
    if std::env::args().any(|arg| arg == "--scalar-suite") {
        return;
    }
    let options = IntegratorOptions::default();
    let problem = InitialValueProblem::new(0., kepler, 0.5, [1.]);
    if selected("kepler_grid") {
        dense_workload(kepler, problem, options);
    }
    if selected("kepler_stm") {
        if std::env::args().any(|arg| arg == "--seed-sweep") {
            stm_workload::<1>("kepler_stm_w1", problem, options, true);
            stm_workload::<4>("kepler_stm_w4", problem, options, true);
            stm_workload::<6>("kepler_stm_w6", problem, options, true);
            stm_workload::<7>("kepler_stm_w7", problem, options, true);
            stm_workload::<14>("kepler_stm_w14", problem, options, true);
            stm_workload::<21>("kepler_stm_w21", problem, options, true);
        } else {
            stm_workload::<6>("kepler_stm", problem, options, false);
        }
    }
    batch_workload(rotating, mu, options);
    leg_workload(options);
}

fn dense_workload(
    kepler: [f64; 6],
    problem: InitialValueProblem<6, 1>,
    options: IntegratorOptions,
) {
    let sweep = std::env::args().any(|arg| arg == "--grid-sweep");
    let sizes: &[u32] = if sweep {
        &[1, 10, 100, 1001, 10001]
    } else {
        &[1001]
    };
    for &points in sizes {
        let name = if sweep {
            format!("kepler_grid_{points}")
        } else {
            "kepler_grid".into()
        };
        let times: Vec<_> = (0..points)
            .map(|i| 0.5 * f64::from(i) / f64::from((points - 1).max(1)))
            .collect();
        let run = || {
            Taylor
                .propagate_dense(&KeplerDynamics, problem, &times, options)
                .unwrap()
        };
        let dense = run();
        let timing = measure(
            || {
                black_box(run());
            },
            32,
        );
        let worst = times
            .iter()
            .zip(&dense.states)
            .map(|(&time, state)| {
                let reference =
                    pykep_core::astro::propagation::propagate_lagrangian(&kepler, time, 1.)
                        .unwrap();
                error(state, &reference)
            })
            .fold(0.0, f64::max);
        println!(
            "{name},taylor,1e-12,{timing},{worst:.17e},{},{},{:.17e},{}",
            dense.stats.accepted_steps,
            dense.stats.rhs_evaluations,
            dense.states.last().unwrap().iter().sum::<f64>(),
            timing.raw()
        );
    }
}

fn stm_workload<const W: usize>(
    name: &str,
    problem: InitialValueProblem<6, 1>,
    options: IntegratorOptions,
    mixed: bool,
) {
    let seeded: SensitivityProblem<6, 1, W> = SensitivityProblem {
        nominal: problem,
        initial_sensitivities: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                f64::from(i == j % 6)
                    + if mixed {
                        0.01 * ((i + j) % 5) as f64
                    } else {
                        0.0
                    }
            })
        }),
        parameter_seeds: [std::array::from_fn(|j| {
            if mixed { 0.01 * (j + 1) as f64 } else { 0.0 }
        })],
    };
    let run = || {
        Taylor
            .propagate_with_sensitivities(&KeplerDynamics, seeded, options)
            .unwrap()
    };
    let stm = run();
    let reference = Dop853
        .propagate_with_sensitivities(
            &KeplerDynamics,
            seeded,
            IntegratorOptions {
                relative_tolerance: 1e-14,
                absolute_tolerance: 1e-14,
                maximum_step: Some(0.005),
                ..options
            },
        )
        .unwrap();
    let stm_error = stm
        .sensitivities
        .iter()
        .zip(&reference.sensitivities)
        .map(|(a, b)| error(a, b))
        .fold(0.0, f64::max);
    let timing = measure(
        || {
            black_box(run());
        },
        256,
    );
    println!(
        "{name},taylor,1e-12,{timing},{stm_error:.17e},{},{},{:.17e},{}",
        stm.stats.accepted_steps,
        stm.stats.rhs_evaluations,
        stm.sensitivities.iter().flatten().sum::<f64>(),
        timing.raw()
    );
}

fn batch_workload(state: [f64; 6], mu: f64, options: IntegratorOptions) {
    if !selected("bcp_population") {
        return;
    }
    let parameters = [mu, 328_900.56, 389.172, -0.925_195_985_520_347];
    for irregular in [false, true] {
        let jobs: Vec<_> = (0..1024)
            .map(|i| {
                let mut state = state;
                state[0] += f64::from(i) * 1e-6;
                let duration = if irregular {
                    0.01 + f64::from(i % 17) * 0.02
                } else {
                    0.25
                };
                InitialValueProblem::new(0.0, state, duration, parameters)
            })
            .collect();
        let sweep = std::env::args().any(|arg| arg == "--batch-sweep");
        let sizes: &[usize] = if sweep {
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64, 256, 1024]
        } else {
            &[1024]
        };
        for &size in sizes {
            let jobs = &jobs[..size];
            for workers in [1, 4] {
                let scalar = || {
                    pykep_core::batch::try_map(jobs, workers, |&p| {
                        Taylor.propagate(&BcpDynamics, p, options)
                    })
                    .unwrap()
                };
                let batch = || {
                    Taylor
                        .propagate_batch(&BcpDynamics, jobs, options, workers)
                        .unwrap()
                };
                let reference = scalar();
                let actual = batch();
                let maximum_error = actual
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| error(&a.state, &b.state))
                    .fold(0.0, f64::max);
                let tag = if irregular { "mixed" } else { "balanced" };
                if sweep {
                    let vector_calls: usize = actual
                        .chunks_exact(4)
                        .map(|group| {
                            group
                                .iter()
                                .map(|lane| lane.stats.accepted_steps)
                                .max()
                                .unwrap_or(0)
                        })
                        .sum();
                    let active: usize = actual
                        .chunks_exact(4)
                        .flatten()
                        .map(|lane| lane.stats.accepted_steps)
                        .sum();
                    eprintln!(
                        "batch={tag} size={size} workers={workers} static_vector_calls={vector_calls} active_lane_steps={active}"
                    );
                }
                let suffix = if sweep {
                    format!("_{size}")
                } else {
                    String::new()
                };
                for (label, run) in [("scalar", &scalar as &dyn Fn() -> _), ("batch", &batch)] {
                    let timing = measure(
                        || {
                            black_box(run());
                        },
                        4,
                    );
                    println!(
                        "bcp_population_{tag}_{workers}{suffix},{label},1e-12,{timing},{maximum_error:.17e},0,0,0,{}",
                        timing.raw()
                    );
                }
            }
        }
    }
}

fn leg_workload(options: IntegratorOptions) {
    use pykep_core::integration::IntegrationMethod;
    use pykep_core::leg::ZohKeplerLeg;
    if !selected("zoh_leg") && !selected("zoh_jacobian") {
        return;
    }
    let counts: &[i32] = if std::env::args().any(|arg| arg == "--leg-sweep") {
        &[1, 20, 32, 100]
    } else {
        &[20, 32]
    };
    for &segments in counts {
        let leg = ZohKeplerLeg::new(
            ZohKeplerDynamics,
            [1., 0., 0., 0., 1., 0., 1.],
            (0..segments)
                .map(|i| [0.05, if i % 2 == 0 { 0.3 } else { -0.3 }, 0.4, 0.])
                .collect(),
            [0.8, 0.6, 0., -0.6, 0.8, 0., 0.99],
            (0..=segments)
                .map(|i| f64::from(i) / f64::from(segments))
                .collect(),
            [0.02],
            0.5,
            options,
        )
        .unwrap();
        let reference = leg.mismatch_constraints().unwrap();
        let jac = leg.mismatch_jacobian().unwrap();
        for (name, method) in [
            ("taylor", IntegrationMethod::Taylor),
            ("dop853", IntegrationMethod::Dop853),
        ] {
            let run = || leg.mismatch_constraints_with_method(method).unwrap();
            let delta = error(&run(), &reference);
            let timing = measure(
                || {
                    black_box(run());
                },
                64,
            );
            println!(
                "zoh_leg_{segments},{name},1e-12,{timing},{delta:.17e},0,0,0,{}",
                timing.raw()
            );
            let run = || leg.mismatch_jacobian_with_method(method).unwrap();
            let actual = run();
            let delta = actual
                .initial_state
                .iter()
                .zip(&jac.initial_state)
                .chain(actual.final_state.iter().zip(&jac.final_state))
                .map(|(a, b)| error(a, b))
                .chain(
                    actual
                        .controls
                        .iter()
                        .zip(&jac.controls)
                        .map(|(a, b)| error(a, b)),
                )
                .chain(
                    actual
                        .time_grid
                        .iter()
                        .zip(&jac.time_grid)
                        .map(|(a, b)| error(a, b)),
                )
                .fold(0.0, f64::max);
            let timing = measure(
                || {
                    black_box(run());
                },
                64,
            );
            println!(
                "zoh_jacobian_{segments},{name},1e-12,{timing},{delta:.17e},0,0,0,{}",
                timing.raw()
            );
        }
    }
}
