// SPDX-License-Identifier: MPL-2.0
//! Standalone objective replay and bounded fcmaes-rust ask/tell comparison.

use fcmaes_core::{De, DeParams, Fitness};
use pykep_core::dynamics::{BcpDynamics, zoh::ZohKeplerDynamics};
use pykep_core::integration::{InitialValueProblem, IntegrationMethod, IntegratorOptions, Taylor};
use pykep_core::leg::ZohKeplerLeg;
use std::{hint::black_box, time::Instant};

const POPULATION: usize = 32;
const BUDGET: usize = 4096;
const BOUNDS: [f64; 4] = [0.02, 0.04, 0.1, 0.1];

type Score = (f64, f64);

fn score(residual: &[f64]) -> Score {
    (
        residual.iter().map(|x| x * x).sum(),
        residual.iter().map(|x| x.abs()).fold(0.0, f64::max),
    )
}

fn evaluate(
    model: &str,
    candidates: &[Vec<f64>],
    workers: usize,
) -> pykep_core::Result<Vec<Score>> {
    let options = IntegratorOptions::default();
    if model == "bcp" {
        let problems: Vec<_> = candidates
            .iter()
            .map(|x| {
                InitialValueProblem::new(
                    0.0,
                    [0.8, -0.2, 0.1, 0.03 + x[0], -0.04 + x[1], 0.02],
                    0.25 + x[2],
                    [
                        0.012_150_585_609_624_04,
                        328_900.56,
                        389.172,
                        -0.925_195_985_520_347 + x[3],
                    ],
                )
            })
            .collect();
        let outputs = Taylor.propagate_batch(&BcpDynamics, &problems, options, workers)?;
        return Ok(outputs
            .into_iter()
            .map(|result| {
                score(&[
                    result.state[0] - 0.77,
                    result.state[1] + 0.19,
                    result.state[2] - 0.1,
                ])
            })
            .collect());
    }
    pykep_core::batch::try_map(candidates, workers, |x| {
        let controls = (0..20)
            .map(|i| {
                [
                    0.05 + x[0],
                    0.3 + x[1] * if i % 2 == 0 { 1.0 } else { -1.0 },
                    0.4 + x[2],
                    x[3],
                ]
            })
            .collect();
        let leg = ZohKeplerLeg::new(
            ZohKeplerDynamics,
            [1., 0., 0., 0., 1., 0., 1.],
            controls,
            [0.8, 0.6, 0., -0.6, 0.8, 0., 0.99],
            (0..=20).map(|i| f64::from(i) / 20.0).collect(),
            [0.02],
            0.5,
            options,
        )?;
        Ok(score(&leg.mismatch_constraints_with_method(
            IntegrationMethod::Taylor,
        )?))
    })
}

fn fixed_population() -> Vec<Vec<f64>> {
    (0..1024)
        .map(|index| {
            BOUNDS
                .iter()
                .enumerate()
                .map(|(j, bound)| {
                    let cell = (index * (31 + 2 * j) + 17 * j) % 997;
                    bound * (2.0 * cell as f64 / 996.0 - 1.0)
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let workers = args
        .iter()
        .find_map(|s| s.strip_prefix("--workers="))
        .map_or(Ok(1), str::parse)?;
    assert!(matches!(workers, 1 | 4));
    let replay_only = args.iter().any(|s| s == "--replay-only");
    let fixed = fixed_population();
    for model in ["bcp", "zoh_kepler"] {
        let output = evaluate(model, &fixed, workers)?;
        if args.iter().any(|s| s == "--values") {
            for (index, (fitness, constraint)) in output.iter().enumerate() {
                println!("value,{model},{index},{fitness:.17e},{constraint:.17e}");
            }
        }
        let mut times = [0.0; 7];
        for elapsed in &mut times {
            let start = Instant::now();
            for _ in 0..8 {
                black_box(evaluate(model, &fixed, workers)?);
            }
            *elapsed = start.elapsed().as_secs_f64() * 1e6 / 8.0;
        }
        times.sort_by(f64::total_cmp);
        println!(
            "replay,{model},{workers},{},{:.6},{:.6},{:.6}",
            fixed.len(),
            times[0],
            times[3],
            times[6]
        );
        if replay_only {
            continue;
        }
        for seed in [11, 19, 42] {
            let mut de = De::new(
                Fitness::bounded(4, 1, &BOUNDS.map(|v| -v), &BOUNDS),
                &[],
                &[],
                None,
                &DeParams {
                    popsize: POPULATION as i32,
                    max_evaluations: BUDGET as u64,
                    seed,
                    ..DeParams::default()
                },
            );
            let started = Instant::now();
            let mut reached = None;
            // Equal ask/tell budgets make the evaluation count independent of termination heuristics.
            for _ in 0..BUDGET / POPULATION {
                let candidates = de.ask();
                let values: Vec<_> = evaluate(model, &candidates, workers)?
                    .into_iter()
                    .map(|x| x.0)
                    .collect();
                de.tell(&values);
                if reached.is_none() && de.result().y < 1e-4 {
                    reached = Some(started.elapsed().as_secs_f64());
                }
            }
            let result = de.result();
            let reached =
                reached.map_or_else(|| "unreached".into(), |seconds| format!("{seconds:.9}"));
            println!(
                "optimizer,{model},{workers},{seed},{},{:.9},{:.17e},{reached}",
                result.evaluations,
                started.elapsed().as_secs_f64(),
                result.y
            );
        }
    }
    Ok(())
}
