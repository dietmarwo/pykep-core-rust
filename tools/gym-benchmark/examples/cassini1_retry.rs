// SPDX-License-Identifier: MPL-2.0

//! Parallel BiteOpt retries on the gym's Cassini1 mission.
//!
//! Run from the repository root:
//!
//! ```bash
//! cargo run --release -p pykep-gym-benchmark --example cassini1_retry
//! ```
//!
//! The astrodynamics introduction walks through this example. The objective
//! is the total delta-v in m/s. A decision that the evaluator rejects, for
//! example an extreme flyby geometry, becomes a large penalty instead of
//! stopping the search.

use fcmaes_core::{BiteParams, RetryBounds, RetryConfig, RetryRunResult, optimize_bite, retry};
use pykep_trajopt::{GymProblem, Problem, TopsOptions};

const PENALTY: f64 = 1e10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = GymProblem::new("gym.cassini1", TopsOptions::default())?;
    let spec = problem.spec();
    let lower: Vec<f64> = spec.bounds.iter().map(|b| b[0]).collect();
    let upper: Vec<f64> = spec.bounds.iter().map(|b| b[1]).collect();
    let objective = |x: &[f64]| {
        problem
            .evaluate(x)
            .map_or(PENALTY, |evaluation| evaluation.objectives[0])
    };

    let bounds = RetryBounds::new(lower, upper)?;
    let config = RetryConfig {
        num_retries: 32,
        workers: 0, // all available cores
        max_evaluations: 50_000,
        seed: 42,
        ..RetryConfig::default()
    };
    let best = retry(&objective, &bounds, &config, |objective, context| {
        let result = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            None,
            &BiteParams {
                max_evaluations: context.max_evaluations,
                seed: context.run_seed,
                ..BiteParams::default()
            },
            6,
        );
        RetryRunResult {
            x: result.x,
            y: result.y,
            evaluations: result.evaluations,
        }
    });

    println!(
        "best delta-v {:.1} m/s after {} runs and {} evaluations",
        best.y, best.runs, best.evaluations
    );
    for (name, value) in spec.decisions.iter().zip(&best.x) {
        println!("  {name:28} {value:10.3}");
    }
    let trajectory = problem.decode(&best.x, 0)?;
    println!(
        "  legs: {}",
        trajectory["trajectory"]["arcs"]
            .as_array()
            .map_or(0, Vec::len)
    );
    Ok(())
}
