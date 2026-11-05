// SPDX-License-Identifier: MPL-2.0

//! Independent trajectory replay and ordered population regression checks.

use pykep_trajopt::{GymProblem, Problem, TopsOptions};
use serde_json::Value;

#[test]
fn feasible_tops_examples_pass_full_forward_validation() {
    for text in [
        include_str!("../data/solutions/twobody.P4.json"),
        include_str!("../data/solutions/mee.P8.json"),
        include_str!("../data/solutions/cr3bp.P0.json"),
        include_str!("../data/solutions/sail.P4.json"),
    ] {
        let record: Value = serde_json::from_str(text).unwrap();
        let id = record["id"].as_str().unwrap();
        let problem = GymProblem::new(
            id,
            TopsOptions {
                segments: Some(record["options"]["nseg"].as_u64().unwrap() as usize),
                ..TopsOptions::default()
            },
        )
        .unwrap();
        let x: Vec<f64> = serde_json::from_value(record["x"].clone()).unwrap();
        for (&value, &[low, high]) in x.iter().zip(&problem.spec().bounds) {
            assert!(low <= value && value <= high, "{id} bounds");
        }
        assert!(problem.evaluate(&x).unwrap().feasible(1e-8, 1e-8), "{id}");
        assert!(
            problem
                .forward_residual(&x)
                .unwrap()
                .iter()
                .all(|v| v.abs() < 1e-6),
            "{id}"
        );
    }
}

#[test]
fn ordered_population_batches_match_scalar_evaluations() {
    for id in ["tops.cr3bp.P0", "tops.sail.P4.moving", "gym.cassini2"] {
        let problem = GymProblem::new(
            id,
            TopsOptions {
                segments: Some(6),
                ..TopsOptions::default()
            },
        )
        .unwrap();
        let spec = problem.spec();
        let population: Vec<Vec<f64>> = (0..9)
            .map(|k| {
                spec.bounds
                    .iter()
                    .map(|&[low, high]| low + (high - low) * (0.1 + 0.08 * f64::from(k)))
                    .collect()
            })
            .collect();
        let batch = problem.evaluate_batch(&population, 3).unwrap();
        assert_eq!(batch.len(), population.len());
        for (x, evaluation) in population.iter().zip(&batch) {
            assert_eq!(
                evaluation.fitness(),
                problem.evaluate(x).unwrap().fitness(),
                "{id}"
            );
        }
    }
}
