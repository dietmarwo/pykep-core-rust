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
