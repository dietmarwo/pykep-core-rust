// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Public Taylor-backend and independent heyoka reference tests.

use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohKeplerDynamics, propagate_schedule_with_method,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{
    AdaptiveIntegrator, InitialValueProblem, IntegrationMethod, IntegratorOptions,
    SensitivityProblem, Taylor,
};
use serde_json::Value;

fn array<const N: usize>(value: &Value) -> [f64; N] {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .unwrap()
        .try_into()
        .unwrap()
}

fn options() -> IntegratorOptions {
    IntegratorOptions {
        relative_tolerance: 1e-14,
        absolute_tolerance: 1e-14,
        maximum_steps: 1_000_000,
        ..IntegratorOptions::default()
    }
}

fn assert_state<const N: usize>(actual: &[f64; N], expected: &[f64; N], tolerance: f64) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "component {index}: actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}

#[test]
fn heyoka_reference_states_and_kepler_stm_match() {
    let document: Value = serde_json::from_str(include_str!("data/taylor-heyoka-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(document["heyoka_version"], "7.10.1");

    for case in document["cases"].as_array().unwrap() {
        let initial = array::<6>(&case["initial_state"]);
        let final_time = case["final_time"].as_f64().unwrap();
        let expected = array::<6>(&case["state"]);
        let actual = match case["name"].as_str().unwrap() {
            "kepler" => {
                Taylor
                    .propagate(
                        &KeplerDynamics,
                        InitialValueProblem::new(
                            0.0,
                            initial,
                            final_time,
                            array(&case["parameters"]),
                        ),
                        options(),
                    )
                    .unwrap()
                    .state
            }
            "cr3bp" => {
                Taylor
                    .propagate(
                        &Cr3bpDynamics,
                        InitialValueProblem::new(
                            0.0,
                            initial,
                            final_time,
                            array(&case["parameters"]),
                        ),
                        options(),
                    )
                    .unwrap()
                    .state
            }
            "bcp" => {
                Taylor
                    .propagate(
                        &BcpDynamics,
                        InitialValueProblem::new(
                            0.0,
                            initial,
                            final_time,
                            array(&case["parameters"]),
                        ),
                        options(),
                    )
                    .unwrap()
                    .state
            }
            name => panic!("unexpected reference case {name}"),
        };
        assert_state(&actual, &expected, 3e-11);
    }

    let case = &document["cases"][0];
    let initial = array::<6>(&case["initial_state"]);
    let sensitivity = Taylor
        .propagate_with_sensitivities(
            &KeplerDynamics,
            SensitivityProblem {
                nominal: InitialValueProblem::new(
                    0.0,
                    initial,
                    case["final_time"].as_f64().unwrap(),
                    array(&case["parameters"]),
                ),
                initial_sensitivities: core::array::from_fn(|row| {
                    core::array::from_fn(|column| if row == column { 1.0 } else { 0.0 })
                }),
                parameter_seeds: [[0.0; 6]],
            },
            options(),
        )
        .unwrap();
    let expected = case["stm"].as_array().unwrap();
    for (row, expected_row) in expected.iter().enumerate() {
        assert_state(
            &sensitivity.sensitivities[row],
            &array::<6>(expected_row),
            3e-7,
        );
    }
}

#[test]
fn runtime_selector_and_zoh_schedule_are_publicly_callable() {
    assert_eq!(
        AdaptiveIntegrator::default().method(),
        IntegrationMethod::Taylor
    );
    let initial = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
    let selected = KeplerDynamics
        .propagate_with_method(0.0, initial, 0.5, 1.0, options(), IntegrationMethod::Taylor)
        .unwrap();
    let direct = Taylor
        .propagate(
            &KeplerDynamics,
            InitialValueProblem::new(0.0, initial, 0.5, [1.0]),
            options(),
        )
        .unwrap();
    assert_eq!(selected.state, direct.state);
    let default = KeplerDynamics
        .propagate(0.0, initial, 0.5, 1.0, options())
        .unwrap();
    assert_eq!(default.state, direct.state);

    let schedule = ControlSchedule::new(vec![0.0, 0.25], vec![[0.02, 0.3, -0.4, 0.5]]).unwrap();
    let result = propagate_schedule_with_method(
        &ZohKeplerDynamics,
        &schedule,
        [0.8, -0.2, 0.1, 0.03, 1.0, 0.02, 1.1],
        [0.01],
        options(),
        IntegrationMethod::Taylor,
    )
    .unwrap();
    assert_eq!(result.time, 0.25);
}
