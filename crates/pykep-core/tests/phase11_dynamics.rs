// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Evaluated-model and C++ Taylor-reference validation for phase 11.

use pykep_core::constants::{
    BCP_MU_EARTH_MOON, BCP_MU_SUN, BCP_SUN_ANGULAR_VELOCITY, BCP_SUN_DISTANCE, CR3BP_MU_EARTH_MOON,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{DifferentiableDynamicsModel, DynamicsModel, IntegratorOptions};
use pykep_core::{CartesianState, Matrix6, PykepError};
use serde_json::Value;

const REFERENCE_OPTIONS: IntegratorOptions = IntegratorOptions {
    relative_tolerance: 2e-13,
    absolute_tolerance: 2e-13,
    initial_step: None,
    maximum_step: Some(0.01),
    maximum_steps: 100_000,
    maximum_rejections: 100,
};

fn parse(encoded: &str) -> f64 {
    let (sign, unsigned) = encoded
        .strip_prefix('-')
        .map_or((1.0, encoded), |rest| (-1.0, rest));
    let unsigned = unsigned.strip_prefix("0x").unwrap();
    let (significand, exponent) = unsigned.split_once('p').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (integer, fraction) = significand.split_once('.').unwrap_or((significand, ""));
    let digits = format!("{integer}{fraction}");
    sign * u64::from_str_radix(&digits, 16).unwrap() as f64
        * 2.0_f64.powi(exponent - 4 * fraction.len() as i32)
}

fn array<const N: usize>(value: &Value) -> [f64; N] {
    let values = value.as_array().unwrap();
    assert_eq!(values.len(), N);
    core::array::from_fn(|index| parse(values[index].as_str().unwrap()))
}

fn assert_close(context: &str, actual: f64, expected: f64, absolute: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= absolute + relative * expected.abs(),
        "{context}: {actual:.17e} != {expected:.17e}"
    );
}

fn assert_state(context: &str, actual: CartesianState, expected: CartesianState, tolerance: f64) {
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert_close(
            &format!("{context}, component {index}"),
            actual,
            expected,
            tolerance,
            tolerance,
        );
    }
}

fn central_state_jacobian<M, const P: usize>(
    model: &M,
    time: f64,
    state: CartesianState,
    parameters: [f64; P],
) -> Matrix6
where
    M: DynamicsModel<6, P>,
{
    let mut output = [[0.0; 6]; 6];
    for column in 0..6 {
        let step = 2e-6 * state[column].abs().max(1.0);
        let mut plus = state;
        let mut minus = state;
        plus[column] += step;
        minus[column] -= step;
        let mut rhs_plus = [0.0; 6];
        let mut rhs_minus = [0.0; 6];
        model.rhs(time, &plus, &parameters, &mut rhs_plus).unwrap();
        model
            .rhs(time, &minus, &parameters, &mut rhs_minus)
            .unwrap();
        for row in 0..6 {
            output[row][column] = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
        }
    }
    output
}

fn verify_jacobians<M, const P: usize>(
    model: &M,
    time: f64,
    state: CartesianState,
    parameters: [f64; P],
) where
    M: DifferentiableDynamicsModel<6, P>,
{
    let mut state_jacobian = [[0.0; 6]; 6];
    let mut parameter_jacobian = [[0.0; P]; 6];
    model
        .jacobians(
            time,
            &state,
            &parameters,
            &mut state_jacobian,
            &mut parameter_jacobian,
        )
        .unwrap();
    let finite_state = central_state_jacobian(model, time, state, parameters);
    for row in 0..6 {
        for column in 0..6 {
            assert_close(
                &format!("state Jacobian [{row},{column}]"),
                state_jacobian[row][column],
                finite_state[row][column],
                2e-8,
                2e-8,
            );
        }
    }
    for column in 0..P {
        let step = 2e-6 * parameters[column].abs().max(1.0);
        let mut plus = parameters;
        let mut minus = parameters;
        plus[column] += step;
        minus[column] -= step;
        let mut rhs_plus = [0.0; 6];
        let mut rhs_minus = [0.0; 6];
        model.rhs(time, &state, &plus, &mut rhs_plus).unwrap();
        model.rhs(time, &state, &minus, &mut rhs_minus).unwrap();
        for row in 0..6 {
            let finite = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
            assert_close(
                &format!("parameter Jacobian [{row},{column}]"),
                parameter_jacobian[row][column],
                finite,
                3e-7,
                3e-7,
            );
        }
    }
}

#[test]
fn evaluated_right_hand_sides_and_frame_conventions_are_exact() {
    let kepler = KeplerDynamics
        .evaluate(&[1.0, 2.0, 2.0, 4.0, 5.0, 6.0], 9.0)
        .unwrap();
    assert_eq!(kepler[..3], [4.0, 5.0, 6.0]);
    assert_eq!(kepler[3..], [-1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]);

    let l4 = [
        0.5 - CR3BP_MU_EARTH_MOON,
        3.0_f64.sqrt() / 2.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let l4_rhs = Cr3bpDynamics.evaluate(&l4, CR3BP_MU_EARTH_MOON).unwrap();
    for value in l4_rhs {
        assert!(value.abs() < 4e-16);
    }

    let state = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let cr3bp = Cr3bpDynamics.evaluate(&state, CR3BP_MU_EARTH_MOON).unwrap();
    let bcp_without_sun = BcpDynamics
        .evaluate(
            1.25,
            &state,
            [
                BCP_MU_EARTH_MOON,
                0.0,
                BCP_SUN_DISTANCE,
                BCP_SUN_ANGULAR_VELOCITY,
            ],
        )
        .unwrap();
    assert_eq!(bcp_without_sun, cr3bp);
}

#[test]
fn cr3bp_potential_jacobi_and_representative_invariant_are_consistent() {
    let state = [
        1.012_380_823_452_34,
        -0.042_352_352_345_4,
        0.226_343_763_21,
        -0.123_262_361_4,
        0.123_462_698_209_365,
        0.123_667_064_622,
    ];
    let potential = Cr3bpDynamics
        .effective_potential(&state, CR3BP_MU_EARTH_MOON)
        .unwrap();
    let jacobi = Cr3bpDynamics
        .jacobi_constant(&state, CR3BP_MU_EARTH_MOON)
        .unwrap();
    let velocity_squared = state[3..].iter().map(|value| value * value).sum::<f64>();
    assert_eq!(jacobi, 2.0 * potential - velocity_squared);

    let final_state = Cr3bpDynamics
        .propagate(0.0, state, 2.45, CR3BP_MU_EARTH_MOON, REFERENCE_OPTIONS)
        .unwrap()
        .state;
    let final_jacobi = Cr3bpDynamics
        .jacobi_constant(&final_state, CR3BP_MU_EARTH_MOON)
        .unwrap();
    assert!((final_jacobi - jacobi).abs() < 2e-11);
}

#[test]
fn analytic_model_jacobians_match_independent_central_differences() {
    verify_jacobians(
        &KeplerDynamics,
        0.4,
        [1.1, -0.3, 0.2, 0.1, 0.7, -0.2],
        [1.3],
    );
    verify_jacobians(
        &Cr3bpDynamics,
        0.4,
        [0.8, -0.3, 0.2, 0.1, 0.7, -0.2],
        [0.012_150_585_609_624_04],
    );
    verify_jacobians(
        &BcpDynamics,
        0.4,
        [0.8, -0.3, 0.2, 0.1, 0.7, -0.2],
        [
            BCP_MU_EARTH_MOON,
            BCP_MU_SUN,
            BCP_SUN_DISTANCE,
            BCP_SUN_ANGULAR_VELOCITY,
        ],
    );
}

#[test]
fn model_domain_errors_are_separate_from_integration_failures() {
    assert!(matches!(
        KeplerDynamics.evaluate(&[0.0; 6], 1.0),
        Err(PykepError::SingularGeometry { .. })
    ));
    assert!(matches!(
        KeplerDynamics.evaluate(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.0),
        Err(PykepError::InvalidInput {
            parameter: "mu",
            ..
        })
    ));
    assert!(matches!(
        Cr3bpDynamics.evaluate(&[0.0; 6], -0.1),
        Err(PykepError::InvalidInput {
            parameter: "mu",
            ..
        })
    ));
    assert!(matches!(
        BcpDynamics.evaluate(0.0, &[0.8, 0.0, 0.0, 0.0, 0.0, 0.0], [0.01, 1.0, 0.0, -0.9]),
        Err(PykepError::InvalidInput {
            parameter: "rho_sun",
            ..
        })
    ));

    let limited = IntegratorOptions {
        maximum_steps: 1,
        ..REFERENCE_OPTIONS
    };
    assert!(matches!(
        Cr3bpDynamics.propagate(
            0.0,
            [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
            10.0,
            CR3BP_MU_EARTH_MOON,
            limited
        ),
        Err(PykepError::IntegrationFailure {
            model: "CR3BP dynamics",
            ..
        })
    ));
}

#[test]
fn sampled_trajectories_and_stms_match_cpp_taylor_reference() {
    let document: Value = serde_json::from_str(include_str!("data/phase11-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    for model in document["models"].as_array().unwrap() {
        let name = model["name"].as_str().unwrap();
        let initial_time = parse(model["initial_time"].as_str().unwrap());
        let initial_state = array(&model["initial_state"]);
        let samples = model["samples"].as_array().unwrap();
        for (sample_index, sample) in samples.iter().enumerate() {
            let final_time = parse(sample["time"].as_str().unwrap());
            let expected = array(&sample["state"]);
            let actual = match name {
                "kepler" => {
                    KeplerDynamics
                        .propagate(
                            initial_time,
                            initial_state,
                            final_time,
                            array::<1>(&model["parameters"])[0],
                            REFERENCE_OPTIONS,
                        )
                        .unwrap()
                        .state
                }
                "cr3bp" => {
                    Cr3bpDynamics
                        .propagate(
                            initial_time,
                            initial_state,
                            final_time,
                            array::<1>(&model["parameters"])[0],
                            REFERENCE_OPTIONS,
                        )
                        .unwrap()
                        .state
                }
                "bcp" => {
                    BcpDynamics
                        .propagate(
                            initial_time,
                            initial_state,
                            final_time,
                            array(&model["parameters"]),
                            REFERENCE_OPTIONS,
                        )
                        .unwrap()
                        .state
                }
                _ => panic!("unexpected oracle model {name}"),
            };
            let tolerance = if name == "cr3bp" { 2e-9 } else { 3e-11 };
            assert_state(
                &format!("{name} sample {sample_index}"),
                actual,
                expected,
                tolerance,
            );
        }

        let final_time = parse(
            samples
                .last()
                .unwrap()
                .get("time")
                .unwrap()
                .as_str()
                .unwrap(),
        );
        let actual_stm = match name {
            "kepler" => {
                KeplerDynamics
                    .propagate_with_stm(
                        initial_time,
                        initial_state,
                        final_time,
                        array::<1>(&model["parameters"])[0],
                        REFERENCE_OPTIONS,
                    )
                    .unwrap()
                    .sensitivities
            }
            "cr3bp" => {
                Cr3bpDynamics
                    .propagate_with_stm(
                        initial_time,
                        initial_state,
                        final_time,
                        array::<1>(&model["parameters"])[0],
                        REFERENCE_OPTIONS,
                    )
                    .unwrap()
                    .sensitivities
            }
            "bcp" => {
                BcpDynamics
                    .propagate_with_stm(
                        initial_time,
                        initial_state,
                        final_time,
                        array(&model["parameters"]),
                        REFERENCE_OPTIONS,
                    )
                    .unwrap()
                    .sensitivities
            }
            _ => unreachable!(),
        };
        let expected_stm = array::<36>(&model["final_stm"]);
        for (index, (&actual, &expected)) in
            actual_stm.iter().flatten().zip(&expected_stm).enumerate()
        {
            let tolerance = if name == "cr3bp" { 2e-7 } else { 2e-9 };
            assert_close(
                &format!("{name} STM component {index}"),
                actual,
                expected,
                tolerance,
                tolerance,
            );
        }
    }
}
