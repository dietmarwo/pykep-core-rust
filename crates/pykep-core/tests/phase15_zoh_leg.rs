// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Generic ZOH-leg parity, sensitivity, cut, and validation tests.

use pykep_core::PykepError;
use pykep_core::dynamics::zoh::{
    ZeroOrderHoldModel, ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics,
    ZohSolarSailDynamics,
};
use pykep_core::integration::{IntegrationMethod, IntegratorOptions};
use pykep_core::leg::{ZohKeplerLeg, ZohLeg, ZohLegMismatchJacobian, evaluate_zoh_mismatch_batch};
use serde_json::Value;

const OPTIONS: IntegratorOptions = IntegratorOptions {
    relative_tolerance: 2e-13,
    absolute_tolerance: 2e-13,
    initial_step: None,
    maximum_step: Some(0.005),
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

fn values(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| parse(item.as_str().unwrap()))
        .collect()
}

fn array<const N: usize>(value: &Value) -> [f64; N] {
    values(value).try_into().unwrap()
}

fn controls<const C: usize>(value: &Value) -> Vec<[f64; C]> {
    values(value)
        .chunks_exact(C)
        .map(|chunk| chunk.try_into().unwrap())
        .collect()
}

fn assert_scaled(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * actual.abs().max(expected.abs()).max(1.0),
            "component {index}: {actual:.17e} != {expected:.17e}"
        );
    }
}

fn validate_case<
    M,
    const N: usize,
    const C: usize,
    const K: usize,
    const P: usize,
    const W: usize,
>(
    model: M,
    case: &Value,
) where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    let leg = ZohLeg::<M, N, C, K, P, W>::new(
        model,
        array(&case["initial_state"]),
        controls(&case["controls"]),
        array(&case["final_state"]),
        values(&case["time_grid"]),
        array(&case["constants"]),
        parse(case["cut"].as_str().unwrap()),
        OPTIONS,
    )
    .unwrap();
    assert_eq!(
        leg.forward_segment_count(),
        case["forward_segments"].as_u64().unwrap() as usize
    );
    assert_scaled(
        &leg.mismatch_constraints().unwrap(),
        &values(&case["mismatch"]),
        3e-9,
    );
    let jacobian = leg.mismatch_jacobian().unwrap();
    assert_jacobian(&jacobian, case, 3e-5);
}

fn assert_jacobian<const N: usize>(
    actual: &ZohLegMismatchJacobian<N>,
    case: &Value,
    tolerance: f64,
) {
    assert_scaled(
        actual.initial_state.as_flattened(),
        &values(&case["initial_jacobian"]),
        tolerance,
    );
    assert_scaled(
        actual.final_state.as_flattened(),
        &values(&case["final_jacobian"]),
        tolerance,
    );
    assert_scaled(
        &actual
            .controls
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &values(&case["controls_jacobian"]),
        tolerance,
    );
    assert_scaled(
        &actual
            .time_grid
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &values(&case["time_grid_jacobian"]),
        tolerance,
    );
}

#[test]
fn every_supported_model_matches_the_cpp_leg_oracle() {
    let document: Value = serde_json::from_str(include_str!("data/phase15-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    for case in document["legs"].as_array().unwrap() {
        match case["name"].as_str().unwrap() {
            "zoh_kepler" => validate_case::<_, 7, 4, 1, 5, 11>(ZohKeplerDynamics, case),
            "zoh_cr3bp" => validate_case::<_, 7, 4, 2, 6, 11>(ZohCr3bpDynamics, case),
            "zoh_equinoctial" => {
                validate_case::<_, 7, 4, 1, 5, 11>(ZohEquinoctialDynamics, case);
            }
            "zoh_solar_sail" => {
                validate_case::<_, 6, 2, 1, 3, 8>(ZohSolarSailDynamics, case);
            }
            other => panic!("unexpected model {other}"),
        }
    }
}

fn kepler_case() -> ZohKeplerLeg {
    ZohKeplerLeg::new(
        ZohKeplerDynamics,
        [1.0, 0.1, -0.05, -0.1, 0.95, 0.03, 1.2],
        vec![
            [0.02, 1.0, 0.0, 0.0],
            [0.01, 0.0, 1.0, 0.0],
            [0.015, 0.0, 0.0, 1.0],
        ],
        [0.4, 0.9, 0.08, -0.8, 0.3, -0.04, 1.1],
        vec![0.1, 0.35, 0.7, 1.0],
        [0.2],
        0.5,
        OPTIONS,
    )
    .unwrap()
}

fn finite_difference(
    base: &ZohKeplerLeg,
    mutate: impl Fn(&mut ZohKeplerInputs, f64),
    step: f64,
) -> [f64; 7] {
    let mut plus = ZohKeplerInputs::from(base);
    let mut minus = plus.clone();
    mutate(&mut plus, step);
    mutate(&mut minus, -step);
    let plus = plus.leg().mismatch_constraints().unwrap();
    let minus = minus.leg().mismatch_constraints().unwrap();
    core::array::from_fn(|row| (plus[row] - minus[row]) / (2.0 * step))
}

#[derive(Clone)]
struct ZohKeplerInputs {
    initial: [f64; 7],
    controls: Vec<[f64; 4]>,
    final_state: [f64; 7],
    grid: Vec<f64>,
}

impl From<&ZohKeplerLeg> for ZohKeplerInputs {
    fn from(leg: &ZohKeplerLeg) -> Self {
        Self {
            initial: leg.initial_state(),
            controls: leg.schedule().controls().to_vec(),
            final_state: leg.final_state(),
            grid: leg.schedule().boundaries().to_vec(),
        }
    }
}

impl ZohKeplerInputs {
    fn leg(&self) -> ZohKeplerLeg {
        ZohKeplerLeg::new(
            ZohKeplerDynamics,
            self.initial,
            self.controls.clone(),
            self.final_state,
            self.grid.clone(),
            [0.2],
            0.5,
            OPTIONS,
        )
        .unwrap()
    }
}

#[test]
fn all_sensitivity_groups_match_scale_adjusted_central_differences() {
    let leg = kepler_case();
    let analytic = leg.mismatch_jacobian().unwrap();

    for column in 0..7 {
        let scale = leg.initial_state()[column].abs().max(1.0);
        let finite = finite_difference(
            &leg,
            |inputs, delta| inputs.initial[column] += delta,
            2e-6 * scale,
        );
        assert_scaled(
            &analytic
                .initial_state
                .iter()
                .map(|row| row[column])
                .collect::<Vec<_>>(),
            &finite,
            2e-5,
        );
    }
    for column in 0..7 {
        let scale = leg.final_state()[column].abs().max(1.0);
        let finite = finite_difference(
            &leg,
            |inputs, delta| inputs.final_state[column] += delta,
            2e-6 * scale,
        );
        assert_scaled(
            &analytic
                .final_state
                .iter()
                .map(|row| row[column])
                .collect::<Vec<_>>(),
            &finite,
            2e-5,
        );
    }
    for column in 0..12 {
        let finite = finite_difference(
            &leg,
            |inputs, delta| inputs.controls[column / 4][column % 4] += delta,
            2e-6,
        );
        assert_scaled(
            &analytic
                .controls
                .iter()
                .map(|row| row[column])
                .collect::<Vec<_>>(),
            &finite,
            3e-5,
        );
    }
    for column in 0..4 {
        let finite = finite_difference(&leg, |inputs, delta| inputs.grid[column] += delta, 2e-6);
        assert_scaled(
            &analytic
                .time_grid
                .iter()
                .map(|row| row[column])
                .collect::<Vec<_>>(),
            &finite,
            3e-5,
        );
    }
}

#[test]
fn histories_cuts_and_batch_evaluation_preserve_order() {
    let base = kepler_case();
    let history = base.state_history(5).unwrap();
    assert_eq!(history.forward.len(), 1);
    assert_eq!(history.backward.len(), 2);
    assert!(
        history
            .forward
            .iter()
            .chain(&history.backward)
            .all(|segment| segment.len() == 5)
    );
    let from_history: [f64; 7] = core::array::from_fn(|row| {
        history.forward.last().unwrap().last().unwrap()[row]
            - history.backward.last().unwrap().last().unwrap()[row]
    });
    assert_scaled(&from_history, &base.mismatch_constraints().unwrap(), 2e-12);
    assert!(base.state_history(1).is_err());

    let make_cut = |cut| {
        ZohKeplerLeg::new(
            ZohKeplerDynamics,
            base.initial_state(),
            base.schedule().controls().to_vec(),
            base.final_state(),
            base.schedule().boundaries().to_vec(),
            base.constants(),
            cut,
            OPTIONS,
        )
        .unwrap()
    };
    let zero = make_cut(0.0);
    let one = make_cut(1.0);
    assert_eq!(zero.forward_segment_count(), 0);
    assert_eq!(zero.backward_segment_count(), 3);
    assert_eq!(one.forward_segment_count(), 3);
    assert_eq!(one.backward_segment_count(), 0);
    let batch = evaluate_zoh_mismatch_batch(&[zero.clone(), base.clone(), one.clone()]).unwrap();
    assert_eq!(batch[0], zero.mismatch_constraints().unwrap());
    assert_eq!(batch[1], base.mismatch_constraints().unwrap());
    assert_eq!(batch[2], one.mismatch_constraints().unwrap());
}

#[test]
fn selected_taylor_zoh_leg_matches_dop853_constraints_and_history() {
    let base = kepler_case();
    let dop853 = base
        .mismatch_constraints_with_method(IntegrationMethod::Dop853)
        .unwrap();
    let taylor = base
        .mismatch_constraints_with_method(IntegrationMethod::Taylor)
        .unwrap();
    assert_eq!(dop853, base.mismatch_constraints().unwrap());
    assert_scaled(&taylor, &dop853, 2e-10);

    let taylor_history = base
        .state_history_with_method(5, IntegrationMethod::Taylor)
        .unwrap();
    assert_eq!(taylor_history.forward.len(), 1);
    assert_eq!(taylor_history.backward.len(), 2);
    assert!(
        taylor_history
            .forward
            .iter()
            .chain(&taylor_history.backward)
            .all(|segment| segment.len() == 5)
    );
    let from_history: [f64; 7] = core::array::from_fn(|row| {
        taylor_history.forward.last().unwrap().last().unwrap()[row]
            - taylor_history.backward.last().unwrap().last().unwrap()[row]
    });
    assert_scaled(&from_history, &taylor, 2e-12);
    assert!(
        base.state_history_with_method(1, IntegrationMethod::Taylor)
            .is_err()
    );
}

#[test]
fn malformed_inputs_and_segment_failures_are_contextual() {
    let base = kepler_case();
    let construct = |controls, grid, cut, options| {
        ZohKeplerLeg::new(
            ZohKeplerDynamics,
            base.initial_state(),
            controls,
            base.final_state(),
            grid,
            base.constants(),
            cut,
            options,
        )
    };
    assert!(construct(vec![], vec![0.0], 0.5, OPTIONS).is_err());
    assert!(construct(vec![[0.0; 4]], vec![0.0, 0.0], 0.5, OPTIONS).is_err());
    assert!(construct(vec![[0.0; 4]], vec![0.0, f64::NAN], 0.5, OPTIONS).is_err());
    assert!(construct(vec![[0.0; 4]], vec![0.0, 1.0], -0.1, OPTIONS).is_err());
    assert!(
        construct(
            vec![[0.0; 4]],
            vec![0.0, 1.0],
            0.5,
            IntegratorOptions {
                relative_tolerance: 0.0,
                ..OPTIONS
            }
        )
        .is_err()
    );

    let limited = ZohKeplerLeg::new(
        ZohKeplerDynamics,
        base.initial_state(),
        vec![[0.01, 1.0, 0.0, 0.0]],
        base.final_state(),
        vec![0.0, 20.0],
        base.constants(),
        1.0,
        IntegratorOptions {
            maximum_step: Some(1e-4),
            maximum_steps: 1,
            ..OPTIONS
        },
    )
    .unwrap();
    let error = limited.mismatch_constraints().unwrap_err();
    assert!(matches!(error, PykepError::IntegrationFailure { .. }));
    let message = error.to_string();
    assert!(message.contains("forward segment 0"));
    assert!(message.contains("[0, 20]"));
}
