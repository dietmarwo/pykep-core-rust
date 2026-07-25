// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Parity against deterministic pykep/kep3 3.0.1 propagation and STM data.

use pykep_core::astro::propagation::{
    propagate_lagrangian, propagate_universal, state_transition_matrix_lagrangian,
    state_transition_matrix_reynolds,
};
use serde_json::Value;

fn parse_hex_binary64(encoded: &str) -> f64 {
    let (sign, unsigned) = encoded
        .strip_prefix('-')
        .map_or((1.0, encoded), |rest| (-1.0, rest));
    let unsigned = unsigned.strip_prefix("0x").unwrap();
    let (significand, exponent) = unsigned.split_once('p').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (integer, fraction) = significand.split_once('.').unwrap_or((significand, ""));
    let digits = format!("{integer}{fraction}");
    let integer_significand = u64::from_str_radix(&digits, 16).unwrap();
    sign * integer_significand as f64 * 2.0_f64.powi(exponent - 4 * fraction.len() as i32)
}

fn array<const N: usize>(value: &Value) -> [f64; N] {
    let source = value.as_array().unwrap();
    assert_eq!(source.len(), N);
    core::array::from_fn(|index| parse_hex_binary64(source[index].as_str().unwrap()))
}

fn assert_close(context: &str, actual: f64, expected: f64, relative: f64) {
    let tolerance = relative * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: {actual:.17e} != {expected:.17e}, tolerance {tolerance:.3e}"
    );
}

#[test]
fn propagation_and_stm_golden_cases_match() {
    let document: Value = serde_json::from_str(include_str!("data/phase5-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    for case in document["cases"].as_array().unwrap() {
        let initial = array::<6>(&case["input"]);
        let time = parse_hex_binary64(case["time"].as_str().unwrap());
        let mu = parse_hex_binary64(case["mu"].as_str().unwrap());
        let category = case["category"].as_str().unwrap();
        let lagrangian = propagate_lagrangian(&initial, time, mu)
            .unwrap_or_else(|error| panic!("{category} Lagrangian propagation failed: {error}"));
        let universal = propagate_universal(&initial, time, mu)
            .unwrap_or_else(|error| panic!("{category} universal propagation failed: {error}"));
        let expected_lagrangian = array::<6>(&case["lagrangian_state"]);
        let expected_universal = array::<6>(&case["universal_state"]);
        for index in 0..6 {
            assert_close(
                &format!("{} lagrangian[{index}]", case["category"]),
                lagrangian[index],
                expected_lagrangian[index],
                2e-12,
            );
            assert_close(
                &format!("{} universal[{index}]", case["category"]),
                universal[index],
                expected_universal[index],
                2e-11,
            );
        }

        let lagrangian_stm = state_transition_matrix_lagrangian(&initial, time, mu).unwrap();
        let reynolds_stm =
            state_transition_matrix_reynolds(&initial, &lagrangian, time, mu).unwrap();
        let expected_lagrangian_stm = array::<36>(&case["lagrangian_stm"]);
        let expected_reynolds_stm = array::<36>(&case["reynolds_stm"]);
        for row in 0..6 {
            for column in 0..6 {
                assert_close(
                    &format!("{} lagrangian STM[{row}][{column}]", case["category"]),
                    lagrangian_stm[row][column],
                    expected_lagrangian_stm[row * 6 + column],
                    2e-10,
                );
                assert_close(
                    &format!("{} Reynolds STM[{row}][{column}]", case["category"]),
                    reynolds_stm[row][column],
                    expected_reynolds_stm[row * 6 + column],
                    2e-9,
                );
            }
        }
    }
}
