// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Parity against deterministic pykep/kep3 3.0.1 foundation data.

use pykep_core::constants;
use pykep_core::math::kepler_equations::{
    elliptic_kepler_derivative, elliptic_kepler_residual, elliptic_kepler_second_derivative,
    hyperbolic_kepler_derivative, hyperbolic_kepler_residual, hyperbolic_kepler_second_derivative,
};
use pykep_core::math::stumpff::{stumpff_c, stumpff_s};
use pykep_core::time::julian::{
    jd_to_mjd, jd_to_mjd2000, mjd_to_jd, mjd_to_mjd2000, mjd2000_to_jd, mjd2000_to_mjd,
};
use pykep_core::{PykepError, Result};
use serde_json::Value;

fn parse_hex_binary64(encoded: &str) -> f64 {
    match encoded {
        "NaN" => return f64::NAN,
        "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
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

fn values(case: &Value) -> Vec<f64> {
    case["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| parse_hex_binary64(value.as_str().unwrap()))
        .collect()
}

fn evaluate(operation: &str, input: &[f64]) -> Result<f64> {
    match operation {
        "stumpff_c" => stumpff_c(input[0]),
        "stumpff_s" => stumpff_s(input[0]),
        "jd2mjd" => jd_to_mjd(input[0]),
        "jd2mjd2000" => jd_to_mjd2000(input[0]),
        "mjd2jd" => mjd_to_jd(input[0]),
        "mjd2mjd2000" => mjd_to_mjd2000(input[0]),
        "mjd20002jd" => mjd2000_to_jd(input[0]),
        "mjd20002mjd" => mjd2000_to_mjd(input[0]),
        "kep_e" => elliptic_kepler_residual(input[0], input[1], input[2]),
        "d_kep_e" => elliptic_kepler_derivative(input[0], input[1]),
        "dd_kep_e" => elliptic_kepler_second_derivative(input[0], input[1]),
        "kep_h" => hyperbolic_kepler_residual(input[0], input[1], input[2]),
        "d_kep_h" => hyperbolic_kepler_derivative(input[0], input[1]),
        "dd_kep_h" => hyperbolic_kepler_second_derivative(input[0], input[1]),
        other => panic!("unknown golden operation {other}"),
    }
}

fn assert_close(operation: &str, actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    let tolerance = 128.0 * f64::EPSILON * scale;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{operation}: {actual:.17e} != {expected:.17e}, tolerance {tolerance:.3e}"
    );
}

#[test]
fn golden_metadata_and_constants_match() {
    let document: Value = serde_json::from_str(include_str!("data/foundations-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    let expected = &document["constants"];
    let actual = [
        ("pi", constants::PI),
        ("half_pi", constants::HALF_PI),
        ("au", constants::ASTRONOMICAL_UNIT),
        ("cavendish", constants::CAVENDISH_CONSTANT),
        ("mu_sun", constants::MU_SUN),
        ("mu_earth", constants::MU_EARTH),
        ("mu_moon", constants::MU_MOON),
        ("day_to_seconds", constants::DAY_TO_SECONDS),
        ("standard_gravity", constants::STANDARD_GRAVITY),
    ];
    for (name, actual) in actual {
        let expected = parse_hex_binary64(expected[name].as_str().unwrap());
        assert_eq!(actual.to_bits(), expected.to_bits(), "{name}");
    }
}

#[test]
fn finite_golden_cases_match_or_document_stability_improvement() {
    let document: Value = serde_json::from_str(include_str!("data/foundations-v1.json")).unwrap();
    let mut compared = 0;
    for case in document["cases"].as_array().unwrap() {
        let operation = case["operation"].as_str().unwrap();
        let input = values(case);
        let expected = parse_hex_binary64(case["output"].as_str().unwrap());

        if input.iter().any(|value| !value.is_finite()) {
            assert!(matches!(
                evaluate(operation, &input),
                Err(PykepError::NonFiniteInput { .. })
            ));
            continue;
        }
        if operation.contains("kep_h") && input.last() == Some(&1.0) {
            assert!(matches!(
                evaluate(operation, &input),
                Err(PykepError::InvalidInput { .. })
            ));
            continue;
        }
        if operation.starts_with("stumpff") && input[0].abs() < 1e-7 && input[0] != 0.0 {
            let stable = evaluate(operation, &input).unwrap();
            let limit = if operation == "stumpff_c" {
                0.5
            } else {
                1.0 / 6.0
            };
            assert!((stable - limit).abs() < 1e-9);
            continue;
        }

        let actual = evaluate(operation, &input).unwrap();
        assert_close(operation, actual, expected);
        compared += 1;
    }
    assert!(compared >= 130, "only compared {compared} golden cases");
}
