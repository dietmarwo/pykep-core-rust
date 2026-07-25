// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Parity against deterministic pykep/kep3 3.0.1 epoch and anomaly data.

use pykep_core::astro::anomalies::{
    eccentric_to_mean_anomaly, eccentric_to_true_anomaly, gudermannian_to_true_anomaly,
    hyperbolic_anomaly_to_mean, hyperbolic_anomaly_to_true, hyperbolic_mean_to_anomaly,
    hyperbolic_mean_to_true, mean_to_eccentric_anomaly, mean_to_true_anomaly,
    true_to_eccentric_anomaly, true_to_gudermannian_anomaly, true_to_hyperbolic_anomaly,
    true_to_hyperbolic_mean, true_to_mean_anomaly,
};
use pykep_core::time::epoch::Epoch;
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
        "m2e" => mean_to_eccentric_anomaly(input[0], input[1]),
        "e2m" => eccentric_to_mean_anomaly(input[0], input[1]),
        "e2f" => eccentric_to_true_anomaly(input[0], input[1]),
        "f2e" => true_to_eccentric_anomaly(input[0], input[1]),
        "m2f" => mean_to_true_anomaly(input[0], input[1]),
        "f2m" => true_to_mean_anomaly(input[0], input[1]),
        "n2h" => hyperbolic_mean_to_anomaly(input[0], input[1]),
        "h2n" => hyperbolic_anomaly_to_mean(input[0], input[1]),
        "h2f" => hyperbolic_anomaly_to_true(input[0], input[1]),
        "f2h" => true_to_hyperbolic_anomaly(input[0], input[1]),
        "n2f" => hyperbolic_mean_to_true(input[0], input[1]),
        "f2n" => true_to_hyperbolic_mean(input[0], input[1]),
        "zeta2f" => gudermannian_to_true_anomaly(input[0], input[1]),
        "f2zeta" => true_to_gudermannian_anomaly(input[0], input[1]),
        other => panic!("unknown golden operation {other}"),
    }
}

fn assert_close(operation: &str, actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    let tolerance = 256.0 * f64::EPSILON * scale;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{operation}: {actual:.17e} != {expected:.17e}, tolerance {tolerance:.3e}"
    );
}

#[test]
fn epoch_golden_cases_match() {
    let document: Value = serde_json::from_str(include_str!("data/phase3-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );

    for case in document["epochs"].as_array().unwrap() {
        let constructor = case["constructor"].as_str().unwrap();
        let input = case["input"].as_str().unwrap();
        let epoch = match constructor {
            "mjd2000" => Epoch::from_mjd2000(input.parse().unwrap()).unwrap(),
            "mjd" => Epoch::from_mjd(input.parse().unwrap()).unwrap(),
            "jd" => Epoch::from_jd(input.parse().unwrap()).unwrap(),
            // The calendar constructor itself has exhaustive unit coverage. The
            // oracle's canonical ISO value retains every calendar component.
            "calendar" => Epoch::from_iso(case["iso"].as_str().unwrap()).unwrap(),
            "iso" => Epoch::from_iso(input).unwrap(),
            other => panic!("unknown epoch constructor {other}"),
        };
        assert_eq!(epoch.to_iso(), case["iso"].as_str().unwrap());
        for (name, actual) in [
            ("mjd2000", epoch.mjd2000()),
            ("mjd", epoch.mjd()),
            ("jd", epoch.jd()),
        ] {
            assert_close(
                name,
                actual,
                parse_hex_binary64(case[name].as_str().unwrap()),
            );
        }
    }
}

#[test]
fn anomaly_golden_cases_match() {
    let document: Value = serde_json::from_str(include_str!("data/phase3-v1.json")).unwrap();
    let mut compared = 0;
    for case in document["anomalies"].as_array().unwrap() {
        let operation = case["operation"].as_str().unwrap();
        let input = values(case);
        let expected = parse_hex_binary64(case["output"].as_str().unwrap());
        match evaluate(operation, &input) {
            Ok(actual) if expected.is_finite() => {
                assert_close(operation, actual, expected);
                compared += 1;
            }
            Err(PykepError::InvalidInput { .. }) if expected.is_nan() => {}
            result => panic!("{operation}: unexpected result {result:?} for {expected:?}"),
        }
    }
    assert!(compared >= 130, "only compared {compared} golden cases");
}

#[test]
fn invalid_oracle_cases_map_to_explicit_errors() {
    let document: Value = serde_json::from_str(include_str!("data/phase3-v1.json")).unwrap();
    let cases = document["invalid_cases"].as_array().unwrap();
    assert!(Epoch::from_iso(cases[0]["input"].as_str().unwrap()).is_err());
    for case in &cases[1..] {
        let result = evaluate(case["operation"].as_str().unwrap(), &values(case));
        assert!(matches!(result, Err(PykepError::InvalidInput { .. })));
    }
}
