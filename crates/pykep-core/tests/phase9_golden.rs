// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! VSOP2013 parity against pykep/kep3 3.0.1 and heyoka 7.10.0.

#![cfg(feature = "vsop2013")]

use pykep_core::ephemeris::{Ephemeris, VSOP2013_MINIMUM_THRESHOLD, Vsop2013};
use serde_json::Value;

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
    core::array::from_fn(|index| parse(values[index].as_str().unwrap()))
}

fn assert_state(context: &str, actual: [f64; 6], expected: [f64; 6]) {
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = if index < 3 { 0.5 } else { 2e-7 };
        let relative = if index < 3 { 3e-12 } else { 5e-11 };
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{context}, component {index}: {actual:.17e} != {expected:.17e}"
        );
    }
}

#[test]
fn every_body_matches_the_expanded_epoch_grid() {
    let document: Value = serde_json::from_str(include_str!("data/phase9-v1.json")).unwrap();
    let threshold = parse(document["threshold"].as_str().unwrap());
    assert_eq!(threshold, VSOP2013_MINIMUM_THRESHOLD);
    let mut case_count = 0;
    for body in document["bodies"].as_array().unwrap() {
        let name = body["name"].as_str().unwrap();
        let provider = Vsop2013::with_threshold(name, threshold).unwrap();
        for case in body["cases"].as_array().unwrap() {
            case_count += 1;
            let epoch = parse(case["epoch"].as_str().unwrap());
            assert_state(
                &format!("{name} at {epoch}"),
                provider.state(epoch).unwrap(),
                array(&case["state"]),
            );
        }
    }
    assert_eq!(case_count, 54);
}

#[test]
fn coefficient_thresholds_match_upstream_selection() {
    let document: Value = serde_json::from_str(include_str!("data/phase9-v1.json")).unwrap();
    for case in document["threshold_cases"].as_array().unwrap() {
        let threshold = parse(case["threshold"].as_str().unwrap());
        let provider = Vsop2013::with_threshold("venus", threshold).unwrap();
        assert_eq!(provider.threshold(), threshold);
        assert_state(
            &format!("venus at threshold {threshold}"),
            provider.state(123.0).unwrap(),
            array(&case["state"]),
        );
    }
}
