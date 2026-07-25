// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Keplerian ephemeris parity against pykep/kep3 3.0.1.

use pykep_core::astro::elements::ClassicalElements;
use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris};
use pykep_core::time::epoch::Epoch;
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

#[test]
fn keplerian_provider_matches_golden_states_and_metadata() {
    let document: Value = serde_json::from_str(include_str!("data/phase7-v1.json")).unwrap();
    let reference =
        Epoch::from_mjd2000(parse(document["reference_epoch"].as_str().unwrap())).unwrap();
    let provider = KeplerianEphemeris::from_classical(
        reference,
        ClassicalElements::from(array::<6>(&document["elements"])),
        parse(document["central_mu"].as_str().unwrap()),
        "oracle",
        Some(parse(document["body_mu"].as_str().unwrap())),
        Some(parse(document["radius"].as_str().unwrap())),
        Some(parse(document["safe_radius"].as_str().unwrap())),
    )
    .unwrap();
    let metadata = provider.metadata();
    assert_eq!(metadata.body_mu, Some(1.2));
    assert_eq!(metadata.radius, Some(2.2));
    assert_eq!(metadata.safe_radius, Some(2.9));
    let expected_period = parse(document["period"].as_str().unwrap());
    let actual_period = provider.period(reference.mjd2000()).unwrap().unwrap();
    assert!((actual_period - expected_period).abs() < 2e-14);
    for case in document["states"].as_array().unwrap() {
        let epoch = parse(case["epoch"].as_str().unwrap());
        let actual = provider.state(epoch).unwrap();
        let expected = array::<6>(&case["state"]);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2e-11 * expected.abs().max(1.0),
                "{actual:.17e} != {expected:.17e}"
            );
        }
    }
}
