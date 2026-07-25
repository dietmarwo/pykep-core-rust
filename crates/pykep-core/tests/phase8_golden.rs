// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! JPL low-precision ephemeris parity against pykep/kep3 3.0.1.

use pykep_core::constants::MU_SUN;
use pykep_core::ephemeris::{ElementRepresentation, Ephemeris, JplLowPrecision};
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

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn all_supported_bodies_match_golden_elements_states_and_metadata() {
    let document: Value = serde_json::from_str(include_str!("data/phase8-v1.json")).unwrap();
    let bodies = document["bodies"].as_array().unwrap();
    assert_eq!(bodies.len(), 8);
    let mut case_count = 0;
    for body in bodies {
        let name = body["name"].as_str().unwrap();
        let provider = JplLowPrecision::new(name).unwrap();
        assert_eq!(provider.name(), format!("{name}(jpl_lp)"));
        let metadata = provider.metadata();
        assert_eq!(metadata.central_mu, Some(MU_SUN));
        assert_eq!(
            metadata.body_mu,
            Some(parse(body["body_mu"].as_str().unwrap()))
        );
        assert_eq!(
            metadata.radius,
            Some(parse(body["radius"].as_str().unwrap()))
        );
        assert_eq!(
            metadata.safe_radius,
            Some(parse(body["safe_radius"].as_str().unwrap()))
        );

        for case in body["cases"].as_array().unwrap() {
            case_count += 1;
            let epoch = parse(case["epoch"].as_str().unwrap());
            let expected_elements = array::<6>(&case["elements"]);
            let actual_elements = provider
                .elements(epoch, ElementRepresentation::ClassicalTrue)
                .unwrap();
            for (actual, expected) in actual_elements.iter().zip(expected_elements) {
                assert_near(*actual, expected, 3e-14);
            }

            let expected_state = array::<6>(&case["state"]);
            let actual_state = provider.state(epoch).unwrap();
            for (actual, expected) in actual_state.iter().zip(expected_state) {
                assert_near(*actual, expected, 3e-13);
            }
        }
    }
    assert_eq!(case_count, 40);
}

#[test]
fn supported_names_batches_and_validity_window_are_explicit() {
    assert_eq!(
        JplLowPrecision::supported_bodies(),
        [
            "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"
        ]
    );
    let earth = JplLowPrecision::new("EaRtH").unwrap();
    let epochs = [-73_047.999, 0.0, 18_262.999];
    let batch = earth.states(&epochs).unwrap();
    for (state, epoch) in batch.iter().zip(epochs) {
        assert_eq!(*state, earth.state(epoch).unwrap());
    }
    assert!(earth.state(-73_048.0).is_err());
    assert!(earth.state(18_263.0).is_err());
    assert!(JplLowPrecision::new("pluto").is_err());
}
