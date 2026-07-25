// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Parity against deterministic pykep/kep3 3.0.1 Phase 6 data.

use pykep_core::astro::encodings::{direct_to_alpha, direct_to_eta};
use pykep_core::astro::flyby::{flyby_constraints, flyby_delta_v, flyby_outgoing_velocity};
use pykep_core::astro::lambert::LambertProblem;
use pykep_core::astro::mima::mima;
use pykep_core::astro::transfers::{bielliptic, hohmann};
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
    assert_eq!(values.len(), N);
    core::array::from_fn(|index| parse(values[index].as_str().unwrap()))
}

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs().max(1.0),
        "{actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn phase6_golden_values_match() {
    let document: Value = serde_json::from_str(include_str!("data/phase6-v1.json")).unwrap();
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    let expected = array::<4>(&document["transfer"]["hohmann"]);
    let actual = hohmann(1.1, 2.2, 1.3).unwrap();
    for (actual, expected) in [
        actual.delta_v,
        actual.time,
        actual.impulses[0],
        actual.impulses[1],
    ]
    .iter()
    .zip(expected)
    {
        close(*actual, expected, 2e-15);
    }
    let expected = array::<5>(&document["transfer"]["bielliptic"]);
    let actual = bielliptic(1.0, 15.0, 25.0, 1.0).unwrap();
    for (actual, expected) in [
        actual.delta_v,
        actual.time,
        actual.impulses[0],
        actual.impulses[1],
        actual.impulses[2],
    ]
    .iter()
    .zip(expected)
    {
        close(*actual, expected, 2e-15);
    }

    let direct = [0.1, 0.2, 0.3];
    let (alpha, total) = direct_to_alpha(&direct).unwrap();
    let expected_alpha = array::<3>(&document["encodings"]["alpha"]);
    for (actual, expected) in alpha.iter().zip(expected_alpha) {
        close(*actual, expected, 2e-15);
    }
    close(
        total,
        parse(document["encodings"]["alpha_total"].as_str().unwrap()),
        0.0,
    );
    let eta = direct_to_eta(&direct, 1.0).unwrap();
    for (actual, expected) in eta.iter().zip(array::<3>(&document["encodings"]["eta"])) {
        close(*actual, expected, 2e-15);
    }

    let incoming = [7200.0, -4567.7655, 1234.4233];
    let outgoing = [7100.0, 220.123, -144.432];
    let mu = 398_600_441_800_000.0;
    let radius = 7_015_800.000_000_001;
    let constraints = flyby_constraints(&incoming, &outgoing, mu, radius).unwrap();
    for (actual, expected) in constraints
        .iter()
        .zip(array::<2>(&document["flyby"]["constraints"]))
    {
        close(*actual, expected, 2e-15);
    }
    close(
        flyby_delta_v(&incoming, &outgoing, mu, radius).unwrap(),
        parse(document["flyby"]["delta_v"].as_str().unwrap()),
        2e-15,
    );
    #[allow(clippy::approx_constant)]
    let velocity =
        flyby_outgoing_velocity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 2.0, 3.1415 / 3.0, 1.0)
            .unwrap();
    for (actual, expected) in velocity
        .iter()
        .zip(array::<3>(&document["flyby"]["outgoing_velocity"]))
    {
        close(*actual, expected, 3e-15);
    }

    let actual = mima(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 10.0, 0.6, 4000.0).unwrap();
    let expected = array::<2>(&document["mima"]);
    close(actual.mass, expected[0], 2e-15);
    close(actual.acceleration, expected[1], 2e-15);

    for case in document["lambert"].as_array().unwrap() {
        let problem = LambertProblem::new(
            array(&case["r0"]),
            array(&case["r1"]),
            parse(case["time"].as_str().unwrap()),
            parse(case["mu"].as_str().unwrap()),
            case["clockwise"].as_bool().unwrap(),
            10,
        )
        .unwrap();
        assert_eq!(
            problem.maximum_revolutions(),
            case["maximum_revolutions"].as_u64().unwrap() as usize
        );
        let expected_solutions = case["solutions"].as_array().unwrap();
        assert_eq!(problem.solutions().len(), expected_solutions.len());
        for (actual, expected) in problem.solutions().iter().zip(expected_solutions) {
            close(actual.x, parse(expected["x"].as_str().unwrap()), 3e-13);
            assert_eq!(
                actual.iterations,
                expected["iterations"].as_u64().unwrap() as usize
            );
            for (actual, expected) in actual
                .departure_velocity
                .iter()
                .zip(array::<3>(&expected["v0"]))
            {
                close(*actual, expected, 3e-13);
            }
            for (actual, expected) in actual
                .arrival_velocity
                .iter()
                .zip(array::<3>(&expected["v1"]))
            {
                close(*actual, expected, 3e-13);
            }
        }
    }
}
