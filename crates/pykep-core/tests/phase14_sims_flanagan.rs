// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Sims–Flanagan mismatch, cut, validation, and gradient tests.

use pykep_core::PykepError;
use pykep_core::leg::{
    SimsFlanaganAlphaLeg, SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint,
};
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

fn values(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| parse(item.as_str().unwrap()))
        .collect()
}

fn controls(value: &Value) -> Vec<[f64; 3]> {
    values(value)
        .chunks_exact(3)
        .map(|chunk| chunk.try_into().unwrap())
        .collect()
}

fn endpoint(case: &Value, prefix: &str) -> SpacecraftEndpoint {
    SpacecraftEndpoint::new(
        array::<6>(&case[format!("{prefix}_state")]),
        parse(case[format!("{prefix}_mass")].as_str().unwrap()),
    )
    .unwrap()
}

fn settings(case: &Value) -> SimsFlanaganSettings {
    SimsFlanaganSettings::new(
        parse(case["time_of_flight"].as_str().unwrap()),
        parse(case["maximum_thrust"].as_str().unwrap()),
        parse(case["exhaust_velocity"].as_str().unwrap()),
        parse(case["mu"].as_str().unwrap()),
        parse(case["cut"].as_str().unwrap()),
    )
    .unwrap()
}

fn fixed_leg(case: &Value) -> SimsFlanaganLeg {
    SimsFlanaganLeg::new(
        endpoint(case, "departure"),
        controls(&case["throttles"]),
        endpoint(case, "arrival"),
        settings(case),
    )
    .unwrap()
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

#[test]
fn fixed_legs_match_cpp_constraints_and_analytic_gradients() {
    let document: Value = serde_json::from_str(include_str!("data/phase14-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    for case in document["fixed"].as_array().unwrap() {
        let leg = fixed_leg(case);
        assert_eq!(
            leg.forward_segment_count(),
            case["forward_segments"].as_u64().unwrap() as usize
        );
        assert_eq!(
            leg.backward_segment_count(),
            leg.segment_count() - leg.forward_segment_count()
        );
        assert_scaled(
            &leg.mismatch_constraints().unwrap(),
            &values(&case["mismatch"]),
            4e-12,
        );
        assert_scaled(
            &leg.throttle_constraints(),
            &values(&case["throttle_constraints"]),
            2e-15,
        );

        let jacobian = leg.mismatch_jacobian().unwrap();
        assert_scaled(
            jacobian.departure.as_flattened(),
            &values(&case["departure_jacobian"]),
            2e-10,
        );
        assert_scaled(
            jacobian.arrival.as_flattened(),
            &values(&case["arrival_jacobian"]),
            2e-10,
        );
        assert_scaled(
            &jacobian
                .controls_and_time
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            &values(&case["controls_time_jacobian"]),
            3e-10,
        );
        assert_scaled(
            &leg.throttle_jacobian()
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            &values(&case["throttle_jacobian"]),
            1e-15,
        );
    }
}

#[test]
fn analytic_gradient_matches_scaled_central_differences() {
    let document: Value = serde_json::from_str(include_str!("data/phase14-v1.json")).unwrap();
    let case = &document["fixed"][2];
    let base = fixed_leg(case);
    let analytic = base.mismatch_jacobian().unwrap();

    for column in 0..7 {
        let original = base.departure().extended_for_test();
        let step = 2e-6 * original[column].abs().max(1.0);
        let mut plus = original;
        let mut minus = original;
        plus[column] += step;
        minus[column] -= step;
        let plus_leg = SimsFlanaganLeg::new(
            SpacecraftEndpoint::new(plus[..6].try_into().unwrap(), plus[6]).unwrap(),
            base.throttles().to_vec(),
            base.arrival(),
            base.settings(),
        )
        .unwrap();
        let minus_leg = SimsFlanaganLeg::new(
            SpacecraftEndpoint::new(minus[..6].try_into().unwrap(), minus[6]).unwrap(),
            base.throttles().to_vec(),
            base.arrival(),
            base.settings(),
        )
        .unwrap();
        let plus = plus_leg.mismatch_constraints().unwrap();
        let minus = minus_leg.mismatch_constraints().unwrap();
        for row in 0..7 {
            let finite = (plus[row] - minus[row]) / (2.0 * step);
            assert!(
                (analytic.departure[row][column] - finite).abs() < 2e-6 * finite.abs().max(1.0)
            );
        }
    }

    for column in 0..7 {
        let original = base.arrival().extended_for_test();
        let step = 2e-6 * original[column].abs().max(1.0);
        let mut plus = original;
        let mut minus = original;
        plus[column] += step;
        minus[column] -= step;
        let plus_leg = SimsFlanaganLeg::new(
            base.departure(),
            base.throttles().to_vec(),
            SpacecraftEndpoint::new(plus[..6].try_into().unwrap(), plus[6]).unwrap(),
            base.settings(),
        )
        .unwrap();
        let minus_leg = SimsFlanaganLeg::new(
            base.departure(),
            base.throttles().to_vec(),
            SpacecraftEndpoint::new(minus[..6].try_into().unwrap(), minus[6]).unwrap(),
            base.settings(),
        )
        .unwrap();
        let plus = plus_leg.mismatch_constraints().unwrap();
        let minus = minus_leg.mismatch_constraints().unwrap();
        for row in 0..7 {
            let finite = (plus[row] - minus[row]) / (2.0 * step);
            assert!((analytic.arrival[row][column] - finite).abs() < 2e-6 * finite.abs().max(1.0));
        }
    }

    for control_column in 0..base.segment_count() * 3 {
        let segment = control_column / 3;
        let component = control_column % 3;
        let step = 2e-6;
        let mut plus_controls = base.throttles().to_vec();
        let mut minus_controls = base.throttles().to_vec();
        plus_controls[segment][component] += step;
        minus_controls[segment][component] -= step;
        let plus = SimsFlanaganLeg::new(
            base.departure(),
            plus_controls,
            base.arrival(),
            base.settings(),
        )
        .unwrap()
        .mismatch_constraints()
        .unwrap();
        let minus = SimsFlanaganLeg::new(
            base.departure(),
            minus_controls,
            base.arrival(),
            base.settings(),
        )
        .unwrap()
        .mismatch_constraints()
        .unwrap();
        for row in 0..7 {
            let finite = (plus[row] - minus[row]) / (2.0 * step);
            assert!(
                (analytic.controls_and_time[row][control_column] - finite).abs()
                    < 3e-6 * finite.abs().max(1.0)
            );
        }
    }

    let step = 2e-6;
    let settings = base.settings();
    let plus_settings = SimsFlanaganSettings::new(
        settings.time_of_flight + step,
        settings.maximum_thrust,
        settings.exhaust_velocity,
        settings.mu,
        settings.cut,
    )
    .unwrap();
    let minus_settings = SimsFlanaganSettings::new(
        settings.time_of_flight - step,
        settings.maximum_thrust,
        settings.exhaust_velocity,
        settings.mu,
        settings.cut,
    )
    .unwrap();
    let plus = SimsFlanaganLeg::new(
        base.departure(),
        base.throttles().to_vec(),
        base.arrival(),
        plus_settings,
    )
    .unwrap()
    .mismatch_constraints()
    .unwrap();
    let minus = SimsFlanaganLeg::new(
        base.departure(),
        base.throttles().to_vec(),
        base.arrival(),
        minus_settings,
    )
    .unwrap()
    .mismatch_constraints()
    .unwrap();
    for row in 0..7 {
        let finite = (plus[row] - minus[row]) / (2.0 * step);
        assert!(
            (analytic.controls_and_time[row][base.segment_count() * 3] - finite).abs()
                < 3e-6 * finite.abs().max(1.0)
        );
    }
}

#[test]
fn alpha_legs_match_cpp_and_normalized_weights() {
    let document: Value = serde_json::from_str(include_str!("data/phase14-v1.json")).unwrap();
    let fixed = document["fixed"].as_array().unwrap();
    let alpha = document["alpha"].as_array().unwrap();
    for (index, case) in alpha.iter().enumerate() {
        let base = if index == 0 { &fixed[0] } else { &fixed[2] };
        let leg = SimsFlanaganAlphaLeg::new(
            endpoint(base, "departure"),
            controls(&base["throttles"]),
            values(&case["durations"]),
            endpoint(base, "arrival"),
            SimsFlanaganSettings::new(
                parse(base["time_of_flight"].as_str().unwrap()),
                parse(base["maximum_thrust"].as_str().unwrap()),
                parse(base["exhaust_velocity"].as_str().unwrap()),
                parse(base["mu"].as_str().unwrap()),
                parse(case["cut"].as_str().unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            leg.forward_segment_count(),
            case["forward_segments"].as_u64().unwrap() as usize
        );
        assert_scaled(
            &leg.mismatch_constraints().unwrap(),
            &values(&case["mismatch"]),
            4e-12,
        );
        assert_scaled(
            &leg.throttle_constraints(),
            &values(&case["throttle_constraints"]),
            2e-15,
        );
    }

    let base = fixed_leg(&fixed[2]);
    let weighted = SimsFlanaganAlphaLeg::from_time_weights(
        base.departure(),
        base.throttles().to_vec(),
        vec![1.0, 2.0, 3.0, 4.0],
        base.arrival(),
        base.settings(),
    )
    .unwrap();
    assert!(
        (weighted.segment_durations().iter().sum::<f64>() - base.settings().time_of_flight).abs()
            < 2e-16
    );
    assert_eq!(weighted.segment_durations(), &[0.13, 0.26, 0.39, 0.52]);
}

#[test]
fn ballistic_one_segment_cuts_and_zero_throttle_gradient_are_well_defined() {
    let departure = SpacecraftEndpoint::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2.0).unwrap();
    let arrival_state =
        pykep_core::astro::propagation::propagate_lagrangian(&departure.state, 0.7, 1.0).unwrap();
    let arrival = SpacecraftEndpoint::new(arrival_state, 2.0).unwrap();
    for cut in [0.0, 0.2, 0.5, 1.0] {
        let leg = SimsFlanaganLeg::new(
            departure,
            vec![[0.0; 3]],
            arrival,
            SimsFlanaganSettings::new(0.7, 0.0, 2.0, 1.0, cut).unwrap(),
        )
        .unwrap();
        assert_scaled(&leg.mismatch_constraints().unwrap(), &[0.0; 7], 2e-14);
        assert!(
            leg.mismatch_jacobian()
                .unwrap()
                .controls_and_time
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn malformed_legs_are_rejected_at_construction() {
    let endpoint = SpacecraftEndpoint::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1.0).unwrap();
    let settings = SimsFlanaganSettings::new(1.0, 0.1, 2.0, 1.0, 0.5).unwrap();
    assert!(matches!(
        SimsFlanaganLeg::new(endpoint, vec![], endpoint, settings),
        Err(PykepError::InvalidInput {
            parameter: "throttles",
            ..
        })
    ));
    assert!(SimsFlanaganSettings::new(-1.0, 0.1, 2.0, 1.0, 0.5).is_err());
    assert!(SimsFlanaganSettings::new(1.0, -0.1, 2.0, 1.0, 0.5).is_err());
    assert!(SimsFlanaganSettings::new(1.0, 0.1, 0.0, 1.0, 0.5).is_err());
    assert!(SimsFlanaganSettings::new(1.0, 0.1, 2.0, 0.0, 0.5).is_err());
    assert!(SimsFlanaganSettings::new(1.0, 0.1, 2.0, 1.0, 1.1).is_err());
    assert!(SpacecraftEndpoint::new([0.0; 6], 1.0).is_err());
    assert!(SpacecraftEndpoint::new(endpoint.state, 0.0).is_err());
    assert!(SimsFlanaganLeg::new(endpoint, vec![[f64::NAN; 3]], endpoint, settings).is_err());
    assert!(matches!(
        SimsFlanaganAlphaLeg::new(
            endpoint,
            vec![[0.0; 3], [0.0; 3]],
            vec![1.0],
            endpoint,
            settings
        ),
        Err(PykepError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(
        SimsFlanaganAlphaLeg::from_time_weights(
            endpoint,
            vec![[0.0; 3]],
            vec![0.0],
            endpoint,
            settings
        )
        .is_err()
    );
}

trait EndpointTestExt {
    fn extended_for_test(self) -> [f64; 7];
}

impl EndpointTestExt for SpacecraftEndpoint {
    fn extended_for_test(self) -> [f64; 7] {
        [
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
            self.state[4],
            self.state[5],
            self.mass,
        ]
    }
}
