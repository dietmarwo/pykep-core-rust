// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Public Taylor-backend and independent heyoka reference tests.

use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohKeplerDynamics, propagate_schedule_with_method,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{
    AdaptiveIntegrator, InitialValueProblem, IntegrationMethod, IntegratorOptions,
    SensitivityProblem, Taylor,
};
use serde_json::Value;

fn array<const N: usize>(value: &Value) -> [f64; N] {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
        .unwrap()
        .try_into()
        .unwrap()
}

fn options() -> IntegratorOptions {
    IntegratorOptions {
        relative_tolerance: 1e-14,
        absolute_tolerance: 1e-14,
        maximum_steps: 1_000_000,
        ..IntegratorOptions::default()
    }
}

fn assert_state<const N: usize>(actual: &[f64; N], expected: &[f64; N], tolerance: f64) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "component {index}: actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}

#[test]
fn heyoka_reference_states_and_kepler_stm_match() {
    let document: Value = serde_json::from_str(include_str!("data/taylor-heyoka-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(document["heyoka_version"], "7.10.1");

    for case in document["cases"].as_array().unwrap() {
        let initial = array::<6>(&case["initial_state"]);
        let final_time = case["final_time"].as_f64().unwrap();
        let expected = array::<6>(&case["state"]);
        let actual = match case["name"].as_str().unwrap() {
            "kepler" => {
                Taylor
                    .propagate(
                        &KeplerDynamics,
                        InitialValueProblem::new(
                            0.0,
                            initial,
                            final_time,
                            array(&case["parameters"]),
                        ),
                        options(),
                    )
                    .unwrap()
                    .state
            }
            "cr3bp" => {
                Taylor
                    .propagate(
                        &Cr3bpDynamics,
                        InitialValueProblem::new(
                            0.0,
                            initial,
                            final_time,
                            array(&case["parameters"]),
                        ),
                        options(),
                    )
                    .unwrap()
                    .state
            }
            "bcp" => {
                Taylor
                    .propagate(
                        &BcpDynamics,
                        InitialValueProblem::new(
                            0.0,
                            initial,
                            final_time,
                            array(&case["parameters"]),
                        ),
                        options(),
                    )
                    .unwrap()
                    .state
            }
            name => panic!("unexpected reference case {name}"),
        };
        assert_state(&actual, &expected, 3e-11);
    }

    let case = &document["cases"][0];
    let initial = array::<6>(&case["initial_state"]);
    let sensitivity = Taylor
        .propagate_with_sensitivities(
            &KeplerDynamics,
            SensitivityProblem {
                nominal: InitialValueProblem::new(
                    0.0,
                    initial,
                    case["final_time"].as_f64().unwrap(),
                    array(&case["parameters"]),
                ),
                initial_sensitivities: core::array::from_fn(|row| {
                    core::array::from_fn(|column| if row == column { 1.0 } else { 0.0 })
                }),
                parameter_seeds: [[0.0; 6]],
            },
            options(),
        )
        .unwrap();
    let expected = case["stm"].as_array().unwrap();
    for (row, expected_row) in expected.iter().enumerate() {
        assert_state(
            &sensitivity.sensitivities[row],
            &array::<6>(expected_row),
            3e-7,
        );
    }
}

#[test]
fn runtime_selector_and_zoh_schedule_are_publicly_callable() {
    assert_eq!(
        AdaptiveIntegrator::default().method(),
        IntegrationMethod::Taylor
    );
    let initial = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
    let selected = KeplerDynamics
        .propagate_with_method(0.0, initial, 0.5, 1.0, options(), IntegrationMethod::Taylor)
        .unwrap();
    let direct = Taylor
        .propagate(
            &KeplerDynamics,
            InitialValueProblem::new(0.0, initial, 0.5, [1.0]),
            options(),
        )
        .unwrap();
    assert_eq!(selected.state, direct.state);
    let default = KeplerDynamics
        .propagate(0.0, initial, 0.5, 1.0, options())
        .unwrap();
    assert_eq!(default.state, direct.state);

    let schedule = ControlSchedule::new(vec![0.0, 0.25], vec![[0.02, 0.3, -0.4, 0.5]]).unwrap();
    let result = propagate_schedule_with_method(
        &ZohKeplerDynamics,
        &schedule,
        [0.8, -0.2, 0.1, 0.03, 1.0, 0.02, 1.1],
        [0.01],
        options(),
        IntegrationMethod::Taylor,
    )
    .unwrap();
    assert_eq!(result.time, 0.25);
}

#[test]
fn batch_keeps_order_with_different_directions_and_a_partial_group() {
    let base = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let jobs = [0.25, -0.25, 0.0, 0.12, 0.3].map(|duration| {
        InitialValueProblem::new(
            0.3,
            base,
            0.3 + duration,
            [
                0.012_150_585_609_624_04,
                328_900.56,
                389.172,
                -0.925_195_985_520_347,
            ],
        )
    });
    let batch = Taylor
        .propagate_batch(&BcpDynamics, &jobs, options(), 2)
        .unwrap();
    for (actual, job) in batch.iter().zip(jobs) {
        let expected = Taylor.propagate(&BcpDynamics, job, options()).unwrap();
        assert_eq!(actual.time, job.final_time);
        assert_state(&actual.state, &expected.state, 3e-13);
    }
    let mut jobs = jobs;
    jobs[1].parameters[0] = f64::NAN;
    assert_eq!(
        Taylor
            .propagate_batch(&BcpDynamics, &jobs, options(), 2)
            .unwrap_err(),
        Taylor
            .propagate(&BcpDynamics, jobs[1], options())
            .unwrap_err(),
    );
}

#[test]
fn circular_orbit_plane_event_selects_direction() {
    use pykep_core::integration::{EventDirection, TaylorEvent, Termination};

    let event = TaylorEvent::Plane {
        normal: [1.0, 0.0, 0.0],
        offset: 0.0,
    };
    for (end, first_sign) in [(6.0, 1.0), (-6.0, -1.0)] {
        let problem = InitialValueProblem::new(0.0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], end, [1.0]);
        let result = Taylor
            .propagate_until_event(
                &KeplerDynamics,
                problem,
                options(),
                event,
                EventDirection::Increasing,
            )
            .unwrap();
        assert_eq!(result.termination, Termination::Event);
        assert!((result.time - first_sign * 3.0 * core::f64::consts::FRAC_PI_2).abs() < 1e-11);
        assert!(result.state[0].abs() < 1e-13);
        let history = Taylor
            .propagate_with_events(
                &KeplerDynamics,
                problem,
                options(),
                event,
                EventDirection::Either,
                2,
            )
            .unwrap();
        assert_eq!(history.hits.len(), 2);
        assert!((history.hits[0].time - first_sign * core::f64::consts::FRAC_PI_2).abs() < 1e-11);
        assert_eq!(
            history.propagation,
            Taylor
                .propagate(&KeplerDynamics, problem, options())
                .unwrap()
        );
        assert!(
            Taylor
                .propagate_with_events(
                    &KeplerDynamics,
                    problem,
                    options(),
                    event,
                    EventDirection::Either,
                    1
                )
                .is_err()
        );
    }
}

#[test]
fn pontryagin_variation_includes_control_dependence() {
    use pykep_core::dynamics::pontryagin::CartesianMassOptimal;
    let mut state = [1., 0., 0., 0., 1., 0., 10., 1., 1., 1., 1., 1., 1., 1.];
    let mut seed = [[0.0]; 14];
    seed[10][0] = 1.0;
    for costate in [-1.0, 1.0] {
        state[13] = costate;
        let mut nominal = InitialValueProblem::new(0.0, state, 0.1, [1.0, 0.01, 1.0, 0.5, 1.0]);
        let result = Taylor
            .propagate_with_sensitivities(
                &CartesianMassOptimal,
                SensitivityProblem {
                    nominal,
                    initial_sensitivities: seed,
                    parameter_seeds: [[0.]; 5],
                },
                options(),
            )
            .unwrap();
        nominal.initial_state[10] += 1e-5;
        let plus = Taylor
            .propagate(&CartesianMassOptimal, nominal, options())
            .unwrap();
        nominal.initial_state[10] -= 2e-5;
        let minus = Taylor
            .propagate(&CartesianMassOptimal, nominal, options())
            .unwrap();
        for (row, column) in result.sensitivities.iter().enumerate() {
            let reference = (plus.state[row] - minus.state[row]) / 2e-5;
            assert!(
                (column[0] - reference).abs() < 2e-8,
                "costate {costate}, row {row}"
            );
        }
    }
}

#[test]
fn complete_zoh_jacobian_includes_shared_switch_time() {
    use pykep_core::leg::ZohKeplerLeg;

    let make = |middle| {
        ZohKeplerLeg::new(
            ZohKeplerDynamics,
            [1., 0., 0., 0., 1., 0., 1.],
            vec![
                [0.1, 0.3, 0.4, 0.],
                [0.2, -0.3, 0.1, 0.],
                [0.05, 0.3, 0.4, 0.],
            ],
            [0.9, 0.4, 0., -0.4, 0.9, 0., 0.99],
            vec![0., middle, 0.3, 0.4],
            [0.02],
            0.7,
            options(),
        )
        .unwrap()
    };
    let jac = make(0.15)
        .mismatch_jacobian_with_method(IntegrationMethod::Taylor)
        .unwrap();
    let plus = make(0.15 + 1e-5)
        .mismatch_constraints_with_method(IntegrationMethod::Taylor)
        .unwrap();
    let minus = make(0.15 - 1e-5)
        .mismatch_constraints_with_method(IntegrationMethod::Taylor)
        .unwrap();
    for i in 0..7 {
        assert!((jac.time_grid[i][1] - (plus[i] - minus[i]) / 2e-5).abs() < 2e-8);
    }
}

#[test]
fn every_nominal_model_matches_official_pykep() {
    use pykep_core::dynamics::{pontryagin::*, zoh::*};
    use pykep_core::integration::TaylorDynamicsModel;

    fn compare<M, const N: usize, const P: usize>(model: M, case: &Value)
    where
        M: TaylorDynamicsModel<N, P>,
    {
        let problem = InitialValueProblem::new(
            0.0,
            array(&case["state"]),
            case["final_time"].as_f64().unwrap(),
            array(&case["parameters"]),
        );
        let result = Taylor
            .propagate(&model, problem, IntegratorOptions::default())
            .unwrap();
        assert_state(&result.state, &array(&case["reference"]), 2e-11);
    }

    let fixtures: Value = serde_json::from_str(include_str!("data/taylor-pykep-v1.json")).unwrap();
    let cases = fixtures["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 11);
    for case in cases {
        match case["model"].as_str().unwrap() {
            "kepler" => compare(KeplerDynamics, case),
            "cr3bp" => compare(Cr3bpDynamics, case),
            "bcp" => compare(BcpDynamics, case),
            "zoh_kepler" => compare(ZohKeplerDynamics, case),
            "zoh_cr3bp" => compare(ZohCr3bpDynamics, case),
            "zoh_equinoctial" => compare(ZohEquinoctialDynamics, case),
            "solar_sail" => compare(ZohSolarSailDynamics, case),
            "cartesian_mass" => compare(CartesianMassOptimal, case),
            "cartesian_time" => compare(CartesianTimeOptimal, case),
            "equinoctial_mass" => compare(EquinoctialMassOptimal, case),
            "equinoctial_time" => compare(EquinoctialTimeOptimal, case),
            name => panic!("unexpected model {name}"),
        }
    }
}

#[test]
fn taylor_batch_keeps_the_first_error_across_full_groups_and_tail() {
    let valid = InitialValueProblem::new(
        0.3,
        [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
        0.55,
        [
            0.012_150_585_609_624_04,
            328_900.56,
            389.172,
            -0.925_195_985_520_347,
        ],
    );
    let mut jobs = [valid; 9];
    jobs[3].initial_time = f64::NAN;
    jobs[6].initial_state[1] = f64::NAN;
    jobs[8].parameters[0] = f64::NAN;
    let first = Taylor
        .propagate(&BcpDynamics, jobs[3], options())
        .unwrap_err();
    for workers in [1, 4] {
        assert_eq!(
            Taylor
                .propagate_batch(&BcpDynamics, &jobs, options(), workers)
                .unwrap_err(),
            first
        );
    }
}

#[test]
fn mixed_seed_widths_match_independent_non_autonomous_variations() {
    use pykep_core::integration::Dop853;

    fn check<const W: usize>() {
        let problem: SensitivityProblem<6, 4, W> = SensitivityProblem {
            nominal: InitialValueProblem::new(
                0.37,
                [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
                0.12,
                [
                    0.012_150_585_609_624_04,
                    328_900.56,
                    389.172,
                    -0.925_195_985_520_347,
                ],
            ),
            initial_sensitivities: std::array::from_fn(|i| {
                std::array::from_fn(|j| f64::from(i == j % 6) + 0.02 * ((i + 2 * j) % 7) as f64)
            }),
            parameter_seeds: std::array::from_fn(|i| {
                std::array::from_fn(|j| [1e-4, 0.5, 0.1, 0.03][i] * (j + 1) as f64)
            }),
        };
        let candidate = Taylor
            .propagate_with_sensitivities(&BcpDynamics, problem, options())
            .unwrap();
        let reference = Dop853
            .propagate_with_sensitivities(
                &BcpDynamics,
                problem,
                IntegratorOptions {
                    maximum_step: Some(0.002),
                    ..options()
                },
            )
            .unwrap();
        assert_eq!(candidate.time, problem.nominal.final_time);
        assert_state(&candidate.state, &reference.state, 2e-12);
        for (actual, expected) in candidate.sensitivities.iter().zip(&reference.sensitivities) {
            assert_state(actual, expected, 5e-9);
        }
    }
    check::<1>();
    check::<7>();
    check::<14>();
    check::<21>();
}

#[test]
fn complete_zoh_taylor_jacobian_matches_a_mixed_leg_perturbation() {
    use pykep_core::leg::ZohKeplerLeg;
    const SEGMENTS: usize = 20;
    let initial = [1., 0., 0., 0., 1., 0., 1.];
    let final_state = [0.8, 0.6, 0., -0.6, 0.8, 0., 0.99];
    let initial_seed = [0.1, -0.3, 0.03, 0.04, -0.07, 0.1, 0.05];
    let final_seed = [0.2, 0.02, -0.1, 0.08, -0.09, 0.03, -0.05];
    let control_seed = [0.02, -0.1, 0.07, 0.03];
    let grid_seed = |i: usize| 0.04 * ((i % 7) as f64 - 3.0);
    let build = |h: f64| {
        ZohKeplerLeg::new(
            ZohKeplerDynamics,
            std::array::from_fn(|i| initial[i] + h * initial_seed[i]),
            (0..SEGMENTS)
                .map(|i| {
                    let control = [0.05, if i % 2 == 0 { 0.3 } else { -0.3 }, 0.4, 0.0];
                    std::array::from_fn(|j| control[j] + h * control_seed[j])
                })
                .collect(),
            std::array::from_fn(|i| final_state[i] + h * final_seed[i]),
            (0..=SEGMENTS)
                .map(|i| i as f64 / SEGMENTS as f64 + h * grid_seed(i))
                .collect(),
            [0.02],
            0.5,
            options(),
        )
        .unwrap()
    };
    let jacobian = build(0.0)
        .mismatch_jacobian_with_method(IntegrationMethod::Taylor)
        .unwrap();
    let predicted: [f64; 7] = std::array::from_fn(|i| {
        jacobian.initial_state[i]
            .iter()
            .zip(initial_seed)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            + jacobian.final_state[i]
                .iter()
                .zip(final_seed)
                .map(|(a, b)| a * b)
                .sum::<f64>()
            + jacobian.controls[i]
                .iter()
                .enumerate()
                .map(|(j, a)| a * control_seed[j % 4])
                .sum::<f64>()
            + jacobian.time_grid[i]
                .iter()
                .enumerate()
                .map(|(j, a)| a * grid_seed(j))
                .sum::<f64>()
    });
    for h in [1e-4, 2e-5] {
        let plus = build(h)
            .mismatch_constraints_with_method(IntegrationMethod::Taylor)
            .unwrap();
        let minus = build(-h)
            .mismatch_constraints_with_method(IntegrationMethod::Taylor)
            .unwrap();
        assert_state(
            &predicted,
            &std::array::from_fn(|i| (plus[i] - minus[i]) / (2.0 * h)),
            2e-8,
        );
    }
}
