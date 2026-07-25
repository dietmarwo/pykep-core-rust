// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Pontryagin state/costate, control, invariant, and sensitivity validation.

use pykep_core::PykepError;
use pykep_core::astro::elements::{
    ModifiedEquinoctialElements, modified_equinoctial_to_cartesian,
    modified_equinoctial_to_cartesian_jacobian,
};
use pykep_core::dynamics::pontryagin::{
    CartesianMassOptimal, CartesianTimeOptimal, EquinoctialMassOptimal, EquinoctialTimeOptimal,
    cartesian_control_mass, cartesian_control_time, cartesian_hamiltonian_mass,
    cartesian_hamiltonian_time, equinoctial_control_mass, equinoctial_control_time,
    equinoctial_hamiltonian_mass, equinoctial_hamiltonian_time,
};
use pykep_core::integration::{
    DifferentiableDynamicsModel, Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions,
    SensitivityProblem,
};
use serde_json::Value;

const OPTIONS: IntegratorOptions = IntegratorOptions {
    relative_tolerance: 2e-12,
    absolute_tolerance: 2e-12,
    initial_step: None,
    maximum_step: Some(0.01),
    maximum_steps: 200_000,
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

fn array<const N: usize>(value: &Value) -> [f64; N] {
    let values = value.as_array().unwrap();
    assert_eq!(values.len(), N);
    core::array::from_fn(|index| parse(values[index].as_str().unwrap()))
}

fn encoded_values(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| parse(item.as_str().unwrap()))
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

fn propagate<M, const P: usize>(
    model: &M,
    initial: [f64; 14],
    parameters: [f64; P],
    final_time: f64,
    options: IntegratorOptions,
) -> [f64; 14]
where
    M: DynamicsModel<14, P>,
{
    Dop853
        .propagate(
            model,
            InitialValueProblem::new(0.0, initial, final_time, parameters),
            options,
        )
        .unwrap()
        .state
}

fn sensitivities<M, const P: usize>(
    model: &M,
    initial: [f64; 14],
    parameters: [f64; P],
    final_time: f64,
    lambda0_parameter: Option<usize>,
) -> ([f64; 14], [[f64; 8]; 14])
where
    M: DifferentiableDynamicsModel<14, P>,
{
    let mut initial_sensitivities = [[0.0; 8]; 14];
    for column in 0..7 {
        initial_sensitivities[column + 7][column] = 1.0;
    }
    let mut parameter_seeds = [[0.0; 8]; P];
    if let Some(index) = lambda0_parameter {
        parameter_seeds[index][7] = 1.0;
    }
    let propagation = Dop853
        .propagate_with_sensitivities(
            model,
            SensitivityProblem {
                nominal: InitialValueProblem::new(0.0, initial, final_time, parameters),
                initial_sensitivities,
                parameter_seeds,
            },
            OPTIONS,
        )
        .unwrap();
    (propagation.state, propagation.sensitivities)
}

#[test]
fn upstream_cartesian_and_equinoctial_trajectories_match() {
    let document: Value = serde_json::from_str(include_str!("data/phase13-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    let cases = document["models"].as_array().unwrap();

    for case in &cases[..4] {
        let initial = array::<14>(&case["initial_state"]);
        let final_time = parse(case["final_time"].as_str().unwrap());
        let expected = array::<14>(&case["final_state"]);
        let actual = match case["name"].as_str().unwrap() {
            "cartesian_mass" => propagate(
                &CartesianMassOptimal,
                initial,
                array::<5>(&case["parameters"]),
                final_time,
                OPTIONS,
            ),
            "cartesian_time" => propagate(
                &CartesianTimeOptimal,
                initial,
                array::<3>(&case["parameters"]),
                final_time,
                OPTIONS,
            ),
            "equinoctial_mass" => propagate(
                &EquinoctialMassOptimal,
                initial,
                array::<5>(&case["parameters"]),
                final_time,
                OPTIONS,
            ),
            "equinoctial_time" => propagate(
                &EquinoctialTimeOptimal,
                initial,
                array::<3>(&case["parameters"]),
                final_time,
                OPTIONS,
            ),
            name => panic!("unexpected case {name}"),
        };
        assert_scaled(&actual, &expected, 3e-10);
    }

    let physical = &cases[4];
    assert_eq!(physical["name"], "equinoctial_mass_physical");
    let physical_options = IntegratorOptions {
        maximum_step: Some(43_200.0),
        ..OPTIONS
    };
    let actual = propagate(
        &EquinoctialMassOptimal,
        array(&physical["initial_state"]),
        array::<5>(&physical["parameters"]),
        parse(physical["final_time"].as_str().unwrap()),
        physical_options,
    );
    assert_scaled(&actual, &array::<14>(&physical["final_state"]), 3e-9);
}

#[test]
fn variational_trajectories_match_cpp_and_finite_differences() {
    let document: Value = serde_json::from_str(include_str!("data/phase13-v1.json")).unwrap();
    let cases = document["models"].as_array().unwrap();
    for case in &cases[..4] {
        let initial = array::<14>(&case["initial_state"]);
        let final_time = parse(case["final_time"].as_str().unwrap());
        let (state, sensitivity) = match case["name"].as_str().unwrap() {
            "cartesian_mass" => sensitivities(
                &CartesianMassOptimal,
                initial,
                array::<5>(&case["parameters"]),
                final_time,
                Some(4),
            ),
            "cartesian_time" => sensitivities(
                &CartesianTimeOptimal,
                initial,
                array::<3>(&case["parameters"]),
                final_time,
                None,
            ),
            "equinoctial_mass" => sensitivities(
                &EquinoctialMassOptimal,
                initial,
                array::<5>(&case["parameters"]),
                final_time,
                Some(4),
            ),
            "equinoctial_time" => sensitivities(
                &EquinoctialTimeOptimal,
                initial,
                array::<3>(&case["parameters"]),
                final_time,
                None,
            ),
            name => panic!("unexpected case {name}"),
        };
        assert_scaled(&state, &array::<14>(&case["final_state"]), 3e-10);
        assert_scaled(
            sensitivity.as_flattened(),
            &encoded_values(&case["sensitivities"]),
            2e-4,
        );

        let step = 2e-6;
        let mut plus = initial;
        let mut minus = initial;
        plus[10] += step;
        minus[10] -= step;
        let (plus, minus) = match case["name"].as_str().unwrap() {
            "cartesian_mass" => (
                propagate(
                    &CartesianMassOptimal,
                    plus,
                    array::<5>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
                propagate(
                    &CartesianMassOptimal,
                    minus,
                    array::<5>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
            ),
            "cartesian_time" => (
                propagate(
                    &CartesianTimeOptimal,
                    plus,
                    array::<3>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
                propagate(
                    &CartesianTimeOptimal,
                    minus,
                    array::<3>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
            ),
            "equinoctial_mass" => (
                propagate(
                    &EquinoctialMassOptimal,
                    plus,
                    array::<5>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
                propagate(
                    &EquinoctialMassOptimal,
                    minus,
                    array::<5>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
            ),
            "equinoctial_time" => (
                propagate(
                    &EquinoctialTimeOptimal,
                    plus,
                    array::<3>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
                propagate(
                    &EquinoctialTimeOptimal,
                    minus,
                    array::<3>(&case["parameters"]),
                    final_time,
                    OPTIONS,
                ),
            ),
            _ => unreachable!(),
        };
        for row in 0..14 {
            let finite_difference = (plus[row] - minus[row]) / (2.0 * step);
            assert!(
                (sensitivity[row][3] - finite_difference).abs()
                    <= 4e-4 * finite_difference.abs().max(1.0),
                "{} sensitivity row {row}: {} != {finite_difference}",
                case["name"],
                sensitivity[row][3]
            );
        }
    }
}

#[test]
fn controls_and_singular_primer_behavior_are_explicit() {
    let cartesian = [1., 0., 0., 0., 1., 0., 10., 1., 1., 1., 1., 1., 1., 1.];
    let mass = cartesian_control_mass(&cartesian, &[1., 0.01, 1., 0.5, 1.]).unwrap();
    assert!(mass.throttle > 0.0 && mass.throttle < 1.0);
    assert!(
        (mass
            .direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 5e-16
    );
    assert_eq!(
        cartesian_control_time(&cartesian, &[1., 0.01, 1.])
            .unwrap()
            .throttle,
        1.0
    );

    let equinoctial = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ];
    let mass = equinoctial_control_mass(&equinoctial, &[1., 1e-4, 1., 1., 1e-4]).unwrap();
    assert!(mass.throttle > 0.0 && mass.throttle < 1.0);
    assert!(
        (mass
            .direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 3e-16
    );
    assert_eq!(
        equinoctial_control_time(&equinoctial, &[1., 1e-4, 1.])
            .unwrap()
            .throttle,
        1.0
    );

    let mut zero_cartesian = cartesian;
    zero_cartesian[10..13].fill(0.0);
    assert!(matches!(
        cartesian_control_mass(&zero_cartesian, &[1., 0.01, 1., 0.5, 1.]),
        Err(PykepError::SingularGeometry { .. })
    ));
    let mut zero_equinoctial = equinoctial;
    zero_equinoctial[7..13].fill(0.0);
    assert!(matches!(
        equinoctial_control_time(&zero_equinoctial, &[1., 1e-4, 1.]),
        Err(PykepError::SingularGeometry { .. })
    ));
}

#[test]
fn autonomous_hamiltonians_are_conserved() {
    let cartesian = [1., 0., 0., 0., 1., 0., 10., 1., 1., 1., 1., 1., 1., 1.];
    let mass_parameters = [1., 0.01, 1., 0.5, 1.];
    let final_state = propagate(
        &CartesianMassOptimal,
        cartesian,
        mass_parameters,
        1.2345,
        OPTIONS,
    );
    let initial_hamiltonian = cartesian_hamiltonian_mass(&cartesian, &mass_parameters).unwrap();
    let final_hamiltonian = cartesian_hamiltonian_mass(&final_state, &mass_parameters).unwrap();
    assert!((final_hamiltonian - initial_hamiltonian).abs() < 2e-10);

    let equinoctial = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ];
    let time_parameters = [1., 1e-4, 1.];
    let final_state = propagate(
        &EquinoctialTimeOptimal,
        equinoctial,
        time_parameters,
        1.0,
        OPTIONS,
    );
    let initial_hamiltonian = equinoctial_hamiltonian_time(&equinoctial, &time_parameters).unwrap();
    let final_hamiltonian = equinoctial_hamiltonian_time(&final_state, &time_parameters).unwrap();
    assert!(
        (final_hamiltonian - initial_hamiltonian).abs()
            <= 2e-9 * initial_hamiltonian.abs().max(1.0)
    );
}

#[test]
fn cartesian_and_equinoctial_canonical_forms_are_consistent_without_thrust() {
    let elements = ModifiedEquinoctialElements::from([1.3, 0.12, -0.08, 0.2, -0.1, 0.7]);
    let cartesian_state = modified_equinoctial_to_cartesian(elements, 1.0, false).unwrap();
    let jacobian = modified_equinoctial_to_cartesian_jacobian(elements, 1.0, false).unwrap();
    let cartesian_costate = [0.3, -0.4, 0.2, 0.5, 0.1, -0.25];
    let equinoctial_costate: [f64; 6] = core::array::from_fn(|column| {
        (0..6)
            .map(|row| jacobian[row][column] * cartesian_costate[row])
            .sum()
    });
    let cartesian = [
        cartesian_state[0],
        cartesian_state[1],
        cartesian_state[2],
        cartesian_state[3],
        cartesian_state[4],
        cartesian_state[5],
        2.0,
        cartesian_costate[0],
        cartesian_costate[1],
        cartesian_costate[2],
        cartesian_costate[3],
        cartesian_costate[4],
        cartesian_costate[5],
        0.2,
    ];
    let values = elements.to_array();
    let equinoctial = [
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        2.0,
        equinoctial_costate[0],
        equinoctial_costate[1],
        equinoctial_costate[2],
        equinoctial_costate[3],
        equinoctial_costate[4],
        equinoctial_costate[5],
        0.2,
    ];
    let cartesian_hamiltonian =
        cartesian_hamiltonian_mass(&cartesian, &[1.0, 0.0, 2.0, 0.1, 1.0]).unwrap();
    let equinoctial_hamiltonian =
        equinoctial_hamiltonian_mass(&equinoctial, &[1.0, 0.0, 2.0, 0.1, 1.0]).unwrap();
    assert!((cartesian_hamiltonian - equinoctial_hamiltonian).abs() < 2e-14);

    let cartesian_time = cartesian_hamiltonian_time(&cartesian, &[1.0, 0.0, 2.0]).unwrap();
    let equinoctial_time = equinoctial_hamiltonian_time(&equinoctial, &[1.0, 0.0, 2.0]).unwrap();
    assert!((cartesian_time - equinoctial_time).abs() < 2e-14);
}

#[test]
fn rhs_jacobians_are_finite_and_use_output_by_input_order() {
    let state = [1., 0., 0., 0., 1., 0., 10., 1., 1., 1., 1., 1., 1., 1.];
    let parameters = [1., 0.01, 1., 0.5, 1.];
    let mut state_jacobian = [[0.0; 14]; 14];
    let mut parameter_jacobian = [[0.0; 5]; 14];
    CartesianMassOptimal
        .jacobians(
            0.0,
            &state,
            &parameters,
            &mut state_jacobian,
            &mut parameter_jacobian,
        )
        .unwrap();
    assert!(
        state_jacobian
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!(
        parameter_jacobian
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!((state_jacobian[0][3] - 1.0).abs() < 1e-12);
    assert!((state_jacobian[3][0] - 2.0).abs() < 2e-9);
}
