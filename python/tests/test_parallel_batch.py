"""Parity tests for the ordered parallel batch extension."""

from collections.abc import Callable

import numpy as np
import pytest

import pykep_rust as pk


def assert_array_close(actual: object, expected: object) -> None:
    """Compare arbitrarily nested numeric outputs."""
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-11, atol=1e-12)


def test_foundation_anomaly_and_element_batches_match_scalars() -> None:
    """Cheap batch surfaces preserve scalar values, shapes, and order."""
    vectors = np.asarray([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    other = np.asarray([[4.0, -1.0, 0.5], [1.0, 3.0, -2.0]])
    assert pk.dot_batch(vectors, other, workers=2) == pytest.approx(
        [pk.dot(left, right) for left, right in zip(vectors, other)]
    )
    assert pk.norm_batch(vectors, workers=2) == pytest.approx(
        [pk.norm(vector) for vector in vectors]
    )
    assert_array_close(
        pk.normalize_batch(vectors, workers=2),
        [pk.normalize(vector) for vector in vectors],
    )
    assert_array_close(
        pk.cross_batch(vectors, other, workers=2),
        [pk.cross(left, right) for left, right in zip(vectors, other)],
    )
    assert_array_close(
        pk.skew_batch(vectors, workers=2),
        [pk.skew(vector) for vector in vectors],
    )
    assert pk.stumpff_c_batch([0.0, 0.2], workers=2) == pytest.approx(
        [pk.stumpff_c(0.0), pk.stumpff_c(0.2)]
    )
    assert pk.stumpff_s_batch([0.0, 0.2], workers=2) == pytest.approx(
        [pk.stumpff_s(0.0), pk.stumpff_s(0.2)]
    )

    elliptic_values = [0.1, 0.2]
    hyperbolic_values = [0.1, 0.2]
    cases: list[
        tuple[
            Callable[..., list[float]],
            Callable[[float, float], float],
            list[float],
            float,
        ]
    ] = [
        (pk.mean_to_eccentric_anomaly_batch, pk.mean_to_eccentric_anomaly, elliptic_values, 0.4),
        (pk.eccentric_to_mean_anomaly_batch, pk.eccentric_to_mean_anomaly, elliptic_values, 0.4),
        (pk.eccentric_to_true_anomaly_batch, pk.eccentric_to_true_anomaly, elliptic_values, 0.4),
        (pk.true_to_eccentric_anomaly_batch, pk.true_to_eccentric_anomaly, elliptic_values, 0.4),
        (pk.mean_to_true_anomaly_batch, pk.mean_to_true_anomaly, elliptic_values, 0.4),
        (pk.true_to_mean_anomaly_batch, pk.true_to_mean_anomaly, elliptic_values, 0.4),
        (
            pk.gudermannian_to_true_anomaly_batch,
            pk.gudermannian_to_true_anomaly,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.true_to_gudermannian_anomaly_batch,
            pk.true_to_gudermannian_anomaly,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.hyperbolic_mean_to_anomaly_batch,
            pk.hyperbolic_mean_to_anomaly,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.hyperbolic_anomaly_to_mean_batch,
            pk.hyperbolic_anomaly_to_mean,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.hyperbolic_anomaly_to_true_batch,
            pk.hyperbolic_anomaly_to_true,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.true_to_hyperbolic_anomaly_batch,
            pk.true_to_hyperbolic_anomaly,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.hyperbolic_mean_to_true_batch,
            pk.hyperbolic_mean_to_true,
            hyperbolic_values,
            1.5,
        ),
        (
            pk.true_to_hyperbolic_mean_batch,
            pk.true_to_hyperbolic_mean,
            hyperbolic_values,
            1.5,
        ),
    ]
    for batch, scalar, values, eccentricity in cases:
        assert batch(values, eccentricity, workers=2) == pytest.approx(
            [scalar(value, eccentricity) for value in values]
        )

    classical = np.asarray(
        [
            [2.0, 0.2, 0.4, 0.3, 0.2, 0.1],
            [2.2, 0.1, 0.5, 0.2, 0.4, 0.2],
        ]
    )
    states = pk.classical_to_cartesian_batch(classical, 1.0, workers=2)
    recovered = pk.cartesian_to_classical_batch(states, 1.0, workers=2)
    assert_array_close(recovered, classical)
    mee = pk.classical_to_modified_equinoctial_batch(classical, workers=2)
    assert_array_close(
        pk.modified_equinoctial_to_classical_batch(mee, workers=2), classical
    )
    direct_mee = pk.cartesian_to_modified_equinoctial_batch(states, 1.0, workers=2)
    assert_array_close(
        pk.modified_equinoctial_to_cartesian_batch(
            direct_mee, 1.0, workers=2
        ),
        states,
    )
    assert_array_close(
        pk.cartesian_to_modified_equinoctial_jacobian_batch(
            states, 1.0, workers=2
        ),
        [pk.cartesian_to_modified_equinoctial_jacobian(state, 1.0) for state in states],
    )
    assert_array_close(
        pk.modified_equinoctial_to_cartesian_jacobian_batch(
            direct_mee, 1.0, workers=2
        ),
        [
            pk.modified_equinoctial_to_cartesian_jacobian(elements, 1.0)
            for elements in direct_mee
        ],
    )


def test_propagation_lambert_and_ephemeris_batches_match_scalars() -> None:
    """All two-body, Lambert, and provider batches expose the same results."""
    classical = np.asarray(
        [
            [2.0, 0.2, 0.4, 0.3, 0.2, 0.1],
            [2.2, 0.1, 0.5, 0.2, 0.4, -0.2],
        ]
    )
    states = pk.classical_to_cartesian_batch(classical, 1.0, workers=1)
    times = np.asarray([0.1, -0.2])
    for batch, scalar in [
        (pk.propagate_lagrangian_batch, pk.propagate_lagrangian),
        (pk.propagate_universal_batch, pk.propagate_universal),
        (pk.propagate_keplerian_batch, pk.propagate_keplerian),
    ]:
        output = batch(states, times, 1.0, workers=2)
        assert_array_close(
            output,
            [scalar(state, time, 1.0) for state, time in zip(states, times)],
        )
    propagated, matrices = pk.propagate_lagrangian_with_stm_batch(
        states, times, 1.0, workers=2
    )
    assert (propagated.shape, matrices.shape) == ((2, 6), (2, 6, 6))
    for index in range(2):
        expected_state, expected_matrix = pk.propagate_lagrangian_with_stm(
            states[index], times[index], 1.0
        )
        assert propagated[index] == pytest.approx(expected_state)
        assert_array_close(matrices[index], expected_matrix)
    grid = pk.propagate_lagrangian_grid(
        states[0], np.asarray([10.0, 10.1, 10.2]), 1.0, workers=2
    )
    assert grid[0] == pytest.approx(states[0])

    initial = np.asarray([[1.0, 0.0, 0.0], [1.1, 0.1, 0.0]])
    final = np.asarray([[0.2, 1.1, 0.3], [0.1, 1.2, 0.2]])
    flight_times = np.asarray([20.0, 22.0])
    problems = pk.lambert_problem_batch(
        initial, final, flight_times, 1.0, maximum_revolutions=2, workers=2
    )
    assert len(problems) == 2
    for index, problem in enumerate(problems):
        scalar = pk.LambertProblem(
            initial[index], final[index], flight_times[index], 1.0, False, 2
        )
        assert problem.initial_position == pytest.approx(scalar.initial_position)
        assert [solution.path for solution in problem.solutions] == [
            solution.path for solution in scalar.solutions
        ]

    planet = pk.Planet.keplerian_from_classical(
        0.0, classical[0], 1.0, "batch"
    )
    epochs = np.asarray([0.0, 0.1])
    assert_array_close(
        planet.states(epochs, workers=2),
        [planet.state(epoch) for epoch in epochs],
    )
    assert_array_close(
        planet.elements_batch(epochs, workers=2),
        [planet.elements(epoch) for epoch in epochs],
    )
    assert planet.period_batch(epochs, workers=2) == pytest.approx(
        [planet.period(epoch) for epoch in epochs]
    )
    with pytest.raises(pk.UnsupportedCapabilityError):
        planet.acceleration_batch(epochs, workers=2)


def test_mission_utility_batches_match_scalars() -> None:
    """Transfer, encoding, flyby, and MIMA batches preserve scalar contracts."""
    r1 = np.asarray([1.0, 1.2])
    r2 = np.asarray([2.0, 2.4])
    delta_v, durations, impulses = pk.hohmann_batch(r1, r2, 1.0, workers=2)
    for index in range(2):
        expected = pk.hohmann(r1[index], r2[index], 1.0)
        assert delta_v[index] == pytest.approx(expected[0])
        assert durations[index] == pytest.approx(expected[1])
        assert impulses[index] == pytest.approx(expected[2])
    bi = pk.bielliptic_batch(r1, r2, r2, 1.0, workers=2)
    for index in range(2):
        expected = pk.bielliptic(r1[index], r2[index], r2[index], 1.0)
        assert bi[0][index] == pytest.approx(expected[0])
        assert bi[1][index] == pytest.approx(expected[1])
        assert bi[2][index] == pytest.approx(expected[2])

    direct = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]]
    alpha = pk.direct_to_alpha_batch(direct, workers=2)
    assert_array_close(
        pk.alpha_to_direct_batch(
            [value for value, _ in alpha], alpha[0][1], workers=2
        ),
        direct,
    )
    eta = pk.direct_to_eta_batch(direct, 1.0, workers=2)
    assert_array_close(pk.eta_to_direct_batch(eta, 1.0, workers=2), direct)

    incoming = np.asarray(
        [[7200.0, -4567.7655, 1234.4233], [7000.0, -4200.0, 1000.0]]
    )
    outgoing = np.asarray(
        [[7100.0, 220.123, -144.432], [6800.0, 500.0, -200.0]]
    )
    mu = 3.986e14
    radius = 7e6
    assert_array_close(
        pk.flyby_constraints_batch(
            incoming, outgoing, mu, radius, workers=2
        ),
        [
            pk.flyby_constraints(a, b, mu, radius)
            for a, b in zip(incoming, outgoing)
        ],
    )
    assert_array_close(
        pk.flyby_constraints_jacobian_batch(
            incoming, outgoing, mu, radius, workers=2
        ),
        [
            pk.flyby_constraints_jacobian(a, b, mu, radius)
            for a, b in zip(incoming, outgoing)
        ],
    )
    assert pk.flyby_delta_v_batch(
        incoming, outgoing, mu, radius, workers=2
    ) == pytest.approx(
        [pk.flyby_delta_v(a, b, mu, radius) for a, b in zip(incoming, outgoing)]
    )
    planet_velocity = np.asarray([[10000.0, 20000.0, -1000.0]] * 2)
    periapsis = np.asarray([radius, radius * 1.1])
    beta = np.asarray([0.2, -0.1])
    assert_array_close(
        pk.flyby_outgoing_velocity_batch(
            incoming, planet_velocity, periapsis, beta, mu, workers=2
        ),
        [
            pk.flyby_outgoing_velocity(a, v, rp, angle, mu)
            for a, v, rp, angle in zip(
                incoming, planet_velocity, periapsis, beta
            )
        ],
    )

    departure = np.asarray([[1.0, 0.0, 0.0], [0.8, 0.1, 0.0]])
    arrival = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.8, 0.1]])
    times = np.asarray([10.0, 12.0])
    masses, accelerations = pk.mima_batch(
        departure, arrival, times, 0.6, 4000.0, workers=2
    )
    for index in range(2):
        assert (masses[index], accelerations[index]) == pytest.approx(
            pk.mima(departure[index], arrival[index], times[index], 0.6, 4000.0)
        )
    initial_states = np.asarray(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * 2
    )
    masses2, accelerations2 = pk.mima2_batch(
        initial_states,
        departure * 0.01,
        arrival * 0.01,
        np.asarray([1.0, 1.2]),
        0.1,
        1.0,
        1.0,
        workers=2,
    )
    for index, time in enumerate([1.0, 1.2]):
        assert (masses2[index], accelerations2[index]) == pytest.approx(
            pk.mima2(
                initial_states[index],
                departure[index] * 0.01,
                arrival[index] * 0.01,
                time,
                0.1,
                1.0,
                1.0,
            )
        )


def test_evaluated_dynamics_and_control_batches_match_scalars() -> None:
    """RHS, adaptive dynamics, ZOH, and Pontryagin batches match scalars."""
    state = np.asarray(
        [1.01238082345234, -0.0423523523454, 0.22634376321,
         -0.1232623614, 0.123462698209365, 0.123667064622]
    )
    states = np.tile(state, (2, 1))
    mu = 0.01215058560962404
    assert_array_close(
        pk.kepler_rhs_batch(states, 1.0, workers=2),
        [pk.kepler_rhs(state, 1.0)] * 2,
    )
    assert_array_close(
        pk.cr3bp_rhs_batch(states, mu, workers=2),
        [pk.cr3bp_rhs(state, mu)] * 2,
    )
    bcp_times = np.asarray([0.0, 0.1])
    assert_array_close(
        pk.bcp_rhs_batch(
            bcp_times,
            states,
            mu,
            0.0,
            pk.BCP_SUN_DISTANCE,
            pk.BCP_SUN_ANGULAR_VELOCITY,
            workers=2,
        ),
        [
            pk.bcp_rhs(
                time,
                state,
                mu,
                0.0,
                pk.BCP_SUN_DISTANCE,
                pk.BCP_SUN_ANGULAR_VELOCITY,
            )
            for time in bcp_times
        ],
    )
    assert pk.cr3bp_effective_potential_batch(
        states, mu, workers=2
    ) == pytest.approx([pk.cr3bp_effective_potential(state, mu)] * 2)
    assert pk.cr3bp_jacobi_constant_batch(states, mu, workers=2) == pytest.approx(
        [pk.cr3bp_jacobi_constant(state, mu)] * 2
    )

    final_times = np.asarray([0.005, 0.01])
    assert_array_close(
        pk.propagate_kepler_dynamics_batch(
            states, final_times, 1.0, workers=2
        ),
        [
            pk.propagate_kepler_dynamics(state, time, 1.0)
            for time in final_times
        ],
    )
    assert_array_close(
        pk.propagate_cr3bp_batch(states, final_times, mu, workers=2),
        [pk.propagate_cr3bp(state, time, mu) for time in final_times],
    )
    bcp_parameters = (
        mu,
        pk.BCP_MU_SUN,
        pk.BCP_SUN_DISTANCE,
        pk.BCP_SUN_ANGULAR_VELOCITY,
    )
    assert_array_close(
        pk.propagate_bcp_batch(
            states, final_times, *bcp_parameters, workers=2
        ),
        [
            pk.propagate_bcp(state, time, *bcp_parameters)
            for time in final_times
        ],
    )
    for batch, scalar, arguments in [
        (
            pk.propagate_kepler_dynamics_with_stm_batch,
            pk.propagate_kepler_dynamics_with_stm,
            (1.0,),
        ),
        (
            pk.propagate_cr3bp_with_stm_batch,
            pk.propagate_cr3bp_with_stm,
            (mu,),
        ),
        (
            pk.propagate_bcp_with_stm_batch,
            pk.propagate_bcp_with_stm,
            bcp_parameters,
        ),
    ]:
        propagated, matrices = batch(states, final_times, *arguments, workers=2)
        assert (propagated.shape, matrices.shape) == ((2, 6), (2, 6, 6))
        for index, final_time in enumerate(final_times):
            expected_state, expected_matrix = scalar(
                states[index], final_time, *arguments
            )
            assert_array_close(propagated[index], expected_state)
            assert_array_close(matrices[index], expected_matrix)

    zoh_state = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5])
    zoh_states = np.tile(zoh_state, (2, 1))
    thrust = np.asarray([0.0, 0.01])
    directions = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert_array_close(
        pk.zoh_kepler_rhs_batch(
            zoh_states, thrust, directions, 0.02, workers=2
        ),
        [
            pk.zoh_kepler_rhs(state, value, direction, 0.02)
            for state, value, direction in zip(zoh_states, thrust, directions)
        ],
    )
    rotating = np.tile(np.asarray([0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 1.5]), (2, 1))
    assert_array_close(
        pk.zoh_cr3bp_rhs_batch(
            rotating, thrust, directions, 0.02, 0.01, workers=2
        ),
        [
            pk.zoh_cr3bp_rhs(state, value, direction, 0.02, 0.01)
            for state, value, direction in zip(rotating, thrust, directions)
        ],
    )
    equinoctial = np.tile(np.asarray([1.2, 0.1, 0.0, 0.0, 0.0, 0.2, 1.0]), (2, 1))
    assert_array_close(
        pk.zoh_equinoctial_rhs_batch(
            equinoctial, thrust, directions, 0.0, workers=2
        ),
        [
            pk.zoh_equinoctial_rhs(state, value, direction, 0.0)
            for state, value, direction in zip(equinoctial, thrust, directions)
        ],
    )
    sail = np.tile(np.asarray([0.8, -0.4, 0.3, 0.2, 0.9, -0.1]), (2, 1))
    alphas = np.asarray([0.25, 0.3])
    betas = np.asarray([-1.1, 0.2])
    assert_array_close(
        pk.zoh_solar_sail_rhs_batch(
            sail, alphas, betas, 0.04, workers=2
        ),
        [
            pk.zoh_solar_sail_rhs(state, alpha, beta, 0.04)
            for state, alpha, beta in zip(sail, alphas, betas)
        ],
    )

    boundaries = [[0.0, 0.01], [0.0, 0.02]]
    standard_controls = [[[0.0, 1.0, 0.0, 0.0]]] * 2
    kepler_batch = pk.propagate_zoh_kepler_batch(
        zoh_states,
        boundaries,
        standard_controls,
        0.02,
        workers=2,
    )
    assert_array_close(
        kepler_batch,
        [
            pk.propagate_zoh_kepler(state, grid, controls, 0.02)
            for state, grid, controls in zip(
                zoh_states, boundaries, standard_controls
            )
        ],
    )
    cr3bp_batch = pk.propagate_zoh_cr3bp_batch(
        rotating,
        boundaries,
        standard_controls,
        0.02,
        0.01,
        workers=2,
    )
    assert_array_close(
        cr3bp_batch,
        [
            pk.propagate_zoh_cr3bp(state, grid, controls, 0.02, 0.01)
            for state, grid, controls in zip(
                rotating, boundaries, standard_controls
            )
        ],
    )
    equinoctial_controls = [[[0.0, 0.0, 0.0, 0.0]]] * 2
    equinoctial_batch = pk.propagate_zoh_equinoctial_batch(
        equinoctial,
        boundaries,
        equinoctial_controls,
        0.0,
        workers=2,
    )
    assert_array_close(
        equinoctial_batch,
        [
            pk.propagate_zoh_equinoctial(state, grid, controls, 0.0)
            for state, grid, controls in zip(
                equinoctial, boundaries, equinoctial_controls
            )
        ],
    )
    sail_controls = [[[0.25, -1.1]]] * 2
    sail_batch = pk.propagate_zoh_solar_sail_batch(
        sail, boundaries, sail_controls, 0.04, workers=2
    )
    assert_array_close(
        sail_batch,
        [
            pk.propagate_zoh_solar_sail(state, grid, controls, 0.04)
            for state, grid, controls in zip(sail, boundaries, sail_controls)
        ],
    )

    cartesian = np.asarray(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    cartesian_states = np.tile(cartesian, (2, 1))
    mass_parameters = [1.0, 0.01, 1.0, 0.5, 1.0]
    assert_array_close(
        pk.pontryagin_cartesian_rhs_batch(
            cartesian_states, pk.Optimality.Mass, mass_parameters, workers=2
        ),
        [
            pk.pontryagin_cartesian_rhs(
                cartesian, pk.Optimality.Mass, mass_parameters
            )
        ]
        * 2,
    )
    cartesian_controls = pk.pontryagin_cartesian_control_batch(
        cartesian_states, pk.Optimality.Mass, mass_parameters, workers=2
    )
    expected_cartesian_control = pk.pontryagin_cartesian_control(
        cartesian, pk.Optimality.Mass, mass_parameters
    )
    for control in cartesian_controls:
        assert control[0] == pytest.approx(expected_cartesian_control[0])
        assert control[1] == pytest.approx(expected_cartesian_control[1])
        assert control[2] == pytest.approx(expected_cartesian_control[2])
    assert pk.pontryagin_cartesian_hamiltonian_batch(
        cartesian_states, pk.Optimality.Mass, mass_parameters, workers=2
    ) == pytest.approx(
        [
            pk.pontryagin_cartesian_hamiltonian(
                state, pk.Optimality.Mass, mass_parameters
            )
            for state in cartesian_states
        ]
    )
    cartesian_final_times = np.asarray([0.001, 0.002])
    cartesian_propagated = pk.propagate_pontryagin_cartesian_batch(
        cartesian_states,
        cartesian_final_times,
        pk.Optimality.Mass,
        mass_parameters,
        workers=2,
    )
    assert_array_close(
        cartesian_propagated,
        [
            pk.propagate_pontryagin_cartesian(
                state, time, pk.Optimality.Mass, mass_parameters
            )
            for state, time in zip(cartesian_states, cartesian_final_times)
        ],
    )

    equinoctial_costate = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    equinoctial_states = np.tile(equinoctial_costate, (2, 1))
    time_parameters = [1.0, 1e-4, 1.0]
    assert_array_close(
        pk.pontryagin_equinoctial_rhs_batch(
            equinoctial_states, pk.Optimality.Time, time_parameters, workers=2
        ),
        [
            pk.pontryagin_equinoctial_rhs(
                state, pk.Optimality.Time, time_parameters
            )
            for state in equinoctial_states
        ],
    )
    equinoctial_controls = pk.pontryagin_equinoctial_control_batch(
        equinoctial_states, pk.Optimality.Time, time_parameters, workers=2
    )
    for state, control in zip(equinoctial_states, equinoctial_controls):
        expected = pk.pontryagin_equinoctial_control(
            state, pk.Optimality.Time, time_parameters
        )
        assert control[0] == pytest.approx(expected[0])
        assert control[1] == pytest.approx(expected[1])
        assert control[2] == pytest.approx(expected[2])
    assert pk.pontryagin_equinoctial_hamiltonian_batch(
        equinoctial_states, pk.Optimality.Time, time_parameters, workers=2
    ) == pytest.approx(
        [
            pk.pontryagin_equinoctial_hamiltonian(
                state, pk.Optimality.Time, time_parameters
            )
            for state in equinoctial_states
        ]
    )
    equinoctial_final_times = np.asarray([0.001, 0.002])
    equinoctial_propagated = pk.propagate_pontryagin_equinoctial_batch(
        equinoctial_states,
        equinoctial_final_times,
        pk.Optimality.Time,
        time_parameters,
        workers=2,
    )
    assert_array_close(
        equinoctial_propagated,
        [
            pk.propagate_pontryagin_equinoctial(
                state, time, pk.Optimality.Time, time_parameters
            )
            for state, time in zip(equinoctial_states, equinoctial_final_times)
        ],
    )


def test_leg_object_batches_match_scalar_methods() -> None:
    """Immutable leg objects can be evaluated through every batch method."""
    departure = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    arrival = [0.2, 1.1, 0.1, -0.9, 0.15, -0.05]
    throttles = [
        [0.1, -0.2, 0.05],
        [0.3, 0.1, -0.15],
        [-0.25, 0.2, 0.1],
        [0.05, -0.1, 0.2],
    ]
    leg = pk.SimsFlanaganLeg(
        departure, 2.0, throttles, arrival, 1.7, 1.3, 0.04, 3.0, 1.0
    )
    assert_array_close(
        pk.SimsFlanaganLeg.mismatch_constraints_batch(
            [leg, leg], workers=2
        ),
        [leg.mismatch_constraints()] * 2,
    )
    assert_array_close(
        pk.SimsFlanaganLeg.throttle_constraints_batch(
            [leg, leg], workers=2
        ),
        [leg.throttle_constraints()] * 2,
    )
    actual_leg_jacobian = pk.SimsFlanaganLeg.mismatch_jacobian_batch(
        [leg], workers=2
    )[0]
    for actual, expected in zip(actual_leg_jacobian, leg.mismatch_jacobian()):
        assert_array_close(actual, expected)
    assert_array_close(
        pk.SimsFlanaganLeg.throttle_jacobian_batch([leg], workers=2)[0],
        leg.throttle_jacobian(),
    )

    alpha = pk.SimsFlanaganAlphaLeg(
        departure,
        2.0,
        throttles,
        [0.1, 0.2, 0.4, 0.6],
        arrival,
        1.7,
        1.3,
        0.04,
        3.0,
        1.0,
    )
    assert_array_close(
        pk.SimsFlanaganAlphaLeg.mismatch_constraints_batch(
            [alpha, alpha], workers=2
        ),
        [alpha.mismatch_constraints()] * 2,
    )
    assert_array_close(
        pk.SimsFlanaganAlphaLeg.throttle_constraints_batch([alpha], workers=2)[0],
        alpha.throttle_constraints(),
    )
    assert_array_close(
        pk.SimsFlanaganAlphaLeg.throttle_jacobian_batch([alpha], workers=2)[0],
        alpha.throttle_jacobian(),
    )

    zoh = pk.ZohLeg(
        pk.ZohModel.Kepler,
        [1.0, 0.1, -0.05, -0.1, 0.95, 0.03, 1.2],
        [[0.02, 1.0, 0.0, 0.0], [0.01, 0.0, 1.0, 0.0]],
        [0.4, 0.9, 0.08, -0.8, 0.3, -0.04, 1.1],
        [0.1, 0.5, 1.0],
        [0.2],
        maximum_step=0.01,
    )
    assert_array_close(
        pk.ZohLeg.mismatch_constraints_batch([zoh, zoh], workers=2),
        [zoh.mismatch_constraints()] * 2,
    )
    actual_zoh_jacobian = pk.ZohLeg.mismatch_jacobian_batch([zoh], workers=2)[0]
    for actual, expected in zip(actual_zoh_jacobian, zoh.mismatch_jacobian()):
        assert_array_close(actual, expected)
    actual_forward, actual_backward = pk.ZohLeg.state_history_batch(
        [zoh], 2, workers=2
    )[0]
    expected_forward, expected_backward = zoh.state_history(2)
    assert_array_close(actual_forward, expected_forward)
    assert_array_close(actual_backward, expected_backward)


def test_batch_workers_shapes_and_error_order_are_explicit() -> None:
    """Worker modes agree and invalid input keeps deterministic error behavior."""
    vectors = np.asarray([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])
    expected = [5.0, 3.0]
    for workers in [0, 1, 2]:
        assert pk.norm_batch(vectors, workers=workers) == pytest.approx(expected)

    with pytest.raises(ValueError, match="workers"):
        pk.norm_batch(np.empty((0, 3)), workers=1025)
    with pytest.raises(ValueError):
        pk.dot_batch(vectors, vectors[:1], workers=2)
    with pytest.raises(pk.SingularGeometryError):
        pk.normalize_batch(
            np.asarray([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
            workers=2,
        )
