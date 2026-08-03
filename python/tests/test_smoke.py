"""Installed-extension and foundation API tests."""

from __future__ import annotations

import ast
import gc
import inspect
import math
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import pykep_rust as pk


def test_status_probe_reports_package_version() -> None:
    """The public facade reports the installed native package version."""
    assert pk.port_status() == f"pykep-core {version('pykep-rust')}"


def test_constants_and_julian_conversions() -> None:
    """Pinned constants and reference-epoch conversions reach Python."""
    assert pk.PI == math.pi
    assert pk.ASTRONOMICAL_UNIT == 149_597_870_700.0
    assert pk.jd_to_mjd(2_451_544.5) == 51_544.0
    assert pk.jd_to_mjd2000(2_451_544.5) == 0.0
    assert pk.mjd2000_to_jd(0.0) == 2_451_544.5
    assert pk.mjd_to_jd(pk.mjd2000_to_mjd(0.0)) == 2_451_544.5
    assert pk.mjd_to_mjd2000(pk.mjd2000_to_mjd(123.0)) == 123.0
    constants = [
        pk.HALF_PI,
        pk.CAVENDISH_CONSTANT,
        pk.MU_SUN,
        pk.MU_EARTH,
        pk.MU_MOON,
        pk.EARTH_ORBITAL_VELOCITY,
        pk.EARTH_J2,
        pk.EARTH_RADIUS,
        pk.DEGREES_TO_RADIANS,
        pk.RADIANS_TO_DEGREES,
        pk.DAY_TO_SECONDS,
        pk.SECONDS_TO_DAY,
        pk.JULIAN_YEAR_DAYS,
        pk.DAYS_TO_JULIAN_YEAR,
        pk.STANDARD_GRAVITY,
        pk.CR3BP_MU_EARTH_MOON,
        pk.BCP_MU_EARTH_MOON,
        pk.BCP_MU_SUN,
    ]
    assert all(math.isfinite(value) and value > 0.0 for value in constants)
    assert pk.DEGREES_TO_RADIANS * pk.RADIANS_TO_DEGREES == pytest.approx(1.0)
    assert pk.DAY_TO_SECONDS * pk.SECONDS_TO_DAY == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must be finite"):
        pk.jd_to_mjd(math.nan)


def test_stumpff_scalar_batch_and_errors() -> None:
    """Stable scalar and ordered batch Stumpff paths agree."""
    values = [-1.0, -1e-12, 0.0, 1e-12, 1.0]
    assert pk.stumpff_c_batch(values) == [pk.stumpff_c(value) for value in values]
    assert pk.stumpff_s_batch(tuple(values)) == [
        pk.stumpff_s(value) for value in values
    ]
    assert abs(pk.stumpff_c(1e-12) - 0.5) < 1e-13
    with pytest.raises(ValueError):
        pk.stumpff_c(math.inf)
    with pytest.raises(OverflowError):
        pk.stumpff_s(-1e7)


def test_kepler_residuals_and_domains() -> None:
    """Elliptic and hyperbolic residual families retain their domains."""
    assert pk.elliptic_kepler_residual(0.0, 0.0, 0.0) == 0.0
    assert pk.elliptic_kepler_derivative(0.0, 0.0) == 1.0
    assert pk.hyperbolic_kepler_residual(0.0, 0.0, 1.5) == 0.0
    with pytest.raises(ValueError, match="0 <= e < 1"):
        pk.elliptic_kepler_residual(0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="e > 1"):
        pk.hyperbolic_kepler_residual(0.0, 0.0, 1.0)
    residual = pk.elliptic_difference_residual(0.4, 0.3, 0.2, 2.0, 4.0, 3.0)
    assert math.isfinite(residual)
    residual_family = [
        pk.elliptic_kepler_second_derivative(0.4, 0.2),
        pk.hyperbolic_kepler_derivative(0.4, 1.5),
        pk.hyperbolic_kepler_second_derivative(0.4, 1.5),
        pk.elliptic_difference_derivative(0.4, 0.2, 2.0, 4.0, 3.0),
        pk.elliptic_difference_second_derivative(0.4, 0.2, 2.0, 4.0, 3.0),
        pk.hyperbolic_difference_residual(0.4, 0.3, 0.2, 2.0, -4.0, 3.0),
        pk.hyperbolic_difference_derivative(0.4, 0.2, 2.0, -4.0, 3.0),
        pk.hyperbolic_difference_second_derivative(0.4, 0.2, 2.0, -4.0, 3.0),
    ]
    assert all(math.isfinite(value) for value in residual_family)
    universal = pk.universal_kepler_residual(0.4, 20.0, 7.0, 0.2, 0.01, 3.0)
    assert math.isfinite(universal)
    assert math.isfinite(pk.universal_kepler_derivative(0.4, 7.0, 0.2, 0.01, 3.0))
    assert math.isfinite(
        pk.universal_kepler_second_derivative(0.4, 7.0, 0.2, 0.01, 3.0)
    )


def test_vector_operations_and_validation() -> None:
    """Three-vector shape, singularity, and finite-value checks are stable."""
    assert pk.dot([1.0, 2.0, 3.0], (4.0, 5.0, 6.0)) == 32.0
    assert pk.norm([3.0, 4.0, 12.0]) == 13.0
    assert pk.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == [0.0, 0.0, 1.0]
    assert pk.skew([1.0, 0.0, 0.0]) == [
        [0.0, -0.0, 0.0],
        [0.0, 0.0, -1.0],
        [-0.0, 1.0, 0.0],
    ]
    with pytest.raises(ValueError, match="expected 3"):
        pk.norm([1.0, 2.0])
    with pytest.raises(pk.SingularGeometryError):
        pk.normalize([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="must be finite"):
        pk.dot([math.nan, 0.0, 0.0], [1.0, 0.0, 0.0])


def test_epoch_construction_formatting_comparison_and_arithmetic() -> None:
    """Epoch values retain microsecond precision and explicit day arithmetic."""
    origin = pk.Epoch()
    assert origin.to_iso() == "2000-01-01T00:00:00.000000"
    assert origin.mjd2000 == 0.0
    assert origin.mjd == 51_544.0
    assert origin.jd == 2_451_544.5
    calendar = pk.Epoch.from_calendar(1980, 10, 17, 11, 36, 21, 121, 841)
    assert str(calendar) == "1980-10-17T11:36:21.121841"
    assert repr(calendar) == "Epoch.from_iso('1980-10-17T11:36:21.121841')"
    cropped = pk.Epoch.from_iso("2064-10")
    assert cropped.to_iso() == "2064-10-01T00:00:00.000000"
    for year in (-44, 12_345):
        extended = pk.Epoch.from_calendar(year, 3, 15)
        assert pk.Epoch.from_iso(extended.to_iso()) == extended
    tomorrow = origin + 1.0
    assert tomorrow > origin
    assert tomorrow.seconds_since(origin) == 86_400.0
    assert tomorrow - 1.0 == origin
    assert origin.add_seconds(0.000_001_9).microseconds_since_mjd2000 == 1
    assert pk.Epoch(51_544.0, "mjd") == origin
    assert pk.Epoch(2_451_544.5, "jd") == origin
    with pytest.raises(ValueError, match="scale"):
        pk.Epoch(0.0, "utc")
    with pytest.raises(ValueError, match="day"):
        pk.Epoch.from_calendar(2023, 2, 29)
    with pytest.raises(ValueError, match="expected YYYY"):
        pk.Epoch.from_iso("2064-10-")


def test_anomaly_conversions_batches_round_trips_and_domains() -> None:
    """Elliptic and hyperbolic solvers cover scalar, batch, and error paths."""
    eccentric = pk.mean_to_eccentric_anomaly(0.1, 0.5)
    assert pk.eccentric_to_mean_anomaly(eccentric, 0.5) == pytest.approx(0.1)
    true_anomaly = pk.eccentric_to_true_anomaly(eccentric, 0.5)
    assert pk.true_to_eccentric_anomaly(true_anomaly, 0.5) == pytest.approx(
        eccentric
    )
    assert pk.true_to_mean_anomaly(
        pk.mean_to_true_anomaly(0.1, 0.5), 0.5
    ) == pytest.approx(0.1)
    means = [-4.0, 0.0, 0.1, 100.0]
    assert pk.mean_to_eccentric_anomaly_batch(means, 0.9) == [
        pk.mean_to_eccentric_anomaly(value, 0.9) for value in means
    ]
    hyperbolic = pk.hyperbolic_mean_to_anomaly(20.0, 10.0)
    assert pk.hyperbolic_anomaly_to_mean(hyperbolic, 10.0) == pytest.approx(20.0)
    hyperbolic_true = pk.hyperbolic_anomaly_to_true(hyperbolic, 10.0)
    assert pk.true_to_hyperbolic_mean(hyperbolic_true, 10.0) == pytest.approx(20.0)
    assert pk.hyperbolic_mean_to_true(20.0, 10.0) == pytest.approx(hyperbolic_true)
    gudermannian = pk.true_to_gudermannian_anomaly(hyperbolic_true, 10.0)
    assert pk.gudermannian_to_true_anomaly(
        gudermannian, 10.0
    ) == pytest.approx(hyperbolic_true)
    assert pk.hyperbolic_mean_to_anomaly_batch(means, 1.5) == [
        pk.hyperbolic_mean_to_anomaly(value, 1.5) for value in means
    ]
    with pytest.raises(ValueError, match="0 <= e < 1"):
        pk.mean_to_eccentric_anomaly(0.1, 1.0)
    with pytest.raises(ValueError, match="e > 1"):
        pk.hyperbolic_mean_to_anomaly(0.1, 1.0)
    with pytest.raises(ValueError, match="asymptote"):
        pk.true_to_hyperbolic_anomaly(math.pi, 1.5)


def test_element_scalar_conversions_and_singularities() -> None:
    """Classical and both equinoctial conventions share one scalar core."""
    classical = [3.0, 0.3, 0.7, 1.1, 0.4, -0.8]
    state = pk.classical_to_cartesian(classical, 1.0)
    reconstructed = pk.classical_to_cartesian(
        pk.cartesian_to_classical(state, 1.0), 1.0
    )
    assert reconstructed == pytest.approx(state, rel=2e-13, abs=2e-13)
    for retrograde in (False, True):
        mee = pk.cartesian_to_modified_equinoctial(state, 1.0, retrograde)
        assert pk.modified_equinoctial_to_cartesian(
            mee, 1.0, retrograde
        ) == pytest.approx(state, rel=2e-13, abs=2e-13)
        via_classical = pk.classical_to_modified_equinoctial(
            classical, retrograde
        )
        assert pk.modified_equinoctial_to_classical(
            via_classical, retrograde
        ) == pytest.approx(classical, rel=2e-13, abs=2e-13)
    circular = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    with pytest.raises(pk.SingularGeometryError):
        pk.cartesian_to_classical(circular, 1.0)
    circular_mee = pk.cartesian_to_modified_equinoctial(circular, 1.0)
    assert pk.modified_equinoctial_to_cartesian(
        circular_mee, 1.0
    ) == pytest.approx(circular)
    with pytest.raises(ValueError, match="greater than zero"):
        pk.classical_to_cartesian(classical, 0.0)


def test_element_jacobians_and_numpy_batches() -> None:
    """Jacobians use output-by-input rows and NumPy batches match scalars."""
    classical = np.array(
        [
            [3.0, 0.3, 0.7, 1.1, 0.4, -0.8],
            [5.0, 0.1, 1.0, 2.1, 1.4, 0.8],
        ],
        dtype=np.float64,
    )
    states = pk.classical_to_cartesian_batch(classical, 1.0)
    assert isinstance(states, np.ndarray)
    assert states.shape == (2, 6)
    for index in range(2):
        assert states[index] == pytest.approx(
            pk.classical_to_cartesian(classical[index], 1.0)
        )
    recovered = pk.cartesian_to_classical_batch(states, 1.0)
    assert recovered == pytest.approx(
        np.array([pk.cartesian_to_classical(row, 1.0) for row in states])
    )
    mee = pk.cartesian_to_modified_equinoctial_batch(states, 1.0)
    assert pk.modified_equinoctial_to_cartesian_batch(
        mee, 1.0
    ) == pytest.approx(states)
    assert pk.classical_to_modified_equinoctial_batch(
        classical
    ) == pytest.approx(
        np.array(
            [pk.classical_to_modified_equinoctial(row) for row in classical]
        )
    )
    assert pk.modified_equinoctial_to_classical_batch(
        mee
    ) == pytest.approx(
        np.array([pk.modified_equinoctial_to_classical(row) for row in mee])
    )
    forward = np.array(
        pk.cartesian_to_modified_equinoctial_jacobian(states[0], 1.0)
    )
    inverse = np.array(
        pk.modified_equinoctial_to_cartesian_jacobian(mee[0], 1.0)
    )
    assert forward.shape == (6, 6)
    assert forward @ inverse == pytest.approx(np.eye(6), abs=3e-13)
    empty = pk.classical_to_cartesian_batch(np.empty((0, 6)), 1.0)
    assert empty.shape == (0, 6)
    with pytest.raises(ValueError, match="expected 6"):
        pk.classical_to_cartesian_batch(np.zeros((2, 5)), 1.0)


def test_numpy_batch_layout_dtype_and_finite_validation() -> None:
    """NumPy wrappers accept strided float64 arrays and reject invalid buffers."""
    classical = np.array(
        [
            [3.0, 0.3, 0.7, 1.1, 0.4, -0.8],
            [5.0, 0.1, 1.0, 2.1, 1.4, 0.8],
        ],
        dtype=np.float64,
    )
    storage = np.empty((2, 12), dtype=np.float64)
    storage[:, ::2] = classical
    storage[:, 1::2] = -99.0
    strided = storage[:, ::2]
    assert not strided.flags.c_contiguous
    strided.flags.writeable = False
    assert pk.classical_to_cartesian_batch(strided, 1.0) == pytest.approx(
        pk.classical_to_cartesian_batch(classical, 1.0)
    )
    with pytest.raises(TypeError):
        pk.classical_to_cartesian_batch(classical.astype(np.int64), 1.0)
    with pytest.raises(TypeError):
        pk.classical_to_cartesian_batch(classical[0], 1.0)
    invalid = classical.copy()
    invalid[1, 2] = math.inf
    with pytest.raises(ValueError, match="finite"):
        pk.classical_to_cartesian_batch(invalid, 1.0)


def test_propagation_stms_and_gil_releasing_batches() -> None:
    """Propagation covers scalar, STM, grid, and NumPy batch interfaces."""
    initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    quarter = pk.propagate_lagrangian(initial, math.pi / 2.0, 1.0)
    assert quarter == pytest.approx([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], abs=2e-15)
    assert pk.propagate_universal(initial, math.pi / 2.0, 1.0) == pytest.approx(
        quarter, abs=2e-14
    )
    final_state, stm = pk.propagate_lagrangian_with_stm(initial, 0.4, 1.0)
    assert np.asarray(stm).shape == (6, 6)
    assert np.asarray(
        pk.state_transition_matrix_lagrangian(initial, 0.4, 1.0)
    ) == pytest.approx(np.asarray(stm))
    reynolds = pk.state_transition_matrix_reynolds(
        initial, final_state, 0.4, 1.0
    )
    assert np.asarray(reynolds) == pytest.approx(np.asarray(stm), rel=2e-13)
    with pytest.raises(OverflowError, match="state_transition_matrix_reynolds"):
        pk.state_transition_matrix_reynolds(
            [1e300] * 6, [1e300] * 6, 1e300, 1e300
        )

    states = np.tile(np.asarray(initial), (32, 1))
    times = np.linspace(-1.0, 1.0, 32)
    batch = pk.propagate_lagrangian_batch(states, times, 1.0)
    universal = pk.propagate_universal_batch(states, times, 1.0)
    assert batch.shape == (32, 6)
    assert universal == pytest.approx(batch, rel=3e-14, abs=3e-14)
    assert batch[7] == pytest.approx(
        pk.propagate_lagrangian(states[7], times[7], 1.0)
    )
    grid = pk.propagate_lagrangian_grid(
        initial, np.array([10.0, 10.25, 10.5]), 1.0
    )
    assert grid[0] == pytest.approx(initial)
    with pytest.raises(ValueError, match="expected 32"):
        pk.propagate_lagrangian_batch(states, times[:-1], 1.0)


def test_evaluated_kepler_cr3bp_and_bcp_dynamics() -> None:
    """Phase 11 models expose RHS, invariants, propagation, and STMs."""
    assert pk.kepler_rhs([1.0, 2.0, 2.0, 4.0, 5.0, 6.0], 9.0) == pytest.approx(
        [4.0, 5.0, 6.0, -1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]
    )
    initial = [
        1.01238082345234,
        -0.0423523523454,
        0.22634376321,
        -0.1232623614,
        0.123462698209365,
        0.123667064622,
    ]
    mu = 0.01215058560962404
    potential = pk.cr3bp_effective_potential(initial, mu)
    jacobi = pk.cr3bp_jacobi_constant(initial, mu)
    assert jacobi == pytest.approx(
        2.0 * potential - sum(value * value for value in initial[3:])
    )
    propagated = pk.propagate_cr3bp(
        initial,
        5.7856656782589234,
        mu,
        relative_tolerance=2e-13,
        absolute_tolerance=2e-13,
        maximum_step=0.01,
    )
    assert propagated == pytest.approx(
        [
            0.43038358727124,
            -1.64650668902846,
            0.10271923139472,
            -0.9315629872575,
            -0.42680151362818,
            0.22257221768767,
        ],
        rel=2e-9,
        abs=2e-9,
    )
    final_state, stm = pk.propagate_cr3bp_with_stm(
        initial,
        0.25,
        mu,
        maximum_step=0.01,
    )
    assert len(final_state) == 6
    assert np.asarray(stm).shape == (6, 6)
    assert len(pk.propagate_kepler_dynamics(initial, 0.01, 1.0)) == 6
    assert np.asarray(
        pk.propagate_kepler_dynamics_with_stm(initial, 0.01, 1.0)[1]
    ).shape == (6, 6)
    assert len(pk.propagate_keplerian(initial, 0.01, 1.0)) == 6
    bcp_parameters = (
        mu,
        pk.BCP_MU_SUN,
        pk.BCP_SUN_DISTANCE,
        pk.BCP_SUN_ANGULAR_VELOCITY,
    )
    assert len(pk.propagate_bcp(initial, 0.01, *bcp_parameters)) == 6
    assert np.asarray(
        pk.propagate_bcp_with_stm(initial, 0.01, *bcp_parameters)[1]
    ).shape == (6, 6)
    assert pk.bcp_rhs(
        0.4,
        initial,
        mu,
        0.0,
        pk.BCP_SUN_DISTANCE,
        pk.BCP_SUN_ANGULAR_VELOCITY,
    ) == pytest.approx(pk.cr3bp_rhs(initial, mu))
    with pytest.raises(pk.SingularGeometryError):
        pk.kepler_rhs([0.0] * 6, 1.0)


def test_zero_order_hold_models_switch_and_reverse() -> None:
    """Phase 12 Python APIs preserve schedule dimensions and switch order."""
    initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5]
    boundaries = [0.0, 0.2, 0.5]
    controls = [[0.01, 1.0, 0.0, 0.0], [0.02, 0.0, 1.0, 0.0]]
    final_state = pk.propagate_zoh_kepler(
        initial,
        boundaries,
        controls,
        0.02,
        relative_tolerance=2e-13,
        absolute_tolerance=2e-13,
        maximum_step=0.01,
    )
    assert len(final_state) == 7
    assert final_state[6] < initial[6]
    recovered = pk.propagate_zoh_kepler(
        final_state,
        boundaries,
        controls,
        0.02,
        backward=True,
        relative_tolerance=2e-13,
        absolute_tolerance=2e-13,
        maximum_step=0.01,
    )
    assert recovered == pytest.approx(initial, rel=2e-11, abs=2e-11)
    assert pk.zoh_kepler_rhs(
        initial, 0.0, [1.0, 0.0, 0.0], 0.02
    )[:6] == pytest.approx(pk.kepler_rhs(initial[:6], 1.0))
    rotating = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 1.5]
    assert len(
        pk.zoh_cr3bp_rhs(rotating, 0.0, [1.0, 0.0, 0.0], 0.02, 0.01)
    ) == 7
    assert len(
        pk.propagate_zoh_cr3bp(
            rotating, [0.0, 0.01], [[0.0, 1.0, 0.0, 0.0]], 0.02, 0.01
        )
    ) == 7
    equinoctial = [1.2, 0.1, 0.0, 0.0, 0.0, 0.2, 1.0]
    assert len(
        pk.zoh_equinoctial_rhs(
            equinoctial, 0.0, [0.0, 0.0, 0.0], 0.0
        )
    ) == 7
    assert len(
        pk.propagate_zoh_equinoctial(
            equinoctial, [0.0, 0.01], [[0.0, 0.0, 0.0, 0.0]], 0.0
        )
    ) == 7
    sail = [0.8, -0.4, 0.3, 0.2, 0.9, -0.1]
    assert len(pk.zoh_solar_sail_rhs(sail, 0.25, -1.1, 0.04)) == 6
    assert len(
        pk.propagate_zoh_solar_sail(
            sail, [0.0, 0.01], [[0.25, -1.1]], 0.04
        )
    ) == 6
    with pytest.raises(ValueError, match="expected 4"):
        pk.propagate_zoh_kepler(initial, boundaries, [[0.0] * 3] * 2, 0.02)
    with pytest.raises(ValueError, match="strictly increasing"):
        pk.propagate_zoh_kepler(initial, [0.0, 0.2, 0.1], controls, 0.02)


def test_pontryagin_enum_controls_and_propagation() -> None:
    """Phase 13 uses a typed optimality enum and one native numerical core."""
    cartesian = [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        10.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert pk.Optimality.Mass != pk.Optimality.Time
    derivative = pk.pontryagin_cartesian_rhs(
        cartesian, pk.Optimality.Mass, [1.0, 0.01, 1.0, 0.5, 1.0]
    )
    assert len(derivative) == 14
    throttle, direction, switching = pk.pontryagin_cartesian_control(
        cartesian, pk.Optimality.Mass, [1.0, 0.01, 1.0, 0.5, 1.0]
    )
    assert 0.0 < throttle < 1.0
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert math.isfinite(switching)
    initial_hamiltonian = pk.pontryagin_cartesian_hamiltonian(
        cartesian, pk.Optimality.Mass, [1.0, 0.01, 1.0, 0.5, 1.0]
    )
    final_state = pk.propagate_pontryagin_cartesian(
        cartesian,
        1.2345,
        pk.Optimality.Mass,
        [1.0, 0.01, 1.0, 0.5, 1.0],
        relative_tolerance=2e-12,
        absolute_tolerance=2e-12,
        maximum_step=0.01,
    )
    assert final_state == pytest.approx(
        [
            0.3296405122183833,
            0.9437361993515112,
            -0.000126441649019,
            -0.9446333908544454,
            0.3295140980732467,
            -0.0000302961826291,
            9.993495810584642,
            -0.2930516116019541,
            0.1612382337718589,
            1.2739955068582864,
            0.9472048639975497,
            0.0186280856006421,
            -0.6140454396473527,
            0.9999296119431648,
        ],
        rel=3e-10,
        abs=3e-10,
    )
    assert pk.pontryagin_cartesian_hamiltonian(
        final_state, pk.Optimality.Mass, [1.0, 0.01, 1.0, 0.5, 1.0]
    ) == pytest.approx(initial_hamiltonian, rel=2e-10, abs=2e-10)

    equinoctial = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    assert len(
        pk.pontryagin_equinoctial_rhs(
            equinoctial, pk.Optimality.Time, [1.0, 1e-4, 1.0]
        )
    ) == 14
    throttle, direction, switching = pk.pontryagin_equinoctial_control(
        equinoctial, pk.Optimality.Time, [1.0, 1e-4, 1.0]
    )
    assert throttle == 1.0
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert math.isfinite(switching)
    assert math.isfinite(
        pk.pontryagin_equinoctial_hamiltonian(
            equinoctial, pk.Optimality.Time, [1.0, 1e-4, 1.0]
        )
    )
    assert len(
        pk.propagate_pontryagin_equinoctial(
            equinoctial,
            0.1,
            pk.Optimality.Time,
            [1.0, 1e-4, 1.0],
            maximum_step=0.01,
        )
    ) == 14
    with pytest.raises(TypeError):
        pk.pontryagin_cartesian_rhs(
            cartesian, "mass", [1.0, 0.01, 1.0, 0.5, 1.0]
        )


def test_sims_flanagan_fixed_alpha_constraints_and_shapes() -> None:
    """Phase 14 exposes validated immutable leg classes and stable Jacobians."""
    departure = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    arrival = [0.2, 1.1, 0.1, -0.9, 0.15, -0.05]
    throttles = [
        [0.1, -0.2, 0.05],
        [0.3, 0.1, -0.15],
        [-0.25, 0.2, 0.1],
        [0.05, -0.1, 0.2],
    ]
    leg = pk.SimsFlanaganLeg(
        departure,
        2.0,
        throttles,
        arrival,
        1.7,
        1.3,
        0.04,
        3.0,
        1.0,
    )
    assert leg.segment_count == 4
    assert leg.forward_segment_count == 2
    assert leg.backward_segment_count == 2
    assert len(leg.mismatch_constraints()) == 7
    assert len(leg.throttle_constraints()) == 4
    departure_jacobian, arrival_jacobian, control_time_jacobian = (
        leg.mismatch_jacobian()
    )
    assert np.asarray(departure_jacobian).shape == (7, 7)
    assert np.asarray(arrival_jacobian).shape == (7, 7)
    assert np.asarray(control_time_jacobian).shape == (7, 13)
    assert np.asarray(leg.throttle_jacobian()).shape == (4, 12)

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
    assert alpha.mismatch_constraints() == pytest.approx(
        [
            0.023919853076296405,
            -0.3060611737971523,
            -0.11324122549191241,
            0.1782908006996728,
            0.16755567783157332,
            -0.026531387267934307,
            0.2951375889873944,
        ],
        rel=4e-12,
        abs=4e-12,
    )
    weighted = pk.SimsFlanaganAlphaLeg.from_time_weights(
        departure,
        2.0,
        throttles,
        [1.0, 2.0, 3.0, 4.0],
        arrival,
        1.7,
        1.3,
        0.04,
        3.0,
        1.0,
    )
    assert sum(weighted.segment_durations) == pytest.approx(1.3)
    with pytest.raises(ValueError):
        pk.SimsFlanaganLeg(
            departure, 2.0, [], arrival, 1.7, 1.3, 0.04, 3.0, 1.0
        )


def test_generic_zoh_leg_constraints_gradients_histories_and_batch() -> None:
    """Phase 15 exposes the native generic leg with stable matrix shapes."""
    leg = pk.ZohLeg(
        pk.ZohModel.Kepler,
        [1.0, 0.1, -0.05, -0.1, 0.95, 0.03, 1.2],
        [
            [0.02, 1.0, 0.0, 0.0],
            [0.01, 0.0, 1.0, 0.0],
            [0.015, 0.0, 0.0, 1.0],
        ],
        [0.4, 0.9, 0.08, -0.8, 0.3, -0.04, 1.1],
        [0.1, 0.35, 0.7, 1.0],
        [0.2],
        maximum_step=0.005,
    )
    assert leg.model == pk.ZohModel.Kepler
    assert (leg.state_dimension, leg.control_dimension) == (7, 4)
    assert (leg.segment_count, leg.forward_segment_count) == (3, 1)
    assert leg.backward_segment_count == 2
    mismatch = leg.mismatch_constraints()
    assert mismatch == pytest.approx(
        [
            0.14639517124860257,
            -0.19845853240981343,
            -0.13033321567377493,
            0.03878657191645779,
            0.0970746217374101,
            0.02578617262269304,
            0.09739999999999971,
        ],
        rel=3e-9,
        abs=3e-9,
    )
    initial, final, controls, time_grid = leg.mismatch_jacobian()
    assert np.asarray(initial).shape == (7, 7)
    assert np.asarray(final).shape == (7, 7)
    assert np.asarray(controls).shape == (7, 12)
    assert np.asarray(time_grid).shape == (7, 4)
    forward, backward = leg.state_history(4)
    assert (len(forward), len(backward)) == (1, 2)
    assert all(len(segment) == 4 for segment in [*forward, *backward])
    assert np.asarray(pk.ZohLeg.mismatch_constraints_batch([leg, leg])) == pytest.approx(
        np.asarray([mismatch, mismatch])
    )
    with pytest.raises(ValueError):
        pk.ZohLeg(
            pk.ZohModel.SolarSail,
            [1.0] * 7,
            [[0.0, 0.0]],
            [1.0] * 6,
            [0.0, 1.0],
            [0.1],
        )


def test_transfers_encodings_flyby_lambert_and_mima() -> None:
    """Phase 6 mission-design APIs preserve branch order and stable shapes."""
    delta_v, duration, impulses = pk.hohmann(1.0, 2.0, 1.0)
    assert delta_v == pytest.approx(sum(impulses))
    assert duration > 0.0
    bi_delta_v, _, bi_impulses = pk.bielliptic(1.0, 2.0, 2.0, 1.0)
    assert bi_delta_v == pytest.approx(delta_v)
    assert len(bi_impulses) == 3

    direct = [0.1, 0.2, 0.3]
    alpha, total = pk.direct_to_alpha(direct)
    assert pk.alpha_to_direct(alpha, total) == pytest.approx(direct)
    for invalid_alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="0 < alpha < 1"):
            pk.alpha_to_direct([invalid_alpha, 0.5], 10.0)
    eta = pk.direct_to_eta(direct, 1.0)
    assert pk.eta_to_direct(eta, 1.0) == pytest.approx(direct)

    incoming = [7200.0, -4567.7655, 1234.4233]
    outgoing = [7100.0, 220.123, -144.432]
    constraints = pk.flyby_constraints(incoming, outgoing, 3.986e14, 7e6)
    assert len(constraints) == 2
    assert np.asarray(
        pk.flyby_constraints_jacobian(incoming, outgoing, 3.986e14, 7e6)
    ).shape == (2, 6)
    assert pk.flyby_delta_v(incoming, outgoing, 3.986e14, 7e6) > 0.0
    assert len(
        pk.flyby_outgoing_velocity(
            incoming, [10_000.0, 20_000.0, -1_000.0], 7e6, 0.2, 3.986e14
        )
    ) == 3

    lambert = pk.LambertProblem(
        [1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, False, 4
    )
    assert isinstance(lambert.solutions[0], pk.LambertSolution)
    assert lambert.solutions[0].path == "zero"
    assert [solution.path for solution in lambert.solutions[1:3]] == [
        "left",
        "right",
    ]
    for solution in lambert.solutions:
        state = [*lambert.initial_position, *solution.departure_velocity]
        endpoint = pk.propagate_lagrangian(state, lambert.time, lambert.mu)
        assert endpoint[:3] == pytest.approx(lambert.final_position, abs=3e-10)

    maximum_mass, acceleration = pk.mima(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 10.0, 0.6, 4000.0
    )
    assert maximum_mass > 0.0
    assert acceleration > 0.0
    maximum_mass2, acceleration2 = pk.mima2(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        1.0,
        0.1,
        1.0,
        1.0,
    )
    assert maximum_mass2 > 0.0
    assert acceleration2 > 0.0


def test_owned_inputs_repeatability_and_thread_safety() -> None:
    """Objects own constructor data and support deterministic concurrent reuse."""
    state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    planet = pk.Planet.keplerian_from_state(0.0, state, 1.0)
    expected_state = planet.state(0.25)
    state[:] = math.nan
    del state

    controls = [[0.0, 1.0, 0.0, 0.0]]
    leg = pk.ZohLeg(
        pk.ZohModel.Kepler,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        controls,
        [0.9, 0.1, 0.0, -0.1, 0.9, 0.0, 1.0],
        [0.0, 0.1],
        [0.0],
    )
    expected_mismatch = leg.mismatch_constraints()
    controls[0][0] = math.nan
    del controls
    gc.collect()

    assert planet.state(0.25) == expected_state
    assert leg.mismatch_constraints() == expected_mismatch
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(lambda _: planet.state(0.25), range(16)))
        mismatches = list(pool.map(lambda _: leg.mismatch_constraints(), range(16)))
    assert all(value == expected_state for value in states)
    assert all(value == expected_mismatch for value in mismatches)


def test_keplerian_planet_scalar_batch_metadata_and_capabilities() -> None:
    """The Python planet owns a thread-safe native ephemeris provider."""
    planet = pk.Planet.keplerian_from_classical(
        12.22,
        [3.0, 0.2, 0.4, 0.3, 0.2, 0.1],
        1.1,
        "oracle",
        1.2,
        2.2,
        2.9,
    )
    assert planet.name == "oracle"
    assert planet.central_mu == 1.1
    assert planet.body_mu == 1.2
    assert planet.radius == 2.2
    assert planet.safe_radius == 2.9
    assert planet.period(12.22) == pytest.approx(2 * math.pi * math.sqrt(27 / 1.1))
    epochs = np.array([-100.0, 12.22, 40.0], dtype=np.float64)
    states = planet.states(epochs)
    assert states.shape == (3, 6)
    for index, epoch in enumerate(epochs):
        assert states[index] == pytest.approx(planet.state(float(epoch)))
    assert len(planet.elements(40.0, "modified_equinoctial")) == 6
    assert not planet.has_acceleration()
    with pytest.raises(pk.UnsupportedCapabilityError):
        planet.acceleration(0.0)


def test_jpl_low_precision_planets_names_window_and_batches() -> None:
    """The eight JPL low-precision providers share scalar and batch behavior."""
    names = [
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ]
    assert pk.Planet.jpl_supported_bodies() == names
    epochs = np.array([-73047.999, 0.0, 18262.999], dtype=np.float64)
    for name in names:
        planet = pk.Planet.jpl_low_precision(name.upper())
        assert planet.name == f"{name}(jpl_lp)"
        assert planet.central_mu is not None
        assert planet.body_mu is not None
        assert planet.radius is not None
        assert planet.safe_radius is not None
        assert planet.safe_radius >= planet.radius
        states = planet.states(epochs)
        assert states.shape == (3, 6)
        for index, epoch in enumerate(epochs):
            assert states[index] == pytest.approx(planet.state(float(epoch)))
        assert len(planet.elements(0.0)) == 6

    earth = pk.Planet.jpl_low_precision("earth", 7_000_000.0)
    assert earth.safe_radius == 7_000_000.0
    with pytest.raises(ValueError):
        earth.state(-73048.0)
    with pytest.raises(ValueError):
        earth.state(18263.0)
    with pytest.raises(ValueError):
        pk.Planet.jpl_low_precision("pluto")


def test_vsop2013_feature_names_thresholds_and_batches() -> None:
    """VSOP2013 availability and threshold limits are explicit in Python."""
    names = [
        "mercury",
        "venus",
        "earth_moon",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "pluto",
    ]
    assert pk.Planet.vsop2013_available()
    assert pk.Planet.vsop2013_supported_bodies() == names
    assert pk.Planet.vsop2013_minimum_threshold() == 1e-9
    venus = pk.Planet.vsop2013("vEnUs", 1e-9)
    assert "venus" in venus.name
    state = venus.state(123.0)
    assert state == pytest.approx(
        [
            103_304_986_899.7981,
            32_220_404_104.1199,
            7_957_719_449.51538,
            -10_696.505905435035,
            30_061.035989651813,
            14_201.00090492195,
        ]
    )
    epochs = np.array([-0.5, 0.0, 0.5, 123.0], dtype=np.float64)
    states = venus.states(epochs)
    assert states.shape == (4, 6)
    for index, epoch in enumerate(epochs):
        assert states[index] == pytest.approx(venus.state(float(epoch)))
    with pytest.raises(ValueError):
        pk.Planet.vsop2013("venus", 1e-10)
    with pytest.raises(ValueError):
        pk.Planet.vsop2013("goofy")


def test_public_api_has_runtime_documentation() -> None:
    """Every exported callable, method, and descriptor is documented."""
    missing: list[str] = []
    assert inspect.getdoc(pk)
    for name in pk.__all__:
        value = getattr(pk, name)
        if callable(value) and not inspect.getdoc(value):
            missing.append(name)
        if inspect.isclass(value):
            for member_name, member in vars(value).items():
                if member_name.startswith("_"):
                    continue
                resolved = getattr(value, member_name)
                if (
                    callable(resolved)
                    or inspect.ismethoddescriptor(member)
                    or inspect.isdatadescriptor(member)
                ) and not inspect.getdoc(resolved):
                    missing.append(f"{name}.{member_name}")
    assert not missing


def test_public_exception_hierarchy_and_exports_are_reachable() -> None:
    """All public exceptions are typed and every declared export is reachable."""
    assert issubclass(pk.ConvergenceError, pk.PykepError)
    assert issubclass(pk.SingularGeometryError, pk.PykepError)
    assert issubclass(pk.UnsupportedCapabilityError, pk.PykepError)
    assert issubclass(pk.IntegrationError, pk.PykepError)
    for name in pk.__all__:
        assert getattr(pk, name) is not None


def test_stub_and_runtime_exports_match() -> None:
    """The stub matches runtime names, signatures, defaults, and annotations."""
    stub_path = Path(pk.__file__).with_name("_pykep_rust.pyi")
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    declared.update(
        target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance((target := node.target), ast.Name)
    )
    assert set(pk.__all__) == declared

    missing_default = object()

    def stub_parameters(
        function: ast.FunctionDef, *, drop_self: bool = False
    ) -> list[tuple[str, object]]:
        positional = [*function.args.posonlyargs, *function.args.args]
        defaults: list[ast.expr | None] = [
            None
        ] * (len(positional) - len(function.args.defaults)) + list(
            function.args.defaults
        )
        parameters = [
            (
                argument.arg,
                missing_default if default is None else ast.literal_eval(default),
            )
            for argument, default in zip(positional, defaults, strict=True)
            if not (drop_self and argument.arg == "self")
        ]
        parameters.extend(
            (
                argument.arg,
                missing_default if default is None else ast.literal_eval(default),
            )
            for argument, default in zip(
                function.args.kwonlyargs, function.args.kw_defaults, strict=True
            )
        )
        return parameters

    def runtime_parameters(value: object) -> list[tuple[str, object]]:
        return [
            (
                parameter.name,
                missing_default
                if parameter.default is inspect.Parameter.empty
                else parameter.default,
            )
            for parameter in inspect.signature(value).parameters.values()
        ]

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert node.returns is not None, node.name
            assert stub_parameters(node) == runtime_parameters(getattr(pk, node.name))
        if not isinstance(node, ast.ClassDef):
            continue
        runtime_class = getattr(pk, node.name)
        for function in (
            member for member in node.body if isinstance(member, ast.FunctionDef)
        ):
            assert function.returns is not None, f"{node.name}.{function.name}"
            if function.name == "__init__":
                try:
                    runtime = runtime_parameters(runtime_class)
                except (TypeError, ValueError):
                    continue
                assert stub_parameters(function, drop_self=True) == runtime
                continue
            if function.name.startswith("__") or any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in function.decorator_list
            ):
                continue
            assert stub_parameters(function) == runtime_parameters(
                getattr(runtime_class, function.name)
            )
