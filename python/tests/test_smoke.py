"""Installed-extension and foundation API tests."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import pykep_rust as pk


def test_status_probe_reports_vsop_ephemerides() -> None:
    """The public facade reports the current native implementation phase."""
    assert pk.port_status() == "phase 9: VSOP2013 ephemerides implemented"


def test_constants_and_julian_conversions() -> None:
    """Pinned constants and reference-epoch conversions reach Python."""
    assert pk.PI == math.pi
    assert pk.ASTRONOMICAL_UNIT == 149_597_870_700.0
    assert pk.jd_to_mjd(2_451_544.5) == 51_544.0
    assert pk.jd_to_mjd2000(2_451_544.5) == 0.0
    assert pk.mjd2000_to_jd(0.0) == 2_451_544.5
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
    universal = pk.universal_kepler_residual(0.4, 20.0, 7.0, 0.2, 0.01, 3.0)
    assert math.isfinite(universal)


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
    means = [-4.0, 0.0, 0.1, 100.0]
    assert pk.mean_to_eccentric_anomaly_batch(means, 0.9) == [
        pk.mean_to_eccentric_anomaly(value, 0.9) for value in means
    ]
    hyperbolic = pk.hyperbolic_mean_to_anomaly(20.0, 10.0)
    assert pk.hyperbolic_anomaly_to_mean(hyperbolic, 10.0) == pytest.approx(20.0)
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

    lambert = pk.LambertProblem(
        [1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, False, 4
    )
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
    """Every exported callable and exception has runtime documentation."""
    missing: list[str] = []
    assert inspect.getdoc(pk)
    for name in pk.__all__:
        value = getattr(pk, name)
        if callable(value) and not inspect.getdoc(value):
            missing.append(name)
    assert not missing


def test_stub_and_runtime_exports_match() -> None:
    """The native extension stub declares every runtime extension symbol."""
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
