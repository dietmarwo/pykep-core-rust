"""Installed-extension and foundation API tests."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

import pykep_rust as pk


def test_status_probe_reports_foundations() -> None:
    """The public facade reports the current native implementation phase."""
    assert pk.port_status() == "phase 2: numerical foundations implemented"


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
