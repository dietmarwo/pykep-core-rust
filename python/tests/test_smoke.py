"""Installed-extension smoke tests."""

from __future__ import annotations

import inspect

import pykep_rust


def test_status_probe_uses_native_core() -> None:
    """The public facade returns the explicit native scaffold status."""
    assert pykep_rust.port_status() == "scaffold: numerical port not started"


def test_public_smoke_api_is_documented() -> None:
    """Every exported scaffold symbol has runtime documentation."""
    assert inspect.getdoc(pykep_rust)
    assert pykep_rust.__all__ == ["port_status"]
    assert inspect.getdoc(pykep_rust.port_status)
