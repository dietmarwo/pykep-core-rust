"""Smoke-test an installed pykep-rust wheel or source distribution."""

from __future__ import annotations

import os
import tempfile
from importlib.metadata import version


def main() -> None:
    """Import from outside the checkout and exercise representative native APIs."""
    os.chdir(tempfile.mkdtemp(prefix="pykep-rust-smoke-"))

    import numpy as np
    import pykep_rust as pk

    installed_version = version("pykep-rust")
    expected_version = os.environ.get("EXPECTED_VERSION")
    if expected_version is not None:
        assert installed_version == expected_version

    assert pk.port_status() == f"pykep-core {installed_version}"
    epoch = pk.Epoch.from_iso("2030-01")
    assert epoch.to_iso() == "2030-01-01T00:00:00.000000"
    states = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    propagated = pk.propagate_lagrangian_batch(states, np.array([0.1]), 1.0)
    assert propagated.shape == (1, 6)
    assert np.isfinite(propagated).all()
    print(f"pykep-rust {installed_version}: {pk.port_status()}")


if __name__ == "__main__":
    main()
