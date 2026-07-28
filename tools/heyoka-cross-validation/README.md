# heyoka cross-validation

This optional development harness generates independent reference states with
the official `heyoka.py` package. It is deliberately absent from normal CI and
from the shipped Rust crate.

From the repository root:

```bash
python -m pip install "heyoka==7.10.1"
python tools/heyoka-cross-validation/generate.py
cargo test -p pykep-core --test taylor_integration heyoka
```

The committed fixture covers the autonomous Kepler and CR3BP systems and the
non-autonomous BCP system. Kepler also carries the first-order STM. The larger
ZOH and Pontryagin families are checked against the already committed upstream
pykep fixtures and against DOP853 in Rust; the harness does not pretend that
those comparisons came from heyoka.
