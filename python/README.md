# Python API

The `pykep_rust` package will be a thin, typed Python interface to
`pykep-core`. It currently exposes only `port_status()` so that packaging can
be validated before numerical APIs are committed.

The existing upstream Python wrapper is deliberately out of scope for the
first porting pass. Compatibility names and migration helpers will be designed
only after the Rust core has stable numerical behavior.

