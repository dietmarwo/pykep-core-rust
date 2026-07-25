# pykep-core

`pykep-core` is the future native Rust implementation of the numerical C++
library in pykep version 3. It currently contains only a status probe used to
validate the workspace and Python binding boundary.

The completed crate is intended to have no C or C++ runtime dependency.

## Current example

```
assert!(pykep_core::PORT_STATUS.contains("scaffold"));
```

The crate is deliberately marked `publish = false` until a useful,
numerically validated module replaces the scaffold-only surface.
