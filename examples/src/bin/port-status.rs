//! Print the implementation status reported by `pykep-core`.
//!
//! Units: none.
//! Expected: the current numbered plan phase.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: works without optional `pykep-core` data features.

fn main() {
    println!("{}", pykep_core::PORT_STATUS);
}
