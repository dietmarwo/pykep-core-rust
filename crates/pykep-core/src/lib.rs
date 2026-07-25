//! Native Rust astrodynamics core for the pykep-rust port.
//!
//! The crate is currently a buildable scaffold. Numerical modules will be
//! introduced in the order documented in the repository's port plan.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Current implementation status exposed by both the Rust and Python smoke
/// tests.
pub const PORT_STATUS: &str = "scaffold: numerical port not started";

#[cfg(test)]
mod tests {
    use super::PORT_STATUS;

    #[test]
    fn status_is_explicit_about_the_scaffold() {
        assert!(PORT_STATUS.contains("not started"));
    }
}
