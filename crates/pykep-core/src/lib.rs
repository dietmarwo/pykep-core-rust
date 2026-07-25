#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]

/// Orbital-mechanics algorithms and coordinate conversions.
pub mod astro;
/// Physical and astrodynamical constants.
pub mod constants;
/// Evaluated Kepler, CR3BP, and bicircular dynamics models.
pub mod dynamics;
/// Ephemeris providers and the object-safe planet interface.
pub mod ephemeris;
/// Stable error categories returned by numerical APIs.
pub mod error;
/// Adaptive integration interfaces used by evaluated dynamics models.
pub mod integration;
/// Small, dependency-free numerical helpers.
pub mod math;
/// Time representations and conversions.
pub mod time;
/// Fixed-shape numerical types used throughout the crate.
pub mod types;

pub use error::{PykepError, Result};
pub use types::{CartesianState, Elements6, Matrix3, Matrix6, Vector3};

/// Current implementation status exposed by both the Rust and Python smoke
/// tests.
pub const PORT_STATUS: &str = "phase 13: Pontryagin dynamics";

#[cfg(test)]
mod tests {
    use super::PORT_STATUS;

    #[test]
    fn status_reports_pontryagin_dynamics() {
        assert!(PORT_STATUS.contains("phase 13"));
        assert!(PORT_STATUS.contains("Pontryagin"));
    }
}
