#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]

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
