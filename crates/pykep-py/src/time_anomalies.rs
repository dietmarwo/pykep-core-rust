// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use pykep_core::astro::anomalies;
use pykep_core::time::epoch::Epoch;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;

/// A microsecond-resolution proleptic-Gregorian epoch.
///
/// The default numeric scale is MJD2000. Julian day values are arithmetic
/// counts and do not model leap seconds, TT, or TDB.
#[pyclass(
    name = "Epoch",
    frozen,
    module = "pykep_rust._pykep_rust",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyEpoch {
    inner: Epoch,
}

#[pymethods]
impl PyEpoch {
    /// Construct an epoch from a numeric day count.
    #[new]
    #[pyo3(signature = (value=0.0, scale="mjd2000"))]
    fn new(value: f64, scale: &str) -> PyResult<Self> {
        let inner = match scale {
            "mjd2000" => Epoch::from_mjd2000(value),
            "mjd" => Epoch::from_mjd(value),
            "jd" => Epoch::from_jd(value),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "scale must be 'mjd2000', 'mjd', or 'jd'",
                ));
            }
        }
        .map_err(to_python)?;
        Ok(Self { inner })
    }

    /// Parse a cropped ISO calendar string.
    #[staticmethod]
    fn from_iso(text: &str) -> PyResult<Self> {
        Epoch::from_iso(text)
            .map(|inner| Self { inner })
            .map_err(to_python)
    }

    /// Construct an epoch from validated calendar components.
    #[staticmethod]
    #[pyo3(signature = (
        year,
        month,
        day,
        hour=0,
        minute=0,
        second=0,
        millisecond=0,
        microsecond=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_calendar(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        millisecond: u32,
        microsecond: u32,
    ) -> PyResult<Self> {
        Epoch::from_calendar(
            year,
            month,
            day,
            hour,
            minute,
            second,
            millisecond,
            microsecond,
        )
        .map(|inner| Self { inner })
        .map_err(to_python)
    }

    /// Return an epoch sampled from the current system clock.
    #[staticmethod]
    fn now() -> PyResult<Self> {
        Epoch::now().map(|inner| Self { inner }).map_err(to_python)
    }

    /// Modified Julian Date 2000, in days.
    #[getter]
    fn mjd2000(&self) -> f64 {
        self.inner.mjd2000()
    }

    /// Modified Julian Date, in days.
    #[getter]
    fn mjd(&self) -> f64 {
        self.inner.mjd()
    }

    /// Julian Date, in days.
    #[getter]
    fn jd(&self) -> f64 {
        self.inner.jd()
    }

    /// Signed internal microseconds from 2000-01-01T00:00:00.
    #[getter]
    fn microseconds_since_mjd2000(&self) -> i64 {
        self.inner.microseconds_since_mjd2000()
    }

    /// Return the canonical six-fractional-digit ISO representation.
    fn to_iso(&self) -> String {
        self.inner.to_iso()
    }

    /// Return a new epoch offset by a finite number of days.
    fn add_days(&self, days: f64) -> PyResult<Self> {
        self.inner
            .checked_add_days(days)
            .map(|inner| Self { inner })
            .map_err(to_python)
    }

    /// Return a new epoch offset backwards by a finite number of days.
    fn sub_days(&self, days: f64) -> PyResult<Self> {
        self.inner
            .checked_sub_days(days)
            .map(|inner| Self { inner })
            .map_err(to_python)
    }

    /// Return a new epoch offset by a finite number of seconds.
    fn add_seconds(&self, seconds: f64) -> PyResult<Self> {
        self.inner
            .checked_add_seconds(seconds)
            .map(|inner| Self { inner })
            .map_err(to_python)
    }

    /// Return `self - other` in seconds.
    fn seconds_since(&self, other: &Self) -> PyResult<f64> {
        self.inner
            .duration_seconds_since(other.inner)
            .map_err(to_python)
    }

    fn __repr__(&self) -> String {
        format!("Epoch.from_iso('{}')", self.inner.to_iso())
    }

    fn __str__(&self) -> String {
        self.inner.to_iso()
    }

    fn __richcmp__(&self, other: &Self, operation: CompareOp) -> bool {
        match operation {
            CompareOp::Lt => self.inner < other.inner,
            CompareOp::Le => self.inner <= other.inner,
            CompareOp::Eq => self.inner == other.inner,
            CompareOp::Ne => self.inner != other.inner,
            CompareOp::Gt => self.inner > other.inner,
            CompareOp::Ge => self.inner >= other.inner,
        }
    }

    fn __hash__(&self) -> i64 {
        self.inner.microseconds_since_mjd2000()
    }

    fn __add__(&self, days: f64) -> PyResult<Self> {
        self.add_days(days)
    }

    fn __sub__(&self, days: f64) -> PyResult<Self> {
        self.sub_days(days)
    }
}

macro_rules! anomaly_wrapper {
    ($python_name:ident, $core_name:ident, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        fn $python_name(anomaly: f64, eccentricity: f64) -> PyResult<f64> {
            anomalies::$core_name(anomaly, eccentricity).map_err(to_python)
        }
    };
}

anomaly_wrapper!(
    mean_to_eccentric_anomaly,
    mean_to_eccentric_anomaly,
    "Convert elliptic mean anomaly to principal eccentric anomaly, in radians."
);
anomaly_wrapper!(
    eccentric_to_mean_anomaly,
    eccentric_to_mean_anomaly,
    "Convert eccentric anomaly to elliptic mean anomaly, in radians."
);
anomaly_wrapper!(
    eccentric_to_true_anomaly,
    eccentric_to_true_anomaly,
    "Convert eccentric anomaly to principal true anomaly, in radians."
);
anomaly_wrapper!(
    true_to_eccentric_anomaly,
    true_to_eccentric_anomaly,
    "Convert true anomaly to principal eccentric anomaly, in radians."
);
anomaly_wrapper!(
    mean_to_true_anomaly,
    mean_to_true_anomaly,
    "Convert elliptic mean anomaly to principal true anomaly, in radians."
);
anomaly_wrapper!(
    true_to_mean_anomaly,
    true_to_mean_anomaly,
    "Convert true anomaly to elliptic mean anomaly, in radians."
);
anomaly_wrapper!(
    gudermannian_to_true_anomaly,
    gudermannian_to_true_anomaly,
    "Convert Gudermannian anomaly to hyperbolic true anomaly, in radians."
);
anomaly_wrapper!(
    true_to_gudermannian_anomaly,
    true_to_gudermannian_anomaly,
    "Convert hyperbolic true anomaly to Gudermannian anomaly, in radians."
);
anomaly_wrapper!(
    hyperbolic_mean_to_anomaly,
    hyperbolic_mean_to_anomaly,
    "Convert hyperbolic mean anomaly to hyperbolic anomaly, in radians."
);
anomaly_wrapper!(
    hyperbolic_anomaly_to_mean,
    hyperbolic_anomaly_to_mean,
    "Convert hyperbolic anomaly to hyperbolic mean anomaly, in radians."
);
anomaly_wrapper!(
    hyperbolic_anomaly_to_true,
    hyperbolic_anomaly_to_true,
    "Convert hyperbolic anomaly to principal true anomaly, in radians."
);
anomaly_wrapper!(
    true_to_hyperbolic_anomaly,
    true_to_hyperbolic_anomaly,
    "Convert true anomaly to hyperbolic anomaly, in radians."
);
anomaly_wrapper!(
    hyperbolic_mean_to_true,
    hyperbolic_mean_to_true,
    "Convert hyperbolic mean anomaly to principal true anomaly, in radians."
);
anomaly_wrapper!(
    true_to_hyperbolic_mean,
    true_to_hyperbolic_mean,
    "Convert true anomaly to hyperbolic mean anomaly, in radians."
);

/// Convert elliptic mean anomalies in input order at one eccentricity.
#[pyfunction]
fn mean_to_eccentric_anomaly_batch(
    python: Python<'_>,
    mean_anomalies: Vec<f64>,
    eccentricity: f64,
) -> PyResult<Vec<f64>> {
    python
        .detach(move || {
            mean_anomalies
                .into_iter()
                .map(|mean| anomalies::mean_to_eccentric_anomaly(mean, eccentricity))
                .collect::<pykep_core::Result<Vec<_>>>()
        })
        .map_err(to_python)
}

/// Convert hyperbolic mean anomalies in input order at one eccentricity.
#[pyfunction]
fn hyperbolic_mean_to_anomaly_batch(
    python: Python<'_>,
    mean_anomalies: Vec<f64>,
    eccentricity: f64,
) -> PyResult<Vec<f64>> {
    python
        .detach(move || {
            mean_anomalies
                .into_iter()
                .map(|mean| anomalies::hyperbolic_mean_to_anomaly(mean, eccentricity))
                .collect::<pykep_core::Result<Vec<_>>>()
        })
        .map_err(to_python)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEpoch>()?;
    module.add_function(wrap_pyfunction!(mean_to_eccentric_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(eccentric_to_mean_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(eccentric_to_true_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(true_to_eccentric_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(mean_to_true_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(true_to_mean_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(gudermannian_to_true_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(true_to_gudermannian_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_mean_to_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_anomaly_to_mean, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_anomaly_to_true, module)?)?;
    module.add_function(wrap_pyfunction!(true_to_hyperbolic_anomaly, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_mean_to_true, module)?)?;
    module.add_function(wrap_pyfunction!(true_to_hyperbolic_mean, module)?)?;
    module.add_function(wrap_pyfunction!(mean_to_eccentric_anomaly_batch, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_mean_to_anomaly_batch, module)?)?;
    Ok(())
}
