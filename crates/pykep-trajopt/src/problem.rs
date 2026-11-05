// SPDX-License-Identifier: MPL-2.0

use pykep_core::{PykepError, Result};
use serde::Serialize;

/// Machine-readable description of a trajectory problem.
#[derive(Debug, Clone, Serialize)]
pub struct ProblemSpec {
    /// Stable problem identifier.
    pub id: String,
    /// Version of this formulation and decision layout.
    pub formulation: &'static str,
    /// Upstream commit for this definition.
    pub source: &'static str,
    /// Human-readable name.
    pub name: String,
    /// Decision labels in input order.
    pub decisions: Vec<String>,
    /// Inclusive lower and upper bound for each decision.
    pub bounds: Vec<[f64; 2]>,
    /// Objective labels and units, all minimized.
    pub objectives: Vec<String>,
    /// Raw equality residual labels.
    pub equalities: Vec<String>,
    /// Raw inequality labels, with feasible values at or below zero.
    pub inequalities: Vec<String>,
    /// Availability of analytic derivatives.
    pub has_gradient: bool,
    /// Numerical settings and physical scales specific to the formulation.
    pub settings: serde_json::Value,
}

impl ProblemSpec {
    /// Midpoint of each finite bound, including fixed coordinates.
    pub fn midpoint(&self) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|b| f64::midpoint(b[0], b[1]))
            .collect()
    }

    /// Verify dimensions and finiteness. Off-box values can be useful for differentiation.
    ///
    /// # Errors
    /// Returns a dimension or nonfinite-input error.
    pub fn check(&self, x: &[f64]) -> Result<()> {
        if x.len() != self.bounds.len() {
            return Err(PykepError::DimensionMismatch {
                expected: self.bounds.len(),
                actual: x.len(),
            });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(PykepError::NonFiniteInput {
                parameter: "decision vector",
            });
        }
        Ok(())
    }
}

/// Logical propagation work of a successful evaluation.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Work {
    /// Nominal or variational segment propagations.
    pub segments: usize,
    /// Propagated moving endpoint states.
    pub endpoints: usize,
    /// Analytical coast propagations.
    pub coasts: usize,
    /// Lambert solves.
    pub lambert: usize,
}

/// Raw evaluation, with objectives and constraints kept separate.
#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    /// Minimized objectives.
    pub objectives: Vec<f64>,
    /// Residuals whose target is zero.
    pub equalities: Vec<f64>,
    /// Residuals whose target is nonpositive.
    pub inequalities: Vec<f64>,
    /// Physical work performed by this call.
    pub work: Work,
}

impl Evaluation {
    /// Return objectives, equalities and inequalities in PyGMO order.
    pub fn fitness(&self) -> Vec<f64> {
        self.objectives
            .iter()
            .chain(&self.equalities)
            .chain(&self.inequalities)
            .copied()
            .collect()
    }

    /// Compare raw residuals against the supplied absolute tolerances.
    pub fn feasible(&self, equality_tolerance: f64, inequality_tolerance: f64) -> bool {
        self.equalities
            .iter()
            .all(|x| x.abs() <= equality_tolerance)
            && self.inequalities.iter().all(|x| *x <= inequality_tolerance)
    }
}

/// An optimizer-independent trajectory objective and its numerical capabilities.
pub trait Problem: Send + Sync {
    /// Description and decision bounds.
    fn spec(&self) -> &ProblemSpec;

    /// Evaluate raw objectives and constraints.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or an unsuccessful numerical solve.
    fn evaluate(&self, x: &[f64]) -> Result<Evaluation>;

    /// Complete gradient in fitness-row by decision-column order.
    ///
    /// # Errors
    /// Returns an unsupported capability or numerical evaluation error.
    fn gradient(&self, _x: &[f64]) -> Result<Vec<Vec<f64>>> {
        Err(PykepError::UnsupportedCapability {
            provider: self.spec().id.clone(),
            capability: "analytic gradient",
        })
    }

    /// Decode states, controls and trajectory information for replay.
    ///
    /// # Errors
    /// Returns an error if the trajectory cannot be reconstructed.
    fn decode(&self, x: &[f64], samples_per_segment: usize) -> Result<serde_json::Value>;

    /// Evaluate an ordered population using the core's bounded worker pool.
    ///
    /// # Errors
    /// Returns the earliest input error in population order.
    fn evaluate_batch(&self, x: &[Vec<f64>], workers: usize) -> Result<Vec<Evaluation>> {
        pykep_core::batch::try_map(x, workers, |decision| self.evaluate(decision))
    }
}

pub(crate) fn invalid(parameter: &'static str, reason: impl Into<String>) -> PykepError {
    PykepError::InvalidInput {
        parameter,
        reason: reason.into(),
    }
}
