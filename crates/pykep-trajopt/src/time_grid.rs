// SPDX-License-Identifier: MPL-2.0

use crate::problem::invalid;
use pykep_core::Result;
use serde::Serialize;

/// Segment duration encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeEncoding {
    /// Equal segment durations.
    #[default]
    Uniform,
    /// Positive duration weights from one logit per segment.
    Softmax,
}

impl std::str::FromStr for TimeEncoding {
    type Err = pykep_core::PykepError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "uniform" => Ok(Self::Uniform),
            "softmax" => Ok(Self::Softmax),
            _ => Err(invalid("time_encoding", "expected uniform or softmax")),
        }
    }
}

pub(crate) struct Grid {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Grid {
    pub fn new(n: usize, tof: f64, logits: &[f64]) -> Result<Self> {
        if !tof.is_finite() || tof <= 0.0 || n == 0 {
            return Err(invalid(
                "time grid",
                "requires a positive finite duration and segment count",
            ));
        }
        if !logits.is_empty() && logits.len() != n {
            return Err(invalid("time grid", "expected one logit per segment"));
        }
        let mut weights = Vec::new();
        let mut nodes = Vec::with_capacity(n + 1);
        if logits.is_empty() {
            nodes.extend((0..=n).map(|i| tof * (i as f64 / n as f64)));
        } else {
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            weights.extend(logits.iter().map(|value| (value - max).exp()));
            let sum: f64 = weights.iter().sum();
            nodes.push(0.0);
            for weight in &mut weights {
                *weight /= sum;
                nodes.push(nodes.last().copied().unwrap_or_default() + tof * *weight);
            }
        }
        nodes[n] = tof;
        if nodes.windows(2).any(|p| p[0] >= p[1] || !p[0].is_finite()) {
            return Err(invalid(
                "time grid",
                "segment duration rounded to zero or overflowed",
            ));
        }
        Ok(Self { nodes, weights })
    }
}

pub(crate) fn contract(
    nodes: &[f64],
    weights: &[f64],
    tof: f64,
    derivatives: &[f64],
    result: &mut [f64],
) -> f64 {
    let d_tof = nodes
        .iter()
        .zip(derivatives)
        .map(|(t, d)| (t / tof) * d)
        .sum::<f64>();
    let mut suffix = 0.0;
    for (j, weight) in weights.iter().enumerate().rev() {
        suffix += derivatives[j + 1];
        result[j] = tof * weight * (suffix - d_tof);
    }
    d_tof
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contracts_softmax_and_duration_derivatives() {
        let x = [0.7, -0.4, 0.1];
        let d = [0., 2., -1.5, 3.];
        let tof = 4.5;
        let mut rows = [0.; 3];
        let grid = Grid::new(3, tof, &x).unwrap();
        let dt = contract(&grid.nodes, &grid.weights, tof, &d, &mut rows);
        let eval = |t, w: &[f64]| {
            Grid::new(3, t, w)
                .unwrap()
                .nodes
                .iter()
                .zip(d)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let h = 1e-5;
        assert!((dt - (eval(tof + h, &x) - eval(tof - h, &x)) / (2. * h)).abs() < 1e-9);
        for (i, row) in rows.iter().enumerate() {
            let mut plus = x;
            let mut minus = x;
            plus[i] += h;
            minus[i] -= h;
            assert!((row - (eval(tof, &plus) - eval(tof, &minus)) / (2. * h)).abs() < 1e-9);
        }
        assert!(rows.iter().sum::<f64>().abs() < 1e-14);
        assert!(Grid::new(2, tof, &[0., -1000.]).is_err());
    }
}
