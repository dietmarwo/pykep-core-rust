// SPDX-License-Identifier: MPL-2.0
use std::ops::Range;

use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint};
use pykep_core::{CartesianState, PykepError, Result, Vector3};

use crate::problem::invalid;

struct Arc {
    left: Range<usize>,
    controls: Range<usize>,
    right: Range<usize>,
    settings: SimsFlanaganSettings,
    forward_segments: usize,
}

/// Multiple shooting for one fixed-duration Sims–Flanagan transfer.
///
/// The decision layout is departure velocity excess, then each arc's
/// Cartesian throttle vectors and its right node `[r, v, mass]`. Only the
/// mass is variable at the final node. Each arc has seven matching rows
/// followed by the squared throttle norms of its segments. A single
/// global cut is retained, including the original backward mass update
/// convention on its far side.
pub struct SimsFlanaganChain {
    departure: SpacecraftEndpoint,
    arrival: CartesianState,
    segments: usize,
    arcs: Vec<Arc>,
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    elements: Vec<Vec<usize>>,
}

impl SimsFlanaganChain {
    /// Builds a uniform-segment chain. The total time and fractional cut
    /// are those of the corresponding single leg.
    ///
    /// # Errors
    /// Rejects invalid endpoints, settings, or empty or oversized layouts.
    pub fn new(
        departure: SpacecraftEndpoint,
        arrival: CartesianState,
        arc_count: usize,
        segments_per_arc: usize,
        settings: SimsFlanaganSettings,
    ) -> Result<Self> {
        let departure = SpacecraftEndpoint::new(departure.state, departure.mass)?;
        SpacecraftEndpoint::new(arrival, departure.mass)?;
        let settings = SimsFlanaganSettings::new(
            settings.time_of_flight,
            settings.maximum_thrust,
            settings.exhaust_velocity,
            settings.mu,
            settings.cut,
        )?;
        let segments = arc_count
            .checked_mul(segments_per_arc)
            .filter(|&count| count > 0 && count <= 100_000)
            .ok_or_else(|| invalid("arc layout", "requires 1 to 100,000 total segments"))?;
        let forward = (segments as f64 * settings.cut) as usize;
        let mut arcs = Vec::with_capacity(arc_count);
        let mut start = 3;
        let mut left = 0..3;
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut elements = Vec::with_capacity(arc_count);
        for i in 0..arc_count {
            let controls = start..start + 3 * segments_per_arc;
            let right = controls.end..controls.end + if i + 1 == arc_count { 1 } else { 7 };
            let local_forward = forward
                .saturating_sub(i * segments_per_arc)
                .min(segments_per_arc);
            let cut = if local_forward == segments_per_arc {
                1.0
            } else {
                (local_forward as f64 + 0.5) / segments_per_arc as f64
            };
            let arc_settings = SimsFlanaganSettings {
                time_of_flight: settings.time_of_flight / arc_count as f64,
                cut,
                ..settings
            };
            let first = indptr.len() - 1;
            for _ in 0..7 {
                indices.extend(left.start..right.end);
                indptr.push(indices.len());
            }
            for control in controls.clone().step_by(3) {
                indices.extend(control..control + 3);
                indptr.push(indices.len());
            }
            elements.push((first..indptr.len() - 1).collect());
            start = right.end;
            arcs.push(Arc {
                left,
                controls,
                right: right.clone(),
                settings: arc_settings,
                forward_segments: local_forward,
            });
            left = right;
        }
        Ok(Self {
            departure,
            arrival,
            segments,
            arcs,
            n: start,
            indptr,
            indices,
            elements,
        })
    }

    /// Number of variables in the arc-ordered decision vector.
    pub const fn dimension(&self) -> usize {
        self.n
    }

    /// Zero-based CSR row pointers and column indices for the constraint Jacobian.
    pub fn jacobian_pattern(&self) -> (&[usize], &[usize]) {
        (&self.indptr, &self.indices)
    }

    /// Constraint row groups for a partitioned SQP Hessian.
    pub fn elements(&self) -> &[Vec<usize>] {
        &self.elements
    }

    fn check(&self, x: &[f64]) -> Result<()> {
        if x.len() != self.n {
            return Err(PykepError::DimensionMismatch {
                expected: self.n,
                actual: x.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PykepError::NonFiniteInput {
                parameter: "decision vector",
            });
        }
        Ok(())
    }

    fn left(&self, x: &[f64], arc: &Arc) -> Result<SpacecraftEndpoint> {
        if arc.left.start == 0 {
            let mut state = self.departure.state;
            for i in 0..3 {
                state[i + 3] += x[i];
            }
            SpacecraftEndpoint::new(state, self.departure.mass)
        } else {
            node(&x[arc.left.clone()])
        }
    }

    fn right(&self, x: &[f64], arc: &Arc) -> Result<SpacecraftEndpoint> {
        if arc.right.len() == 1 {
            SpacecraftEndpoint::new(self.arrival, x[arc.right.start])
        } else {
            node(&x[arc.right.clone()])
        }
    }

    fn leg(&self, x: &[f64], arc: &Arc) -> Result<SimsFlanaganLeg> {
        SimsFlanaganLeg::new(
            self.left(x, arc)?,
            controls(x, arc),
            self.right(x, arc)?,
            arc.settings,
        )
    }

    /// Returns negative arrival mass and the raw rows, in element order.
    ///
    /// # Errors
    /// Returns an error for invalid decisions or failed propagation.
    pub fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.check(x)?;
        let mut rows = Vec::with_capacity(self.indptr.len() - 1);
        for arc in &self.arcs {
            let leg = self.leg(x, arc)?;
            rows.extend(leg.mismatch_constraints()?);
            rows.extend(
                leg.throttles()
                    .iter()
                    .map(|u| u.iter().map(|v| v * v).sum::<f64>()),
            );
        }
        if rows.iter().any(|v| !v.is_finite()) {
            return Err(PykepError::NumericalOverflow {
                operation: "shooting constraints",
            });
        }
        Ok((-x[self.n - 1], rows))
    }

    /// Returns the analytic constraint Jacobian in the declared CSR order.
    /// The objective gradient is -1 in the last column and zero elsewhere.
    ///
    /// # Errors
    /// Returns an error for invalid decisions or failed variational propagation.
    pub fn jacobian_values(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.check(x)?;
        let mut values = Vec::with_capacity(self.indices.len());
        for arc in &self.arcs {
            let leg = self.leg(x, arc)?;
            let jac = leg.mismatch_jacobian()?;
            let left = if arc.left.start == 0 { 3..6 } else { 0..7 };
            let right = if arc.right.len() == 1 { 6..7 } else { 0..7 };
            for i in 0..7 {
                values.extend_from_slice(&jac.departure[i][left.clone()]);
                values.extend_from_slice(&jac.controls_and_time[i][..arc.controls.len()]);
                values.extend_from_slice(&jac.arrival[i][right.clone()]);
            }
            values.extend(x[arc.controls.clone()].iter().map(|v| 2.0 * v));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(PykepError::NumericalOverflow {
                operation: "shooting Jacobian",
            });
        }
        Ok(values)
    }

    /// Lifts a single-leg control/mass/excess decision into arc coordinates.
    ///
    /// Interior nodes are propagated from their corresponding endpoint.
    /// The cut mismatch remains in one arc if the original point is infeasible.
    ///
    /// # Errors
    /// Rejects invalid controls or endpoints and propagates numerical errors.
    pub fn lift(
        &self,
        throttle: &[Vector3],
        arrival_mass: f64,
        departure_excess: Vector3,
    ) -> Result<Vec<f64>> {
        if throttle.len() != self.segments {
            return Err(PykepError::DimensionMismatch {
                expected: self.segments,
                actual: throttle.len(),
            });
        }
        let mut x = vec![0.0; self.n];
        x[..3].copy_from_slice(&departure_excess);
        x[self.n - 1] = arrival_mass;
        for (i, arc) in self.arcs.iter().enumerate() {
            let width = arc.controls.len() / 3;
            for (out, u) in x[arc.controls.clone()]
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(&throttle[i * width..(i + 1) * width])
            {
                out.copy_from_slice(u);
            }
        }
        self.check(&x)?;
        let departure = self.left(&x, &self.arcs[0])?;
        let arrival = SpacecraftEndpoint::new(self.arrival, arrival_mass)?;
        let mut current = departure;
        for arc in self
            .arcs
            .iter()
            .take(self.arcs.len() - 1)
            .take_while(|arc| arc.forward_segments == arc.controls.len() / 3)
        {
            let leg = SimsFlanaganLeg::new(current, controls(&x, arc), arrival, arc.settings)?;
            current = leg.matchpoint_states()?[0];
            write_node(&mut x[arc.right.clone()], current);
        }
        current = arrival;
        for arc in self.arcs.iter().rev().take(self.arcs.len() - 1) {
            // A fully backward arc can seed its left node, except when that
            // node was already set by the forward half at a boundary cut.
            if arc.forward_segments != 0 || x[arc.left.end - 1] != 0.0 {
                break;
            }
            let leg = SimsFlanaganLeg::new(departure, controls(&x, arc), current, arc.settings)?;
            current = leg.matchpoint_states()?[1];
            write_node(&mut x[arc.left.clone()], current);
        }
        Ok(x)
    }
}

fn controls(x: &[f64], arc: &Arc) -> Vec<Vector3> {
    x[arc.controls.clone()].as_chunks::<3>().0.to_vec()
}

fn node(x: &[f64]) -> Result<SpacecraftEndpoint> {
    SpacecraftEndpoint::new(x[..6].try_into().expect("six state entries"), x[6])
}

fn write_node(out: &mut [f64], endpoint: SpacecraftEndpoint) {
    out[..6].copy_from_slice(&endpoint.state);
    out[6] = endpoint.mass;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lift_preserves_the_single_leg_cut_for_boundary_and_interior_cuts() {
        let dep = SpacecraftEndpoint::new([1., 0., 0., 0., 1., 0.], 1.).unwrap();
        let arrival = [0.96, 0.28, 0.01, -0.28, 0.96, 0.];
        let excess = [0.0, 0.02, 0.001];
        let settings = SimsFlanaganSettings::new(0.3, 0.12, 1.5, 1., 0.5).unwrap();
        for arcs in [3, 4] {
            let throttle = vec![[0.04, 0.22, -0.01]; arcs * 4];
            let chain = SimsFlanaganChain::new(dep, arrival, arcs, 4, settings).unwrap();
            let x = chain.lift(&throttle, 0.98, excess).unwrap();
            let a = SpacecraftEndpoint::new(arrival, 0.98).unwrap();
            let d = chain.left(&x, &chain.arcs[0]).unwrap();
            let expected = SimsFlanaganLeg::new(d, throttle, a, settings)
                .unwrap()
                .mismatch_constraints()
                .unwrap();
            let (_, rows) = chain.evaluate(&x).unwrap();
            for (arc, group) in rows.as_chunks::<11>().0.iter().enumerate() {
                for (j, &v) in group[..7].iter().enumerate() {
                    let target = if arc == arcs / 2 { expected[j] } else { 0.0 };
                    assert!((v - target).abs() < 2e-13, "arc {arc} row {j}");
                }
            }
            assert_eq!(
                chain.jacobian_values(&x).unwrap().len(),
                chain.jacobian_pattern().1.len()
            );
        }
    }
}
