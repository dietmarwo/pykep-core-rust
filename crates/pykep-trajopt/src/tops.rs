// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
// Decision layouts and normalization follow the pinned PyKEP TOPS transcriptions.

use core::f64::consts::{FRAC_PI_2, TAU};

use pykep_core::Result;
use pykep_core::dynamics::zoh::{
    ZeroOrderHoldModel, ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics,
    ZohSolarSailDynamics, propagate_schedule_with_method,
};
use pykep_core::integration::{IntegrationMethod, IntegratorOptions, TaylorDynamicsModel};
use pykep_core::leg::ZohLeg;
use serde::Serialize;
use serde_json::{Value, json};

use crate::boundary::{BoundaryPair, Endpoints};
use crate::problem::invalid;
use crate::time_grid::Grid;
use crate::tops_data::{RawCase, TopsFamily, case, required};
use crate::{Evaluation, Problem, ProblemSpec, TimeEncoding, UPSTREAM, Work};

/// Numerical and transcription settings for TOPS.
#[derive(Debug, Clone, Copy)]
pub struct TopsOptions {
    /// Segment count, or the upstream family default.
    pub segments: Option<usize>,
    /// Fraction assigned to the forward half, rounded down to a segment boundary.
    pub cut: f64,
    /// Decision encoding of the time grid.
    pub time_encoding: TimeEncoding,
    /// Direction constraints as inequalities when true, or the family default.
    pub direction_inequalities: Option<bool>,
    /// Nominal and sensitivity propagation backend.
    pub method: IntegrationMethod,
    /// Local error target for nominal propagation.
    pub tolerance: f64,
    /// Local error target for sensitivity propagation.
    pub derivative_tolerance: f64,
}

impl Default for TopsOptions {
    fn default() -> Self {
        Self {
            segments: None,
            cut: 0.5,
            time_encoding: TimeEncoding::Uniform,
            direction_inequalities: None,
            method: IntegrationMethod::Taylor,
            tolerance: 1e-13,
            derivative_tolerance: 1e-13,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct Scales {
    pub length: f64,
    pub time: f64,
    pub mass: f64,
}

impl Scales {
    pub fn velocity(self) -> f64 {
        self.length / self.time
    }

    fn acceleration(self) -> f64 {
        self.velocity() / self.time
    }

    pub fn cartesian(self, mut state: [f64; 6]) -> [f64; 6] {
        for (i, value) in state.iter_mut().enumerate() {
            *value /= if i < 3 { self.length } else { self.velocity() };
        }
        state
    }

    fn physical(self, raw: &RawCase) -> Self {
        Self {
            length: self.length * raw.length,
            time: self.time * raw.time,
            mass: self.mass * raw.mass.unwrap_or(1.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    mass: Option<usize>,
    directions: Option<[usize; 2]>,
    controls: usize,
    width: usize,
    tof: usize,
}

impl Layout {
    fn new(n: usize, sail: bool, moving: bool) -> Self {
        let mass = (!sail).then_some(usize::from(moving));
        let prefix = usize::from(moving) + usize::from(!sail);
        let directions = moving.then_some([prefix + 1, prefix + 5]);
        let controls = prefix + if moving { 8 } else { 0 };
        let width = if sail { 2 } else { 4 };
        Self {
            mass,
            directions,
            controls,
            width,
            tof: controls + n * width,
        }
    }
}

/// A fixed or moving boundary TOPS nonlinear program.
#[derive(Debug, Clone)]
pub struct TopsProblem {
    spec: ProblemSpec,
    family: TopsFamily,
    options: TopsOptions,
    segments: usize,
    layout: Layout,
    scales: Scales,
    initial: [f64; 6],
    final_state: [f64; 6],
    initial_mass: f64,
    thrust: f64,
    constants: [f64; 2],
    boundary: Option<BoundaryPair>,
    direction_inequalities: bool,
}

impl TopsProblem {
    /// Construct a fixed-boundary TOPS problem.
    ///
    /// # Errors
    /// Returns an error for invalid options or an unknown instance.
    pub fn fixed(family: TopsFamily, id: &str, options: TopsOptions) -> Result<Self> {
        Self::new(family, id, options, false)
    }

    /// Construct a moving-boundary TOPS problem.
    ///
    /// # Errors
    /// Returns an error for invalid options or an unknown instance.
    pub fn moving(family: TopsFamily, id: &str, options: TopsOptions) -> Result<Self> {
        Self::new(family, id, options, true)
    }

    fn new(family: TopsFamily, id: &str, options: TopsOptions, moving: bool) -> Result<Self> {
        let raw = case(family, id)?;
        let sail = family == TopsFamily::Sail;
        let cr3bp = family == TopsFamily::Cr3bp;
        let default_segments = if cr3bp && !moving { 60 } else { 10 };
        let segments = options.segments.unwrap_or(default_segments);
        if !(1..=512).contains(&segments) {
            return Err(invalid("segments", "expected 1..=512"));
        }
        if !(0.0..=1.0).contains(&options.cut)
            || [options.tolerance, options.derivative_tolerance]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid(
                "TOPS options",
                "cut and error tolerances are outside their domains",
            ));
        }
        let normalise = !(cr3bp || moving && sail);
        let length = if !normalise {
            1.0
        } else if family == TopsFamily::Mee {
            raw.state_s[0]
        } else {
            raw.state_s[..3].iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        let raw_mass = if sail { 1.0 } else { required(raw.m_s)? };
        let scales = Scales {
            length,
            time: if normalise {
                (length.powi(3) / required(raw.mu)?).sqrt()
            } else {
                1.0
            },
            mass: if normalise { raw_mass } else { 1.0 },
        };
        let initial_mass = raw_mass / scales.mass;
        let thrust = if sail {
            0.0
        } else {
            required(raw.max_thrust)? / (scales.mass * scales.acceleration())
        };
        let constants = [
            if sail {
                required(raw.c_sail)? / scales.acceleration()
            } else {
                scales.velocity() / required(raw.veff)?
            },
            if cr3bp { required(raw.mu_cr3bp)? } else { 0.0 },
        ];
        let scale_state = |mut state: [f64; 6]| {
            if family == TopsFamily::Mee {
                state[0] /= scales.length;
                state
            } else {
                scales.cartesian(state)
            }
        };
        let layout = Layout::new(segments, sail, moving);
        let direction_inequalities = options.direction_inequalities.unwrap_or(moving || !cr3bp);
        let boundary = moving
            .then(|| BoundaryPair::new(family, raw, scales, options))
            .transpose()?;
        let mut bounds = Vec::with_capacity(layout.tof + 1 + segments);
        let mut decisions = Vec::with_capacity(bounds.capacity());
        let mut push = |label: String, bound| {
            decisions.push(label);
            bounds.push(bound);
        };
        if let Some(b) = &boundary {
            push("departure_offset".into(), [0.0, b.window()]);
        }
        if !sail {
            let min_mass = if moving && family == TopsFamily::Twobody {
                2.0 / 3.0
            } else {
                initial_mass * 0.5
            };
            push("final_mass".into(), [min_mass, initial_mass]);
        }
        if moving {
            for endpoint in ["departure", "arrival"] {
                push(format!("{endpoint}_vinf"), [0.0, 0.0]);
                for axis in ["x", "y", "z"] {
                    push(format!("{endpoint}_direction_{axis}"), [-1.0, 1.0]);
                }
            }
        }
        for i in 0..segments {
            if sail {
                push(format!("alpha_{i}"), [-FRAC_PI_2 + 1e-3, FRAC_PI_2 - 1e-3]);
                push(format!("beta_{i}"), [0.0, TAU]);
            } else {
                push(format!("throttle_{i}"), [0.0, 1.0]);
                for axis in ["x", "y", "z"] {
                    push(format!("direction_{i}_{axis}"), [-1.0, 1.0]);
                }
            }
        }
        push(
            "time_of_flight".into(),
            raw.tof_bounds.map(|v| v / scales.time),
        );
        if options.time_encoding == TimeEncoding::Softmax {
            for i in 0..segments {
                push(format!("logit_{i}"), [-1.0, 1.0]);
            }
        }
        let mut equalities: Vec<_> = if family == TopsFamily::Mee {
            ["p", "f", "g", "h", "k", "longitude"]
                .map(|s| format!("match_{s}"))
                .to_vec()
        } else {
            ["x", "y", "z", "vx", "vy", "vz"]
                .map(|s| format!("match_{s}"))
                .to_vec()
        };
        let mut direction_rows = Vec::new();
        if !sail {
            equalities.push("match_mass".into());
            direction_rows.extend((0..segments).map(|i| format!("direction_norm_{i}")));
        }
        if moving {
            direction_rows.extend([
                "departure_direction_norm".into(),
                "arrival_direction_norm".into(),
            ]);
        }
        let inequalities = if direction_inequalities {
            direction_rows
        } else {
            equalities.extend(direction_rows);
            Vec::new()
        };
        let formulation = if moving && cr3bp && options.time_encoding == TimeEncoding::Softmax {
            "tops-v1-cr3bp-moving-softmax"
        } else {
            "tops-v1"
        };
        Ok(Self {
            spec: ProblemSpec {
                id: format!(
                    "tops.{}.{id}{}",
                    family.id(),
                    if moving { ".moving" } else { "" }
                ),
                formulation,
                source: UPSTREAM,
                name: raw.info.clone(),
                decisions,
                bounds,
                objectives: vec![
                    if sail {
                        "flight_time [normalized]"
                    } else {
                        "negative_final_mass [normalized]"
                    }
                    .into(),
                ],
                equalities,
                inequalities,
                has_gradient: true,
                settings: json!({
                    "family": family, "moving": moving, "segments": segments, "cut": options.cut,
                    "time_encoding": options.time_encoding, "direction_inequalities": direction_inequalities,
                    "source_units_si": {"length": raw.length, "time": raw.time, "mass": raw.mass},
                    "integration_units_si": scales.physical(raw),
                    "backend": match options.method { IntegrationMethod::Taylor => "taylor", IntegrationMethod::Dop853 => "dop853" },
                    "tolerance": options.tolerance, "derivative_tolerance": options.derivative_tolerance,
                    "frame": if cr3bp { "rotating CR3BP" } else if family == TopsFamily::Mee {
                        "prograde MEE with unwrapped longitude"
                    } else { "inertial Cartesian" }
                }),
            },
            family,
            options,
            segments,
            layout,
            scales,
            initial: scale_state(raw.state_s),
            final_state: scale_state(raw.state_f),
            initial_mass,
            thrust,
            constants,
            boundary,
            direction_inequalities,
        })
    }

    /// Segment count.
    pub const fn segments(&self) -> usize {
        self.segments
    }

    /// Time-of-flight decision index.
    pub const fn flight_time_index(&self) -> usize {
        self.layout.tof
    }

    /// Independent forward terminal-state residual across the full control schedule.
    ///
    /// # Errors
    /// Returns an error if the controls or the tighter forward integration fail.
    pub fn forward_residual(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.apply(x, Operation::Forward)
            .map(|result| result.residual)
    }

    fn endpoints(&self, x: &[f64], derivatives: bool) -> Result<Endpoints> {
        match &self.boundary {
            Some(b) => b.evaluate(
                x,
                self.layout.directions.unwrap_or_default(),
                x[self.layout.tof],
                derivatives,
            ),
            None => Ok(Endpoints::fixed(self.initial, self.final_state)),
        }
    }

    fn apply(&self, x: &[f64], operation: Operation) -> Result<Output> {
        self.spec.check(x)?;
        if let Operation::Decode(samples) = operation
            && self.segments.saturating_mul(samples) > 500_000
        {
            return Err(invalid("history samples", "too many segment samples"));
        }
        match self.family {
            TopsFamily::Twobody => {
                self.run::<_, 7, 4, 1, 5, 11>(ZohKeplerDynamics, [self.constants[0]], x, operation)
            }
            TopsFamily::Mee => self.run::<_, 7, 4, 1, 5, 11>(
                ZohEquinoctialDynamics,
                [self.constants[0]],
                x,
                operation,
            ),
            TopsFamily::Cr3bp => {
                self.run::<_, 7, 4, 2, 6, 11>(ZohCr3bpDynamics, self.constants, x, operation)
            }
            TopsFamily::Sail => self.run::<_, 6, 2, 1, 3, 8>(
                ZohSolarSailDynamics,
                [self.constants[0]],
                x,
                operation,
            ),
        }
    }

    fn run<M, const N: usize, const C: usize, const K: usize, const P: usize, const W: usize>(
        &self,
        model: M,
        constants: [f64; K],
        x: &[f64],
        operation: Operation,
    ) -> Result<Output>
    where
        M: ZeroOrderHoldModel<N, C, K, P> + TaylorDynamicsModel<N, P> + Copy,
    {
        let layout = self.layout;
        let tof = x[layout.tof];
        let grid = Grid::new(self.segments, tof, &x[layout.tof + 1..])?;
        let endpoints = self.endpoints(x, matches!(operation, Operation::Gradient))?;
        let final_mass = layout.mass.map_or(1.0, |i| x[i]);
        let initial = core::array::from_fn(|i| {
            if i < 6 {
                endpoints.initial[i]
            } else {
                self.initial_mass
            }
        });
        let final_state = core::array::from_fn(|i| {
            if i < 6 {
                endpoints.final_state[i]
            } else {
                final_mass
            }
        });
        let controls = x[layout.controls..layout.tof]
            .chunks_exact(C)
            .map(|c| {
                core::array::from_fn(|i| {
                    if C == 4 && i == 0 {
                        self.thrust * c[i]
                    } else {
                        c[i]
                    }
                })
            })
            .collect();
        let tolerance = match operation {
            Operation::Gradient => self.options.derivative_tolerance,
            Operation::Forward => self.options.tolerance.min(5e-15),
            _ => self.options.tolerance,
        };
        let options = IntegratorOptions {
            relative_tolerance: tolerance,
            absolute_tolerance: tolerance,
            maximum_steps: 1000,
            ..IntegratorOptions::default()
        };
        let weights = grid.weights;
        let leg = ZohLeg::<M, N, C, K, P, W>::new(
            model,
            initial,
            controls,
            final_state,
            grid.nodes,
            constants,
            self.options.cut,
            options,
        )?;
        let method = self.options.method;
        let mut output = Output::default();
        match operation {
            Operation::Fitness => {
                let mut equalities = leg.mismatch_constraints_with_method(method)?.to_vec();
                let directions = self
                    .direction_indices()
                    .map(|i| x[i..i + 3].iter().map(|v| v * v).sum::<f64>() - 1.0);
                let inequalities = if self.direction_inequalities {
                    directions.collect()
                } else {
                    equalities.extend(directions);
                    Vec::new()
                };
                output.evaluation = Some(Evaluation {
                    objectives: vec![layout.mass.map_or(tof, |_| -final_mass)],
                    equalities,
                    inequalities,
                    work: Work {
                        segments: self.segments,
                        endpoints: usize::from(self.boundary.is_some()) * 2,
                        ..Work::default()
                    },
                });
            }
            Operation::Gradient => {
                let jac = leg.mismatch_jacobian_with_method(method)?;
                let mut rows = vec![
                    vec![0.0; x.len()];
                    1 + self.spec.equalities.len() + self.spec.inequalities.len()
                ];
                let (objective_col, objective_derivative) =
                    layout.mass.map_or((layout.tof, 1.0), |i| (i, -1.0));
                rows[0][objective_col] = objective_derivative;
                for (r, row) in rows[1..=N].iter_mut().enumerate() {
                    if let Some(mass) = layout.mass {
                        row[mass] = jac.final_state[r][6];
                    }
                    row[layout.controls..layout.tof].copy_from_slice(&jac.controls[r]);
                    if C == 4 {
                        for control in row[layout.controls..layout.tof].chunks_exact_mut(C) {
                            control[0] *= self.thrust;
                        }
                    }
                    row[layout.tof] = crate::time_grid::contract(
                        leg.schedule().boundaries(),
                        &weights,
                        tof,
                        &jac.time_grid[r],
                        &mut row[layout.tof + 1..],
                    );
                    endpoints.add_derivatives(
                        row,
                        &jac.initial_state[r],
                        &jac.final_state[r],
                        layout.tof,
                    );
                }
                for (row, i) in rows[1 + N..].iter_mut().zip(self.direction_indices()) {
                    for j in i..i + 3 {
                        row[j] = 2.0 * x[j];
                    }
                }
                output.gradient = rows;
            }
            Operation::Decode(samples) => {
                let mut decoded = json!({
                    "grid": leg.schedule().boundaries(), "controls": leg.schedule().controls().iter().map(|c| c.as_slice()).collect::<Vec<_>>(),
                    "initial": initial.as_slice(), "target": final_state.as_slice(), "constants": constants.as_slice(),
                    "departure_epoch_source": endpoints.departure_epoch,
                    "arrival_epoch_source": endpoints.arrival_epoch, "forward_segments": self.segments - leg.backward_segment_count(),
                    "source_to_integrator_scales": self.scales,
                });
                if samples > 0 {
                    let history = leg.state_history_with_method(samples, method)?;
                    let serialise = |segments: Vec<Vec<[f64; N]>>| {
                        segments
                            .into_iter()
                            .map(|segment| {
                                segment.into_iter().map(|s| s.to_vec()).collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    };
                    decoded["forward"] = json!(serialise(history.forward));
                    decoded["backward"] = json!(serialise(history.backward));
                }
                output.decoded = decoded;
            }
            Operation::Forward => {
                let result = propagate_schedule_with_method(
                    &model,
                    leg.schedule(),
                    initial,
                    constants,
                    options,
                    IntegrationMethod::Dop853,
                )?;
                output.residual = result
                    .state
                    .iter()
                    .zip(final_state)
                    .map(|(v, t)| v - t)
                    .collect();
            }
        }
        Ok(output)
    }

    fn direction_indices(&self) -> impl Iterator<Item = usize> + '_ {
        let n = if self.layout.width == 4 {
            self.segments
        } else {
            0
        };
        (0..n)
            .map(|i| self.layout.controls + 4 * i + 1)
            .chain(self.layout.directions.into_iter().flatten())
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Fitness,
    Gradient,
    Decode(usize),
    Forward,
}

#[derive(Default)]
struct Output {
    evaluation: Option<Evaluation>,
    gradient: Vec<Vec<f64>>,
    decoded: Value,
    residual: Vec<f64>,
}

impl Problem for TopsProblem {
    fn spec(&self) -> &ProblemSpec {
        &self.spec
    }

    fn evaluate(&self, x: &[f64]) -> Result<Evaluation> {
        self.apply(x, Operation::Fitness)?
            .evaluation
            .ok_or_else(|| invalid("evaluation", "missing result"))
    }

    fn gradient(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        Ok(self.apply(x, Operation::Gradient)?.gradient)
    }

    fn decode(&self, x: &[f64], samples_per_segment: usize) -> Result<Value> {
        Ok(self
            .apply(x, Operation::Decode(samples_per_segment))?
            .decoded)
    }
}
