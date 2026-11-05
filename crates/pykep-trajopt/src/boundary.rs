// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
// Moving-boundary endpoints follow `pykep/trajopt/_zoh_pl2pl.py` and
// `_zoh_ss_pl2pl.py` at c01e19debad5ea22b7079189f786f473046a79d5.

use core::f64::consts::TAU;

use pykep_core::astro::elements::{
    ModifiedEquinoctialElements, cartesian_to_modified_equinoctial,
    cartesian_to_modified_equinoctial_jacobian, modified_equinoctial_to_cartesian,
};
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::dynamics::Cr3bpDynamics;
use pykep_core::integration::{
    AdaptiveIntegrator, DynamicsModel, InitialValueProblem, IntegratorOptions,
};
use pykep_core::{PykepError, Result};

use crate::tops::{Scales, TopsOptions};
use crate::tops_data::{RawCase, TopsFamily, required};

#[derive(Debug, Clone)]
pub(crate) struct BoundaryPair {
    family: TopsFamily,
    scales: Scales,
    options: TopsOptions,
    state: [[f64; 6]; 2],
    reference_arrival: f64,
    half_window: f64,
    mu: f64,
    revs: f64,
    maximum_steps: usize,
}

impl BoundaryPair {
    pub fn new(
        family: TopsFamily,
        raw: &RawCase,
        scales: Scales,
        options: TopsOptions,
        maximum_steps: usize,
    ) -> Result<Self> {
        let mut state = [raw.state_s, raw.state_f];
        let mut revs = 0.0;
        let mu = match family {
            TopsFamily::Cr3bp => required(raw.mu_cr3bp)?,
            TopsFamily::Mee => {
                let mu = required(raw.mu)?;
                revs = ((raw.state_f[5] - raw.state_s[5]) / TAU).floor();
                for y in &mut state {
                    *y = modified_equinoctial_to_cartesian(
                        ModifiedEquinoctialElements::new(y[0], y[1], y[2], y[3], y[4], y[5]),
                        mu,
                        false,
                    )?;
                }
                mu
            }
            _ => {
                // The two-body and sail endpoint arcs are propagated in the
                // source's normalized units, which the upstream data fix at mu = 1.
                if required(raw.mu)? != 1.0 {
                    return Err(PykepError::InvalidInput {
                        parameter: "TOPS endpoint mu",
                        reason: "moving two-body and sail endpoints require mu = 1".into(),
                    });
                }
                1.0
            }
        };
        // The upstream MEE moving constructor fixes the departure epoch.
        let half_window = if family == TopsFamily::Mee {
            0.0
        } else {
            raw.period_s / 4.0
        };
        Ok(Self {
            family,
            scales,
            options,
            state,
            reference_arrival: raw.tof_bounds[0],
            half_window,
            mu,
            revs,
            maximum_steps,
        })
    }

    pub fn window(&self) -> f64 {
        2.0 * self.half_window / self.scales.time
    }

    pub fn evaluate(
        &self,
        x: &[f64],
        directions: [usize; 2],
        tof: f64,
        gradient: bool,
    ) -> Result<Endpoints> {
        let t0 = x[0] * self.scales.time - self.half_window;
        let t1 = t0 + tof * self.scales.time;
        let (initial, first) =
            self.endpoint(0, t0, &x[directions[0] - 1..directions[0] + 3], gradient)?;
        let (mut final_state, last) =
            self.endpoint(1, t1, &x[directions[1] - 1..directions[1] + 3], gradient)?;
        if self.family == TopsFamily::Mee {
            if final_state[5] < initial[5] {
                final_state[5] += TAU;
            }
            final_state[5] += self.revs * TAU;
        }
        Ok(Endpoints {
            initial,
            final_state,
            departure_epoch: t0,
            arrival_epoch: t1,
            jacobian: gradient.then_some(BoundaryJacobian {
                first,
                last,
                directions,
            }),
        })
    }

    fn endpoint(
        &self,
        index: usize,
        time: f64,
        offset: &[f64],
        gradient: bool,
    ) -> Result<([f64; 6], [[f64; 6]; 5])> {
        let anchor = if index == 0 {
            0.0
        } else {
            self.reference_arrival
        };
        let cr3bp = self.family == TopsFamily::Cr3bp;
        let state = if cr3bp {
            let tolerance = self.options.tolerance;
            AdaptiveIntegrator::new(self.options.method)
                .propagate(
                    &Cr3bpDynamics,
                    InitialValueProblem::new(anchor, self.state[index], time, [self.mu]),
                    IntegratorOptions {
                        relative_tolerance: tolerance,
                        absolute_tolerance: tolerance,
                        maximum_steps: self.maximum_steps,
                        ..IntegratorOptions::default()
                    },
                )?
                .state
        } else {
            propagate_lagrangian(&self.state[index], time - anchor, self.mu)?
        };
        let mut transformed = self.scales.cartesian(state);
        for (i, v) in transformed[3..].iter_mut().enumerate() {
            *v += offset[0] * offset[i + 1];
        }
        let mut jac = [[0.0; 6]; 5];
        if gradient {
            let rhs = if cr3bp {
                let mut rhs = [0.0; 6];
                Cr3bpDynamics.rhs(time, &state, &[self.mu], &mut rhs)?;
                rhs
            } else {
                let r3 = state[..3].iter().map(|v| v * v).sum::<f64>().powf(1.5);
                [
                    state[3],
                    state[4],
                    state[5],
                    -self.mu * state[0] / r3,
                    -self.mu * state[1] / r3,
                    -self.mu * state[2] / r3,
                ]
            };
            jac[0] = self.scales.cartesian(rhs).map(|v| v * self.scales.time);
            jac[1][3..].copy_from_slice(&offset[1..]);
            for i in 0..3 {
                jac[i + 2][i + 3] = offset[0];
            }
        }
        if self.family == TopsFamily::Mee {
            if gradient {
                let map = cartesian_to_modified_equinoctial_jacobian(&transformed, 1.0, false)?;
                for column in &mut jac {
                    *column = core::array::from_fn(|row| dot(&map[row], column));
                }
            }
            transformed = cartesian_to_modified_equinoctial(&transformed, 1.0, false)?.to_array();
        }
        Ok((transformed, jac))
    }
}

fn dot(row: &[f64], col: &[f64; 6]) -> f64 {
    row.iter().zip(col).map(|(a, b)| a * b).sum()
}

struct BoundaryJacobian {
    first: [[f64; 6]; 5],
    last: [[f64; 6]; 5],
    directions: [usize; 2],
}

pub(crate) struct Endpoints {
    pub initial: [f64; 6],
    pub final_state: [f64; 6],
    pub departure_epoch: f64,
    pub arrival_epoch: f64,
    jacobian: Option<BoundaryJacobian>,
}

impl Endpoints {
    pub fn fixed(initial: [f64; 6], final_state: [f64; 6]) -> Self {
        Self {
            initial,
            final_state,
            departure_epoch: 0.0,
            arrival_epoch: 0.0,
            jacobian: None,
        }
    }

    pub fn add_derivatives(&self, row: &mut [f64], d_initial: &[f64], d_final: &[f64], tof: usize) {
        if let Some(jac) = &self.jacobian {
            let arrival = dot(d_final, &jac.last[0]);
            row[0] = dot(d_initial, &jac.first[0]) + arrival;
            row[tof] += arrival;
            for i in 0..4 {
                row[jac.directions[0] - 1 + i] = dot(d_initial, &jac.first[i + 1]);
                row[jac.directions[1] - 1 + i] = dot(d_final, &jac.last[i + 1]);
            }
        }
    }
}
