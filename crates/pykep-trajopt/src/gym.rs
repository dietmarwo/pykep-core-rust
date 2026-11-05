// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
// The problem catalog mirrors upstream `pykep/trajopt/gym/` at
// c01e19debad5ea22b7079189f786f473046a79d5.

use std::sync::Arc;

use pykep_core::astro::elements::ClassicalElements;
use pykep_core::constants::{ASTRONOMICAL_UNIT, MU_SUN};
use pykep_core::ephemeris::KeplerianEphemeris;
use pykep_core::time::epoch::Epoch;
use pykep_core::{PykepError, Result};

use crate::impulsive::{
    Arrival, FlightTimes, ImpulseOptions, ImpulsiveProblem, MgaDsmOptions, MgaOptions, MissionBody,
};
use crate::problem::invalid;
use crate::{Evaluation, Problem, ProblemSpec, TopsFamily, TopsOptions, TopsProblem};

/// Named current PyKEP gym problems.
pub const CLASSICAL_IDS: [&str; 14] = [
    "cassini1",
    "cassini1_a",
    "cassini1_n",
    "cassini2",
    "eve_mga1dsm",
    "eve_mga1dsm_a",
    "eve_mga1dsm_n",
    "rosetta",
    "messenger",
    "em3imp",
    "em5imp",
    "em7imp",
    "juice",
    "juice_mo",
];

/// A named catalog instance, built lazily.
#[derive(Clone)]
pub enum GymProblem {
    /// TOPS fixed or moving boundary nonlinear program.
    Tops(Box<TopsProblem>),
    /// Multiple gravity assists or impulses.
    Impulsive(Box<ImpulsiveProblem>),
}

impl GymProblem {
    /// Build an instance from a family-qualified ID.
    ///
    /// # Errors
    /// Returns an error for unknown IDs or invalid TOPS configuration.
    pub fn new(id: &str, options: TopsOptions) -> Result<Self> {
        if let Some(rest) = id.strip_prefix("tops.") {
            let (family, name) = rest.split_once('.').ok_or_else(|| invalid("gym ID", id))?;
            let family = family.parse::<TopsFamily>()?;
            return if let Some(name) = name.strip_suffix(".moving") {
                TopsProblem::moving(family, name, options)
            } else {
                TopsProblem::fixed(family, name, options)
            }
            .map(|p| Self::Tops(Box::new(p)));
        }
        classical(id.strip_prefix("gym.").unwrap_or(id)).map(|p| Self::Impulsive(Box::new(p)))
    }

    /// All problem IDs, without constructing an integrator or any trajectory.
    ///
    /// # Errors
    /// Returns an error if embedded TOPS data are invalid.
    pub fn ids() -> Result<Vec<String>> {
        let mut ids: Vec<_> = CLASSICAL_IDS
            .iter()
            .map(|name| format!("gym.{name}"))
            .collect();
        for family in TopsFamily::ALL {
            for case in family.ids()? {
                ids.push(format!("tops.{}.{case}", family.id()));
                ids.push(format!("tops.{}.{case}.moving", family.id()));
            }
        }
        Ok(ids)
    }

    /// Check the full forward TOPS trajectory with tighter DOP853 integration.
    ///
    /// # Errors
    /// Returns an unsupported capability for an impulsive problem, or an integration error.
    pub fn forward_residual(&self, x: &[f64]) -> Result<Vec<f64>> {
        match self {
            Self::Tops(p) => p.forward_residual(x),
            _ => Err(PykepError::UnsupportedCapability {
                provider: self.spec().id.clone(),
                capability: "TOPS forward residual",
            }),
        }
    }

    fn problem(&self) -> &dyn Problem {
        match self {
            Self::Tops(p) => p.as_ref(),
            Self::Impulsive(p) => p.as_ref(),
        }
    }
}

impl Problem for GymProblem {
    fn spec(&self) -> &ProblemSpec {
        self.problem().spec()
    }
    fn evaluate(&self, x: &[f64]) -> Result<Evaluation> {
        self.problem().evaluate(x)
    }
    fn gradient(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.problem().gradient(x)
    }
    fn decode(&self, x: &[f64], samples: usize) -> Result<serde_json::Value> {
        self.problem().decode(x, samples)
    }
}

fn bodies(names: &[&str], overrides: &[(&str, f64)]) -> Result<Vec<MissionBody>> {
    names
        .iter()
        .map(|name| {
            MissionBody::planet(
                name,
                overrides
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, ratio)| *ratio),
            )
        })
        .collect()
}

fn classical(name: &str) -> Result<ImpulsiveProblem> {
    let id = format!("gym.{name}");
    let cassini = ["earth", "venus", "venus", "earth", "jupiter", "saturn"];
    let saturn = Arrival::Insertion {
        periapsis: 108_950_000.0,
        eccentricity: 0.98,
    };
    let default = MgaDsmOptions {
        departure: [0.0, 3000.0],
        times: FlightTimes::Direct(vec![[10.0, 500.0]; 2]),
        launch_speed: [500.0, 2500.0],
        charge_departure: false,
        arrival: Arrival::Rendezvous,
        dsm_fraction: [0.01, 0.99],
        maximum_periapsis_ratio: 30.0,
        multi_objective: false,
        juice_mass: false,
    };
    match name {
        "cassini1" | "cassini1_a" | "cassini1_n" => {
            let times = match name {
                "cassini1_a" => FlightTimes::Alpha([4000., 7000.]),
                "cassini1_n" => FlightTimes::Eta(7000.),
                _ => FlightTimes::Direct(vec![
                    [30., 400.],
                    [100., 470.],
                    [30., 400.],
                    [400., 2000.],
                    [1000., 6000.],
                ]),
            };
            ImpulsiveProblem::mga(
                &id,
                bodies(&cassini, &[])?,
                MgaOptions {
                    departure: [-1000., 0.],
                    times,
                    free_launch_speed: 3000.,
                    arrival: saturn,
                    multi_objective: false,
                },
            )
        }
        "eve_mga1dsm" | "eve_mga1dsm_a" | "eve_mga1dsm_n" => {
            let times = match name {
                "eve_mga1dsm_a" => FlightTimes::Alpha([300., 700.]),
                "eve_mga1dsm_n" => FlightTimes::Eta(700.),
                _ => default.times.clone(),
            };
            ImpulsiveProblem::mga_1dsm(
                &id,
                bodies(&["earth", "venus", "earth"], &[])?,
                MgaDsmOptions {
                    times,
                    dsm_fraction: [0.1, 0.9],
                    ..default
                },
            )
        }
        "cassini2" => ImpulsiveProblem::mga_1dsm(
            &id,
            bodies(
                &cassini,
                &[("earth", 1.15), ("venus", 1.05), ("jupiter", 1.7)],
            )?,
            MgaDsmOptions {
                departure: [-1000., 0.],
                times: FlightTimes::Direct(vec![
                    [100., 400.],
                    [100., 500.],
                    [30., 300.],
                    [400., 1600.],
                    [800., 2200.],
                ]),
                launch_speed: [3000., 5000.],
                arrival: saturn,
                dsm_fraction: [0.01, 0.9],
                maximum_periapsis_ratio: 70.,
                ..default
            },
        ),
        "rosetta" => {
            let mut sequence = bodies(
                &["earth", "earth", "mars", "earth", "earth"],
                &[("mars", 1.05)],
            )?;
            let comet = KeplerianEphemeris::from_classical_mean(
                Epoch::from_mjd(52_504.237_540_000_12)?,
                ClassicalElements::new(
                    3.502_949_728_362_75 * ASTRONOMICAL_UNIT,
                    0.631_935_6,
                    7.12723_f64.to_radians(),
                    50.92302_f64.to_radians(),
                    11.36788_f64.to_radians(),
                    0.0,
                ),
                MU_SUN,
                "Churyumov-Gerasimenko",
                None,
                None,
                None,
            )?;
            sequence.push(MissionBody::new(Arc::new(comet))?);
            ImpulsiveProblem::mga_1dsm(
                &id,
                sequence,
                MgaDsmOptions {
                    departure: [1460., 1825.],
                    times: FlightTimes::Direct(vec![
                        [300., 500.],
                        [150., 800.],
                        [150., 800.],
                        [300., 800.],
                        [700., 1850.],
                    ]),
                    launch_speed: [3000., 5000.],
                    dsm_fraction: [0.01, 0.9],
                    maximum_periapsis_ratio: 9.,
                    ..default
                },
            )
        }
        "messenger" => {
            let mut times = vec![[100., 500.]; 6];
            times[5][1] = 600.;
            ImpulsiveProblem::mga_1dsm(
                &id,
                bodies(
                    &[
                        "earth", "venus", "venus", "mercury", "mercury", "mercury", "mercury",
                    ],
                    &[("venus", 1.1), ("mercury", 1.05)],
                )?,
                MgaDsmOptions {
                    departure: [1900., 2300.],
                    times: FlightTimes::Direct(times),
                    launch_speed: [2500., 4050.],
                    arrival: Arrival::Insertion {
                        periapsis: 2_640_000.,
                        eccentricity: 0.704,
                    },
                    maximum_periapsis_ratio: 6.,
                    ..default
                },
            )
        }
        "em3imp" | "em5imp" | "em7imp" => {
            let impulses = match name {
                "em3imp" => 3,
                "em5imp" => 5,
                _ => 7,
            };
            ImpulsiveProblem::impulses(
                &id,
                [
                    MissionBody::planet("earth", None)?,
                    MissionBody::planet("mars", None)?,
                ],
                ImpulseOptions {
                    impulses,
                    departure: Some([10000., 11000.]),
                    flight_time: [200., 700.],
                    magnitude: [0., 4000.],
                    multi_objective: false,
                },
            )
        }
        "juice" | "juice_mo" => {
            let multi_objective = name == "juice_mo";
            let times = if multi_objective {
                FlightTimes::Alpha([2000., 3000.])
            } else {
                FlightTimes::Direct(vec![
                    [200., 500.],
                    [30., 300.],
                    [200., 500.],
                    [30., 300.],
                    [500., 800.],
                    [900., 1200.],
                ])
            };
            ImpulsiveProblem::mga_1dsm(
                &id,
                bodies(
                    &[
                        "earth", "earth", "venus", "earth", "mars", "earth", "jupiter",
                    ],
                    &[("earth", 1.05), ("venus", 1.05), ("mars", 1.05)],
                )?,
                MgaDsmOptions {
                    departure: [8000., 8400.],
                    times,
                    launch_speed: [1000., 4000.],
                    arrival: Arrival::Insertion {
                        periapsis: 1_070_400_000.,
                        eccentricity: 0.985_314_079_963_58,
                    },
                    maximum_periapsis_ratio: 10.,
                    multi_objective,
                    juice_mass: true,
                    ..default
                },
            )
        }
        _ => Err(invalid("gym ID", name)),
    }
}
