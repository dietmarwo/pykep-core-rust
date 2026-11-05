// SPDX-License-Identifier: MPL-2.0
//! Native trajectory problems with explicit bounds, constraints and units.

mod boundary;
mod gym;
mod impulsive;
mod launcher;
pub use gym::{CLASSICAL_IDS, GymProblem};
pub use impulsive::{
    Arrival, FlightTimes, ImpulseOptions, ImpulsiveProblem, MgaDsmOptions, MgaOptions, MissionBody,
};
pub use launcher::Launcher;
mod problem;
mod tops;
mod tops_data;
pub use tops::{TopsOptions, TopsProblem};
pub use tops_data::TopsFamily;
mod time_grid;
pub use problem::{Evaluation, Problem, ProblemSpec, Work};
pub use time_grid::TimeEncoding;

/// PyKEP revision defining the catalog and comparison workloads.
pub const UPSTREAM: &str = "c01e19debad5ea22b7079189f786f473046a79d5";
