// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
// Transcriptions follow the pinned ESA PyKEP trajectory definitions.

use core::f64::consts::{PI, TAU};
use std::sync::Arc;

use pykep_core::astro::encodings::{alpha_to_direct, eta_to_direct};
use pykep_core::astro::flyby::{flyby_delta_v, flyby_outgoing_velocity};
use pykep_core::astro::lambert::{LambertProblem, LambertSolution};
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::constants::DAY_TO_SECONDS;
use pykep_core::ephemeris::{Ephemeris, EphemerisMetadata, JplLowPrecision};
use pykep_core::{CartesianState, Result, Vector3};
use serde::Serialize;
use serde_json::{Value, json};

use crate::launcher::Launcher;
use crate::problem::invalid;
use crate::{Evaluation, Problem, ProblemSpec, UPSTREAM, Work};

/// Ephemeris and physical metadata for a mission encounter.
#[derive(Clone)]
pub struct MissionBody {
    provider: Arc<dyn Ephemeris>,
    metadata: EphemerisMetadata,
}

impl MissionBody {
    /// Wrap an ephemeris with a central gravitational parameter.
    ///
    /// # Errors
    /// Returns an error when the provider lacks a positive central parameter.
    pub fn new(provider: Arc<dyn Ephemeris>) -> Result<Self> {
        let metadata = provider.metadata();
        positive("central mu", metadata.central_mu)?;
        Ok(Self { provider, metadata })
    }

    /// Construct a JPL low-precision planet, optionally overriding safe radius as a ratio.
    ///
    /// # Errors
    /// Returns an error for an unknown planet or invalid safe-radius ratio.
    pub fn planet(name: &str, safe_radius_ratio: Option<f64>) -> Result<Self> {
        let mut planet = JplLowPrecision::new(name)?;
        if let Some(ratio) = safe_radius_ratio {
            planet.set_safe_radius(ratio * positive("radius", planet.metadata().radius)?)?;
        }
        Self::new(Arc::new(planet))
    }
}

/// Encoding and bounds for inter-encounter flight durations, in days.
#[derive(Debug, Clone, Serialize)]
pub enum FlightTimes {
    /// One duration per leg.
    Direct(Vec<[f64; 2]>),
    /// Total-duration bounds and one logarithmic weight per leg.
    Alpha([f64; 2]),
    /// Maximum mission duration and one fraction of remaining time per leg.
    Eta(f64),
}

impl FlightTimes {
    fn validate(&self, n: usize) -> Result<()> {
        match self {
            Self::Direct(bounds) => {
                if bounds.len() != n {
                    return Err(invalid("flight time", "expected one bound per leg"));
                }
                for bound in bounds {
                    positive_bounds("flight time", *bound)?;
                }
            }
            Self::Alpha(total) => positive_bounds("total flight time", *total)?,
            Self::Eta(maximum) => {
                positive("maximum flight time", Some(*maximum))?;
            }
        }
        Ok(())
    }

    fn bounds(&self, i: usize) -> [f64; 2] {
        match self {
            Self::Direct(bounds) => bounds[i],
            _ => [1e-3, 1.0 - 1e-3],
        }
    }

    fn decode(&self, encoded: &[f64], total: f64) -> Result<Vec<f64>> {
        let times = match self {
            Self::Direct(_) => encoded.to_vec(),
            Self::Alpha(_) => alpha_to_direct(encoded, total)?,
            Self::Eta(maximum) => eta_to_direct(encoded, *maximum)?,
        };
        if times.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return Err(invalid(
                "flight time",
                "requires positive finite leg durations",
            ));
        }
        Ok(times)
    }
}

/// Treatment of terminal hyperbolic excess speed.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum Arrival {
    /// Match the target's velocity.
    Rendezvous,
    /// Exclude arrival cost.
    Ignore,
    /// Burn at the specified target periapsis.
    Insertion {
        /// Target periapsis in metres.
        periapsis: f64,
        /// Target eccentricity, less than one.
        eccentricity: f64,
    },
}

impl Arrival {
    fn validate(self, target: &MissionBody) -> Result<()> {
        if let Self::Insertion {
            periapsis,
            eccentricity,
        } = self
        {
            positive("insertion periapsis", Some(periapsis))?;
            positive("target mu", target.metadata.body_mu)?;
            if !(0.0..1.0).contains(&eccentricity) {
                return Err(invalid("target eccentricity", "expected 0 <= e < 1"));
            }
        }
        Ok(())
    }

    fn cost(self, vinf: f64, target: &MissionBody) -> Result<f64> {
        match self {
            Self::Ignore => Ok(0.0),
            Self::Rendezvous => Ok(vinf),
            Self::Insertion {
                periapsis,
                eccentricity,
            } => {
                let scale = positive("target mu", target.metadata.body_mu)? / periapsis;
                Ok(
                    ((vinf * vinf + 2.0 * scale).sqrt() - ((1.0 + eccentricity) * scale).sqrt())
                        .abs(),
                )
            }
        }
    }
}

/// Options for the multiple-gravity-assist formulation without DSMs.
#[derive(Debug, Clone, Serialize)]
pub struct MgaOptions {
    /// Departure epoch bounds in MJD2000 days.
    pub departure: [f64; 2],
    /// Flight-duration encoding.
    pub times: FlightTimes,
    /// Launch excess velocity allowance in m/s.
    pub free_launch_speed: f64,
    /// Final maneuver convention.
    pub arrival: Arrival,
    /// Also minimize total flight time in days.
    pub multi_objective: bool,
}

/// Options for one deep-space maneuver per leg.
#[derive(Debug, Clone, Serialize)]
pub struct MgaDsmOptions {
    /// Departure epoch bounds in MJD2000 days.
    pub departure: [f64; 2],
    /// Flight-duration encoding.
    pub times: FlightTimes,
    /// Launch excess speed bounds in m/s.
    pub launch_speed: [f64; 2],
    /// Include launch excess speed in the objective.
    pub charge_departure: bool,
    /// Final maneuver convention.
    pub arrival: Arrival,
    /// Lower and upper DSM coast fractions, strictly between zero and one.
    pub dsm_fraction: [f64; 2],
    /// Upper flyby periapsis in units of planet radius.
    pub maximum_periapsis_ratio: f64,
    /// Also minimize total flight time in days.
    pub multi_objective: bool,
    /// Use the Ariane 5 launch mass and PyKEP JUICE mass objective.
    pub juice_mass: bool,
}

/// Options for a transfer with a fixed maximum number of impulses.
#[derive(Debug, Clone, Serialize)]
pub struct ImpulseOptions {
    /// Number of impulses including the two Lambert endpoint burns.
    pub impulses: usize,
    /// Departure epoch bounds in MJD2000, or independent free phases.
    pub departure: Option<[f64; 2]>,
    /// Total flight time bounds in days.
    pub flight_time: [f64; 2],
    /// Interior impulse magnitude bounds in m/s.
    pub magnitude: [f64; 2],
    /// Also minimize total flight time.
    pub multi_objective: bool,
}

#[derive(Clone)]
enum Formulation {
    Mga(MgaOptions),
    Dsm(MgaDsmOptions),
    Impulses(ImpulseOptions),
}

/// A reusable impulsive trajectory problem.
#[derive(Clone)]
pub struct ImpulsiveProblem {
    spec: ProblemSpec,
    bodies: Vec<MissionBody>,
    mu: f64,
    formulation: Formulation,
}

#[derive(Serialize)]
struct ArcRecord {
    start: CartesianState,
    end: CartesianState,
    duration_seconds: f64,
    clockwise: Option<bool>,
}

#[derive(Default, Serialize)]
struct Trace {
    epochs_mjd2000: Vec<f64>,
    body_states: Vec<CartesianState>,
    arcs: Vec<ArcRecord>,
    delta_v_mps: Vec<f64>,
    impulse_vectors_mps: Vec<Vector3>,
    launch_excess_mps: Option<Vector3>,
    flyby_periapsis_m: Vec<f64>,
    flyby_beta_rad: Vec<f64>,
    launch_declination_deg: Option<f64>,
    launch_mass_kg: Option<f64>,
    final_mass_kg: Option<f64>,
    total_days: f64,
}

type Decisions = Vec<(String, [f64; 2])>;

impl ImpulsiveProblem {
    /// Build an MGA problem with powered flyby costs and zero-revolution Lambert arcs.
    ///
    /// # Errors
    /// Returns an error for inconsistent bodies, geometry settings or bounds.
    pub fn mga(id: &str, bodies: Vec<MissionBody>, options: MgaOptions) -> Result<Self> {
        let n = sequence(&bodies, options.arrival)?;
        options.times.validate(n)?;
        nonnegative("launch allowance", options.free_launch_speed)?;
        let mut columns = vec![("departure_epoch [MJD2000]".into(), options.departure)];
        if let FlightTimes::Alpha(bounds) = options.times {
            columns.push(("total_days".into(), bounds));
        }
        for i in 0..n {
            columns.push((format!("flight_time_encoded_{i}"), options.times.bounds(i)));
        }
        Self::new(id, bodies, columns, Formulation::Mga(options))
    }

    /// Build an MGA problem with one DSM and a zero-revolution Lambert arc per leg.
    ///
    /// # Errors
    /// Returns an error for invalid bounds or incompatible body metadata.
    pub fn mga_1dsm(id: &str, bodies: Vec<MissionBody>, options: MgaDsmOptions) -> Result<Self> {
        let n = sequence(&bodies, options.arrival)?;
        options.times.validate(n)?;
        positive_bounds("launch speed", options.launch_speed)?;
        positive_bounds("DSM fraction", options.dsm_fraction)?;
        if options.dsm_fraction[1] >= 1.0 {
            return Err(invalid(
                "DSM fraction",
                "upper fraction must be less than one",
            ));
        }
        let mut columns = vec![("departure_epoch [MJD2000]".into(), options.departure)];
        columns.extend([
            ("launch_u".into(), [0.0, 1.0]),
            ("launch_v".into(), [0.0, 1.0]),
            ("launch_speed [m/s]".into(), options.launch_speed),
        ]);
        for (i, body) in bodies.iter().take(n).enumerate() {
            if i > 0 {
                columns.push((format!("flyby_beta_{i}"), [-TAU, TAU]));
                let body = &body.metadata;
                let low = positive("safe radius", body.safe_radius)?
                    / positive("body radius", body.radius)?;
                columns.push((
                    format!("flyby_radius_ratio_{i}"),
                    [low, options.maximum_periapsis_ratio],
                ));
            }
            columns.push((format!("dsm_fraction_{i}"), options.dsm_fraction));
            columns.push((format!("flight_time_encoded_{i}"), options.times.bounds(i)));
        }
        if let FlightTimes::Alpha(bounds) = options.times {
            columns.push(("total_days".into(), bounds));
        }
        Self::new(id, bodies, columns, Formulation::Dsm(options))
    }

    /// Build a multiple-impulse transfer, with the last two burns set by Lambert closure.
    ///
    /// # Errors
    /// Returns an error for fewer than three impulses, invalid bounds or missing ephemeris data.
    pub fn impulses(id: &str, bodies: [MissionBody; 2], options: ImpulseOptions) -> Result<Self> {
        if !(3..=128).contains(&options.impulses) {
            return Err(invalid("impulses", "expected 3..=128"));
        }
        sequence(&bodies, Arrival::Rendezvous)?;
        positive_bounds("flight time", options.flight_time)?;
        nonnegative("impulse magnitude", options.magnitude[0])?;
        let phase_bounds = |body: &MissionBody| -> Result<[f64; 2]> {
            let period = positive("orbital period", body.provider.period(0.0)?)?;
            Ok([0.0, 2.0 * period / DAY_TO_SECONDS])
        };
        let departure = match options.departure {
            Some(bounds) => bounds,
            None => phase_bounds(&bodies[0])?,
        };
        let mut columns = vec![
            ("departure_epoch [MJD2000]".into(), departure),
            ("total_days".into(), options.flight_time),
        ];
        for i in 0..options.impulses - 1 {
            columns.push((format!("time_alpha_{i}"), [1e-3, 1.0 - 1e-3]));
            if i < options.impulses - 2 {
                columns.extend([
                    (format!("impulse_u_{i}"), [0.0, 1.0]),
                    (format!("impulse_v_{i}"), [0.0, 1.0]),
                    (format!("impulse_speed_{i} [m/s]"), options.magnitude),
                ]);
            }
        }
        if options.departure.is_none() {
            columns.push((
                "arrival_ephemeris_epoch [MJD2000]".into(),
                phase_bounds(&bodies[1])?,
            ));
        }
        Self::new(
            id,
            Vec::from(bodies),
            columns,
            Formulation::Impulses(options),
        )
    }

    fn new(
        id: &str,
        bodies: Vec<MissionBody>,
        columns: Decisions,
        formulation: Formulation,
    ) -> Result<Self> {
        let mu = positive("central mu", bodies[0].metadata.central_mu)?;
        let (decisions, bounds): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
        if bounds
            .iter()
            .any(|b| !b.iter().all(|v| v.is_finite()) || b[0] > b[1])
        {
            return Err(invalid("decision bounds", "requires finite lower <= upper"));
        }
        let (multi, mass, variant, options) = match &formulation {
            Formulation::Mga(x) => (x.multi_objective, false, "mga-v1", json!(x)),
            Formulation::Dsm(x) => (x.multi_objective, x.juice_mass, "mga-1dsm-v1", json!(x)),
            Formulation::Impulses(x) => (x.multi_objective, false, "impulses-v1", json!(x)),
        };
        let mut objectives = vec![
            if mass {
                "negative_log_final_mass"
            } else {
                "delta_v [m/s]"
            }
            .into(),
        ];
        if multi {
            objectives.push("flight_time [day]".into());
        }
        Ok(Self {
            spec: ProblemSpec {
                id: id.into(),
                formulation: variant,
                source: UPSTREAM,
                name: id.into(),
                decisions,
                bounds,
                objectives,
                equalities: Vec::new(),
                inequalities: Vec::new(),
                has_gradient: false,
                settings: json!({"sequence": bodies.iter().map(|b| b.provider.name()).collect::<Vec<_>>(),
                    "central_mu_si": mu, "frame": "heliocentric inertial", "options": options}),
            },
            bodies,
            mu,
            formulation,
        })
    }

    fn run(&self, x: &[f64], record: bool) -> Result<(Evaluation, Trace)> {
        self.spec.check(x)?;
        let mut trace = Trace::default();
        let (cost, work) = match &self.formulation {
            Formulation::Mga(opt) => self.run_mga(x, opt, record, &mut trace)?,
            Formulation::Dsm(opt) => self.run_dsm(x, opt, record, &mut trace)?,
            Formulation::Impulses(opt) => self.run_impulses(x, opt, record, &mut trace)?,
        };
        if !cost.is_finite() {
            return Err(invalid("mission objective", "numerical overflow"));
        }
        let mut objectives = vec![cost];
        if self.spec.objectives.len() > 1 {
            objectives.push(trace.total_days);
        }
        Ok((
            Evaluation {
                objectives,
                equalities: Vec::new(),
                inequalities: Vec::new(),
                work,
            },
            trace,
        ))
    }

    fn encounters(&self, first: f64, times: &[f64], trace: &mut Trace) -> Result<()> {
        let mut epoch = first;
        trace.epochs_mjd2000.push(epoch);
        for time in times {
            epoch += time;
            trace.epochs_mjd2000.push(epoch);
        }
        trace.total_days = times.iter().sum();
        for (body, &epoch) in self.bodies.iter().zip(&trace.epochs_mjd2000) {
            trace.body_states.push(body.provider.state(epoch)?);
        }
        Ok(())
    }

    fn run_mga(
        &self,
        x: &[f64],
        opt: &MgaOptions,
        record: bool,
        trace: &mut Trace,
    ) -> Result<(f64, Work)> {
        let (encoded, total) = if matches!(opt.times, FlightTimes::Alpha(_)) {
            (&x[2..], x[1])
        } else {
            (&x[1..], 0.0)
        };
        let times = opt.times.decode(encoded, total)?;
        self.encounters(x[0], &times, trace)?;
        let mut previous = [0.0; 3];
        for (i, &days) in times.iter().enumerate() {
            let a = trace.body_states[i];
            let b = trace.body_states[i + 1];
            let seconds = days * DAY_TO_SECONDS;
            let lambert = transfer(position(a), position(b), seconds, self.mu, false)?;
            let departing = sub(lambert.departure_velocity, velocity(a));
            let cost = if i == 0 {
                trace.launch_excess_mps = Some(departing);
                (norm(departing) - opt.free_launch_speed).max(0.0)
            } else {
                let body = &self.bodies[i].metadata;
                flyby_delta_v(
                    &sub(previous, velocity(a)),
                    &departing,
                    positive("body mu", body.body_mu)?,
                    positive("safe radius", body.safe_radius)?,
                )?
            };
            trace.delta_v_mps.push(cost);
            if record {
                trace.arcs.push(lambert_arc(
                    position(a),
                    position(b),
                    seconds,
                    &lambert,
                    false,
                ));
            }
            previous = lambert.arrival_velocity;
        }
        let target = &self.bodies[times.len()];
        let speed = norm(sub(previous, velocity(trace.body_states[times.len()])));
        trace.delta_v_mps.push(opt.arrival.cost(speed, target)?);
        Ok((
            trace.delta_v_mps.iter().sum(),
            Work {
                lambert: times.len(),
                ..Work::default()
            },
        ))
    }

    fn run_dsm(
        &self,
        x: &[f64],
        opt: &MgaDsmOptions,
        record: bool,
        trace: &mut Trace,
    ) -> Result<(f64, Work)> {
        let n = self.bodies.len() - 1;
        let encoded: Vec<_> = (0..n).map(|i| x[5 + 4 * i]).collect();
        let times = opt.times.decode(&encoded, x[x.len() - 1])?;
        self.encounters(x[0], &times, trace)?;
        let launch = uvv(x[1], x[2], x[3])?;
        trace.launch_excess_mps = Some(launch);
        let mut previous = [0.0; 3];
        for (i, &days) in times.iter().enumerate() {
            let a = trace.body_states[i];
            let b = trace.body_states[i + 1];
            let departure = if i == 0 {
                add(velocity(a), launch)
            } else {
                let body = &self.bodies[i].metadata;
                let rp = x[3 + 4 * i] * positive("body radius", body.radius)?;
                let beta = x[2 + 4 * i];
                if record {
                    trace.flyby_periapsis_m.push(rp);
                    trace.flyby_beta_rad.push(beta);
                }
                flyby_outgoing_velocity(
                    &previous,
                    &velocity(a),
                    rp,
                    beta,
                    positive("body mu", body.body_mu)?,
                )?
            };
            let fraction = x[4 + 4 * i];
            if !(0.0..1.0).contains(&fraction) {
                return Err(invalid(
                    "DSM fraction",
                    "expected a fraction strictly between zero and one",
                ));
            }
            let start = join(position(a), departure);
            let coast_time = days * fraction * DAY_TO_SECONDS;
            let coast = propagate_lagrangian(&start, coast_time, self.mu)?;
            let arc_time = days * (1.0 - fraction) * DAY_TO_SECONDS;
            let lambert = transfer(position(coast), position(b), arc_time, self.mu, false)?;
            let burn = sub(lambert.departure_velocity, velocity(coast));
            trace.delta_v_mps.push(
                norm(burn)
                    + if i == 0 && opt.charge_departure {
                        x[3]
                    } else {
                        0.0
                    },
            );
            if record {
                trace.impulse_vectors_mps.push(burn);
                trace.arcs.push(ArcRecord {
                    start,
                    end: coast,
                    duration_seconds: coast_time,
                    clockwise: None,
                });
                trace.arcs.push(lambert_arc(
                    position(coast),
                    position(b),
                    arc_time,
                    &lambert,
                    false,
                ));
            }
            previous = lambert.arrival_velocity;
        }
        let speed = norm(sub(previous, velocity(trace.body_states[n])));
        trace
            .delta_v_mps
            .push(opt.arrival.cost(speed, &self.bodies[n])?);
        let delta_v = trace.delta_v_mps.iter().sum::<f64>();
        let objective = if opt.juice_mass {
            positive("launch speed", Some(x[3]))?;
            let (s, c) = 0.409_072_975_f64.sin_cos();
            let declination = ((-launch[1] * s + launch[2] * c) / x[3])
                .clamp(-1.0, 1.0)
                .asin()
                .to_degrees();
            let launch_mass = Launcher::Ariane5.mass(x[3] / 1000.0, declination)?;
            let final_mass = launch_mass * (-(delta_v + 275.0) / (312.0 * 9.80665)).exp();
            let final_mass = if final_mass == 0.0 {
                1e-320
            } else {
                final_mass
            };
            trace.launch_declination_deg = Some(declination);
            trace.launch_mass_kg = Some(launch_mass);
            trace.final_mass_kg = Some(final_mass);
            -final_mass.ln()
        } else {
            delta_v
        };
        Ok((
            objective,
            Work {
                coasts: n,
                lambert: n,
                ..Work::default()
            },
        ))
    }

    fn run_impulses(
        &self,
        x: &[f64],
        opt: &ImpulseOptions,
        record: bool,
        trace: &mut Trace,
    ) -> Result<(f64, Work)> {
        let n = opt.impulses - 1;
        let weights: Vec<_> = (0..n).map(|i| x[2 + 4 * i]).collect();
        let times = alpha_to_direct(&weights, x[1])?;
        let arrival_epoch = if opt.departure.is_some() {
            x[0] + x[1]
        } else {
            x[x.len() - 1]
        };
        let initial = self.bodies[0].provider.state(x[0])?;
        let target = self.bodies[1].provider.state(arrival_epoch)?;
        trace.epochs_mjd2000 = vec![x[0], arrival_epoch];
        trace.body_states = vec![initial, target];
        trace.total_days = x[1];
        let mut state = initial;
        for (i, &days) in times[..n - 1].iter().enumerate() {
            let burn = uvv(x[3 + 4 * i], x[4 + 4 * i], x[5 + 4 * i])?;
            let start = join(position(state), add(velocity(state), burn));
            let seconds = days * DAY_TO_SECONDS;
            state = propagate_lagrangian(&start, seconds, self.mu)?;
            trace.delta_v_mps.push(norm(burn));
            if record {
                trace.impulse_vectors_mps.push(burn);
                trace.arcs.push(ArcRecord {
                    start,
                    end: state,
                    duration_seconds: seconds,
                    clockwise: None,
                });
            }
        }
        let seconds = times[n - 1] * DAY_TO_SECONDS;
        let clockwise = state[0] * state[4] - state[1] * state[3] < 0.0;
        let lambert = transfer(
            position(state),
            position(target),
            seconds,
            self.mu,
            clockwise,
        )?;
        let burns = [
            sub(lambert.departure_velocity, velocity(state)),
            sub(velocity(target), lambert.arrival_velocity),
        ];
        trace.delta_v_mps.extend(burns.map(norm));
        if record {
            trace.impulse_vectors_mps.extend(burns);
            trace.arcs.push(lambert_arc(
                position(state),
                position(target),
                seconds,
                &lambert,
                clockwise,
            ));
        }
        Ok((
            trace.delta_v_mps.iter().sum(),
            Work {
                coasts: n - 1,
                lambert: 1,
                ..Work::default()
            },
        ))
    }
}

impl Problem for ImpulsiveProblem {
    fn spec(&self) -> &ProblemSpec {
        &self.spec
    }
    fn evaluate(&self, x: &[f64]) -> Result<Evaluation> {
        self.run(x, false).map(|(value, _)| value)
    }
    fn decode(&self, x: &[f64], _samples: usize) -> Result<Value> {
        let (value, trace) = self.run(x, true)?;
        Ok(json!({"evaluation": value, "trajectory": trace, "central_mu_si": self.mu}))
    }
}

fn sequence(bodies: &[MissionBody], arrival: Arrival) -> Result<usize> {
    if !(2..=129).contains(&bodies.len()) {
        return Err(invalid("encounters", "expected 2..=129 bodies"));
    }
    if bodies
        .iter()
        .any(|body| body.metadata.central_mu != bodies[0].metadata.central_mu)
    {
        return Err(invalid(
            "encounters",
            "bodies must share the same central gravitational parameter",
        ));
    }
    for body in &bodies[1..bodies.len() - 1] {
        positive("body mu", body.metadata.body_mu)?;
        positive("safe radius", body.metadata.safe_radius)?;
    }
    arrival.validate(&bodies[bodies.len() - 1])?;
    Ok(bodies.len() - 1)
}

fn positive(label: &'static str, value: Option<f64>) -> Result<f64> {
    value
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| invalid(label, "requires a positive finite value"))
}

fn nonnegative(label: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid(label, "requires a nonnegative finite value"))
    }
}

fn positive_bounds(label: &'static str, bounds: [f64; 2]) -> Result<()> {
    positive(label, Some(bounds[0]))?;
    if !bounds[1].is_finite() || bounds[1] < bounds[0] {
        return Err(invalid(label, "invalid upper bound"));
    }
    Ok(())
}

fn uvv(u: f64, v: f64, speed: f64) -> Result<Vector3> {
    nonnegative("impulse speed", speed)?;
    if !(0.0..=1.0).contains(&v) || !(0.0..=1.0).contains(&u) {
        return Err(invalid(
            "uvV direction",
            "both coordinates must be in [0,1]",
        ));
    }
    let (s, c) = (TAU * u).sin_cos();
    let latitude = (2.0 * v - 1.0).acos() - PI * 0.5;
    let (sl, cl) = latitude.sin_cos();
    Ok([speed * cl * c, speed * cl * s, speed * sl])
}

fn transfer(a: Vector3, b: Vector3, dt: f64, mu: f64, clockwise: bool) -> Result<LambertSolution> {
    let p = LambertProblem::new(a, b, dt, mu, clockwise, 0)?;
    p.solutions()
        .first()
        .cloned()
        .ok_or_else(|| invalid("Lambert transfer", "no zero-revolution solution"))
}

fn lambert_arc(
    a: Vector3,
    b: Vector3,
    dt: f64,
    transfer: &LambertSolution,
    clockwise: bool,
) -> ArcRecord {
    ArcRecord {
        start: join(a, transfer.departure_velocity),
        end: join(b, transfer.arrival_velocity),
        duration_seconds: dt,
        clockwise: Some(clockwise),
    }
}

fn position(s: CartesianState) -> Vector3 {
    [s[0], s[1], s[2]]
}
fn velocity(s: CartesianState) -> Vector3 {
    [s[3], s[4], s[5]]
}
fn join(a: Vector3, b: Vector3) -> CartesianState {
    [a[0], a[1], a[2], b[0], b[1], b[2]]
}
fn add(a: Vector3, b: Vector3) -> Vector3 {
    core::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: Vector3, b: Vector3) -> Vector3 {
    core::array::from_fn(|i| a[i] - b[i])
}
fn norm(a: Vector3) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}
