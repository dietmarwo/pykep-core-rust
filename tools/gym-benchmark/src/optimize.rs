// SPDX-License-Identifier: MPL-2.0

use fcmaes_core::{Cmaes, CmaesParams, De, DeParams, Fitness, Mode, ModeParams};
use pykep_trajopt::{Evaluation, Problem, Work};
use serde::Serialize;
use std::io::{self, Write};
use std::time::{Duration, Instant};

const BAD: f64 = 1e100;
const EQ_TOL: f64 = 1e-6;
const INEQ_TOL: f64 = 1e-8;

#[derive(Serialize, Clone)]
pub(crate) struct Candidate {
    pub x: Vec<f64>,
    pub evaluation: Evaluation,
    merit: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct Config<'a> {
    pub optimizer: &'a str,
    pub evaluations: usize,
    pub workers: usize,
    pub seed: u64,
    pub seconds: Option<f64>,
}

pub(crate) struct Run {
    pub actual: usize,
    pub failed: usize,
    pub feasible: usize,
    pub elapsed: f64,
    pub best: Option<Candidate>,
    pub best_feasible: Option<Candidate>,
    pub pareto: Vec<Candidate>,
    pub work: Work,
}

struct Search<'a, W> {
    problem: &'a dyn Problem,
    bounds: &'a [[f64; 2]],
    free: Vec<usize>,
    cfg: Config<'a>,
    started: Instant,
    deadline: Option<Duration>,
    log: W,
    run: Run,
}

impl<W: Write> Search<'_, W> {
    fn done(&self, limit: usize) -> bool {
        self.run.actual >= limit || self.deadline.is_some_and(|d| self.started.elapsed() >= d)
    }

    fn population(&mut self, batch: &[Vec<f64>]) -> pykep_core::Result<Vec<Vec<f64>>> {
        let problem = self.problem;
        let spec = problem.spec();
        let mut candidates = Vec::with_capacity(batch.len());
        for point in batch {
            let mut x = spec.midpoint();
            for (&column, &value) in self.free.iter().zip(point) {
                let b = self.bounds[column];
                x[column] = b[0] + value.clamp(0.0, 1.0) * (b[1] - b[0]);
            }
            candidates.push(x);
        }
        let results =
            pykep_core::batch::try_map(&candidates, self.cfg.workers, |x| Ok(problem.evaluate(x)))?;
        let mut selection = Vec::with_capacity(batch.len());
        for (x, result) in candidates.into_iter().zip(results) {
            self.run.actual += 1;
            match result {
                Ok(value) => {
                    let constraints = value
                        .equalities
                        .iter()
                        .map(|v| v.abs() - EQ_TOL)
                        .chain(value.inequalities.iter().map(|v| v - INEQ_TOL));
                    let violation = constraints.clone().map(|v| v.max(0.0).powi(2)).sum::<f64>();
                    let merit = (value.objectives[0] + 1e3 * violation).min(BAD);
                    let weights = if self.cfg.optimizer == "mode" {
                        value
                            .objectives
                            .iter()
                            .copied()
                            .chain(constraints)
                            .collect()
                    } else {
                        vec![merit]
                    };
                    self.run.work.segments += value.work.segments;
                    self.run.work.endpoints += value.work.endpoints;
                    self.run.work.coasts += value.work.coasts;
                    self.run.work.lambert += value.work.lambert;
                    let candidate = Candidate {
                        x,
                        evaluation: value,
                        merit,
                    };
                    if candidate.evaluation.feasible(EQ_TOL, INEQ_TOL) {
                        self.run.feasible += 1;
                        if self.run.best_feasible.as_ref().is_none_or(|best| {
                            candidate.evaluation.objectives[0] < best.evaluation.objectives[0]
                        }) {
                            self.run.best_feasible = Some(candidate.clone());
                        }
                        if spec.objectives.len() > 1
                            && !self
                                .run
                                .pareto
                                .iter()
                                .any(|other| dominates(other, &candidate))
                        {
                            self.run
                                .pareto
                                .retain(|other| !dominates(&candidate, other));
                            self.run.pareto.push(candidate.clone());
                        }
                    }
                    if self.run.best.as_ref().is_none_or(|best| merit < best.merit) {
                        self.run.best = Some(candidate);
                    }
                    selection.push(weights);
                }
                Err(_) => {
                    self.run.failed += 1;
                    selection.push(vec![
                        BAD;
                        if self.cfg.optimizer == "mode" {
                            spec.objectives.len() + spec.equalities.len() + spec.inequalities.len()
                        } else {
                            1
                        }
                    ]);
                }
            }
        }
        Ok(selection)
    }

    fn record(&mut self) -> io::Result<()> {
        let best = self
            .run
            .best_feasible
            .as_ref()
            .map(|v| v.evaluation.objectives[0].to_string())
            .unwrap_or_default();
        writeln!(
            self.log,
            "{},{:.9},{},{}",
            self.run.actual,
            self.started.elapsed().as_secs_f64(),
            best,
            self.run.feasible
        )
    }
}

fn dominates(a: &Candidate, b: &Candidate) -> bool {
    a.evaluation
        .objectives
        .iter()
        .zip(&b.evaluation.objectives)
        .all(|(a, b)| a <= b)
}

pub(crate) fn run(
    problem: &dyn Problem,
    cfg: Config<'_>,
    mut log: impl Write,
) -> Result<Run, Box<dyn std::error::Error>> {
    let spec = problem.spec();
    let bounds = &spec.bounds;
    let free: Vec<_> = (0..bounds.len())
        .filter(|&i| bounds[i][0] < bounds[i][1])
        .collect();
    if free.is_empty() {
        return Err("optimizer requires a free decision".into());
    }
    let dim = free.len();
    if cfg.optimizer != "mode" && spec.objectives.len() != 1 {
        return Err("use mode for multiple objectives".into());
    }
    writeln!(
        log,
        "evaluations,elapsed_seconds,best_feasible_objective,feasible_evaluations"
    )?;
    let popsize = 32;
    let fitness = |outputs| Fitness::bounded(dim, outputs, &vec![0.0; dim], &vec![1.0; dim]);
    let mut search = Search {
        problem,
        bounds,
        free,
        cfg,
        started: Instant::now(),
        deadline: cfg.seconds.map(Duration::from_secs_f64),
        log,
        run: Run {
            actual: 0,
            failed: 0,
            feasible: 0,
            elapsed: 0.0,
            best: None,
            best_feasible: None,
            pareto: Vec::new(),
            work: Work::default(),
        },
    };
    if cfg.optimizer == "mode" {
        let ncon = spec.equalities.len() + spec.inequalities.len();
        let mut mode = Mode::try_new(
            fitness(spec.objectives.len() + ncon),
            spec.objectives.len(),
            ncon,
            None,
            &ModeParams {
                popsize,
                seed: cfg.seed,
                ..ModeParams::default()
            },
        )?;
        while !search.done(cfg.evaluations) {
            let batch = mode.ask();
            let n = batch.len().min(cfg.evaluations - search.run.actual);
            let y = search.population(&batch[..n])?;
            search.record()?;
            if n < batch.len() || mode.tell(&y) != 0 {
                break;
            }
        }
    } else if cfg.optimizer == "de-cma" || cfg.optimizer == "de" {
        let limit = if cfg.optimizer == "de" {
            cfg.evaluations
        } else {
            cfg.evaluations.div_ceil(2)
        };
        let mut de = De::new(
            fitness(1),
            &[],
            &[],
            None,
            &DeParams {
                popsize,
                seed: cfg.seed,
                max_evaluations: limit as u64,
                ..DeParams::default()
            },
        );
        while !search.done(limit) {
            let batch = de.ask();
            let n = batch.len().min(limit - search.run.actual);
            let y: Vec<_> = search
                .population(&batch[..n])?
                .into_iter()
                .map(|v| v[0])
                .collect();
            search.record()?;
            if n < batch.len() || de.tell(&y) != 0 {
                break;
            }
        }
        if cfg.optimizer == "de-cma" && !search.done(cfg.evaluations) {
            let guess: Vec<_> = search
                .free
                .iter()
                .map(|&i| {
                    search.run.best.as_ref().map_or(0.5, |p| {
                        (p.x[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
                    })
                })
                .collect();
            let mut cma = Cmaes::new(
                fitness(1),
                &guess,
                &vec![0.15; dim],
                &CmaesParams {
                    popsize,
                    seed: cfg.seed.wrapping_add(1),
                    max_evaluations: (cfg.evaluations - search.run.actual) as u64,
                    ..CmaesParams::default()
                },
            );
            while !search.done(cfg.evaluations) {
                let batch = cma.ask();
                let n = batch.len().min(cfg.evaluations - search.run.actual);
                let y: Vec<_> = search
                    .population(&batch[..n])?
                    .into_iter()
                    .map(|v| v[0])
                    .collect();
                search.record()?;
                if n < batch.len() || cma.tell(&y) != 0 {
                    break;
                }
            }
        }
    } else {
        return Err("unknown optimizer. Expected de, de-cma or mode".into());
    }
    search.log.flush()?;
    search.run.elapsed = search.started.elapsed().as_secs_f64();
    Ok(search.run)
}
