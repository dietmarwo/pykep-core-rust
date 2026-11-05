// SPDX-License-Identifier: MPL-2.0
//! Bounded native trajectory problem replay and optimizer studies.

mod optimize;

use pykep_core::integration::IntegrationMethod;
use pykep_trajopt::{GymProblem, Problem, TopsOptions};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap, error::Error, fs, hint::black_box, io::Write, path::Path, time::Instant,
};

type AnyResult<T> = Result<T, Box<dyn Error>>;

fn main() -> AnyResult<()> {
    let argv: Vec<_> = std::env::args().collect();
    let Some(command) = argv.get(1).filter(|s| *s != "--help") else {
        println!("pykep-gym-benchmark list|spec|evaluate|replay|optimize|validate");
        println!("  --problem ID --nseg N --cut FRACTION --time-encoding uniform|softmax");
        println!("  --backend taylor|dop853 --tolerance T --derivative-tolerance T --workers N");
        println!("  evaluate: --decisions FILE --gradient true --samples-per-segment N");
        println!("  replay: --count N --seed N --passes N --decisions FILE");
        println!(
            "  optimize: --optimizer de|de-cma|mode --evaluations N --seconds S --seed N --output DIR"
        );
        println!("  validate: --run FILE");
        return Ok(());
    };
    if command == "list" {
        for id in GymProblem::ids()? {
            println!("{id}");
        }
        return Ok(());
    }
    let (pairs, remainder) = argv[2..].as_chunks::<2>();
    if !remainder.is_empty() {
        return Err("expected --option value pairs".into());
    }
    let flags: BTreeMap<_, _> = pairs
        .iter()
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect();
    let flag = |name, default| flags.get(name).copied().unwrap_or(default);
    let integer = |name, default| flag(name, default).parse::<usize>();
    let number = |name, default| flag(name, default).parse::<f64>();
    let mut options = TopsOptions {
        segments: flags.get("--nseg").map(|v| v.parse()).transpose()?,
        cut: number("--cut", "0.5")?,
        time_encoding: flag("--time-encoding", "uniform").parse()?,
        direction_inequalities: flags
            .get("--direction-inequalities")
            .map(|v| v.parse())
            .transpose()?,
        method: match flag("--backend", "taylor") {
            "taylor" => IntegrationMethod::Taylor,
            "dop853" => IntegrationMethod::Dop853,
            _ => return Err("unknown integration backend".into()),
        },
        tolerance: number("--tolerance", "1e-13")?,
        derivative_tolerance: number("--derivative-tolerance", "1e-13")?,
        maximum_steps: flags
            .get("--maximum-steps")
            .map(|v| v.parse())
            .transpose()?,
    };
    let run: Option<Value> = flags
        .get("--run")
        .map(|path| read_json(Path::new(path)))
        .transpose()?;
    let id = run
        .as_ref()
        .and_then(|r| r["pykep"]["problem"].as_str())
        .unwrap_or(flag("--problem", "gym.eve_mga1dsm"));
    if let Some(r) = &run {
        let s = &r["pykep"]["settings"];
        options.segments = s["segments"].as_u64().map(|n| n as usize);
        options.cut = s["cut"].as_f64().unwrap_or(options.cut);
        options.time_encoding = s["time_encoding"].as_str().unwrap_or("uniform").parse()?;
        options.direction_inequalities = s["direction_inequalities"].as_bool();
        options.method = match s["backend"].as_str().unwrap_or("taylor") {
            "taylor" => IntegrationMethod::Taylor,
            "dop853" => IntegrationMethod::Dop853,
            _ => return Err("unknown recorded integration backend".into()),
        };
        options.derivative_tolerance = s["derivative_tolerance"]
            .as_f64()
            .unwrap_or(options.derivative_tolerance);
        options.maximum_steps = s["maximum_steps"].as_u64().map(|n| n as usize);
        // Replay the recorded decisions with a tighter nominal tolerance.
        options.tolerance = 5e-15;
    }
    let problem = GymProblem::new(id, options)?;
    let spec = problem.spec();
    let workers = integer("--workers", "1")?;
    let seed = flag("--seed", "42").parse::<u64>()?;
    match command.as_str() {
        "spec" => println!("{}", serde_json::to_string_pretty(spec)?),
        "evaluate" | "validate" => {
            let x: Vec<f64> = if let Some(r) = &run {
                serde_json::from_value(r["best"]["x"].clone())?
            } else if let Some(path) = flags.get("--decisions") {
                serde_json::from_value(read_json(Path::new(path))?)?
            } else {
                spec.midpoint()
            };
            let evaluation = problem.evaluate(&x)?;
            let samples = integer("--samples-per-segment", "0")?;
            let decoded = problem.decode(&x, samples)?;
            let mut result = json!({"spec":spec,"x":x,"evaluation":evaluation,"decoded":decoded});
            if flag("--gradient", "false").parse::<bool>()? {
                result["gradient"] = json!(problem.gradient(&x)?);
            }
            let mut failure = None;
            if command == "validate" {
                let r = run.as_ref().ok_or("validate requires --run FILE")?;
                let forward = if matches!(problem, GymProblem::Tops(_)) {
                    let residual = problem.forward_residual(&x)?;
                    result["forward_residual"] = json!(residual);
                    Some(residual.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
                } else {
                    None
                };
                failure = validate_replay(r, &evaluation, forward, &mut result)?;
            }
            println!("{}", serde_json::to_string_pretty(&result)?);
            if let Some(reason) = failure {
                return Err(reason.into());
            }
        }
        "replay" => {
            let count = integer("--count", "64")?;
            let passes = integer("--passes", "7")?;
            if !(1..=10_000).contains(&count) || !(1..=100).contains(&passes) {
                return Err("replay count or passes exceeds limits".into());
            }
            let population: Vec<Vec<f64>> = if let Some(path) = flags.get("--decisions") {
                serde_json::from_value(read_json(Path::new(path))?)?
            } else {
                population(spec, count, seed)
            };
            if population.is_empty() || population.len() > 10_000 {
                return Err("invalid population size".into());
            }
            let evaluate =
                || pykep_core::batch::try_map(&population, workers, |x| Ok(problem.evaluate(x)));
            let warm = evaluate()?;
            let failed = warm.iter().filter(|v| v.is_err()).count();
            let mut samples = Vec::with_capacity(passes);
            for _ in 0..passes {
                let t = Instant::now();
                black_box(evaluate()?);
                samples.push(t.elapsed().as_secs_f64() * 1e9 / population.len() as f64);
            }
            println!(
                "{}",
                json!({"problem":spec,"seed":seed,"workers":workers,
                "count":population.len(),"failed":failed,"ns_per_call":samples,
                "first_fitness":warm.first().and_then(|v|v.as_ref().ok()).map(|v|v.fitness())})
            );
        }
        "optimize" => {
            let optimizer = flag("--optimizer", "de-cma");
            let evaluations = integer("--evaluations", "4096")?;
            let seconds = flags
                .get("--seconds")
                .map(|v| v.parse::<f64>())
                .transpose()?;
            if !(1..=1_000_000).contains(&evaluations)
                || seconds.is_some_and(|s| !(0.001..=3600.0).contains(&s))
            {
                return Err("invalid optimizer budget".into());
            }
            let folder = Path::new(flags.get("--output").ok_or("optimize requires --output")?);
            fs::create_dir_all(folder)?;
            let convergence = fs::File::create_new(folder.join("convergence.csv"))?;
            let cfg = optimize::Config {
                optimizer,
                evaluations,
                workers,
                seed,
                seconds,
            };
            let result = optimize::run(&problem, cfg, std::io::BufWriter::new(convergence))?;
            let best = result.best_feasible.as_ref().or(result.best.as_ref());
            let mut artifacts = json!({"convergence":"convergence.csv","spec":"spec.json"});
            if let Some(candidate) = best {
                write_json(
                    &folder.join("trajectory.json"),
                    &problem.decode(&candidate.x, 0)?,
                )?;
                artifacts["trajectory"] = json!("trajectory.json");
            }
            if !result.pareto.is_empty() {
                let mut out = fs::File::create(folder.join("pareto.csv"))?;
                let objectives = (0..spec.objectives.len()).map(|i| format!("objective_{i}"));
                let decisions = (0..spec.bounds.len()).map(|i| format!("decision_{i}"));
                writeln!(
                    out,
                    "point_id,feasible,selected,{}",
                    objectives.chain(decisions).collect::<Vec<_>>().join(",")
                )?;
                for (i, candidate) in result.pareto.iter().enumerate() {
                    let values = candidate
                        .evaluation
                        .objectives
                        .iter()
                        .chain(&candidate.x)
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    let selected = best.is_some_and(|best| best.x == candidate.x);
                    writeln!(out, "{i},true,{selected},{values}")?;
                }
                artifacts["pareto"] = json!("pareto.csv");
            }
            write_json(&folder.join("spec.json"), &serde_json::to_value(spec)?)?;
            let record = json!({
                "schema_version":1,"tutorial":"pykep-trajectory-gym","formulation":spec.formulation,
                "command":argv.join(" "),"seed":seed,"workers":workers,"requested_evaluations":evaluations,
                "actual_evaluations":result.actual,"elapsed_seconds":result.elapsed,"status":"completed",
                "objectives":spec.objectives.iter().enumerate().map(|(i,label)| {
                    let (label, unit) = label.strip_suffix(']').and_then(|s|s.rsplit_once(" [")).unwrap_or((label,"1"));
                    json!({"column":format!("objective_{i}"),"label":label,"unit":unit})
                }).collect::<Vec<_>>(),
                "descriptors":[],"feasible_results":result.feasible,"failed_evaluations":result.failed,
                "best":best,"artifacts":artifacts,
                "pykep":{"problem":spec.id,"settings":spec.settings,"optimizer":optimizer,"work_of_successful_calls":result.work,
                    "equality_tolerance":1e-6,"inequality_tolerance":1e-8,"wall_budget_seconds":seconds,
                    "reference_revision":spec.source,"fcmaes_core":"0.2.1","pareto_points":result.pareto.len()}
            });
            write_json(&folder.join("run.json"), &record)?;
            println!(
                "{}",
                json!({"output":folder,"actual_evaluations":result.actual,"feasible_results":result.feasible,
                "failed_evaluations":result.failed,"elapsed_seconds":result.elapsed,
                "best_objectives":best.map(|c|&c.evaluation.objectives)})
            );
        }
        _ => return Err("unknown command, see --help".into()),
    }
    Ok(())
}

fn population(spec: &pykep_trajopt::ProblemSpec, count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            spec.bounds
                .iter()
                .map(|bound| {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    let u = ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
                    bound[0] + u * (bound[1] - bound[0])
                })
                .collect()
        })
        .collect()
}

/// Compare a replayed best decision with its recorded evaluation.
///
/// Returns a failure reason when the replayed objectives differ, a recorded
/// feasible point becomes infeasible, or the forward residual is too large.
fn validate_replay(
    run: &Value,
    evaluation: &pykep_trajopt::Evaluation,
    forward_residual: Option<f64>,
    result: &mut Value,
) -> AnyResult<Option<String>> {
    let recorded = |key: &str| -> AnyResult<Vec<f64>> {
        serde_json::from_value(run["best"]["evaluation"][key].clone())
            .map_err(|e| format!("run record lacks best.evaluation.{key}: {e}").into())
    };
    let (objectives, equalities, inequalities) = (
        recorded("objectives")?,
        recorded("equalities")?,
        recorded("inequalities")?,
    );
    let equality = run["pykep"]["equality_tolerance"].as_f64().unwrap_or(1e-6);
    let inequality = run["pykep"]["inequality_tolerance"]
        .as_f64()
        .unwrap_or(1e-8);
    let objective_difference = objectives
        .iter()
        .zip(&evaluation.objectives)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0_f64, f64::max);
    let recorded_feasible = equalities.iter().all(|v| v.abs() <= equality)
        && inequalities.iter().all(|v| *v <= inequality);
    let feasible = evaluation.feasible(equality, inequality);
    let mut failures = Vec::new();
    if objectives.len() != evaluation.objectives.len() || objective_difference > 1e-6 {
        failures.push(format!(
            "objective differs from the record by {objective_difference:e}"
        ));
    }
    if recorded_feasible && !feasible {
        failures.push("recorded feasible point is infeasible on replay".into());
    }
    if let Some(residual) = forward_residual
        && feasible
        && residual > 1e-6
    {
        failures.push(format!("forward residual {residual:e} exceeds 1e-6"));
    }
    result["validation"] = json!({
        "objective_difference": objective_difference,
        "recorded_feasible": recorded_feasible,
        "feasible": feasible,
        "forward_residual": forward_residual,
        "passed": failures.is_empty(),
        "failures": failures,
    });
    Ok((!failures.is_empty()).then(|| failures.join("; ")))
}

fn read_json(path: &Path) -> AnyResult<Value> {
    if path.metadata()?.len() > 32 << 20 {
        return Err("JSON input exceeds 32 MiB".into());
    }
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

fn write_json(path: &Path, value: &Value) -> AnyResult<()> {
    let temp = path.with_extension("tmp");
    let mut output = fs::File::create(&temp)?;
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    fs::rename(temp, path)?;
    Ok(())
}
