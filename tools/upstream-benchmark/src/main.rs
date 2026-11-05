// SPDX-License-Identifier: MPL-2.0
//! Matched input kernel benchmark with independent output verification.

use pykep_core::astro::anomalies::{eccentric_to_mean_anomaly, mean_to_eccentric_anomaly};
use pykep_core::astro::lambert::LambertProblem;
use pykep_core::astro::propagation::{propagate_lagrangian, propagate_lagrangian_with_stm};
use pykep_core::dynamics::zoh::ZohKeplerDynamics;
use pykep_core::integration::{IntegrationMethod, IntegratorOptions};
use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint, ZohKeplerLeg};
use pykep_core::{PykepError, Result};
use serde_json::json;
use std::{error::Error, fs, hint::black_box, path::Path, time::Instant};

const LIMIT: u64 = 128 << 20;

struct Dataset {
    suite: String,
    width: usize,
    count: usize,
    values: Vec<f64>,
}

impl Dataset {
    fn read(path: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        if path.metadata()?.len() > LIMIT {
            return Err("fixture exceeds 128 MiB".into());
        }
        let bytes = fs::read(path)?;
        let newline = bytes
            .iter()
            .position(|b| *b == b'\n')
            .ok_or("missing fixture header")?;
        let mut words = std::str::from_utf8(&bytes[..newline])?.split_whitespace();
        if words.next() != Some("PYKEP_BENCH_V1") {
            return Err("bad fixture magic".into());
        }
        let suite = words.next().ok_or("missing suite")?.to_owned();
        let count = words.next().ok_or("missing count")?.parse::<usize>()?;
        let width = words.next().ok_or("missing width")?.parse::<usize>()?;
        let data = &bytes[newline + 1..];
        if count == 0
            || width == 0
            || words.next().is_some()
            || count.checked_mul(width).and_then(|v| v.checked_mul(8)) != Some(data.len())
        {
            return Err("invalid fixture dimensions".into());
        }
        let values: Vec<_> = data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| f64::from_le_bytes(*v))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite fixture value".into());
        }
        Ok(Self {
            suite,
            width,
            count,
            values,
        })
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> {
        self.values.chunks_exact(self.width)
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.values[i * self.width..(i + 1) * self.width]
    }
}

struct Options {
    passes: usize,
    repetitions: usize,
    values: usize,
}

fn measure<T>(
    data: &Dataset,
    operation: &str,
    options: &Options,
    run: impl Fn(usize) -> Result<T>,
    view: impl Fn(T) -> Vec<f64>,
) {
    // Validate every input once. Limit serialized individual records.
    let mut failures = 0usize;
    let mut outputs = 0usize;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for i in 0..data.count {
        match run(i) {
            Ok(value) => {
                let values = view(value);
                outputs += values.len();
                for value in &values {
                    max_abs = max_abs.max(value.abs());
                    sum_abs += value.abs();
                }
                if i < options.values {
                    println!(
                        "{}",
                        json!({"suite":data.suite,"operation":operation,"case":i,"values":values})
                    );
                }
            }
            Err(error) => {
                failures += 1;
                if failures <= 8 {
                    println!(
                        "{}",
                        json!({"suite":data.suite,"operation":operation,"case":i,"error":error.to_string()})
                    );
                }
            }
        }
    }
    let mut samples = Vec::with_capacity(options.passes);
    for _ in 0..options.passes {
        let start = Instant::now();
        for _ in 0..options.repetitions {
            for i in 0..data.count {
                let _ = black_box(run(black_box(i)));
            }
        }
        samples
            .push(start.elapsed().as_secs_f64() * 1e9 / (data.count * options.repetitions) as f64);
    }
    println!(
        "{}",
        json!({
            "suite":data.suite,"operation":operation,"cases":data.count,"failed":failures,
            "output_scalars":outputs,"max_abs":max_abs,"sum_abs":sum_abs,"ns_per_call":samples,
            "repetitions":options.repetitions,
        })
    );
}

fn sf(row: &[f64], controls: &[f64], tof: f64, mf: f64) -> Result<SimsFlanaganLeg> {
    SimsFlanaganLeg::new(
        SpacecraftEndpoint::new(array(&row[..6]), row[6])?,
        controls.chunks_exact(3).map(array).collect(),
        SpacecraftEndpoint::new(array(&row[7..13]), mf)?,
        SimsFlanaganSettings::new(tof, row[15], row[16], row[17], row[18])?,
    )
}

fn sf_fit(row: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    let n = x.len() - 2;
    let leg = sf(row, &x[..n], x[n], x[n + 1])?;
    Ok([
        vec![-x[n + 1]],
        leg.mismatch_constraints()?.to_vec(),
        leg.throttle_constraints(),
    ]
    .concat())
}

fn sf_fd(row: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    let mut plus = x.to_vec();
    let mut minus = x.to_vec();
    let n = x.len();
    let m = sf_fit(row, x)?.len();
    let mut grad = vec![0.0; n * m];
    for i in 0..n {
        let h = x[i].abs().max(1.0) * 1e-8;
        plus[i] += h;
        minus[i] -= h;
        let u = sf_fit(row, &plus)?;
        let l = sf_fit(row, &minus)?;
        for r in 0..m {
            grad[r * n + i] = (u[r] - l[r]) / (2.0 * h);
        }
        plus[i] = x[i];
        minus[i] = x[i];
    }
    Ok(grad)
}

fn sf_x(row: &[f64]) -> Vec<f64> {
    let mut x = row[19..].to_vec();
    x.extend([row[14], row[13]]);
    x
}

fn array<const N: usize>(values: &[f64]) -> [f64; N] {
    core::array::from_fn(|i| values[i])
}

fn zoh(row: &[f64], n: usize, tol: f64) -> Result<ZohKeplerLeg> {
    ZohKeplerLeg::new(
        ZohKeplerDynamics,
        array(row),
        row[15 + n..].chunks_exact(4).map(array).collect(),
        array(&row[7..]),
        row[14..15 + n].to_vec(),
        [1.0 / 3000.0],
        0.5,
        IntegratorOptions {
            absolute_tolerance: tol,
            relative_tolerance: tol,
            maximum_steps: 1000,
            ..IntegratorOptions::default()
        },
    )
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help") {
        println!(
            "pykep-upstream-benchmark FIXTURE [--passes=5] [--repetitions=1] [--values=0] [--zoh-derivative-tol=1e-8] [--zoh-backend=taylor]"
        );
        return Ok(());
    }
    let flag = |name: &str, default: usize| -> std::result::Result<usize, Box<dyn Error>> {
        Ok(args
            .iter()
            .find_map(|a| a.strip_prefix(name))
            .map(str::parse)
            .transpose()?
            .unwrap_or(default))
    };
    let options = Options {
        passes: flag("--passes=", 5)?,
        repetitions: flag("--repetitions=", 1)?,
        values: flag("--values=", 0)?,
    };
    if !(1..=100).contains(&options.passes)
        || !(1..=100000).contains(&options.repetitions)
        || options.values > 1024
    {
        return Err("invalid sample controls".into());
    }
    let deriv_tol = args
        .iter()
        .find_map(|a| a.strip_prefix("--zoh-derivative-tol="))
        .map(str::parse)
        .transpose()?
        .unwrap_or(1e-8);
    let zoh_method = match args
        .iter()
        .find_map(|a| a.strip_prefix("--zoh-backend="))
        .unwrap_or("taylor")
    {
        "taylor" => IntegrationMethod::Taylor,
        "dop853" => IntegrationMethod::Dop853,
        _ => return Err("invalid ZOH backend".into()),
    };
    let data = Dataset::read(Path::new(&args[0]))?;
    let suite = &data.suite;
    if suite.starts_with("anomaly") && data.width == 2 {
        measure(
            &data,
            "m2e",
            &options,
            |i| {
                let r = data.row(i);
                mean_to_eccentric_anomaly(r[0], r[1])
            },
            |e| vec![e.sin(), e.cos()],
        );
        if suite.ends_with("-error") {
            measure(
                &data,
                "m2e-roundtrip",
                &options,
                |i| {
                    let r = data.row(i);
                    let e = mean_to_eccentric_anomaly(r[0], r[1])?;
                    let m = eccentric_to_mean_anomaly(e, r[1])?;
                    Ok(0.5 * ((m.sin() - r[0].sin()).abs() + (m.cos() - r[0].cos()).abs()))
                },
                |v| vec![v],
            );
        }
    } else if (suite.starts_with("lagrangian")
        || suite.starts_with("roundtrip")
        || suite.starts_with("stm"))
        && data.width == 7
    {
        measure(
            &data,
            "state",
            &options,
            |i| {
                let r = data.row(i);
                propagate_lagrangian(&array(r), r[6], 1.0)
            },
            |v| v.to_vec(),
        );
        if suite.starts_with("stm") {
            measure(
                &data,
                "state-stm",
                &options,
                |i| {
                    let r = data.row(i);
                    propagate_lagrangian_with_stm(&array(r), r[6], 1.0)
                },
                |(y, j)| y.into_iter().chain(j.into_iter().flatten()).collect(),
            );
        }
        if suite.starts_with("roundtrip") {
            measure(
                &data,
                "roundtrip-error",
                &options,
                |i| {
                    let r = data.row(i);
                    let end = propagate_lagrangian(&array(r), r[6], 1.0)?;
                    let start = propagate_lagrangian(&end, -r[6], 1.0)?;
                    Ok(start.iter().zip(r).map(|(a, b)| (a - b).abs()).sum::<f64>())
                },
                |v| vec![v],
            );
        }
    } else if (suite == "lambert0" || suite == "lambert20") && data.width == 9 {
        let max_revs = if suite == "lambert0" { 0 } else { 20 };
        measure(
            &data,
            "solve",
            &options,
            |i| {
                let r = data.row(i);
                LambertProblem::new(array(r), array(&r[3..]), r[6], r[8], r[7] != 0.0, max_revs)
            },
            |l| {
                l.solutions()
                    .iter()
                    .flat_map(|s| s.departure_velocity.into_iter().chain(s.arrival_velocity))
                    .collect()
            },
        );
        let mut max_error = 0.0_f64;
        let mut failures = 0;
        for r in data.rows().take(128) {
            let l =
                LambertProblem::new(array(r), array(&r[3..]), r[6], r[8], r[7] != 0.0, max_revs)?;
            for s in l.solutions() {
                let state = array::<6>(&[
                    r[0],
                    r[1],
                    r[2],
                    s.departure_velocity[0],
                    s.departure_velocity[1],
                    s.departure_velocity[2],
                ]);
                match propagate_lagrangian(&state, r[6], r[8]) {
                    Ok(end) => {
                        for j in 0..3 {
                            max_error =
                                max_error.max((end[j] - r[3 + j]).abs() / r[3 + j].abs().max(1.0));
                        }
                    }
                    Err(_) => failures += 1,
                }
            }
        }
        println!(
            "{}",
            json!({"suite":suite,"validation":"lambert_closure","maximum_scaled_error":max_error,"failed":failures})
        );
    } else if suite.starts_with("sf") && data.width >= 22 && (data.width - 19) % 3 == 0 {
        measure(
            &data,
            "construct",
            &options,
            |i| {
                let r = data.row(i);
                sf(r, &r[19..], r[14], r[13])
            },
            |l| vec![l.segment_count() as f64],
        );
        let legs: Vec<_> = data
            .rows()
            .map(|r| sf(r, &r[19..], r[14], r[13]))
            .collect::<Result<_>>()?;
        measure(
            &data,
            "mismatch",
            &options,
            |i| legs[i].mismatch_constraints(),
            |v| v.to_vec(),
        );
        measure(
            &data,
            "jacobian",
            &options,
            |i| legs[i].mismatch_jacobian(),
            |j| {
                j.departure
                    .into_iter()
                    .flatten()
                    .chain(j.arrival.into_iter().flatten())
                    .chain(j.controls_and_time.into_iter().flatten())
                    .collect()
            },
        );
        let population: Vec<_> = data.rows().map(sf_x).collect();
        measure(
            &data,
            "udp-analytic",
            &options,
            |i| {
                let r = data.row(i);
                let x = &population[i];
                let n = x.len() - 2;
                let leg = sf(r, &x[..n], x[n], x[n + 1])?;
                let j = leg.mismatch_jacobian()?;
                let mut out = vec![0.0; x.len() * (8 + n / 3)];
                out[n + 1] = -1.0;
                for k in 0..7 {
                    let row = &mut out[(k + 1) * x.len()..(k + 2) * x.len()];
                    row[..n + 1].copy_from_slice(&j.controls_and_time[k]);
                    row[n + 1] = j.arrival[k][6];
                }
                for k in 0..n {
                    out[(8 + k / 3) * x.len() + k] = 2.0 * x[k];
                }
                Ok(out)
            },
            |v| v,
        );
        measure(
            &data,
            "udp-numerical",
            &options,
            |i| sf_fd(data.row(i), &population[i]),
            |v| v,
        );
    } else if let Some(n) = suite.strip_prefix("zoh") {
        let n = n.parse::<usize>()?;
        if !(1..=512).contains(&n) || data.width != 15 + 5 * n {
            return Err("invalid ZOH shape".into());
        }
        let legs: Vec<_> = data
            .rows()
            .map(|r| zoh(r, n, 1e-14))
            .collect::<Result<_>>()?;
        let deriv: Vec<_> = data
            .rows()
            .map(|r| zoh(r, n, deriv_tol))
            .collect::<Result<_>>()?;
        measure(
            &data,
            "mismatch",
            &options,
            |i| legs[i].mismatch_constraints_with_method(zoh_method),
            |v| v.to_vec(),
        );
        measure(
            &data,
            "jacobian",
            &options,
            |i| deriv[i].mismatch_jacobian_with_method(zoh_method),
            |j| {
                j.initial_state
                    .into_iter()
                    .flatten()
                    .chain(j.final_state.into_iter().flatten())
                    .chain(j.controls.into_iter().flatten())
                    .chain(j.time_grid.into_iter().flatten())
                    .collect()
            },
        );
    } else {
        return Err(PykepError::InvalidInput {
            parameter: "fixture",
            reason: "unknown suite or shape".into(),
        }
        .into());
    }
    Ok(())
}
