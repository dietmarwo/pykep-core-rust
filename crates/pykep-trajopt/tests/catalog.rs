// SPDX-License-Identifier: MPL-2.0

//! Native catalog, launcher and finite-difference Jacobian regression checks.
//!
//! These checks run without the optional PyKEP oracle, so every supported
//! platform exercises the complete TOPS chain rule in CI.

use pykep_trajopt::{GymProblem, Launcher, Problem, TimeEncoding, TopsOptions};

/// Deterministic values in `[0.25, 0.75)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        0.25 + 0.5 * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// An interior point with nonzero thrust directions. Endpoint `vinf`
/// coordinates are fixed at zero by their bounds, so they are moved off the box
/// (accepted for differentiation) to make the endpoint columns observable.
fn interior(problem: &GymProblem, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    problem
        .spec()
        .decisions
        .iter()
        .zip(&problem.spec().bounds)
        .map(|(name, &[low, high])| {
            if name.ends_with("_vinf") {
                0.02
            } else if low == high {
                low
            } else {
                low + (high - low) * rng.next()
            }
        })
        .collect()
}

fn central_difference(problem: &GymProblem, x: &[f64], direction: &[f64], h: f64) -> Vec<f64> {
    let shifted = |sign: f64| {
        let point: Vec<f64> = x
            .iter()
            .zip(direction)
            .map(|(v, d)| v + sign * h * d)
            .collect();
        problem.evaluate(&point).unwrap().fitness()
    };
    let (plus, minus) = (shifted(1.0), shifted(-1.0));
    plus.iter()
        .zip(&minus)
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect()
}

fn assert_directional(problem: &GymProblem, x: &[f64], direction: &[f64], label: &str) {
    let jacobian = problem.gradient(x).unwrap();
    let analytic: Vec<f64> = jacobian
        .iter()
        .map(|row| row.iter().zip(direction).map(|(a, d)| a * d).sum())
        .collect();
    let numerical = central_difference(problem, x, direction, 1e-6);
    for (row, (a, n)) in analytic.iter().zip(&numerical).enumerate() {
        assert!(
            (a - n).abs() <= 2e-5 * n.abs().max(1.0),
            "{label} row {row} ({}): analytic {a:.12e}, central difference {n:.12e}",
            if row == 0 { "objective" } else { "constraint" }
        );
    }
}

#[test]
fn launcher_tables_match_upstream_probes() {
    // (launcher, speed km/s, declination deg, pinned PyKEP 3.1.0 mass kg)
    let probes = [
        ("ariane5", 0.0, -90.0, 0.049_787_068_4),
        ("ariane5", 2.5, 0.0, 5180.0),
        ("ariane5", 3.2, -13.0, 1_407.793_244_489_800_1),
        ("ariane5", 4.17, 37.0, 0.215_788_212_382_400_04),
        ("ariane5", -1.0, 130.0, 5302.0),
        ("ariane5", 9.0, -100.0, 6.144_212_35e-6),
        ("ariane64", 2.5, 0.0, 7700.0),
        ("ariane64", 3.2, -13.0, 6_340.000_000_000_001),
        ("atlas501", 2.5, 0.0, 2335.0),
        ("atlas501", 3.2, -13.0, 2_131.000_000_000_000_5),
        ("atlas501", 4.17, 37.0, 3.700_000_000_000_000_6),
        ("atlas551", 2.5, 0.0, 5_434.710_813_977_31),
        ("atlas551", 3.2, -13.0, 5_036.825_816_588_07),
        ("atlas551", 4.17, 37.0, 4_370.196_148_944_353),
        ("soyuzf", 2.5, 0.0, 2109.7),
        ("soyuzf", 3.2, -13.0, 1_930.285_999_999_999_8),
        ("soyuzf", 4.17, 37.0, 1_579.349_8),
        ("falcon9", 0.0, -90.0, 8300.0),
        ("falcon9", 2.5, 0.0, 6_343.551_350_592_483),
        ("falcon9", 3.2, -13.0, 4_801.077_491_646_922),
        ("falcon9", 4.17, 37.0, 3_251.963_822_051_885),
        ("falcon9", 9.0, -100.0, 0.1),
    ];
    for (name, speed, declination, expected) in probes {
        let launcher = Launcher::ALL
            .into_iter()
            .find(|table| table.id() == name)
            .unwrap();
        let mass = launcher.mass(speed, declination).unwrap();
        assert!(
            (mass - expected).abs() <= 1e-12 * f64::max(1.0, expected),
            "{name}({speed}, {declination}) = {mass:.17e}, expected {expected:.17e}"
        );
    }
}

#[test]
fn every_catalog_problem_evaluates_at_its_midpoint() {
    let ids = GymProblem::ids().unwrap();
    assert_eq!(ids.len(), 82);
    for id in ids {
        let problem = GymProblem::new(
            &id,
            TopsOptions {
                segments: Some(4),
                ..TopsOptions::default()
            },
        )
        .unwrap();
        let spec = problem.spec();
        let evaluation = problem.evaluate(&spec.midpoint()).unwrap();
        assert_eq!(evaluation.objectives.len(), spec.objectives.len(), "{id}");
        assert_eq!(evaluation.equalities.len(), spec.equalities.len(), "{id}");
        assert_eq!(
            evaluation.inequalities.len(),
            spec.inequalities.len(),
            "{id}"
        );
        assert!(
            evaluation.fitness().iter().all(|value| value.is_finite()),
            "{id}"
        );
        assert_eq!(spec.has_gradient, id.starts_with("tops."), "{id}");
    }
}

#[test]
fn tops_jacobians_match_central_differences_for_every_family() {
    for family in ["twobody", "mee", "cr3bp", "sail"] {
        for suffix in ["", ".moving"] {
            for encoding in [TimeEncoding::Uniform, TimeEncoding::Softmax] {
                let id = format!("tops.{family}.P1{suffix}");
                let problem = GymProblem::new(
                    &id,
                    TopsOptions {
                        segments: Some(4),
                        time_encoding: encoding,
                        ..TopsOptions::default()
                    },
                )
                .unwrap();
                let label = format!("{id} {encoding:?}");
                let x = interior(&problem, 7);
                let mut rng = Lcg(11);
                for trial in 0..3 {
                    let direction: Vec<f64> = x
                        .iter()
                        .map(|v| (rng.next() - 0.5) * v.abs().clamp(0.1, 2.0))
                        .collect();
                    assert_directional(&problem, &x, &direction, &format!("{label} d{trial}"));
                }
                // Check each endpoint, flight-time and time-grid column alone.
                let spec = problem.spec();
                for (column, name) in spec.decisions.iter().enumerate() {
                    if name.starts_with("throttle")
                        || name.starts_with("direction_")
                        || name.starts_with("alpha")
                        || name.starts_with("beta")
                    {
                        continue;
                    }
                    let mut direction = vec![0.0; x.len()];
                    direction[column] = 1.0;
                    assert_directional(&problem, &x, &direction, &format!("{label} {name}"));
                }
            }
        }
    }
}
