// SPDX-License-Identifier: MPL-2.0

//! Moving-CR3BP stress vectors with close lunar passes.
//!
//! These are the two decisions that exceeded the strict source-comparison
//! gate. Both pass close to the Moon, where the transition matrices are
//! ill-conditioned.

use pykep_core::integration::IntegrationMethod;
use pykep_trajopt::{GymProblem, Problem, TopsOptions};

const P1: [f64; 75] = [
    0.8257805455719971,
    0.75,
    0.0,
    0.8,
    0.3,
    -0.25,
    0.0,
    0.8,
    0.3,
    -0.25,
    0.261225138050941,
    -0.2767905133641079,
    0.7371332803623483,
    -0.1414974865284799,
    0.23307943012017104,
    0.7280369465049267,
    0.22820273273576436,
    0.24059450803314683,
    0.2710645425906931,
    0.7596648804708392,
    -0.09798534907621684,
    0.23088555768093733,
    0.22876655630328968,
    -0.5171464019709985,
    0.4895322331902808,
    0.3645789236861987,
    0.23553173696956398,
    -0.30604124492013374,
    -0.4133700556982179,
    0.6127511350130549,
    0.21455788114986038,
    0.24451785640891877,
    -0.22386902429179067,
    -0.7280753243036252,
    0.274240377442102,
    0.7458454232009415,
    -0.1778000878801965,
    0.22825804134795113,
    0.22932915704358947,
    -0.7691607816577813,
    0.13057211081641054,
    -0.1770384586369243,
    0.2575880736571453,
    -0.3710874620614475,
    -0.6648905755469184,
    0.24538667048085797,
    0.2342046685200871,
    -0.2900272402309815,
    0.4698648058154802,
    0.57888795477189,
    0.2364565625696554,
    0.5512971814057692,
    -0.4316137178514448,
    -0.38701552467117906,
    0.20437442809177775,
    0.35079042758459084,
    0.3211957062925197,
    0.6432568648481487,
    0.24640218299835648,
    -0.2941523417899265,
    -0.4304490016348932,
    0.6067850169631961,
    0.20978461671507992,
    0.6246776050238072,
    0.46845316568462203,
    -0.17415372904929832,
    0.22502693009159905,
    0.4961588933796904,
    -0.5123220040369688,
    0.3624258775250262,
    0.2976525602280446,
    0.3924708580515821,
    -0.48119259746611853,
    -0.5044009414385197,
    8.5,
];

const P13: [f64; 75] = [
    0.925686725,
    0.75,
    0.0,
    0.8,
    0.3,
    -0.25,
    0.0,
    0.8,
    0.3,
    -0.25,
    0.261225138050941,
    -0.2767905133641079,
    0.7371332803623483,
    -0.1414974865284799,
    0.23307943012017104,
    0.7280369465049267,
    0.22820273273576436,
    0.24059450803314683,
    0.2710645425906931,
    0.7596648804708392,
    -0.09798534907621684,
    0.23088555768093733,
    0.22876655630328968,
    -0.5171464019709985,
    0.4895322331902808,
    0.3645789236861987,
    0.23553173696956398,
    -0.30604124492013374,
    -0.4133700556982179,
    0.6127511350130549,
    0.21455788114986038,
    0.24451785640891877,
    -0.22386902429179067,
    -0.7280753243036252,
    0.274240377442102,
    0.7458454232009415,
    -0.1778000878801965,
    0.22825804134795113,
    0.22932915704358947,
    -0.7691607816577813,
    0.13057211081641054,
    -0.1770384586369243,
    0.2575880736571453,
    -0.3710874620614475,
    -0.6648905755469184,
    0.24538667048085797,
    0.2342046685200871,
    -0.2900272402309815,
    0.4698648058154802,
    0.57888795477189,
    0.2364565625696554,
    0.5512971814057692,
    -0.4316137178514448,
    -0.38701552467117906,
    0.20437442809177775,
    0.35079042758459084,
    0.3211957062925197,
    0.6432568648481487,
    0.24640218299835648,
    -0.2941523417899265,
    -0.4304490016348932,
    0.6067850169631961,
    0.20978461671507992,
    0.6246776050238072,
    0.46845316568462203,
    -0.17415372904929832,
    0.22502693009159905,
    0.4961588933796904,
    -0.5123220040369688,
    0.3624258775250262,
    0.2976525602280446,
    0.3924708580515821,
    -0.48119259746611853,
    -0.5044009414385197,
    15.349316929558674,
];

fn problem(id: &str, method: IntegrationMethod, tolerance: f64) -> GymProblem {
    GymProblem::new(
        id,
        TopsOptions {
            segments: Some(16),
            method,
            tolerance,
            derivative_tolerance: tolerance,
            ..TopsOptions::default()
        },
    )
    .unwrap()
}

fn scaled_difference(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// With centered-difference model Jacobians, the DOP853 gradient differed
/// from the Taylor gradient by a scaled 50 (P1) and 477 (P13) on these
/// vectors. Analytic CR3BP model Jacobians bring both within the
/// conditioning spread of the Taylor gradient itself.
#[test]
fn dop853_and_taylor_gradients_agree_through_close_lunar_passes() {
    for (id, x) in [("tops.cr3bp.P1.moving", P1), ("tops.cr3bp.P13.moving", P13)] {
        let taylor = problem(id, IntegrationMethod::Taylor, 1e-13)
            .gradient(&x)
            .unwrap();
        let dop853 = problem(id, IntegrationMethod::Dop853, 1e-13)
            .gradient(&x)
            .unwrap();
        let difference = scaled_difference(&dop853, &taylor);
        assert!(difference < 2e-4, "{id}: scaled difference {difference:e}");
    }
}

/// Prints the Taylor gradient spread across tolerances, which bounds how
/// closely any double-precision reference can agree on these vectors.
#[test]
#[ignore = "diagnostic: cargo test --release -p pykep-trajopt --test cr3bp_conditioning -- --ignored --nocapture"]
fn taylor_gradient_tolerance_spread() {
    for (id, x) in [("tops.cr3bp.P1.moving", P1), ("tops.cr3bp.P13.moving", P13)] {
        let reference = problem(id, IntegrationMethod::Taylor, 1e-16)
            .gradient(&x)
            .unwrap();
        for tolerance in [1e-12, 1e-13, 1e-14, 1e-15] {
            let gradient = problem(id, IntegrationMethod::Taylor, tolerance)
                .gradient(&x)
                .unwrap();
            println!(
                "{id} tolerance {tolerance:e}: scaled difference from 1e-16 {:.2e}",
                scaled_difference(&gradient, &reference)
            );
        }
    }
}
