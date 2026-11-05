// SPDX-License-Identifier: MPL-2.0

use super::*;
use crate::dynamics::{
    BcpDynamics, Cr3bpDynamics, KeplerDynamics, pontryagin::EquinoctialMassOptimal,
    zoh::ZohSolarSailDynamics,
};
use std::{hint::black_box, time::Instant};

fn measure(mut operation: impl FnMut()) -> f64 {
    let mut samples = [0.0; 7];
    operation();
    for elapsed in &mut samples {
        let start = Instant::now();
        for _ in 0..1024 {
            operation();
        }
        *elapsed = start.elapsed().as_secs_f64() * 1e6 / 1024.0;
    }
    samples.sort_by(f64::total_cmp);
    samples[3]
}

fn profile<M: TaylorCoefficientModel<N, P>, const N: usize, const P: usize>(
    model: M,
    state: [f64; N],
    parameters: [f64; P],
) {
    let options = IntegratorOptions::default();
    let order = selected_order(options);
    let mut jet = [[0.0; MAX_ORDER + 1]; N];
    let coefficients = measure(|| {
        model
            .coefficients(
                0.0,
                black_box(&state),
                black_box(&parameters),
                order,
                &mut jet,
            )
            .unwrap();
        black_box(&jet);
    });
    let controller = measure(|| {
        black_box(suggested_step(black_box(&jet), &state, order, options));
    });
    let evaluation = measure(|| {
        black_box(evaluate(black_box(&jet), order, black_box(0.01)));
    });
    println!(
        "{},{order},{coefficients:.6},{controller:.6},{evaluation:.6}",
        M::NAME
    );
}

#[test]
#[ignore = "developer-only release timings for isolated Taylor step stages"]
fn profile_taylor_step_stages() {
    let rotating = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let mu = 0.012_150_585_609_624_04;
    println!("model,order,coefficient_us,controller_us,fused_horner_us");
    profile(KeplerDynamics, [0.8, -0.2, 0.1, 0.03, 1.0, 0.02], [1.0]);
    profile(Cr3bpDynamics, rotating, [mu]);
    profile(
        BcpDynamics,
        rotating,
        [mu, 328_900.56, 389.172, -0.925_195_985_520_347],
    );
    profile(
        ZohSolarSailDynamics,
        [0.8, -0.4, 0.3, 0.2, 0.9, -0.1],
        [0.25, -1.1, 0.04],
    );
    profile(
        EquinoctialMassOptimal,
        [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ],
        [1.0, 1e-4, 1.0, 1.0, 1e-4],
    );
}

#[test]
fn dense_horner_can_recover_from_an_overflowing_product() {
    let mut jet = [[0.0; MAX_ORDER + 1]; 1];
    jet[0][0] = -f64::MAX;
    jet[0][1] = f64::MAX;
    assert_eq!(evaluate_dense(&jet, 1, 2.0), [f64::MAX]);
}
