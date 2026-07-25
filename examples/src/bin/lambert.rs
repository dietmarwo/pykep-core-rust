// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Solve a dimensionless multi-revolution Lambert transfer.

use pykep_core::astro::lambert::LambertProblem;

fn main() -> Result<(), pykep_core::PykepError> {
    let problem = LambertProblem::new([1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, false, 4)?;
    println!("{} ordered solutions", problem.solutions().len());
    for solution in problem.solutions() {
        println!(
            "{} rev {:?}: departure {:?}",
            solution.revolutions, solution.path, solution.departure_velocity
        );
    }
    Ok(())
}
