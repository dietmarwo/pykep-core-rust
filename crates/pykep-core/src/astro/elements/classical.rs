// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/ic2par2ic.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

use core::f64::consts::PI;

use super::{
    ClassicalElements, cross, dot, join_state, norm, singular, split_state, validate_mu,
    validate_output, validate_six, validate_state,
};
use crate::{CartesianState, PykepError, Result};

fn validate_classical_shape(elements: ClassicalElements) -> Result<()> {
    validate_six("elements", &elements.to_array())?;
    let a = elements.semi_major_axis;
    let eccentricity = elements.eccentricity;
    if eccentricity < 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "eccentricity",
            reason: "must be non-negative".into(),
        });
    }
    if eccentricity == 1.0 || a == 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "parabolic classical elements require a different representation".into(),
        });
    }
    if (eccentricity < 1.0 && a < 0.0) || (eccentricity > 1.0 && a > 0.0) {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "the convention requires a > 0 for e < 1 and a < 0 for e > 1".into(),
        });
    }
    Ok(())
}

fn validate_classical(elements: ClassicalElements) -> Result<()> {
    validate_classical_shape(elements)?;
    if !(0.0..=PI).contains(&elements.inclination) {
        return Err(PykepError::InvalidInput {
            parameter: "inclination",
            reason: "must lie in 0..=pi".into(),
        });
    }
    Ok(())
}

/// Converts a Cartesian state to classical elements `[a, e, i, Ω, ω, ν]`.
///
/// Position, velocity, and `mu` must use a consistent unit system. The three
/// orientation angles are returned in `[0, 2π)` and inclination in `[0, π]`.
///
/// # Errors
///
/// Returns an error for non-finite input, non-positive `mu`, zero
/// position/angular momentum, a circular orbit, an equatorial orbit, an exact
/// parabola, or numerical overflow. Classical `Ω`, `ω`, and `ν` are undefined
/// for circular or equatorial singularities; use modified equinoctial
/// elements there.
pub fn cartesian_to_classical(state: &CartesianState, mu: f64) -> Result<ClassicalElements> {
    validate_state(state)?;
    validate_mu(mu)?;
    let (position, velocity) = split_state(state);
    let radius = norm(&position);
    if radius == 0.0 {
        return Err(singular("cartesian_to_classical"));
    }
    let angular_momentum = cross(&position, &velocity);
    let h_norm = norm(&angular_momentum);
    if h_norm == 0.0 {
        return Err(singular("cartesian_to_classical"));
    }
    let node = [-angular_momentum[1], angular_momentum[0], 0.0];
    let node_norm = norm(&node);
    if node_norm == 0.0 {
        return Err(singular("cartesian_to_classical"));
    }
    let node_unit = [
        node[0] / node_norm,
        node[1] / node_norm,
        node[2] / node_norm,
    ];
    let velocity_cross_h = cross(&velocity, &angular_momentum);
    let eccentricity_vector = [
        velocity_cross_h[0] / mu - position[0] / radius,
        velocity_cross_h[1] / mu - position[1] / radius,
        velocity_cross_h[2] / mu - position[2] / radius,
    ];
    let eccentricity = norm(&eccentricity_vector);
    if eccentricity == 0.0 {
        return Err(singular("cartesian_to_classical"));
    }
    let denominator = 1.0 - eccentricity * eccentricity;
    if denominator == 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "state",
            reason: "exactly parabolic states have no finite semi-major axis".into(),
        });
    }
    let semilatus_rectum = h_norm * h_norm / mu;
    let semi_major_axis = semilatus_rectum / denominator;
    let inclination = (angular_momentum[2] / h_norm).clamp(-1.0, 1.0).acos();

    let mut argument_periapsis = (dot(&node_unit, &eccentricity_vector) / eccentricity)
        .clamp(-1.0, 1.0)
        .acos();
    if eccentricity_vector[2] < 0.0 {
        argument_periapsis = 2.0 * PI - argument_periapsis;
    }
    let mut longitude_ascending_node = node_unit[0].clamp(-1.0, 1.0).acos();
    if node_unit[1] < 0.0 {
        longitude_ascending_node = 2.0 * PI - longitude_ascending_node;
    }
    let cosine_true_anomaly = dot(&eccentricity_vector, &position) / (eccentricity * radius);
    let sine_true_anomaly = dot(&position, &velocity) * h_norm / (eccentricity * radius * mu);
    let mut true_anomaly = sine_true_anomaly.atan2(cosine_true_anomaly);
    if true_anomaly < 0.0 {
        true_anomaly += 2.0 * PI;
    }

    let values = validate_output(
        "cartesian_to_classical",
        [
            semi_major_axis,
            eccentricity,
            inclination,
            longitude_ascending_node,
            argument_periapsis,
            true_anomaly,
        ],
    )?;
    Ok(values.into())
}

/// Converts classical elements `[a, e, i, Ω, ω, ν]` to a Cartesian state.
///
/// Position uses the unit of `a`; velocity uses the corresponding unit per
/// time implied by `mu`.
///
/// # Errors
///
/// Returns an error for non-finite input, non-positive `mu`, inconsistent
/// ellipse/hyperbola signs, exact parabolic elements, inclination outside
/// `[0, π]`, a hyperbolic anomaly beyond an asymptote, or overflow.
pub fn classical_to_cartesian(elements: ClassicalElements, mu: f64) -> Result<CartesianState> {
    validate_classical(elements)?;
    classical_to_cartesian_validated(elements, mu)
}

pub(crate) fn classical_to_cartesian_unbounded_inclination(
    elements: ClassicalElements,
    mu: f64,
) -> Result<CartesianState> {
    validate_classical_shape(elements)?;
    classical_to_cartesian_validated(elements, mu)
}

fn classical_to_cartesian_validated(
    elements: ClassicalElements,
    mu: f64,
) -> Result<CartesianState> {
    validate_mu(mu)?;
    let ClassicalElements {
        semi_major_axis: a,
        eccentricity,
        inclination,
        longitude_ascending_node: node,
        argument_periapsis: periapsis,
        true_anomaly,
    } = elements;

    let cosine_true = true_anomaly.cos();
    let radial_denominator = 1.0 + eccentricity * cosine_true;
    if eccentricity > 1.0 && radial_denominator <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "true_anomaly",
            reason: "lies at or beyond a hyperbolic asymptote".into(),
        });
    }
    let semilatus_rectum = a * (1.0 - eccentricity * eccentricity);
    if semilatus_rectum <= 0.0 || !semilatus_rectum.is_finite() {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "must define a positive finite semilatus rectum".into(),
        });
    }
    let radius = semilatus_rectum / radial_denominator;
    let angular_momentum = (semilatus_rectum * mu).sqrt();
    let sine_true = true_anomaly.sin();
    let perifocal_position = [radius * cosine_true, radius * sine_true, 0.0];
    let perifocal_velocity = [
        -mu / angular_momentum * sine_true,
        mu / angular_momentum * (eccentricity + cosine_true),
        0.0,
    ];

    let (cosine_node, sine_node) = (node.cos(), node.sin());
    let (cosine_periapsis, sine_periapsis) = (periapsis.cos(), periapsis.sin());
    let (cosine_inclination, sine_inclination) = (inclination.cos(), inclination.sin());
    let rotation = [
        [
            cosine_node * cosine_periapsis - sine_node * sine_periapsis * cosine_inclination,
            -cosine_node * sine_periapsis - sine_node * cosine_periapsis * cosine_inclination,
            sine_node * sine_inclination,
        ],
        [
            sine_node * cosine_periapsis + cosine_node * sine_periapsis * cosine_inclination,
            -sine_node * sine_periapsis + cosine_node * cosine_periapsis * cosine_inclination,
            -cosine_node * sine_inclination,
        ],
        [
            sine_periapsis * sine_inclination,
            cosine_periapsis * sine_inclination,
            cosine_inclination,
        ],
    ];
    let rotate = |vector: [f64; 3]| {
        [
            dot(&rotation[0], &vector),
            dot(&rotation[1], &vector),
            dot(&rotation[2], &vector),
        ]
    };
    validate_output(
        "classical_to_cartesian",
        join_state(rotate(perifocal_position), rotate(perifocal_velocity)),
    )
}
