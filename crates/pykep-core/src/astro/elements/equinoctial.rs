// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/mee2par2mee.cpp and ic2mee2ic.cpp at pykep
// commit 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Modified-equinoctial conversions and analytic Jacobians.
//!
//! Direct Cartesian conversion evaluates the semilatus rectum as `|h|² / mu`,
//! the algebraically equivalent symbolic-upstream form. The pinned numeric C++
//! path obtains it from `a * (1 - e²)`, which can differ in the last bits.

use core::f64::consts::PI;

use super::automatic_differentiation;
use super::{
    ClassicalElements, ModifiedEquinoctialElements, cross, dot, join_state, norm, singular,
    split_state, validate_mu, validate_output, validate_six, validate_state,
};
use crate::{CartesianState, Matrix6, PykepError, Result};

fn direction(retrograde: bool) -> f64 {
    if retrograde { -1.0 } else { 1.0 }
}

fn wrap_positive(mut angle: f64) -> f64 {
    if angle < 0.0 {
        angle += 2.0 * PI;
    } else if angle > 2.0 * PI {
        angle -= 2.0 * PI;
    }
    angle
}

fn validate_mee(elements: ModifiedEquinoctialElements) -> Result<()> {
    validate_six("elements", &elements.to_array())?;
    if elements.semilatus_rectum > 0.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "semilatus_rectum",
            reason: "must be greater than zero".into(),
        })
    }
}

/// Converts classical `[a, e, i, Ω, ω, ν]` elements to modified
/// equinoctial `[p, f, g, h, k, L]` elements.
///
/// Set `retrograde` to select the convention whose nonsingular pole is at
/// inclination `π`. The prograde convention is singular at `π`; the
/// retrograde convention is singular at zero.
///
/// # Errors
///
/// Returns an error for non-finite or inconsistent classical elements, exact
/// parabolic elements, an inclination outside `[0, π]`, the selected
/// convention's singular pole, or overflow.
pub fn classical_to_modified_equinoctial(
    elements: ClassicalElements,
    retrograde: bool,
) -> Result<ModifiedEquinoctialElements> {
    validate_six("elements", &elements.to_array())?;
    let ClassicalElements {
        semi_major_axis,
        eccentricity,
        inclination,
        longitude_ascending_node: node,
        argument_periapsis,
        true_anomaly,
    } = elements;
    if eccentricity < 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "eccentricity",
            reason: "must be non-negative".into(),
        });
    }
    if eccentricity == 1.0 || semi_major_axis == 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "parabolic classical elements require a different representation".into(),
        });
    }
    if (eccentricity < 1.0 && semi_major_axis < 0.0)
        || (eccentricity > 1.0 && semi_major_axis > 0.0)
    {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "the convention requires a > 0 for e < 1 and a < 0 for e > 1".into(),
        });
    }
    if !(0.0..=PI).contains(&inclination) {
        return Err(PykepError::InvalidInput {
            parameter: "inclination",
            reason: "must lie in 0..=pi".into(),
        });
    }
    let half = inclination / 2.0;
    let tangent = if retrograde {
        let sine = half.sin();
        if sine.abs() <= 8.0 * f64::EPSILON {
            return Err(singular("classical_to_modified_equinoctial"));
        }
        half.cos() / sine
    } else {
        let cosine = half.cos();
        if cosine.abs() <= 8.0 * f64::EPSILON {
            return Err(singular("classical_to_modified_equinoctial"));
        }
        half.sin() / cosine
    };
    let orientation = direction(retrograde);
    let eccentricity_longitude = argument_periapsis + orientation * node;
    let values = validate_output(
        "classical_to_modified_equinoctial",
        [
            semi_major_axis * (1.0 - eccentricity * eccentricity),
            eccentricity * eccentricity_longitude.cos(),
            eccentricity * eccentricity_longitude.sin(),
            tangent * node.cos(),
            tangent * node.sin(),
            true_anomaly + argument_periapsis + orientation * node,
        ],
    )?;
    if values[0] <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "must define a positive semilatus rectum".into(),
        });
    }
    Ok(values.into())
}

/// Converts modified equinoctial `[p, f, g, h, k, L]` elements to classical
/// `[a, e, i, Ω, ω, ν]` elements.
///
/// # Errors
///
/// Returns an error for non-finite input, non-positive `p`, circular or
/// equatorial singularities (whose classical angles are undefined), an exact
/// parabola, or overflow.
pub fn modified_equinoctial_to_classical(
    elements: ModifiedEquinoctialElements,
    retrograde: bool,
) -> Result<ClassicalElements> {
    validate_mee(elements)?;
    let eccentricity = (elements.f * elements.f + elements.g * elements.g).sqrt();
    let inclination_parameter = (elements.h * elements.h + elements.k * elements.k).sqrt();
    if eccentricity == 0.0 || inclination_parameter == 0.0 {
        return Err(singular("modified_equinoctial_to_classical"));
    }
    let denominator = 1.0 - eccentricity * eccentricity;
    if denominator == 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "exactly parabolic elements have no finite semi-major axis".into(),
        });
    }
    let orientation = direction(retrograde);
    let mut eccentricity_longitude = elements.g.atan2(elements.f);
    if eccentricity_longitude < 0.0 {
        eccentricity_longitude += 2.0 * PI;
    }
    let inclination = if retrograde {
        PI - 2.0 * inclination_parameter.atan()
    } else {
        2.0 * inclination_parameter.atan()
    };
    let mut node = elements.k.atan2(elements.h);
    if node < 0.0 {
        node += 2.0 * PI;
    }
    let argument_periapsis = wrap_positive(eccentricity_longitude - orientation * node);
    let true_anomaly = elements.true_longitude - orientation * node - argument_periapsis;
    let values = validate_output(
        "modified_equinoctial_to_classical",
        [
            elements.semilatus_rectum / denominator,
            eccentricity,
            inclination,
            node,
            argument_periapsis,
            true_anomaly,
        ],
    )?;
    Ok(values.into())
}

/// Converts a Cartesian state directly to modified equinoctial elements.
///
/// Unlike classical elements, this conversion remains defined for circular
/// and equatorial states away from the selected convention's singular pole.
/// Its semilatus rectum is evaluated as `|h|² / mu` for improved conditioning
/// near the parabolic boundary.
///
/// # Errors
///
/// Returns an error for non-finite input, non-positive `mu`, zero position or
/// angular momentum, the selected prograde/retrograde pole, or overflow.
pub fn cartesian_to_modified_equinoctial(
    state: &CartesianState,
    mu: f64,
    retrograde: bool,
) -> Result<ModifiedEquinoctialElements> {
    validate_state(state)?;
    validate_mu(mu)?;
    let (position, velocity) = split_state(state);
    let radius = norm(&position);
    if radius == 0.0 {
        return Err(singular("cartesian_to_modified_equinoctial"));
    }
    let angular_momentum = cross(&position, &velocity);
    let angular_norm = norm(&angular_momentum);
    if angular_norm == 0.0 {
        return Err(singular("cartesian_to_modified_equinoctial"));
    }
    let unit_angular = [
        angular_momentum[0] / angular_norm,
        angular_momentum[1] / angular_norm,
        angular_momentum[2] / angular_norm,
    ];
    let orientation = direction(retrograde);
    let pole_denominator = 1.0 + orientation * unit_angular[2];
    if pole_denominator.abs() <= 8.0 * f64::EPSILON {
        return Err(singular("cartesian_to_modified_equinoctial"));
    }
    let k = unit_angular[0] / pole_denominator;
    let h = -unit_angular[1] / pole_denominator;
    let frame_denominator = k * k + h * h + 1.0;
    let frame_f = [
        (1.0 - k * k + h * h) / frame_denominator,
        2.0 * k * h / frame_denominator,
        -2.0 * orientation * k / frame_denominator,
    ];
    let frame_g = [
        2.0 * orientation * k * h / frame_denominator,
        orientation * (1.0 + k * k - h * h) / frame_denominator,
        2.0 * h / frame_denominator,
    ];
    let velocity_cross_h = cross(&velocity, &angular_momentum);
    let eccentricity_vector = [
        velocity_cross_h[0] / mu - position[0] / radius,
        velocity_cross_h[1] / mu - position[1] / radius,
        velocity_cross_h[2] / mu - position[2] / radius,
    ];
    let f = dot(&eccentricity_vector, &frame_f);
    let g = dot(&eccentricity_vector, &frame_g);

    let determinants = [
        frame_g[1] * frame_f[0] - frame_f[1] * frame_g[0],
        frame_g[2] * frame_f[0] - frame_f[2] * frame_g[0],
        frame_g[2] * frame_f[1] - frame_f[2] * frame_g[1],
    ];
    let index = determinants
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .abs()
                .partial_cmp(&right.1.abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map_or(0, |(index, _)| index);
    let (x, y) = match index {
        0 => (
            (frame_g[1] * position[0] - frame_g[0] * position[1]) / determinants[0],
            (-frame_f[1] * position[0] + frame_f[0] * position[1]) / determinants[0],
        ),
        1 => (
            (frame_g[2] * position[0] - frame_g[0] * position[2]) / determinants[1],
            (-frame_f[2] * position[0] + frame_f[0] * position[2]) / determinants[1],
        ),
        _ => (
            (frame_g[2] * position[1] - frame_g[1] * position[2]) / determinants[2],
            (-frame_f[2] * position[1] + frame_f[1] * position[2]) / determinants[2],
        ),
    };
    let values = validate_output(
        "cartesian_to_modified_equinoctial",
        [angular_norm * angular_norm / mu, f, g, h, k, y.atan2(x)],
    )?;
    Ok(values.into())
}

/// Converts modified equinoctial elements directly to a Cartesian state.
///
/// # Errors
///
/// Returns an error for non-finite input, non-positive `p` or `mu`, a
/// longitude at or beyond an orbital asymptote, or overflow.
pub fn modified_equinoctial_to_cartesian(
    elements: ModifiedEquinoctialElements,
    mu: f64,
    retrograde: bool,
) -> Result<CartesianState> {
    validate_mee(elements)?;
    validate_mu(mu)?;
    let orientation = direction(retrograde);
    let denominator = 1.0 + elements.k * elements.k + elements.h * elements.h;
    let frame_f = [
        (1.0 - elements.k * elements.k + elements.h * elements.h) / denominator,
        2.0 * elements.k * elements.h / denominator,
        -2.0 * orientation * elements.k / denominator,
    ];
    let frame_g = [
        2.0 * orientation * elements.k * elements.h / denominator,
        orientation * (1.0 + elements.k * elements.k - elements.h * elements.h) / denominator,
        2.0 * elements.h / denominator,
    ];
    let sine_longitude = elements.true_longitude.sin();
    let cosine_longitude = elements.true_longitude.cos();
    let radial_denominator = 1.0 + elements.g * sine_longitude + elements.f * cosine_longitude;
    if radial_denominator <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "true_longitude",
            reason: "lies at or beyond an orbital asymptote".into(),
        });
    }
    let radius = elements.semilatus_rectum / radial_denominator;
    let equinoctial_position = [radius * cosine_longitude, radius * sine_longitude];
    let speed_scale = (mu / elements.semilatus_rectum).sqrt();
    let equinoctial_velocity = [
        -speed_scale * (elements.g + sine_longitude),
        speed_scale * (elements.f + cosine_longitude),
    ];
    let combine = |components: [f64; 2]| {
        [
            components[0].mul_add(frame_f[0], components[1] * frame_g[0]),
            components[0].mul_add(frame_f[1], components[1] * frame_g[1]),
            components[0].mul_add(frame_f[2], components[1] * frame_g[2]),
        ]
    };
    validate_output(
        "modified_equinoctial_to_cartesian",
        join_state(combine(equinoctial_position), combine(equinoctial_velocity)),
    )
}

/// Returns the analytic Jacobian `∂[p,f,g,h,k,L]/∂[x,y,z,vx,vy,vz]`.
///
/// The result is row-major, with output components as rows and Cartesian
/// input components as columns. It is evaluated with forward-mode automatic
/// differentiation of the branch-free upstream expressions.
///
/// # Errors
///
/// Returns the same validation and singularity errors as
/// [`cartesian_to_modified_equinoctial`], plus numerical overflow.
pub fn cartesian_to_modified_equinoctial_jacobian(
    state: &CartesianState,
    mu: f64,
    retrograde: bool,
) -> Result<Matrix6> {
    automatic_differentiation::cartesian_to_modified_equinoctial_jacobian(state, mu, retrograde)
}

/// Returns the analytic Jacobian `∂[x,y,z,vx,vy,vz]/∂[p,f,g,h,k,L]`.
///
/// The result is row-major, with Cartesian output components as rows and
/// element input components as columns.
///
/// # Errors
///
/// Returns the same validation errors as
/// [`modified_equinoctial_to_cartesian`], plus numerical overflow.
pub fn modified_equinoctial_to_cartesian_jacobian(
    elements: ModifiedEquinoctialElements,
    mu: f64,
    retrograde: bool,
) -> Result<Matrix6> {
    automatic_differentiation::modified_equinoctial_to_cartesian_jacobian(elements, mu, retrograde)
}
