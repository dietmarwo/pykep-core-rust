// SPDX-License-Identifier: MPL-2.0

use super::{kernel::product, *};
use crate::integration::{EventDirection, TaylorEvent};

fn invalid(reason: &'static str) -> PykepError {
    PykepError::InvalidInput {
        parameter: "event",
        reason: reason.into(),
    }
}

fn value(coefficients: &[f64], time: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &c| acc.mul_add(time, c))
}

fn derivative(coefficients: &[f64]) -> Vec<f64> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| i as f64 * c)
        .collect()
}

// The derivative roots partition the interval into monotone pieces. Degree
// is bounded by twice MAX_ORDER, and each bracket has a fixed work limit.
fn roots(coefficients: &[f64], model: &'static str, event_level: bool) -> Result<Vec<f64>> {
    let len = coefficients
        .iter()
        .rposition(|&c| c != 0.0)
        .map_or(0, |i| i + 1);
    let coefficients = &coefficients[..len];
    match len {
        0 if event_level => return integration_failure(model, "event is zero throughout a step"),
        0 | 1 => return Ok(Vec::new()),
        2 => {
            let root = -coefficients[0] / coefficients[1];
            return Ok(if (0.0..=1.0).contains(&root) {
                vec![root]
            } else {
                Vec::new()
            });
        }
        _ => {}
    }
    let critical = roots(&derivative(coefficients), model, false)?;
    let scale: f64 = coefficients.iter().map(|x| x.abs()).sum();
    let mut result = Vec::new();
    let mut left = 0.0_f64;
    let mut left_value = coefficients[0];
    for right in critical
        .into_iter()
        .filter(|&u| u > 0.0 && u < 1.0)
        .chain([1.0])
    {
        let right_value = value(coefficients, right);
        if right_value == 0.0 {
            result.push(right);
        } else if event_level && right < 1.0 && right_value.abs() <= 32.0 * f64::EPSILON * scale {
            return integration_failure(model, "ambiguous Taylor event near a stationary point");
        } else if left_value != 0.0
            && left_value.is_sign_positive() != right_value.is_sign_positive()
        {
            let mut lo = left;
            let mut hi = right;
            for _ in 0..64 {
                let mid = lo.midpoint(hi);
                if mid == lo || mid == hi {
                    break;
                }
                let middle = value(coefficients, mid);
                if middle == 0.0 {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if middle.is_sign_positive() == left_value.is_sign_positive() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            result.push(lo.midpoint(hi));
        }
        left = right;
        left_value = right_value;
    }
    Ok(result)
}

fn polynomial<const N: usize>(
    jet: &[[f64; MAX_ORDER + 1]; N],
    order: usize,
    step: f64,
    event: TaylorEvent,
    model: &'static str,
) -> Result<Vec<f64>> {
    let mut scaled = [[0.0; MAX_ORDER + 1]; 6];
    for (row, input) in scaled.iter_mut().zip(jet) {
        let mut power = 1.0;
        for (value, &coefficient) in row[..=order].iter_mut().zip(input) {
            *value = if coefficient == 0.0 {
                0.0
            } else {
                coefficient * power
            };
            power *= step;
        }
    }
    let mut coefficients = match event {
        TaylorEvent::Plane { normal, offset } => {
            let mut c: Vec<_> = (0..=order)
                .map(|n| (0..3).map(|i| normal[i] * scaled[i][n]).sum::<f64>())
                .collect();
            c[0] -= offset;
            c
        }
        TaylorEvent::Sphere { center, radius } => {
            for (row, origin) in scaled.iter_mut().zip(center) {
                row[0] -= origin;
            }
            let mut c = dot_polynomials(&scaled, order, 0);
            c[0] -= radius * radius;
            c
        }
        TaylorEvent::RadialVelocity => dot_polynomials(&scaled, order, 3),
    };
    let scale = coefficients
        .iter()
        .try_fold(0.0_f64, |norm, &x| x.is_finite().then(|| norm.max(x.abs())))
        .ok_or(PykepError::NumericalOverflow { operation: model })?;
    if scale > 0.0 {
        coefficients.iter_mut().for_each(|x| *x /= scale);
    }
    Ok(coefficients)
}

fn dot_polynomials(scaled: &[[f64; MAX_ORDER + 1]; 6], order: usize, offset: usize) -> Vec<f64> {
    (0..=2 * order)
        .map(|n| {
            (0..3)
                .map(|i| {
                    if n <= order {
                        product(&scaled[i], &scaled[i + offset], n)
                    } else {
                        (n - order..=order)
                            .map(|k| scaled[i][k] * scaled[i + offset][n - k])
                            .sum()
                    }
                })
                .sum()
        })
        .collect()
}

pub(crate) fn propagate<M, const N: usize, const P: usize>(
    model: &M,
    problem: InitialValueProblem<N, P>,
    options: IntegratorOptions,
    event: TaylorEvent,
    direction: EventDirection,
    mut observe: impl FnMut(f64, [f64; N]) -> Result<bool>,
) -> Result<Propagation<N>>
where
    M: TaylorCoefficientModel<N, P>,
{
    validate_problem(model, &problem, options)?;
    if !M::CARTESIAN_EVENTS {
        return Err(invalid("Taylor events require a built-in Cartesian model"));
    }
    match event {
        TaylorEvent::Plane { normal, offset }
            if !normal.iter().all(|x| x.is_finite())
                || normal == [0.0; 3]
                || !offset.is_finite() =>
        {
            return Err(invalid(
                "plane requires a finite nonzero normal and finite offset",
            ));
        }
        TaylorEvent::Sphere { center, radius }
            if !center.iter().all(|x| x.is_finite()) || !radius.is_finite() || radius <= 0.0 =>
        {
            return Err(invalid(
                "sphere requires a finite center and positive finite radius",
            ));
        }
        _ => {}
    }
    let mut workspace = Workspace::new(problem.initial_time, problem.initial_state);
    let mut stats = IntegrationStats::default();
    let order = selected_order(options);
    let mut last_event = None;
    while workspace.time != problem.final_time {
        let start = workspace.time;
        let step = advance(
            model,
            &problem.parameters,
            problem.final_time,
            options,
            order,
            &mut workspace,
            &mut stats,
        )?;
        let coefficients = polynomial(&workspace.jet, order, step, event, M::NAME)?;
        let slope = derivative(&coefficients);
        for root in roots(&coefficients, M::NAME, true)? {
            if root <= 0.0 {
                continue;
            }
            let crossing = value(&slope, root);
            if match direction {
                EventDirection::Either => true,
                EventDirection::Increasing => crossing > 0.0,
                EventDirection::Decreasing => crossing < 0.0,
            } {
                let offset = root * step;
                let time = if root == 1.0 {
                    workspace.time
                } else {
                    start + offset
                };
                if last_event == Some(time) {
                    continue;
                }
                // A root within rounding of the step start is either the excluded
                // initial event, for example after restarting from a returned
                // event state, or the event already reported at the end of the
                // previous step.
                let negligible = 4.0 * f64::EPSILON * start.abs().max(step.abs());
                if offset.abs() <= negligible
                    && (start == problem.initial_time
                        || last_event
                            .is_some_and(|previous| (previous - start).abs() <= negligible))
                {
                    continue;
                }
                let state = evaluate(&workspace.jet, order, offset);
                ensure_finite_state(M::NAME, &state)?;
                model.validate(time, &state, &problem.parameters)?;
                last_event = Some(time);
                if observe(time, state)? {
                    return Ok(Propagation {
                        time,
                        state,
                        stats,
                        termination: Termination::Event,
                    });
                }
            }
        }
    }
    Ok(Propagation {
        time: workspace.time,
        state: workspace.state,
        stats,
        termination: Termination::FinalTime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolates_two_crossings_despite_same_endpoint_sign() {
        let roots = roots(&[0.1875, -1.0, 1.0], "test", true).unwrap();
        assert_eq!(roots, [0.25, 0.75]);
    }
}
