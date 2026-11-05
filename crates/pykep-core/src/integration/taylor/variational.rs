// SPDX-License-Identifier: MPL-2.0

use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{kernel::Coefficient, *};

const FALLBACK_WIDTH: usize = 4;
const MAX_SHARED_WIDTH: usize = 16;

#[derive(Clone, Copy)]
struct Tangent<const LANES: usize> {
    value: f64,
    derivative: [f64; LANES],
}

impl<const LANES: usize> Tangent<LANES> {
    fn chain(self, value: f64, slope: f64) -> Self {
        Self {
            value,
            derivative: self.derivative.map(|d| slope * d),
        }
    }

    fn horner<const FUSED: bool>(self, step: f64, coefficient: Self) -> Self {
        Self {
            value: self.value.mul_add(step, coefficient.value),
            derivative: core::array::from_fn(|i| {
                if FUSED {
                    self.derivative[i].mul_add(step, coefficient.derivative[i])
                } else {
                    self.derivative[i] * step + coefficient.derivative[i]
                }
            }),
        }
    }
}

impl<const LANES: usize> Add for Tangent<LANES> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: core::array::from_fn(|i| self.derivative[i] + rhs.derivative[i]),
        }
    }
}

impl<const LANES: usize> Sub for Tangent<LANES> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivative: core::array::from_fn(|i| self.derivative[i] - rhs.derivative[i]),
        }
    }
}

impl<const LANES: usize> Mul for Tangent<LANES> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: core::array::from_fn(|i| {
                self.derivative[i] * rhs.value + self.value * rhs.derivative[i]
            }),
        }
    }
}

impl<const LANES: usize> Div for Tangent<LANES> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        Self {
            value,
            derivative: core::array::from_fn(|i| {
                (self.derivative[i] - value * rhs.derivative[i]) / rhs.value
            }),
        }
    }
}

impl<const LANES: usize> Neg for Tangent<LANES> {
    type Output = Self;
    fn neg(self) -> Self {
        self.chain(-self.value, -1.0)
    }
}

impl<const LANES: usize> Coefficient for Tangent<LANES> {
    const ZERO: Self = Self {
        value: 0.0,
        derivative: [0.0; LANES],
    };
    fn primal(self) -> f64 {
        self.value
    }
    fn splat(value: f64) -> Self {
        Self {
            value,
            ..Self::ZERO
        }
    }
    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        self.chain(value, exponent * value / self.value)
    }
    fn exp(self) -> Self {
        let value = self.value.exp();
        self.chain(value, value)
    }
    fn sin(self) -> Self {
        let (sin, cos) = self.value.sin_cos();
        self.chain(sin, cos)
    }
    fn cos(self) -> Self {
        let (sin, cos) = self.value.sin_cos();
        self.chain(cos, -sin)
    }
}

fn pack<const N: usize, const W: usize, const LANES: usize>(
    value: &[f64; N],
    derivative: &[[f64; W]; N],
    offset: usize,
) -> [Tangent<LANES>; N] {
    core::array::from_fn(|row| Tangent {
        value: value[row],
        derivative: core::array::from_fn(|lane| {
            derivative[row].get(offset + lane).copied().unwrap_or(0.0)
        }),
    })
}

pub(super) fn propagate<M, const N: usize, const P: usize, const W: usize>(
    model: &M,
    problem: SensitivityProblem<N, P, W>,
    options: IntegratorOptions,
) -> Result<SensitivityPropagation<N, W>>
where
    M: TaylorCoefficientModel<N, P>,
{
    // Common shooting and ZOH widths fit in one dual jet. This computes every
    // nominal graph coefficient once, alongside all its seeded derivatives.
    // Larger caller-defined widths retain bounded-size kernel scratch.
    if W <= MAX_SHARED_WIDTH {
        propagate_blocks::<M, N, P, W, W>(model, problem, options)
    } else {
        propagate_blocks::<M, N, P, W, FALLBACK_WIDTH>(model, problem, options)
    }
}

fn propagate_blocks<M, const N: usize, const P: usize, const W: usize, const LANES: usize>(
    model: &M,
    problem: SensitivityProblem<N, P, W>,
    options: IntegratorOptions,
) -> Result<SensitivityPropagation<N, W>>
where
    M: TaylorCoefficientModel<N, P>,
{
    let nominal = problem.nominal;
    let mut time = nominal.initial_time;
    let mut state = nominal.initial_state;
    let mut sensitivities = problem.initial_sensitivities;
    let mut stats = IntegrationStats::default();
    let order = selected_order(options);
    let mut inline = [[Tangent::<LANES>::ZERO; MAX_ORDER + 1]; N];
    let mut overflow = Vec::new();
    let blocks = W.div_ceil(LANES);
    let jets = if blocks == 1 {
        std::slice::from_mut(&mut inline)
    } else {
        overflow
            .try_reserve_exact(blocks)
            .map_err(|_| PykepError::NumericalOverflow {
                operation: "Taylor sensitivity workspace",
            })?;
        overflow.resize(blocks, inline);
        overflow.as_mut_slice()
    };

    while time != nominal.final_time {
        if stats.accepted_steps >= options.maximum_steps {
            return integration_failure(M::NAME, "maximum step count exceeded");
        }
        let mut tails = [0.0_f64; 2];
        for (block, jet) in jets.iter_mut().enumerate() {
            let offset = block * LANES;
            model.generated_coefficients(
                Tangent::splat(time),
                &pack(&state, &sensitivities, offset),
                &pack(&nominal.parameters, &problem.parameter_seeds, offset),
                order,
                jet,
            );
            for row in jet.iter() {
                for (i, norm) in tails.iter_mut().enumerate() {
                    let coefficient = row[order - 1 + i];
                    *norm = norm.max(scaled_coefficient(
                        coefficient.value,
                        row[0].value,
                        options,
                    )?);
                    for lane in 0..LANES {
                        *norm = norm.max(scaled_coefficient(
                            coefficient.derivative[lane],
                            row[0].derivative[lane],
                            options,
                        )?);
                    }
                }
            }
        }
        let remaining = nominal.final_time - time;
        let step = select_step(
            M::NAME,
            remaining,
            time,
            SAFETY_FACTOR * step_from_norms(tails, order),
            options,
            &stats,
        )?;
        for (block, jet) in jets.iter().enumerate() {
            for (row, coefficients) in jet.iter().enumerate() {
                let evaluate = |fused| {
                    let coefficients = coefficients[..=order].iter().rev();
                    if fused {
                        coefficients.fold(Tangent::ZERO, |acc, &c| acc.horner::<true>(step, c))
                    } else {
                        coefficients.fold(Tangent::ZERO, |acc, &c| acc.horner::<false>(step, c))
                    }
                };
                let mut result = evaluate(!BASELINE_X86);
                if BASELINE_X86 && result.derivative.iter().any(|value| !value.is_finite()) {
                    result = evaluate(true);
                }
                if block == 0 {
                    state[row] = result.value;
                }
                for (dst, src) in sensitivities[row][block * LANES..]
                    .iter_mut()
                    .zip(result.derivative)
                {
                    *dst = src;
                }
            }
        }
        time = if step == remaining {
            nominal.final_time
        } else {
            time + step
        };
        ensure_finite_state(M::NAME, &state)?;
        ensure_finite_matrix("sensitivities", &sensitivities)?;
        model.validate(time, &state, &nominal.parameters)?;
        stats.accepted_steps += 1;
        stats.rhs_evaluations = stats
            .rhs_evaluations
            .saturating_add(order.saturating_mul(jets.len()));
    }
    Ok(SensitivityPropagation {
        time,
        state,
        sensitivities,
        stats,
    })
}
