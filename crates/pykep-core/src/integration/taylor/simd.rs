// SPDX-License-Identifier: MPL-2.0

use super::{kernel::Coefficient, *};
use wide::f64x4;

impl Coefficient for f64x4 {
    const ZERO: Self = Self::new([0.0; 4]);

    fn primal(self) -> f64 {
        self.as_array_ref()[0]
    }

    #[inline]
    fn splat(value: f64) -> Self {
        Self::new([value; 4])
    }

    #[inline]
    fn powf(self, exponent: f64) -> Self {
        Self::new(self.to_array().map(|x| x.powf(exponent)))
    }

    #[inline]
    fn exp(self) -> Self {
        Self::new(self.to_array().map(f64::exp))
    }

    #[inline]
    fn sin(self) -> Self {
        Self::new(self.to_array().map(f64::sin))
    }

    #[inline]
    fn cos(self) -> Self {
        Self::new(self.to_array().map(f64::cos))
    }
}

pub(super) fn propagate_group<M, const N: usize, const P: usize>(
    model: &M,
    jobs: &[InitialValueProblem<N, P>; 4],
    options: IntegratorOptions,
) -> Result<[Propagation<N>; 4]>
where
    M: TaylorCoefficientModel<N, P>,
{
    let order = selected_order(options);
    let mut times = jobs.map(|job| job.initial_time);
    let mut states = jobs.map(|job| job.initial_state);
    let mut stats = [IntegrationStats::default(); 4];
    let mut errors = jobs
        .each_ref()
        .map(|job| validate_problem(model, job, options).err());
    let mut jet = [[f64x4::ZERO; MAX_ORDER + 1]; N];
    loop {
        let active = core::array::from_fn::<_, 4, _>(|i| {
            errors[i].is_none() && times[i] != jobs[i].final_time
        });
        let Some(first) = active.iter().position(|&value| value) else {
            break;
        };
        let source = core::array::from_fn(|i| if active[i] { i } else { first });
        let time = f64x4::new(source.map(|i| times[i]));
        let state = core::array::from_fn(|row| f64x4::new(source.map(|i| states[i][row])));
        let params =
            core::array::from_fn(|row| f64x4::new(source.map(|i| jobs[i].parameters[row])));
        model.generated_coefficients(time, &state, &params, order, &mut jet);
        let mut steps = [0.0; 4];
        for (i, step) in steps.iter_mut().enumerate() {
            if !active[i] {
                continue;
            }
            let suggestion =
                suggested_step_from_coefficients(&states[i], order, options, |row, n| {
                    jet[row][n].as_array_ref()[i]
                });
            match select_step(
                M::NAME,
                jobs[i].final_time - times[i],
                times[i],
                M::NOMINAL_STEP_SAFETY * suggestion,
                options,
                &stats[i],
            ) {
                Ok(value) => *step = value,
                Err(error) => errors[i] = Some(error),
            }
        }
        let step = f64x4::new(steps);
        let next = jet.each_ref().map(|coefficients| {
            coefficients[..=order]
                .iter()
                .rev()
                .fold(f64x4::ZERO, |value, &c| value.mul_add(step, c))
        });
        for (i, &step) in steps.iter().enumerate() {
            if step == 0.0 {
                continue;
            }
            let mut state = next.each_ref().map(|row| row.as_array_ref()[i]);
            if !state.iter().all(|value| value.is_finite()) {
                // A separate multiply can overflow where a fused operation stays finite.
                state = jet.each_ref().map(|row| {
                    row[..=order]
                        .iter()
                        .rev()
                        .fold(0.0_f64, |value, c| value.mul_add(step, c.as_array_ref()[i]))
                });
            }
            match accept_state(
                model,
                &jobs[i].parameters,
                times[i],
                jobs[i].final_time,
                step,
                &state,
            ) {
                Ok(time) => {
                    times[i] = time;
                    states[i] = state;
                    stats[i].rhs_evaluations += order;
                    stats[i].accepted_steps += 1;
                }
                Err(error) => errors[i] = Some(error),
            }
        }
    }
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(error);
    }
    Ok(core::array::from_fn(|i| Propagation {
        state: states[i],
        time: times[i],
        stats: stats[i],
        termination: Termination::FinalTime,
    }))
}
