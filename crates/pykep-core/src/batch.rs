// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Shared ordered parallel-batch execution.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::{PykepError, Result};

const MAX_EXPLICIT_WORKERS: usize = 1024;

fn worker_pool(workers: usize) -> Result<Arc<ThreadPool>> {
    if workers > MAX_EXPLICIT_WORKERS {
        return Err(PykepError::InvalidInput {
            parameter: "workers",
            reason: format!("must not exceed {MAX_EXPLICIT_WORKERS}"),
        });
    }
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools = pools
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(pool) = pools.get(&workers) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(move |index| format!("pykep-batch-{workers}-{index}"))
            .build()
            .map_err(|error| PykepError::InvalidInput {
                parameter: "workers",
                reason: format!("could not create the worker pool: {error}"),
            })?,
    );
    pools.insert(workers, Arc::clone(&pool));
    Ok(pool)
}

/// Apply a fallible operation to an ordered batch.
///
/// Zero workers uses Rayon's shared global pool, one worker executes serially,
/// and larger values use a cached pool with exactly that many threads.
///
/// # Errors
///
/// Returns an error when `workers` exceeds 1024, an explicit worker pool
/// cannot be created, or an item operation fails. Item failures are reported
/// in input order.
pub fn try_map<T, R>(
    items: &[T],
    workers: usize,
    operation: impl Fn(&T) -> Result<R> + Sync + Send,
) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
{
    if workers > MAX_EXPLICIT_WORKERS {
        return Err(PykepError::InvalidInput {
            parameter: "workers",
            reason: format!("must not exceed {MAX_EXPLICIT_WORKERS}"),
        });
    }
    let results: Vec<Result<R>> = if workers == 1 || items.len() <= 1 {
        items.iter().map(operation).collect()
    } else if workers == 0 {
        items.par_iter().map(operation).collect()
    } else {
        worker_pool(workers)?.install(|| items.par_iter().map(operation).collect())
    };
    // Collect after the parallel work so a failure is reported in input order.
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_preserves_values_and_input_order() {
        let input: Vec<_> = (0..64).collect();
        for workers in [0, 1, 2] {
            let output = try_map(&input, workers, |value| Ok(value * value)).unwrap();
            assert_eq!(
                output,
                input.iter().map(|value| value * value).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn errors_and_worker_limits_are_deterministic() {
        let input: Vec<_> = (0..16).collect();
        let error = try_map(&input, 2, |value| {
            if *value == 3 || *value == 7 {
                Err(PykepError::InvalidInput {
                    parameter: "item",
                    reason: value.to_string(),
                })
            } else {
                Ok(*value)
            }
        })
        .unwrap_err();
        assert_eq!(
            error,
            PykepError::InvalidInput {
                parameter: "item",
                reason: "3".into()
            }
        );
        assert!(try_map(&input, MAX_EXPLICIT_WORKERS + 1, |value| Ok(*value)).is_err());
        assert!(try_map::<i32, i32>(&[], MAX_EXPLICIT_WORKERS + 1, |value| Ok(*value)).is_err());
    }
}
