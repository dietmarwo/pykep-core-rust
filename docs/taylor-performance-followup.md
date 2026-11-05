# Taylor performance follow-up

This report records the preceding Taylor implementation rounds. For the
subsequent native PyKEP 3.1.0 and trajectory-gym measurements, see the
[matched benchmark report](matched-benchmarks.md).

The follow-up reduces generated coefficient storage, evaluates four-lane
Taylor polynomials without copying their histories into scalar workspaces,
and shares nominal coefficients across seeded sensitivity directions.

Measurements use Rust 1.88.0 on an Intel Xeon Platinum 8573C, affinity to
CPUs 2–5, and the repository's release profile. B0 is GitHub main at
`542158b575ad9d235841d501772c179177d69660`. B1 adds the first Taylor patch.
The final source adds the accepted follow-up changes. Paired results use
three ABBA rounds and seven warmed samples per pass. The ratios are medians
of paired round ratios. Host scheduling affects the spread of short timings.

## Scalar propagation at requested tolerance `1e-12`

The endpoint metric is `max(abs(actual - reference) / max(1, abs(reference)))`.
All rows meet the `2e-11` fixed-tolerance gate against the tight DOP853
reference. B0 comparisons use the default crate features for both sides.
B1 comparisons enable `simd` in both builds and call the scalar API here.

### Portable build

| Model | B1 µs | Final µs | B1 speedup | B0 speedup | Scaled endpoint error |
| --- | ---: | ---: | ---: | ---: | ---: |
| kepler | 3.740 | 4.285 | 0.91× | 0.99× | 6.66e-16 |
| cr3bp | 18.285 | 11.514 | 1.62× | 1.70× | 8.60e-15 |
| bcp | 7.698 | 5.904 | 1.34× | 1.69× | 6.94e-16 |
| zoh_kepler | 4.637 | 4.805 | 0.95× | 1.10× | 2.83e-15 |
| zoh_cr3bp | 10.424 | 9.750 | 1.07× | 1.68× | 3.03e-15 |
| zoh_equinoctial | 13.530 | 14.026 | 0.96× | 1.07× | 3.03e-15 |
| solar_sail | 25.385 | 19.759 | 1.36× | 1.33× | 7.11e-15 |
| cartesian_mass | 9.500 | 8.441 | 1.13× | 1.87× | 5.55e-16 |
| cartesian_time | 8.120 | 6.277 | 1.29× | 1.64× | 2.31e-15 |
| equinoctial_mass | 583.367 | 446.650 | 1.22× | 2.04× | 9.21e-14 |
| equinoctial_time | 516.446 | 437.909 | 1.15× | 1.94× | 9.15e-14 |

| Comparison | Geometric mean | Median | Slowest ratio |
| --- | ---: | ---: | ---: |
| B1 to final | 1.16× | 1.15× | 0.91× (kepler) |
| B0 to final | 1.51× | 1.68× | 0.99× (kepler) |

### Native build

| Model | B1 µs | Final µs | B1 speedup | B0 speedup | Scaled endpoint error |
| --- | ---: | ---: | ---: | ---: | ---: |
| kepler | 3.086 | 3.227 | 0.95× | 1.14× | 6.66e-16 |
| cr3bp | 19.069 | 10.545 | 2.14× | 2.21× | 8.60e-15 |
| bcp | 8.642 | 5.816 | 1.49× | 1.41× | 6.94e-16 |
| zoh_kepler | 3.735 | 3.705 | 1.01× | 0.94× | 2.83e-15 |
| zoh_cr3bp | 10.062 | 8.400 | 1.16× | 1.63× | 3.03e-15 |
| zoh_equinoctial | 14.660 | 14.217 | 1.03× | 0.99× | 3.03e-15 |
| solar_sail | 25.567 | 19.716 | 1.23× | 1.37× | 7.11e-15 |
| cartesian_mass | 7.883 | 8.407 | 0.91× | 1.82× | 5.55e-16 |
| cartesian_time | 7.691 | 6.437 | 1.24× | 1.33× | 2.31e-15 |
| equinoctial_mass | 582.622 | 498.064 | 1.20× | 1.98× | 9.21e-14 |
| equinoctial_time | 555.152 | 448.806 | 1.18× | 1.91× | 9.15e-14 |

| Comparison | Geometric mean | Median | Slowest ratio |
| --- | ---: | ---: | ---: |
| B1 to final | 1.20× | 1.18× | 0.91× (cartesian_mass) |
| B0 to final | 1.47× | 1.41× | 0.94× (zoh_kepler) |

## Complete workload comparisons

Grid errors use analytical Kepler values. BCP batch deltas compare against scalar calls.
Jacobian errors cover endpoint, control, and time-grid columns. Both builds enable SIMD.

### Portable application workloads

| Workload | B1 µs | Final µs | Speedup | Maximum error or delta |
| --- | ---: | ---: | ---: | ---: |
| kepler_grid | 366.18 | 69.75 | 5.04× | 1.32e-15 |
| bcp_population_balanced_1 | 3448.23 | 1741.56 | 2.06× | 1.11e-16 |
| bcp_population_mixed_1 | 3835.83 | 1914.77 | 2.08× | 1.25e-16 |
| bcp_population_balanced_4 | 1508.66 | 849.20 | 1.89× | 1.11e-16 |
| bcp_population_mixed_4 | 1651.59 | 825.40 | 1.96× | 1.25e-16 |
| kepler_stm | 49.59 | 20.41 | 2.42× | 2.55e-15 |
| zoh_jacobian_20 | 414.14 | 219.98 | 1.84× | 4.04e-11 |
| zoh_jacobian_32 | 705.77 | 342.06 | 2.12× | 3.34e-11 |

### Native application workloads

| Workload | B1 µs | Final µs | Speedup | Maximum error or delta |
| --- | ---: | ---: | ---: | ---: |
| kepler_grid | 49.91 | 42.31 | 1.06× | 1.32e-15 |
| bcp_population_balanced_1 | 2781.32 | 1364.36 | 1.91× | 0.00e+00 |
| bcp_population_mixed_1 | 2863.75 | 1615.07 | 1.74× | 0.00e+00 |
| bcp_population_balanced_4 | 1055.61 | 634.36 | 1.63× | 0.00e+00 |
| bcp_population_mixed_4 | 1692.33 | 700.40 | 2.25× | 0.00e+00 |
| kepler_stm | 52.18 | 17.94 | 2.86× | 2.55e-15 |
| zoh_jacobian_20 | 434.88 | 229.00 | 1.99× | 4.04e-11 |
| zoh_jacobian_32 | 657.29 | 342.88 | 1.86× | 3.34e-11 |

## Small scalar confirmations

The focused runs use the same sources with four times as many calls per sample.
The native Kepler case remains slower than B1. The default-feature comparison against B0 still shows a gain.

| Build | Model | B1 µs | Final µs | B1 speedup |
| --- | --- | ---: | ---: | ---: |
| portable | kepler | 3.913 | 4.020 | 0.98× |
| portable | zoh_kepler | 5.216 | 5.064 | 1.02× |
| portable | zoh_equinoctial | 13.872 | 12.780 | 1.09× |
| native | kepler | 3.041 | 3.219 | 0.92× |
| native | cartesian_mass | 8.704 | 8.664 | 1.00× |

## Short-arc matched-error sweep

Each configuration meets scaled endpoint error `1e-11` on its fixture.
These timings use the fastest tested tolerance from one seven-sample pass per configuration.
The paired fixed-tolerance comparisons above give a better view of timing variation.

| Model | B1 tolerance | Final tolerance | B1 µs | Final µs | Final error |
| --- | ---: | ---: | ---: | ---: | ---: |
| kepler | 1e-09 | 1e-09 | 3.896 | 3.491 | 3.07e-12 |
| cr3bp | 1e-09 | 1e-12 | 14.633 | 11.602 | 8.60e-15 |
| bcp | 1e-09 | 1e-09 | 7.367 | 5.305 | 1.88e-12 |
| zoh_kepler | 1e-09 | 1e-09 | 3.485 | 3.563 | 3.28e-12 |
| zoh_cr3bp | 1e-09 | 1e-12 | 8.302 | 10.561 | 3.03e-15 |
| zoh_equinoctial | 1e-09 | 1e-09 | 11.103 | 10.872 | 2.09e-13 |
| solar_sail | 1e-09 | 1e-12 | 13.742 | 19.673 | 7.11e-15 |
| cartesian_mass | 1e-09 | 1e-09 | 5.977 | 8.449 | 5.55e-16 |
| cartesian_time | 1e-09 | 1e-09 | 6.108 | 5.177 | 2.31e-15 |
| equinoctial_mass | 1e-09 | 1e-12 | 374.187 | 590.888 | 9.21e-14 |
| equinoctial_time | 1e-09 | 1e-12 | 314.934 | 420.812 | 9.15e-14 |

## CPU build comparison

Both columns use the final source and the `simd` feature. The native column uses `-C target-cpu=native`.

| Workload | Portable µs | Native µs | Portable / native |
| --- | ---: | ---: | ---: |
| kepler_grid | 80.95 | 45.04 | 1.94× |
| bcp_population_balanced_1 | 1814.81 | 1380.00 | 1.32× |
| bcp_population_mixed_1 | 1953.42 | 1385.15 | 1.42× |
| kepler_stm | 21.51 | 17.13 | 1.29× |
| zoh_jacobian_20 | 241.96 | 206.80 | 1.08× |

## Portable mixed sensitivity seed widths

| Workload | B1 µs | Final µs | Speedup | Final error |
| --- | ---: | ---: | ---: | ---: |
| kepler_stm_w1 | 22.05 | 20.43 | 1.42× | 1.67e-15 |
| kepler_stm_w14 | 109.68 | 41.52 | 2.65× | 2.39e-15 |
| kepler_stm_w21 | 173.19 | 111.57 | 1.58× | 2.39e-15 |
| kepler_stm_w4 | 33.15 | 17.57 | 1.78× | 2.16e-15 |
| kepler_stm_w6 | 70.84 | 22.53 | 3.14× | 2.16e-15 |
| kepler_stm_w7 | 55.22 | 30.28 | 1.89× | 2.16e-15 |

## Cumulative portable grid and STM

| Workload | B0 µs | Final µs | Speedup | Final error |
| --- | ---: | ---: | ---: | ---: |
| kepler_grid | 1061.59 | 74.24 | 14.62× | 1.32e-15 |
| kepler_stm | 58.89 | 21.25 | 2.72× | 2.55e-15 |

## Cumulative native grid and STM

| Workload | B0 µs | Final µs | Speedup | Final error |
| --- | ---: | ---: | ---: | ---: |
| kepler_grid | 787.78 | 42.44 | 18.89× | 1.32e-15 |
| kepler_stm | 44.22 | 17.79 | 2.49× | 2.55e-15 |

## Complete Python ZOH history calls

| Workload | B1 µs | Final µs | Speedup | Final error |
| --- | ---: | ---: | ---: | ---: |
| zoh_history_10 | 16.24 | 8.97 | 1.81× | 6.11e-16 |
| zoh_history_1001 | 1490.78 | 761.22 | 1.96× | 7.22e-16 |
| zoh_history_101 | 130.69 | 64.76 | 2.22× | 6.66e-16 |
| zoh_history_2 | 5.33 | 4.55 | 1.17× | 6.11e-16 |

## fcmaes objective replay

The same 1,024 candidates were evaluated in the same order. Times include construction, propagation, and scoring.

| Objective | Workers | B1 µs | Final µs | Speedup | Maximum score/residual delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| bcp | 1 | 4255.43 | 2573.56 | 1.65× | 1.11e-16 |
| zoh_kepler | 1 | 33630.62 | 29975.51 | 1.12× | 0.00e+00 |
| bcp | 4 | 1784.45 | 1038.34 | 1.72× | 1.11e-16 |
| zoh_kepler | 4 | 9870.46 | 9796.29 | 1.01× | 0.00e+00 |

Each optimizer cell uses seeds 11, 19, and 42 and 4,096 evaluations per run.
These short campaigns compare wall time and final objectives. Time-to-target records are in the raw data.

| Objective | Workers | Median B1 ms | Median final ms | Largest final objective delta | Target hits B1 / final |
| --- | ---: | ---: | ---: | ---: | ---: |
| bcp | 1 | 21.71 | 13.37 | 3.12e-19 | 3 / 3 |
| zoh_kepler | 1 | 110.62 | 114.27 | 0.00e+00 | 0 / 0 |
| bcp | 4 | 42.26 | 19.45 | 3.12e-19 | 3 / 3 |
| zoh_kepler | 4 | 68.35 | 68.34 | 0.00e+00 | 0 / 0 |

## Python calls at matched endpoint accuracy

PyKEP 3.0.1 with heyoka.py 7.10.1 reuses the compiled integrator. Both sides include Python calls.
Rows select the fastest tested tolerance at scaled endpoint error `1e-11`.

| Model | PyKEP µs | Rust Taylor µs | Rust / PyKEP |
| --- | ---: | ---: | ---: |
| kepler | 4.023 | 3.473 | 0.86× |
| cr3bp | 4.866 | 9.682 | 1.99× |
| bcp | 4.564 | 5.469 | 1.20× |
| zoh_kepler | 5.430 | 5.446 | 1.00× |
| zoh_cr3bp | 5.681 | 7.913 | 1.39× |
| zoh_equinoctial | 8.694 | 12.387 | 1.42× |
| solar_sail | 7.403 | 22.481 | 3.04× |
| cartesian_mass | 4.653 | 8.618 | 1.85× |
| cartesian_time | 4.135 | 6.129 | 1.48× |
| equinoctial_mass | 622.882 | 508.383 | 0.82× |
| equinoctial_time | 554.198 | 413.126 | 0.75× |

See the comparison guide for reproduction commands. The raw measurements
are in the measurement archive attached to the [0.1.5 release](https://github.com/dietmarwo/pykep-core-rust/releases/tag/v0.1.5).
[Comparison guide](https://github.com/dietmarwo/pykep-core-rust/blob/main/tools/taylor-comparison/README.md)
