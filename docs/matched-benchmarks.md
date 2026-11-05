# Matched PyKEP benchmarks

The 0.1.5 benchmark tree was measured against PyKEP 3.1.0 at
`c01e19debad5ea22b7079189f786f473046a79d5`. Warmed heyoka led on nine
scalar Taylor fixtures. Rust led on both equinoctial Pontryagin fixtures
and on the complete ZOH Jacobian in the three leg pilots. The nominal
ZOH mismatch and full Jacobian give different results.
[Source workloads](https://github.com/esa/pykep/tree/c01e19debad5ea22b7079189f786f473046a79d5/benchmark)

## Native scalar Taylor

The reference uses heyoka 7.10.0 ABI 42. The host was an Intel Xeon
Platinum 8573C under Linux. The Rust binary uses Rust 1.88, release thin
LTO and the portable x86-64 target with SIMD compiled in. The native
driver uses Clang 20.1.2, libstdc++ 13 and `-O3 -march=x86-64`. Heyoka
generates its model kernels at runtime. The SDK used a development-only
header adapter to link the Clang build to the packaged heyoka ABI.
Each process is pinned to one
logical CPU. The study uses ten paired rounds in alternating order.
Upstream resets and reuses a warmed integrator. The raw samples include
separate cold setup, first call and reset-only results.

Both arms use the same initial states, parameters, times and tight DOP853
endpoint vector. Both requested `1e-12` and passed the `1e-11` scaled
endpoint-error gate. The ratio is the median of paired Rust / upstream
times. The interval resamples whole pairs.

| Model | Rust µs | Warmed heyoka µs | Rust / heyoka time | 95% paired interval | Scaled errors Rust / heyoka |
| --- | --- | --- | --- | --- | --- |
| kepler | 4.6 | 2.02 | 2.26× | 1.73–3.31 | 6.7e-16 / 7.3e-15 |
| cr3bp | 10.8 | 4.59 | 2.57× | 1.98–3.24 | 8.6e-15 / 5.1e-15 |
| bcp | 7.01 | 4.22 | 1.94× | 1.31–2.22 | 6.9e-16 / 6.0e-15 |
| zoh_kepler | 5.11 | 2.92 | 1.77× | 1.25–2.11 | 2.8e-15 / 8.1e-15 |
| zoh_cr3bp | 11.2 | 6.54 | 1.67× | 1.50–1.84 | 3.0e-15 / 1.1e-14 |
| zoh_equinoctial | 15 | 6.73 | 2.19× | 1.69–3.12 | 3.0e-15 / 3.0e-15 |
| solar_sail | 21.3 | 9.09 | 2.49× | 2.00–2.85 | 7.1e-15 / 1.1e-14 |
| cartesian_mass | 8.88 | 4.02 | 2.28× | 1.91–2.54 | 5.6e-16 / 5.6e-16 |
| cartesian_time | 8.24 | 3.72 | 2.27× | 1.71–2.39 | 2.3e-15 / 2.3e-15 |
| equinoctial_mass | 536 | 673 | 0.78× | 0.69–0.87 | 9.2e-14 / 2.8e-13 |
| equinoctial_time | 516 | 623 | 0.83× | 0.75–0.91 | 9.1e-14 / 2.8e-13 |

## Seven upstream benchmark programs

The full-size shared-input matrix uses the upstream case counts.
The runners check finiteness and aggregate output counts for every
case and retain complete sampled vectors. The numbers are orientation
results from one ABBA block. Small differences are sensitive to host load.

| Workload | Cases per row | Upstream / Rust time by case band | Max sampled scaled difference |
| --- | --- | --- | --- |
| Anomaly conversion | 1,000,000 | 1.21× / 1.25× / 0.69× | 1.1e-15 |
| Anomaly round trip | 100,000 | 0.48× / 1.15× / 1.47× | 9.4e-16 |
| Lagrange state | 1,000,000 | 1.31× / 2.94× / 1.15× / 2.28× | 2.4e-15 |
| Lagrange round trip | 100,000 | 1.54× / 1.23× / 1.57× | 5.1e-15 |
| State and 6 × 6 STM | 100,000 | 2.47× / 2.49× / 1.58× | 3.5e-14 |
| Lambert 0 / ≤20 revolutions | 50,000 | 0.84× / 0.79× | 3.3e-16 |
| SF simple mismatch | 1 | 1.33× | 1.9e-16 |
| SF complete Jacobian | 1,000 | 4.05× / 4.47× / 3.29× / 2.53× | 1.8e-15 |
| SF numerical UDP gradient | 1,000 | 1.05× / 1.14× / 1.12× / 1.17× | 1.1e-07 |

The initial sweep found convergence exhaustion in the previous Lagrange
anomaly solve. With safeguarded Newton iteration, the repeat passed the
failure-count, output-count and sampled-value gates. The measurement
archive preserves the initial failed cases and the repeat.

### Complete ZOH legs

Each segment count uses 200 shared cases, seed 424242, cached legs and
five passes per arm. The complete Jacobian includes both endpoint-state
blocks, every segment control and all grid-node derivatives. The table
uses `1e-13` for both variational tolerances and `1e-14` for nominal
integration. The measurement archive also records the source `1e-8`
derivative setting and a separate Rust DOP853 arm. That arm predates the
analytic DOP853 model Jacobians for the Kepler and CR3BP ZOH models in
0.1.5; the Taylor rows are unaffected.

| Segments | Operation | Rust µs | Upstream µs | Upstream / Rust time | Sampled scaled difference |
| --- | --- | --- | --- | --- | --- |
| 20 | mismatch | 36.9 | 23.9 | 0.65× | 4.4e-16 |
| 20 | jacobian | 272 | 556 | 2.05× | 1.9e-15 |
| 40 | mismatch | 67.2 | 51.2 | 0.76× | 2.2e-16 |
| 40 | jacobian | 525 | 1.16e+03 | 2.22× | 1.0e-15 |
| 80 | mismatch | 126 | 102 | 0.81× | 2.2e-16 |
| 80 | jacobian | 1.1e+03 | 2.13e+03 | 1.93× | 1.6e-15 |

## Python gym evaluation

Four TOPS cases use 16 segments, uniform time and the same first vector
in each Python implementation. The source variational tolerance is
`1e-13`. Fitness and derivative agreement was checked on four vectors
for each case. Each cell is the median of seven inner samples in one
process. The measurement archive has the softmax results, two classical examples,
construction, ordered batch, fitness plus gradient, and history timings.

| Problem | Rust fitness µs | PyKEP fitness µs | Rust gradient µs | PyKEP gradient µs |
| --- | --- | --- | --- | --- |
| tops.twobody.P4 | 27.2 | 61.2 | 301 | 670 |
| tops.mee.P8 | 5.03e+03 | 2.9e+03 | 3.8e+04 | 8.28e+04 |
| tops.cr3bp.P0 | 434 | 817 | 5.49e+03 | 1.21e+04 |
| tops.sail.P4 | 1.54e+03 | 1.01e+03 | 6.91e+03 | 1.1e+04 |
| gym.eve_mga1dsm | 6.11 | 66.7 | — | — |
| gym.juice_mo | 22.6 | 530 | — | — |

## Gym validation

The catalog exposes 82 stable IDs. The primary comparison covers
14 classical configurations and both time encodings for each of the
68 TOPS IDs. It uses 16 segments and two vectors per configuration.
A five-segment DOP853 arm exercises an alternative cut and direction mode.

| Source catalog | Configurations | Direct pass | Rust extension | Discrepancy | Nonfinite/failed |
| --- | --- | --- | --- | --- | --- |
| Classical | 14 | 14 | 0 | 0 | 0 |
| Fixed TOPS | 68 | 68 | 0 | 0 | 0 |
| Moving TOPS | 68 | 34 | 14 | 12 | 8 |

The moving discrepancies are ten moving-MEE rows and two moving-CR3BP
stress vectors whose trajectories pass close to the Moon. The CR3BP pair
differs from the source gradient by a scaled `1.07e-6` and `4.23e-6`,
inside the `2.4e-5` spread of the Rust Taylor gradient across tolerances
on the same vectors; see [TOPS known limitations](tops.md#known-limitations).
Moving CR3BP softmax has a separate Rust formulation ID and an independent
gradient check. The source moving CR3BP constructor selects uniform time.
Moving MEE retains the original source discrepancies and a separate
comparison using the source symbolic MEE conversion and finite
differences. The P5 and P8 fixed-time anchors have off-anchor full-column
checks. [TOPS guide](tops.md)

## Optimizer comparisons

The same native fcmaes DE-to-CMA adapter ran the two Rust backends on four
fixed-boundary TOPS cases with 20 held-out seeds per case and budget.
The fixed-work arm requests 128 evaluations. The wall arm uses an
80 ms budget checked between populations. The four feasible tutorials
use a separate least-squares run and a tighter independent forward replay.

| Case | Transcription | 128 calls, ms Taylor / DOP853 | 80 ms, calls Taylor / DOP853 | Runs with feasible candidate |
| --- | --- | --- | --- | --- |
| tops.twobody.P4 | 16 uniform | 9.24 / 14.2 | 1616 / 896 | 0/80 |
| tops.mee.P8 | 24 softmax | 350 / 141 | 32 / 96 | 0/80 |
| tops.cr3bp.P0 | 30 uniform | 85.2 / 72.8 | 128 / 192 | 0/80 |
| tops.sail.P4 | 24 softmax | 103 / 51.2 | 128 / 224 | 0/80 |

For Sims–Flanagan, both Python evaluator arms use the same PyGMO NLopt
SLSQP, starting points, `1e-8` raw feasibility threshold and maximum
1,000 solver evaluations. Each segment count has 100 starts. The raw
counts include the calls made by numerical differentiation.

| Segments | Rust analytic | PyKEP analytic | Rust numerical | PyKEP numerical |
| --- | --- | --- | --- | --- |
| 5 | 84/100 | 84/100 | 84/100 | 84/100 |
| 10 | 83/100 | 83/100 | 83/100 | 83/100 |
| 15 | 84/100 | 84/100 | 85/100 | 85/100 |

## Benchmark notebooks

The five-segment SF notebook checks all Jacobian groups against both
the C++ leg and symbolic heyoka. Its legacy Isp assignment uses the
current exhaust-velocity constructor in the reproduction. Complete
derivative differences were below `2.4e-13` in both checks.

The provider study ran TLE and the available SPICE notebook cells as
reference-only rows. The SPICE wrapper batch cell fails on its
list-to-time multiplication. The separately labelled provider
comparison covers Keplerian, JPL low-precision and VSOP2013 scalar
and batch paths.

The measurement archive attached to the [0.1.5 release](https://github.com/dietmarwo/pykep-core-rust/releases/tag/v0.1.5) contains
the paired samples, per-case gym results, optimizer records, fixture
hashes and SDK provenance, including the development-only heyoka header
adapter. [Run the comparisons](https://github.com/dietmarwo/pykep-core-rust/blob/main/tools/upstream-comparison/README.md)
