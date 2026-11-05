# TOPS low-thrust problems

TOPS transcribes a trajectory into a chronological sequence of held controls.
The leg integrates from both ends and returns the difference at the selected
cut. The source JSON stores physical scales and reference states. The Rust
catalog retains the source units, then applies the normalization of each
pinned PyKEP constructor. The problem metadata records both scale systems.

| Family | Control per segment | State | Default fixed constraints and objective |
|---|---|---|---|
| Two-body Cartesian | Throttle fraction and three direction components | Position, velocity, mass | Seven mismatch equalities, direction inequalities, minimize negative final mass |
| MEE | Throttle fraction and three direction components | Prograde modified equinoctial elements, mass | Seven mismatch equalities with unwrapped longitude, direction inequalities, minimize negative final mass |
| CR3BP | Throttle fraction and three direction components | Rotating position, velocity, mass | Seven mismatch equalities and direction equalities, minimize negative final mass |
| Solar sail | Sail cone and clock angles | Position and velocity | Six mismatch equalities, minimize flight time |

Direction constraints use the squared norm minus one. The evaluator keeps the
decision components as supplied. The solar-sail catalog's six fixed-boundary
entries all fix flight time, so their default optimization task is feasibility.

Uniform grids divide total duration equally. Softmax grids use one logit per
segment and positive weights that sum to one. The Jacobian contracts the leg's
state, control and grid derivatives with the decoding maps. The softmax
contraction uses a reverse cumulative sum for each residual row.

Moving problems add the departure offset, endpoint relative-velocity
magnitudes and their direction components. The endpoint states follow
Keplerian or CR3BP flow. The normalized epoch chain rule includes both the
departure shift and the arrival shift due to flight time. MEE additionally
applies the coordinate Jacobian and preserves the source revolution offset.
The moving CR3BP source factory uses uniform time even when it receives
`softmax`. The Rust softmax formulation carries the version
`tops-v1-cr3bp-moving-softmax`.

The moving MEE Jacobian follows the selected longitude winding branch.
At the P5 and P8 fixed-time anchors, varying the frozen epoch or TOF can
change that branch. Optimizer adapters eliminate those frozen coordinates.
The benchmark report records free-column checks at the anchors and full
derivative checks away from the transition.

## Feasible replay examples

The bundled decisions were generated using a bounded least-squares search
with the Rust Jacobian. They satisfy the raw bounds and constraints, and a
tighter independent forward DOP853 integration checks the terminal state.

| Case | Segments | Maximum absolute forward residual |
|---|---:|---:|
| Cartesian P4 | 16 | 4.88e-14 |
| MEE P8 | 24 | 1.59e-13 |
| CR3BP P0 | 30 | 1.59e-12 |
| Solar sail P4 | 24 | 8.15e-14 |

### Cartesian two-body P4

The first coordinate is final mass. Each segment contributes a throttle
fraction and three direction components. The last coordinate is time
of flight. The integrator uses the initial radius and mass to set its
units. Solve for the seven state and mass matching rows before comparing
the final-mass objective.

```bash
python tools/gym-benchmark/solve_tops.py tops.twobody.P4 --nseg 16 \
  --seed 42 --evaluations 320 --output artifacts/gym/twobody-solved.json
```

### Modified equinoctial P8

The integrator normalizes the initial semi-latus rectum and uses prograde
MEE. The longitude residual includes its winding, so leave the longitude
unwrapped in the terminal check. The saved control directions are expressed
in the local radial, transverse and normal frame.

```bash
python tools/gym-benchmark/solve_tops.py tops.mee.P8 --nseg 24 \
  --seed 42 --evaluations 320 --output artifacts/gym/mee-solved.json
```

### Rotating CR3BP P0

The dynamics use the dataset's rotating coordinates and normalized mass
ratio. The default fixed-boundary directions have unit-norm equality
constraints. Inspect the terminal residual in the rotating frame after
replaying the chronological controls.

```bash
python tools/gym-benchmark/solve_tops.py tops.cr3bp.P0 --nseg 30 \
  --seed 11 --evaluations 900 --output artifacts/gym/cr3bp-solved.json
```

### Solar sail P4

Each segment holds cone and clock angles in radians. There is no
spacecraft mass coordinate in the six-state integration. The source
fixes the TOF, so the six terminal-state rows determine success.

```bash
python tools/gym-benchmark/solve_tops.py tops.sail.P4 --nseg 24 \
  --seed 11 --evaluations 900 --output artifacts/gym/sail-solved.json
```

The commands use SciPy's bounded least-squares solver with the analytic Rust
Jacobian and a `1e-8` feasibility gate. Each uses one thread when the
environment sets `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

After installing the wheel built from this checkout, replay a case from the
repository root. The other three fixture names are `mee.P8.json`,
`cr3bp.P0.json` and `sail.P4.json`.

```bash
python python/examples/gym/replay.py \
  --solution crates/pykep-trajopt/data/solutions/twobody.P4.json \
  --output artifacts/gym/twobody-trajectory.json
```

Use `--solution` with either the checked-in vector or a newly solved record.
The replay includes the original decision, scaled controls, chronological
time grid, endpoint states and separate forward/backward sample paths.
The backward paths are ordered from the arrival end toward the cut. Use
`tools/gym-benchmark/solve_tops.py` with SciPy to regenerate a feasible
vector. Its output records the requested budget and actual fitness and
gradient calls.

```bash
python python/examples/gym/plot.py artifacts/gym/twobody-trajectory.json \
  artifacts/gym/twobody-trajectory.png
python python/examples/gym/replay.py --problem tops.twobody.P4.moving --nseg 16
```

Plotting uses Matplotlib in a separate process. The moving-boundary
command illustrates a finite midpoint evaluation.

Set the TOPS backend explicitly to `taylor` or `dop853`. Nominal and
derivative tolerances are separate options. See
[measured objective and derivative performance](matched-benchmarks.md).
