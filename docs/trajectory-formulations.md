# Impulsive trajectory formulations

The impulsive problems use the core ephemeris, Lambert, Lagrange propagation,
and flyby routines. Encounter epochs are MJD2000 days, coast durations are
seconds, states use metres and metres per second, and maneuver magnitudes are
metres per second.

- MGA joins successive encounter positions with zero-revolution Lambert
  arcs, applies powered flyby costs, then adds the configured launch and
  arrival costs. Direct, alpha and eta encodings specify different bound
  boxes. They evaluate the same physical trajectory when decoded into the
  same leg durations.
- MGA-1DSM propagates a coast before each maneuver and closes the leg using
  a Lambert arc. Later legs begin with an unpowered flyby. The decoded trace
  includes the DSM vectors, flyby radii and angles, and each coast/Lambert
  arc. The launch uvV vector uses two angles encoded in the unit interval
  and a magnitude in metres per second.
- N-impulse problems use alpha weights for the coast times. The final two
  burns come from the last Lambert arc and the arrival velocity match. The
  Earth–Mars gym instances use three, five or seven impulses.
- JUICE evaluates the Ariane 5 table at the decoded launch speed and
  declination. It applies the pinned insertion orbit, the added 275 m/s
  cost and rocket-equation mass loss. The first minimized objective is
  negative log final mass. The multiobjective variant also minimizes
  total flight time in days.

The launcher interpolation matches the pinned source conventions. The
two-dimensional splines clamp to their tabulated domain. The one-dimensional
Atlas 551 and Falcon 9 tables return 0.1 kg outside their speed range.
`Launcher::mass` and `launcher_mass` accept speed in kilometres per second
and declination in degrees.

Run the generic replay example with any classical ID and inspect the returned
arc endpoint closure. Use MODE to produce a JUICE trade-off study.

```bash
python python/examples/gym/replay.py --problem gym.cassini1
python python/examples/gym/replay.py --problem gym.rosetta
python python/examples/gym/replay.py --problem gym.em7imp
cargo run --release --locked -p pykep-gym-benchmark -- optimize \
  --problem gym.juice_mo --optimizer mode --evaluations 4096 \
  --seed 42 --workers 1 --output artifacts/gym/juice
```

The `gym.*` namespace identifies the current PyKEP definitions. The legacy
fcmaes GTOP objectives use separate result identifiers and kernels.

| Current gym | Related legacy GTOP | Definition details |
|---|---|---|
| `gym.cassini1`, `gym.cassini1_a`, `gym.cassini1_n` | `cassini1` | Fixed encounter sequence with distinct encoding, ephemeris and launch conventions. The legacy mixed-integer variant includes sequence decisions. |
| `gym.cassini2` | `cassini2` | Current gym charges Saturn insertion. Legacy GTOP uses rendezvous in the DSM formulation. |
| `gym.messenger` | `messenger`, `messenger_full` | The reduced legacy case has a different sequence. The full case shares the insertion target. Ephemeris, constants and bounds follow each implementation's definition. |
| `gym.rosetta` | `rosetta` | Both use a Keplerian comet endpoint. Their ephemeris and objective conventions are versioned separately. |
| EVE, JUICE, Earth–Mars impulse, TOPS | Native gym IDs | These use the definitions in the pinned PyKEP gym. |

See the [legacy implementations](https://github.com/dietmarwo/fcmaes-rust/blob/8b48919d7e0776c20e60fadcd01b1bae8dc086cd/crates/fcmaes-gtop/src/lib.rs).
See the [current upstream definitions](https://github.com/esa/pykep/tree/c01e19debad5ea22b7079189f786f473046a79d5/pykep/trajopt/gym).
