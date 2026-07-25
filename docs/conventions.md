# Numerical conventions

Unless an API states otherwise:

- scalar computations use IEEE-754 binary64 (`f64`);
- position is in metres and velocity is in metres per second;
- gravitational parameters are in cubic metres per square second;
- durations are in seconds, while Julian-date conversions operate in days;
- angles are in radians;
- scalar ephemeris epochs will use MJD2000;
- Cartesian state ordering is `[x, y, z, vx, vy, vz]`;
- classical element ordering is `[a, e, i, Ω, ω, ν]`;
- modified equinoctial ordering is `[p, f, g, h, k, L]`;
- matrices use row-major nested arrays in Rust and C-contiguous row-major
  arrays at the Python boundary;
- public functions reject NaN and positive or negative infinity;
- deterministic batch APIs preserve input order and do not create an implicit
  worker pool.

Some upstream functions propagate non-finite values or use them as invalid
domain sentinels. The Rust API reports explicit errors instead. Algorithm
guides document any narrower valid domain and all intentional deviations.

Classical conversion reports circular and equatorial states as singular
because their node/periapsis angles are undefined. Modified equinoctial
elements cover those states: choose the prograde convention except at
inclination `π`, and the retrograde convention except at inclination zero.
Element Jacobians are `6 × 6`, with output components as rows and input
components as columns.

Two-body propagation accepts any caller-consistent position, velocity, time,
and gravitational-parameter units. Negative durations propagate backward.
Time grids are relative to their first entry. State-transition matrices use
`∂state_final/∂state_initial`, with output components as rows.
