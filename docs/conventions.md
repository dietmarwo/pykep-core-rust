# Numerical conventions

Unless an API states otherwise:

- scalar computations use IEEE-754 binary64 (`f64`);
- position is in metres and velocity is in metres per second;
- gravitational parameters are in cubic metres per square second;
- durations are in seconds, while Julian-date conversions operate in days;
- angles are in radians;
- scalar ephemeris epochs will use MJD2000;
- Cartesian state ordering is `[x, y, z, vx, vy, vz]`;
- matrices use row-major nested arrays in Rust and C-contiguous row-major
  arrays at the Python boundary;
- public functions reject NaN and positive or negative infinity;
- deterministic batch APIs preserve input order and do not create an implicit
  worker pool.

Some upstream functions propagate non-finite values or use them as invalid
domain sentinels. The Rust API reports explicit errors instead. Algorithm
guides document any narrower valid domain and all intentional deviations.
