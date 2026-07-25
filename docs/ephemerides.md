# Ephemerides

## JPL low-precision planets

`JplLowPrecision` and `Planet.jpl_low_precision()` evaluate the eight
heliocentric approximate planetary models carried by pykep/kep3 3.0.1:
Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Body lookup
is ASCII case-insensitive. The returned name is the lowercase body followed by
`(jpl_lp)`.

Inputs are MJD2000 day counts and must satisfy the open interval
`-73048 < epoch < 18263`, approximately 1800–2050. Cartesian output is
`[x, y, z, vx, vy, vz]` in metres and metres per second, heliocentric and
referred to the mean ecliptic and equinox of J2000. The underlying JPL table
uses Julian ephemeris date/JDTDB; this library does not perform time-scale
conversion. The `earth` coefficients describe the Earth–Moon barycentre, as
in the source table.

The coefficients and rates come from the
[JPL Solar System Dynamics approximate-position table](https://ssd.jpl.nasa.gov/planets/approx_pos.html).
JPL describes these as lower-accuracy fitted formulae and warns against using
them outside their fitted interval. Its nominal 1800–2050 errors are:

| Body | Longitude (arcsec) | Latitude (arcsec) | Distance (1000 km) |
|---|---:|---:|---:|
| Mercury | 15 | 1 | 1 |
| Venus | 20 | 1 | 4 |
| Earth–Moon barycentre | 20 | 8 | 6 |
| Mars | 40 | 2 | 25 |
| Jupiter | 400 | 10 | 600 |
| Saturn | 600 | 25 | 1500 |
| Uranus | 50 | 2 | 1000 |
| Neptune | 10 | 1 | 200 |

These models are suitable for approximate mission design, not precision
navigation. Use a high-precision integrated ephemeris when those error bounds
are too large.

The Rust provider exposes true-anomaly, mean-anomaly, and both modified
equinoctial element forms. Python scalar and NumPy batch state evaluation call
the same provider. Batch order is preserved and no implicit thread pool is
created.
