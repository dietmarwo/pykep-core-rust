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
equinoctial element forms. Python scalar and NumPy state, element, period, and
optional-acceleration batches call the same provider. Batch order is
preserved; `workers=0` uses the shared pool, one is serial, and larger values
select an exact cached worker count.

## VSOP2013

`Vsop2013` and `Planet.vsop2013()` implement the analytical theory for
Mercury, Venus, the Earth–Moon barycentre (`earth_moon`), Mars, Jupiter,
Saturn, Uranus, Neptune, and Pluto. The coefficient source is the
[IMCCE VSOP2013 solution](https://ftp.imcce.fr/pub/ephem/planets/vsop2013/solution/);
the exact adaptation and license chain are recorded beside the embedded data
and in ADR 0003.

Input is an MJD2000 day count interpreted as TDB. The evaluator applies
`T = (mjd2000 - 0.5) / 365250`, because the theory is measured in thousands of
Julian years from J2000 at JD 2451545.0, twelve hours after the MJD2000 origin.
Output is heliocentric ICRF `[x, y, z, vx, vy, vz]` in metres and metres per
second. The provider does not convert UTC, TAI, TT, or TDB.

The theory was fitted to INPOP10a over 1890–2000. IMCCE also publishes
comparison errors over −4000 to +8000, but accuracy degrades with distance
from the fit interval and especially for Pluto. There is no artificial hard
date cutoff; callers must choose a time span appropriate to their accuracy
requirements. Over the fit interval, the published largest heliocentric
longitude/latitude/distance differences are:

| Body | Longitude (mas) | Latitude (mas) | Distance (km) |
|---|---:|---:|---:|
| Mercury | 0.06 | 0.01 | 0.008 |
| Venus | 0.02 | 0.05 | 0.002 |
| Earth–Moon barycentre | 0.02 | 0.08 | 0.011 |
| Mars | 0.93 | 0.06 | 0.162 |
| Jupiter | 0.20 | 0.02 | 0.277 |
| Saturn | 0.24 | 0.05 | 0.592 |
| Uranus | 2.19 | 0.13 | 5.962 |
| Neptune | 0.38 | 0.05 | 2.764 |
| Pluto | 10.83 | 3.19 | 118.419 |

The `vsop2013` Cargo feature is enabled by default. It embeds 4.3 MiB of
coefficients and supports thresholds greater than or equal to `1e-9`; the
upstream default is `1e-5`. Smaller thresholds are rejected because the
remaining 2.5 million terms are intentionally not embedded. Disable default
features to omit the data and retain the Keplerian and JPL low-precision
providers. `Vsop2013::available()` and the corresponding Python static method
make the build configuration queryable.
