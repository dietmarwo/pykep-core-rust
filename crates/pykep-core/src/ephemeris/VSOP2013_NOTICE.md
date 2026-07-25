# VSOP2013 coefficient notice

`vsop2013.bin` contains a thresholded, binary encoding of the original 17
integer multipliers and binary64 S/C coefficients used by heyoka 7.10.0,
commit `15bd82547b5b2d701727335c3b3dbcbeafa20018`. It retains all terms whose
coefficient magnitude is at least `1e-9`, in original series order.

The heyoka coefficient sources and evaluator are licensed under MPL-2.0:

> Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo
> (dario.izzo@gmail.com)

The heyoka sources were generated from the authoritative
[IMCCE VSOP2013 solution files](https://ftp.imcce.fr/pub/ephem/planets/vsop2013/solution/).
The source data describe the nine bodies, variables, time powers, units, time
coordinate, frame, truncation rule, and reference implementation.

The development-only generator is `oracle/generate_vsop2013_data.py` in the
staging source tree. Generation performs no rounding or transformation of the
retained coefficients. The committed binary has SHA-256
`f78e83ba43ef4a25e154bb5c82f7b9609667e9de0e26a82950027579114939bf`.

This file and `vsop2013.bin` are part of the MPL-2.0-covered source form of
pykep-rust. The full license is at the repository root.
