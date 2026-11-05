# Reference SDK provenance

The comparison used a clean checkout of ESA PyKEP
`c01e19debad5ea22b7079189f786f473046a79d5`. The C++ release library and
CPython 3.12 extension were compiled from that source. They share the
heyoka ABI 42 DSO from the heyoka.py 7.10.1 wheel. An installed historical
PyKEP 3.0.1 wheel supplied its OpenBLAS DSO and a separate historical
comparison. The new oracle process imports the compiled PyKEP 3.1.0
module from the reference prefix.

The standard SDK setup is in `tools/upstream-comparison/README.md`. The
0.1.5 measurements used a local compatibility build because the available
C++ compiler and standard library provided different parts of the required
C++23 support. The `configure_clang_sdk.py` and `zig-cxx` wrappers that show
the exact local prefix layout, link flags and imported-target shims use
developer scratch paths; they are kept in the measurement archive attached
to the 0.1.5 GitHub release rather than in the repository. The measured
build adds `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`.

The header patch `sdk-heyoka-header-adapter.patch` in this directory
applies to the heyoka 7.10.0 C++ headers and changes two development build
details. It provides a `ranges::to` fallback for libstdc++ 13. It replaces
tanuki's empty structural-config base marker with a nested marker to match
the packaged GNU ABI symbol spelling in Clang. The packaged heyoka DSO
and its numerical kernels were used as supplied.

| Dependency | Actual version or source |
|---|---|
| CMake | 3.30.9 |
| C++ compiler | Zig 0.15.2 Clang 20.1.2 frontend, libstdc++ 13 |
| heyoka C++ headers | 7.10.0, `15bd82547b5b2d701727335c3b3dbcbeafa20018` |
| heyoka.py | 7.10.1, shared ABI 42 runtime |
| mp++ | 2.0.0, `a85be780cffa57382cd36ee692d55b35d2088505` |
| spdlog | 1.17.0, `79524ddd08a4ec981b7fea76afd08ee05f83755d` |
| xtl | 0.8.2, `c52350e283b98e5d69dbc50726925a1f8e16c57c` |
| xtensor | 0.27.1, `f115228a7fc277b9be1bef34572831f8fd53ac73` |
| xtensor-blas | 0.23.0, `836a8047413eb64ce8e034ab6ab7ce1608562bfc` |
| Boost and fmt | 1.88.0 and 12.1.0 |
| PyGMO and NLopt Python distributions | 2.20.0 and 2.10.0 |
| NumPy and SciPy in the oracle | 2.3.5 and 1.17.0 |
| spiceypy and sgp4 | 8.3.0 and 2.27 |

Reproduce the warmed scalar study with `taylor_study.py --rounds 10 --cpu 2`
on an available logical core. The raw benchmark passes record the
command, executable hash, timeout, affinity and observed numerical output.
Keep the selected CPU and optimizer worker count fixed within a study.
