# ADR 0001: Fixed numerical types

- Status: accepted
- Date: 2026-07-25

## Context

The common pykep values are three-vectors, six-element Cartesian states or
orbital elements, and small fixed matrices. Dynamic allocation and a
general-purpose public matrix type would add cost and couple the API to a
dependency without improving these fixed-shape contracts.

## Decision

Parity work starts with documented aliases over stack-allocated Rust arrays:

- `Vector3 = [f64; 3]`;
- `CartesianState = [f64; 6]`;
- `Elements6 = [f64; 6]`;
- `Matrix3 = [[f64; 3]; 3]`;
- `Matrix6 = [[f64; 6]; 6]`.

Small matrix helpers use const generics internally and in deliberately generic
public operations. Variable-sized solution families, controls, grids, and
batches use owned vectors. No external matrix type appears in the public API.

## Consequences

The common hot path is allocation-free and Python conversion remains
straightforward. Aliases do not prevent semantic mix-ups, so public functions
must use meaningful parameter names and named result structs where multiple
arrays would be ambiguous. Newtypes remain an option before API stabilization
if later phases demonstrate enough safety benefit.

Phase 4 exercised that option for semantically distinct element sets:
`ClassicalElements` and `ModifiedEquinoctialElements` provide named fields and
lossless `[f64; 6]` conversions. Cartesian states retain the fixed alias
because their ordering is unambiguous in every consuming API.
