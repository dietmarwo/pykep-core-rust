// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion infrastructure smoke benchmark.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn status_probe(criterion: &mut Criterion) {
    criterion.bench_function("status_probe", |bencher| {
        bencher.iter(|| black_box(pykep_core::PORT_STATUS).len());
    });
}

criterion_group!(benches, status_probe);
criterion_main!(benches);
