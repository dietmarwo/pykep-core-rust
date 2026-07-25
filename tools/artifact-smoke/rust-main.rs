fn main() -> pykep_core::Result<()> {
    let state = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let propagated = pykep_core::astro::propagation::propagate_lagrangian(&state, 0.25, 1.0)?;
    assert!(propagated.into_iter().all(f64::is_finite));
    println!("{}", pykep_core::PORT_STATUS);
    Ok(())
}
