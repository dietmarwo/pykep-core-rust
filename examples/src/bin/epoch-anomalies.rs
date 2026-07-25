//! Demonstrates Phase 3 epoch and anomaly conversion APIs.

use pykep_core::Result;
use pykep_core::astro::anomalies::{
    eccentric_to_mean_anomaly, hyperbolic_anomaly_to_mean, hyperbolic_mean_to_anomaly,
    mean_to_eccentric_anomaly,
};
use pykep_core::time::epoch::Epoch;

fn main() -> Result<()> {
    let departure = Epoch::from_iso("2030-01-15T12:30:00")?;
    let arrival = departure.checked_add_days(180.0)?;
    println!("departure: {departure}");
    println!("arrival:   {arrival}");

    let eccentric = mean_to_eccentric_anomaly(0.7, 0.8)?;
    println!(
        "elliptic round trip: {}",
        eccentric_to_mean_anomaly(eccentric, 0.8)?
    );

    let hyperbolic = hyperbolic_mean_to_anomaly(4.0, 1.5)?;
    println!(
        "hyperbolic round trip: {}",
        hyperbolic_anomaly_to_mean(hyperbolic, 1.5)?
    );
    Ok(())
}
