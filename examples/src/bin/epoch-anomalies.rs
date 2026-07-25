//! Convert an epoch and round-trip elliptic and hyperbolic anomalies.
//!
//! Units: epochs use MJD2000 days and anomalies are radians.
//! Expected: arrival 180 days later and round trips equal to 0.7 and 4.0.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: default `pykep-core`; no external data or runtime.

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
