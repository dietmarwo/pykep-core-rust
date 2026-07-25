// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from include/kep3/core_astro/convert_julian_dates.hpp at pykep
// commit 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Conversions among Julian date, modified Julian date, and MJD2000.
//!
//! These are arithmetic day-count conversions only. They do not implement a
//! time scale, leap seconds, UTC, TT, or TDB.

use crate::Result;
use crate::error::ensure_finite;

const JD_MINUS_MJD: f64 = 2_400_000.5;
const JD_AT_MJD2000_ZERO: f64 = 2_451_544.5;
const MJD_AT_MJD2000_ZERO: f64 = 51_544.0;

/// Converts a Julian date to modified Julian date, in days.
///
/// # Errors
///
/// Returns an error if `julian_date` is NaN or infinite.
pub fn jd_to_mjd(julian_date: f64) -> Result<f64> {
    ensure_finite("julian_date", julian_date)?;
    Ok(julian_date - JD_MINUS_MJD)
}

/// Converts a Julian date to MJD2000, in days from 2000-01-01 00:00.
///
/// # Errors
///
/// Returns an error if `julian_date` is NaN or infinite.
pub fn jd_to_mjd2000(julian_date: f64) -> Result<f64> {
    ensure_finite("julian_date", julian_date)?;
    Ok(julian_date - JD_AT_MJD2000_ZERO)
}

/// Converts a modified Julian date to Julian date, in days.
///
/// # Errors
///
/// Returns an error if `modified_julian_date` is NaN or infinite.
pub fn mjd_to_jd(modified_julian_date: f64) -> Result<f64> {
    ensure_finite("modified_julian_date", modified_julian_date)?;
    Ok(modified_julian_date + JD_MINUS_MJD)
}

/// Converts a modified Julian date to MJD2000, in days.
///
/// # Errors
///
/// Returns an error if `modified_julian_date` is NaN or infinite.
pub fn mjd_to_mjd2000(modified_julian_date: f64) -> Result<f64> {
    ensure_finite("modified_julian_date", modified_julian_date)?;
    Ok(modified_julian_date - MJD_AT_MJD2000_ZERO)
}

/// Converts MJD2000 to Julian date, in days.
///
/// # Errors
///
/// Returns an error if `mjd2000` is NaN or infinite.
pub fn mjd2000_to_jd(mjd2000: f64) -> Result<f64> {
    ensure_finite("mjd2000", mjd2000)?;
    Ok(mjd2000 + JD_AT_MJD2000_ZERO)
}

/// Converts MJD2000 to modified Julian date, in days.
///
/// # Errors
///
/// Returns an error if `mjd2000` is NaN or infinite.
pub fn mjd2000_to_mjd(mjd2000: f64) -> Result<f64> {
    ensure_finite("mjd2000", mjd2000)?;
    Ok(mjd2000 + MJD_AT_MJD2000_ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PykepError;

    #[test]
    fn reference_epoch_conversions_are_exact() {
        assert_eq!(jd_to_mjd(2_451_544.5), Ok(51_544.0));
        assert_eq!(jd_to_mjd2000(2_451_544.5), Ok(0.0));
        assert_eq!(mjd_to_jd(51_544.0), Ok(2_451_544.5));
        assert_eq!(mjd_to_mjd2000(51_544.0), Ok(0.0));
        assert_eq!(mjd2000_to_jd(0.0), Ok(2_451_544.5));
        assert_eq!(mjd2000_to_mjd(0.0), Ok(51_544.0));
    }

    #[test]
    fn finite_conversions_round_trip() {
        for value in [-100_000.25, 0.0, 51_544.5, 1_000_000.75] {
            assert_eq!(mjd_to_jd(jd_to_mjd(value).unwrap()).unwrap(), value);
            assert_eq!(mjd2000_to_jd(jd_to_mjd2000(value).unwrap()).unwrap(), value);
        }
    }

    #[test]
    fn non_finite_day_counts_are_rejected() {
        assert_eq!(
            jd_to_mjd(f64::NAN),
            Err(PykepError::NonFiniteInput {
                parameter: "julian_date"
            })
        );
        assert!(mjd2000_to_jd(f64::INFINITY).is_err());
    }
}
