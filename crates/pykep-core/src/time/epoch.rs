// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from include/kep3/epoch.hpp and src/epoch.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Microsecond-resolution epochs relative to MJD2000.

use core::fmt;
use core::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::ensure_finite;
use crate::{PykepError, Result};

const MICROSECONDS_PER_SECOND: i64 = 1_000_000;
const SECONDS_PER_DAY: i64 = 86_400;
const MICROSECONDS_PER_DAY: i64 = SECONDS_PER_DAY * MICROSECONDS_PER_SECOND;
const MJD_AT_MJD2000: f64 = 51_544.0;
const JD_AT_MJD2000: f64 = 2_451_544.5;
const MJD_OFFSET_MICROSECONDS: i64 = 51_544 * MICROSECONDS_PER_DAY;
const JD_OFFSET_MICROSECONDS: i64 = 211_813_444_800_000_000;
const UNIX_SECONDS_AT_MJD2000: i64 = 946_684_800;
const UNIX_DAYS_AT_MJD2000: i64 = 10_957;

/// A proleptic-Gregorian epoch with one-microsecond resolution.
///
/// The internal value is a signed count of microseconds from
/// 2000-01-01T00:00:00, which is MJD2000 zero. Julian day getters are
/// arithmetic day counts and do not imply UTC, leap-second, TT, or TDB
/// handling.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Epoch {
    microseconds_mjd2000: i64,
}

impl Epoch {
    /// Constructs MJD2000 zero (`2000-01-01T00:00:00.000000`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            microseconds_mjd2000: 0,
        }
    }

    fn from_scaled_days(
        value: f64,
        parameter: &'static str,
        offset_microseconds: i64,
    ) -> Result<Self> {
        ensure_finite(parameter, value)?;
        let scaled = value * MICROSECONDS_PER_DAY as f64;
        if !scaled.is_finite() || scaled >= i64::MAX as f64 || scaled < i64::MIN as f64 {
            return Err(PykepError::InvalidInput {
                parameter,
                reason: "day count is outside the representable epoch range".into(),
            });
        }
        let absolute_microseconds = scaled.trunc() as i64;
        let microseconds_mjd2000 = absolute_microseconds
            .checked_sub(offset_microseconds)
            .ok_or_else(|| PykepError::InvalidInput {
                parameter,
                reason: "day count is outside the representable epoch range".into(),
            })?;
        Ok(Self {
            microseconds_mjd2000,
        })
    }

    /// Constructs an epoch from MJD2000 days.
    ///
    /// Fractional values are truncated to microsecond resolution, matching the
    /// pinned upstream representation.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN, infinity, or an out-of-range day count.
    pub fn from_mjd2000(days: f64) -> Result<Self> {
        Self::from_scaled_days(days, "mjd2000", 0)
    }

    /// Constructs an epoch from modified Julian date days.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN, infinity, or an out-of-range day count.
    pub fn from_mjd(days: f64) -> Result<Self> {
        Self::from_scaled_days(days, "mjd", MJD_OFFSET_MICROSECONDS)
    }

    /// Constructs an epoch from Julian date days.
    ///
    /// The internal epoch is microsecond-granular, but a binary64 Julian date
    /// near J2000 has roughly 40 microseconds between adjacent representable
    /// values. Use [`Self::from_mjd2000`] or [`Self::from_calendar`] when
    /// single-microsecond construction is required.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN, infinity, or an out-of-range day count.
    pub fn from_jd(days: f64) -> Result<Self> {
        Self::from_scaled_days(days, "jd", JD_OFFSET_MICROSECONDS)
    }

    /// Constructs an epoch from a validated proleptic-Gregorian calendar.
    ///
    /// `hour`, `minute`, and `second` use their conventional ranges.
    /// `millisecond` and `microsecond` must each be in `0..=999`; their sum is
    /// the six-digit fractional second.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid date/component or a date outside the
    /// signed microsecond representation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_calendar(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        millisecond: u32,
        microsecond: u32,
    ) -> Result<Self> {
        if !(-32_767..=32_767).contains(&year) {
            return Err(invalid_calendar("year", "must be in -32767..=32767"));
        }
        if !(1..=12).contains(&month) {
            return Err(invalid_calendar("month", "must be in 1..=12"));
        }
        let maximum_day = days_in_month(year, month);
        if day == 0 || day > maximum_day {
            return Err(invalid_calendar(
                "day",
                &format!("must be in 1..={maximum_day} for the selected month"),
            ));
        }
        for (name, value, upper) in [
            ("hour", hour, 23),
            ("minute", minute, 59),
            ("second", second, 59),
            ("millisecond", millisecond, 999),
            ("microsecond", microsecond, 999),
        ] {
            if value > upper {
                return Err(invalid_calendar(name, &format!("must be in 0..={upper}")));
            }
        }

        let days_from_unix = days_from_civil(year as i64, month, day);
        let days_from_mjd2000 = days_from_unix - UNIX_DAYS_AT_MJD2000;
        let day_microseconds = days_from_mjd2000
            .checked_mul(MICROSECONDS_PER_DAY)
            .ok_or_else(epoch_range_error)?;
        let second_of_day = i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second);
        let time_microseconds = second_of_day * MICROSECONDS_PER_SECOND
            + i64::from(millisecond) * 1_000
            + i64::from(microsecond);
        let microseconds_mjd2000 = day_microseconds
            .checked_add(time_microseconds)
            .ok_or_else(epoch_range_error)?;
        Ok(Self {
            microseconds_mjd2000,
        })
    }

    /// Parses a cropped ISO calendar string.
    ///
    /// Accepted forms are `YYYY-MM`, `YYYY-MM-DD`, and incremental suffixes
    /// through `YYYY-MM-DDTHH:MM:SS.ffffff`. Missing day/time components
    /// default to the first day at midnight. The year has four or five ASCII
    /// digits and may have a leading minus sign, covering every year emitted
    /// by [`Self::to_iso`].
    ///
    /// # Errors
    ///
    /// Returns an error for malformed text or an invalid calendar value.
    pub fn from_iso(text: &str) -> Result<Self> {
        const SUFFIX_LENGTHS: [usize; 11] = [3, 6, 9, 12, 15, 17, 18, 19, 20, 21, 22];
        if !text.is_ascii() {
            return Err(invalid_iso());
        }
        let bytes = text.as_bytes();
        let separator_search_start = usize::from(bytes.first() == Some(&b'-'));
        let year_end = bytes[separator_search_start..]
            .iter()
            .position(|&byte| byte == b'-')
            .map(|index| index + separator_search_start)
            .ok_or_else(invalid_iso)?;
        let year_digits = &text[separator_search_start..year_end];
        let suffix = &text[year_end..];
        if !(4..=5).contains(&year_digits.len())
            || !year_digits.bytes().all(|byte| byte.is_ascii_digit())
            || !SUFFIX_LENGTHS.contains(&suffix.len())
        {
            return Err(invalid_iso());
        }
        let has = |index: usize, expected: u8| suffix.as_bytes().get(index) == Some(&expected);
        if !has(0, b'-')
            || (suffix.len() >= 6 && !has(3, b'-'))
            || (suffix.len() >= 9 && !has(6, b'T'))
            || (suffix.len() >= 12 && !has(9, b':'))
            || (suffix.len() >= 15 && !has(12, b':'))
            || (suffix.len() >= 17 && !has(15, b'.'))
        {
            return Err(invalid_iso());
        }

        let year = text[..year_end].parse::<i32>().map_err(|_| invalid_iso())?;
        let month = parse_component::<u32>(suffix, 1, 3)?;
        let day = if suffix.len() >= 6 {
            parse_component(suffix, 4, 6)?
        } else {
            1
        };
        let hour = if suffix.len() >= 9 {
            parse_component(suffix, 7, 9)?
        } else {
            0
        };
        let minute = if suffix.len() >= 12 {
            parse_component(suffix, 10, 12)?
        } else {
            0
        };
        let second = if suffix.len() >= 15 {
            parse_component(suffix, 13, 15)?
        } else {
            0
        };
        let fractional_microseconds = if suffix.len() >= 17 {
            let fraction = &suffix[16..];
            if !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(invalid_iso());
            }
            let parsed = fraction.parse::<u32>().map_err(|_| invalid_iso())?;
            parsed * 10_u32.pow((6 - fraction.len()) as u32)
        } else {
            0
        };
        Self::from_calendar(
            year,
            month,
            day,
            hour,
            minute,
            second,
            fractional_microseconds / 1_000,
            fractional_microseconds % 1_000,
        )
    }

    /// Returns the epoch as MJD2000 days.
    #[must_use]
    pub fn mjd2000(self) -> f64 {
        self.microseconds_mjd2000 as f64 / MICROSECONDS_PER_DAY as f64
    }

    /// Returns the epoch as modified Julian date days.
    #[must_use]
    pub fn mjd(self) -> f64 {
        self.mjd2000() + MJD_AT_MJD2000
    }

    /// Returns the epoch as Julian date days.
    #[must_use]
    pub fn jd(self) -> f64 {
        self.mjd2000() + JD_AT_MJD2000
    }

    /// Returns the signed internal microsecond count from MJD2000 zero.
    #[must_use]
    pub const fn microseconds_since_mjd2000(self) -> i64 {
        self.microseconds_mjd2000
    }

    /// Formats a proleptic-Gregorian ISO timestamp with six fractional digits.
    ///
    /// The result is accepted by [`Self::from_iso`] across the complete
    /// supported year range.
    #[must_use]
    pub fn to_iso(self) -> String {
        let days = self.microseconds_mjd2000.div_euclid(MICROSECONDS_PER_DAY);
        let within_day = self.microseconds_mjd2000.rem_euclid(MICROSECONDS_PER_DAY);
        let (year, month, day) = civil_from_days(days + UNIX_DAYS_AT_MJD2000);
        let seconds = within_day / MICROSECONDS_PER_SECOND;
        let fractional = within_day % MICROSECONDS_PER_SECOND;
        let hour = seconds / 3_600;
        let minute = (seconds % 3_600) / 60;
        let second = seconds % 60;
        let year = if year >= 0 {
            format!("{year:04}")
        } else {
            format!("-{:04}", -year)
        };
        format!("{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{fractional:06}")
    }

    /// Adds a finite number of days with microsecond truncation.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite duration or representational overflow.
    pub fn checked_add_days(self, days: f64) -> Result<Self> {
        let duration = duration_microseconds(days, MICROSECONDS_PER_DAY as f64, "days")?;
        self.checked_add_microseconds(duration)
    }

    /// Subtracts a finite number of days with microsecond truncation.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite duration or representational overflow.
    pub fn checked_sub_days(self, days: f64) -> Result<Self> {
        let duration = duration_microseconds(days, MICROSECONDS_PER_DAY as f64, "days")?;
        self.checked_add_microseconds(duration.checked_neg().ok_or_else(epoch_range_error)?)
    }

    /// Adds a finite number of seconds with microsecond truncation.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite duration or representational overflow.
    pub fn checked_add_seconds(self, seconds: f64) -> Result<Self> {
        let duration = duration_microseconds(seconds, MICROSECONDS_PER_SECOND as f64, "seconds")?;
        self.checked_add_microseconds(duration)
    }

    /// Adds a signed integer microsecond duration.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting epoch is outside the representation.
    pub fn checked_add_microseconds(self, microseconds: i64) -> Result<Self> {
        let microseconds_mjd2000 = self
            .microseconds_mjd2000
            .checked_add(microseconds)
            .ok_or_else(epoch_range_error)?;
        Ok(Self {
            microseconds_mjd2000,
        })
    }

    /// Returns `self - other` as signed integer microseconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the difference exceeds the signed representation.
    pub fn duration_microseconds_since(self, other: Self) -> Result<i64> {
        self.microseconds_mjd2000
            .checked_sub(other.microseconds_mjd2000)
            .ok_or_else(epoch_range_error)
    }

    /// Returns `self - other` in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying microsecond difference overflows.
    pub fn duration_seconds_since(self, other: Self) -> Result<f64> {
        Ok(self.duration_microseconds_since(other)? as f64 / MICROSECONDS_PER_SECOND as f64)
    }

    /// Samples the system clock and truncates it to microsecond resolution.
    ///
    /// # Errors
    ///
    /// Returns an error only when the system clock lies outside the
    /// representation.
    pub fn now() -> Result<Self> {
        let unix_microseconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => i64::try_from(duration.as_micros()).map_err(|_| epoch_range_error())?,
            Err(error) => {
                let magnitude =
                    i64::try_from(error.duration().as_micros()).map_err(|_| epoch_range_error())?;
                magnitude.checked_neg().ok_or_else(epoch_range_error)?
            }
        };
        let y2k_microseconds = UNIX_SECONDS_AT_MJD2000 * MICROSECONDS_PER_SECOND;
        let microseconds_mjd2000 = unix_microseconds
            .checked_sub(y2k_microseconds)
            .ok_or_else(epoch_range_error)?;
        Ok(Self {
            microseconds_mjd2000,
        })
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_iso())
    }
}

impl FromStr for Epoch {
    type Err = PykepError;

    fn from_str(text: &str) -> Result<Self> {
        Self::from_iso(text)
    }
}

fn invalid_calendar(parameter: &'static str, reason: &str) -> PykepError {
    PykepError::InvalidInput {
        parameter,
        reason: reason.into(),
    }
}

fn invalid_iso() -> PykepError {
    PykepError::InvalidInput {
        parameter: "iso",
        reason: "expected YYYY-MM[-DD[THH[:MM[:SS[.ffffff]]]]] with an optional negative and four- or five-digit year".into(),
    }
}

fn epoch_range_error() -> PykepError {
    PykepError::InvalidInput {
        parameter: "epoch",
        reason: "value is outside the signed microsecond representation".into(),
    }
}

fn parse_component<T: FromStr>(text: &str, start: usize, end: usize) -> Result<T> {
    let component = &text[start..end];
    if !component.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid_iso());
    }
    component.parse::<T>().map_err(|_| invalid_iso())
}

fn duration_microseconds(value: f64, scale: f64, parameter: &'static str) -> Result<i64> {
    ensure_finite(parameter, value)?;
    let scaled = value * scale;
    if !scaled.is_finite() || scaled >= i64::MAX as f64 || scaled < i64::MIN as f64 {
        return Err(PykepError::InvalidInput {
            parameter,
            reason: "duration is outside the signed microsecond range".into(),
        });
    }
    Ok(scaled.trunc() as i64)
}

const fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

const fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let adjusted_year = year - i64::from(month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = i64::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days_from_unix: i64) -> (i64, i64, i64) {
    let shifted = days_from_unix + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_epoch_and_cpp_calendar_cases_match() {
        assert_eq!(Epoch::new().to_iso(), "2000-01-01T00:00:00.000000");
        let epoch = Epoch::from_calendar(1980, 10, 17, 11, 36, 21, 121, 841).unwrap();
        assert_eq!(epoch.to_iso(), "1980-10-17T11:36:21.121841");
        assert_eq!(Epoch::from_mjd2000(0.0).unwrap(), Epoch::new());
        assert_eq!(Epoch::from_mjd(51_544.0).unwrap(), Epoch::new());
        assert_eq!(Epoch::from_jd(2_451_544.5).unwrap(), Epoch::new());
    }

    #[test]
    fn iso_crops_and_fractional_precision_are_supported() {
        assert_eq!(
            Epoch::from_iso("2064-10").unwrap().to_iso(),
            "2064-10-01T00:00:00.000000"
        );
        assert_eq!(
            Epoch::from_iso("2064-10-17T11:36:21.1").unwrap().to_iso(),
            "2064-10-17T11:36:21.100000"
        );
        let complete = "2064-10-17T11:36:21.121834";
        assert_eq!(Epoch::from_iso(complete).unwrap().to_iso(), complete);
        for &(year, expected) in &[
            (-44, "-0044-03-15T00:00:00.000000"),
            (12_345, "12345-03-15T00:00:00.000000"),
        ] {
            let epoch = Epoch::from_calendar(year, 3, 15, 0, 0, 0, 0, 0).unwrap();
            assert_eq!(epoch.to_iso(), expected);
            assert_eq!(Epoch::from_iso(expected).unwrap(), epoch);
        }
    }

    #[test]
    fn invalid_calendar_values_are_rejected() {
        for text in [
            "2064-10-",
            "2064-02-30",
            "2064/10",
            "2064-10-17 11",
            "2064-10-17T24",
            "2064-10-17T11:36:21.1218343",
        ] {
            assert!(Epoch::from_iso(text).is_err(), "{text}");
        }
        assert!(Epoch::from_calendar(2001, 2, 29, 0, 0, 0, 0, 0).is_err());
    }

    #[test]
    fn negative_dates_and_arithmetic_keep_microsecond_precision() {
        let value = Epoch::from_mjd2000(-123.456).unwrap();
        assert_eq!(value.to_iso(), "1999-08-30T13:03:21.600000");
        let tomorrow = Epoch::new().checked_add_days(1.0).unwrap();
        assert_eq!(tomorrow.to_iso(), "2000-01-02T00:00:00.000000");
        assert_eq!(
            tomorrow.duration_microseconds_since(Epoch::new()).unwrap(),
            MICROSECONDS_PER_DAY
        );
        assert_eq!(tomorrow.checked_sub_days(1.0).unwrap(), Epoch::new());
        assert_eq!(
            Epoch::new()
                .checked_add_seconds(0.000_001_9)
                .unwrap()
                .microseconds_since_mjd2000(),
            1
        );
    }

    #[test]
    fn date_conversion_round_trips_across_leap_boundaries() {
        for &(year, month, day) in &[
            (1900, 2, 28),
            (2000, 2, 29),
            (2001, 3, 1),
            (2400, 2, 29),
            (-44, 3, 15),
        ] {
            let epoch = Epoch::from_calendar(year, month, day, 0, 0, 0, 0, 0).unwrap();
            assert_eq!(Epoch::from_mjd2000(epoch.mjd2000()).unwrap(), epoch);
        }
    }
}
