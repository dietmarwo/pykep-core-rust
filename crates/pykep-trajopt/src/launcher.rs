// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
// Launcher tables are extracted from upstream `pykep/trajopt/_launchers.py` at
// c01e19debad5ea22b7079189f786f473046a79d5.

use pykep_core::{PykepError, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::OnceLock};

/// Tabulated launch mass models from the pinned PyKEP source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launcher {
    /// Ariane 5 from Kourou.
    Ariane5,
    /// Ariane 64 from Kourou.
    Ariane64,
    /// Atlas 501.
    Atlas501,
    /// Atlas 551.
    Atlas551,
    /// Soyuz-Fregat.
    SoyuzF,
    /// Falcon 9.
    Falcon9,
}

#[derive(Deserialize)]
struct Table {
    speed: Vec<f64>,
    declination: Vec<f64>,
    mass: Vec<Vec<f64>>,
}

static TABLES: OnceLock<Option<BTreeMap<String, Table>>> = OnceLock::new();

impl Launcher {
    /// Available tables.
    pub const ALL: [Self; 6] = [
        Self::Ariane5,
        Self::Ariane64,
        Self::Atlas501,
        Self::Atlas551,
        Self::SoyuzF,
        Self::Falcon9,
    ];

    /// Stable table identifier.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Ariane5 => "ariane5",
            Self::Ariane64 => "ariane64",
            Self::Atlas501 => "atlas501",
            Self::Atlas551 => "atlas551",
            Self::SoyuzF => "soyuzf",
            Self::Falcon9 => "falcon9",
        }
    }

    /// Interpolate delivered mass in kg at excess velocity in km/s and declination in degrees.
    ///
    /// Bivariate tables clamp to their boundary. One-dimensional tables use
    /// the upstream 0.1 kg fill value beyond the speed interval.
    ///
    /// # Errors
    /// Returns an error for nonfinite coordinates or invalid embedded data.
    pub fn mass(self, speed: f64, declination: f64) -> Result<f64> {
        if !speed.is_finite() || !declination.is_finite() {
            return Err(PykepError::NonFiniteInput {
                parameter: "launcher coordinates",
            });
        }
        let table = TABLES
            .get_or_init(|| serde_json::from_str(include_str!("../data/launchers.json")).ok())
            .as_ref()
            .and_then(|tables| tables.get(self.id()))
            .ok_or(PykepError::DataUnavailable {
                dataset: "launcher tables",
            })?;
        if table.declination.is_empty()
            && !(table.speed[0]..=table.speed[table.speed.len() - 1]).contains(&speed)
        {
            return Ok(0.1);
        }
        let (j, u) = bracket(&table.speed, speed);
        let row = |i: usize| table.mass[i][j] * (1.0 - u) + table.mass[i][j + 1] * u;
        if table.declination.is_empty() {
            return Ok(row(0));
        }
        let (i, v) = bracket(&table.declination, declination);
        Ok(row(i) * (1.0 - v) + row(i + 1) * v)
    }
}

fn bracket(axis: &[f64], v: f64) -> (usize, f64) {
    let v = v.clamp(axis[0], axis[axis.len() - 1]);
    let i = axis
        .partition_point(|x| *x <= v)
        .saturating_sub(1)
        .min(axis.len() - 2);
    (i, (v - axis[i]) / (axis[i + 1] - axis[i]))
}
