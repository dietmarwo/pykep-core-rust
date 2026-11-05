// SPDX-License-Identifier: MPL-2.0

use pykep_core::{PykepError, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::OnceLock};

use crate::problem::invalid;

/// Dynamics family of a TOPS problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TopsFamily {
    /// Cartesian two-body, with mass and thrust controls.
    Twobody,
    /// Prograde modified equinoctial, with mass and thrust controls.
    Mee,
    /// Rotating circular restricted three-body, with mass and thrust controls.
    Cr3bp,
    /// Cartesian solar sail, with two orientation angles.
    Sail,
}

impl TopsFamily {
    /// Families in catalog order.
    pub const ALL: [Self; 4] = [Self::Twobody, Self::Mee, Self::Cr3bp, Self::Sail];

    /// Stable family identifier.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Twobody => "twobody",
            Self::Mee => "mee",
            Self::Cr3bp => "cr3bp",
            Self::Sail => "sail",
        }
    }

    /// Instance keys ordered by source problem number.
    ///
    /// # Errors
    /// Returns an error if the embedded catalog is invalid.
    pub fn ids(self) -> Result<Vec<String>> {
        let mut ids: Vec<_> = data(self)?.keys().cloned().collect();
        ids.sort_by_key(|id| id[1..].parse::<usize>().unwrap_or(usize::MAX));
        Ok(ids)
    }
}

impl std::str::FromStr for TopsFamily {
    type Err = PykepError;

    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|family| family.id() == s)
            .ok_or_else(|| invalid("TOPS family", "expected twobody, mee, cr3bp or sail"))
    }
}

#[derive(Deserialize)]
pub(crate) struct RawCase {
    pub info: String,
    pub state_s: [f64; 6],
    pub state_f: [f64; 6],
    pub tof_bounds: [f64; 2],
    pub period_s: f64,
    pub mu: Option<f64>,
    pub mu_cr3bp: Option<f64>,
    pub m_s: Option<f64>,
    pub veff: Option<f64>,
    pub max_thrust: Option<f64>,
    pub c_sail: Option<f64>,
    #[serde(rename = "L")]
    pub length: f64,
    #[serde(rename = "TIME")]
    pub time: f64,
    #[serde(rename = "MASS")]
    pub mass: Option<f64>,
}

type Data = BTreeMap<String, RawCase>;
static DATA: [OnceLock<Option<Data>>; 4] = [const { OnceLock::new() }; 4];

fn data(family: TopsFamily) -> Result<&'static Data> {
    let index = family as usize;
    let bytes: &[u8] = match family {
        TopsFamily::Twobody => include_bytes!("../data/tops/_tops_twobody.json"),
        TopsFamily::Mee => include_bytes!("../data/tops/_tops_mee.json"),
        TopsFamily::Cr3bp => include_bytes!("../data/tops/_tops_cr3bp.json"),
        TopsFamily::Sail => include_bytes!("../data/tops/_tops_ss.json"),
    };
    DATA[index]
        .get_or_init(|| serde_json::from_slice(bytes).ok())
        .as_ref()
        .ok_or(PykepError::DataUnavailable { dataset: "TOPS" })
}

pub(crate) fn case(family: TopsFamily, id: &str) -> Result<&'static RawCase> {
    data(family)?
        .get(id)
        .ok_or_else(|| invalid("TOPS instance", id))
}

pub(crate) fn required(value: Option<f64>) -> Result<f64> {
    value
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or(PykepError::DataUnavailable {
            dataset: "TOPS scales or parameters",
        })
}
