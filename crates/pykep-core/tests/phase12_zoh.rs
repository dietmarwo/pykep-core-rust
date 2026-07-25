// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Zero-order-hold model and switching-contract validation.

use pykep_core::PykepError;
use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics,
    ZohSensitivitySeeds, ZohSolarSailDynamics, propagate_schedule, propagate_schedule_backward,
    propagate_schedule_with_sensitivities,
};
use pykep_core::dynamics::{Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{DynamicsModel, IntegratorOptions};
use serde_json::Value;

const OPTIONS: IntegratorOptions = IntegratorOptions {
    relative_tolerance: 2e-13,
    absolute_tolerance: 2e-13,
    initial_step: None,
    maximum_step: Some(0.01),
    maximum_steps: 100_000,
    maximum_rejections: 100,
};

fn assert_state<const N: usize>(actual: [f64; N], expected: [f64; N], tolerance: f64) {
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "component {index}: {actual:.17e} != {expected:.17e}"
        );
    }
}

fn parse(encoded: &str) -> f64 {
    let (sign, unsigned) = encoded
        .strip_prefix('-')
        .map_or((1.0, encoded), |rest| (-1.0, rest));
    let unsigned = unsigned.strip_prefix("0x").unwrap();
    let (significand, exponent) = unsigned.split_once('p').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (integer, fraction) = significand.split_once('.').unwrap_or((significand, ""));
    let digits = format!("{integer}{fraction}");
    sign * u64::from_str_radix(&digits, 16).unwrap() as f64
        * 2.0_f64.powi(exponent - 4 * fraction.len() as i32)
}

fn array<const N: usize>(value: &Value) -> [f64; N] {
    let values = value.as_array().unwrap();
    assert_eq!(values.len(), N);
    core::array::from_fn(|index| parse(values[index].as_str().unwrap()))
}

fn assert_sensitivities<const N: usize, const W: usize>(
    actual: [[f64; W]; N],
    expected: &[f64],
    tolerance: f64,
) {
    assert_eq!(expected.len(), N * W);
    for (index, (&actual, &expected)) in actual.iter().flatten().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "sensitivity {index}: {actual:.17e} != {expected:.17e}"
        );
    }
}

#[test]
fn schedules_validate_once_and_own_switches_deterministically() {
    let schedule = ControlSchedule::new(vec![0.0, 1.0, 3.0], vec![[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(schedule.len(), 2);
    assert!(!schedule.is_empty());
    assert_eq!(schedule.initial_time(), 0.0);
    assert_eq!(schedule.final_time(), 3.0);
    assert_eq!(schedule.control_at(0.0).unwrap(), [1.0, 2.0]);
    assert_eq!(schedule.control_at(0.999).unwrap(), [1.0, 2.0]);
    assert_eq!(schedule.control_at(1.0).unwrap(), [3.0, 4.0]);
    assert_eq!(schedule.control_at(3.0).unwrap(), [3.0, 4.0]);
    assert!(schedule.control_at(-0.1).is_err());
    assert!(schedule.control_at(f64::NAN).is_err());

    assert!(matches!(
        ControlSchedule::<2>::new(vec![0.0], vec![]),
        Err(PykepError::InvalidInput {
            parameter: "controls",
            ..
        })
    ));
    assert!(matches!(
        ControlSchedule::new(vec![0.0, 1.0], vec![[0.0], [1.0]]),
        Err(PykepError::DimensionMismatch { .. })
    ));
    assert!(ControlSchedule::new(vec![0.0, 1.0, 1.0], vec![[0.0], [1.0]]).is_err());
    assert!(ControlSchedule::new(vec![0.0, f64::INFINITY], vec![[0.0]]).is_err());
    assert!(ControlSchedule::new(vec![0.0, 1.0], vec![[f64::NAN]]).is_err());
}

#[test]
fn all_upstream_single_segment_regressions_are_represented() {
    let kepler_initial = [
        0.273_027_499_717_861_67,
        -0.230_376_670_222_849_58,
        -0.905_109_157_146_030_8,
        0.910_505_479_231_563_7,
        0.812_101_873_206_247_9,
        -0.086_060_890_292_185_55,
        0.920_800_214_245_817,
    ];
    let kepler_control = [
        0.032_171_291_371_278_32,
        0.736_829_960_492_892_4,
        0.651_813_096_864_986_6,
        -0.179_502_913_835_174_22,
    ];
    let kepler = propagate_schedule(
        &ZohKeplerDynamics,
        &ControlSchedule::new(vec![0.0, 0.159_992_799_610_298_58], vec![kepler_control]).unwrap(),
        kepler_initial,
        [0.572_330_848_010_41],
        OPTIONS,
    )
    .unwrap();
    assert_state(
        kepler.state,
        [
            0.414_697_325_701_195_66,
            -0.097_621_603_345_794_81,
            -0.906_676_177_493_356_2,
            0.857_349_216_227_101_7,
            0.843_326_670_227_938_2,
            0.064_314_045_044_228_4,
            0.917_854_327_228_336,
        ],
        2e-12,
    );

    let cr3bp_initial = [
        0.750_582_277_078_230_7,
        -0.563_387_291_054_110_9,
        -0.361_721_725_165_381_7,
        -0.618_076_526_405_485_7,
        -0.641_867_652_950_951_5,
        -0.093_395_468_373_456_8,
        1.045_013_508_427_182_3,
    ];
    let cr3bp = propagate_schedule(
        &ZohCr3bpDynamics,
        &ControlSchedule::new(
            vec![0.0, 1.947_313_369_265_437_2],
            vec![[
                0.001_342_869_355_874_922,
                -0.764_720_337_256_993_6,
                -0.532_433_704_397_195_7,
                -0.362_928_582_792_027_4,
            ]],
        )
        .unwrap(),
        cr3bp_initial,
        [0.426_535_468_540_315_1, 0.486_381_383_021_788_2],
        OPTIONS,
    )
    .unwrap();
    assert_state(
        cr3bp.state,
        [
            0.018_228_316_270_594_77,
            -0.152_000_558_963_279_9,
            -0.386_399_174_033_396_96,
            -0.721_640_487_914_744_4,
            0.630_823_381_562_888_9,
            -0.066_579_632_776_004_98,
            1.043_898_123_530_023_8,
        ],
        3e-10,
    );

    let equinoctial_initial = [
        1.054_274_337_943_761_8,
        -0.710_842_284_896_237,
        -0.381_216_006_131_334_44,
        -0.363_231_478_692_526_75,
        0.296_896_967_886_767_1,
        3.311_918_879_648_033_4,
        0.691_032_778_029_835_6,
    ];
    let equinoctial = propagate_schedule(
        &ZohEquinoctialDynamics,
        &ControlSchedule::new(
            vec![0.0, 1.195_089_119_785_219],
            vec![[
                0.046_078_053_456_509_51,
                -0.132_670_099_899_888_2,
                0.827_758_966_310_986_9,
                -0.545_173_126_891_192_6,
            ]],
        )
        .unwrap(),
        equinoctial_initial,
        [0.897_848_121_507_555_6],
        OPTIONS,
    )
    .unwrap();
    assert_state(
        equinoctial.state,
        [
            1.178_047_723_382_589_7,
            -0.742_109_858_493_202_4,
            -0.505_186_086_091_061_5,
            -0.365_569_021_878_626_4,
            0.315_766_032_943_834_7,
            5.412_410_892_975_656,
            0.641_590_634_029_158_5,
        ],
        2e-11,
    );

    let solar_sail = propagate_schedule(
        &ZohSolarSailDynamics,
        &ControlSchedule::new(vec![0.0, 0.75], vec![[0.25, -1.1]]).unwrap(),
        [0.8, -0.4, 0.3, 0.2, 0.9, -0.1],
        [0.04],
        OPTIONS,
    )
    .unwrap();
    assert_state(
        solar_sail.state,
        [
            0.616_738_469_029_390_4,
            0.324_622_547_028_165_3,
            0.122_031_764_181_760_16,
            -0.788_614_844_568_717_3,
            0.871_449_474_076_843_9,
            -0.375_150_386_713_481_5,
        ],
        2e-11,
    );
}

#[test]
fn zero_control_reduces_to_uncontrolled_models() {
    let cartesian = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let mut zoh_kepler_rhs = [0.0; 7];
    ZohKeplerDynamics
        .rhs(
            0.0,
            &[0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 2.0],
            &[0.0; 5],
            &mut zoh_kepler_rhs,
        )
        .unwrap();
    assert_eq!(
        &zoh_kepler_rhs[..6],
        &KeplerDynamics.evaluate(&cartesian, 1.0).unwrap()
    );
    assert_eq!(zoh_kepler_rhs[6], 0.0);

    let mut zoh_cr3bp_rhs = [0.0; 7];
    ZohCr3bpDynamics
        .rhs(
            0.0,
            &[0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 2.0],
            &[0.0, 0.0, 0.0, 0.0, 0.4, 0.01],
            &mut zoh_cr3bp_rhs,
        )
        .unwrap();
    assert_state(
        zoh_cr3bp_rhs[..6].try_into().unwrap(),
        Cr3bpDynamics.evaluate(&cartesian, 0.01).unwrap(),
        2e-16,
    );
    assert_eq!(zoh_cr3bp_rhs[6], 0.0);

    let mut equinoctial_rhs = [0.0; 7];
    ZohEquinoctialDynamics
        .rhs(
            0.0,
            &[1.2, 0.1, -0.2, 0.3, -0.1, 0.7, 2.0],
            &[0.0; 5],
            &mut equinoctial_rhs,
        )
        .unwrap();
    assert_eq!(equinoctial_rhs[..5], [0.0; 5]);
    assert_eq!(equinoctial_rhs[6], 0.0);
    assert!(equinoctial_rhs[5] > 0.0);

    let mut sail_rhs = [0.0; 6];
    ZohSolarSailDynamics
        .rhs(0.0, &cartesian, &[0.2, -0.7, 0.0], &mut sail_rhs)
        .unwrap();
    assert_state(
        sail_rhs,
        KeplerDynamics.evaluate(&cartesian, 1.0).unwrap(),
        2e-16,
    );
}

#[test]
fn multiple_switches_reverse_exactly_and_match_manual_segments() {
    let controls = vec![
        [0.01, 1.0, 0.0, 0.0],
        [0.02, 0.0, 1.0, 0.0],
        [0.015, 0.0, 0.0, -1.0],
    ];
    let schedule = ControlSchedule::new(vec![0.0, 0.2, 0.5, 0.9], controls.clone()).unwrap();
    let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5];
    let complete =
        propagate_schedule(&ZohKeplerDynamics, &schedule, initial, [0.02], OPTIONS).unwrap();

    let first = propagate_schedule(
        &ZohKeplerDynamics,
        &ControlSchedule::new(vec![0.0, 0.2], vec![controls[0]]).unwrap(),
        initial,
        [0.02],
        OPTIONS,
    )
    .unwrap();
    let second = propagate_schedule(
        &ZohKeplerDynamics,
        &ControlSchedule::new(vec![0.2, 0.5], vec![controls[1]]).unwrap(),
        first.state,
        [0.02],
        OPTIONS,
    )
    .unwrap();
    let third = propagate_schedule(
        &ZohKeplerDynamics,
        &ControlSchedule::new(vec![0.5, 0.9], vec![controls[2]]).unwrap(),
        second.state,
        [0.02],
        OPTIONS,
    )
    .unwrap();
    assert_eq!(complete.state, third.state);

    let reversed = propagate_schedule_backward(
        &ZohKeplerDynamics,
        &schedule,
        complete.state,
        [0.02],
        OPTIONS,
    )
    .unwrap();
    assert_state(reversed.state, initial, 2e-11);
}

#[test]
fn segment_control_sensitivities_activate_at_their_switch() {
    let schedule = ControlSchedule::new(
        vec![0.0, 0.2, 0.5],
        vec![[0.01, 1.0, 0.0, 0.0], [0.02, 0.0, 1.0, 0.0]],
    )
    .unwrap();
    let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5];
    let mut control_seeds = vec![[[0.0; 3]; 4]; 2];
    control_seeds[0][0][0] = 1.0;
    control_seeds[1][0][1] = 1.0;
    let mut initial_seeds = [[0.0; 3]; 7];
    initial_seeds[0][2] = 1.0;
    let seeds = ZohSensitivitySeeds {
        initial_state: initial_seeds,
        segment_controls: control_seeds,
        constants: [[0.0; 3]],
    };
    let sensitivities = propagate_schedule_with_sensitivities(
        &ZohKeplerDynamics,
        &schedule,
        initial,
        [0.02],
        &seeds,
        OPTIONS,
    )
    .unwrap();

    let first_only = ControlSchedule::new(vec![0.0, 0.2], vec![[0.01, 1.0, 0.0, 0.0]]).unwrap();
    let prefix_seeds = ZohSensitivitySeeds {
        initial_state: initial_seeds,
        segment_controls: vec![[[1.0, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]]],
        constants: [[0.0; 3]],
    };
    let prefix = propagate_schedule_with_sensitivities(
        &ZohKeplerDynamics,
        &first_only,
        initial,
        [0.02],
        &prefix_seeds,
        OPTIONS,
    )
    .unwrap();
    assert!(prefix.sensitivities.iter().all(|row| row[1] == 0.0));
    assert!(
        sensitivities
            .sensitivities
            .iter()
            .any(|row| row[1].abs() > 1e-8)
    );

    for control_column in 0..2 {
        let step = 2e-6;
        let mut plus_controls = schedule.controls().to_vec();
        let mut minus_controls = schedule.controls().to_vec();
        plus_controls[control_column][0] += step;
        minus_controls[control_column][0] -= step;
        let plus = propagate_schedule(
            &ZohKeplerDynamics,
            &ControlSchedule::new(schedule.boundaries().to_vec(), plus_controls).unwrap(),
            initial,
            [0.02],
            OPTIONS,
        )
        .unwrap();
        let minus = propagate_schedule(
            &ZohKeplerDynamics,
            &ControlSchedule::new(schedule.boundaries().to_vec(), minus_controls).unwrap(),
            initial,
            [0.02],
            OPTIONS,
        )
        .unwrap();
        for row in 0..7 {
            let finite = (plus.state[row] - minus.state[row]) / (2.0 * step);
            assert!((sensitivities.sensitivities[row][control_column] - finite).abs() < 2e-7);
        }
    }

    let bad_seeds = ZohSensitivitySeeds {
        initial_state: [[0.0; 1]; 7],
        segment_controls: vec![[[0.0; 1]; 4]],
        constants: [[0.0; 1]],
    };
    assert!(matches!(
        propagate_schedule_with_sensitivities(
            &ZohKeplerDynamics,
            &schedule,
            initial,
            [0.02],
            &bad_seeds,
            OPTIONS
        ),
        Err(PykepError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn singular_states_are_reported_before_integration() {
    let schedule = ControlSchedule::new(vec![0.0, 1.0], vec![[0.0; 4]]).unwrap();
    assert!(matches!(
        propagate_schedule(
            &ZohKeplerDynamics,
            &schedule,
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0],
            OPTIONS
        ),
        Err(PykepError::SingularGeometry { .. })
    ));
    assert!(matches!(
        propagate_schedule(
            &ZohEquinoctialDynamics,
            &schedule,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0],
            OPTIONS
        ),
        Err(PykepError::InvalidInput {
            parameter: "semilatus_rectum",
            ..
        })
    ));
}

#[test]
fn constant_segment_states_and_variations_match_cpp_taylor_reference() {
    let document: Value = serde_json::from_str(include_str!("data/phase12-v1.json")).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(
        document["upstream_commit"],
        "53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e"
    );
    let models = document["models"].as_array().unwrap();

    let kepler = &models[0];
    assert_eq!(kepler["name"], "zoh_kepler");
    let kepler_parameters = array::<5>(&kepler["parameters"]);
    let mut initial_seeds = [[0.0; 11]; 7];
    for (index, row) in initial_seeds.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut control_seeds = [[0.0; 11]; 4];
    for (index, row) in control_seeds.iter_mut().enumerate() {
        row[index + 7] = 1.0;
    }
    let result = propagate_schedule_with_sensitivities(
        &ZohKeplerDynamics,
        &ControlSchedule::new(
            vec![0.0, parse(kepler["final_time"].as_str().unwrap())],
            vec![kepler_parameters[..4].try_into().unwrap()],
        )
        .unwrap(),
        array(&kepler["initial_state"]),
        [kepler_parameters[4]],
        &ZohSensitivitySeeds {
            initial_state: initial_seeds,
            segment_controls: vec![control_seeds],
            constants: [[0.0; 11]],
        },
        OPTIONS,
    )
    .unwrap();
    assert_state(result.state, array(&kepler["final_state"]), 2e-12);
    assert_sensitivities(
        result.sensitivities,
        &encoded_values(&kepler["sensitivities"]),
        2e-7,
    );

    let cr3bp = &models[1];
    assert_eq!(cr3bp["name"], "zoh_cr3bp");
    let parameters = array::<6>(&cr3bp["parameters"]);
    let mut initial_seeds = [[0.0; 11]; 7];
    for (index, row) in initial_seeds.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut control_seeds = [[0.0; 11]; 4];
    for (index, row) in control_seeds.iter_mut().enumerate() {
        row[index + 7] = 1.0;
    }
    let result = propagate_schedule_with_sensitivities(
        &ZohCr3bpDynamics,
        &ControlSchedule::new(
            vec![0.0, parse(cr3bp["final_time"].as_str().unwrap())],
            vec![parameters[..4].try_into().unwrap()],
        )
        .unwrap(),
        array(&cr3bp["initial_state"]),
        [parameters[4], parameters[5]],
        &ZohSensitivitySeeds {
            initial_state: initial_seeds,
            segment_controls: vec![control_seeds],
            constants: [[0.0; 11]; 2],
        },
        OPTIONS,
    )
    .unwrap();
    assert_state(result.state, array(&cr3bp["final_state"]), 3e-10);
    assert_sensitivities(
        result.sensitivities,
        &encoded_values(&cr3bp["sensitivities"]),
        2e-5,
    );

    let equinoctial = &models[2];
    assert_eq!(equinoctial["name"], "zoh_equinoctial");
    let parameters = array::<5>(&equinoctial["parameters"]);
    let mut initial_seeds = [[0.0; 11]; 7];
    for (index, row) in initial_seeds.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut control_seeds = [[0.0; 11]; 4];
    for (index, row) in control_seeds.iter_mut().enumerate() {
        row[index + 7] = 1.0;
    }
    let result = propagate_schedule_with_sensitivities(
        &ZohEquinoctialDynamics,
        &ControlSchedule::new(
            vec![0.0, parse(equinoctial["final_time"].as_str().unwrap())],
            vec![parameters[..4].try_into().unwrap()],
        )
        .unwrap(),
        array(&equinoctial["initial_state"]),
        [parameters[4]],
        &ZohSensitivitySeeds {
            initial_state: initial_seeds,
            segment_controls: vec![control_seeds],
            constants: [[0.0; 11]],
        },
        OPTIONS,
    )
    .unwrap();
    assert_state(result.state, array(&equinoctial["final_state"]), 2e-11);
    assert_sensitivities(
        result.sensitivities,
        &encoded_values(&equinoctial["sensitivities"]),
        2e-6,
    );

    let sail = &models[3];
    assert_eq!(sail["name"], "zoh_solar_sail");
    let parameters = array::<3>(&sail["parameters"]);
    let mut initial_seeds = [[0.0; 8]; 6];
    for (index, row) in initial_seeds.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut control_seeds = [[0.0; 8]; 2];
    for (index, row) in control_seeds.iter_mut().enumerate() {
        row[index + 6] = 1.0;
    }
    let result = propagate_schedule_with_sensitivities(
        &ZohSolarSailDynamics,
        &ControlSchedule::new(
            vec![0.0, parse(sail["final_time"].as_str().unwrap())],
            vec![parameters[..2].try_into().unwrap()],
        )
        .unwrap(),
        array(&sail["initial_state"]),
        [parameters[2]],
        &ZohSensitivitySeeds {
            initial_state: initial_seeds,
            segment_controls: vec![control_seeds],
            constants: [[0.0; 8]],
        },
        OPTIONS,
    )
    .unwrap();
    assert_state(result.state, array(&sail["final_state"]), 2e-11);
    assert_sensitivities(
        result.sensitivities,
        &encoded_values(&sail["sensitivities"]),
        2e-6,
    );
}

fn encoded_values(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| parse(value.as_str().unwrap()))
        .collect()
}
