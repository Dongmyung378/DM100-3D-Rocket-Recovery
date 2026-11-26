"""Phase 1 tests: MuJoCo MJCF dynamic 3-axis inertia and 3D 4-leg touchdown / tip-over."""

from __future__ import annotations

from math import radians
from time import perf_counter

import numpy as np

from dm100_3d.config import DM100VehicleConfig, mass_properties_3d
from dm100_3d.coords import axis_angle_to_quat_wxyz, quat_multiply_wxyz
from dm100_3d.plant_mujoco import MuJoCoRocketBody, run_4leg_drop_test


def test_dynamic_composite_mass_com_inertia_matches_exact_spec() -> None:
    """Composite MuJoCo (booster + 4 legs + 4 fins) must match mass_properties_3d() to <1e-6."""
    cfg = DM100VehicleConfig()
    body = MuJoCoRocketBody(cfg, timestep_s=0.01, gear_deployed=False, upper_attached=True)

    test_cases = [
        (100_000.0, 0.0, True, False),  # Launchpad stacked
        (55_000.0, 0.0, True, False),  # Mid-ascent
        (18_500.0, 0.0, False, False),  # Post-separation boostback
        (10_500.0, 0.5, False, False),  # Mid gear deployment
        (8_800.0, 1.0, False, True),  # Pre-touchdown landing booster
        (7_200.0, 1.0, False, True),  # Nearly dry landing booster
    ]
    t0 = perf_counter()
    for mass_kg, dep, upper, struts in test_cases:
        expected = mass_properties_3d(mass_kg, dep, vehicle=cfg, upper_attached=upper)
        body.update_mass_properties(mass_kg, dep, upper_attached=upper, struts_active=struts)
        got_m, got_com, got_i = body.measure_composite_mass_properties()

        assert np.isclose(got_m, expected.total_mass_kg, rtol=1e-9, atol=1e-6)
        assert np.allclose(got_com, [0.0, 0.0, expected.cg_station_m], atol=1e-6)
        assert np.allclose(got_i, np.diag(expected.inertia_diag_kg_m2), rtol=1e-7, atol=1e-3)
    elapsed_ms = (perf_counter() - t0) * 1000.0
    # 6 mj_setConst updates should be well under 50 ms
    assert elapsed_ms < 50.0


def test_vertical_drop_matches_2d_contact_envelope_and_stands_5s() -> None:
    """1.5 m/s vertical touchdown on 4 legs matches 2D stroke/load bounds and settles upright."""
    res = run_4leg_drop_test(
        mass_kg=8_800.0,
        initial_foot_clearance_m=0.10,
        vel_enu_m_s=(0.0, 0.0, -1.5),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    assert res.outcome == "landed"
    assert res.standing_stable is True
    assert not res.body_ground_collision
    assert res.legs_in_contact_final == (True, True, True, True)
    # 2D reference at 1.5 m/s vertical impact: max stroke ~0.35-0.45 m << 0.90 m, load ~1.5-2.3 g
    assert 0.25 < res.max_leg_stroke_m < 0.55
    assert 1.1 < res.max_contact_load_g < 2.6
    assert res.max_leg_force_n < 100_000.0
    assert res.final_tilt_deg < 0.1
    assert res.final_speed_m_s < 0.01
    # All 4 legs share static load symmetrically
    assert np.allclose(res.final_leg_strokes_m, res.final_leg_strokes_m[0], atol=1e-3)


def test_3d_tilted_asymmetric_drop_settles_upright() -> None:
    """Combined 3 deg pitch + 2 deg roll asymmetric touchdown settles cleanly on all 4 legs."""
    q_pitch = axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), radians(3.0))
    q_roll_tilt = axis_angle_to_quat_wxyz((1.0, 0.0, 0.0), radians(-2.0))
    q0 = tuple(float(v) for v in quat_multiply_wxyz(q_pitch, q_roll_tilt))

    res = run_4leg_drop_test(
        mass_kg=8_800.0,
        initial_foot_clearance_m=0.10,
        vel_enu_m_s=(0.0, 0.0, -1.2),
        quat_wxyz=q0,
        omega_body_rad_s=(radians(0.3), radians(-0.4), radians(0.2)),
    )
    assert res.outcome == "landed"
    assert res.standing_stable is True
    assert not res.body_ground_collision
    assert res.max_leg_stroke_m < 0.70
    assert res.max_contact_load_g < 3.0
    assert res.final_tilt_deg < 1.5
    assert res.final_speed_m_s < 0.05
    assert res.final_rate_deg_s < 0.5


def test_3d_lateral_slip_drop_arrested_by_coulomb_friction() -> None:
    """Combined horizontal velocity (0.5 m/s X, -0.35 m/s Y) arrests smoothly without tip-over."""
    q0 = tuple(float(v) for v in axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), radians(2.0)))
    res = run_4leg_drop_test(
        mass_kg=8_800.0,
        initial_foot_clearance_m=0.08,
        vel_enu_m_s=(0.50, -0.35, -1.0),
        quat_wxyz=q0,
    )
    assert res.outcome == "landed"
    assert res.standing_stable is True
    assert not res.body_ground_collision
    assert res.max_leg_stroke_m < 0.75
    assert res.final_tilt_deg < 1.0
    assert res.final_speed_m_s < 0.08
    assert res.final_rate_deg_s < 0.5


def test_extreme_tilt_and_lateral_speed_genuinely_tips_over() -> None:
    """Excessive tilt + horizontal velocity must trigger physical 3D tip-over / hull collision."""
    q0 = tuple(float(v) for v in axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), radians(24.0)))
    res = run_4leg_drop_test(
        mass_kg=8_800.0,
        initial_foot_clearance_m=0.10,
        vel_enu_m_s=(4.0, 1.5, -1.8),
        quat_wxyz=q0,
        omega_body_rad_s=(0.0, radians(15.0), 0.0),
    )
    assert res.standing_stable is False
    assert res.outcome in {"tip_over", "body_ground_collision"}
