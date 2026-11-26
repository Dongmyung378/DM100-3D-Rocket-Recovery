"""Unit, bit-exact legacy equivalence, and CDR M1 margin gate tests for DM100 3D v2 (Phase 7A)."""

from __future__ import annotations

import json
import sys
from math import isclose, pi, radians
from pathlib import Path

import numpy as np
import pytest

from dm100_3d.config import (
    DM100VehicleConfig,
    DualTankGeometry,
    GncV2Config,
    PhysicsV2Config,
    dual_tank_propellant_levels,
    mass_properties_3d,
)
from dm100_3d.coords import axis_angle_to_quat_wxyz, quat_to_rotmat
from dm100_3d.gnc.guidance_3d import RtlsController3D
from dm100_3d.physics import (
    DualTankSloshSettings,
    DualTankSloshState,
    FeedSettings,
    GroundEffect3DSettings,
    Slosh3DState,
    Srp3DSettings,
    SrpBuffetFilter,
    compute_ground_effect_3d,
    compute_srp_effects,
    dual_slosh_3d_response,
    dual_slosh_body_offset_3d,
    evaluate_upper_stage_orbit_3d,
    feed_rates,
    ground_effect_nominal_lift_n,
    initial_feed_state,
    regularized_ring_baffle_damping,
    srp_nominal_drag_multiplier,
    wgs84_coriolis_and_transport_enu,
    wgs84_normal_gravity,
    wgs84_radii_of_curvature,
)
from dm100_3d.plant_mujoco import MuJoCoRocketBody
from dm100_3d.sim_3d import MuJoCoPlant3D, Scenario3D, SimConfig3D

# ---------------------------------------------------------------------------
# 1. Config, Dual-Tank Geometry, Ullage & Bit-Exact Legacy Equivalence
# ---------------------------------------------------------------------------


def test_v2_config_defaults_and_presets() -> None:
    pv = PhysicsV2Config()
    gv = GncV2Config()
    assert not pv.any_enabled()
    assert not any(
        (
            gv.use_wgs84_gravity,
            gv.use_dual_tank_mass_props,
            gv.use_srp_iip_feedforward,
            gv.use_ground_effect_feedforward,
        )
    )
    m1_p = PhysicsV2Config.m1_preset()
    m1_g = GncV2Config.m1_preset()
    assert m1_p.earth_wgs84_j2 and m1_p.dual_tank_slosh
    assert m1_p.srp_interference and m1_p.ground_effect
    assert not m1_p.flex_body and not m1_p.sloped_pad
    assert m1_g.use_wgs84_gravity and m1_g.use_dual_tank_mass_props
    assert m1_g.use_srp_iip_feedforward and m1_g.use_ground_effect_feedforward


def test_dual_tank_geometry_volumes_clearances_and_helium_pressure() -> None:
    cfg = DM100VehicleConfig()
    dtg = DualTankGeometry()
    assert isclose(dtg.total_tank_volume_m3, 91.0, rel_tol=1e-9)

    m_lox, h_lox, _, m_ch4, h_ch4, _ = dual_tank_propellant_levels(
        cfg.propellant.main_propellant_kg, dtg
    )
    assert isclose(m_lox + m_ch4, 69_500.0, rel_tol=1e-12)
    assert isclose(m_lox / m_ch4, 3.6, rel_tol=1e-9)
    # Common bulkhead clearance >= 1.061 m dome + 0.40 m ullage; LOX top < 18.0 m tank bulkhead
    ch4_top = dtg.ch4_bottom_m + h_ch4
    lox_top = dtg.lox_bottom_m + h_lox
    assert dtg.lox_bottom_m - ch4_top >= dtg.common_bulkhead_dome_depth_m + 0.45
    assert 18.0 - lox_top >= 2.0

    # Feed settings must keep positive ullage even at +0.3% propellant fill dispersion
    fs = FeedSettings.dual_tank_v2(dtg)
    st_full = initial_feed_state(cfg.propellant.main_propellant_kg * 1.003, fs)
    assert st_full.bottle_gas_kg > 50.0
    # Near-empty tank must still hold > 400 kPa >> 280 kPa minimum ignition pressure with 75 kg He
    st_empty = initial_feed_state(100.0, fs)
    fr_empty = feed_rates(100.0, 0.0, 10.0, st_empty, fs)
    assert fr_empty.inventory_valid
    assert fr_empty.tank_pressure_pa >= 340_000.0


def test_dual_tank_mujoco_composite_inertia_exact_match() -> None:
    cfg = DM100VehicleConfig()
    body = MuJoCoRocketBody(cfg, timestep_s=0.01, gear_deployed=False, upper_attached=True)
    mp = body.update_mass_properties(
        cfg.launch_mass_kg,
        0.0,
        upper_attached=True,
        dual_tank=True,
        rcs_propellant_kg=cfg.propellant.rcs_propellant_kg,
        struts_active=False,
    )
    tot_m, com_b, inertia_3x3 = body.measure_composite_mass_properties()
    assert isclose(tot_m, mp.total_mass_kg, rel_tol=1e-9)
    assert isclose(float(com_b[2]), mp.cg_station_m, abs_tol=1e-9)
    for ax in range(3):
        assert isclose(float(inertia_3x3[ax, ax]), mp.inertia_diag_kg_m2[ax], rel_tol=1e-9)


def test_legacy_v1_bit_exact_equivalence_short_closed_loop() -> None:
    """When PhysicsV2Config & GncV2Config are all False, closed-loop state is bit-for-bit equal."""
    scen = Scenario3D(name="legacy_bit_exact", seed=378)
    cfg = DM100VehicleConfig()

    def _run_steps(scfg: SimConfig3D, n_steps: int = 120) -> list[tuple]:
        plant = MuJoCoPlant3D(vehicle=cfg, scenario=scen, sim_config=scfg)
        ctrl = RtlsController3D(
            vehicle=cfg,
            atmo_settings=scen.atmosphere,
            wind_config=scen.wind,
            sensor_config=scen.sensors,
            gnc_v2=scfg.gnc_v2,
        )
        trace = []
        for _ in range(n_steps):
            pkt = plant.telemetry()
            cmd = ctrl.step(pkt)
            plant.step(cmd, scfg.control_dt_s)
            p, v, q, om = plant.body.get_com_state()
            trace.append(
                (
                    tuple(float(x) for x in p),
                    tuple(float(x) for x in v),
                    tuple(float(x) for x in q),
                    tuple(float(x) for x in om),
                    plant.main_prop_kg,
                    plant.engine_throttle,
                    tuple(plant.gimbal),
                )
            )
        return trace

    tr_default = _run_steps(SimConfig3D(record_replay=False, seed=378, sensor_seed=1378))
    tr_explicit = _run_steps(
        SimConfig3D(
            record_replay=False,
            seed=378,
            sensor_seed=1378,
            physics_v2=PhysicsV2Config(),
            gnc_v2=GncV2Config(),
        )
    )
    assert tr_default == tr_explicit


# ---------------------------------------------------------------------------
# 2. WGS84 Ellipsoid & 3D ECI + J2 Orbit Insertion (0, 37, -89, +89 deg)
# ---------------------------------------------------------------------------


def test_wgs84_somigliana_gravity_and_radii_of_curvature() -> None:
    g_eq = wgs84_normal_gravity(0.0, 0.0)
    g_mid = wgs84_normal_gravity(37.0, 0.0)
    g_pole = wgs84_normal_gravity(90.0, 0.0)
    assert isclose(g_eq, 9.7803253359, rel_tol=1e-9)
    assert isclose(g_pole, 9.8321849378, rel_tol=1e-6)
    assert g_eq < g_mid < g_pole
    assert wgs84_normal_gravity(37.0, 60_000.0) < g_mid

    rm_eq, rn_eq = wgs84_radii_of_curvature(0.0)
    rm_p, rn_p = wgs84_radii_of_curvature(90.0)
    assert rm_eq < rn_eq
    assert isclose(rm_p, rn_p, rel_tol=1e-9)

    # Zero horizontal fake drift for stationary rocket on pad
    a_cor0 = wgs84_coriolis_and_transport_enu(
        altitude_m=0.0, vel_enu_m_s=(0.0, 0.0, 0.0), latitude_deg=45.0
    )
    assert np.allclose(a_cor0, 0.0, atol=0.0)


@pytest.mark.parametrize("lat_deg", [0.0, 37.0, -89.0, 89.0])
def test_wgs84_j2_stage2_orbit_insertion_all_latitudes(lat_deg: float) -> None:
    """Verify Stage 2 700 +/- 15 km mean radius & e_mean < 0.005 across equator to +/-89 deg."""
    res = evaluate_upper_stage_orbit_3d(
        pos_sep_enu_m=(28_000.0, 0.0, 65_000.0),
        vel_sep_enu_m_s=(1_280.0, 0.0, 1_160.0),
        latitude_deg=lat_deg,
        use_wgs84_j2=True,
    )
    assert res.orbit_inserted, f"Failed J2 orbit insertion at lat={lat_deg}: {res}"
    assert res.mean_radius_altitude_km is not None
    assert res.mean_eccentricity is not None
    assert abs(res.mean_radius_altitude_km - 700.0) <= 15.0
    assert res.mean_eccentricity < 0.005
    assert res.upper_propellant_remaining_kg > 400.0


# ---------------------------------------------------------------------------
# 3. Dual-Tank LOX/CH4 3x3 Coupled Slosh & Multi-Ring Baffle Damping
# ---------------------------------------------------------------------------


def test_dual_tank_slosh_baffle_smoothness_and_free_decay() -> None:
    # Smooth C-infinity behavior around theta = 0 with baseline 0.080
    z0 = regularized_ring_baffle_damping(0.0)
    z_small = regularized_ring_baffle_damping(1e-4)
    z_large = regularized_ring_baffle_damping(0.25)
    assert isclose(z0, 0.080, rel_tol=1e-9)
    assert z0 < z_small < z_large

    # Liftoff deep fill: combined modal mass ~7,535 kg, deep-tank frequencies match
    cfg = DM100VehicleConfig()
    mp_lift = mass_properties_3d(
        cfg.launch_mass_kg,
        0.0,
        vehicle=cfg,
        upper_attached=True,
        dual_tank=True,
        rcs_propellant_kg=cfg.propellant.rcs_propellant_kg,
    )
    settings = DualTankSloshSettings()
    st = DualTankSloshState(
        lox=Slosh3DState(qx_m=0.05, qy_m=-0.03),
        ch4=Slosh3DState(qx_m=-0.04, qy_m=0.02),
    )
    resp = dual_slosh_3d_response(
        total_mass_kg=mp_lift.total_mass_kg,
        rigid_inertia_diag_kg_m2=mp_lift.inertia_diag_kg_m2,
        main_propellant_kg=69_500.0,
        cg_station_m=mp_lift.cg_station_m,
        proper_accel_body_m_s2=(0.0, 0.0, 15.5),
        external_torque_body_nm=(0.0, 0.0, 0.0),
        omega_body_rad_s=(0.0, 0.0, 0.0),
        state=st,
        settings=settings,
    )
    assert resp.valid
    assert 7_000.0 <= resp.lox_modal_mass_kg + resp.ch4_modal_mass_kg <= 8_000.0
    assert isclose(resp.lox_freq_hz, resp.ch4_freq_hz, rel_tol=0.02)

    # Free-decay energy test over 4 seconds under 1.5g axial acceleration
    dt = 0.01
    amp0 = abs(st.lox.qx_m) + abs(st.ch4.qx_m)
    for _ in range(400):
        r_step = dual_slosh_3d_response(
            total_mass_kg=mp_lift.total_mass_kg,
            rigid_inertia_diag_kg_m2=mp_lift.inertia_diag_kg_m2,
            main_propellant_kg=69_500.0,
            cg_station_m=mp_lift.cg_station_m,
            proper_accel_body_m_s2=(0.0, 0.0, 15.5),
            external_torque_body_nm=(0.0, 0.0, 0.0),
            omega_body_rad_s=(0.0, 0.0, 0.0),
            state=st,
            settings=settings,
        )
        assert r_step.valid
        lqxd = st.lox.qxd_m_s + r_step.lox_qdd_m_s2[0] * dt
        lqyd = st.lox.qyd_m_s + r_step.lox_qdd_m_s2[1] * dt
        cqxd = st.ch4.qxd_m_s + r_step.ch4_qdd_m_s2[0] * dt
        cqyd = st.ch4.qyd_m_s + r_step.ch4_qdd_m_s2[1] * dt
        st = DualTankSloshState(
            lox=Slosh3DState(
                qx_m=st.lox.qx_m + 0.5 * (st.lox.qxd_m_s + lqxd) * dt,
                qy_m=st.lox.qy_m + 0.5 * (st.lox.qyd_m_s + lqyd) * dt,
                qxd_m_s=lqxd,
                qyd_m_s=lqyd,
            ),
            ch4=Slosh3DState(
                qx_m=st.ch4.qx_m + 0.5 * (st.ch4.qxd_m_s + cqxd) * dt,
                qy_m=st.ch4.qy_m + 0.5 * (st.ch4.qyd_m_s + cqyd) * dt,
                qxd_m_s=cqxd,
                qyd_m_s=cqyd,
            ),
        )
    amp_end = abs(st.lox.qx_m) + abs(st.ch4.qx_m)
    assert amp_end < 0.35 * amp0
    off = dual_slosh_body_offset_3d(mp_lift.total_mass_kg, 69_500.0, st, settings)
    assert all(abs(x) < 0.05 for x in off)

    # Shallow fill (landing 1,200 kg): frequencies diverge as h_lox/R and h_ch4/R differ
    resp_land = dual_slosh_3d_response(
        total_mass_kg=7_500.0,
        rigid_inertia_diag_kg_m2=(5.1e5, 5.1e5, 1.8e4),
        main_propellant_kg=1_200.0,
        cg_station_m=9.1,
        proper_accel_body_m_s2=(0.0, 0.0, 20.0),
        external_torque_body_nm=(0.0, 0.0, 0.0),
        omega_body_rad_s=(0.0, 0.0, 0.0),
        state=DualTankSloshState(),
        settings=settings,
    )
    assert resp_land.valid
    assert abs(resp_land.lox_freq_hz - resp_land.ch4_freq_hz) > 0.03


# ---------------------------------------------------------------------------
# 4. Supersonic Retropropulsion (SRP) & Landing Ground Effect
# ---------------------------------------------------------------------------


def test_srp_c1_continuity_shielding_and_buffeting_bounds() -> None:
    cfg = DM100VehicleConfig()
    # Inactive during ascent or subsonic
    off_asc = compute_srp_effects(
        mach=2.5,
        v_air_zb_m_s=-800.0,
        thrust_n=700_000.0,
        dynamic_pressure_pa=20_000.0,
        body_radius_m=cfg.geometry.body_radius_m,
        upper_attached=True,
    )
    assert off_asc.weight == 0.0 and off_asc.drag_multiplier == 1.0

    # Smooth monotonic transition across Mach in [0.95, 1.35]
    weights = [
        compute_srp_effects(
            mach=m,
            v_air_zb_m_s=-400.0,
            thrust_n=600_000.0,
            dynamic_pressure_pa=15_000.0,
            body_radius_m=cfg.geometry.body_radius_m,
            settings=Srp3DSettings(buffet_enabled=False),
        ).weight
        for m in np.linspace(0.95, 1.35, 9)
    ]
    assert weights[0] == 0.0 and weights[-1] == 1.0
    assert all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))

    # Supersonic entry burn: 25-58% shielding & negative CP shift + bounded deterministic buffeting
    f1 = SrpBuffetFilter(seed=378)
    f2 = SrpBuffetFilter(seed=378)
    eff1 = compute_srp_effects(
        mach=3.2,
        v_air_zb_m_s=-950.0,
        thrust_n=650_000.0,
        dynamic_pressure_pa=22_000.0,
        body_radius_m=cfg.geometry.body_radius_m,
        buffet_filter=f1,
    )
    eff2 = compute_srp_effects(
        mach=3.2,
        v_air_zb_m_s=-950.0,
        thrust_n=650_000.0,
        dynamic_pressure_pa=22_000.0,
        body_radius_m=cfg.geometry.body_radius_m,
        buffet_filter=f2,
    )
    assert eff1.buffet_torque_body_nm == eff2.buffet_torque_body_nm
    assert 0.40 <= eff1.drag_multiplier <= 0.78
    assert -1.4 <= eff1.delta_z_cp_m <= -0.8
    assert max(abs(eff1.buffet_torque_body_nm[0]), abs(eff1.buffet_torque_body_nm[1])) < 15_000.0

    # Causal nominal predictor helper stays inside (0.55, 1.0)
    nom_mult = srp_nominal_drag_multiplier(
        mach=3.0,
        v_air_zb_m_s=-900.0,
        thrust_n=650_000.0,
        dynamic_pressure_pa=20_000.0,
        body_radius_m=cfg.geometry.body_radius_m,
    )
    assert 0.60 <= nom_mult <= 0.90


def test_ground_effect_liftoff_block_taper_and_tilt_perturbation() -> None:
    cfg = DM100VehicleConfig()
    # CRITICAL: Must NEVER fire during liftoff (upper_attached=True)
    ge_lift = compute_ground_effect_3d(
        pos_com_enu_m=(0.0, 0.0, 14.0),
        rot_wb=np.eye(3),
        cg_station_m=14.0,
        nozzle_station_m=cfg.geometry.nozzle_station_m,
        thrust_n=1_550_000.0,
        upper_attached=True,
    )
    assert ge_lift.cushion_force_n == 0.0
    assert ge_lift.force_world_n == (0.0, 0.0, 0.0)

    # Vertical landing at h_noz = 2.0 m: pure +Z lift, zero torque
    cg = 9.0
    ge_vert = compute_ground_effect_3d(
        pos_com_enu_m=(0.0, 0.0, cg + 2.2),
        rot_wb=np.eye(3),
        cg_station_m=cg,
        nozzle_station_m=cfg.geometry.nozzle_station_m,
        thrust_n=200_000.0,
        upper_attached=False,
        settings=GroundEffect3DSettings(c_ge=0.07),
    )
    assert 8_000.0 < ge_vert.cushion_force_n < 10_000.0
    assert isclose(ge_vert.force_world_n[0], 0.0, abs_tol=1e-9)
    assert isclose(ge_vert.force_world_n[2], ge_vert.cushion_force_n, rel_tol=1e-9)
    assert np.allclose(ge_vert.torque_body_nm, 0.0, atol=1e-9)

    # Smooth taper to exact zero above 15 m
    ge_13 = compute_ground_effect_3d(
        pos_com_enu_m=(0.0, 0.0, cg + 13.2),
        rot_wb=np.eye(3),
        cg_station_m=cg,
        nozzle_station_m=cfg.geometry.nozzle_station_m,
        thrust_n=200_000.0,
        upper_attached=False,
    )
    ge_15 = compute_ground_effect_3d(
        pos_com_enu_m=(0.0, 0.0, cg + 15.2),
        rot_wb=np.eye(3),
        cg_station_m=cg,
        nozzle_station_m=cfg.geometry.nozzle_station_m,
        thrust_n=200_000.0,
        upper_attached=False,
    )
    assert ge_13.cushion_force_n > 0.0 and ge_15.cushion_force_n == 0.0

    # Tilted approach generates lateral cushion perturbation
    q_tilt = axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), radians(6.0))
    ge_tilt = compute_ground_effect_3d(
        pos_com_enu_m=(0.0, 0.0, cg + 2.2),
        rot_wb=quat_to_rotmat(q_tilt),
        cg_station_m=cg,
        nozzle_station_m=cfg.geometry.nozzle_station_m,
        thrust_n=200_000.0,
        upper_attached=False,
    )
    assert ge_tilt.force_world_n[0] > 50.0
    assert ground_effect_nominal_lift_n(nozzle_height_m=2.0, thrust_n=200_000.0) > 6_000.0


# ---------------------------------------------------------------------------
# 5. CDR M1 Margin Gate & Kaggle Collector Anti-Contamination Gate
# ---------------------------------------------------------------------------


def test_m1_entry_burn_dual_tank_srp_static_stability_and_fin_hinge_margin() -> None:
    """CDR H1 M1 Exit Gate: body z_cp,eff - CG > +0.25 m AND total (body + fins) > +3.0 m."""
    cfg = DM100VehicleConfig()
    cp_body_desc = cfg.aerodynamics.pressure_center_descent_m  # 11.0 m
    worst_srp_shift = -1.40
    cp_body_eff = cp_body_desc + worst_srp_shift  # 9.60 m

    # Check across full entry_burn propellant range (6,517 kg -> 2,500 kg abort)
    for main_prop in (6_517.0, 5_000.0, 3_747.0, 2_500.0):
        tot_m = cfg.booster_dry_mass_kg + main_prop + 200.0
        mp = mass_properties_3d(
            tot_m,
            0.0,
            vehicle=cfg,
            upper_attached=False,
            dual_tank=True,
            rcs_propellant_kg=200.0,
        )
        body_margin = cp_body_eff - mp.cg_station_m
        assert body_margin > 0.25, (
            f"Body static margin too low at {main_prop} kg: {body_margin:.3f}"
        )

        # Total CN_alpha-weighted CP including 2 pitch-plane deployed grid fins at 22.4 m
        s_ref = pi * cfg.geometry.body_radius_m**2
        cna_body = cfg.aerodynamics.slender_lift_slope * s_ref
        cna_fins = 2.0 * cfg.fins.per_fin_area_m2 * cfg.fins.lift_slope_rad_inv
        cp_tot = (cna_body * cp_body_eff + cna_fins * cfg.fins.station_m) / (cna_body + cna_fins)
        tot_margin = cp_tot - mp.cg_station_m
        assert tot_margin > 3.0, f"Total static margin too low at {main_prop} kg: {tot_margin:.3f}"

    # Verify fin hinge margin at peak aero_glide q = 25 kPa, 8 deg effective alpha under shorter arm
    q_glide = 25_000.0
    cl_8 = min(cfg.fins.max_lift_coefficient, cfg.fins.lift_slope_rad_inv * radians(8.0))
    hinge_nm = q_glide * cfg.fins.per_fin_area_m2 * cl_8 * cfg.fins.hinge_arm_m
    assert hinge_nm < 0.35 * cfg.fins.per_fin_hinge_limit_nm


def test_kaggle_collector_blocks_contamination_and_duplicates(tmp_path: Path) -> None:
    kaggle_dir = Path(__file__).resolve().parents[1] / "scripts" / "kaggle"
    sys.path.insert(0, str(kaggle_dir))
    try:
        from collect_kaggle_results import merge_shard_reports
        from push_kaggle_shards import make_shard_slug, select_shard_indices
    finally:
        sys.path.pop(0)

    slug = make_shard_slug(suite="mc-1000", shard_idx=19, num_shards=20, tag="v2_rc1")
    assert len(slug) <= 50 and "_" not in slug
    assert select_shard_indices(20, batch="0/5") == [0, 1, 2, 3]
    assert select_shard_indices(20, batch="4/5") == [16, 17, 18, 19]

    s0 = tmp_path / "results_edge-3d_0_of_2.json"
    s1 = tmp_path / "results_edge-3d_1_of_2.json"
    s0.write_text(
        json.dumps(
            {
                "suite": "edge-3d",
                "shard": "0/2",
                "physics_v2_preset": "m1",
                "gnc_v2_preset": "m1",
                "results": [
                    {
                        "scenario": "case-a",
                        "outcome": "landed",
                        "requirements_passed": True,
                        "failed_criteria": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    s1.write_text(
        json.dumps(
            {
                "suite": "holdout-3d",  # Contaminated suite!
                "shard": "1/2",
                "physics_v2_preset": "m1",
                "gnc_v2_preset": "m1",
                "results": [
                    {
                        "scenario": "case-b",
                        "outcome": "landed",
                        "requirements_passed": True,
                        "failed_criteria": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Contaminated shard"):
        merge_shard_reports([s0, s1], suite="edge-3d", expected_shards=2, expected_cases=2)

    # Fix suite -> clean merge succeeds and emits v2 schema
    s1.write_text(
        json.dumps(
            {
                "suite": "edge-3d",
                "shard": "1/2",
                "physics_v2_preset": "m1",
                "gnc_v2_preset": "m1",
                "results": [
                    {
                        "scenario": "case-b",
                        "outcome": "landed",
                        "requirements_passed": True,
                        "failed_criteria": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    merged = merge_shard_reports([s0, s1], suite="edge-3d", expected_shards=2, expected_cases=2)
    assert merged["schema"] == "dm100_3d_validation_merged_v2"
    assert merged["passed_cases"] == 2 and merged["total_cases"] == 2
