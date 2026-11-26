"""Tier-1 unit tests, 9-regime Bode CI gate, and M2 corner stress tests (Phase 7B + 8A)."""

from __future__ import annotations

from math import degrees, isclose, pi, radians, sin

import numpy as np

from dm100_3d.config import DM100VehicleConfig, GncV2Config, PhysicsV2Config
from dm100_3d.coords import axis_angle_to_quat_wxyz, quat_to_rotmat
from dm100_3d.gnc.mekf import CausalEstimator3D, SensorConfig3D
from dm100_3d.gnc.notch_filter import StructuralFilter3D
from dm100_3d.gnc.stability_analysis import verify_all_nine_regimes
from dm100_3d.interfaces import ActuatorCommand3D, TelemetryPacket3D
from dm100_3d.physics import (
    AtmosphereLUT,
    DrydenTurbulence3D,
    DrydenTurbulenceSettings,
    FlexBodyState,
    SensorSuiteV2,
    SensorV2Settings,
    Wind3DLUT,
    compute_base_slant_radar_range_m,
    compute_flex_modal_properties,
    free_free_mode_eval,
    project_flex_state_at_separation,
    step_flex_body_3d,
)
from dm100_3d.plant_mujoco import MuJoCoRocketBody
from dm100_3d.sim_3d import MuJoCoPlant3D, Scenario3D, SimConfig3D

# ---------------------------------------------------------------------------
# 1. NASA SP-8003 Flex-Body Modes, Follower-Force Moment & L2 Projection
# ---------------------------------------------------------------------------


def test_flex_body_eigenfunctions_generalized_mass_and_follower_force() -> None:
    cfg = DM100VehicleConfig()
    # Normalized free-free boundary values |phi_k(0)| = |phi_k(L)| = 1.0
    for k in range(2):
        p0, s0, _ = free_free_mode_eval(0.0, 33.0, k)
        pl, _, _ = free_free_mode_eval(33.0, 33.0, k)
        assert isclose(p0, 1.0, rel_tol=1e-9)
        assert isclose(abs(pl), 1.0, rel_tol=1e-9)
        # Explicit linear extrapolation at z_noz = -0.2 m (CDR §2.1)
        pn, sn, kn = free_free_mode_eval(-0.2, 33.0, k)
        assert isclose(pn, p0 - 0.2 * s0, rel_tol=1e-12)
        assert isclose(sn, s0, rel_tol=1e-12)
        assert kn == 0.0

    # Generalized mass M_k / m in [0.21, 0.35] across ascent & booster regimes
    for upper, prop in ((True, 69_500.0), (True, 12_300.0), (False, 12_300.0), (False, 1_500.0)):
        m_tot = (
            cfg.booster_dry_mass_kg
            + prop
            + 200.0
            + (cfg.upper_stage.total_mass_kg if upper else 0.0)
        )
        props = compute_flex_modal_properties(
            total_mass_kg=m_tot,
            main_propellant_kg=prop,
            rcs_propellant_kg=200.0,
            gear_extension=0.0,
            upper_attached=upper,
            vehicle=cfg,
        )
        for mk in props.gen_mass_kg:
            assert 0.21 * m_tot - 1e-6 <= mk <= 0.35 * m_tot + 1e-6
        if upper:
            assert 2.3 <= props.freq_hz[0] <= 4.3
            assert 6.1 <= props.freq_hz[1] <= 10.6
        else:
            assert 4.4 <= props.freq_hz[0] <= 6.9
            assert 10.4 <= props.freq_hz[1] <= 12.6

    # Multi-rate 500 Hz step returns 5 samples per 0.01 s, zero roll coupling, follower offset
    props = compute_flex_modal_properties(
        total_mass_kg=cfg.launch_mass_kg,
        main_propellant_kg=69_500.0,
        rcs_propellant_kg=300.0,
        gear_extension=0.0,
        upper_attached=True,
        vehicle=cfg,
    )
    st0 = FlexBodyState(eta_x=(0.02, 0.0), etad_x=(0.0, 0.0))
    samples = step_flex_body_3d(
        state=st0,
        dt_s=0.01,
        modal_props=props,
        cg_station_m=15.0,
        thrust_axial_n=1_500_000.0,
        tvc_lateral_force_body_xy_n=(20_000.0, 0.0),
        fin_force_body_xy_n=(0.0, 0.0),
        aero_force_body_xy_n=(0.0, 0.0),
        rcs_torque_body_xy_nm=(0.0, 0.0),
        cp_station_m=11.5,
        vehicle=cfg,
    )
    assert len(samples) == 5
    # Zero roll coupling in IMU omega & rigid torque
    assert all(fs.imu_omega_delta_rad_s[2] == 0.0 for fs in samples)
    assert all(fs.rigid_torque_delta_body_nm[2] == 0.0 for fs in samples)
    # Non-zero follower-force pitch moment & IMU rate contamination
    assert abs(samples[-1].rigid_torque_delta_body_nm[1]) > 1_000.0
    assert abs(samples[-1].imu_omega_delta_rad_s[1]) > 1e-4

    # Stage separation L2 projection preserves physical continuity on [0, 24 m]
    st_sep = project_flex_state_at_separation(FlexBodyState(eta_x=(0.04, -0.01), etad_x=(0.1, 0.0)))
    assert all(np.isfinite(st_sep.eta_x)) and all(np.isfinite(st_sep.etad_x))
    assert 0.005 < abs(st_sep.eta_x[0]) < 0.08


# ---------------------------------------------------------------------------
# 2. Single-Path 4th-Order Elliptic CSI Filter & Mandatory 9-Regime Bode Gate
# ---------------------------------------------------------------------------


def test_elliptic_csi_filter_attenuation_roll_bypass_and_bumpless_switch() -> None:
    filt = StructuralFilter3D(upper_attached=True)
    # Feed pure 5.5 Hz bending sine on X/Y and 5.5 Hz test signal on Roll Z for 4 s
    xs_out, zs_out = [], []
    for k in range(200):
        t = k * 0.02
        val = sin(2.0 * pi * 5.5 * t)
        y = filt.filter_sample((val, -val, val), upper_attached=True)
        if k >= 120:
            xs_out.append(abs(float(y[0])))
            zs_out.append(abs(float(y[2])))
    # Pitch/Yaw heavily attenuated (~ -42 dB -> amplitude < 0.015), Roll exact unity bypass!
    assert max(xs_out) < 0.015
    assert max(zs_out) > 0.95

    # Bumpless regime switch to booster (2.25 Hz) preserves steady-state DC continuity
    filt.reset((0.25, -0.15), upper_attached=True)
    y_before = filt.filter_sample((0.25, -0.15, 0.1), upper_attached=True)
    y_after = filt.filter_sample((0.25, -0.15, 0.1), upper_attached=False)
    dc_expected = 0.25 * (10.0 ** (-0.5 / 20.0))
    assert isclose(float(y_before[0]), dc_expected, abs_tol=1e-6)
    assert isclose(float(y_after[0]), float(y_before[0]), abs_tol=1e-6)


def test_cdr_nine_regime_bode_stability_ci_gate() -> None:
    """Mandatory CI Gate (CDR Rev 3.1 §3.1): All 9 regimes GM>=6 dB, PM>=35 deg, Flex<=-10 dB."""
    reports = verify_all_nine_regimes()
    assert len(reports) == 9
    for rep in reports:
        assert rep.worst_gm_db >= 6.0, f"{rep.name} GM={rep.worst_gm_db:.2f} dB < 6.0"
        assert rep.worst_pm_deg >= 35.0, f"{rep.name} PM={rep.worst_pm_deg:.1f} deg < 35.0"
        assert rep.worst_flex1_peak_db <= -10.0, f"{rep.name} Flex1={rep.worst_flex1_peak_db:.1f}"
        assert rep.worst_flex2_peak_db <= -10.0, f"{rep.name} Flex2={rep.worst_flex2_peak_db:.1f}"
        assert rep.passed, f"Regime failed Bode gate: {rep}"


# ---------------------------------------------------------------------------
# 3. Dryden Clamped Turbulence & Finite 15 m Sloped Pad (z = 0 Outer Terrain)
# ---------------------------------------------------------------------------


def test_dryden_clamping_determinism_and_jet_shear() -> None:
    s = DrydenTurbulenceSettings(sigma_m_s=2.5, jet_shear_peak_m_s=20.0, jet_shear_azimuth_deg=90.0)
    d1 = DrydenTurbulence3D(seed=378, settings=s)
    d2 = DrydenTurbulence3D(seed=378, settings=s)
    # Near-ground zero-airspeed clamping must remain finite & bounded
    g1 = d1.step(altitude_m=0.0, airspeed_m_s=0.0, dt_s=0.01)
    g2 = d2.step(altitude_m=0.0, airspeed_m_s=0.0, dt_s=0.01)
    assert g1 == g2
    assert all(np.isfinite(g1)) and max(abs(x) for x in g1) < 12.0

    # At 10.5 km jet center, +Y jet shear peak ~ +20 m/s
    g_jet = d1.step(altitude_m=10_500.0, airspeed_m_s=350.0, dt_s=0.01)
    assert 12.0 <= g_jet[1] <= 28.0
    # In vacuum (50 km), Dryden gust tapers cleanly to 0
    g_vac = d1.step(altitude_m=50_000.0, airspeed_m_s=1_200.0, dt_s=0.01)
    assert all(abs(x) < 1e-6 for x in g_vac)


def test_finite_15m_sloped_pad_zero_ascent_penetration_and_outer_z0() -> None:
    cfg = DM100VehicleConfig()
    body = MuJoCoRocketBody(
        cfg,
        timestep_s=0.002,
        gear_deployed=False,
        upper_attached=True,
        sloped_pad_enabled=True,
        pad_slope_deg=2.5,
        pad_slope_azimuth_deg=35.0,
        pad_radius_m=15.0,
    )
    # During ascent (upper_attached=True): flat launchpad z = 0 -> ZERO t=0 nozzle penetration
    body.update_ground_surface(upper_attached=True, pos_xy_m=(0.0, 0.0))
    z_asc, n_asc = body.surface_height_and_normal(5.0, 0.0, upper_attached=True)
    assert z_asc == 0.0 and np.allclose(n_asc, (0.0, 0.0, 1.0))

    # Post-separation: finite r=15 m sloped pad active, outer terrain at r=25 km is exact z = 0.0 m
    body.update_ground_surface(upper_attached=False, pos_xy_m=(0.0, 0.0))
    z_pad, n_pad = body.surface_height_and_normal(8.0, 0.0, upper_attached=False)
    assert abs(z_pad) > 0.25
    assert isclose(degrees(np.arccos(float(n_pad[2]))), 2.5, abs_tol=1e-6)
    z_sea, n_sea = body.surface_height_and_normal(25_000.0, 0.0, upper_attached=False)
    assert z_sea == 0.0 and np.allclose(n_sea, (0.0, 0.0, 1.0))

    # Vertical touchdown drop test onto 2.5 deg sloped pad must settle cleanly on all 4 legs
    scen = Scenario3D(name="sloped_pad_drop", seed=77)
    pv2 = PhysicsV2Config(
        earth_wgs84_j2=True,
        dual_tank_slosh=True,
        sloped_pad=True,
        pad_slope_deg=2.5,
        pad_slope_azimuth_deg=45.0,
    )
    scfg = SimConfig3D(
        record_replay=False,
        seed=77,
        sensor_seed=1077,
        physics_v2=pv2,
        gnc_v2=GncV2Config.m1_preset(),
    )
    plant = MuJoCoPlant3D(vehicle=cfg, scenario=scen, sim_config=scfg)
    plant.on_pad = False
    plant.upper_attached = False
    plant.main_prop_kg = 2_000.0
    plant.rcs_prop_kg = 150.0
    plant.gear_extension = 1.0
    plant.fin_extension = 1.0
    mp = plant.body.update_mass_properties(
        plant._total_mass(),
        1.0,
        upper_attached=False,
        dual_tank=True,
        rcs_propellant_kg=150.0,
        struts_active=True,
    )
    plant.body.update_ground_surface(upper_attached=False, pos_xy_m=(0.0, 0.0))
    p0 = np.array([0.0, 0.0, mp.cg_station_m - cfg.landing_gear.foot_station_m + 0.35])
    plant.body.set_pose_from_com(p0, (0.0, 0.0, -0.50), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    cmd_off = ActuatorCommand3D(deploy_fins=True, deploy_gear=True, phase="touchdown_settle")
    for _ in range(320):
        plant.step(cmd_off, scfg.control_dt_s)
        if plant.done:
            break
    assert plant.outcome == "landed"
    assert plant.standing_stable is True
    assert plant.first_contact_record is not None
    assert plant.first_contact_record["tilt_pad_deg"] > 2.0


# ---------------------------------------------------------------------------
# 4. 500->50 Hz Sensors v2, 30 s GPS Blackout Innovation 3-Fix & Slant Radar
# ---------------------------------------------------------------------------


def test_sensors_v2_aa_filter_zero_oracle_leak_and_base_radar() -> None:
    sc = SensorConfig3D(position_noise_m=0.1, velocity_noise_m_s=0.05, gyro_noise_deg_s=0.01)
    suite = SensorSuiteV2(
        sensor_seed=99,
        base_config=sc,
        settings=SensorV2Settings(gps_blackout_duration_s=20.0),
    )
    # 18 Hz Butterworth AA strongly attenuates a 60 Hz aliased flex tone at 500 Hz
    peaks = []
    for k in range(200):
        t = k * 0.002
        om_aa, _ = suite.step_imu_500hz(
            omega_true_body_rad_s=(0.0, 0.0, 0.0),
            accel_true_body_m_s2=(0.0, 0.0, 9.81),
            flex_omega_delta_rad_s=(sin(2.0 * pi * 60.0 * t), 0.0, 0.0),
        )
        if k >= 100:
            peaks.append(abs(float(om_aa[0])))
    assert max(peaks) < 0.12

    # In-flight plasma blackout suppresses GPS and NEVER leaks true quaternion
    q_true = axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), radians(25.0))
    r_wb = quat_to_rotmat(q_true)
    out_bo = suite.sample_50hz(
        time_s=420.0,
        pos_com_enu_m=np.array([1000.0, 0.0, 45_000.0]),
        vel_com_enu_m_s=np.array([-200.0, 0.0, -1100.0]),
        quat_wxyz=q_true,
        rot_wb=r_wb,
        cg_station_m=9.5,
        dynamic_pressure_pa=15_000.0,
        heat_flux_w_m2=25_000.0,
        thrust_n=0.0,
        upper_attached=False,
        on_pad=False,
        pad_normal_enu=np.array([0.0, 0.0, 1.0]),
        sloped_pad_enabled=False,
        true_fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
        true_main_prop_kg=4000.0,
        true_chamber_ratios_7=(0.0,) * 7,
    )
    assert out_bo["gps_valid"] is False
    assert out_bo["gps_pos_enu_m"] is None
    assert out_bo["attitude_valid"] is False
    assert out_bo["quat_wxyz"] == (1.0, 0.0, 0.0, 0.0)

    # Hull-base slant radar intersects flat ground accurately at h_base = 100 m, tilt = 10 deg
    q_10 = axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), radians(10.0))
    ok, rng, r_hit = compute_base_slant_radar_range_m(
        pos_base_enu_m=(0.0, 0.0, 100.0),
        rot_wb=quat_to_rotmat(q_10),
        pad_normal_enu=(0.0, 0.0, 1.0),
        sloped_pad_enabled=False,
    )
    assert ok
    assert isclose(rng, 100.0 / float(np.cos(radians(10.0))), rel_tol=1e-6)
    assert isclose(r_hit, 100.0 * float(np.tan(radians(10.0))), rel_tol=1e-6)


def test_strapdown_mekf_v2_high_speed_3fix_gps_reacquisition_and_radar_gate() -> None:
    """CDR C3: Innovation 3-Fix GPS reacquisition MUST pass at v = 1,200 m/s (no lockout!)."""
    cfg = DM100VehicleConfig()
    est = CausalEstimator3D(
        vehicle=cfg,
        forecast_atmo=AtmosphereLUT(),
        forecast_wind=Wind3DLUT(forecast=True),
        sensor_config=SensorConfig3D(position_noise_m=0.2, velocity_noise_m_s=0.05),
        gnc_v2=GncV2Config.m2_preset(),
    )
    # Initialize at high speed descending (v = -1,200 m/s)
    p = np.array([5000.0, 0.0, 50_000.0], dtype=np.float64)
    v = np.array([-200.0, 0.0, -1_200.0], dtype=np.float64)
    t = 0.0
    dt = 0.02
    g = 9.78

    def _pkt(
        ts: float, pos: np.ndarray, vel: np.ndarray, gps_ok: bool, spike_m: float = 0.0
    ) -> TelemetryPacket3D:
        gp = (float(pos[0] + spike_m), float(pos[1]), float(pos[2])) if gps_ok else None
        gv = (float(vel[0]), float(vel[1]), float(vel[2])) if gps_ok else None
        return TelemetryPacket3D(
            time_s=ts,
            measurement_time_s=ts,
            pos_enu_m=gp or (0.0, 0.0, 0.0),
            vel_enu_m_s=gv or (0.0, 0.0, 0.0),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            omega_body_rad_s=(0.0, 0.0, 0.0),
            accel_body_m_s2=(0.0, 0.0, 0.0),  # Free-fall ballistic descent
            main_propellant_kg=5000.0,
            rcs_propellant_kg=200.0,
            gear_extension=0.0,
            fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
            fin_extension=1.0,
            gimbal_rad=(0.0, 0.0),
            engine_throttle=0.0,
            upper_attached=False,
            on_pad=False,
            touchdown_detected=False,
            active_engines=0,
            restarts_used=1,
            gps_valid=gps_ok,
            gps_pos_enu_m=gp,
            gps_vel_enu_m_s=gv,
            attitude_valid=(ts == 0.0),
        )

    # 1. Init at t = 0
    est.observe(_pkt(t, p, v, True))
    # 2. Simulate 10 s GPS blackout (500 steps) while moving at 1,200 m/s
    for _ in range(500):
        t += dt
        p = p + v * dt + 0.5 * np.array([0.0, 0.0, -g]) * dt * dt
        v = v + np.array([0.0, 0.0, -g]) * dt
        out = est.observe(_pkt(t, p, v, False))
    assert out.gps_blackout is True
    assert est._gps_outage_s >= 9.9

    # 3. Inject 1 bad 500 m GPS outlier spike -> MUST reject and remain in reacquisition!
    t += dt
    p = p + v * dt + 0.5 * np.array([0.0, 0.0, -g]) * dt * dt
    v = v + np.array([0.0, 0.0, -g]) * dt
    est.observe(_pkt(t, p, v, True, spike_m=500.0))
    assert est._in_gps_blackout is True

    # 4. Feed 3 clean consecutive GPS fixes at v ~ 1,300 m/s -> MUST reacquire cleanly!
    for _ in range(4):
        t += dt
        p = p + v * dt + 0.5 * np.array([0.0, 0.0, -g]) * dt * dt
        v = v + np.array([0.0, 0.0, -g]) * dt
        out = est.observe(_pkt(t, p, v, True, spike_m=0.0))
    assert out.gps_blackout is False
    assert np.linalg.norm(est.p - p) < 50.0


def test_m2_presets_and_legacy_bit_exactness_preserved() -> None:
    p2 = PhysicsV2Config.m2_preset()
    g2 = GncV2Config.m2_preset()
    assert p2.flex_body and p2.dryden_turbulence and p2.sloped_pad and p2.sensors_v2
    assert not p2.faults_v2
    assert g2.use_structural_filter and g2.use_strapdown_mekf
    assert not g2.use_online_mpc and not g2.use_fdir_v2
    # Default config remains completely legacy
    assert not PhysicsV2Config().any_enabled()
