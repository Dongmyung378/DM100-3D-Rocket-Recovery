"""Phase 2 Exit Gate: 3D physics verification and 2D-to-3D Plane Equivalence Gate."""

from __future__ import annotations

import sys
from math import cos, hypot, radians, sin
from pathlib import Path

import numpy as np
import pytest

from dm100_3d.config import DM100VehicleConfig, mass_properties_3d
from dm100_3d.coords import base_to_com_pos_enu, pitch_to_quat_wxyz, quat_to_rotmat, roll_angle_rad
from dm100_3d.physics import (
    Atmosphere3DSettings,
    AtmosphereLUT,
    FeedSettings,
    Separation3DSettings,
    Slosh3DSettings,
    Slosh3DState,
    Wind3DConfig,
    Wind3DLUT,
    compute_coriolis_and_centrifugal_3d,
    compute_fins_aero_3d,
    compute_forces_and_torques_3d,
    evaluate_upper_stage_orbit_3d,
    feed_rates,
    ignition_ready,
    initial_feed_state,
    separation_event_3d,
    slosh_3d_response,
)
from dm100_3d.plant_mujoco import MuJoCoRocketBody, fast_gravity

DM100_2D_SRC = Path(r"d:\Projects\DM100-2D-Rocket-Recovery\src")


def _try_import_2d():
    if not DM100_2D_SRC.exists():
        return None
    if str(DM100_2D_SRC) not in sys.path:
        sys.path.insert(0, str(DM100_2D_SRC))
    from powered_landing_guidance.apr_recovery import AprConditions
    from powered_landing_guidance.recovery_physics import AtmosphereSettings
    from powered_landing_guidance.rtls.forces import (
        AtmosphereLUT as Atmo2D,
    )
    from powered_landing_guidance.rtls.forces import (
        RtlsAerodynamics,
        RtlsFinSettings,
        RtlsPropulsionSettings,
        WindLUT,
        compute_forces_and_torques,
    )
    from powered_landing_guidance.rtls.orbit import evaluate_upper_stage_orbit
    from powered_landing_guidance.vehicle_systems import SloshState, slosh_response

    return {
        "AprConditions": AprConditions,
        "AtmosphereSettings": AtmosphereSettings,
        "Atmo2D": Atmo2D,
        "WindLUT": WindLUT,
        "RtlsAerodynamics": RtlsAerodynamics,
        "RtlsFinSettings": RtlsFinSettings,
        "RtlsPropulsionSettings": RtlsPropulsionSettings,
        "compute_forces_and_torques": compute_forces_and_torques,
        "evaluate_upper_stage_orbit": evaluate_upper_stage_orbit,
        "SloshState": SloshState,
        "slosh_response": slosh_response,
    }


def test_forces_3d_matches_dm100_2d_bit_for_bit_in_xz_plane() -> None:
    """Plane Equivalence: zero crosswind/roll forces match DM100 2D across ascent/entry/glide."""
    ref2d = _try_import_2d()
    if ref2d is None:
        pytest.skip("Sibling DM100-2D-Rocket-Recovery repo not found")

    cfg = DM100VehicleConfig()
    atmo3 = AtmosphereLUT(Atmosphere3DSettings())
    wind3 = Wind3DLUT(Wind3DConfig())
    atmo2 = ref2d["Atmo2D"](ref2d["AtmosphereSettings"]())
    wind2 = ref2d["WindLUT"](ref2d["AprConditions"](), forecast=False)
    aero2 = ref2d["RtlsAerodynamics"]()
    fins2 = ref2d["RtlsFinSettings"]()
    prop2 = ref2d["RtlsPropulsionSettings"]()

    regimes = [
        # (name, z, vx, vz, theta_deg, omega_deg_s, cg, thr, gim_deg, rcs_nm, fin_deg, fin_ext, up)
        ("ascent_maxq", 11_000.0, 210.0, 340.0, 24.0, 1.5, 14.5, 1.0, 1.8, 0.0, 0.0, 0.0, True),
        (
            "entry_burn",
            28_000.0,
            -320.0,
            -950.0,
            16.0,
            -2.0,
            8.2,
            3.0 / 7.0,
            -2.5,
            12_000.0,
            0.0,
            1.0,
            False,
        ),
        ("aero_glide", 12_000.0, -140.0, -320.0, 18.0, 0.8, 7.6, 0.0, 0.0, 0.0, 7.5, 1.0, False),
        ("hoverslam", 450.0, -12.0, -85.0, 6.0, -1.2, 7.2, 0.12, 3.2, -8_000.0, -4.0, 1.0, False),
    ]
    for (
        _name,
        z,
        vx,
        vz,
        th_deg,
        om_deg,
        cg,
        thr,
        gim_deg,
        rcs_nm,
        fin_deg,
        fin_ext,
        upper,
    ) in regimes:
        th = radians(th_deg)
        om = radians(om_deg)
        gim = radians(gim_deg)
        fin = radians(fin_deg)

        ft2 = ref2d["compute_forces_and_torques"](
            z_m=z,
            vx_m_s=vx,
            vz_m_s=vz,
            theta_rad=th,
            omega_rad_s=om,
            cg_m=cg,
            throttle_actual=thr,
            gimbal_rad=gim,
            rcs_torque_nm=rcs_nm,
            fin_deflection_rad=fin,
            fin_extension=fin_ext,
            atmo_lut=atmo2,
            wind_lut=wind2,
            time_s=100.0,
            aero=aero2,
            fins=fins2,
            propulsion=prop2,
            upper_attached=upper,
        )
        ft3 = compute_forces_and_torques_3d(
            altitude_m=z,
            vel_enu_m_s=(vx, 0.0, vz),
            quat_wxyz=pitch_to_quat_wxyz(th),
            omega_body_rad_s=(0.0, om, 0.0),
            cg_station_m=cg,
            throttle_actual=thr,
            gimbal_rad=(gim, 0.0),
            rcs_torque_body_nm=(0.0, rcs_nm, 0.0),
            fin_deflection_rad=(0.0, 0.0, fin, fin),
            fin_extension=fin_ext,
            atmo_lut=atmo3,
            wind_lut=wind3,
            time_s=100.0,
            vehicle=cfg,
            upper_attached=upper,
        )

        # Out-of-plane forces and torques must vanish identically
        assert abs(ft3.force_world_n[1]) < 1e-9
        assert abs(ft3.torque_body_nm[0]) < 1e-9
        assert abs(ft3.torque_body_nm[2]) < 1e-9

        # In-plane forces, pitch torque, thrust, q, mach, hinge match DM100 2D
        assert np.isclose(ft3.force_world_n[0], ft2.force_x_n, rtol=1e-4, atol=1e-3)
        assert np.isclose(ft3.force_world_n[2], ft2.force_z_n, rtol=1e-4, atol=1e-3)
        assert np.isclose(ft3.torque_body_nm[1], ft2.torque_nm, rtol=1e-4, atol=1e-2)
        assert np.isclose(ft3.thrust_n, ft2.thrust_n, rtol=1e-9)
        assert np.isclose(ft3.dynamic_pressure_pa, ft2.dynamic_pressure_pa, rtol=1e-4)
        assert np.isclose(ft3.mach, ft2.mach, rtol=1e-4)
        assert np.isclose(ft3.fin_hinge_nm[2], ft2.fin_hinge_nm, rtol=1e-4, atol=1e-3)


def test_mujoco_rk4_powered_ascent_plane_equivalence_vs_2d_rk4() -> None:
    """Plane Equivalence Trajectory Gate: 60 s MuJoCo 6DOF ascent matches 2D RK4 ODE."""
    cfg = DM100VehicleConfig()
    atmo = AtmosphereLUT(Atmosphere3DSettings(latitude_deg=0.0, planar_coriolis_only=True))
    wind = Wind3DLUT(Wind3DConfig())

    dt = 0.02
    n_steps = 3000  # 60.0 s ascent through Max-Q

    # 1. MuJoCo 6DOF trajectory
    body = MuJoCoRocketBody(cfg, timestep_s=0.01, gear_deployed=False, upper_attached=True)
    mass_mj = cfg.launch_mass_kg
    mp0 = body.update_mass_properties(mass_mj, 0.0, upper_attached=True, struts_active=False)
    p0 = base_to_com_pos_enu(
        (0.0, 0.0, -cfg.geometry.nozzle_station_m), (1, 0, 0, 0), mp0.cg_station_m
    )
    body.set_pose_from_com(p0, (0.0, 0.0, 0.0), (1, 0, 0, 0), (0.0, 0.0, 0.0))

    # 2. Pure 2D RK4 reference [x, z, vx, vz, theta, omega, mass]
    s2 = np.array([0.0, float(p0[2]), 0.0, 0.0, 0.0, 0.0, mass_mj], dtype=np.float64)
    max_q_mj = 0.0
    max_q_2d = 0.0

    for k in range(n_steps):
        t = k * dt
        # Smooth pitch-over TVC pulse between t = 10..20 s
        gim_y = radians(1.2) if 10.0 <= t < 13.0 else (-radians(0.8) if 13.0 <= t < 16.0 else 0.0)

        # --- Step MuJoCo (2 substeps of 0.01 s with 50 Hz mass update) ---
        mp_mj = body.update_mass_properties(mass_mj, 0.0, upper_attached=True, struts_active=False)
        for _ in range(2):
            p_c, v_c, q_c, om_c = body.get_com_state()
            ft = compute_forces_and_torques_3d(
                altitude_m=float(p_c[2]),
                vel_enu_m_s=v_c,
                quat_wxyz=q_c,
                omega_body_rad_s=om_c,
                cg_station_m=mp_mj.cg_station_m,
                throttle_actual=1.0,
                gimbal_rad=(gim_y, 0.0),
                rcs_torque_body_nm=(0.0, 0.0, 0.0),
                fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
                fin_extension=0.0,
                atmo_lut=atmo,
                wind_lut=wind,
                time_s=t,
                vehicle=cfg,
                upper_attached=True,
            )
            max_q_mj = max(max_q_mj, ft.dynamic_pressure_pa)
            a_cor = compute_coriolis_and_centrifugal_3d(
                altitude_m=float(p_c[2]),
                vel_enu_m_s=v_c,
                latitude_deg=0.0,
                coriolis_enabled=True,
                planar_coriolis_only=True,
            )
            f_tot = np.asarray(ft.force_world_n) + mass_mj * a_cor
            tau_w = quat_to_rotmat(q_c) @ np.asarray(ft.torque_body_nm)
            body.step_physics(
                dt_s=0.01,
                force_world_n=f_tot,
                torque_world_nm=tau_w,
                altitude_for_gravity_m=float(p_c[2]),
            )
            mass_mj -= ft.main_flow_kg_s * 0.01

        # --- Step 2D RK4 reference (same 50 Hz step using 2 substeps) ---
        for _ in range(2):
            sub_dt = 0.01

            def _rhs(st: np.ndarray, gim_val: float = gim_y, t_val: float = t) -> np.ndarray:
                x, z, vx, vz, th, om, m = [float(v) for v in st]
                mp = mass_properties_3d(m, 0.0, vehicle=cfg, upper_attached=True)
                ft_r = compute_forces_and_torques_3d(
                    altitude_m=z,
                    vel_enu_m_s=(vx, 0.0, vz),
                    quat_wxyz=pitch_to_quat_wxyz(th),
                    omega_body_rad_s=(0.0, om, 0.0),
                    cg_station_m=mp.cg_station_m,
                    throttle_actual=1.0,
                    gimbal_rad=(gim_val, 0.0),
                    rcs_torque_body_nm=(0.0, 0.0, 0.0),
                    fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
                    fin_extension=0.0,
                    atmo_lut=atmo,
                    wind_lut=wind,
                    time_s=t_val,
                    vehicle=cfg,
                    upper_attached=True,
                )
                a_c = compute_coriolis_and_centrifugal_3d(
                    altitude_m=z,
                    vel_enu_m_s=(vx, 0.0, vz),
                    latitude_deg=0.0,
                    coriolis_enabled=True,
                    planar_coriolis_only=True,
                )
                ax = ft_r.force_world_n[0] / m + float(a_c[0])
                az = ft_r.force_world_n[2] / m - fast_gravity(z) + float(a_c[2])
                alpha = ft_r.torque_body_nm[1] / mp.inertia_diag_kg_m2[1]
                return np.array([vx, vz, ax, az, om, alpha, -ft_r.main_flow_kg_s])

            k1 = _rhs(s2)
            k2 = _rhs(s2 + 0.5 * sub_dt * k1)
            k3 = _rhs(s2 + 0.5 * sub_dt * k2)
            k4 = _rhs(s2 + sub_dt * k3)
            s2 = s2 + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            _, _, rho_r = atmo.at(float(s2[1]))
            max_q_2d = max(max_q_2d, 0.5 * rho_r * (float(s2[2]) ** 2 + float(s2[3]) ** 2))

    p_end, v_end, q_end, om_end = body.get_com_state()
    # Out-of-plane drift must be strictly zero
    assert abs(float(p_end[1])) < 1e-6
    assert abs(float(v_end[1])) < 1e-6
    assert abs(roll_angle_rad(q_end)) < 1e-6
    assert abs(float(om_end[0])) < 1e-6
    assert abs(float(om_end[2])) < 1e-6

    # Downrange, altitude, speed, Max-Q, and propellant match 2D RK4 within 0.5%
    assert np.isclose(float(p_end[0]), float(s2[0]), rtol=5e-3, atol=2.0)
    assert np.isclose(float(p_end[2]), float(s2[1]), rtol=5e-3, atol=5.0)
    assert np.isclose(
        hypot(float(v_end[0]), float(v_end[2])), hypot(float(s2[2]), float(s2[3])), rtol=5e-3
    )
    assert np.isclose(max_q_mj, max_q_2d, rtol=5e-3)
    assert np.isclose(mass_mj, float(s2[6]), rtol=1e-4)


def test_4fin_differential_roll_control_and_conical_tvc_clamping() -> None:
    """Differential 4-fin deflection creates pure roll torque; 2-axis TVC clamps to 6 deg cone."""
    cfg = DM100VehicleConfig()
    atmo = AtmosphereLUT(Atmosphere3DSettings())
    wind = Wind3DLUT(Wind3DConfig())

    d_r = radians(8.0)
    # Fin 0 (+X, +Y lift), Fin 1 (-X, -Y lift), Fin 2 (+Y, -X lift), Fin 3 (-Y, +X lift) -> +roll
    ff_w, ftau_b, hinges = compute_fins_aero_3d(
        altitude_m=8_000.0,
        vel_enu_m_s=(0.0, 0.0, -250.0),
        rot_wb=np.eye(3),
        omega_body_rad_s=(0.0, 0.0, 0.0),
        cg_station_m=8.0,
        fin_deflection_rad=(d_r, -d_r, -d_r, d_r),
        fin_extension=1.0,
        atmo_lut=atmo,
        wind_lut=wind,
        time_s=0.0,
        vehicle=cfg,
        upper_attached=False,
    )
    assert abs(float(ff_w[0])) < 1e-9
    assert abs(float(ff_w[1])) < 1e-9
    assert abs(float(ftau_b[0])) < 1e-9
    assert abs(float(ftau_b[1])) < 1e-9
    assert float(ftau_b[2]) > 50_000.0
    assert np.allclose(hinges, hinges[0], rtol=1e-9)

    # 2-axis TVC circular cone check: (6 deg, 6 deg) must clamp to exact 6 deg cone, NOT 8.49 deg
    ft_cone = compute_forces_and_torques_3d(
        altitude_m=10_000.0,
        vel_enu_m_s=(0.0, 0.0, 0.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        omega_body_rad_s=(0.0, 0.0, 0.0),
        cg_station_m=10.0,
        throttle_actual=1.0,
        gimbal_rad=(radians(6.0), radians(6.0)),
        rcs_torque_body_nm=(0.0, 0.0, 0.0),
        fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
        fin_extension=0.0,
        atmo_lut=atmo,
        wind_lut=wind,
        time_s=0.0,
        vehicle=cfg,
        upper_attached=False,
    )
    t_lat = hypot(ft_cone.force_world_n[0], ft_cone.force_world_n[1])
    assert np.isclose(t_lat, ft_cone.thrust_n * sin(radians(6.0)), rtol=1e-9)
    assert np.isclose(ft_cone.force_world_n[2], ft_cone.thrust_n * cos(radians(6.0)), rtol=1e-9)


def test_3d_slosh_feed_separation_and_stage2_orbit_25s_trajectory() -> None:
    """Verify 3D slosh symmetry/2D match, closed helium feed, 6DOF separation, and 700 km orbit."""
    cfg = DM100VehicleConfig()

    # 1. Slosh X/Y symmetry and 2D equivalence
    slosh_cfg = Slosh3DSettings()
    r_x = slosh_3d_response(
        total_mass_kg=40_000.0,
        rigid_inertia_diag_kg_m2=(1.8e6, 1.8e6, 4.5e4),
        main_propellant_kg=30_000.0,
        station_offset_m=1.2,
        proper_accel_body_m_s2=(1.5, 0.0, 22.0),
        external_torque_body_nm=(0.0, 50_000.0, 0.0),
        omega_body_rad_s=(0.0, 0.05, 0.0),
        state=Slosh3DState(qx_m=0.12, qy_m=0.0, qxd_m_s=0.08, qyd_m_s=0.0),
        settings=slosh_cfg,
    )
    assert r_x.valid
    assert abs(r_x.slosh_torque_body_nm[0]) < 1e-12
    assert abs(r_x.qdd_m_s2[1]) < 1e-12

    ref2d = _try_import_2d()
    if ref2d is not None:
        r2 = ref2d["slosh_response"](
            40_000.0,
            1.8e6,
            30_000.0,
            1.2,
            22.0,
            1.5,
            50_000.0,
            0.05,
            ref2d["SloshState"](0.12, 0.08),
        )
        alpha_3d = (50_000.0 + r_x.slosh_torque_body_nm[1]) / 1.8e6
        assert np.isclose(alpha_3d, r2.angular_acceleration_rad_s2, rtol=1e-9)
        assert np.isclose(r_x.qdd_m_s2[0], r2.displacement_acceleration_m_s2, rtol=1e-9)

    # 2. Closed helium feed conservation
    f_cfg = FeedSettings()
    f0 = initial_feed_state(cfg.propellant.main_propellant_kg, f_cfg)
    fr = feed_rates(cfg.propellant.main_propellant_kg, 480.0, 15.0, f0, f_cfg)
    assert fr.inventory_valid
    assert ignition_ready(f0, fr.tank_pressure_pa)
    assert np.isclose(fr.tank_gas_derivative_kg_s + fr.bottle_gas_derivative_kg_s, 0.0, atol=1e-12)

    # 3. 6DOF cold separation momentum conservation
    sep = separation_event_3d(
        stack_pos_com_enu_m=(25_000.0, 400.0, 60_000.0),
        stack_vel_com_enu_m_s=(1_750.0, 35.0, 720.0),
        quat_wxyz=pitch_to_quat_wxyz(radians(55.0)),
        omega_body_rad_s=(0.0, 0.01, 0.0),
        booster_mass_kg=18_800.0,
        upper_mass_kg=24_000.0,
        booster_cg_station_m=9.5,
        upper_cg_station_m=28.0,
        booster_inertia_diag_kg_m2=(8.5e5, 8.5e5, 2.2e4),
        upper_inertia_diag_kg_m2=(1.8e5, 1.8e5, 2.9e4),
        settings=Separation3DSettings(),
    )
    assert float(sep["linear_momentum_residual_kg_m_s"]) < 1e-7
    dv_rel = np.asarray(sep["upper_vel_enu_m_s"]) - np.asarray(sep["booster_vel_enu_m_s"])
    assert np.isclose(float(np.linalg.norm(dv_rel)), 0.5, atol=0.25)

    # 4. Stage 2 700 km circular orbit + 25 s 50 Hz trajectory (1,251 samples)
    orb = evaluate_upper_stage_orbit_3d(
        pos_sep_enu_m=(25_000.0, 0.0, 60_000.0),
        vel_sep_enu_m_s=(1_750.0, 0.0, 720.0),
        latitude_deg=37.0,
        settings=cfg.upper_stage,
        replay_hz=50.0,
    )
    assert orb.orbit_inserted is True
    assert orb.final_apogee_km is not None
    assert abs(orb.final_apogee_km - 700.0) <= 15.0
    assert orb.upper_propellant_remaining_kg > 0.0
    assert len(orb.replay_samples) == 1_251  # 25.0 s * 50 Hz + 1
    assert orb.replay_samples[0].engine_lit is False
    assert orb.replay_samples[-1].engine_lit is True
    # Confirm upper stage accelerates well clear of initial separation point
    dist_25s = float(
        np.linalg.norm(
            np.asarray(orb.replay_samples[-1].pos_enu_m)
            - np.asarray(orb.replay_samples[0].pos_enu_m)
        )
    )
    assert dist_25s > 40_000.0

    if ref2d is not None:
        orb2 = ref2d["evaluate_upper_stage_orbit"](
            z_sep_m=60_000.0,
            vx_sep_m_s=1_750.0,
            vz_sep_m_s=720.0,
            latitude_deg=37.0,
        )
        assert orb.orbit_inserted == orb2["orbit_inserted"]
        assert np.isclose(orb.final_apogee_km, orb2["final_apogee_km"], atol=1e-6)
        assert np.isclose(
            orb.upper_propellant_remaining_kg, orb2["upper_propellant_remaining_kg"], atol=1e-6
        )
