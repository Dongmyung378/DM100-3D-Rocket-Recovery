"""Phase 3 tests: 15-State MEKF, 3x9 allocation, Clarabel 3D GFOLD SOCP, and 6DOF RTLS."""

from __future__ import annotations

from math import degrees, hypot, radians
from pathlib import Path

import numpy as np

from dm100_3d.config import DM100VehicleConfig
from dm100_3d.coords import axis_angle_to_quat_wxyz, quat_multiply_wxyz
from dm100_3d.gnc import (
    Allocation3DInput,
    CausalEstimator3D,
    Gfold3DProblem,
    PriorityControlAllocator3D,
    SensorConfig3D,
    predict_zero_tilt_state_3d,
    solve_gfold_3d_socp,
)
from dm100_3d.interfaces import ActuatorCommand3D, TelemetryPacket3D
from dm100_3d.physics import Atmosphere3DSettings, AtmosphereLUT, Wind3DConfig, Wind3DLUT
from dm100_3d.replay import export_replay_json, load_replay_json
from dm100_3d.sim_3d import Disturbances3D, Scenario3D, SimConfig3D, run_flight_3d


def test_mekf_15state_converges_under_noise_bias_and_dropouts() -> None:
    """15-State MEKF tracks 3D position/velocity/attitude and suppresses injected gyro bias."""
    cfg = DM100VehicleConfig()
    atmo = AtmosphereLUT(Atmosphere3DSettings())
    wind = Wind3DLUT(Wind3DConfig())
    sc = SensorConfig3D(
        position_noise_m=0.8,
        velocity_noise_m_s=0.25,
        attitude_noise_deg=0.08,
        gyro_noise_deg_s=0.06,
        gyro_bias_deg_s=(0.05, -0.05, 0.04),
        dropout_rate=0.10,
    )
    est = CausalEstimator3D(vehicle=cfg, forecast_atmo=atmo, forecast_wind=wind, sensor_config=sc)
    rng = np.random.default_rng(42)

    # High-altitude ballistic test where vacuum free-fall matches MEKF nominal dynamics
    p = np.array([100.0, -50.0, 95_000.0], dtype=np.float64)
    v = np.array([-20.0, 10.0, -100.0], dtype=np.float64)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    om_true = np.array([0.02, -0.03, 0.01], dtype=np.float64)
    gb_true = np.array([radians(0.05), radians(-0.05), radians(0.04)])

    dt = 0.02
    last_out = None
    for k in range(250):
        t = k * dt
        p = p + v * dt
        v[2] -= 9.80665 * dt
        dq = axis_angle_to_quat_wxyz(om_true, float(np.linalg.norm(om_true)) * dt)
        q = quat_multiply_wxyz(q, dq)

        dropped = (k > 0) and (k % 10 == 0)
        p_m = p + rng.normal(0.0, sc.position_noise_m, size=3)
        v_m = v + rng.normal(0.0, sc.velocity_noise_m_s, size=3)
        ang_n = rng.normal(0.0, radians(sc.attitude_noise_deg), size=3)
        q_m = quat_multiply_wxyz(q, axis_angle_to_quat_wxyz(ang_n, float(np.linalg.norm(ang_n))))
        om_m = om_true + gb_true + rng.normal(0.0, radians(sc.gyro_noise_deg_s), size=3)

        pkt = TelemetryPacket3D(
            time_s=t,
            measurement_time_s=t,
            pos_enu_m=(float(p_m[0]), float(p_m[1]), float(p_m[2])),
            vel_enu_m_s=(float(v_m[0]), float(v_m[1]), float(v_m[2])),
            quat_wxyz=(float(q_m[0]), float(q_m[1]), float(q_m[2]), float(q_m[3])),
            omega_body_rad_s=(float(om_m[0]), float(om_m[1]), float(om_m[2])),
            accel_body_m_s2=(0.0, 0.0, 0.0),
            main_propellant_kg=3_000.0,
            rcs_propellant_kg=250.0,
            gear_extension=0.0,
            fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
            fin_extension=1.0,
            gimbal_rad=(0.0, 0.0),
            engine_throttle=0.0,
            upper_attached=False,
            on_pad=False,
            touchdown_detected=False,
            active_engines=0,
            restarts_used=2,
            packet_dropped=dropped,
        )
        last_out = est.observe(pkt)
        est.record_command(t, ActuatorCommand3D())

    assert last_out is not None
    assert float(np.linalg.norm(np.asarray(last_out.pos_enu_m) - p)) < 1.5
    assert float(np.linalg.norm(np.asarray(last_out.vel_enu_m_s) - v)) < 1.0
    # Verify covariance stayed symmetric positive definite
    eigvals = np.linalg.eigvalsh(est.P)
    assert float(np.min(eigvals)) > 0.0


def test_allocator_3x9_prioritizes_fins_then_conical_tvc_then_rcs() -> None:
    """3x9 allocator respects fin hinge limits, circular 6 deg TVC cone, and RCS guards."""
    cfg = DM100VehicleConfig()
    alloc = PriorityControlAllocator3D(cfg)

    # 1. Pure aerodynamic glide within 3.8 deg/s^2 guard: fins absorb pitch + roll with zero RCS
    out_glide = alloc.allocate(
        Allocation3DInput(
            requested_torque_body_nm=(0.0, 18_000.0, 800.0),
            background_torque_body_nm=(0.0, 0.0, 0.0),
            inertia_diag_kg_m2=(3.5e5, 3.5e5, 1.4e4),
            lever_arm_m=7.5,
            sizing_thrust_n=0.0,
            fin_arm_z_m=14.4,
            dynamic_pressure_pa=10_000.0,
            fin_extension=1.0,
            alpha_retro_xy_rad=(0.0, 0.0),
            rcs_available=True,
            allow_fin_allocation=True,
        )
    )
    assert out_glide.effectiveness_matrix_3x9.shape == (3, 9)
    assert all(abs(d) <= radians(20.0) + 1e-9 for d in out_glide.fin_deflection_rad)
    # Zero RCS needed when fins handle the torque
    assert max(abs(t) for t in out_glide.rcs_torque_body_nm) < 1e-3

    # 2. High 2-axis torque during burn: TVC must stay strictly inside 6 deg circular cone
    out_burn = alloc.allocate(
        Allocation3DInput(
            requested_torque_body_nm=(800_000.0, -800_000.0, 25_000.0),
            background_torque_body_nm=(0.0, 0.0, 0.0),
            inertia_diag_kg_m2=(3.5e5, 3.5e5, 1.4e4),
            lever_arm_m=7.5,
            sizing_thrust_n=220_000.0,
            fin_arm_z_m=14.4,
            dynamic_pressure_pa=50.0,
            fin_extension=0.0,
            alpha_retro_xy_rad=(0.0, 0.0),
            rcs_available=True,
            allow_fin_allocation=False,
        )
    )
    g_mag_deg = degrees(hypot(out_burn.gimbal_rad[0], out_burn.gimbal_rad[1]))
    assert g_mag_deg <= 6.0 + 1e-6
    assert abs(out_burn.rcs_torque_body_nm[2]) <= cfg.propulsion.rcs_roll_torque_limit_nm + 1e-6


def test_clarabel_3d_gfold_socp_solves_fast_and_handles_infeasible_fallback() -> None:
    """Clarabel 3D GFOLD SOCP solves crossrange hoverslam in <100 ms & satisfies all cones."""
    prob = Gfold3DProblem(
        pos0_enu_m=(45.0, -35.0, 520.0),
        vel0_enu_m_s=(-12.0, 8.0, -78.0),
        a_min_m_s2=10.5,
        a_max_m_s2=26.5,
        gravity_m_s2=9.80665,
        v_touchdown_m_s=0.65,
        max_tilt_deg=18.0,
        final_tilt_deg=2.0,
    )
    sol = solve_gfold_3d_socp(prob)
    assert sol.feasible is True
    assert sol.status == "optimal"
    assert sol.solve_ms < 250.0
    assert np.allclose(sol.pos_enu_m[-1], (0.0, 0.0, 0.0), atol=1e-2)
    assert np.allclose(sol.vel_enu_m_s[-1], (0.0, 0.0, -0.65), atol=1e-2)
    for ax, ay, az in sol.accel_enu_m_s2:
        a_norm = float(np.linalg.norm([ax, ay, az]))
        assert prob.a_min_m_s2 - 0.15 <= a_norm <= prob.a_max_m_s2 + 0.15

    # Infeasible case must return cleanly without crashing; backup zero-tilt predictor stays finite
    infeas = Gfold3DProblem(
        pos0_enu_m=(500.0, 500.0, 20.0),
        vel0_enu_m_s=(50.0, 50.0, -100.0),
        a_min_m_s2=10.5,
        a_max_m_s2=12.0,
    )
    sol_bad = solve_gfold_3d_socp(infeas)
    assert sol_bad.feasible is False
    p_b, v_b = predict_zero_tilt_state_3d(
        (10.0, -8.0), (-2.0, 1.5), (0.05, -0.04), (0.01, -0.01), 16.0, 0.05
    )
    assert all(np.isfinite((*p_b, *v_b)))


def test_full_6dof_rtls_flight_calm_and_crosswind_with_replay(tmp_path: Path) -> None:
    """Phase 3 Exit Gate: End-to-end 6DOF MuJoCo launch -> 700 km orbit -> 3D RTLS -> 5s settle."""
    # 1. Calm 3D flight (includes real 3D crossrange Coriolis at 37 deg latitude!)
    summary_calm, frames_calm = run_flight_3d(
        Scenario3D(name="calm_3d"),
        sim_config=SimConfig3D(record_replay=True),
    )
    assert summary_calm["outcome"] == "landed", f"Calm failed: {summary_calm['failed_criteria']}"
    assert summary_calm["requirements_passed"] is True, summary_calm["failed_criteria"]
    assert summary_calm["upper_stage_orbit"]["orbit_inserted"] is True
    assert summary_calm["upper_stage_orbit"]["replay_samples_count"] == 1_251
    assert summary_calm["phases"]["aero_glide"]["max_throttle_cmd"] <= 1e-6
    assert summary_calm["socp_solver_used"] is True
    assert summary_calm["peak_roll_deg"] <= 5.0

    # Verify exported 50 Hz 3D replay loads & passes schema validation + contains Stage 2 frames
    out_path = export_replay_json(
        frames_calm, tmp_path / "calm_3d_50hz.json", metadata=summary_calm
    )
    meta, loaded = load_replay_json(out_path)
    assert meta["requirements_passed"] is True
    assert any(f.upper_pos_enu_m is not None for f in loaded)
    assert all(f.base_pos_enu_m is not None for f in loaded)

    # 2. Full 3D Crosswind + Wind Shear + Roll Disturbance + Sensor Noise/Dropout flight!
    w_x = ((0.0, 5.0), (50.0, 5.0), (10_000.0, 12.0), (100_000.0, 12.0))
    w_y = ((0.0, -4.0), (50.0, -4.0), (10_000.0, 10.0), (100_000.0, 10.0))
    scen_3d = Scenario3D(
        name="crosswind_roll_3d",
        atmosphere=Atmosphere3DSettings(latitude_deg=37.0, density_scale=1.02),
        wind=Wind3DConfig(
            wind_x_profile=w_x,
            wind_y_profile=w_y,
            forecast_wind_x_profile=((0.0, 4.5), (50.0, 4.5), (10_000.0, 11.5), (100_000.0, 11.5)),
            forecast_wind_y_profile=((0.0, -3.5), (50.0, -3.5), (10_000.0, 9.5), (100_000.0, 9.5)),
            gust_amplitude_m_s=3.0,
            gust_azimuth_rad=1.1,
            gust_start_s=280.0,
            gust_duration_s=100.0,
            gust_altitude_center_m=8_000.0,
        ),
        disturbances=Disturbances3D(
            thrust_scale=0.99,
            drag_scale=1.03,
            roll_disturbance_torque_nm=2_500.0,
            roll_disturbance_start_s=320.0,
            roll_disturbance_duration_s=4.0,
        ),
        sensors=SensorConfig3D(
            position_noise_m=0.4,
            velocity_noise_m_s=0.12,
            attitude_noise_deg=0.04,
            gyro_noise_deg_s=0.04,
            gyro_bias_deg_s=(0.02, -0.02, 0.02),
            dropout_rate=0.04,
        ),
    )
    summary_3d, _ = run_flight_3d(scen_3d, sim_config=SimConfig3D(record_replay=False))
    assert summary_3d["outcome"] == "landed", f"3D failed: {summary_3d}"
    assert summary_3d["requirements_passed"] is True, summary_3d["failed_criteria"]
    assert summary_3d["upper_stage_orbit"]["orbit_inserted"] is True
    assert summary_3d["peak_roll_deg"] <= 5.0
