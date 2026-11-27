"""Tier-1 unit & integration tests for DM100 3D v2 M3 (Phase 8B: MPC + 7-Engine/Fin-Jam FDIR)."""

from __future__ import annotations

from math import atan2, degrees, isclose, pi, radians, sqrt

import numpy as np

from dm100_3d.config import DM100VehicleConfig, GncV2Config, PhysicsV2Config
from dm100_3d.gnc.allocation_3d import Allocation3DInput, PriorityControlAllocator3D
from dm100_3d.gnc.fdir_3d import CausalFdir3D
from dm100_3d.gnc.mpc_gfold_3d import MpcGfold3DConfig, OnlineGfoldMpc3D
from dm100_3d.interfaces import ActuatorCommand3D, TelemetryPacket3D
from dm100_3d.physics.faults_3d import (
    ENGINE_OUTER_RIM_RADIUS_M,
    ENGINE_RING_RADIUS_M,
    SYMMETRIC_3ENG_MASKS,
    EngineClusterFaultModel3D,
    FaultSettings3D,
)
from dm100_3d.sim_3d import Scenario3D, SimConfig3D, run_flight_3d
from dm100_3d.validation_suites import build_edge_v2_suite, get_validation_suite

# ---------------------------------------------------------------------------
# 1. 7-Engine Cluster Physics, Asymmetric EO Moments, Side-Loads & Fin-Jam
# ---------------------------------------------------------------------------


def test_engine_cluster_geometry_symmetry_eo_moments_and_sideload() -> None:
    cfg = DM100VehicleConfig()
    # Exact outer rim check: 0.92 m ring + sqrt(1.4113 / (7*pi)) = 1.173 m (CDR §2.6)
    r_noz_single = sqrt(cfg.propulsion.nozzle_exit_area_m2 / (7.0 * pi))
    assert isclose(ENGINE_RING_RADIUS_M + r_noz_single, ENGINE_OUTER_RIM_RADIUS_M, abs_tol=1e-3)

    # All 7 engines unbiased -> exact zero asymmetric moment
    model = EngineClusterFaultModel3D(
        seed=42, vehicle=cfg, settings=FaultSettings3D(side_load_peak_n=2_000.0)
    )
    out = None
    for k in range(60):
        out = model.step_cluster(
            time_s=k * 0.01,
            dt_s=0.01,
            phase="ascent",
            req_total_throttle=1.0,
            active_engines=7,
            engine_mask_7=(True,) * 7,
            engine_throttles_cmd_7=None,
            engine_tau_s=0.03,
            ambient_pressure_pa=101_325.0,
            cg_station_m=15.0,
            gimbal_rad=(0.0, 0.0),
            thrust_scale=1.0,
            delivered_ratio=1.0,
            main_prop_kg=50_000.0,
            touchdown_detected=False,
        )
    assert out is not None
    assert np.linalg.norm(out.asymmetric_torque_body_nm) < 1e-6
    assert out.total_thrust_n > 1_200_000.0
    # After 0.60 s (> 0.15 s pulse duration), ignition side-load pulse has fully decayed
    assert np.linalg.norm(out.side_load_force_body_n) < 1e-9

    # Atmospheric scaling of 0.15 s ignition side-load pulse: non-zero at SL, zero in vacuum
    sl_model = EngineClusterFaultModel3D(
        seed=42, vehicle=cfg, settings=FaultSettings3D(side_load_peak_n=2_500.0)
    )
    vac_model = EngineClusterFaultModel3D(
        seed=42, vehicle=cfg, settings=FaultSettings3D(side_load_peak_n=2_500.0)
    )
    peak_sl = 0.0
    peak_vac = 0.0
    for k in range(25):
        o_sl = sl_model.step_cluster(
            time_s=k * 0.01,
            dt_s=0.01,
            phase="landing_burn",
            req_total_throttle=1.0 / 7.0,
            active_engines=1,
            engine_mask_7=(True, False, False, False, False, False, False),
            engine_throttles_cmd_7=None,
            engine_tau_s=0.03,
            ambient_pressure_pa=101_325.0,
            cg_station_m=8.0,
            gimbal_rad=(0.0, 0.0),
            thrust_scale=1.0,
            delivered_ratio=1.0,
            main_prop_kg=3_000.0,
            touchdown_detected=False,
        )
        o_vac = vac_model.step_cluster(
            time_s=k * 0.01,
            dt_s=0.01,
            phase="boostback",
            req_total_throttle=1.0 / 7.0,
            active_engines=1,
            engine_mask_7=(True, False, False, False, False, False, False),
            engine_throttles_cmd_7=None,
            engine_tau_s=0.03,
            ambient_pressure_pa=0.0,
            cg_station_m=8.0,
            gimbal_rad=(0.0, 0.0),
            thrust_scale=1.0,
            delivered_ratio=1.0,
            main_prop_kg=3_000.0,
            touchdown_detected=False,
        )
        peak_sl = max(peak_sl, float(np.linalg.norm(o_sl.side_load_force_body_n)))
        peak_vac = max(peak_vac, float(np.linalg.norm(o_vac.side_load_force_body_n)))
    assert 2_300.0 <= peak_sl <= 2_550.0
    assert peak_vac == 0.0

    # Symmetric 3-engine triplets produce exact zero asymmetric torque
    for sym in SYMMETRIC_3ENG_MASKS:
        m3 = EngineClusterFaultModel3D(
            seed=7, vehicle=cfg, settings=FaultSettings3D(side_load_peak_n=0.0)
        )
        for k in range(30):
            o3 = m3.step_cluster(
                time_s=k * 0.01,
                dt_s=0.01,
                phase="entry_burn",
                req_total_throttle=3.0 / 7.0,
                active_engines=3,
                engine_mask_7=sym,
                engine_throttles_cmd_7=None,
                engine_tau_s=0.03,
                ambient_pressure_pa=1_000.0,
                cg_station_m=9.5,
                gimbal_rad=(0.0, 0.0),
                thrust_scale=1.0,
                delivered_ratio=1.0,
                main_prop_kg=7_000.0,
                touchdown_detected=False,
            )
        assert np.linalg.norm(o3.asymmetric_torque_body_nm) < 1e-6

    # 7->6 EO (engine 1 @ +X failed): trim angle ~ 0.60 deg; 3->2 EO: trim angle ~ 2.84 deg
    m_eo = EngineClusterFaultModel3D(
        seed=11,
        vehicle=cfg,
        settings=FaultSettings3D(
            side_load_peak_n=0.0,
            engine_out_index=1,
            engine_out_time_s=0.0,
            engine_out_spindown_tau_s=0.05,
        ),
    )
    for k in range(80):
        o_eo = m_eo.step_cluster(
            time_s=k * 0.01,
            dt_s=0.01,
            phase="ascent",
            req_total_throttle=1.0,
            active_engines=7,
            engine_mask_7=(True,) * 7,
            engine_throttles_cmd_7=None,
            engine_tau_s=0.02,
            ambient_pressure_pa=50_000.0,
            cg_station_m=14.6,
            gimbal_rad=(0.0, 0.0),
            thrust_scale=1.0,
            delivered_ratio=1.0,
            main_prop_kg=40_000.0,
            touchdown_detected=False,
        )
    assert o_eo.engine_failed_7[1]
    assert o_eo.chamber_pressure_ratios_7[1] < 1e-3
    arm_asc = 14.6 - cfg.geometry.nozzle_station_m
    trim_7to6_deg = degrees(
        atan2(abs(o_eo.asymmetric_torque_body_nm[1]), o_eo.total_thrust_n * arm_asc)
    )
    assert 0.50 <= trim_7to6_deg <= 0.70

    # 3->2 EO on [0, 1, 4] with engine 1 dead -> [0, 4] remaining
    m_32 = EngineClusterFaultModel3D(
        seed=12,
        vehicle=cfg,
        settings=FaultSettings3D(
            side_load_peak_n=0.0,
            engine_out_index=1,
            engine_out_time_s=0.0,
            engine_out_spindown_tau_s=0.05,
        ),
    )
    for k in range(80):
        o_32 = m_32.step_cluster(
            time_s=k * 0.01,
            dt_s=0.01,
            phase="landing_burn",
            req_total_throttle=3.0 / 7.0,
            active_engines=3,
            engine_mask_7=SYMMETRIC_3ENG_MASKS[0],
            engine_throttles_cmd_7=None,
            engine_tau_s=0.02,
            ambient_pressure_pa=101_325.0,
            cg_station_m=9.0,
            gimbal_rad=(0.0, 0.0),
            thrust_scale=1.0,
            delivered_ratio=1.0,
            main_prop_kg=2_500.0,
            touchdown_detected=False,
        )
    arm_land = 9.0 - cfg.geometry.nozzle_station_m
    trim_3to2_deg = degrees(
        atan2(abs(o_32.asymmetric_torque_body_nm[1]), o_32.total_thrust_n * arm_land)
    )
    assert 2.50 <= trim_3to2_deg <= 3.30

    # Fin-Jam rate-limited runaway & hard lock
    m_fj = EngineClusterFaultModel3D(
        seed=13,
        vehicle=cfg,
        settings=FaultSettings3D(
            fin_jam_index=2,
            fin_jam_phase="aero_glide",
            fin_jam_phase_elapsed_s=0.10,
            fin_jam_angle_deg=10.0,
        ),
    )
    fins = [0.0, 0.0, 0.0, 0.0]
    for k in range(40):
        fins = m_fj.apply_fin_jam_dynamics(
            time_s=k * 0.01,
            dt_s=0.01,
            phase="aero_glide",
            fin_deflection=fins,
            cmd_fin_deflection_rad=(0.0, 0.0, radians(-8.0), 0.0),
            fin_rate_rad_s=radians(60.0),
            fin_tau_s=0.08,
            fin_limit_rad=radians(20.0),
        )
    assert m_fj.fin_jammed[2]
    assert isclose(degrees(fins[2]), 10.0, abs_tol=0.1)


# ---------------------------------------------------------------------------
# 2. Causal FDIR: Zero False Positives & Deterministic EO / Fin-Jam Isolation
# ---------------------------------------------------------------------------


def _make_pkt(
    t: float,
    *,
    upper: bool = True,
    prop_kg: float = 50_000.0,
    fin_ext: float = 0.0,
    fin_def: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    p7: tuple[float, ...] = (1.0,) * 7,
    active_engines: int = 7,
) -> TelemetryPacket3D:
    return TelemetryPacket3D(
        time_s=t,
        measurement_time_s=t,
        pos_enu_m=(0.0, 0.0, 5_000.0),
        vel_enu_m_s=(0.0, 0.0, 200.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        omega_body_rad_s=(0.0, 0.0, 0.0),
        accel_body_m_s2=(0.0, 0.0, 15.0),
        main_propellant_kg=prop_kg,
        rcs_propellant_kg=250.0,
        gear_extension=0.0,
        fin_deflection_rad=fin_def,
        fin_extension=fin_ext,
        gimbal_rad=(0.0, 0.0),
        engine_throttle=float(sum(p7) / 7.0),
        upper_attached=upper,
        on_pad=False,
        touchdown_detected=False,
        active_engines=active_engines,
        restarts_used=0,
        engine_chamber_pressure_ratio_7=p7,  # type: ignore[arg-type]
    )


def test_causal_fdir_zero_false_positives_and_isolation() -> None:
    cfg = DM100VehicleConfig()
    fdir = CausalFdir3D(vehicle=cfg, observation_delay_steps=1, action_delay_steps=1)

    # Phase A: Nominal flight through 0.36 min throttle, cutoff, 0.20 s restart, +/-1.5% dispersion
    mask7 = (True,) * 7
    mask1 = (True, False, False, False, False, False, False)
    cmd_hi = ActuatorCommand3D(
        phase="ascent", throttle_cmd=1.0, active_engines=7, engine_mask_7=mask7
    )
    cmd_lo = ActuatorCommand3D(
        phase="landing_burn", throttle_cmd=0.36 / 7.0, active_engines=1, engine_mask_7=mask1
    )
    cmd_off = ActuatorCommand3D(phase="coast", throttle_cmd=0.0, active_engines=0)

    # Simulate 2.0 s full throttle, 0.5 s cutoff, 1.5 s restart to 0.36 min throttle
    p_sim = [0.0] * 7
    restart_t = 0.0
    was_on = False
    for k in range(200):
        t = k * 0.02
        if k < 100:
            cmd = cmd_hi
            tgt = [0.985 + 0.005 * i for i in range(7)]  # +/-1.5% per-engine bias
            upper = True
            n_act = 7
        elif k < 125:
            cmd = cmd_off
            tgt = [0.0] * 7
            upper = False
            was_on = False
            n_act = 0
        else:
            cmd = cmd_lo
            upper = False
            n_act = 1
            if not was_on:
                was_on = True
                restart_t = 0.20
            if restart_t > 0.0:
                restart_t = max(0.0, restart_t - 0.02)
                tgt = [0.0] * 7
            else:
                tgt = [0.36 * 0.985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        fdir.record_command(cmd)
        # Lagged chamber pressure with 0.055 s engine_lag + 0.030 s sensor lag
        for i in range(7):
            p_sim[i] += (tgt[i] - p_sim[i]) * (1.0 - np.exp(-0.02 / 0.075))
        st = fdir.update(
            pkt=_make_pkt(t, upper=upper, p7=tuple(p_sim), active_engines=n_act),
            q_pa=5_000.0,
            vel_enu_m_s=(200.0, 0.0, 300.0),
            pos_enu_m=(1_000.0, 0.0, 12_000.0),
            accel_dist_world_m_s2=(0.1, 0.0, 0.1),
            torque_bias_body_nm=(1_000.0, 1_000.0, 0.0),
            mass_kg=60_000.0,
        )
        assert not any(st.engine_failed_7), f"False positive EO at t={t:.2f}s"
        assert not any(st.fin_jammed_4)

    # Phase B: Genuine Engine-Out on engine 3 during steady 7-engine climb -> isolates in <= 0.16 s
    fdir_eo = CausalFdir3D(vehicle=cfg)
    for k in range(60):
        t = k * 0.02
        fdir_eo.record_command(cmd_hi)
        p7 = [1.0] * 7
        if t >= 0.80:
            p7[3] = 0.15
        st = fdir_eo.update(
            pkt=_make_pkt(t, upper=True, prop_kg=45_000.0, p7=tuple(p7), active_engines=7),
            q_pa=15_000.0,
            vel_enu_m_s=(50.0, 0.0, 180.0),
            pos_enu_m=(500.0, 0.0, 5_000.0),
            accel_dist_world_m_s2=(0.0, 0.0, -2.0) if t >= 0.80 else (0.0, 0.0, 0.0),
            torque_bias_body_nm=(30_000.0, 0.0, 0.0) if t >= 0.80 else (0.0, 0.0, 0.0),
            mass_kg=75_000.0,
        )
    assert st.engine_failed_7[3]
    assert sum(st.engine_failed_7) == 1
    assert st.engine_out_detected_time_s is not None
    assert 0.80 <= st.engine_out_detected_time_s <= 0.96
    # Healthy 3-engine triplet selector must skip any triplet containing engine 3
    sym_mask, n_eng = fdir_eo.select_3eng_mask()
    assert n_eng == 3 and not sym_mask[3]

    # Phase B2: Healthy 55 deg/s fin slew (+20 -> 0 deg) with 1-step resolver lag -> ZERO false FJ!
    fdir_slew = CausalFdir3D(vehicle=cfg)
    f_true = [radians(20.0)] * 4
    f_prev_meas = list(f_true)
    cmd_zero = ActuatorCommand3D(
        phase="entry_burn", throttle_cmd=0.0, active_engines=0, fin_deflection_rad=(0.0,) * 4
    )
    for k in range(40):
        t = k * 0.02
        fdir_slew.record_command(cmd_zero)
        st_s = fdir_slew.update(
            pkt=_make_pkt(
                t,
                upper=False,
                fin_ext=1.0,
                fin_def=tuple(f_prev_meas),  # type: ignore[arg-type]
                p7=(0.0,) * 7,
                active_engines=0,
            ),
            q_pa=800.0,
            vel_enu_m_s=(0.0, 0.0, -300.0),
            pos_enu_m=(0.0, 0.0, 30_000.0),
            accel_dist_world_m_s2=(0.0, 0.0, 0.0),
            torque_bias_body_nm=(0.0, 0.0, 0.0),
            mass_kg=10_000.0,
        )
        assert not any(st_s.fin_jammed_4), f"False Fin-Jam during healthy 55 deg/s slew at k={k}"
        f_prev_meas = list(f_true)
        # Plant 2x sub-stepped healthy fin motion toward 0 deg
        for _ in range(2):
            for j in range(4):
                step_j = max(
                    -radians(60.0) * 0.01,
                    min(radians(60.0) * 0.01, (0.0 - f_true[j]) * (1.0 - np.exp(-0.01 / 0.08))),
                )
                f_true[j] += step_j

    # Phase C: Fin-Jam detection in 4 steps (80 ms) during glide
    fdir_fj = CausalFdir3D(vehicle=cfg)
    cmd_gl = ActuatorCommand3D(
        phase="aero_glide",
        throttle_cmd=0.0,
        active_engines=0,
        fin_deflection_rad=(0.0, 0.0, radians(-6.0), 0.0),
        deploy_fins=True,
    )
    for k in range(30):
        t = k * 0.02
        fdir_fj.record_command(cmd_gl)
        # Fin 2 stuck at +8 deg while commanded to -6 deg
        f_meas = (0.0, 0.0, radians(8.0), 0.0)
        st_fj = fdir_fj.update(
            pkt=_make_pkt(
                t, upper=False, fin_ext=1.0, fin_def=f_meas, p7=(0.0,) * 7, active_engines=0
            ),
            q_pa=4_500.0,
            vel_enu_m_s=(-40.0, 0.0, -180.0),
            pos_enu_m=(1_500.0, 0.0, 9_000.0),
            accel_dist_world_m_s2=(0.0, 0.0, 0.0),
            torque_bias_body_nm=(0.0, 0.0, 0.0),
            mass_kg=16_500.0,
        )
    assert st_fj.fin_jammed_4 == (False, False, True, False)
    assert isclose(degrees(st_fj.fin_jam_angles_rad_4[2]), 8.0, abs_tol=0.2)


# ---------------------------------------------------------------------------
# 3. 3x4 Individual Fin Jacobian + 3x3 10x Roll-Weighted QP Allocator
# ---------------------------------------------------------------------------


def test_allocator_3x4_roll_priority_qp_cancels_jammed_fin_roll() -> None:
    cfg = DM100VehicleConfig()
    alloc = PriorityControlAllocator3D(cfg)
    # Realistic 3x4 fin torque Jacobian [N*m/rad]:
    # fins 0,1 (+X/-X) pitch/roll; fins 2,3 (+Y/-Y) yaw/roll
    b_3x4 = np.array(
        [
            [0.0, 0.0, 120_000.0, -120_000.0],  # tx
            [-120_000.0, 120_000.0, 0.0, 0.0],  # ty
            [24_000.0, 24_000.0, 24_000.0, 24_000.0],  # tz (roll: all 4 fins same sign)
        ],
        dtype=np.float64,
    )
    jam_rad = radians(10.0)
    out = alloc.allocate(
        Allocation3DInput(
            requested_torque_body_nm=(0.0, 0.0, 0.0),
            background_torque_body_nm=(0.0, 0.0, 0.0),
            inertia_diag_kg_m2=(1.0e6, 1.0e6, 4.0e4),
            lever_arm_m=9.0,
            sizing_thrust_n=0.0,
            fin_arm_z_m=13.4,
            dynamic_pressure_pa=8_000.0,
            fin_extension=1.0,
            alpha_retro_xy_rad=(0.0, 0.0),
            rcs_available=True,
            allow_fin_allocation=True,
            fin_jacobian_3x4=b_3x4,
            fin_ref_deflection_rad=(0.0, 0.0, 0.0, 0.0),
            fin_ref_torque_body_nm=(0.0, 0.0, 0.0),
            fin_jammed_mask=(False, False, True, False),
            fin_jammed_angles_rad=(0.0, 0.0, jam_rad, 0.0),
        )
    )
    u_cmd = np.array(out.fin_deflection_rad, dtype=np.float64)
    # Jammed fin 2 locked at +10 deg
    assert isclose(u_cmd[2], jam_rad, abs_tol=1e-9)
    # Net roll torque with 10x roll priority weight must be essentially cancelled (< 50 N*m!)
    net_tau = b_3x4 @ u_cmd
    assert abs(float(net_tau[2])) < 50.0
    # All healthy fins stay well inside +/-20 deg physical stop
    assert np.all(np.abs(u_cmd) <= radians(20.0) + 1e-9)

    # Also test severe saturation + ill-conditioned inertia ratio (Ixx/Izz = 25 -> cond(H) ~ 1e6):
    # Exact active-set 3x3 QP must STILL match req_guarded[2] to < 50 N*m even when a fin saturates!
    out_sat = alloc.allocate(
        Allocation3DInput(
            requested_torque_body_nm=(160_000.0, 0.0, -500.0),
            background_torque_body_nm=(0.0, 0.0, 0.0),
            inertia_diag_kg_m2=(300_000.0, 300_000.0, 12_000.0),
            lever_arm_m=9.0,
            sizing_thrust_n=0.0,
            fin_arm_z_m=13.4,
            dynamic_pressure_pa=18_000.0,
            fin_extension=1.0,
            alpha_retro_xy_rad=(0.0, 0.0),
            rcs_available=True,
            allow_fin_allocation=True,
            angular_accel_guard_deg_s2=35.0,
            fin_jacobian_3x4=b_3x4 * 2.0,
            fin_ref_deflection_rad=(0.0, 0.0, 0.0, 0.0),
            fin_ref_torque_body_nm=(0.0, 0.0, 0.0),
            fin_jammed_mask=(False, False, True, False),
            fin_jammed_angles_rad=(0.0, 0.0, jam_rad, 0.0),
        )
    )
    u_sat = np.array(out_sat.fin_deflection_rad, dtype=np.float64)
    net_sat = (b_3x4 * 2.0) @ u_sat
    assert abs(float(net_sat[2]) - (-500.0)) < 50.0
    assert abs(u_sat[3]) >= radians(19.9)  # Fin 3 genuinely saturated at -20 deg!
    assert np.all(np.abs(u_sat) <= radians(20.0) + 1e-9)


# ---------------------------------------------------------------------------
# 4. Online 2 Hz N=12 Log-Mass Clarabel GFOLD MPC
# ---------------------------------------------------------------------------


def test_online_logmass_gfold_mpc_optimality_timing_and_fallback() -> None:
    mpc = OnlineGfoldMpc3D(MpcGfold3DConfig())

    # Realistic DM100 3-engine landing burn initiation (h=1500 m, vz=-150 m/s, T in [239, 664] kN)
    sol1 = mpc.solve(
        time_s=500.0,
        pos0_enu_m=(28.0, -18.0, 1_500.0),
        vel0_enu_m_s=(-5.0, 3.0, -150.0),
        mass0_kg=16_200.0,
        dry_mass_kg=14_500.0,
        t_min_n=239_140.0,
        t_max_n=664_285.0,
        isp_eff_s=325.0,
        gravity_m_s2=9.80665,
        v_touchdown_m_s=1.2,
        a_aero_now_enu_m_s2=(0.08, -0.05, 0.35),
        ground_effect_coeff=0.06,
    )
    assert sol1 is not None
    assert sol1.feasible
    assert sol1.lossless_ok
    assert sol1.lossless_ratio_0 >= 0.98
    assert sol1.iterations <= 35
    assert sol1.solve_ms < 50.0  # Generous CI headroom; typical ~3-6 ms
    assert sol1.k_vert == 9
    assert float(np.hypot(sol1.pos_enu_m[-1][0], sol1.pos_enu_m[-1][1])) < 1.0
    # Last nodes enforce near-vertical thrust cone <= 3 deg
    for k in range(sol1.k_vert, len(sol1.accel_enu_m_s2)):
        ax, ay, az = sol1.accel_enu_m_s2[k]
        tilt_k = degrees(atan2(float(np.hypot(ax, ay)), float(az)))
        assert tilt_k <= 3.1

    # Second replan 0.5 s later reuses cached CSC solver via solver.update()
    sol2 = mpc.solve(
        time_s=500.5,
        pos0_enu_m=(25.0, -16.0, 1_425.0),
        vel0_enu_m_s=(-4.5, 2.7, -142.0),
        mass0_kg=16_120.0,
        dry_mass_kg=14_500.0,
        t_min_n=239_140.0,
        t_max_n=664_285.0,
        isp_eff_s=325.0,
        gravity_m_s2=9.80665,
        v_touchdown_m_s=1.2,
        a_aero_now_enu_m_s2=(0.07, -0.04, 0.32),
        ground_effect_coeff=0.06,
    )
    assert sol2 is not None and sol2.feasible and sol2.lossless_ok and sol2.iterations <= 35
    assert mpc.total_solves == 2
    assert mpc.total_valid_solves == 2

    # 20 ms lead query & 1.5 s fallback hold chain
    q_ok = mpc.sample_reference(501.0)
    assert q_ok is not None
    p_ref, v_ref, a_ref = q_ok
    assert p_ref.shape == (3,) and v_ref.shape == (3,) and a_ref.shape == (3,)
    # Beyond 1.5 s stale hold window -> cleanly returns None (triggers v1 fallback!)
    assert mpc.sample_reference(502.5) is None


# ---------------------------------------------------------------------------
# 5. Edge-v2 48-Case Validation Suite Registry
# ---------------------------------------------------------------------------


def test_edge_v2_suite_has_48_unique_cases_with_eo_and_fin_jam() -> None:
    suite = build_edge_v2_suite()
    assert len(suite) == 48
    names = [s.name for s in suite]
    assert len(set(names)) == 48
    assert len(get_validation_suite("edge-v2")) == 48
    eo_cases = [
        s for s in suite if s.physics_v2 is not None and s.physics_v2.engine_out_index is not None
    ]
    fj_cases = [
        s for s in suite if s.physics_v2 is not None and s.physics_v2.fin_jam_index is not None
    ]
    assert len(eo_cases) == 4
    assert len(fj_cases) == 4
    assert any(s.expected_mission_mode == "DOWNRANGE_SPLASHDOWN_ABORT" for s in eo_cases)


def test_run_flight_3d_full_v2_summary_smoke() -> None:
    """Fast 0.1 s end-to-end smoke check verifying run_flight_3d + summary + M3 metrics."""
    assert not GncV2Config().any_enabled()
    assert GncV2Config.full_v2().any_enabled()
    summary, _ = run_flight_3d(
        Scenario3D(name="m3-smoke", seed=1),
        sim_config=SimConfig3D(
            deadline_s=0.10,
            record_replay=False,
            seed=1,
            sensor_seed=2,
            physics_v2=PhysicsV2Config.full_v2(),
            gnc_v2=GncV2Config.full_v2(),
        ),
    )
    assert "mpc_metrics" in summary
    assert "fdir_metrics" in summary
    assert summary["mission_mode"] == "RTLS"
    assert summary["fdir_state"] == "NOMINAL"
