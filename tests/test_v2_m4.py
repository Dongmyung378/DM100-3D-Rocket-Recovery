"""Tier-1 unit & contract tests for DM100 3D v2 M4 (Phase 9: 54-scalar V&V & Report Generator)."""

from __future__ import annotations

import json
import sys
from math import isclose
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dm100_3d.config import DM100VehicleConfig, GncV2Config, PhysicsV2Config
from dm100_3d.physics.atmosphere_3d import Atmosphere3DSettings, AtmosphereLUT, standard_atmosphere
from dm100_3d.sim_3d import Disturbances3D, MuJoCoPlant3D, Scenario3D, SimConfig3D
from dm100_3d.validation_suites import (
    MC_1000_SCALAR_NAMES,
    build_envelope_boundary_suite,
    build_fault_mc_160_suite,
    build_mc_1000_suite,
    get_validation_suite,
    sample_mc_1000_lhs_matrix,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_v2_report import (  # noqa: E402
    clopper_pearson_one_sided_lcb,
    compute_2d_cep_and_ellipses,
    compute_replicated_lhs_deff_binary,
    compute_replicated_lhs_deff_continuous,
    generate_v2_report,
)

# ---------------------------------------------------------------------------
# 1. Three Neutral-Default 54-Scalar Hooks (Zero Legacy Double-Counting)
# ---------------------------------------------------------------------------


def test_upper_air_density_scale_has_zero_low_alt_double_counting() -> None:
    s_base = Atmosphere3DSettings(sea_level_temperature_k=295.0, sea_level_pressure_pa=100_000.0)
    s_hi = Atmosphere3DSettings(
        sea_level_temperature_k=295.0,
        sea_level_pressure_pa=100_000.0,
        upper_air_density_scale=1.03,
    )
    # Below and up to 10 km: bit-exact identical (ideal-gas T0/P0 only, zero double-counting!)
    for alt in (0.0, 2_500.0, 8_000.0, 10_000.0):
        assert standard_atmosphere(alt, s_base) == standard_atmosphere(alt, s_hi)

    # At 12 km (mid-ramp): +1.5%; at >= 14 km: exact +3.0%
    _, _, rho_12_b = standard_atmosphere(12_000.0, s_base)
    _, _, rho_12_h = standard_atmosphere(12_000.0, s_hi)
    assert isclose(rho_12_h / rho_12_b, 1.015, rel_tol=1e-9)

    _, _, rho_25_b = standard_atmosphere(25_000.0, s_base)
    _, _, rho_25_h = standard_atmosphere(25_000.0, s_hi)
    assert isclose(rho_25_h / rho_25_b, 1.03, rel_tol=1e-9)

    lut_b = AtmosphereLUT(s_base)
    lut_h = AtmosphereLUT(s_hi)
    assert lut_b.at(5_000.0) == lut_h.at(5_000.0)
    assert isclose(lut_h.at(25_000.0)[2] / lut_b.at(25_000.0)[2], 1.03, rel_tol=1e-6)


def test_propellant_load_scale_cap_and_inertia_roll_decoupling() -> None:
    cfg = DM100VehicleConfig()
    pv2 = PhysicsV2Config.full_v2()
    # Nominal propellant scale = 1.002 (< 91 m^3 cap)
    sc_nom = Scenario3D(
        disturbances=Disturbances3D(
            propellant_load_scale=1.002, inertia_scale=1.02, inertia_roll_scale=0.97
        ),
        physics_v2=pv2,
    )
    plant_nom = MuJoCoPlant3D(
        vehicle=cfg, scenario=sc_nom, sim_config=SimConfig3D(physics_v2=pv2, gnc_v2=GncV2Config())
    )
    assert isclose(plant_nom.main_prop_kg, cfg.propellant.main_propellant_kg * 1.002, rel_tol=1e-9)

    # Excessive propellant load must be capped at dual-tank 91.0 m^3 physical volume
    sc_over = Scenario3D(
        disturbances=Disturbances3D(propellant_load_scale=1.50),
        physics_v2=pv2,
    )
    plant_over = MuJoCoPlant3D(
        vehicle=cfg, scenario=sc_over, sim_config=SimConfig3D(physics_v2=pv2, gnc_v2=GncV2Config())
    )
    max_cap = (
        0.98
        * pv2.dual_tank_geometry.total_tank_volume_m3
        * pv2.dual_tank_geometry.bulk_propellant_density_kg_m3
    )
    assert isclose(plant_over.main_prop_kg, max_cap, rel_tol=1e-9)

    # Independent lateral vs roll inertia scaling
    sc_ref = Scenario3D(
        disturbances=Disturbances3D(inertia_scale=1.0, inertia_roll_scale=1.0),
        physics_v2=pv2,
    )
    plant_ref = MuJoCoPlant3D(
        vehicle=cfg, scenario=sc_ref, sim_config=SimConfig3D(physics_v2=pv2, gnc_v2=GncV2Config())
    )
    i_ref = plant_ref.body.current_mass_props.inertia_diag_kg_m2
    # Compare with same propellant mass
    sc_roll = Scenario3D(
        disturbances=Disturbances3D(inertia_scale=1.04, inertia_roll_scale=0.96),
        physics_v2=pv2,
    )
    plant_roll = MuJoCoPlant3D(
        vehicle=cfg, scenario=sc_roll, sim_config=SimConfig3D(physics_v2=pv2, gnc_v2=GncV2Config())
    )
    i_sc = plant_roll.body.current_mass_props.inertia_diag_kg_m2
    # On pad (upper attached): booster lateral scales up while booster roll scales down
    assert i_sc[0] > i_ref[0] and i_sc[1] > i_ref[1] and i_sc[2] < i_ref[2]
    # Post-separation booster-only exact ratio check
    mp_b_ref = plant_ref.body.update_mass_properties(
        16_000.0,
        0.0,
        upper_attached=False,
        inertia_scale=1.0,
        inertia_roll_scale=1.0,
        dual_tank=True,
        rcs_propellant_kg=250.0,
        dual_tank_geom=pv2.dual_tank_geometry,
    )
    mp_b_sc = plant_roll.body.update_mass_properties(
        16_000.0,
        0.0,
        upper_attached=False,
        inertia_scale=1.04,
        inertia_roll_scale=0.96,
        dual_tank=True,
        rcs_propellant_kg=250.0,
        dual_tank_geom=pv2.dual_tank_geometry,
    )
    assert isclose(
        mp_b_sc.inertia_diag_kg_m2[0] / mp_b_ref.inertia_diag_kg_m2[0], 1.04, rel_tol=1e-6
    )
    assert isclose(
        mp_b_sc.inertia_diag_kg_m2[2] / mp_b_ref.inertia_diag_kg_m2[2], 0.96, rel_tol=1e-6
    )


# ---------------------------------------------------------------------------
# 2. mc-1000 54-Scalar 10x100 Replicated LHS & Iman-Conover Correlations
# ---------------------------------------------------------------------------


def test_mc_1000_54_scalar_lhs_and_iman_conover_correlations() -> None:
    assert len(MC_1000_SCALAR_NAMES) == 54
    assert len(set(MC_1000_SCALAR_NAMES)) == 54

    mat1 = sample_mc_1000_lhs_matrix()
    mat2 = sample_mc_1000_lhs_matrix()
    assert mat1.shape == (1_000, 54)
    # Strictly deterministic
    assert np.array_equal(mat1, mat2)

    # Verify all 5 Iman-Conover rank correlations from CDR §4.2
    rho_dryden_w = float(spearmanr(np.abs(mat1[:, 0]), mat1[:, 6]).statistic)
    rho_t0_p0 = float(spearmanr(mat1[:, 7], mat1[:, 8]).statistic)
    rho_ilat_m = float(spearmanr(mat1[:, 10], mat1[:, 12]).statistic)
    rho_iroll_m = float(spearmanr(mat1[:, 10], mat1[:, 13]).statistic)
    rho_f1_f2 = float(spearmanr(mat1[:, 24], mat1[:, 25]).statistic)

    assert 0.58 <= rho_dryden_w <= 0.72, f"rho_dryden_w={rho_dryden_w}"
    assert -0.42 <= rho_t0_p0 <= -0.28, f"rho_t0_p0={rho_t0_p0}"
    assert 0.63 <= rho_ilat_m <= 0.77, f"rho_ilat_m={rho_ilat_m}"
    assert 0.63 <= rho_iroll_m <= 0.77, f"rho_iroll_m={rho_iroll_m}"
    assert 0.73 <= rho_f1_f2 <= 0.87, f"rho_f1_f2={rho_f1_f2}"

    # Verify untruncated normal tails exist (> 2.8 sigma) & physical bounds hold
    assert float(np.max(np.abs(mat1[:, 0]))) > 7.5  # wind_ground > 2.8 sigma
    assert float(np.min(mat1[:, 4])) >= 0.0  # gust_amp >= 0
    assert float(np.min(mat1[:, 6])) >= 1.0 and float(np.max(mat1[:, 6])) <= 2.5
    assert set(np.unique(mat1[:, 45]).astype(int)) == {0, 1, 2}

    # Build full 1,000 Scenario3D suite and verify replicate blocks + fixed lat=37.0 deg
    suite = build_mc_1000_suite()
    assert len(suite) == 1_000
    assert len({s.name for s in suite}) == 1_000
    assert all(s.atmosphere.latitude_deg == 37.0 for s in suite)
    assert all(s.atmosphere.density_scale == 1.0 for s in suite)
    assert all(s.suite_metadata is not None for s in suite)
    rep_counts = [sum(1 for s in suite if s.suite_metadata["replicate_id"] == r) for r in range(10)]  # type: ignore[index]
    assert rep_counts == [100] * 10
    assert len(get_validation_suite("mc-1000")) == 1_000


# ---------------------------------------------------------------------------
# 3. fault-mc-160 & envelope-boundary Suite Registry & Metadata
# ---------------------------------------------------------------------------


def test_fault_mc_160_and_envelope_boundary_suites() -> None:
    fsuite = build_fault_mc_160_suite()
    assert len(fsuite) == 160
    assert len({s.name for s in fsuite}) == 160
    assert len(get_validation_suite("fault-mc-160")) == 160
    strata = {}
    for s in fsuite:
        assert s.suite_metadata is not None
        st = s.suite_metadata["stratum"]
        strata[st] = strata.get(st, 0) + 1
    assert strata == {
        "stratum_1_eo_ascent_abort": 40,
        "stratum_2_eo_ascent_rtls": 40,
        "stratum_3_eo_postsep_3to2": 40,
        "stratum_4_fin_jam_glide": 40,
    }

    esuite = build_envelope_boundary_suite()
    assert len(esuite) == 100
    assert len({s.name for s in esuite}) == 100
    assert len(get_validation_suite("envelope-boundary")) == 100
    grids = {}
    for s in esuite:
        assert s.suite_metadata is not None
        gid = s.suite_metadata["grid_id"]
        grids[gid] = grids.get(gid, 0) + 1
    assert grids == {
        "g1_wind_slope": 25,
        "g2_finjam_wind": 25,
        "g3_blackout_gyro": 25,
        "g4_flex_dryden": 25,
    }


# ---------------------------------------------------------------------------
# 4. Statistical Report Generator (Clopper-Pearson, DEFF, CEP, 6 Charts)
# ---------------------------------------------------------------------------


def test_v2_report_statistics_and_6_chart_generation(tmp_path: Path) -> None:
    # Exact CDR §4.1 Clopper-Pearson check: 996/1,000 one-sided 95% LCB == 99.0870%
    lcb_996 = clopper_pearson_one_sided_lcb(996, 1_000, alpha=0.05)
    assert isclose(lcb_996, 0.990870, abs_tol=1e-4)
    lcb_1000 = clopper_pearson_one_sided_lcb(1_000, 1_000, alpha=0.05)
    assert isclose(lcb_1000, 0.05 ** (1.0 / 1_000), rel_tol=1e-9)

    # DEFF binary: 1000/1000 -> DEFF_bin = 0.0, p_disp = 1.0
    rep_ids = [i // 100 for i in range(1_000)]
    db_all = compute_replicated_lhs_deff_binary([True] * 1_000, rep_ids)
    assert db_all["deff_bin"] == 0.0 and db_all["p_disp"] == 1.0

    # 1 failure in 4 replicates -> DEFF_bin < 1.88, p_disp >= 0.05
    flags = [True] * 1_000
    for idx in (5, 105, 205, 305):
        flags[idx] = False
    db_4 = compute_replicated_lhs_deff_binary(flags, rep_ids)
    assert db_4["deff_bin"] < 1.88 and db_4["p_disp"] >= 0.05

    # Continuous DEFF on stratified replicates should be <= 1.05
    rng = np.random.default_rng(7)
    strat_vals = []
    for _ in range(10):
        # Perfectly stratified replicate means -> tiny between-replicate variance
        strat_vals.extend((0.12 + rng.normal(0.0, 0.03, size=100)).tolist())
        # Center each replicate mean closely
        strat_vals[-100:] = (
            np.asarray(strat_vals[-100:]) - np.mean(strat_vals[-100:]) + 0.12
        ).tolist()
    dc = compute_replicated_lhs_deff_continuous(strat_vals, rep_ids)
    assert dc["deff_cont"] <= 0.1

    # 2D CEP multipliers match CDR §4.1 exact chi^2_2 values
    xy = rng.normal(0.0, [0.12, 0.08], size=(1_000, 2))
    cep = compute_2d_cep_and_ellipses(xy)
    assert isclose(cep["ellipses"]["cep50"]["scale_sigma"], 1.17741, abs_tol=1e-4)
    assert isclose(cep["ellipses"]["cep95"]["scale_sigma"], 2.44775, abs_tol=1e-4)
    assert isclose(cep["ellipses"]["sigma3_98p89"]["scale_sigma"], 3.0, abs_tol=1e-9)
    assert isclose(cep["ellipses"]["cep99"]["scale_sigma"], 3.03485, abs_tol=1e-4)
    assert isclose(cep["ellipses"]["cep99p7"]["scale_sigma"], 3.40861, abs_tol=1e-4)

    # Synthetic end-to-end 6-chart report generation
    mc_results = []
    for i in range(100):
        mc_results.append(
            {
                "scenario": f"mc1000-r{i // 10:02d}-c{i % 10:03d}",
                "requirements_passed": True,
                "catastrophic_failure": False,
                "pos_xy_m": [float(xy[i, 0]), float(xy[i, 1])],
                "position_error_m": float(np.hypot(xy[i, 0], xy[i, 1])),
                "peak_angular_acceleration_deg_s2": 3.1,
                "peak_roll_deg": 1.4,
                "peak_fin_hinge_nm": 900.0,
                "main_remaining_kg": 2_800.0,
                "post_contact": {"max_leg_stroke_m": 0.45},
                "physics_v2_metrics": {"max_flex_tip_deflection_m": 0.04},
                "mpc_metrics": {"max_solve_ms": 6.5, "max_iterations": 12},
                "fdir_metrics": {"engine_failed_7": [False] * 7, "fin_jammed_4": [False] * 4},
                "suite_metadata": {"replicate_id": i // 10},
            }
        )
    mc_path = tmp_path / "mc.json"
    mc_path.write_text(json.dumps({"results": mc_results}), encoding="utf-8")

    fault_results = []
    for st_i, st_name in enumerate(
        (
            "stratum_1_eo_ascent_abort",
            "stratum_2_eo_ascent_rtls",
            "stratum_3_eo_postsep_3to2",
            "stratum_4_fin_jam_glide",
        ),
        start=1,
    ):
        for k in range(10):
            fault_results.append(
                {
                    "scenario": f"faultmc-s{st_i}-{k:02d}",
                    "requirements_passed": True,
                    "catastrophic_failure": False,
                    "mission_mode": "DOWNRANGE_SPLASHDOWN_ABORT" if st_i == 1 else "RTLS",
                    "suite_metadata": {"stratum": st_name},
                }
            )
    fault_path = tmp_path / "fault.json"
    fault_path.write_text(json.dumps({"results": fault_results}), encoding="utf-8")

    env_results = []
    for s in build_envelope_boundary_suite():
        assert s.suite_metadata is not None
        r_i = int(s.suite_metadata["row"])
        c_i = int(s.suite_metadata["col"])
        tier = (
            "NOMINAL_PASS"
            if (r_i + c_i <= 4)
            else ("SURVIVED_OUT_OF_SPEC" if r_i + c_i <= 6 else "DESTROYED")
        )
        env_results.append(
            {
                "scenario": s.name,
                "outcome": "landed" if tier != "DESTROYED" else "tip_over",
                "outcome_tier": tier,
                "position_error_m": 0.25 if tier == "NOMINAL_PASS" else 3.5,
                "suite_metadata": s.suite_metadata,
            }
        )
    env_path = tmp_path / "env.json"
    env_path.write_text(json.dumps({"results": env_results}), encoding="utf-8")

    charts_dir = tmp_path / "charts"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    rep = generate_v2_report(
        mc_1000_path=mc_path,
        fault_mc_path=fault_path,
        envelope_path=env_path,
        out_json=out_json,
        out_md=out_md,
        charts_dir=charts_dir,
    )
    assert rep["schema"] == "dm100_3d_validation_merged_v2"
    assert len(rep["charts"]) == 6
    for ch in rep["charts"]:
        assert Path(ch).exists() and Path(ch).stat().st_size > 5_000
    assert out_json.exists() and out_md.exists()


def test_causal_asymmetric_gimbal_roll_ff_matches_cluster_physics() -> None:
    """3-axis causal asymmetric thrust-offset FF must match 6-engine gimbal roll coupling."""
    from math import radians

    from dm100_3d.config import DM100VehicleConfig
    from dm100_3d.gnc.fdir_3d import CausalFdir3D
    from dm100_3d.physics.faults_3d import EngineClusterFaultModel3D, FaultSettings3D

    veh = DM100VehicleConfig()
    fdir = CausalFdir3D(vehicle=veh)
    fdir.engine_failed[1] = True
    mask_6 = fdir.healthy_ascent_mask()
    gim = (radians(2.5), radians(-3.2))

    cluster = EngineClusterFaultModel3D(
        seed=42,
        vehicle=veh,
        settings=FaultSettings3D(
            engine_out_index=1,
            engine_out_time_s=0.0,
            side_load_peak_n=0.0,
        ),
    )
    cluster.engine_failed[1] = True
    cluster.engine_throttles[:] = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cl_out = cluster.step_cluster(
        time_s=50.0,
        dt_s=0.01,
        phase="ascent",
        req_total_throttle=6.0 / 7.0,
        active_engines=6,
        engine_mask_7=mask_6,
        engine_throttles_cmd_7=None,
        engine_tau_s=0.04,
        ambient_pressure_pa=10_000.0,
        cg_station_m=10.5,
        gimbal_rad=gim,
        thrust_scale=1.0,
        delivered_ratio=1.0,
        main_prop_kg=30_000.0,
        touchdown_detected=False,
    )
    ff = fdir.causal_asymmetric_torque_ff_nm(
        engine_mask_7=mask_6,
        total_thrust_n=cl_out.total_thrust_n,
        full_thrust_7_n=0.0,
        gimbal_rad=gim,
    )
    assert abs(cl_out.asymmetric_torque_body_nm[2]) > 5_000.0
    np.testing.assert_allclose(ff, cl_out.asymmetric_torque_body_nm, rtol=1e-6, atol=1e-3)


def test_fin_jam_aero_hinge_load_derating() -> None:
    """High supersonic dynamic pressure physically derates hydraulic fin-jam runaway rate."""
    from math import radians

    from dm100_3d.config import DM100VehicleConfig
    from dm100_3d.physics.faults_3d import EngineClusterFaultModel3D, FaultSettings3D

    veh = DM100VehicleConfig()
    s = FaultSettings3D(fin_jam_index=0, fin_jam_time_s=0.0, fin_jam_angle_deg=-10.0)
    cl_vac = EngineClusterFaultModel3D(seed=42, vehicle=veh, settings=s)
    cl_hiq = EngineClusterFaultModel3D(seed=42, vehicle=veh, settings=s)
    f_rate = radians(60.0)
    f_lim = radians(20.0)
    d_vac = cl_vac.apply_fin_jam_dynamics(
        time_s=0.0,
        dt_s=0.02,
        phase="aero_glide",
        fin_deflection=[0.0, 0.0, 0.0, 0.0],
        cmd_fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
        fin_rate_rad_s=f_rate,
        fin_tau_s=0.08,
        fin_limit_rad=f_lim,
        dynamic_pressure_pa=0.0,
    )
    d_hiq = cl_hiq.apply_fin_jam_dynamics(
        time_s=0.0,
        dt_s=0.02,
        phase="aero_glide",
        fin_deflection=[0.0, 0.0, 0.0, 0.0],
        cmd_fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
        fin_rate_rad_s=f_rate,
        fin_tau_s=0.08,
        fin_limit_rad=f_lim,
        dynamic_pressure_pa=12_500.0,
    )
    assert np.isclose(d_vac[0], -radians(1.2), atol=1e-9)
    assert np.isclose(d_hiq[0], -radians(0.6), atol=1e-9)
    assert abs(d_hiq[0]) < 0.55 * abs(d_vac[0])


def test_mekf_separation_reset_clears_cross_covariance() -> None:
    """reset_for_separation must zero stale pos/vel <-> attitude cross-covariance."""
    from dm100_3d.config import DM100VehicleConfig, GncV2Config
    from dm100_3d.gnc.mekf import CausalEstimator3D
    from dm100_3d.physics import Atmosphere3DSettings, AtmosphereLUT, Wind3DConfig, Wind3DLUT

    est = CausalEstimator3D(
        vehicle=DM100VehicleConfig(),
        forecast_atmo=AtmosphereLUT(Atmosphere3DSettings()),
        forecast_wind=Wind3DLUT(Wind3DConfig()),
        gnc_v2=GncV2Config.full_v2(),
    )
    est.P[0:6, 6:15] = 1.25
    est.P[6:15, 0:6] = 1.25
    est.reset_for_separation(
        np.array([100.0, 0.0, 70_000.0]), np.array([1200.0, 0.0, 900.0]), 130.0
    )
    assert float(np.max(np.abs(est.P[0:6, 6:15]))) == 0.0
    assert float(np.max(np.abs(est.P[6:15, 0:6]))) == 0.0
