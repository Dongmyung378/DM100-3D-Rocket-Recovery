"""9-Regime Frequency-Domain Bode, Sensitivity (Ms), & Flex-Attenuation Stability Gate (M2).

Verifies DM100 3D v2 CDR Rev 3.1 §3.1 across ALL 9 flight regimes & axes under simultaneous
worst-case dispersions:
  - 2-step / 40 ms loop transport delay + 50 Hz ZOH
  - 18 Hz 2nd-order Butterworth anti-aliasing filter (500 Hz)
  - Single-path 4th-order Elliptic CSI filter (P/Y: 1.20 Hz ascent / 2.25 Hz booster; Roll bypass)
  - Matched DOB (identical H_filt on model torque, tau_xy = 0.50 s, tau_z = 0.25 s)
  - Coupled dual-tank LOX/CH4 slosh dipoles with -20% ring-baffle damping (zeta0 = 0.064)
  - +/-10% flex frequency dispersion, M_k / m in [0.21, 0.25, 0.35], +6% TVC / +13% FIN gain

Mandatory CI Gate Requirements (all 9 regimes):
  - Gain Margin (GM) >= 6.0 dB
  - Phase Margin (PM) >= 35.0 deg
  - Mode 1 & Mode 2 flexible bending open-loop peaks <= -10.0 dB
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, pi, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from dm100_3d.config import (
    DM100VehicleConfig,
    DualTankGeometry,
    dual_tank_propellant_levels,
    mass_properties_3d,
)
from dm100_3d.gnc.notch_filter import StructuralFilterConfig, design_elliptic_lpf_sos
from dm100_3d.physics.flex_3d import (
    FlexBodySettings,
    compute_flex_modal_properties,
    free_free_mode_eval,
)
from dm100_3d.physics.slosh_3d import DualTankSloshSettings, _tank_modal_params


@dataclass(frozen=True, slots=True)
class FlightRegimeSpec:
    name: str
    axis: str  # "pitch_yaw" or "roll"
    actuator: str  # "tvc", "fin", or "rcs"
    upper_attached: bool
    main_prop_kg: float
    rcs_prop_kg: float
    gear_ext: float
    thrust_n: float
    dynamic_pressure_pa: float
    axial_accel_m_s2: float
    kp: float
    kd: float


@dataclass(frozen=True, slots=True)
class RegimeMarginReport:
    name: str
    axis: str
    actuator: str
    worst_gm_db: float
    worst_pm_deg: float
    worst_flex1_peak_db: float
    worst_flex2_peak_db: float
    worst_ms: float
    passed: bool


# Locked CDR Rev 3.1 gains: Ascent P/Y (0.40, 1.20), Booster P/Y (0.55, 1.65), Roll (1.20, 2.50)
CDR_NINE_REGIMES: tuple[FlightRegimeSpec, ...] = (
    FlightRegimeSpec(
        name="ascent_liftoff_tvc_py",
        axis="pitch_yaw",
        actuator="tvc",
        upper_attached=True,
        main_prop_kg=69_000.0,
        rcs_prop_kg=300.0,
        gear_ext=0.0,
        thrust_n=1_550_000.0,
        dynamic_pressure_pa=500.0,
        axial_accel_m_s2=15.5,
        kp=0.40,
        kd=1.20,
    ),
    FlightRegimeSpec(
        name="ascent_maxq_tvc_py",
        axis="pitch_yaw",
        actuator="tvc",
        upper_attached=True,
        main_prop_kg=38_000.0,
        rcs_prop_kg=300.0,
        gear_ext=0.0,
        thrust_n=1_350_000.0,
        dynamic_pressure_pa=30_000.0,
        axial_accel_m_s2=22.0,
        kp=0.40,
        kd=1.20,
    ),
    FlightRegimeSpec(
        name="entry_burn_tvc_py",
        axis="pitch_yaw",
        actuator="tvc",
        upper_attached=False,
        main_prop_kg=5_200.0,
        rcs_prop_kg=220.0,
        gear_ext=0.0,
        thrust_n=660_000.0,
        dynamic_pressure_pa=22_000.0,
        axial_accel_m_s2=42.0,
        kp=0.55,
        kd=1.65,
    ),
    FlightRegimeSpec(
        name="entry_burn_fin_py",
        axis="pitch_yaw",
        actuator="fin",
        upper_attached=False,
        main_prop_kg=4_800.0,
        rcs_prop_kg=220.0,
        gear_ext=0.0,
        thrust_n=660_000.0,
        dynamic_pressure_pa=22_000.0,
        axial_accel_m_s2=42.0,
        kp=0.55,
        kd=1.65,
    ),
    FlightRegimeSpec(
        name="aero_glide_fin_py",
        axis="pitch_yaw",
        actuator="fin",
        upper_attached=False,
        main_prop_kg=3_500.0,
        rcs_prop_kg=200.0,
        gear_ext=0.0,
        thrust_n=0.0,
        dynamic_pressure_pa=18_000.0,
        axial_accel_m_s2=12.0,
        kp=0.55,
        kd=1.65,
    ),
    FlightRegimeSpec(
        name="landing_3eng_tvc_py",
        axis="pitch_yaw",
        actuator="tvc",
        upper_attached=False,
        main_prop_kg=3_000.0,
        rcs_prop_kg=180.0,
        gear_ext=0.0,
        thrust_n=640_000.0,
        dynamic_pressure_pa=4_500.0,
        axial_accel_m_s2=45.0,
        kp=0.55,
        kd=1.65,
    ),
    FlightRegimeSpec(
        name="landing_1eng_tvc_py",
        axis="pitch_yaw",
        actuator="tvc",
        upper_attached=False,
        main_prop_kg=2_000.0,
        rcs_prop_kg=160.0,
        gear_ext=1.0,
        thrust_n=180_000.0,
        dynamic_pressure_pa=200.0,
        axial_accel_m_s2=18.0,
        kp=0.55,
        kd=1.65,
    ),
    FlightRegimeSpec(
        name="ascent_roll_rcs",
        axis="roll",
        actuator="rcs",
        upper_attached=True,
        main_prop_kg=45_000.0,
        rcs_prop_kg=300.0,
        gear_ext=0.0,
        thrust_n=1_450_000.0,
        dynamic_pressure_pa=25_000.0,
        axial_accel_m_s2=20.0,
        kp=1.20,
        kd=2.50,
    ),
    FlightRegimeSpec(
        name="aero_glide_roll_fin",
        axis="roll",
        actuator="fin",
        upper_attached=False,
        main_prop_kg=3_500.0,
        rcs_prop_kg=200.0,
        gear_ext=0.0,
        thrust_n=0.0,
        dynamic_pressure_pa=18_000.0,
        axial_accel_m_s2=12.0,
        kp=1.20,
        kd=2.50,
    ),
)


def _compute_bode_margins(
    freqs_hz: NDArray[np.float64],
    loop_tf: NDArray[np.complex128],
    *,
    f1_hz: float,
    f2_hz: float,
) -> tuple[float, float, float, float, float]:
    """Compute `(gm_db, pm_deg, flex1_peak_db, flex2_peak_db, peak_ms)` from L(jw)."""
    mag = np.abs(loop_tf)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    phase_deg = np.rad2deg(np.unwrap(np.angle(loop_tf)))

    # Normalize phase near low-frequency rigid-body -180 + phase_lead region
    # (for double integrator + PD lead, low-freq phase starts near -90 deg).
    while phase_deg[0] > 45.0:
        phase_deg -= 360.0
    while phase_deg[0] < -270.0:
        phase_deg += 360.0

    # Gain Margin: minimum -20*log10(|L|) at all -180 deg (+ 360*k) crossings above 0.05 Hz
    gm_candidates: list[float] = []
    # Check crossings of (-180 - 360*k) for k in {0, 1, 2}
    for target_ph in (-180.0, -540.0):
        diff = phase_deg - target_ph
        cross = np.where((diff[:-1] * diff[1:] <= 0.0) & (freqs_hz[:-1] >= 0.05))[0]
        for idx in cross:
            denom = float(diff[idx + 1] - diff[idx])
            frac = 0.0 if abs(denom) < 1e-12 else float(-diff[idx] / denom)
            m_at = float(mag[idx] + frac * (mag[idx + 1] - mag[idx]))
            if m_at > 1e-9:
                gm_candidates.append(-20.0 * log10(m_at))
    gm_db = min(gm_candidates) if gm_candidates else 40.0

    # Phase Margin: minimum (180 + phase_mod) at all 0 dB gain crossings
    pm_candidates: list[float] = []
    gdiff = mag_db
    gcross = np.where((gdiff[:-1] * gdiff[1:] <= 0.0) & (freqs_hz[:-1] >= 0.03))[0]
    for idx in gcross:
        denom = float(gdiff[idx + 1] - gdiff[idx])
        frac = 0.0 if abs(denom) < 1e-12 else float(-gdiff[idx] / denom)
        ph_at = float(phase_deg[idx] + frac * (phase_deg[idx + 1] - phase_deg[idx]))
        # Distance to nearest -180 - 360*k critical phase
        wrapped = ((ph_at + 180.0) % 360.0) - 180.0
        pm_candidates.append(180.0 + wrapped if wrapped <= 0.0 else 180.0 - wrapped)
    pm_deg = min(pm_candidates) if pm_candidates else 90.0

    # Flex 1 & Flex 2 open-loop peak gain [dB] in +/-25% windows around modal frequencies
    def _band_peak(f_center: float) -> float:
        mask = (freqs_hz >= 0.78 * f_center) & (freqs_hz <= 1.25 * f_center)
        if not np.any(mask):
            return -60.0
        return float(np.max(mag_db[mask]))

    flex1_db = _band_peak(f1_hz)
    flex2_db = _band_peak(f2_hz)

    # Peak sensitivity Ms = max |1 / (1 + L(jw))|
    ms = float(np.max(1.0 / np.maximum(np.abs(1.0 + loop_tf), 1e-6)))
    return gm_db, pm_deg, flex1_db, flex2_db, ms


def evaluate_regime_loop_tf(
    regime: FlightRegimeSpec,
    freqs_hz: NDArray[np.float64],
    *,
    vehicle: DM100VehicleConfig | None = None,
    filter_cfg: StructuralFilterConfig | None = None,
    delay_s: float = 0.04,
    flex_freq_scale: float = 1.0,
    flex_mk_frac: float = 0.25,
    baffle_damping_scale: float = 1.0,
    act_gain_scale: float = 1.0,
    include_matched_dob: bool = True,
) -> tuple[NDArray[np.complex128], float, float]:
    """Compute closed-form open-loop transfer function `L(jw)` for one regime and dispersion."""
    veh = vehicle or DM100VehicleConfig()
    fcfg = filter_cfg or StructuralFilterConfig()
    w = 2.0 * pi * freqs_hz
    s = 1j * w
    ts = 1.0 / fcfg.sample_rate_hz

    tot_m = (
        veh.booster_dry_mass_kg
        + regime.main_prop_kg
        + regime.rcs_prop_kg
        + (veh.upper_stage.total_mass_kg if regime.upper_attached else 0.0)
    )
    dtg = DualTankGeometry()
    mp = mass_properties_3d(
        tot_m,
        regime.gear_ext,
        vehicle=veh,
        upper_attached=regime.upper_attached,
        dual_tank=True,
        rcs_propellant_kg=regime.rcs_prop_kg,
        dual_tank_geom=dtg,
    )
    i_ax = mp.inertia_diag_kg_m2[2] if regime.axis == "roll" else mp.inertia_diag_kg_m2[1]

    # 1. Actuator dynamics + gain dispersion
    if regime.actuator == "tvc":
        g_act = act_gain_scale / (veh.propulsion.gimbal_tau_s * s + 1.0)
        z_act = veh.geometry.nozzle_station_m
        act_arm = max(mp.cg_station_m - z_act, 1.0)
    elif regime.actuator == "fin":
        g_act = act_gain_scale / (veh.fins.actuator_time_constant_s * s + 1.0)
        z_act = veh.fins.station_m
        act_arm = max(z_act - mp.cg_station_m, 1.0)
    else:
        g_act = np.full_like(s, act_gain_scale, dtype=np.complex128)
        z_act = veh.geometry.rcs_station_m
        act_arm = max(z_act - mp.cg_station_m, 1.0)

    # 2. Digital ZOH (50 Hz) + transport delay (2-step / 40 ms)
    zoh = (1.0 - np.exp(-s * ts)) / np.where(np.abs(s * ts) > 1e-12, s * ts, 1.0)
    g_delay = np.exp(-s * delay_s) * zoh

    # 3. 18 Hz 2nd-order Butterworth AA filter (at 500 Hz)
    sos_aa = signal.butter(2, 18.0 / 250.0, btype="low", output="sos")
    _, h_aa = signal.sosfreqz(sos_aa, worN=freqs_hz, fs=500.0)

    # 4. Single-path 4th-order Elliptic LPF (P/Y ONLY; Roll strictly bypassed!)
    if regime.axis == "pitch_yaw":
        fc = fcfg.ascent_cutoff_hz if regime.upper_attached else fcfg.booster_cutoff_hz
        sos_csi = design_elliptic_lpf_sos(
            fc,
            sample_rate_hz=fcfg.sample_rate_hz,
            order=fcfg.order,
            rp_db=fcfg.passband_ripple_db,
            rs_db=fcfg.stopband_atten_db,
        )
        _, h_filt = signal.sosfreqz(sos_csi, worN=freqs_hz, fs=fcfg.sample_rate_hz)
    else:
        h_filt = np.ones_like(s, dtype=np.complex128)

    # 5. Rigid-body + dual-tank slosh dipole + 2-mode flex-body plant (tau -> theta)
    fprops = compute_flex_modal_properties(
        total_mass_kg=tot_m,
        main_propellant_kg=regime.main_prop_kg,
        rcs_propellant_kg=regime.rcs_prop_kg,
        gear_extension=regime.gear_ext,
        upper_attached=regime.upper_attached,
        vehicle=veh,
        settings=FlexBodySettings(
            freq_mode1_scale=flex_freq_scale,
            freq_mode2_scale=flex_freq_scale,
        ),
        dual_tank_geom=dtg,
    )
    f1_hz, f2_hz = fprops.freq_hz

    if regime.axis == "roll":
        # Roll axis has zero lateral bending & zero axisymmetric cylindrical slosh coupling
        p_theta = 1.0 / (i_ax * s * s)
    else:
        # Coupled rigid + 2-tank slosh exact 3x3 frequency response
        m_lox, h_lox, _, m_ch4, h_ch4, _ = dual_tank_propellant_levels(regime.main_prop_kg, dtg)
        sl_set = DualTankSloshSettings(geometry=dtg, damping_scale=baffle_damping_scale)
        m1, lp1, z1 = _tank_modal_params(
            m_lox, h_lox, dtg.lox_bottom_m, dtg.tank_inner_radius_m, sl_set
        )
        m2, lp2, z2 = _tank_modal_params(
            m_ch4, h_ch4, dtg.ch4_bottom_m, dtg.tank_inner_radius_m, sl_set
        )
        d1 = z1 - mp.cg_station_m
        d2 = z2 - mp.cg_station_m
        mu1 = m1 * (1.0 - m1 / tot_m)
        mu2 = m2 * (1.0 - m2 / tot_m)
        c12 = -(m1 * m2 / tot_m)
        w1_sl = sqrt(max(regime.axial_accel_m_s2 / max(lp1, 0.2), 0.01))
        w2_sl = sqrt(max(regime.axial_accel_m_s2 / max(lp2, 0.2), 0.01))
        z_baf = dtg.ring_baffle_zeta0 * baffle_damping_scale

        # Schur complement reduction of 2 slosh modes into rotational impedance
        a11 = mu1 * s * s + (2.0 * z_baf * m1 * w1_sl) * s + m1 * w1_sl * w1_sl
        a22 = mu2 * s * s + (2.0 * z_baf * m2 * w2_sl) * s + m2 * w2_sl * w2_sl
        a12 = c12 * s * s
        det_sl = a11 * a22 - a12 * a12
        b1_s = m1 * d1 * s * s
        b2_s = m2 * d2 * s * s
        # Equivalent slosh impedance subtracted from I_ax * s^2
        z_slosh = (b1_s * (a22 * b1_s - a12 * b2_s) + b2_s * (a11 * b2_s - a12 * b1_s)) / np.where(
            np.abs(det_sl) > 1e-12, det_sl, 1.0
        )
        p_rig_slosh = 1.0 / (i_ax * s * s - z_slosh)

        # Additive 2-mode elastic slope at IMU station z_imu = 21.8 m (force F = tau / act_arm)
        z_imu = min(21.8, fprops.length_m)
        p_flex = np.zeros_like(s, dtype=np.complex128)
        for k in range(2):
            phi_a = free_free_mode_eval(z_act, fprops.length_m, k)[0]
            sig_imu = free_free_mode_eval(z_imu, fprops.length_m, k)[1]
            wk = 2.0 * pi * fprops.freq_hz[k]
            zk = fprops.zeta[k]
            mk = flex_mk_frac * tot_m
            # Sign of actuator force relative to positive pitch torque:
            # TVC at nozzle (z < cg): +tau_y requires -F_x; FIN/RCS (z > cg): +tau_y requires +F_x.
            sgn_f = -1.0 if z_act < mp.cg_station_m else 1.0
            gain_k = (sgn_f * phi_a * sig_imu) / (act_arm * mk)
            p_flex = p_flex + gain_k / (s * s + 2.0 * zk * wk * s + wk * wk)

        p_theta = p_rig_slosh + p_flex

    # 6. Outer attitude PD controller (I_nom * (kd*kp + kd*s)) + Matched DOB
    c_pd = i_ax * (regime.kd * regime.kp + regime.kd * s)
    if include_matched_dob:
        tau_dob = 0.25 if regime.axis == "roll" else 0.50
        q_dob = 1.0 / (tau_dob * s + 1.0)
        # Matched DOB filters measured angular accel and delayed model torque with identical H_filt
        g_act_nom = g_act / act_gain_scale
        g_meas = g_act * g_delay * h_aa * p_theta
        l_pd = c_pd * h_filt * g_meas
        l_dob = q_dob * h_filt * (i_ax * s * s * g_meas - g_act_nom * g_delay)
        loop_tf = l_pd + l_dob
    else:
        loop_tf = c_pd * h_filt * g_act * g_delay * h_aa * p_theta

    return loop_tf, f1_hz, f2_hz


def verify_all_nine_regimes(
    *,
    vehicle: DM100VehicleConfig | None = None,
    filter_cfg: StructuralFilterConfig | None = None,
    gm_min_db: float = 6.0,
    pm_min_deg: float = 35.0,
    flex_max_db: float = -10.0,
) -> list[RegimeMarginReport]:
    """Run full worst-case dispersion sweep across all 9 CDR flight regimes (CI Gate)."""
    veh = vehicle or DM100VehicleConfig()
    fcfg = filter_cfg or StructuralFilterConfig()
    freqs = np.geomspace(0.02, 24.8, 1600)
    reports: list[RegimeMarginReport] = []

    for reg in CDR_NINE_REGIMES:
        worst_gm = 999.0
        worst_pm = 999.0
        worst_f1 = -999.0
        worst_f2 = -999.0
        worst_ms = 0.0

        act_gains = (
            (1.0, 1.13)
            if reg.actuator == "fin"
            else ((1.0, 1.06) if reg.actuator == "tvc" else (1.0,))
        )
        flex_scales = (0.90, 1.0, 1.10) if reg.axis == "pitch_yaw" else (1.0,)
        mk_fracs = (0.21, 0.25, 0.35) if reg.axis == "pitch_yaw" else (0.25,)
        baffle_scales = (0.80, 1.0) if reg.axis == "pitch_yaw" else (1.0,)

        for g_act in act_gains:
            for f_sc in flex_scales:
                for mk_f in mk_fracs:
                    for b_sc in baffle_scales:
                        ltf, f1_hz, f2_hz = evaluate_regime_loop_tf(
                            reg,
                            freqs,
                            vehicle=veh,
                            filter_cfg=fcfg,
                            delay_s=0.04,
                            flex_freq_scale=f_sc,
                            flex_mk_frac=mk_f,
                            baffle_damping_scale=b_sc,
                            act_gain_scale=g_act,
                            include_matched_dob=True,
                        )
                        gm, pm, f1_pk, f2_pk, ms = _compute_bode_margins(
                            freqs, ltf, f1_hz=f1_hz, f2_hz=f2_hz
                        )
                        worst_gm = min(worst_gm, gm)
                        worst_pm = min(worst_pm, pm)
                        worst_f1 = max(worst_f1, f1_pk)
                        worst_f2 = max(worst_f2, f2_pk)
                        worst_ms = max(worst_ms, ms)

        passed = (
            worst_gm >= gm_min_db
            and worst_pm >= pm_min_deg
            and worst_f1 <= flex_max_db
            and worst_f2 <= flex_max_db
        )
        reports.append(
            RegimeMarginReport(
                name=reg.name,
                axis=reg.axis,
                actuator=reg.actuator,
                worst_gm_db=float(worst_gm),
                worst_pm_deg=float(worst_pm),
                worst_flex1_peak_db=float(worst_f1),
                worst_flex2_peak_db=float(worst_f2),
                worst_ms=float(worst_ms),
                passed=bool(passed),
            )
        )
    return reports
