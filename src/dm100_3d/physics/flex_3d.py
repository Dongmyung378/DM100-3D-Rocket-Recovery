"""NASA SP-8003 2-plane x 2-mode free-free Euler-Bernoulli flexible-body dynamics (M2).

Implements DM100 3D v2 CDR Rev 3.1 §2.1:
  - Orthogonal X_b / Y_b bending planes, 2 free-free beam modes per plane (8 modal states).
  - Exact normalized free-free eigenfunctions (|phi_k(0)| = |phi_k(L)| = 1) with explicit
    linear extrapolation outside [0, L] (e.g., nozzle station z_noz = -0.2 m).
  - Discrete mass-weighted generalized mass M_k(m) in [0.21, 0.35] * m_tot.
  - Mass-scheduled frequencies:
      * Ascent stack (33 m): f1 = 2.4 -> 4.2 Hz, f2 = 6.2 -> 10.5 Hz, zeta = (0.008, 0.012)
      * Booster alone (24 m): f1 = 4.5 -> 6.8 Hz, f2 = 10.5 -> 12.5 Hz, zeta = (0.010, 0.015)
  - Follower-force rigid-body moment correction including both elastic nozzle slope
    AND lateral nozzle displacement offset (-dx_noz * T_z, +dy_noz * T_z).
  - Commanded lateral thrust T*gamma only in modal RHS (prevents spurious Beck flutter).
  - Exact L2 inner-product modal projection across stage separation (33 m -> 24 m).
  - Exact closed-form exponential integrator at multi-rate dt_flex = 0.002 s (500 Hz).
  - IMU measurement contamination at z_imu = 21.8 m (zero roll coupling: d_omega_z = 0).
  - Criterion #20 structural integrity metrics: ||y_tip|| <= 0.18 m & M_allow / M_max >= 1.25.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, cosh, exp, hypot, pi, sin, sinh, sqrt

import numpy as np
from numpy.typing import NDArray

from dm100_3d.config import DM100VehicleConfig, DualTankGeometry, dual_tank_propellant_levels

# Free-free Euler-Bernoulli beam eigenvalues: cos(beta*L)*cosh(beta*L) = 1
_BETA_L: tuple[float, float] = (4.730040744862704, 7.853204624095838)
_ALPHA_K: tuple[float, float] = tuple((cosh(bl) - cos(bl)) / (sinh(bl) - sin(bl)) for bl in _BETA_L)
# Precomputed 12-point Gauss-Legendre nodes/weights on [0, 1] for fast discrete M_k & L2 projection
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_U = 0.5 * (_GL_X + 1.0)
_GL_WU = 0.5 * _GL_W


def free_free_mode_eval(z_m: float, length_m: float, mode_idx: int) -> tuple[float, float, float]:
    """Evaluate normalized free-free beam `(phi_k(z), sigma_k(z)=dphi/dz, kappa_k(z)=d2phi/dz2)`.

    Normalized so `|phi_k(0)| = |phi_k(L)| = 1.0`. Stations outside `[0, L]` (such as
    nozzle exit `z_noz = -0.2 m`) use explicit tangent linear extrapolation (CDR §2.1).
    """
    bl = _BETA_L[mode_idx]
    ak = _ALPHA_K[mode_idx]
    beta = bl / max(float(length_m), 1e-6)

    if z_m < 0.0:
        phi0 = 1.0
        sig0 = -ak * beta
        return phi0 + z_m * sig0, sig0, 0.0
    if z_m > length_m:
        c_bl = cos(bl)
        ch_bl = cosh(bl)
        s_bl = sin(bl)
        sh_bl = sinh(bl)
        phi_l = 0.5 * (ch_bl + c_bl - ak * (sh_bl + s_bl))
        sig_l = 0.5 * beta * (sh_bl - s_bl - ak * (ch_bl + c_bl))
        return phi_l + (z_m - length_m) * sig_l, sig_l, 0.0

    bu = bl * (z_m / length_m)
    ch = cosh(bu)
    sh = sinh(bu)
    c = cos(bu)
    s = sin(bu)
    phi = 0.5 * (ch + c - ak * (sh + s))
    sig = 0.5 * beta * (sh - s - ak * (ch + c))
    kap = 0.5 * (beta * beta) * (ch - c - ak * (sh - s))
    return float(phi), float(sig), float(kap)


def _compute_stage_sep_projection(
    l_stack: float = 33.0, l_booster: float = 24.0
) -> NDArray[np.float64]:
    """Compute exact 2x2 L2 projection matrix P mapping 33 m stack modes -> 24 m booster modes."""
    gl_x, gl_w = np.polynomial.legendre.leggauss(32)
    zs = 0.5 * l_booster * (gl_x + 1.0)
    ws = 0.5 * l_booster * gl_w
    phi_33 = np.zeros((2, len(zs)), dtype=np.float64)
    phi_24 = np.zeros((2, len(zs)), dtype=np.float64)
    for k in range(2):
        for j, z in enumerate(zs):
            phi_33[k, j] = free_free_mode_eval(float(z), l_stack, k)[0]
            phi_24[k, j] = free_free_mode_eval(float(z), l_booster, k)[0]

    p_mat = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        denom = float(np.sum(ws * phi_24[i] * phi_24[i]))
        for j in range(2):
            num = float(np.sum(ws * phi_24[i] * phi_33[j]))
            p_mat[i, j] = num / max(denom, 1e-12)
    return p_mat


_SEP_PROJ_33_TO_24: NDArray[np.float64] = _compute_stage_sep_projection(33.0, 24.0)


@dataclass(frozen=True, slots=True)
class FlexBodySettings:
    """Parameters for 2-plane x 2-mode NASA SP-8003 flexible-body dynamics."""

    dt_flex_s: float = 0.002
    imu_station_m: float = 21.8
    freq_mode1_scale: float = 1.0
    freq_mode2_scale: float = 1.0
    damping_scale: float = 1.0
    # Ascent stack (33 m, full 69.5 t -> MECO 12.3 t)
    ascent_f1_full_hz: float = 2.4
    ascent_f1_meco_hz: float = 4.2
    ascent_f2_full_hz: float = 6.2
    ascent_f2_meco_hz: float = 10.5
    ascent_zeta1: float = 0.008
    ascent_zeta2: float = 0.012
    # Booster alone (24 m, sep 12.3 t -> empty 0 t)
    booster_f1_sep_hz: float = 4.5
    booster_f1_empty_hz: float = 6.8
    booster_f2_sep_hz: float = 10.5
    booster_f2_empty_hz: float = 12.5
    booster_zeta1: float = 0.010
    booster_zeta2: float = 0.015
    # Structural integrity limits (Criterion #20)
    max_tip_deflection_m: float = 0.18
    allowable_bending_moment_nm: float = 5_200_000.0
    min_bending_safety_factor: float = 1.25


@dataclass(frozen=True, slots=True)
class FlexBodyState:
    """8-state modal coordinates for X_b and Y_b bending planes (modes 1 & 2)."""

    eta_x: tuple[float, float] = (0.0, 0.0)
    etad_x: tuple[float, float] = (0.0, 0.0)
    eta_y: tuple[float, float] = (0.0, 0.0)
    etad_y: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class FlexModalProperties:
    length_m: float
    freq_hz: tuple[float, float]
    zeta: tuple[float, float]
    gen_mass_kg: tuple[float, float]
    ei_eff_nm2: float


@dataclass(frozen=True, slots=True)
class FlexStepSample:
    """Instantaneous 500 Hz sample returned per flex sub-step (feeds 18 Hz AA IMU filter)."""

    state: FlexBodyState
    rigid_torque_delta_body_nm: tuple[float, float, float]
    rigid_force_delta_body_n: tuple[float, float, float]
    imu_omega_delta_rad_s: tuple[float, float, float]
    imu_accel_delta_m_s2: tuple[float, float, float]
    tip_deflection_xy_m: tuple[float, float]
    tip_deflection_norm_m: float
    max_bending_moment_nm: float
    modal_freq_hz: tuple[float, float]


def compute_flex_modal_properties(
    *,
    total_mass_kg: float,
    main_propellant_kg: float,
    rcs_propellant_kg: float,
    gear_extension: float,
    upper_attached: bool,
    vehicle: DM100VehicleConfig,
    settings: FlexBodySettings | None = None,
    dual_tank_geom: DualTankGeometry | None = None,
) -> FlexModalProperties:
    """Compute mass-scheduled `(f1, f2)`, `(zeta1, zeta2)`, and discrete `M_k = sum m_i phi_k^2`."""
    s_cfg = settings or FlexBodySettings()
    dtg = dual_tank_geom or DualTankGeometry()
    g = vehicle.geometry
    length_m = g.body_length_m + g.upper_length_m if upper_attached else g.body_length_m
    m_main = max(0.0, float(main_propellant_kg))
    m_rcs = max(0.0, float(rcs_propellant_kg))
    m_tot = max(float(total_mass_kg), 1.0)

    if upper_attached:
        s_fill = max(0.0, min(1.0, (m_main - 12_300.0) / max(69_500.0 - 12_300.0, 1.0)))
        f1 = (
            s_cfg.ascent_f1_meco_hz + s_fill * (s_cfg.ascent_f1_full_hz - s_cfg.ascent_f1_meco_hz)
        ) * s_cfg.freq_mode1_scale
        f2 = (
            s_cfg.ascent_f2_meco_hz + s_fill * (s_cfg.ascent_f2_full_hz - s_cfg.ascent_f2_meco_hz)
        ) * s_cfg.freq_mode2_scale
        z1 = s_cfg.ascent_zeta1 * s_cfg.damping_scale
        z2 = s_cfg.ascent_zeta2 * s_cfg.damping_scale
    else:
        s_fill = max(0.0, min(1.0, m_main / 12_300.0))
        f1 = (
            s_cfg.booster_f1_empty_hz
            + s_fill * (s_cfg.booster_f1_sep_hz - s_cfg.booster_f1_empty_hz)
        ) * s_cfg.freq_mode1_scale
        f2 = (
            s_cfg.booster_f2_empty_hz
            + s_fill * (s_cfg.booster_f2_sep_hz - s_cfg.booster_f2_empty_hz)
        ) * s_cfg.freq_mode2_scale
        z1 = s_cfg.booster_zeta1 * s_cfg.damping_scale
        z2 = s_cfg.booster_zeta2 * s_cfg.damping_scale

    # Discrete mass-weighted generalized mass M_k = sum_i m_i * phi_k(z_i)^2
    gear_z = g.gear_retracted_station_m + max(0.0, min(1.0, float(gear_extension))) * (
        g.gear_deployed_station_m - g.gear_retracted_station_m
    )
    m_lox, h_lox, _, m_ch4, h_ch4, _ = dual_tank_propellant_levels(m_main, dtg)
    m_air = g.airframe_mass_kg
    m_up = vehicle.upper_stage.total_mass_kg if upper_attached else 0.0

    def _uniform_seg_mk(mass_seg: float, z_lo: float, z_hi: float, k_idx: int) -> float:
        if mass_seg <= 1e-6:
            return 0.0
        if abs(z_hi - z_lo) < 1e-4:
            p = free_free_mode_eval(0.5 * (z_lo + z_hi), length_m, k_idx)[0]
            return mass_seg * p * p
        acc = 0.0
        for u, w in zip(_GL_U, _GL_WU, strict=True):
            z = z_lo + float(u) * (z_hi - z_lo)
            p = free_free_mode_eval(z, length_m, k_idx)[0]
            acc += float(w) * p * p
        return mass_seg * acc

    gen_m: list[float] = []
    for k_idx in range(2):
        mk = 0.0
        # Point masses
        for mp_kg, zp_m in (
            (g.engine_mass_kg, g.engine_station_m),
            (g.gear_mass_kg, gear_z),
            (vehicle.fins.mass_kg, vehicle.fins.station_m),
            (m_rcs, g.rcs_station_m),
        ):
            p = free_free_mode_eval(zp_m, length_m, k_idx)[0]
            mk += mp_kg * p * p
        # Distributed segments: airframe [0, 24], CH4, LOX, upper stage [24, 33]
        mk += _uniform_seg_mk(m_air, 0.0, g.body_length_m, k_idx)
        mk += _uniform_seg_mk(m_ch4, dtg.ch4_bottom_m, dtg.ch4_bottom_m + h_ch4, k_idx)
        mk += _uniform_seg_mk(m_lox, dtg.lox_bottom_m, dtg.lox_bottom_m + h_lox, k_idx)
        if upper_attached:
            mk += _uniform_seg_mk(m_up, g.body_length_m, length_m, k_idx)
        # Keep physical generalized mass strictly bounded inside CDR [0.21, 0.35] * m_tot
        mk = max(0.21 * m_tot, min(0.35 * m_tot, mk))
        gen_m.append(float(mk))

    # Effective bending stiffness consistent with 1st free-free mode frequency
    beta1 = _BETA_L[0] / length_m
    omega1 = 2.0 * pi * max(f1, 0.5)
    rho_lin = m_tot / length_m
    ei_eff = (omega1 * omega1) * rho_lin / (beta1**4)

    return FlexModalProperties(
        length_m=length_m,
        freq_hz=(float(max(f1, 0.5)), float(max(f2, 1.0))),
        zeta=(float(max(z1, 1e-4)), float(max(z2, 1e-4))),
        gen_mass_kg=(gen_m[0], gen_m[1]),
        ei_eff_nm2=float(ei_eff),
    )


def project_flex_state_at_separation(state: FlexBodyState) -> FlexBodyState:
    """Project 33 m ascent-stack modal state onto 24 m booster basis via exact L2 inner product."""
    ex = _SEP_PROJ_33_TO_24 @ np.asarray(state.eta_x, dtype=np.float64)
    edx = _SEP_PROJ_33_TO_24 @ np.asarray(state.etad_x, dtype=np.float64)
    ey = _SEP_PROJ_33_TO_24 @ np.asarray(state.eta_y, dtype=np.float64)
    edy = _SEP_PROJ_33_TO_24 @ np.asarray(state.etad_y, dtype=np.float64)
    return FlexBodyState(
        eta_x=(float(ex[0]), float(ex[1])),
        etad_x=(float(edx[0]), float(edx[1])),
        eta_y=(float(ey[0]), float(ey[1])),
        etad_y=(float(edy[0]), float(edy[1])),
    )


def _exact_oscillator_step(
    eta: float, etad: float, omega: float, zeta: float, u_acc: float, dt: float
) -> tuple[float, float, float]:
    """Exact closed-form step of `etadd + 2*zeta*omega*etad + omega^2*eta = u_acc` over `dt`."""
    w2 = omega * omega
    eta_eq = u_acc / max(w2, 1e-9)
    y0 = eta - eta_eq
    v0 = etad
    wd = omega * sqrt(max(1.0 - zeta * zeta, 1e-9))
    e_dec = exp(-zeta * omega * dt)
    c = cos(wd * dt)
    s = sin(wd * dt)
    coeff_s = (v0 + zeta * omega * y0) / wd
    y_new = e_dec * (y0 * c + coeff_s * s)
    v_new = e_dec * ((v0) * c - ((zeta * omega * v0 + w2 * y0) / wd) * s)
    eta_new = eta_eq + y_new
    etadd_new = u_acc - 2.0 * zeta * omega * v_new - w2 * eta_new
    return float(eta_new), float(v_new), float(etadd_new)


def step_flex_body_3d(
    *,
    state: FlexBodyState,
    dt_s: float,
    modal_props: FlexModalProperties,
    cg_station_m: float,
    thrust_axial_n: float,
    tvc_lateral_force_body_xy_n: tuple[float, float],
    fin_force_body_xy_n: tuple[float, float],
    aero_force_body_xy_n: tuple[float, float],
    rcs_torque_body_xy_nm: tuple[float, float],
    cp_station_m: float,
    vehicle: DM100VehicleConfig,
    engine_out_torque_body_xy_nm: tuple[float, float] = (0.0, 0.0),
    buffet_torque_body_xy_nm: tuple[float, float] = (0.0, 0.0),
    settings: FlexBodySettings = FlexBodySettings(),
) -> list[FlexStepSample]:
    """Advance flex-body ODE over `dt_s` using exact internal `dt_flex_s = 0.002 s` sub-steps.

    Returns the list of 500 Hz `FlexStepSample` records (length 5 for dt_s=0.01 s, 1 for 0.002 s)
    so `sensors_v2.py` can run its 18 Hz 2nd-order Butterworth anti-aliasing filter at true 500 Hz.
    """
    n_sub = max(1, int(round(dt_s / max(settings.dt_flex_s, 1e-4))))
    h = dt_s / n_sub
    length_m = modal_props.length_m
    g = vehicle.geometry
    z_noz = g.nozzle_station_m
    z_fin = vehicle.fins.station_m
    z_rcs = g.rcs_station_m
    z_imu = min(settings.imu_station_m, length_m)
    z_cp = max(0.0, min(length_m, float(cp_station_m)))

    # Equivalent lateral RCS force injecting pure couple at z_rcs (CDR §2.1)
    rcs_arm = max(z_rcs - cg_station_m, 2.0)
    f_rcs_x = float(rcs_torque_body_xy_nm[1]) / rcs_arm
    f_rcs_y = -float(rcs_torque_body_xy_nm[0]) / rcs_arm

    # Precompute mode shapes and slopes at excitation, nozzle, IMU, tip, and mid-span stations
    phi_noz = [0.0, 0.0]
    sig_noz = [0.0, 0.0]
    phi_imu = [0.0, 0.0]
    sig_imu = [0.0, 0.0]
    phi_tip = [0.0, 0.0]
    kap_mid = [0.0, 0.0]
    u_x = [0.0, 0.0]
    u_y = [0.0, 0.0]

    for k in range(2):
        p_noz, s_noz, _ = free_free_mode_eval(z_noz, length_m, k)
        p_fin, _, _ = free_free_mode_eval(z_fin, length_m, k)
        p_rcs, _, _ = free_free_mode_eval(z_rcs, length_m, k)
        p_cp, s_cp, _ = free_free_mode_eval(z_cp, length_m, k)
        p_imu, s_imu, _ = free_free_mode_eval(z_imu, length_m, k)
        p_tip, _, _ = free_free_mode_eval(length_m, length_m, k)
        # Peak curvature occurs near mid-span (0.5*L for mode 1, 0.31*L for mode 2)
        z_peak_k = 0.50 * length_m if k == 0 else 0.31 * length_m
        _, _, k_mid = free_free_mode_eval(z_peak_k, length_m, k)

        phi_noz[k] = p_noz
        sig_noz[k] = s_noz
        phi_imu[k] = p_imu
        sig_imu[k] = s_imu
        phi_tip[k] = p_tip
        kap_mid[k] = abs(k_mid)

        mk = modal_props.gen_mass_kg[k]
        # Commanded TVC lateral force ONLY (no elastic follower force on RHS -> zero Beck flutter)
        qx = (
            p_noz * float(tvc_lateral_force_body_xy_n[0])
            + p_fin * float(fin_force_body_xy_n[0])
            + p_rcs * f_rcs_x
            + p_cp * float(aero_force_body_xy_n[0])
            + s_noz * float(engine_out_torque_body_xy_nm[1])
            + s_cp * float(buffet_torque_body_xy_nm[1])
        )
        qy = (
            p_noz * float(tvc_lateral_force_body_xy_n[1])
            + p_fin * float(fin_force_body_xy_n[1])
            + p_rcs * f_rcs_y
            + p_cp * float(aero_force_body_xy_n[1])
            - s_noz * float(engine_out_torque_body_xy_nm[0])
            - s_cp * float(buffet_torque_body_xy_nm[0])
        )
        u_x[k] = qx / mk
        u_y[k] = qy / mk

    om = (2.0 * pi * modal_props.freq_hz[0], 2.0 * pi * modal_props.freq_hz[1])
    ze = modal_props.zeta
    lever_noz = cg_station_m - z_noz
    t_ax = max(0.0, float(thrust_axial_n))

    eta_x = [state.eta_x[0], state.eta_x[1]]
    etad_x = [state.etad_x[0], state.etad_x[1]]
    eta_y = [state.eta_y[0], state.eta_y[1]]
    etad_y = [state.etad_y[0], state.etad_y[1]]

    samples: list[FlexStepSample] = []
    for _ in range(n_sub):
        etadd_x = [0.0, 0.0]
        etadd_y = [0.0, 0.0]
        for k in range(2):
            eta_x[k], etad_x[k], etadd_x[k] = _exact_oscillator_step(
                eta_x[k], etad_x[k], om[k], ze[k], u_x[k], h
            )
            eta_y[k], etad_y[k], etadd_y[k] = _exact_oscillator_step(
                eta_y[k], etad_y[k], om[k], ze[k], u_y[k], h
            )

        # Follower-force elastic nozzle tilt + nozzle lateral offset moment (NASA SP-8003)
        dx_noz = phi_noz[0] * eta_x[0] + phi_noz[1] * eta_x[1]
        dy_noz = phi_noz[0] * eta_y[0] + phi_noz[1] * eta_y[1]
        dgam_x = sig_noz[0] * eta_x[0] + sig_noz[1] * eta_x[1]
        dgam_y = sig_noz[0] * eta_y[0] + sig_noz[1] * eta_y[1]

        df_x = t_ax * dgam_x
        df_y = t_ax * dgam_y
        # r_noz x F_tvc - r_noz0 x F_tvc0 (right-handed body frame):
        dtau_x = dy_noz * t_ax + lever_noz * df_y
        dtau_y = -lever_noz * df_x - dx_noz * t_ax

        # IMU contamination at z_imu (exact zero roll coupling: d_om_z == 0)
        d_om_x = -(sig_imu[0] * etad_y[0] + sig_imu[1] * etad_y[1])
        d_om_y = +(sig_imu[0] * etad_x[0] + sig_imu[1] * etad_x[1])
        d_acc_x = phi_imu[0] * etadd_x[0] + phi_imu[1] * etadd_x[1]
        d_acc_y = phi_imu[0] * etadd_y[0] + phi_imu[1] * etadd_y[1]

        # Tip deflection & worst elastic bending moment along barrel
        tip_x = phi_tip[0] * eta_x[0] + phi_tip[1] * eta_x[1]
        tip_y = phi_tip[0] * eta_y[0] + phi_tip[1] * eta_y[1]
        tip_norm = hypot(tip_x, tip_y)

        # Conservative modal curvature bound for Criterion #20 bending moment
        curv_x = kap_mid[0] * abs(eta_x[0]) + kap_mid[1] * abs(eta_x[1])
        curv_y = kap_mid[0] * abs(eta_y[0]) + kap_mid[1] * abs(eta_y[1])
        m_bend_max = modal_props.ei_eff_nm2 * hypot(curv_x, curv_y)

        cur_state = FlexBodyState(
            eta_x=(eta_x[0], eta_x[1]),
            etad_x=(etad_x[0], etad_x[1]),
            eta_y=(eta_y[0], eta_y[1]),
            etad_y=(etad_y[0], etad_y[1]),
        )
        samples.append(
            FlexStepSample(
                state=cur_state,
                rigid_torque_delta_body_nm=(float(dtau_x), float(dtau_y), 0.0),
                rigid_force_delta_body_n=(float(df_x), float(df_y), 0.0),
                imu_omega_delta_rad_s=(float(d_om_x), float(d_om_y), 0.0),
                imu_accel_delta_m_s2=(float(d_acc_x), float(d_acc_y), 0.0),
                tip_deflection_xy_m=(float(tip_x), float(tip_y)),
                tip_deflection_norm_m=float(tip_norm),
                max_bending_moment_nm=float(m_bend_max),
                modal_freq_hz=modal_props.freq_hz,
            )
        )
    return samples
