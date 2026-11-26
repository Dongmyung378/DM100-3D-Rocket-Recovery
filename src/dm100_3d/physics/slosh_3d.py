"""2-DOF spherical-pendulum propellant slosh model (reduces identically to DM100 2D in XZ)."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt, tanh

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dm100_3d.config import DualTankGeometry, dual_tank_propellant_levels


@dataclass(frozen=True, slots=True)
class Slosh3DSettings:
    modal_mass_fraction: float = 0.03
    tank_radius_m: float = 1.5
    tank_length_m: float = 13.0
    pendulum_length_m: float = 0.8
    liquid_density_kg_m3: float = 826.0
    surface_tension_n_m: float = 0.025
    damping_ratio: float = 0.12
    viscous_decay_rate_s_inv: float = 0.08
    maximum_mode_force_n: float = 100_000.0


@dataclass(frozen=True, slots=True)
class Slosh3DState:
    qx_m: float = 0.0
    qy_m: float = 0.0
    qxd_m_s: float = 0.0
    qyd_m_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Slosh3DResponse:
    slosh_torque_body_nm: tuple[float, float, float]
    qdd_m_s2: tuple[float, float]
    modal_mass_kg: float
    constraint_force_n: float
    valid: bool


def slosh_displacement_3d(
    state: Slosh3DState,
    settings: Slosh3DSettings = Slosh3DSettings(),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (u_body_m, du_dqx, du_dqy) for spherical pendulum arc-length coordinates (qx, qy)."""
    l_p = settings.pendulum_length_m
    r_arc = sqrt(state.qx_m * state.qx_m + state.qy_m * state.qy_m)
    if r_arc < 1e-12:
        return (
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
    theta = r_arc / l_p
    ux = state.qx_m / r_arc
    uy = state.qy_m / r_arc
    st = sin(theta)
    ct = cos(theta)
    u = np.array([l_p * st * ux, l_p * st * uy, l_p * (1.0 - ct)], dtype=np.float64)
    # Exact tangents along qx and qy (reduces identically to 2D [cos(th), 0, sin(th)]
    # for all arc lengths, without artificial 86 deg zero-g integrator windup!)
    theta_w = ((theta + pi) % (2.0 * pi)) - pi
    sinc = 1.0 if abs(theta_w) < 1e-9 else sin(theta_w) / theta_w
    du_dqx = np.array(
        [ct * ux * ux + sinc * uy * uy, (ct - sinc) * ux * uy, st * ux], dtype=np.float64
    )
    du_dqy = np.array(
        [(ct - sinc) * ux * uy, ct * uy * uy + sinc * ux * ux, st * uy], dtype=np.float64
    )
    return u, du_dqx, du_dqy


def slosh_body_offset_3d(
    total_mass_kg: float,
    main_propellant_kg: float,
    state: Slosh3DState,
    settings: Slosh3DSettings = Slosh3DSettings(),
) -> tuple[float, float, float]:
    """Body-frame COM offset (-eta * u_slosh) matching DM100 2D `body_frame_offset`."""
    if total_mass_kg <= 0.0 or main_propellant_kg <= 0.0:
        return (0.0, 0.0, 0.0)
    eta = settings.modal_mass_fraction * max(main_propellant_kg, 0.0) / total_mass_kg
    u, _, _ = slosh_displacement_3d(state, settings)
    return (float(-eta * u[0]), float(-eta * u[1]), float(-eta * u[2]))


def slosh_3d_response(
    *,
    total_mass_kg: float,
    rigid_inertia_diag_kg_m2: tuple[float, float, float],
    main_propellant_kg: float,
    station_offset_m: float,
    proper_accel_body_m_s2: ArrayLike,
    external_torque_body_nm: ArrayLike,
    omega_body_rad_s: ArrayLike,
    state: Slosh3DState,
    settings: Slosh3DSettings = Slosh3DSettings(),
) -> Slosh3DResponse:
    """Evaluate 3D coupled spherical-pendulum slosh acceleration & reaction torque on booster.

    Returns the equivalent `slosh_torque_body_nm` to add to MuJoCo `xfrc_applied` so
    MuJoCo's rigid-body rotational acceleration equals the exact coupled pendulum-booster
    angular acceleration (matching DM100 2D `slosh_response` identically in XZ plane).
    """
    m = settings.modal_mass_fraction * max(main_propellant_kg, 0.0)
    if m <= 1e-9 or total_mass_kg <= 0.0:
        return Slosh3DResponse(
            slosh_torque_body_nm=(0.0, 0.0, 0.0),
            qdd_m_s2=(0.0, 0.0),
            modal_mass_kg=0.0,
            constraint_force_n=0.0,
            valid=True,
        )

    a_b = np.asarray(proper_accel_body_m_s2, dtype=np.float64).reshape(3)
    tau_ext = np.asarray(external_torque_body_nm, dtype=np.float64).reshape(3)
    om = np.asarray(omega_body_rad_s, dtype=np.float64).reshape(3)
    ixx, iyy, _ = rigid_inertia_diag_kg_m2

    u, tx, ty = slosh_displacement_3d(state, settings)
    mu = m * (1.0 - m / total_mass_kg)
    l_p = settings.pendulum_length_m

    depth = main_propellant_kg / (settings.liquid_density_kg_m3 * pi * settings.tank_radius_m**2)
    k_wave = 1.841183781 / settings.tank_radius_m
    omega_cap_sq = (
        (settings.surface_tension_n_m / settings.liquid_density_kg_m3)
        * (k_wave**3)
        * tanh(k_wave * depth)
    )
    axial = float(a_b[2])
    freq_sq = axial / l_p + omega_cap_sq
    damping = m * (
        2.0 * settings.damping_ratio * sqrt(max(0.0, freq_sq)) + settings.viscous_decay_rate_s_inv
    )

    # Couple X-slosh with pitch (omega_y, tau_y) and Y-slosh with yaw/roll-tilt (omega_x, tau_x)
    # Using exact 2D reduced-mass metric per orthogonal plane (exact 2D equivalence in XZ)
    def _solve_axis(
        inertia_ax: float,
        u_lat: float,
        u_z: float,
        t_lat: float,
        t_z: float,
        lat_acc: float,
        tau_ax: float,
        om_ax: float,
        qd_ax: float,
        sign_couple: float,
    ) -> tuple[float, float, float]:
        eff_i = inertia_ax + 2.0 * m * station_offset_m * u_z + mu * (u_lat * u_lat + u_z * u_z)
        b_c = sign_couple * (m * station_offset_m * t_lat + mu * (u_z * t_lat - u_lat * t_z))
        det = eff_i * mu - b_c * b_c
        if det <= 1e-6 or eff_i <= 1e-6:
            return 0.0, 0.0, 0.0
        ai = 2.0 * (m * station_offset_m + mu * l_p) * t_z
        bi = -sign_couple * (m * station_offset_m / l_p + mu) * t_z
        rot_rhs = tau_ax - ai * qd_ax * om_ax - bi * qd_ax * qd_ax
        mod_rhs = (
            0.5 * ai * om_ax * om_ax
            - m * (lat_acc * t_lat + axial * t_z)
            - m * l_p * omega_cap_sq * t_z
            - damping * qd_ax
        )
        alpha = (mu * rot_rhs - b_c * mod_rhs) / det
        qdd = (eff_i * mod_rhs - b_c * rot_rhs) / det
        # Equivalent reaction torque so rigid MuJoCo body experiences `alpha`
        tau_slosh = inertia_ax * alpha - tau_ax
        return tau_slosh, qdd, alpha

    # XZ plane: +q_x couples positively with +omega_y (clockwise pitch in ENU)
    tau_y_slosh, qdd_x, alpha_y = _solve_axis(
        iyy,
        float(u[0]),
        float(u[2]),
        float(tx[0]),
        float(tx[2]),
        float(a_b[0]),
        float(tau_ext[1]),
        float(om[1]),
        state.qxd_m_s,
        1.0,
    )
    # YZ plane: +q_y couples negatively with +omega_x (since r=[0,0,z] x [0,Fy,0] = [-z*Fy, 0, 0])
    tau_x_slosh, qdd_y, alpha_x = _solve_axis(
        ixx,
        float(u[1]),
        float(u[2]),
        float(ty[1]),
        float(ty[2]),
        float(a_b[1]),
        float(tau_ext[0]),
        float(om[0]),
        state.qyd_m_s,
        -1.0,
    )

    # Approximate constraint force magnitude for envelope monitoring
    a_mode = np.array(
        [
            float(a_b[0]) + alpha_y * station_offset_m + qdd_x,
            float(a_b[1]) - alpha_x * station_offset_m + qdd_y,
            axial,
        ],
        dtype=np.float64,
    )
    c_force = m * float(np.linalg.norm(a_mode))
    valid = (
        all(isfinite(v) for v in (tau_x_slosh, tau_y_slosh, qdd_x, qdd_y, c_force))
        and sqrt(float(u[0]) ** 2 + float(u[1]) ** 2) <= settings.tank_radius_m
        and -1e-12 <= float(u[2]) <= settings.tank_length_m
        and c_force <= settings.maximum_mode_force_n
    )
    return Slosh3DResponse(
        slosh_torque_body_nm=(float(tau_x_slosh), float(tau_y_slosh), 0.0),
        qdd_m_s2=(float(qdd_x), float(qdd_y)),
        modal_mass_kg=float(m),
        constraint_force_n=float(c_force),
        valid=bool(valid),
    )


# ---------------------------------------------------------------------------
# DM100 3D v2 (M1): Common-Bulkhead LOX + CH4 3x3 Coupled Dual-Tank Slosh
# ---------------------------------------------------------------------------

_XI1_CYL = 1.841183781


@dataclass(frozen=True, slots=True)
class DualTankSloshSettings:
    """Dual-tank (upper LOX + lower CH4) multi-ring baffle slosh settings (CDR Rev 3.1)."""

    geometry: DualTankGeometry = DualTankGeometry()
    damping_scale: float = 1.0
    surface_tension_lox_n_m: float = 0.0132
    surface_tension_ch4_n_m: float = 0.0130
    viscous_decay_rate_s_inv: float = 0.05
    zero_g_wall_drag_s_inv: float = 0.65
    wall_confinement_angle_rad: float = 0.25
    wall_confinement_radius_frac: float = 0.38
    wall_stiffness_s_inv2: float = 28.0
    wall_damping_s_inv: float = 6.0
    wall_ramp_width_m: float = 0.20
    maximum_mode_force_n: float = 300_000.0
    min_pendulum_length_m: float = 0.40
    max_pendulum_length_m: float = 8.00


@dataclass(frozen=True, slots=True)
class DualTankSloshState:
    lox: Slosh3DState = Slosh3DState()
    ch4: Slosh3DState = Slosh3DState()


@dataclass(frozen=True, slots=True)
class DualTankSloshResponse:
    slosh_torque_body_nm: tuple[float, float, float]
    lox_qdd_m_s2: tuple[float, float]
    ch4_qdd_m_s2: tuple[float, float]
    lox_modal_mass_kg: float
    ch4_modal_mass_kg: float
    lox_freq_hz: float
    ch4_freq_hz: float
    constraint_force_n: float
    valid: bool


def regularized_ring_baffle_damping(
    theta_rad: float,
    *,
    zeta0: float = 0.080,
    kb: float = 0.08,
    eps_rad: float = 1e-3,
    damping_scale: float = 1.0,
) -> float:
    """Smoothly regularized nonlinear ring-baffle damping (C-infinity at theta = 0)."""
    th2 = float(theta_rad) * float(theta_rad)
    e2 = float(eps_rad) * float(eps_rad)
    nl = (th2 + e2) ** 0.25 - sqrt(float(eps_rad))
    return max(0.01, float(damping_scale) * (float(zeta0) + float(kb) * nl))


def _tank_modal_params(
    liquid_mass_kg: float,
    liquid_depth_m: float,
    tank_bottom_m: float,
    tank_radius_m: float,
    settings: DualTankSloshSettings,
) -> tuple[float, float, float]:
    """Return Dodge / NASA SP-106 `(modal_mass_kg, pendulum_length_m, action_station_m)`."""
    if liquid_mass_kg <= 1e-6 or liquid_depth_m <= 1e-4:
        return 0.0, settings.min_pendulum_length_m, tank_bottom_m
    r = tank_radius_m
    h = liquid_depth_m
    xh = _XI1_CYL * h / r
    th = tanh(xh)
    # NASA SP-106 fundamental lateral cylindrical slosh mode
    frac = (2.0 * r / (_XI1_CYL * (_XI1_CYL * _XI1_CYL - 1.0) * h)) * th
    frac = max(0.0, min(0.85, frac))
    m_mode = liquid_mass_kg * frac
    l_p = r / max(_XI1_CYL * th, 1e-6)
    l_p = max(settings.min_pendulum_length_m, min(settings.max_pendulum_length_m, l_p))
    # Modal lateral force station relative to hull base (matches NASA SP-106 deep/shallow limits)
    z_act = tank_bottom_m + h - (r / _XI1_CYL) * th
    return m_mode, l_p, z_act


def _pendulum_kinematics(
    state: Slosh3DState, l_p: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    r_arc = sqrt(state.qx_m * state.qx_m + state.qy_m * state.qy_m)
    if r_arc < 1e-12:
        return (
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            0.0,
        )
    theta = r_arc / l_p
    ux = state.qx_m / r_arc
    uy = state.qy_m / r_arc
    st = sin(theta)
    ct = cos(theta)
    u = np.array([l_p * st * ux, l_p * st * uy, l_p * (1.0 - ct)], dtype=np.float64)
    theta_w = ((theta + pi) % (2.0 * pi)) - pi
    sinc = 1.0 if abs(theta_w) < 1e-9 else sin(theta_w) / theta_w
    du_dqx = np.array(
        [ct * ux * ux + sinc * uy * uy, (ct - sinc) * ux * uy, st * ux], dtype=np.float64
    )
    du_dqy = np.array(
        [(ct - sinc) * ux * uy, ct * uy * uy + sinc * ux * ux, st * uy], dtype=np.float64
    )
    return u, du_dqx, du_dqy, theta


def dual_slosh_body_offset_3d(
    total_mass_kg: float,
    main_propellant_kg: float,
    state: DualTankSloshState,
    settings: DualTankSloshSettings = DualTankSloshSettings(),
) -> tuple[float, float, float]:
    """Combined body-frame COM shift `-(m_lox * u_lox + m_ch4 * u_ch4) / M_tot`."""
    if total_mass_kg <= 0.0 or main_propellant_kg <= 0.0:
        return (0.0, 0.0, 0.0)
    dtg = settings.geometry
    m_lox_liq, h_lox, _, m_ch4_liq, h_ch4, _ = dual_tank_propellant_levels(main_propellant_kg, dtg)
    m1, lp1, _ = _tank_modal_params(
        m_lox_liq, h_lox, dtg.lox_bottom_m, dtg.tank_inner_radius_m, settings
    )
    m2, lp2, _ = _tank_modal_params(
        m_ch4_liq, h_ch4, dtg.ch4_bottom_m, dtg.tank_inner_radius_m, settings
    )
    u1, _, _, _ = _pendulum_kinematics(state.lox, lp1)
    u2, _, _, _ = _pendulum_kinematics(state.ch4, lp2)
    off = -(m1 * u1 + m2 * u2) / total_mass_kg
    return (float(off[0]), float(off[1]), float(off[2]))


def dual_slosh_3d_response(
    *,
    total_mass_kg: float,
    rigid_inertia_diag_kg_m2: tuple[float, float, float],
    main_propellant_kg: float,
    cg_station_m: float,
    proper_accel_body_m_s2: ArrayLike,
    external_torque_body_nm: ArrayLike,
    omega_body_rad_s: ArrayLike,
    state: DualTankSloshState,
    settings: DualTankSloshSettings = DualTankSloshSettings(),
) -> DualTankSloshResponse:
    """Solve exact 3x3 symmetric coupled dual-tank (LOX + CH4) slosh dynamics per plane."""
    if main_propellant_kg <= 1e-6 or total_mass_kg <= 0.0:
        return DualTankSloshResponse(
            slosh_torque_body_nm=(0.0, 0.0, 0.0),
            lox_qdd_m_s2=(0.0, 0.0),
            ch4_qdd_m_s2=(0.0, 0.0),
            lox_modal_mass_kg=0.0,
            ch4_modal_mass_kg=0.0,
            lox_freq_hz=0.0,
            ch4_freq_hz=0.0,
            constraint_force_n=0.0,
            valid=True,
        )

    dtg = settings.geometry
    m_lox_liq, h_lox, _, m_ch4_liq, h_ch4, _ = dual_tank_propellant_levels(main_propellant_kg, dtg)
    m1, lp1, z1 = _tank_modal_params(
        m_lox_liq, h_lox, dtg.lox_bottom_m, dtg.tank_inner_radius_m, settings
    )
    m2, lp2, z2 = _tank_modal_params(
        m_ch4_liq, h_ch4, dtg.ch4_bottom_m, dtg.tank_inner_radius_m, settings
    )
    if m1 <= 1e-9 and m2 <= 1e-9:
        return DualTankSloshResponse(
            slosh_torque_body_nm=(0.0, 0.0, 0.0),
            lox_qdd_m_s2=(0.0, 0.0),
            ch4_qdd_m_s2=(0.0, 0.0),
            lox_modal_mass_kg=0.0,
            ch4_modal_mass_kg=0.0,
            lox_freq_hz=0.0,
            ch4_freq_hz=0.0,
            constraint_force_n=0.0,
            valid=True,
        )

    a_b = np.asarray(proper_accel_body_m_s2, dtype=np.float64).reshape(3)
    tau_ext = np.asarray(external_torque_body_nm, dtype=np.float64).reshape(3)
    om = np.asarray(omega_body_rad_s, dtype=np.float64).reshape(3)
    ixx, iyy, _ = rigid_inertia_diag_kg_m2
    axial = float(a_b[2])
    r_in = dtg.tank_inner_radius_m
    k_wave = _XI1_CYL / r_in

    u1, t1x, t1y, th1 = _pendulum_kinematics(state.lox, lp1)
    u2, t2x, t2y, th2 = _pendulum_kinematics(state.ch4, lp2)

    d1 = z1 - float(cg_station_m)
    d2 = z2 - float(cg_station_m)
    mu1 = m1 * (1.0 - m1 / total_mass_kg)
    mu2 = m2 * (1.0 - m2 / total_mass_kg)

    om_cap1_sq = (
        (settings.surface_tension_lox_n_m / dtg.lox_density_kg_m3)
        * (k_wave**3)
        * tanh(k_wave * max(h_lox, 1e-4))
    )
    om_cap2_sq = (
        (settings.surface_tension_ch4_n_m / dtg.ch4_density_kg_m3)
        * (k_wave**3)
        * tanh(k_wave * max(h_ch4, 1e-4))
    )
    freq1_sq = axial / lp1 + om_cap1_sq
    freq2_sq = axial / lp2 + om_cap2_sq

    zeta1 = regularized_ring_baffle_damping(
        th1,
        zeta0=dtg.ring_baffle_zeta0,
        kb=dtg.ring_baffle_kb,
        eps_rad=dtg.ring_baffle_eps_rad,
        damping_scale=settings.damping_scale,
    )
    zeta2 = regularized_ring_baffle_damping(
        th2,
        zeta0=dtg.ring_baffle_zeta0,
        kb=dtg.ring_baffle_kb,
        eps_rad=dtg.ring_baffle_eps_rad,
        damping_scale=settings.damping_scale,
    )
    # Smooth microgravity wall/baffle boundary drag when axial settling acceleration < 1.0 m/s^2
    # (zero effect during powered ascent/entry/landing where axial >> 1 m/s^2).
    u_zero_g = max(0.0, min(1.0, 1.0 - max(0.0, axial) / 1.0))
    visc_eff = settings.viscous_decay_rate_s_inv + settings.zero_g_wall_drag_s_inv * (
        u_zero_g * u_zero_g * (3.0 - 2.0 * u_zero_g)
    )
    damp1 = m1 * (2.0 * zeta1 * sqrt(max(0.0, freq1_sq)) + visc_eff)
    damp2 = m2 * (2.0 * zeta2 * sqrt(max(0.0, freq2_sq)) + visc_eff)

    # Physical cylindrical tank-wall / ring-baffle wave-breaking confinement:
    # Strictly ZERO inside small linear angles (preserving exact linearized Bode margins),
    # and smoothly arrests liquid inside the 1.50 m tank wall during large zero-g flip maneuvers.
    def _wall_confinement_force(st_j: Slosh3DState, m_j: float, lp_j: float) -> tuple[float, float]:
        r_arc = sqrt(st_j.qx_m * st_j.qx_m + st_j.qy_m * st_j.qy_m)
        q_lim = min(
            settings.wall_confinement_angle_rad * lp_j,
            settings.wall_confinement_radius_frac * r_in,
        )
        if m_j <= 1e-9 or r_arc <= q_lim or r_arc <= 1e-12:
            return 0.0, 0.0
        exc = r_arc - q_lim
        nx = st_j.qx_m / r_arc
        ny = st_j.qy_m / r_arc
        v_rad = st_j.qxd_m_s * nx + st_j.qyd_m_s * ny
        # C1-smooth quadratic spring + smoothly ramped outward-velocity baffle dissipation
        # (zero force AND zero damping jump at r_arc == q_lim!)
        ramp = min(1.0, exc / max(settings.wall_ramp_width_m, 0.05))
        f_spring = m_j * settings.wall_stiffness_s_inv2 * exc * (1.0 + exc / max(q_lim, 0.10))
        f_damp = m_j * settings.wall_damping_s_inv * ramp * max(0.0, v_rad)
        f_mag = f_spring + f_damp
        return -f_mag * nx, -f_mag * ny

    f_wall1_x, f_wall1_y = _wall_confinement_force(state.lox, m1, lp1)
    f_wall2_x, f_wall2_y = _wall_confinement_force(state.ch4, m2, lp2)

    def _solve_plane_3x3(
        inertia_ax: float,
        u1_lat: float,
        u1_z: float,
        t1_lat: float,
        t1_z: float,
        u2_lat: float,
        u2_z: float,
        t2_lat: float,
        t2_z: float,
        lat_acc: float,
        tau_ax: float,
        om_ax: float,
        qd1: float,
        qd2: float,
        f_wall1: float,
        f_wall2: float,
        sign_couple: float,
    ) -> tuple[float, float, float, float]:
        eff_i = (
            inertia_ax
            + 2.0 * m1 * d1 * u1_z
            + mu1 * (u1_lat * u1_lat + u1_z * u1_z)
            + 2.0 * m2 * d2 * u2_z
            + mu2 * (u2_lat * u2_lat + u2_z * u2_z)
        )
        b1 = sign_couple * (m1 * d1 * t1_lat + mu1 * (u1_z * t1_lat - u1_lat * t1_z))
        b2 = sign_couple * (m2 * d2 * t2_lat + mu2 * (u2_z * t2_lat - u2_lat * t2_z))
        c12 = -(m1 * m2 / total_mass_kg) * (t1_lat * t2_lat + t1_z * t2_z)

        ai1 = 2.0 * (m1 * d1 + mu1 * lp1) * t1_z
        bi1 = -sign_couple * (m1 * d1 / lp1 + mu1) * t1_z
        ai2 = 2.0 * (m2 * d2 + mu2 * lp2) * t2_z
        bi2 = -sign_couple * (m2 * d2 / lp2 + mu2) * t2_z

        rot_rhs = tau_ax - ai1 * qd1 * om_ax - bi1 * qd1 * qd1 - ai2 * qd2 * om_ax - bi2 * qd2 * qd2
        mod1_rhs = (
            0.5 * ai1 * om_ax * om_ax
            - m1 * (lat_acc * t1_lat + axial * t1_z)
            - m1 * lp1 * om_cap1_sq * t1_z
            - damp1 * qd1
            + f_wall1
        )
        mod2_rhs = (
            0.5 * ai2 * om_ax * om_ax
            - m2 * (lat_acc * t2_lat + axial * t2_z)
            - m2 * lp2 * om_cap2_sq * t2_z
            - damp2 * qd2
            + f_wall2
        )

        # Solve exact active sub-system (3x3 when both tanks active, 2x2 if one empty)
        if m1 > 1e-9 and m2 > 1e-9:
            mat = np.array(
                [
                    [eff_i, b1, b2],
                    [b1, mu1, c12],
                    [b2, c12, mu2],
                ],
                dtype=np.float64,
            )
            rhs = np.array([rot_rhs, mod1_rhs, mod2_rhs], dtype=np.float64)
            try:
                sol = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                return 0.0, 0.0, 0.0, 0.0
            alpha, qdd1, qdd2 = float(sol[0]), float(sol[1]), float(sol[2])
        elif m1 > 1e-9:
            det = eff_i * mu1 - b1 * b1
            if det <= 1e-6:
                return 0.0, 0.0, 0.0, 0.0
            alpha = (mu1 * rot_rhs - b1 * mod1_rhs) / det
            qdd1 = (eff_i * mod1_rhs - b1 * rot_rhs) / det
            qdd2 = 0.0
        else:
            det = eff_i * mu2 - b2 * b2
            if det <= 1e-6:
                return 0.0, 0.0, 0.0, 0.0
            alpha = (mu2 * rot_rhs - b2 * mod2_rhs) / det
            qdd1 = 0.0
            qdd2 = (eff_i * mod2_rhs - b2 * rot_rhs) / det

        tau_slosh = inertia_ax * alpha - tau_ax
        return tau_slosh, qdd1, qdd2, alpha

    # XZ plane: +q_x couples positively with +omega_y
    tau_y_sl, qdd1_x, qdd2_x, alpha_y = _solve_plane_3x3(
        iyy,
        float(u1[0]),
        float(u1[2]),
        float(t1x[0]),
        float(t1x[2]),
        float(u2[0]),
        float(u2[2]),
        float(t2x[0]),
        float(t2x[2]),
        float(a_b[0]),
        float(tau_ext[1]),
        float(om[1]),
        state.lox.qxd_m_s,
        state.ch4.qxd_m_s,
        f_wall1_x,
        f_wall2_x,
        1.0,
    )
    # YZ plane: +q_y couples negatively with +omega_x
    tau_x_sl, qdd1_y, qdd2_y, alpha_x = _solve_plane_3x3(
        ixx,
        float(u1[1]),
        float(u1[2]),
        float(t1y[1]),
        float(t1y[2]),
        float(u2[1]),
        float(u2[2]),
        float(t2y[1]),
        float(t2y[2]),
        float(a_b[1]),
        float(tau_ext[0]),
        float(om[0]),
        state.lox.qyd_m_s,
        state.ch4.qyd_m_s,
        f_wall1_y,
        f_wall2_y,
        -1.0,
    )

    a_mode1 = np.array(
        [float(a_b[0]) + alpha_y * d1 + qdd1_x, float(a_b[1]) - alpha_x * d1 + qdd1_y, axial],
        dtype=np.float64,
    )
    a_mode2 = np.array(
        [float(a_b[0]) + alpha_y * d2 + qdd2_x, float(a_b[1]) - alpha_x * d2 + qdd2_y, axial],
        dtype=np.float64,
    )
    c_force = max(m1 * float(np.linalg.norm(a_mode1)), m2 * float(np.linalg.norm(a_mode2)))
    rad1 = sqrt(float(u1[0]) ** 2 + float(u1[1]) ** 2)
    rad2 = sqrt(float(u2[0]) ** 2 + float(u2[1]) ** 2)
    valid = (
        all(isfinite(v) for v in (tau_x_sl, tau_y_sl, qdd1_x, qdd1_y, qdd2_x, qdd2_y, c_force))
        and rad1 <= r_in + 1e-6
        and rad2 <= r_in + 1e-6
        and c_force <= settings.maximum_mode_force_n
    )
    f1_hz = sqrt(max(0.0, freq1_sq)) / (2.0 * pi) if m1 > 1e-9 else 0.0
    f2_hz = sqrt(max(0.0, freq2_sq)) / (2.0 * pi) if m2 > 1e-9 else 0.0

    return DualTankSloshResponse(
        slosh_torque_body_nm=(float(tau_x_sl), float(tau_y_sl), 0.0),
        lox_qdd_m_s2=(float(qdd1_x), float(qdd1_y)),
        ch4_qdd_m_s2=(float(qdd2_x), float(qdd2_y)),
        lox_modal_mass_kg=float(m1),
        ch4_modal_mass_kg=float(m2),
        lox_freq_hz=float(f1_hz),
        ch4_freq_hz=float(f2_hz),
        constraint_force_n=float(c_force),
        valid=bool(valid),
    )
