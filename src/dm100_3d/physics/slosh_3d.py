"""2-DOF spherical-pendulum propellant slosh model (reduces identically to DM100 2D in XZ)."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt, tanh

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
    theta = min(r_arc / l_p, 0.48 * pi)
    ux = state.qx_m / r_arc
    uy = state.qy_m / r_arc
    st = sin(theta)
    ct = cos(theta)
    u = np.array([l_p * st * ux, l_p * st * uy, l_p * (1.0 - ct)], dtype=np.float64)
    # Exact tangents along qx and qy (reduces to [cos(th), 0, sin(th)] when qy == 0)
    sinc = st / theta
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
