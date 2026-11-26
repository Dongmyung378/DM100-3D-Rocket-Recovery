"""Pure 3D external aerodynamics, 4-fin, 7-engine TVC, 3-axis RCS, and Coriolis forces.

Exact 2D Plane Equivalence guarantee:
  When `crosswind_y == 0`, `roll == 0`, `yaw == 0`, `gimbal_rad = (gamma_y, 0.0)`,
  `rcs_torque_nm = (0.0, tau_y, 0.0)`, and symmetric planar fin commands
  `fin_deflection_rad = (0.0, 0.0, delta_pitch, delta_pitch)`:
    - Body aero (Fx, Fz, Tau_y, q, mach, alpha_deg) matches DM100 2D `forces.py` identically.
    - 4-fin aero summed matches DM100 2D `compute_fin_aero` (Fx, Fz, Tau_y, hinge) identically.
    - Thrust, 2-axis TVC, and RCS match DM100 2D `compute_forces_and_torques` identically.
    - All out-of-plane components (Fy, Tau_x, Tau_z) are identically zero.
"""

from __future__ import annotations

from math import atan2, cos, degrees, exp, hypot, pi, radians, sin, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dm100_3d.config import (
    EARTH_RADIUS_M,
    G0,
    OMEGA_EARTH,
    R_AIR,
    SUTTON_GRAVES_EARTH_K,
    DM100VehicleConfig,
)
from dm100_3d.coords import quat_to_rotmat
from dm100_3d.interfaces import ForceTorqueSample3D
from dm100_3d.physics.atmosphere_3d import AtmosphereLUT, Wind3DLUT


def compute_stagnation_heat_flux_w_m2(
    rho_kg_m3: float,
    speed_m_s: float,
    nose_radius_m: float = 1.55,
) -> float:
    """Sutton-Graves convective stagnation heat flux [W/m^2] for Earth entry."""
    if rho_kg_m3 <= 0.0 or speed_m_s <= 0.0:
        return 0.0
    return SUTTON_GRAVES_EARTH_K * sqrt(rho_kg_m3 / max(nose_radius_m, 0.1)) * (speed_m_s**3)


def compute_engine_thrust_3d(
    throttle_total: float,
    pressure_pa: float,
    vehicle: DM100VehicleConfig,
    *,
    thrust_scale: float = 1.0,
) -> tuple[float, float]:
    """Compute scalar (thrust_n, demanded_main_flow_kg_s) matching DM100 2D identically."""
    if throttle_total <= 0.0:
        return 0.0, 0.0
    p = vehicle.propulsion
    thrust = (
        throttle_total
        * (p.sea_level_thrust_n + p.nozzle_exit_area_m2 * (101_325.0 - pressure_pa))
        * thrust_scale
    )
    flow = throttle_total * p.sea_level_thrust_n * thrust_scale / (p.sea_level_isp_s * G0)
    return thrust, flow


def compute_body_aero_3d(
    *,
    altitude_m: float,
    vel_enu_m_s: ArrayLike,
    rot_wb: NDArray[np.float64],
    omega_body_rad_s: ArrayLike,
    cg_station_m: float,
    atmo_lut: AtmosphereLUT,
    wind_lut: Wind3DLUT,
    time_s: float,
    vehicle: DM100VehicleConfig,
    drag_scale: float = 1.0,
    upper_attached: bool = False,
    body_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float, float, float]:
    """Compute 3D body aero (f_world_n, tau_body_nm, q_pa, heat_w_m2, mach, alpha_deg, phi_deg)."""
    t_k, _, rho = atmo_lut.at(altitude_m)
    wx, wy, wz = wind_lut.at(altitude_m, time_s)
    v_enu = np.asarray(vel_enu_m_s, dtype=np.float64).reshape(3)
    v_rel_w = np.array([v_enu[0] - wx, v_enu[1] - wy, v_enu[2] - wz], dtype=np.float64)
    speed = float(np.linalg.norm(v_rel_w))
    q = 0.5 * rho * speed * speed
    heat_flux = compute_stagnation_heat_flux_w_m2(rho, speed, vehicle.geometry.body_radius_m)
    mach = speed / sqrt(1.4 * R_AIR * t_k) if speed > 1e-9 else 0.0
    if speed <= 1e-9:
        z3 = np.zeros(3, dtype=np.float64)
        return z3, z3, 0.0, 0.0, 0.0, 0.0, 0.0

    v_rel_b = rot_wb.T @ v_rel_w
    vx_b, vy_b, vz_b = float(v_rel_b[0]), float(v_rel_b[1]), float(v_rel_b[2])
    v_perp = hypot(vx_b, vy_b)

    aero = vehicle.aerodynamics
    g = vehicle.geometry
    corr = 1.0 + aero.transonic_drag_rise * exp(-(((mach - 1.0) / 0.2) ** 2))
    r = g.body_radius_m
    length = g.body_length_m
    axial_area = pi * r * r
    normal_area = 2.0 * r * length

    ascending_flow = vz_b >= 0.0 or upper_attached
    axial_cd = aero.axial_cd_ascent if ascending_flow else aero.axial_cd_descent
    cp_m = aero.pressure_center_ascent_m if ascending_flow else aero.pressure_center_descent_m

    scale = -0.5 * rho * corr * drag_scale
    fz_b = scale * axial_cd * axial_area * abs(vz_b) * vz_b

    fx_b = 0.0
    fy_b = 0.0
    if v_perp > 1e-12:
        f_perp = scale * aero.normal_cd * normal_area * v_perp * v_perp
        if not upper_attached and aero.slender_lift_slope > 0.0:
            sin_a = v_perp / speed
            cos_a = abs(vz_b) / speed
            f_perp += scale * aero.slender_lift_slope * axial_area * speed * speed * sin_a * cos_a
        ux = vx_b / v_perp
        uy = vy_b / v_perp
        fx_b = f_perp * ux
        fy_b = f_perp * uy

    f_body = np.array([fx_b, fy_b, fz_b], dtype=np.float64)
    f_world = rot_wb @ f_body

    # CP moment about COM: r_cp x F_body with r_cp = (ox, oy, cp - cg + oz)
    ox, oy, oz = body_offset_body_m
    rx, ry, rz = ox, oy, (cp_m - cg_station_m) + oz
    tau_cp = np.array(
        [
            ry * fz_b - rz * fy_b,
            rz * fx_b - rx * fz_b,
            rx * fy_b - ry * fx_b,
        ],
        dtype=np.float64,
    )

    # 3-axis aerodynamic rotational damping
    om = np.asarray(omega_body_rad_s, dtype=np.float64).reshape(3)
    damp_coeff = 0.5 * rho * drag_scale * normal_area * length * length * aero.rotation_damping
    roll_coeff = 0.5 * rho * drag_scale * (2.0 * pi * r * length) * r * r * aero.roll_damping
    tau_damp = np.array(
        [
            -damp_coeff * (speed + length * abs(float(om[0]))) * float(om[0]),
            -damp_coeff * (speed + length * abs(float(om[1]))) * float(om[1]),
            -roll_coeff * (speed + r * abs(float(om[2]))) * float(om[2]),
        ],
        dtype=np.float64,
    )
    tau_body = tau_cp + tau_damp

    # Signed alpha matches 2D sign in XZ plane when vy_b == 0; total_alpha >= 0 in 3D
    v_ax_ref = max(vz_b, 1e-9) if ascending_flow else max(-vz_b, 1e-9)
    alpha_tot_deg = degrees(atan2(v_perp, v_ax_ref))
    if abs(vy_b) < 1e-12 and vx_b < 0.0:
        alpha_tot_deg = -alpha_tot_deg
    aero_roll_deg = degrees(atan2(vy_b, vx_b)) if v_perp > 1e-9 else 0.0

    return f_world, tau_body, q, heat_flux, mach, alpha_tot_deg, aero_roll_deg


def compute_fins_aero_3d(
    *,
    altitude_m: float,
    vel_enu_m_s: ArrayLike,
    rot_wb: NDArray[np.float64],
    omega_body_rad_s: ArrayLike,
    cg_station_m: float,
    fin_deflection_rad: tuple[float, float, float, float],
    fin_extension: float,
    atmo_lut: AtmosphereLUT,
    wind_lut: Wind3DLUT,
    time_s: float,
    vehicle: DM100VehicleConfig,
    upper_attached: bool = False,
    drag_scale: float = 1.0,
    body_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float, float, float]]:
    """Compute independent 4-grid-fin forces, body torques, and per-fin hinge moments.

    Fin indexing:
      0: +X fin (hinge +X) -> lift along +/-Y_b -> produces tau_x and +roll (tau_z)
      1: -X fin (hinge -X) -> lift along +/-Y_b -> produces tau_x and -roll (tau_z)
      2: +Y fin (hinge +Y) -> lift along +/-X_b -> produces tau_y (pitch) and -roll (tau_z)
      3: -Y fin (hinge -Y) -> lift along +/-X_b -> produces tau_y (pitch) and +roll (tau_z)

    Collective deflection (0, 0, delta, delta) in XZ plane matches 2D `compute_fin_aero` exactly.
    Differential deflection (+d_r, -d_r, -d_r, +d_r) yields pure roll torque about +z_b.
    """
    z3 = np.zeros(3, dtype=np.float64)
    if upper_attached or fin_extension <= 1e-9:
        return z3, z3, (0.0, 0.0, 0.0, 0.0)

    _, _, rho = atmo_lut.at(altitude_m)
    wx, wy, wz = wind_lut.at(altitude_m, time_s)
    v_enu = np.asarray(vel_enu_m_s, dtype=np.float64).reshape(3)
    v_rel_w = np.array([v_enu[0] - wx, v_enu[1] - wy, v_enu[2] - wz], dtype=np.float64)
    speed_cg = float(np.linalg.norm(v_rel_w))
    if speed_cg < 1e-6:
        return z3, z3, (0.0, 0.0, 0.0, 0.0)

    v_rel_b_cg = rot_wb.T @ v_rel_w
    om = np.asarray(omega_body_rad_s, dtype=np.float64).reshape(3)

    fins = vehicle.fins
    ox, oy, oz = body_offset_body_m
    arm_z = fins.station_m - cg_station_m + oz
    r_c = fins.radial_center_m
    area_fin = fins.per_fin_area_m2 * fin_extension
    max_a = radians(fins.control_max_alpha_deg)
    # 4 fins share the calibrated 2D total base drag (4 * 0.5 * 1.5 = 3.0 m^2 base drag area)
    per_fin_cd0 = 0.5 * fins.base_drag_coefficient

    # (mount_pos_body, lift_axis_body, radial_sign_for_roll)
    fin_specs = (
        (np.array([r_c + ox, oy, arm_z]), np.array([0.0, 1.0, 0.0])),  # fin 0: +X -> lift +Y
        (np.array([-r_c + ox, oy, arm_z]), np.array([0.0, 1.0, 0.0])),  # fin 1: -X -> lift +Y
        (np.array([ox, r_c + oy, arm_z]), np.array([1.0, 0.0, 0.0])),  # fin 2: +Y -> lift +X
        (np.array([ox, -r_c + oy, arm_z]), np.array([1.0, 0.0, 0.0])),  # fin 3: -Y -> lift +X
    )

    total_f_b = np.zeros(3, dtype=np.float64)
    total_tau_b = np.zeros(3, dtype=np.float64)
    hinges = [0.0, 0.0, 0.0, 0.0]

    for idx, (r_fin, lift_axis) in enumerate(fin_specs):
        # Local air velocity includes roll rate coupling (omega_z x r_fin); pitch/yaw handled
        # at CG consistent with 2D compute_fin_aero so pure XZ plane matches 2D exactly.
        v_local_b = v_rel_b_cg + np.array(
            [-float(om[2]) * float(r_fin[1]), float(om[2]) * float(r_fin[0]), 0.0],
            dtype=np.float64,
        )
        spd = float(np.linalg.norm(v_local_b))
        if spd < 1e-6:
            continue
        q_loc = 0.5 * rho * spd * spd
        v_lat = float(v_local_b @ lift_axis)
        v_ax = float(v_local_b[2])

        alpha_retro = atan2(v_lat, max(-v_ax, 1e-3)) if v_ax < 0.0 else 0.0
        alpha_eff = float(fin_deflection_rad[idx]) - alpha_retro
        if alpha_eff > max_a:
            alpha_eff = max_a
        elif alpha_eff < -max_a:
            alpha_eff = -max_a

        cl = fins.lift_slope_rad_inv * alpha_eff
        if cl > fins.max_lift_coefficient:
            cl = fins.max_lift_coefficient
        elif cl < -fins.max_lift_coefficient:
            cl = -fins.max_lift_coefficient

        lift = q_loc * area_fin * cl * drag_scale
        drag = q_loc * area_fin * (per_fin_cd0 + fins.induced_drag_factor * cl * cl) * drag_scale

        # Lift direction orthogonal to v_local_b in the (lift_axis, +z_b) plane:
        # For lift_axis = +x_b and v = (vx, 0, vz), unit normal = (-vz, 0, vx)/sqrt(vx^2+vz^2)
        in_plane_spd = hypot(v_lat, v_ax)
        if in_plane_spd > 1e-9:
            n_lat = -v_ax / in_plane_spd
            n_ax = v_lat / in_plane_spd
        else:
            n_lat, n_ax = 1.0, 0.0

        f_fin_b = lift * (n_lat * lift_axis + np.array([0.0, 0.0, n_ax])) - drag * (v_local_b / spd)
        total_f_b += f_fin_b
        total_tau_b += np.cross(r_fin, f_fin_b)
        hinges[idx] = abs(lift) * fins.hinge_arm_m

    total_f_world = rot_wb @ total_f_b
    return total_f_world, total_tau_b, (hinges[0], hinges[1], hinges[2], hinges[3])


def compute_coriolis_and_centrifugal_3d(
    *,
    altitude_m: float,
    vel_enu_m_s: ArrayLike,
    latitude_deg: float,
    coriolis_enabled: bool = True,
    planar_coriolis_only: bool = False,
) -> NDArray[np.float64]:
    """Return inertial correction acceleration [m/s^2] in ENU (Coriolis + spherical centrifugal)."""
    v = np.asarray(vel_enu_m_s, dtype=np.float64).reshape(3)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    r_orb = EARTH_RADIUS_M + max(float(altitude_m), 0.0)
    ax = 0.0
    ay = 0.0
    az = (vx * vx + vy * vy) / r_orb

    if coriolis_enabled:
        lat = radians(latitude_deg)
        om_y = OMEGA_EARTH * cos(lat)
        om_z = 0.0 if planar_coriolis_only else OMEGA_EARTH * sin(lat)
        # -2 * [0, om_y, om_z] x [vx, vy, vz]
        ax += -2.0 * (om_y * vz - om_z * vy)
        ay += -2.0 * (om_z * vx)
        az += 2.0 * (om_y * vx)

    return np.array([ax, ay, az], dtype=np.float64)


def compute_forces_and_torques_3d(
    *,
    altitude_m: float,
    vel_enu_m_s: ArrayLike,
    quat_wxyz: ArrayLike,
    omega_body_rad_s: ArrayLike,
    cg_station_m: float,
    throttle_actual: float,
    gimbal_rad: tuple[float, float],
    rcs_torque_body_nm: tuple[float, float, float],
    fin_deflection_rad: tuple[float, float, float, float],
    fin_extension: float,
    atmo_lut: AtmosphereLUT,
    wind_lut: Wind3DLUT,
    time_s: float,
    vehicle: DM100VehicleConfig,
    upper_attached: bool = False,
    thrust_scale: float = 1.0,
    drag_scale: float = 1.0,
    delivered_ratio: float = 1.0,
    body_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ForceTorqueSample3D:
    """Evaluate all pure 3D external forces [N] (ENU world) and torques [N*m] (body frame)."""
    rot_wb = quat_to_rotmat(quat_wxyz)
    _, pressure_pa, _ = atmo_lut.at(altitude_m)
    thrust, main_flow = compute_engine_thrust_3d(
        throttle_actual, pressure_pa, vehicle, thrust_scale=thrust_scale
    )
    thrust *= delivered_ratio
    main_flow *= delivered_ratio

    # 3-axis RCS propellant flow (L1 torque demand divided by thruster arm * Isp * g0)
    rcs_vec = np.asarray(rcs_torque_body_nm, dtype=np.float64).reshape(3)
    rcs_arm_xy = vehicle.geometry.rcs_arm_m
    rcs_arm_roll = vehicle.geometry.body_radius_m
    rcs_flow = (
        (abs(float(rcs_vec[0])) + abs(float(rcs_vec[1]))) / rcs_arm_xy
        + abs(float(rcs_vec[2])) / rcs_arm_roll
    ) / (vehicle.propulsion.rcs_isp_s * G0)

    bf_w, btau_b, q_pa, heat_w_m2, mach, alpha_deg, aero_roll_deg = compute_body_aero_3d(
        altitude_m=altitude_m,
        vel_enu_m_s=vel_enu_m_s,
        rot_wb=rot_wb,
        omega_body_rad_s=omega_body_rad_s,
        cg_station_m=cg_station_m,
        atmo_lut=atmo_lut,
        wind_lut=wind_lut,
        time_s=time_s,
        vehicle=vehicle,
        drag_scale=drag_scale,
        upper_attached=upper_attached,
        body_offset_body_m=body_offset_body_m,
    )
    ff_w, ftau_b, hinges = compute_fins_aero_3d(
        altitude_m=altitude_m,
        vel_enu_m_s=vel_enu_m_s,
        rot_wb=rot_wb,
        omega_body_rad_s=omega_body_rad_s,
        cg_station_m=cg_station_m,
        fin_deflection_rad=fin_deflection_rad,
        fin_extension=fin_extension,
        atmo_lut=atmo_lut,
        wind_lut=wind_lut,
        time_s=time_s,
        vehicle=vehicle,
        upper_attached=upper_attached,
        drag_scale=drag_scale,
        body_offset_body_m=body_offset_body_m,
    )

    # 2-axis conical TVC: gimbal_rad = (gamma_pitch_y, gamma_yaw_x)
    g_lim = radians(vehicle.propulsion.gimbal_limit_deg)
    gy, gx = float(gimbal_rad[0]), float(gimbal_rad[1])
    g_mag = hypot(gy, gx)
    if g_mag > g_lim and g_mag > 1e-12:
        scale_g = g_lim / g_mag
        gy *= scale_g
        gx *= scale_g
        g_mag = g_lim

    if g_mag > 1e-12:
        s_g = sin(g_mag) / g_mag
        t_body = np.array([thrust * gy * s_g, thrust * gx * s_g, thrust * cos(g_mag)])
    else:
        t_body = np.array([0.0, 0.0, thrust], dtype=np.float64)

    t_world = rot_wb @ t_body
    ox, oy, oz = body_offset_body_m
    r_noz_b = np.array(
        [ox, oy, vehicle.geometry.nozzle_station_m - cg_station_m + oz], dtype=np.float64
    )
    tvc_tau_b = np.cross(r_noz_b, t_body)

    aero_f_w = bf_w + ff_w
    total_f_w = aero_f_w + t_world
    total_tau_b = tvc_tau_b + rcs_vec + btau_b + ftau_b

    return ForceTorqueSample3D(
        force_world_n=(float(total_f_w[0]), float(total_f_w[1]), float(total_f_w[2])),
        torque_body_nm=(float(total_tau_b[0]), float(total_tau_b[1]), float(total_tau_b[2])),
        aero_force_world_n=(float(aero_f_w[0]), float(aero_f_w[1]), float(aero_f_w[2])),
        aero_torque_body_nm=(float(btau_b[0]), float(btau_b[1]), float(btau_b[2])),
        fin_force_world_n=(float(ff_w[0]), float(ff_w[1]), float(ff_w[2])),
        fin_torque_body_nm=(float(ftau_b[0]), float(ftau_b[1]), float(ftau_b[2])),
        fin_hinge_nm=hinges,
        thrust_n=float(thrust),
        main_flow_kg_s=float(main_flow),
        rcs_flow_kg_s=float(rcs_flow),
        dynamic_pressure_pa=float(q_pa),
        heat_flux_w_m2=float(heat_w_m2),
        mach=float(mach),
        total_alpha_deg=float(alpha_deg),
        aero_roll_deg=float(aero_roll_deg),
    )
