"""8-Phase Causal 6DOF RTLS Guidance & Control Core (`RtlsController3D`).

Implements `ControllerProtocol3D` strictly causally (TelemetryPacket3D -> ActuatorCommand3D):
  1. ascent           : 7-engine 3D gravity turn + roll hold + Max-Q throttle bucket
  2. flip             : Low-q 3-axis RCS flip toward 3D pad-return vector
  3. boostback        : 3-engine 3D predictor-corrector canceling both X & Y drift/Coriolis
  4. coast            : 3D retrograde alignment + grid-fin deployment at 58 km
  5. entry_burn       : 3-engine q/heat-flux guarded burn with 0.85 s ignition ramp
  6. aero_glide       : 0% main throttle! 4-grid-fin 3D glide steering (x_iip, y_iip) -> (0, 0)
  7. landing_burn     : 3->1 Hoverslam with Clarabel 3D GFOLD SOCP + 3D zero-tilt guidance
  8. touchdown_settle : Immediate MECO on first leg contact + 5 s 4-leg upright settle
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import asin, atan2, cos, degrees, exp, hypot, log, pi, radians, sin, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from dm100_3d.config import (
    EARTH_RADIUS_M,
    G0,
    OMEGA_EARTH,
    R_AIR,
    DM100VehicleConfig,
    GncV2Config,
    MassProperties3D,
    mass_properties_3d,
)
from dm100_3d.coords import (
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_to_rotmat,
    roll_angle_rad,
    rotmat_to_quat_wxyz,
    tilt_angle_rad,
)
from dm100_3d.gnc.allocation_3d import Allocation3DInput, PriorityControlAllocator3D
from dm100_3d.gnc.fdir_3d import CausalFdir3D, FdirStatus3D
from dm100_3d.gnc.mekf import CausalEstimator3D, EstimatedState3D, SensorConfig3D
from dm100_3d.gnc.mpc_gfold_3d import OnlineGfoldMpc3D
from dm100_3d.gnc.socp_gfold import (
    Gfold3DProblem,
    Gfold3DSolution,
    predict_zero_tilt_state_3d,
    solve_gfold_3d_socp,
)
from dm100_3d.interfaces import ActuatorCommand3D, GuidanceDiagnostics3D, TelemetryPacket3D
from dm100_3d.mjcf.builder import LEG_UNIT_DIRS
from dm100_3d.physics import (
    Atmosphere3DSettings,
    AtmosphereLUT,
    Wind3DConfig,
    Wind3DLUT,
    compute_body_aero_3d,
    compute_coriolis_and_centrifugal_3d,
    compute_engine_thrust_3d,
    compute_fins_aero_3d,
    compute_stagnation_heat_flux_w_m2,
    ground_effect_nominal_lift_n,
    srp_nominal_cp_shift_m,
    srp_nominal_drag_multiplier,
    wgs84_coriolis_and_transport_enu,
    wgs84_normal_gravity,
    wgs84_radii_of_curvature,
)
from dm100_3d.plant_mujoco import fast_gravity


def _smoothstep(u: float) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


def pointing_dir_to_quat_wxyz(dir_enu: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build gimbal-lock-free zero-roll ENU quaternion aligning body +z_b with `dir_enu`.

    Keeps body +y_b aligned with North (+Y) projection so planar XZ pitch and 180 deg flips
    produce zero parasitic roll rotation.
    """
    nrm = float(np.linalg.norm(dir_enu))
    zb = np.array([0.0, 0.0, 1.0], dtype=np.float64) if nrm < 1e-12 else dir_enu / nrm
    y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(zb @ y_ref)) > 0.98:
        y_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    yb = y_ref - float(y_ref @ zb) * zb
    yb /= float(np.linalg.norm(yb))
    xb = np.cross(yb, zb)
    xb /= float(np.linalg.norm(xb))
    rot = np.column_stack([xb, yb, zb])
    return rotmat_to_quat_wxyz(rot)


@dataclass(frozen=True, slots=True)
class Guidance3DSettings:
    control_dt_s: float = 0.02
    # Ascent gravity turn (matches DM100 2D AscentGuidanceSettings)
    pitch_start_s: float = 8.0
    pitch_duration_s: float = 10.0
    final_pitch_deg: float = 6.4
    separation_alignment_preview_s: float = 8.0
    separation_alignment_blend_s: float = 4.0
    # Flip & attitude limits
    flip_max_q_pa: float = 250.0
    flip_rate_limit_deg_s: float = 11.0
    flip_accel_limit_deg_s2: float = 2.8
    attitude_rate_limit_deg_s: float = 10.0
    attitude_accel_limit_deg_s2: float = 2.8
    high_q_accel_limit_deg_s2: float = 1.5
    high_q_threshold_pa: float = 5_000.0
    roll_rate_limit_deg_s: float = 8.0
    roll_accel_limit_deg_s2: float = 2.4
    # Boostback
    boostback_throttle_total: float = 3.0 / 7.0
    boostback_max_load_g: float = 3.85
    boostback_replan_period_s: float = 0.40
    boostback_miss_tolerance_m: float = 70.0
    # Entry burn
    entry_q_trigger_pa: float = 26_500.0
    entry_q_cutoff_pa: float = 25_500.0
    entry_heat_trigger_w_m2: float = 62_000.0
    entry_heat_cutoff_w_m2: float = 58_000.0
    entry_min_ignite_q_pa: float = 350.0
    entry_min_ignite_heat_w_m2: float = 10_000.0
    entry_max_load_g: float = 4.10
    # Aero glide
    fin_deploy_altitude_m: float = 58_000.0
    glide_iip_period_s: float = 0.25
    glide_max_alpha_deg: float = 16.0
    glide_handoff_altitude_m: float = 950.0
    # 3->1 Hoverslam
    landing_v_switch_m_s: float = 100.0
    landing_touchdown_speed_m_s: float = 0.65
    landing_divert_max_tilt_deg: float = 20.0
    landing_vertical_align_height_m: float = 48.0
    landing_vertical_body_height_m: float = 20.0
    gear_deploy_height_m: float = 400.0
    use_socp_reference: bool = True
    ignition_delay_budget_s: float = 0.28
    ignition_height_margin_m: float = 80.0


class RtlsController3D:
    """Strictly causal 6DOF RTLS controller implementing ControllerProtocol3D."""

    def __init__(
        self,
        *,
        vehicle: DM100VehicleConfig | None = None,
        atmo_settings: Atmosphere3DSettings | None = None,
        wind_config: Wind3DConfig | None = None,
        sensor_config: SensorConfig3D | None = None,
        guidance_settings: Guidance3DSettings | None = None,
        gnc_v2: GncV2Config | None = None,
    ) -> None:
        self.vehicle = vehicle or DM100VehicleConfig()
        self.gnc_v2 = gnc_v2 or GncV2Config()
        base_atmo = atmo_settings or Atmosphere3DSettings()
        # Causal forecast atmosphere NEVER sees hidden density_scale or upper_air_density_scale != 1
        self.forecast_atmo_settings = Atmosphere3DSettings(
            sea_level_temperature_k=base_atmo.sea_level_temperature_k,
            sea_level_pressure_pa=base_atmo.sea_level_pressure_pa,
            latitude_deg=base_atmo.latitude_deg,
            coriolis_enabled=base_atmo.coriolis_enabled,
            planar_coriolis_only=base_atmo.planar_coriolis_only,
            density_scale=1.0,
            upper_air_density_scale=1.0,
        )
        self.forecast_atmo = AtmosphereLUT(self.forecast_atmo_settings, step_m=100.0)
        self.forecast_wind = Wind3DLUT(wind_config or Wind3DConfig(), forecast=True)
        self.sensor_config = sensor_config or SensorConfig3D()
        self.settings = guidance_settings or Guidance3DSettings()

        self.allocator = PriorityControlAllocator3D(self.vehicle)
        self.estimator = CausalEstimator3D(
            vehicle=self.vehicle,
            forecast_atmo=self.forecast_atmo,
            forecast_wind=self.forecast_wind,
            latitude_deg=self.forecast_atmo_settings.latitude_deg,
            coriolis_enabled=self.forecast_atmo_settings.coriolis_enabled,
            planar_coriolis_only=self.forecast_atmo_settings.planar_coriolis_only,
            sensor_config=self.sensor_config,
            control_dt_s=self.settings.control_dt_s,
            gnc_v2=self.gnc_v2,
        )

        # Mission phase state
        self.phase: str = "ascent"
        self.cutoff_pending: bool = False
        self.sep_align_started_s: float | None = None
        self.flip_started_s: float | None = None
        self.boost_started_s: float | None = None
        self.boost_dir_xy: NDArray[np.float64] = np.array([-1.0, 0.0], dtype=np.float64)
        self.boost_end_s: float = 0.0
        self.next_replan_s: float = 0.0
        self.entry_burn_done: bool = False
        self.entry_single_engine: bool = False
        self.entry_burn_started_s: float | None = None
        self.glide_started_s: float | None = None
        self.landing_started_s: float | None = None
        self.landing_single_engine: bool = False
        self.in_vertical_body: bool = False
        self._vb_started_s: float | None = None
        self._land_bxy_smooth: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._land_bz_smooth: float = 0.0
        self._vb_wind_trim_xy: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._eb_steer_xy_deg: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._eb_iip_vel_xy_m_s: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._glide_alpha_xy_deg: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self.cached_drag_scale: float = 1.0
        self.last_iip_time_s: float = -1e9
        self.last_peak_q_time_s: float = -1e9
        self.cached_iip_xy_m: tuple[float, float] = (0.0, 0.0)
        self.cached_unpowered_q_pa: float | None = None
        self.cached_unpowered_qdot_w_m2: float = 0.0
        self.socp_solution: Gfold3DSolution | None = None
        self.socp_plan_start_s: float = 0.0
        self.last_diag = GuidanceDiagnostics3D()
        self._was_upper_attached: bool = True
        # M3 online 2 Hz log-mass GFOLD MPC & causal FDIR state (None when flags disabled)
        self.fdir: CausalFdir3D | None = (
            CausalFdir3D(
                vehicle=self.vehicle,
                control_dt_s=self.settings.control_dt_s,
                observation_delay_steps=self.sensor_config.observation_delay_steps,
                action_delay_steps=self.sensor_config.action_delay_steps,
            )
            if self.gnc_v2.use_fdir_v2
            else None
        )
        self.online_mpc: OnlineGfoldMpc3D | None = (
            OnlineGfoldMpc3D() if self.gnc_v2.use_online_mpc else None
        )
        self._fdir_status: FdirStatus3D | None = None
        self._land_mask_7: tuple[bool, bool, bool, bool, bool, bool, bool] | None = None
        self._td_started_s: float | None = None
        self._prev_eo_count: int = 0
        self._prev_fj_count: int = 0
        self._mpc_terminal_latched: bool = False

    def _nominal_mass(self, est: EstimatedState3D) -> float:
        return (
            self.vehicle.booster_dry_mass_kg
            + max(est.main_propellant_kg, 0.0)
            + max(est.rcs_propellant_kg, 0.0)
            + (self.vehicle.upper_stage.total_mass_kg if est.upper_attached else 0.0)
        )

    def _gravity(self, altitude_m: float) -> float:
        if self.gnc_v2.use_wgs84_gravity:
            return wgs84_normal_gravity(self.forecast_atmo_settings.latitude_deg, altitude_m)
        return fast_gravity(altitude_m)

    def _coriolis(
        self, altitude_m: float, vel_enu_m_s: tuple[float, float, float]
    ) -> NDArray[np.float64]:
        if self.gnc_v2.use_wgs84_gravity:
            return wgs84_coriolis_and_transport_enu(
                altitude_m=altitude_m,
                vel_enu_m_s=vel_enu_m_s,
                latitude_deg=self.forecast_atmo_settings.latitude_deg,
                coriolis_enabled=self.forecast_atmo_settings.coriolis_enabled,
                planar_coriolis_only=self.forecast_atmo_settings.planar_coriolis_only,
            )
        return compute_coriolis_and_centrifugal_3d(
            altitude_m=altitude_m,
            vel_enu_m_s=vel_enu_m_s,
            latitude_deg=self.forecast_atmo_settings.latitude_deg,
            coriolis_enabled=self.forecast_atmo_settings.coriolis_enabled,
            planar_coriolis_only=self.forecast_atmo_settings.planar_coriolis_only,
        )

    def _mass_props(
        self, mass: float, gear_ext: float, upper_attached: bool, rcs_kg: float
    ) -> MassProperties3D:
        if self.gnc_v2.use_dual_tank_mass_props:
            return mass_properties_3d(
                mass,
                gear_ext,
                vehicle=self.vehicle,
                upper_attached=upper_attached,
                dual_tank=True,
                rcs_propellant_kg=rcs_kg,
                dual_tank_geom=self.gnc_v2.dual_tank_geometry,
            )
        return mass_properties_3d(
            mass, gear_ext, vehicle=self.vehicle, upper_attached=upper_attached
        )

    def _foot_height_m(self, est: EstimatedState3D, cg_m: float) -> float:
        g = self.vehicle.geometry
        gear = self.vehicle.landing_gear
        dep = max(0.0, min(1.0, est.gear_extension))
        span = g.body_radius_m + dep * (gear.half_span_m - g.body_radius_m)
        f_st = gear.anchor_station_m + dep * (gear.foot_station_m - gear.anchor_station_m)
        r_wb = quat_to_rotmat(est.quat_wxyz)
        z_com = est.pos_enu_m[2]
        return min(
            z_com + float((r_wb @ np.array([ux * span, uy * span, f_st - cg_m]))[2])
            for ux, uy in LEG_UNIT_DIRS
        )

    def _air_vel(self, est: EstimatedState3D) -> NDArray[np.float64]:
        wx, wy, wz = self.forecast_wind.at(est.pos_enu_m[2], est.time_s)
        return np.array(
            [est.vel_enu_m_s[0] - wx, est.vel_enu_m_s[1] - wy, est.vel_enu_m_s[2] - wz],
            dtype=np.float64,
        )

    def _retro_dir(self, v_air: NDArray[np.float64], min_vz: float = 20.0) -> NDArray[np.float64]:
        d = np.array([-float(v_air[0]), -float(v_air[1]), max(-float(v_air[2]), min_vz)])
        return d / max(float(np.linalg.norm(d)), 1e-9)

    def _predict_unpowered_peak_q_3d(
        self,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        mass0: float,
        time0: float,
        *,
        dt: float = 0.30,
    ) -> tuple[float, float]:
        z = float(pos[2])
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        t = time0
        peak_q = 0.0
        peak_qdot = 0.0
        r = self.vehicle.geometry.body_radius_m
        ax_area = pi * r * r
        fin_area_eff = 0.5 * self.vehicle.fins.installed_area_m2
        cd_desc = self.vehicle.aerodynamics.axial_cd_descent
        fin_cd0 = self.vehicle.fins.base_drag_coefficient
        tdr = self.vehicle.aerodynamics.transonic_drag_rise
        lat = radians(self.forecast_atmo_settings.latitude_deg)
        cor_on = self.forecast_atmo_settings.coriolis_enabled
        om_y = OMEGA_EARTH * cos(lat) if cor_on else 0.0
        om_z = (
            0.0
            if (not cor_on or self.forecast_atmo_settings.planar_coriolis_only)
            else OMEGA_EARTH * sin(lat)
        )
        declining = 0

        for _ in range(700):
            if z <= 800.0:
                break
            t_k, _, rho = self.forecast_atmo.at(z)
            wx, wy, _ = self.forecast_wind.at(z, t)
            rx, ry, rz = vx - wx, vy - wy, vz
            spd = sqrt(rx * rx + ry * ry + rz * rz)
            q = 0.5 * rho * spd * spd
            qdot = compute_stagnation_heat_flux_w_m2(rho, spd, r)
            if q > peak_q:
                peak_q = q
                declining = 0
            elif z < 18_000.0 and q < 0.85 * peak_q:
                declining += 1
                if declining >= 6:
                    break
            peak_qdot = max(peak_qdot, qdot)
            mach = spd / sqrt(1.4 * R_AIR * t_k) if spd > 1.0 else 0.0
            corr = 1.0 + tdr * exp(-(((mach - 1.0) / 0.2) ** 2))
            fin_d = (
                fin_area_eff * fin_cd0
                if (z < self.settings.fin_deploy_altitude_m and vz < 0.0)
                else 0.0
            )
            drag = q * (corr * cd_desc * ax_area + fin_d)
            inv_s = 1.0 / max(spd, 1.0)
            if self.gnc_v2.use_wgs84_gravity:
                h_c = z if z > 0.0 else 0.0
                r_m_w, r_n_w = wgs84_radii_of_curvature(self.forecast_atmo_settings.latitude_deg)
                a_cent_z = (vx * vx) / (r_n_w + h_c) + (vy * vy) / (r_m_w + h_c)
            else:
                r_orb = EARTH_RADIUS_M + (z if z > 0.0 else 0.0)
                a_cent_z = (vx * vx + vy * vy) / r_orb
            ax = -drag * (rx * inv_s) / mass0 - 2.0 * (om_y * vz - om_z * vy)
            ay = -drag * (ry * inv_s) / mass0 - 2.0 * (om_z * vx)
            az = -drag * (rz * inv_s) / mass0 - self._gravity(z) + 2.0 * om_y * vx + a_cent_z
            z += vz * dt + 0.5 * az * dt * dt
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            t += dt
        return peak_q, peak_qdot

    def _predict_rtls_iip_3d(
        self,
        est: EstimatedState3D,
        burn_duration_s: float,
        boost_dir_xy: NDArray[np.float64],
        *,
        include_entry_burn: bool = True,
        handoff_z_m: float = 950.0,
        dt: float = 0.40,
        use_nav_bias: bool = False,
    ) -> tuple[float, float, float]:
        """Predict 3D horizontal impact point (x_iip, y_iip) at `handoff_z_m` and peak_q."""
        x, y, z = est.pos_enu_m
        vx, vy, vz = est.vel_enu_m_s
        mass = self._nominal_mass(est)
        t = est.time_s
        elapsed = 0.0
        peak_q = 0.0
        r = self.vehicle.geometry.body_radius_m
        ax_area = pi * r * r
        fin_area_eff = 0.5 * self.vehicle.fins.installed_area_m2
        cd_desc = self.vehicle.aerodynamics.axial_cd_descent
        fin_cd0 = self.vehicle.fins.base_drag_coefficient
        tdr = self.vehicle.aerodynamics.transonic_drag_rise
        lat = radians(self.forecast_atmo_settings.latitude_deg)
        cor_on = self.forecast_atmo_settings.coriolis_enabled
        om_y = OMEGA_EARTH * cos(lat) if cor_on else 0.0
        om_z = (
            0.0
            if (not cor_on or self.forecast_atmo_settings.planar_coriolis_only)
            else OMEGA_EARTH * sin(lat)
        )

        entry_active = False
        entry_done = self.entry_burn_done or (not include_entry_burn)
        bb_thr_cap = self.settings.boostback_throttle_total
        ign_wait = (
            self.settings.ignition_delay_budget_s
            if (burn_duration_s > 0.0 and est.engine_throttle < 0.05)
            else 0.0
        )

        bx, by = 0.0, 0.0
        if use_nav_bias and est.engine_throttle < 0.02 and z < 10_000.0:
            # Gate horizontal accel-bias by small body AoA (mirrors 2D): prevents high-AoA
            # lift-scale residuals from creating positive-feedback IIP runaway in thin air.
            r_wb0 = quat_to_rotmat(est.quat_wxyz)
            v_air0 = self._air_vel(est)
            v_air_b0 = r_wb0.T @ v_air0
            v_perp0 = hypot(float(v_air_b0[0]), float(v_air_b0[1]))
            alpha_b0_deg = degrees(atan2(v_perp0, max(-float(v_air_b0[2]), 1.0)))
            zero_aoa_w = max(0.0, 1.0 - alpha_b0_deg / 7.0)
            bx = zero_aoa_w * max(-1.6, min(1.6, est.accel_bias_world_m_s2[0]))
            by = zero_aoa_w * max(-1.6, min(1.6, est.accel_bias_world_m_s2[1]))

        drag_scale_est = 1.0
        if use_nav_bias and est.engine_throttle < 0.02 and z < 22_000.0:
            t0_k, _, rho0 = self.forecast_atmo.at(z)
            wx0, wy0, _ = self.forecast_wind.at(z, t)
            spd0 = sqrt((vx - wx0) ** 2 + (vy - wy0) ** 2 + vz * vz)
            q0 = 0.5 * rho0 * spd0 * spd0
            if q0 > 1_500.0 and vz < -50.0:
                m0 = spd0 / sqrt(1.4 * R_AIR * t0_k)
                c0 = 1.0 + tdr * exp(-(((m0 - 1.0) / 0.2) ** 2))
                f_nom_z = q0 * (c0 * cd_desc * ax_area + fin_area_eff * fin_cd0) * (-vz / spd0)
                a_nom_z = f_nom_z / max(mass, 1.0)
                if a_nom_z > 2.5:
                    drag_scale_est = max(
                        0.75, min(1.35, 1.0 + est.accel_bias_world_m_s2[2] / a_nom_z)
                    )
                    self.cached_drag_scale = drag_scale_est

        while z > handoff_z_m and elapsed < 580.0:
            step = dt
            boosting = elapsed < burn_duration_s - 1e-9
            if boosting and elapsed + step > burn_duration_s:
                step = max(1e-4, burn_duration_s - elapsed)

            t_k, p_pa, rho = self.forecast_atmo.at(z)
            wx, wy, _ = self.forecast_wind.at(z, t)
            rx, ry, rz = vx - wx, vy - wy, vz
            spd = sqrt(rx * rx + ry * ry + rz * rz)
            q = 0.5 * rho * spd * spd
            peak_q = max(peak_q, q)
            qdot = compute_stagnation_heat_flux_w_m2(rho, spd, r)
            mach = spd / sqrt(1.4 * R_AIR * t_k) if spd > 1.0 else 0.0
            corr = 1.0 + tdr * exp(-(((mach - 1.0) / 0.2) ** 2))
            fin_d = (
                fin_area_eff * fin_cd0
                if (not boosting and z < self.settings.fin_deploy_altitude_m and vz < 0.0)
                else 0.0
            )
            drag = drag_scale_est * q * (corr * cd_desc * ax_area + fin_d)
            inv_s = 1.0 / max(spd, 1.0)
            fx = -drag * (rx * inv_s)
            fy = -drag * (ry * inv_s)
            fz = -drag * (rz * inv_s)
            flow = 0.0

            if boosting:
                if elapsed >= ign_wait:
                    full_t, _ = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)
                    min_m = max(
                        self.vehicle.booster_dry_mass_kg - 1_000.0,
                        mass - 0.01 * self.vehicle.launch_mass_kg,
                    )
                    max_bb_t = self.settings.boostback_max_load_g * G0 * min_m / 1.03
                    lo3, _ = self.vehicle.propulsion.throttle_bounds(3)
                    ramp = (
                        _smoothstep((elapsed - ign_wait) / 2.00)
                        if est.engine_throttle < 0.05
                        else 1.0
                    )
                    bb_req = lo3 + ramp * max(0.0, bb_thr_cap - lo3)
                    bb_thr = self.vehicle.propulsion.quantize_throttle(
                        min(bb_req, max_bb_t / max(full_t, 1.0)), 3
                    )
                    thrust, flow = compute_engine_thrust_3d(bb_thr, p_pa, self.vehicle)
                    fx += float(boost_dir_xy[0]) * thrust
                    fy += float(boost_dir_xy[1]) * thrust
            elif not entry_done and vz < 0.0 and z < 68_000.0:
                if not entry_active:
                    if (
                        q >= self.settings.entry_min_ignite_q_pa
                        or qdot >= self.settings.entry_min_ignite_heat_w_m2
                    ):
                        pq, pqd = self._predict_unpowered_peak_q_3d(
                            (x, y, z), (vx, vy, vz), mass, t, dt=0.45
                        )
                        if (
                            pq > self.settings.entry_q_trigger_pa
                            or pqd > self.settings.entry_heat_trigger_w_m2
                        ):
                            entry_active = True
                else:
                    pq, pqd = self._predict_unpowered_peak_q_3d(
                        (x, y, z), (vx, vy, vz), mass, t, dt=0.45
                    )
                    if (
                        pq <= self.settings.entry_q_cutoff_pa
                        and pqd <= self.settings.entry_heat_cutoff_w_m2
                    ) or z < 11_000.0:
                        entry_active = False
                        entry_done = True
                if entry_active:
                    full_t, _ = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)
                    min_m = max(
                        self.vehicle.booster_dry_mass_kg - 1_000.0,
                        mass - 0.01 * self.vehicle.launch_mass_kg,
                    )
                    max_t = max(
                        0.0, (self.settings.entry_max_load_g * G0 * min_m - 1.15 * drag) / 1.03
                    )
                    raw_thr = max(0.0, max_t / max(full_t, 1.0))
                    lo3, _ = self.vehicle.propulsion.throttle_bounds(3)
                    eb_eng = 1 if (q > 12_000.0 and raw_thr < lo3) else 3
                    e_thr = self.vehicle.propulsion.quantize_throttle(raw_thr, eb_eng)
                    thrust, flow = compute_engine_thrust_3d(e_thr, p_pa, self.vehicle)
                    if self.gnc_v2.use_srp_iip_feedforward:
                        body_drag_only = drag_scale_est * q * corr * cd_desc * ax_area
                        srp_mult = srp_nominal_drag_multiplier(
                            mach=mach,
                            v_air_zb_m_s=-spd,
                            thrust_n=thrust,
                            dynamic_pressure_pa=q,
                            body_radius_m=r,
                            eta_nom=self.gnc_v2.srp_nominal_eta,
                        )
                        delta_drag = body_drag_only * (1.0 - srp_mult)
                        fx += delta_drag * (rx * inv_s)
                        fy += delta_drag * (ry * inv_s)
                        fz += delta_drag * (rz * inv_s)
                    fx += thrust * (-rx * inv_s)
                    fy += thrust * (-ry * inv_s)
                    fz += thrust * (-rz * inv_s)

            mid_m = max(self.vehicle.booster_dry_mass_kg, mass - 0.5 * flow * step)
            if self.gnc_v2.use_wgs84_gravity:
                h_c = z if z > 0.0 else 0.0
                r_m_w, r_n_w = wgs84_radii_of_curvature(self.forecast_atmo_settings.latitude_deg)
                a_cent_z = (vx * vx) / (r_n_w + h_c) + (vy * vy) / (r_m_w + h_c)
            else:
                r_orb = EARTH_RADIUS_M + (z if z > 0.0 else 0.0)
                a_cent_z = (vx * vx + vy * vy) / r_orb
            bias_alt_m = 7_000.0 if self.gnc_v2.use_structural_filter else 14_000.0
            ax = fx / mid_m - 2.0 * (om_y * vz - om_z * vy) + (bx if z < bias_alt_m else 0.0)
            ay = fy / mid_m - 2.0 * (om_z * vx) + (by if z < bias_alt_m else 0.0)
            az = fz / mid_m - self._gravity(z) + 2.0 * om_y * vx + a_cent_z
            x += vx * step + 0.5 * ax * step * step
            y += vy * step + 0.5 * ay * step * step
            z += vz * step + 0.5 * az * step * step
            vx += ax * step
            vy += ay * step
            vz += az * step
            mass = max(self.vehicle.booster_dry_mass_kg, mass - flow * step)
            t += step
            elapsed += step

        rem_h = max(0.0, z)
        scale_rem = (rem_h / max(-vz, 30.0)) * 0.55
        return x + vx * scale_rem, y + vy * scale_rem, peak_q

    def _plan_boostback_3d(self, est: EstimatedState3D) -> None:
        """Solve 3D horizontal boostback direction and burn duration targeting (-150 m, 0 m)."""
        target_xy = np.array([-150.0, 0.0], dtype=np.float64)
        x0, y0, _ = self._predict_rtls_iip_3d(
            est,
            0.0,
            self.boost_dir_xy,
            include_entry_burn=True,
            handoff_z_m=self.settings.glide_handoff_altitude_m,
        )
        miss_vec0 = np.array([x0, y0], dtype=np.float64) - target_xy
        miss_dist0 = float(np.linalg.norm(miss_vec0))
        if miss_dist0 <= self.settings.boostback_miss_tolerance_m:
            self.boost_end_s = est.time_s
            self.next_replan_s = est.time_s + self.settings.boostback_replan_period_s
            return

        # Aim opposite the unpowered IIP miss vector (cancels X downrange AND Y crossrange/Coriolis)
        b_dir = -miss_vec0 / miss_dist0
        self.boost_dir_xy = b_dir

        # Solve burn duration along `b_dir` where projection of miss onto -b_dir crosses 0
        cache: dict[float, float] = {}

        def eval_proj(dur: float) -> float:
            key = round(dur, 4)
            if key not in cache:
                xi, yi, _ = self._predict_rtls_iip_3d(
                    est,
                    dur,
                    b_dir,
                    include_entry_burn=True,
                    handoff_z_m=self.settings.glide_handoff_altitude_m,
                )
                m_vec = np.array([xi, yi], dtype=np.float64) - target_xy
                cache[key] = float(m_vec @ (-b_dir))
            return cache[key]

        prev_t = 0.0
        prev_val = eval_proj(0.0)
        root: float | None = None
        for lim in (4.0, 10.0, 18.0, 28.0, 40.0, 54.0, 70.0):
            val = eval_proj(lim)
            if prev_val * val <= 0.0:
                root = float(brentq(eval_proj, prev_t, lim, xtol=0.02, maxiter=20))
                break
            prev_t, prev_val = lim, val
        dur = root if root is not None else 28.0

        # Refine 2D direction using exact residual vector (sign-consistent across all 360 deg)
        xi_r, yi_r, _ = self._predict_rtls_iip_3d(
            est,
            dur,
            b_dir,
            include_entry_burn=True,
            handoff_z_m=self.settings.glide_handoff_altitude_m,
        )
        res_vec = np.array([xi_r, yi_r], dtype=np.float64) - target_xy
        res_norm = float(np.linalg.norm(res_vec))
        if dur > 1.0 and res_norm > 40.0:
            eff_miss = miss_vec0 + 0.85 * res_vec
            eff_norm = float(np.linalg.norm(eff_miss))
            if eff_norm > 1e-6:
                b_new = -eff_miss / eff_norm
                # Limit per-replan azimuth step to 0.30 rad for smooth attitude tracking
                dot_b = max(-1.0, min(1.0, float(b_dir @ b_new)))
                cross_b = float(b_dir[0] * b_new[1] - b_dir[1] * b_new[0])
                d_ang = max(-0.30, min(0.30, atan2(cross_b, dot_b)))
                c_y, s_y = cos(d_ang), sin(d_ang)
                b_dir = np.array(
                    [c_y * b_dir[0] - s_y * b_dir[1], s_y * b_dir[0] + c_y * b_dir[1]],
                    dtype=np.float64,
                )
                b_dir /= float(np.linalg.norm(b_dir))
                self.boost_dir_xy = b_dir
                if res_norm > 150.0:
                    cache.clear()
                    prev_t = 0.0
                    prev_val = eval_proj(0.0)
                    for lim in (4.0, 10.0, 18.0, 28.0, 40.0, 54.0, 70.0):
                        val = eval_proj(lim)
                        if prev_val * val <= 0.0:
                            dur = float(brentq(eval_proj, prev_t, lim, xtol=0.02, maxiter=20))
                            break
                        prev_t, prev_val = lim, val

        self.boost_end_s = est.time_s + dur
        self.next_replan_s = est.time_s + self.settings.boostback_replan_period_s

    def _attitude_allocate_3d(
        self,
        est: EstimatedState3D,
        target_quat_wxyz: NDArray[np.float64],
        throttle_cmd: float,
        *,
        rate_limit_deg_s: float | None = None,
        accel_limit_deg_s2: float | None = None,
        allow_fin: bool = False,
        thrust_tilt_cmd_body_xy: tuple[float, float] | None = None,
        rcs_deadband_nm: float = 0.0,
        high_q_guard: bool = True,
        accel_guard_deg_s2: float = 3.5,
        use_xy_torque_bias: bool = False,
        engine_mask_7: tuple[bool, ...] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float, float], tuple[float, float, float, float]]:
        """3-axis quaternion attitude controller -> PriorityControlAllocator3D (3x9)."""
        s = self.settings
        mass = self._nominal_mass(est)
        mp = self._mass_props(mass, est.gear_extension, est.upper_attached, est.rcs_propellant_kg)
        cg = mp.cg_station_m
        i_diag = mp.inertia_diag_kg_m2

        # Conservative lower inertia (97%) for angular-acceleration safety guards
        i_low = (0.97 * i_diag[0], 0.97 * i_diag[1], 0.97 * i_diag[2])
        q_est = normalize_quat_wxyz(est.quat_wxyz, canonical=True)
        r_wb = quat_to_rotmat(q_est)
        om_b = np.asarray(est.omega_body_rad_s, dtype=np.float64)

        t_k_a, p_pa_a, rho_a = self.forecast_atmo.at(est.pos_enu_m[2])
        cur_t, _ = compute_engine_thrust_3d(est.engine_throttle, p_pa_a, self.vehicle)
        cmd_t, _ = compute_engine_thrust_3d(throttle_cmd, p_pa_a, self.vehicle)
        fut_t = cmd_t + (cur_t - cmd_t) * exp(
            -s.control_dt_s / self.vehicle.propulsion.nominal_engine_tau_s
        )
        srp_drag_mult = 1.0
        srp_cp_shift = 0.0
        if (
            self.gnc_v2.use_srp_iip_feedforward
            and (not est.upper_attached)
            and est.engine_throttle >= 1e-3
        ):
            t_srp_mid = 0.5 * (cur_t + fut_t) if throttle_cmd > 1e-3 else cur_t
            if t_srp_mid > 1.0:
                v_air_s = self._air_vel(est)
                spd_s = float(np.linalg.norm(v_air_s))
                if spd_s > 1.0:
                    q_s = 0.5 * rho_a * spd_s * spd_s
                    mach_s = spd_s / sqrt(1.4 * R_AIR * t_k_a)
                    vz_b_s = float((r_wb.T @ v_air_s)[2])
                    r_body = self.vehicle.geometry.body_radius_m
                    srp_drag_mult = srp_nominal_drag_multiplier(
                        mach=mach_s,
                        v_air_zb_m_s=vz_b_s,
                        thrust_n=t_srp_mid,
                        dynamic_pressure_pa=q_s,
                        body_radius_m=r_body,
                        eta_nom=self.gnc_v2.srp_nominal_eta,
                        upper_attached=False,
                    )
                    srp_cp_shift = srp_nominal_cp_shift_m(
                        mach=mach_s,
                        v_air_zb_m_s=vz_b_s,
                        thrust_n=t_srp_mid,
                        dynamic_pressure_pa=q_s,
                        body_radius_m=r_body,
                        upper_attached=False,
                    )

        _, btau_b, q_pa, _, _, _, _ = compute_body_aero_3d(
            altitude_m=est.pos_enu_m[2],
            vel_enu_m_s=est.vel_enu_m_s,
            rot_wb=r_wb,
            omega_body_rad_s=om_b,
            cg_station_m=cg,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=est.time_s,
            vehicle=self.vehicle,
            upper_attached=est.upper_attached,
            axial_drag_multiplier=srp_drag_mult,
            cp_shift_m=srp_cp_shift,
        )

        # Decoupled 2-axis pointing error (body X/Y) + direct geometric zero-roll error (body Z)
        # Prevents large 130 deg 3D flip geodesics from inducing parasitic geometric roll!
        q_des = normalize_quat_wxyz(target_quat_wxyz, canonical=True)
        q_err = quat_multiply_wxyz(quat_conjugate_wxyz(q_est), q_des)
        if q_err[0] < 0.0:
            q_err = -q_err
        err_vec = 2.0 * q_err[1:4]
        err_vec[2] = -roll_angle_rad(q_est)

        r_lim_xy = radians(rate_limit_deg_s or s.attitude_rate_limit_deg_s)
        a_lim_xy = radians(accel_limit_deg_s2 or s.attitude_accel_limit_deg_s2)
        if high_q_guard and not est.upper_attached and q_pa >= s.high_q_threshold_pa:
            a_lim_xy = min(a_lim_xy, radians(s.high_q_accel_limit_deg_s2))
        r_lims = (r_lim_xy, r_lim_xy, radians(s.roll_rate_limit_deg_s))
        a_lims = (a_lim_xy, a_lim_xy, radians(s.roll_accel_limit_deg_s2))

        des_alpha = np.zeros(3, dtype=np.float64)
        for ax in range(3):
            e = float(err_vec[ax])
            stop_r = sqrt(2.0 * 0.78 * a_lims[ax] * abs(e))
            if self.gnc_v2.use_structural_filter:
                # Locked CDR Rev 3.1 §3.1 CSI-certified gains (all 9 regimes GM>=6 dB, PM>=35 deg)
                if ax == 2:
                    kp, kd = 1.20, 2.50
                elif est.upper_attached:
                    kp, kd = 0.40, 1.20
                else:
                    kp, kd = 0.55, 1.65
            else:
                kp = 3.2 if ax == 2 else 1.8
                kd = 5.5 if ax == 2 else 4.2
            des_r = (1.0 if e >= 0.0 else -1.0) * min(r_lims[ax], kp * abs(e), stop_r)
            des_alpha[ax] = max(-a_lims[ax], min(a_lims[ax], kd * (des_r - float(om_b[ax]))))

        # Clamp pitch/yaw norm to a_lim_xy, and full 3D vector norm inside accel_guard_deg_s2
        xy_norm = hypot(float(des_alpha[0]), float(des_alpha[1]))
        if xy_norm > a_lim_xy and xy_norm > 1e-12:
            des_alpha[0] *= a_lim_xy / xy_norm
            des_alpha[1] *= a_lim_xy / xy_norm
        alpha_norm = float(np.linalg.norm(des_alpha))
        max_vec_acc = radians(min(accel_guard_deg_s2, 3.6))
        if alpha_norm > max_vec_acc and alpha_norm > 1e-12:
            des_alpha *= max_vec_acc / alpha_norm

        # Gyroscopic coupling feedforward: omega x (I * omega)
        i_vec = np.asarray(i_low, dtype=np.float64)
        gyro_ff = np.cross(om_b, i_vec * om_b)
        tb = np.asarray(est.torque_bias_body_nm, dtype=np.float64).copy()
        if not use_xy_torque_bias:
            tb[0] = 0.0
            tb[1] = 0.0
        if (
            self.fdir is not None
            and any(self.fdir.fin_jammed)
            and self.fdir.fin_jam_detected_time_s is not None
            and (est.time_s - self.fdir.fin_jam_detected_time_s) <= 1.05
        ):
            # Suppress Roll DOB during the 1.05 s Fin-Jam runaway & healthy-fin reconfiguration
            # slew so transient resolver lag cannot wind up wrong-sign roll bias.
            tb[2] = 0.0
        if (
            self.fdir is not None
            and any(self.fdir.engine_failed)
            and self.fdir.engine_out_detected_time_s is not None
            and (est.time_s - self.fdir.engine_out_detected_time_s) <= 0.40
        ):
            # Suppress P/Y DOB during the 0.40 s post-EO turbopump spindown & 3->3_sym swap
            # so transient asymmetric thrust mismatch cannot charge wrong-sign P/Y bias.
            tb[0] = 0.0
            tb[1] = 0.0
        # M3 causal 3-axis asymmetric engine-offset moment feedforward (matches mekf.py tau_model +
        # conical gimbal roll coupling + causal chamber-pressure spindown residual)
        if self.fdir is not None and engine_mask_7 is not None and est.engine_throttle >= 1e-3:
            t_asym = 0.5 * (cur_t + fut_t) if throttle_cmd > 1e-3 else cur_t
            full_t_a, _ = compute_engine_thrust_3d(1.0, p_pa_a, self.vehicle)
            if t_asym > 1.0 or full_t_a > 1.0:
                asym_ff = self.fdir.causal_asymmetric_torque_ff_nm(
                    engine_mask_7=engine_mask_7,
                    total_thrust_n=t_asym,
                    full_thrust_7_n=full_t_a,
                    gimbal_rad=est.gimbal_rad,
                )
                btau_b = btau_b + np.asarray(asym_ff, dtype=np.float64)
        req_tau = i_vec * des_alpha + gyro_ff - btau_b - tb
        bg_tau = btau_b - gyro_ff + tb

        if throttle_cmd <= 1e-3 or (est.engine_throttle < 1e-3 and throttle_cmd > 0.0):
            sizing_t = 0.0
        else:
            sizing_t = max(cur_t, fut_t)

        # Compute body-frame retrograde AoA components for fin bounds & weathercock term
        v_air_w = self._air_vel(est)
        v_air_b = r_wb.T @ v_air_w
        if float(v_air_b[2]) < 0.0:
            ax_ret = atan2(float(v_air_b[0]), max(-float(v_air_b[2]), 1e-3))
            ay_ret = atan2(float(v_air_b[1]), max(-float(v_air_b[2]), 1e-3))
        else:
            ax_ret, ay_ret = 0.0, 0.0

        # Exact operating-point linearization of 4-fin aero (includes fin drag weathercock
        # torque and cos(alpha) lift projection so RCS seamlessly fills any true fin deficit).
        fin_tau0_exact: tuple[float, float, float] | None = None
        fin_eff_exact: tuple[float, float, float] | None = None
        fin_jac_3x4: NDArray[np.float64] | None = None
        fin_ref_def: tuple[float, float, float, float] | None = None
        fin_ref_tau: tuple[float, float, float] | None = None
        fin_jam_mask: tuple[bool, bool, bool, bool] | None = None
        fin_jam_ang: tuple[float, float, float, float] | None = None
        has_fin_jam = self.fdir is not None and any(self.fdir.fin_jammed)
        if (
            not est.upper_attached
            and est.fin_extension >= 0.05
            and q_pa >= self.vehicle.fins.control_min_q_pa
        ):
            dx_ref = max(-0.25, min(0.25, ax_ret))
            dy_ref = max(-0.25, min(0.25, ay_ret))
            u_fin = radians(1.0)

            def _eval_fin(d4: tuple[float, float, float, float]) -> NDArray[np.float64]:
                _, t_f, _ = compute_fins_aero_3d(
                    altitude_m=est.pos_enu_m[2],
                    vel_enu_m_s=est.vel_enu_m_s,
                    rot_wb=r_wb,
                    omega_body_rad_s=om_b,
                    cg_station_m=cg,
                    fin_deflection_rad=d4,
                    fin_extension=est.fin_extension,
                    atmo_lut=self.forecast_atmo,
                    wind_lut=self.forecast_wind,
                    time_s=est.time_s,
                    vehicle=self.vehicle,
                )
                return t_f

            if has_fin_jam and self.fdir is not None:
                # Exact per-column 3x4 Jacobian around ACTUAL operating fin angles (eliminates
                # 14-deg linearization roll drift when healthy fins deflect far from feather!)
                jam_m = tuple(self.fdir.fin_jammed)
                jam_a = tuple(self.fdir.fin_jam_angles_rad)
                f_lim_op = radians(self.vehicle.fins.deflection_limit_deg)
                d_base = [
                    jam_a[j]
                    if jam_m[j]
                    else max(-f_lim_op, min(f_lim_op, float(est.fin_deflection_rad[j])))
                    for j in range(4)
                ]
                d_ref_t = (d_base[0], d_base[1], d_base[2], d_base[3])
                t_ref = _eval_fin(d_ref_t)
                jac = np.zeros((3, 4), dtype=np.float64)
                for j in range(4):
                    dp = list(d_base)
                    dm = list(d_base)
                    dp[j] += u_fin
                    dm[j] -= u_fin
                    tp = _eval_fin((dp[0], dp[1], dp[2], dp[3]))
                    tm = _eval_fin((dm[0], dm[1], dm[2], dm[3]))
                    jac[:, j] = (tp - tm) / (2.0 * u_fin)
                fin_jac_3x4 = jac
                fin_ref_def = d_ref_t
                fin_ref_tau = (float(t_ref[0]), float(t_ref[1]), float(t_ref[2]))
                fin_jam_mask = jam_m  # type: ignore[assignment]
                fin_jam_ang = jam_a  # type: ignore[assignment]
            else:
                t_ref = _eval_fin((dy_ref, dy_ref, dx_ref, dx_ref))
                t_yp = _eval_fin((dy_ref + u_fin, dy_ref + u_fin, dx_ref, dx_ref))
                t_ym = _eval_fin((dy_ref - u_fin, dy_ref - u_fin, dx_ref, dx_ref))
                t_pp = _eval_fin((dy_ref, dy_ref, dx_ref + u_fin, dx_ref + u_fin))
                t_pm = _eval_fin((dy_ref, dy_ref, dx_ref - u_fin, dx_ref - u_fin))
                t_rp = _eval_fin((dy_ref + u_fin, dy_ref - u_fin, dx_ref - u_fin, dx_ref + u_fin))
                t_rm = _eval_fin((dy_ref - u_fin, dy_ref + u_fin, dx_ref + u_fin, dx_ref - u_fin))
                eff_x = float(t_yp[0] - t_ym[0]) / (2.0 * u_fin)
                eff_y = float(t_pp[1] - t_pm[1]) / (2.0 * u_fin)
                eff_z = float(t_rp[2] - t_rm[2]) / (2.0 * u_fin)
                fin_tau0_exact = (
                    float(t_ref[0]) - eff_x * dy_ref,
                    float(t_ref[1]) - eff_y * dx_ref,
                    float(t_ref[2]),
                )
                fin_eff_exact = (eff_x, eff_y, eff_z)

        lever = cg - self.vehicle.geometry.nozzle_station_m
        fin_arm_z = self.vehicle.fins.station_m - cg
        out = self.allocator.allocate(
            Allocation3DInput(
                requested_torque_body_nm=(float(req_tau[0]), float(req_tau[1]), float(req_tau[2])),
                background_torque_body_nm=(float(bg_tau[0]), float(bg_tau[1]), float(bg_tau[2])),
                inertia_diag_kg_m2=i_low,
                lever_arm_m=lever,
                sizing_thrust_n=sizing_t,
                fin_arm_z_m=fin_arm_z,
                dynamic_pressure_pa=q_pa,
                fin_extension=est.fin_extension,
                alpha_retro_xy_rad=(ax_ret, ay_ret),
                rcs_available=est.rcs_propellant_kg > 0.0,
                allow_fin_allocation=allow_fin and (est.fin_extension >= 0.05),
                thrust_tilt_cmd_body_xy=thrust_tilt_cmd_body_xy,
                current_gimbal_rad=est.gimbal_rad,
                current_thrust_n=cur_t,
                control_dt_s=s.control_dt_s,
                angular_accel_guard_deg_s2=accel_guard_deg_s2,
                rcs_deadband_nm=rcs_deadband_nm,
                fin_zero_torque_body_nm=fin_tau0_exact,
                fin_axis_eff_nm_per_rad=fin_eff_exact,
                fin_jacobian_3x4=fin_jac_3x4,
                fin_ref_deflection_rad=fin_ref_def,
                fin_ref_torque_body_nm=fin_ref_tau,
                fin_jammed_mask=fin_jam_mask,
                fin_jammed_angles_rad=fin_jam_ang,
                current_fin_deflection_rad=(est.fin_deflection_rad if has_fin_jam else None),
                fin_jam_elapsed_s=(
                    max(0.0, est.time_s - self.fdir.fin_jam_detected_time_s)
                    if (
                        has_fin_jam
                        and self.fdir is not None
                        and self.fdir.fin_jam_detected_time_s is not None
                    )
                    else None
                ),
                vb_gimbal_max_step_deg=(1.0 if self.gnc_v2.use_structural_filter else 1.6),
            )
        )
        # Guard pre-separation RCS torque against lighter post-separation booster inertia
        rcs_cmd = out.rcs_torque_body_nm
        if est.upper_attached and self.cutoff_pending:
            mp_sep = self._mass_props(
                mass - self.vehicle.upper_stage.total_mass_kg,
                est.gear_extension,
                False,
                est.rcs_propellant_kg,
            )
            lims_sep = [i_s * radians(2.0) for i_s in mp_sep.inertia_diag_kg_m2]
            rcs_cmd = (
                max(-lims_sep[0], min(lims_sep[0], rcs_cmd[0])),
                max(-lims_sep[1], min(lims_sep[1], rcs_cmd[1])),
                max(-lims_sep[2], min(lims_sep[2], rcs_cmd[2])),
            )
        return out.gimbal_rad, rcs_cmd, out.fin_deflection_rad

    def _compute_hoverslam_ignition_height(self, est: EstimatedState3D, mass: float) -> float:
        s = self.settings
        vz_down = max(0.0, -est.vel_enu_m_s[2])
        v_td = 0.50 if self.gnc_v2.use_strapdown_mekf else s.landing_touchdown_speed_m_s
        if vz_down <= v_td:
            return 0.0

        g_loc = self._gravity(est.pos_enu_m[2])
        _, p_pa, rho = self.forecast_atmo.at(min(max(est.pos_enu_m[2], 0.0), 3_000.0))
        r = self.vehicle.geometry.body_radius_m
        drag_eff = min(1.0, max(0.75, self.cached_drag_scale))
        cda = drag_eff * (
            self.vehicle.aerodynamics.axial_cd_descent * (pi * r * r)
            + self.vehicle.fins.base_drag_coefficient * 0.5 * self.vehicle.fins.installed_area_m2
        )

        t_lag = (
            s.ignition_delay_budget_s
            + (self.sensor_config.observation_delay_steps + self.sensor_config.action_delay_steps)
            * s.control_dt_s
        )
        dt_lag = t_lag / 10.0
        v = vz_down
        m = mass
        dist = 0.0
        for _ in range(10):
            drag = 0.5 * rho * v * v * cda
            dv = g_loc - drag / m
            dist += v * dt_lag + 0.5 * dv * dt_lag * dt_lag
            v = max(v_td, v + dv * dt_lag)

        full_t, _ = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)
        _, hi3 = self.vehicle.propulsion.throttle_bounds(3)
        thr_3 = self.vehicle.propulsion.quantize_throttle(
            min(hi3, 3.75 * G0 * m / max(full_t, 1.0)), 3
        )
        t_3, f_3 = compute_engine_thrust_3d(thr_3, p_pa, self.vehicle)
        lo1, hi1 = self.vehicle.propulsion.throttle_bounds(1)
        t_min_1, _ = compute_engine_thrust_3d(lo1, p_pa, self.vehicle)
        t_max_1, _ = compute_engine_thrust_3d(hi1, p_pa, self.vehicle)

        m_b = max(self.vehicle.booster_dry_mass_kg, m - 700.0)
        m_light = max(self.vehicle.booster_dry_mass_kg, m_b - 0.011 * self.vehicle.launch_mass_kg)
        a_min_1 = max(0.5, t_min_1 / m_b - g_loc)
        a_min_w = max(a_min_1, 1.04 * t_min_1 / m_light - g_loc)
        a_max_1 = max(a_min_w + 2.5, t_max_1 / m_b - g_loc)
        a_lo = min(a_max_1 - 3.0, a_min_w + 0.45)
        a_hi = min(a_max_1 - 2.0, max(a_lo + 2.0, a_min_1 + 0.58 * (a_max_1 - a_min_1)))
        h_shape = 680.0
        v_top = sqrt(v_td * v_td + (a_lo + a_hi) * h_shape)
        v_sw = min(s.landing_v_switch_m_s, v_top)

        k2 = (a_hi - a_lo) / h_shape
        rhs_v = max(0.0, min(v, v_sw) ** 2 - v_td * v_td)
        h_stage_b = (
            (-a_lo + sqrt(a_lo * a_lo + k2 * rhs_v)) / k2 if k2 > 1e-6 else rhs_v / (2.0 * a_lo)
        )

        dt = 0.05
        h_stage_a = 0.0
        while v > v_sw:
            drag = 0.5 * rho * v * v * cda
            a_dec = max(2.0, (t_3 + 0.7 * drag) / m - g_loc)
            step_t = min(dt, (v - v_sw) / a_dec)
            h_stage_a += v * step_t - 0.5 * a_dec * step_t * step_t
            v -= a_dec * step_t
            m = max(self.vehicle.booster_dry_mass_kg, m - f_3 * step_t)

        return dist + h_stage_a + h_stage_b + s.ignition_height_margin_m

    def _ascent_guidance(self, est: EstimatedState3D) -> ActuatorCommand3D:
        s = self.settings
        ag = self.vehicle.ascent_gate
        mass = self._nominal_mass(est)
        mp = self._mass_props(mass, est.gear_extension, True, est.rcs_propellant_kg)
        r_wb = quat_to_rotmat(est.quat_wxyz)
        bf_w, _, q_pa, _, _, _, _ = compute_body_aero_3d(
            altitude_m=est.pos_enu_m[2],
            vel_enu_m_s=est.vel_enu_m_s,
            rot_wb=r_wb,
            omega_body_rad_s=est.omega_body_rad_s,
            cg_station_m=mp.cg_station_m,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=est.time_s,
            vehicle=self.vehicle,
            upper_attached=True,
        )
        v_air = self._air_vel(est)
        speed = float(np.linalg.norm(v_air))
        _, p_pa, rho = self.forecast_atmo.at(est.pos_enu_m[2])
        full_t, full_flow = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)

        shutdown_delay = (
            self.sensor_config.observation_delay_steps + self.sensor_config.action_delay_steps + 1
        ) * s.control_dt_s + 0.05
        gauge_margin_kg = 85.0 if self.gnc_v2.use_strapdown_mekf else 0.0
        shutdown_res = full_flow * shutdown_delay + 20.0 + gauge_margin_kg
        fdir_eo_ascent = self.fdir is not None and any(self.fdir.engine_failed)
        is_abort_ascent = (
            self.fdir is not None and self.fdir.mission_mode == "DOWNRANGE_SPLASHDOWN_ABORT"
        )
        if is_abort_ascent:
            sep_min_prop_target = 3_500.0
            v_inert_now = float(np.linalg.norm(est.vel_enu_m_s))
            if (
                est.main_propellant_kg <= sep_min_prop_target + shutdown_res
                or (v_inert_now >= 1_635.0 and est.pos_enu_m[2] >= 48_000.0)
                or est.pos_enu_m[2] >= ag.separation_altitude_m + 18_000.0
            ):
                self.cutoff_pending = True
        elif fdir_eo_ascent:
            sep_min_prop_target = 9_500.0
            v_inert_now = float(np.linalg.norm(est.vel_enu_m_s))
            if (
                est.main_propellant_kg <= sep_min_prop_target + shutdown_res
                or (
                    est.main_propellant_kg <= ag.separation_min_main_propellant_kg + shutdown_res
                    and v_inert_now >= 1_635.0
                )
                or est.pos_enu_m[2] >= ag.separation_altitude_m + 0.75 * ag.separation_tolerance_m
            ):
                self.cutoff_pending = True
        else:
            sep_min_prop_target = ag.separation_min_main_propellant_kg
            if (
                est.main_propellant_kg <= ag.separation_min_main_propellant_kg + shutdown_res
                or est.pos_enu_m[2] >= ag.separation_altitude_m + 0.75 * ag.separation_tolerance_m
            ):
                self.cutoff_pending = True

        # 3D Gravity Turn: pitch in XZ corridor + inertial zero-crossrange centering along Y
        flow_x = atan2(float(v_air[0]), max(float(v_air[2]), 1.0)) if speed > 20.0 else 0.0
        flow_y = atan2(float(v_air[1]), max(float(v_air[2]), 1.0)) if speed > 20.0 else 0.0
        flow_inert_x = (
            atan2(est.vel_enu_m_s[0], max(est.vel_enu_m_s[2], 1.0))
            if est.vel_enu_m_s[2] > 20.0
            else 0.0
        )
        flow_inert_y = (
            atan2(est.vel_enu_m_s[1], max(est.vel_enu_m_s[2], 1.0))
            if est.vel_enu_m_s[2] > 20.0
            else 0.0
        )

        wx_1k, wy_1k, _ = self.forecast_wind.at(1_000.0, 0.0)
        wx_4k, wy_4k, _ = self.forecast_wind.at(4_000.0, 0.0)
        kick_deg = max(3.5, min(9.5, s.final_pitch_deg + 0.16 * (0.65 * wx_1k + 0.35 * wx_4k)))
        kick_y_deg = max(-3.0, min(3.0, 0.16 * (0.65 * wy_1k + 0.35 * wy_4k)))
        t_kick_end = s.pitch_start_s + s.pitch_duration_s
        if est.time_s < s.pitch_start_s:
            pitch_x = 0.0
            pitch_y = 0.0
        elif est.time_s < t_kick_end:
            u_k = _smoothstep((est.time_s - s.pitch_start_s) / s.pitch_duration_s)
            pitch_x = radians(kick_deg) * u_k
            pitch_y = radians(kick_y_deg) * u_k
        else:
            gt_blend = _smoothstep((est.time_s - t_kick_end) / 4.0)
            pitch_x = (1.0 - gt_blend) * radians(kick_deg) + gt_blend * flow_inert_x
            pitch_y = (1.0 - gt_blend) * radians(kick_y_deg) + gt_blend * flow_inert_y
            if est.pos_enu_m[2] > 10_000.0:
                rem = max(0.0, est.main_propellant_kg - ag.separation_min_main_propellant_kg)
                tot_burn = max(
                    self.vehicle.propellant.main_propellant_kg
                    - ag.separation_min_main_propellant_kg,
                    1.0,
                )
                frac = max(0.0, min(1.0, 1.0 - rem / tot_burn))
                target_gt = radians(38.5) * frac
                trim_lim = radians(2.3 + 14.7 * _smoothstep((13_500.0 - q_pa) / 8_500.0))
                pitch_x = flow_inert_x + max(
                    -trim_lim, min(trim_lim, 1.60 * (target_gt - flow_inert_x))
                )
                pitch_y = flow_inert_y + max(-trim_lim, min(trim_lim, 1.60 * (0.0 - flow_inert_y)))

        # Protect 3D Max-Q structural q * total_alpha bound with circular norm clamp
        if q_pa > 100.0:
            dep = _smoothstep((est.vel_enu_m_s[2] - 40.0) / 60.0)
            ang_budget = 16.0 - 13.5 * _smoothstep((q_pa - 2_500.0) / 5_500.0)
            allow = radians(
                min(ang_budget, 0.55 * self.vehicle.safety.max_q_alpha_kpa_deg * 1_000.0 / q_pa)
            )
            dx_a = pitch_x - dep * flow_x
            dy_a = pitch_y - dep * flow_y
            d_norm = hypot(dx_a, dy_a)
            if d_norm > allow and d_norm > 1e-12:
                sc_a = allow / d_norm
                pitch_x = dep * flow_x + dx_a * sc_a
                pitch_y = dep * flow_y + dy_a * sc_a

        time_to_res = (est.main_propellant_kg - sep_min_prop_target - shutdown_res) / max(
            full_flow, 1.0
        )
        if (
            self.sep_align_started_s is None
            and est.vel_enu_m_s[2] > 100.0
            and time_to_res <= s.separation_alignment_preview_s
        ):
            self.sep_align_started_s = est.time_s
        if self.sep_align_started_s is not None:
            blend = _smoothstep(
                (est.time_s - self.sep_align_started_s) / s.separation_alignment_blend_s
            )
            pitch_x += blend * (flow_x - pitch_x)
            pitch_y += blend * (flow_y - pitch_y)

        ascent_mask_7: tuple[bool, bool, bool, bool, bool, bool, bool] | None = None
        if self.cutoff_pending:
            pitch_x, pitch_y = flow_x, flow_y
            throttle = 0.0
            active_engines = 0
            if self.fdir is not None:
                ascent_mask_7 = (False, False, False, False, False, False, False)
        else:
            min_m = mass - 0.01 * self.vehicle.launch_mass_kg
            thrust = min(full_t, 3.55 * G0 * min_m / 1.03)
            if q_pa > 0.55 * self.vehicle.safety.max_q_pa and speed > 10.0:
                _, _, rho_hi = self.forecast_atmo.at(est.pos_enu_m[2] + 20.0)
                grad = (log(max(rho_hi, 1e-30)) - log(max(rho, 1e-30))) / 20.0
                rob_q = 0.77 * self.vehicle.safety.max_q_pa / 1.10
                cap = sqrt(2.0 * rob_q / max(rho, 1e-9))
                des = -0.5 * cap * grad * est.vel_enu_m_s[2] + 0.7 * (cap - speed)
                free_v = (
                    (float(bf_w[0]) / mass) * float(v_air[0])
                    + (float(bf_w[1]) / mass) * float(v_air[1])
                    + (float(bf_w[2]) / mass - self._gravity(est.pos_enu_m[2])) * float(v_air[2])
                ) / speed
                d_vec = np.array(
                    [sin(pitch_x), sin(pitch_y) * cos(pitch_x), cos(pitch_x) * cos(pitch_y)]
                )
                d_vec /= float(np.linalg.norm(d_vec))
                proj = float(d_vec @ v_air) / speed
                if proj > 0.05:
                    thrust = min(thrust, max(0.0, mass * (des - free_v) / proj))
            if self.fdir is not None:
                ascent_mask_7 = self.fdir.healthy_ascent_mask()
                active_engines = sum(1 for v in ascent_mask_7 if v)
            else:
                active_engines = self.vehicle.propulsion.engine_count
            throttle = self.vehicle.propulsion.quantize_throttle(
                thrust / max(full_t, 1.0), active_engines
            )

        dir_des = np.array(
            [sin(pitch_x), sin(pitch_y) * cos(pitch_x), cos(pitch_x) * cos(pitch_y)],
            dtype=np.float64,
        )
        q_des = pointing_dir_to_quat_wxyz(dir_des)
        gim, rcs, fins = self._attitude_allocate_3d(
            est,
            q_des,
            throttle,
            rate_limit_deg_s=8.0,
            accel_limit_deg_s2=2.5,
            allow_fin=False,
            use_xy_torque_bias=(self.gnc_v2.use_structural_filter and (not est.on_pad)),
            engine_mask_7=ascent_mask_7,
        )
        self.last_diag = GuidanceDiagnostics3D(
            phase="ascent",
            tilt_target_deg=degrees(tilt_angle_rad(q_des)),
            roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
        )
        return ActuatorCommand3D(
            throttle_cmd=throttle,
            active_engines=active_engines,
            gimbal_rad=gim,
            rcs_torque_nm=rcs,
            fin_deflection_rad=fins,
            deploy_fins=False,
            deploy_gear=False,
            phase="ascent",
            engine_mask_7=ascent_mask_7,
            mission_mode=(self.fdir.mission_mode if self.fdir is not None else "RTLS"),
            fdir_state=(
                self._fdir_status.fdir_state if self._fdir_status is not None else "NOMINAL"
            ),
        )

    def _landing_burn_guidance(
        self, est: EstimatedState3D, mass: float, cg: float, h: float
    ) -> ActuatorCommand3D:
        s = self.settings
        _, p_pa, _ = self.forecast_atmo.at(est.pos_enu_m[2])
        g_loc = self._gravity(est.pos_enu_m[2])
        vz_down = max(0.0, -est.vel_enu_m_s[2])

        # Smooth causal horizontal/vertical acceleration bias compensation
        blend_b = _smoothstep((950.0 - h) / 250.0)
        b_vec = np.asarray(est.accel_bias_world_m_s2, dtype=np.float64)
        b_norm = float(np.linalg.norm(b_vec))
        b_sc = min(1.0, 3.5 / max(b_norm, 1e-9)) * blend_b
        b_xy = b_sc * b_vec[0:2]
        b_z = float(b_sc * b_vec[2])
        a_bxy = 1.0 - exp(-s.control_dt_s / (0.60 if self.in_vertical_body else 1.50))
        self._land_bxy_smooth += a_bxy * (b_xy - self._land_bxy_smooth)
        self._land_bz_smooth += (1.0 - exp(-s.control_dt_s / 0.50)) * (b_z - self._land_bz_smooth)

        t_min_1, _ = compute_engine_thrust_3d(
            self.vehicle.propulsion.throttle_bounds(1)[0], p_pa, self.vehicle
        )
        t_max_1, _ = compute_engine_thrust_3d(
            self.vehicle.propulsion.throttle_bounds(1)[1], p_pa, self.vehicle
        )
        m_light = max(self.vehicle.booster_dry_mass_kg, mass - 0.01 * self.vehicle.launch_mass_kg)
        a_min_1_chk = max(0.5, t_min_1 / mass - g_loc)
        a_min_1_w = max(a_min_1_chk, 1.03 * t_min_1 / m_light - g_loc)
        a_min_1_eff = a_min_1_w + 0.25
        a_max_1_net = max(a_min_1_eff + 2.5, t_max_1 / mass - g_loc)
        a_lo = min(a_max_1_net - 3.0, a_min_1_eff + 0.16)
        a_hi = min(
            a_max_1_net - 2.0,
            max(a_lo + 2.0, a_min_1_chk + 0.58 * (a_max_1_net - a_min_1_chk)),
        )
        h_shape = 680.0
        h_vref = max(0.0, h)
        h_c = min(h_vref, h_shape)
        v_td = (
            0.40
            if self.gnc_v2.use_online_mpc
            else (0.50 if self.gnc_v2.use_strapdown_mekf else s.landing_touchdown_speed_m_s)
        )
        v_stage_b_cap = sqrt(
            v_td * v_td
            + 2.0 * a_lo * h_vref
            + ((a_hi - a_lo) / h_shape) * h_c * h_c
            + 2.0 * a_hi * max(0.0, h_vref - h_shape)
        )

        is_abort = self.fdir is not None and self.fdir.mission_mode == "DOWNRANGE_SPLASHDOWN_ABORT"
        land_mask_7: tuple[bool, bool, bool, bool, bool, bool, bool] | None = None
        if self.fdir is not None and self._land_mask_7 is None:
            self._land_mask_7, _ = self.fdir.select_3eng_mask()

        # M3 FDIR Engine-Out mid-landing reconfiguration (CDR §3.4: 3->1 if h > h_stop_1 + 80 m,
        # else hold 2-engine trimmed burn only until 1-engine stopping margin clears or h <= 350 m)
        if (
            self.fdir is not None
            and (not self.landing_single_engine)
            and self._land_mask_7 is not None
        ):
            has_failed_in_mask = any(
                self._land_mask_7[i] and self.fdir.engine_failed[i] for i in range(7)
            )
            in_asym_2eng = sum(1 for v in self._land_mask_7 if v) == 2 and bool(
                self._land_mask_7[0]
            )
            if has_failed_in_mask or in_asym_2eng:
                h_stop_1 = vz_down * vz_down / (2.0 * max(a_max_1_net - 1.0, 2.0)) + 60.0
                margin_1 = 60.0 if in_asym_2eng else 80.0
                if h > h_stop_1 + margin_1 or h <= 350.0:
                    self.landing_single_engine = True
                elif has_failed_in_mask:
                    self._land_mask_7, _ = self.fdir.select_3eng_mask(
                        current_mask=self._land_mask_7, allow_2eng_trim=True
                    )

        # Monotone 3 -> 1 engine switch + one-shot Clarabel 3D GFOLD SOCP reference plan
        if not self.landing_single_engine and (
            vz_down <= min(s.landing_v_switch_m_s, v_stage_b_cap) or h <= 200.0
        ):
            self.landing_single_engine = True
            if s.use_socp_reference and (not is_abort) and h > 30.0:
                prob = Gfold3DProblem(
                    pos0_enu_m=(est.pos_enu_m[0], est.pos_enu_m[1], max(h, 5.0)),
                    vel0_enu_m_s=(
                        est.vel_enu_m_s[0],
                        est.vel_enu_m_s[1],
                        min(est.vel_enu_m_s[2], -2.0),
                    ),
                    a_min_m_s2=t_min_1 / mass,
                    a_max_m_s2=t_max_1 / mass,
                    gravity_m_s2=g_loc,
                    v_touchdown_m_s=v_td,
                    max_tilt_deg=s.landing_divert_max_tilt_deg,
                )
                self.socp_solution = solve_gfold_3d_socp(prob)
                self.socp_plan_start_s = est.time_s

        if self.fdir is not None:
            if self.landing_single_engine:
                land_mask_7, active_engines = self.fdir.select_1eng_mask()
                self._land_mask_7 = land_mask_7
            else:
                assert self._land_mask_7 is not None
                land_mask_7 = self._land_mask_7
                active_engines = sum(1 for v in land_mask_7 if v)
                if active_engines <= 0:
                    land_mask_7, active_engines = self.fdir.select_1eng_mask()
                    self._land_mask_7 = land_mask_7
        else:
            active_engines = 1 if self.landing_single_engine else 3
        t_min, _ = compute_engine_thrust_3d(
            self.vehicle.propulsion.throttle_bounds(active_engines)[0], p_pa, self.vehicle
        )
        t_max, _ = compute_engine_thrust_3d(
            self.vehicle.propulsion.throttle_bounds(active_engines)[1], p_pa, self.vehicle
        )
        full_t, _ = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)

        r_wb = quat_to_rotmat(est.quat_wxyz)
        bf_w, btau_b_now, _, _, _, _, _ = compute_body_aero_3d(
            altitude_m=est.pos_enu_m[2],
            vel_enu_m_s=est.vel_enu_m_s,
            rot_wb=r_wb,
            omega_body_rad_s=est.omega_body_rad_s,
            cg_station_m=cg,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=est.time_s,
            vehicle=self.vehicle,
        )
        ff_w, ftau_b_now, _ = compute_fins_aero_3d(
            altitude_m=est.pos_enu_m[2],
            vel_enu_m_s=est.vel_enu_m_s,
            rot_wb=r_wb,
            omega_body_rad_s=est.omega_body_rad_s,
            cg_station_m=cg,
            fin_deflection_rad=est.fin_deflection_rad,
            fin_extension=est.fin_extension,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=est.time_s,
            vehicle=self.vehicle,
        )
        aero_f = bf_w + ff_w
        tau_aero_b_now = btau_b_now + ftau_b_now
        lever_trim = max(cg - self.vehicle.geometry.nozzle_station_m, 1.0)
        max_gim_rad = radians(self.vehicle.propulsion.gimbal_limit_deg)
        a_cor = self._coriolis(est.pos_enu_m[2], est.vel_enu_m_s)

        # Current body +z_b tilt projections along ENU X and Y
        zb_w = r_wb[:, 2]
        om_w = r_wb @ np.asarray(est.omega_body_rad_s, dtype=np.float64)
        tilt_x = atan2(float(zb_w[0]), max(float(zb_w[2]), 0.1))
        tilt_y = atan2(float(zb_w[1]), max(float(zb_w[2]), 0.1))
        # d(zb)/dt = om_w x zb -> rate of tilt_x is +om_wy, rate of tilt_y is -om_wx
        rate_tx = float(om_w[1])
        rate_ty = -float(om_w[0])

        a_avg_b = a_lo + 0.5 * (a_hi - a_lo) * min(h_vref / h_shape, 1.0)
        vertical_body = False
        vb_dir_xy = np.zeros(2, dtype=np.float64)
        mpc_ref: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None = None

        if not self.landing_single_engine:
            v_sw = min(s.landing_v_switch_m_s, v_stage_b_cap)
            t_go = max(0.0, (vz_down - v_sw) / 28.0) + max(1.0, (v_sw - v_td) / max(a_avg_b, 1.5))
            k2 = (a_hi - a_lo) / h_shape
            rhs_sw = max(0.0, v_sw * v_sw - v_td * v_td)
            h_sw = (-a_lo + sqrt(a_lo * a_lo + k2 * rhs_sw)) / max(k2, 1e-6)
            dh = max(h - h_sw, 25.0)
            az_net = max(
                t_min / mass - g_loc,
                min(2.85 * G0, (vz_down * vz_down - v_sw * v_sw) / (2.0 * dh)),
            )
            a_t_sm = max(t_min / mass, min(t_max / mass, az_net + g_loc))
            if is_abort:
                # Downrange ocean splashdown abort: null horizontal drift only (no pad divert)
                a_xy_cmd = np.array(
                    [-0.45 * est.vel_enu_m_s[0], -0.45 * est.vel_enu_m_s[1]], dtype=np.float64
                )
            else:
                p_lead, v_lead = predict_zero_tilt_state_3d(
                    (est.pos_enu_m[0], est.pos_enu_m[1]),
                    (est.vel_enu_m_s[0], est.vel_enu_m_s[1]),
                    (tilt_x, tilt_y),
                    (rate_tx, rate_ty),
                    a_t_sm,
                    radians(2.8),
                    x_horizon_cap_s=2.4,
                )
                t_align = max(3.2, t_go - 2.8)
                a_xy_cmd = np.array(
                    [
                        -6.5 * p_lead[0] / (t_align * t_align) - 5.5 * v_lead[0] / t_align,
                        -6.5 * p_lead[1] / (t_align * t_align) - 5.5 * v_lead[1] / t_align,
                    ],
                    dtype=np.float64,
                )
            a_xy_mag = float(np.linalg.norm(a_xy_cmd))
            if a_xy_mag > 7.5:
                a_xy_cmd *= 7.5 / a_xy_mag
            max_tilt = radians(s.landing_divert_max_tilt_deg)
        else:
            a_ff = a_lo + (a_hi - a_lo) * (h_c / h_shape)
            if vz_down <= 0.78 * v_td:
                az_net = -3.5
            else:
                k_v = (
                    (2.7 if h <= 15.0 else 2.1)
                    if vz_down > v_stage_b_cap
                    else (1.8 if h > 15.0 else 1.45)
                )
                az_net = a_ff + k_v * (vz_down - v_stage_b_cap)
            az_net = max(-3.5, min(a_max_1_net, az_net))
            a_t_sm = max(t_min / mass, min(t_max / mass, a_ff + g_loc - float(aero_f[2]) / mass))

            # Causal sign-monotone 1st-order attitude-lag lead state for Online GFOLD MPC:
            # projects actual current net horizontal acceleration (thrust tilt + aero + bias + Cor)
            # over the CSI attitude loop lead horizon without `th_tr` false-reversal bias under
            # strong opposing wind shear.
            p_lead_mpc = (est.pos_enu_m[0], est.pos_enu_m[1])
            v_lead_mpc = (est.vel_enu_m_s[0], est.vel_enu_m_s[1])
            if (
                self.online_mpc is not None
                and (not self.in_vertical_body)
                and (not is_abort)
                and h_vref > 12.0
            ):
                tau_lead_mpc = (2.15 if self.gnc_v2.use_structural_filter else 0.55) * min(
                    1.0, max(0.65, (h_vref - 20.0) / 60.0)
                )
                # Include mid-horizon angular-rate projection (capped +/-20 deg) so CSI attitude
                # inertia during large landing-burn EO / blackout diverts unwinds without overshoot.
                dt_mid = min(0.70, 0.40 * tau_lead_mpc)
                tx_mid = max(-radians(20.0), min(radians(20.0), tilt_x + rate_tx * dt_mid))
                ty_mid = max(-radians(20.0), min(radians(20.0), tilt_y + rate_ty * dt_mid))
                a_now_xy = (
                    a_t_sm * sin(tx_mid)
                    + float(aero_f[0]) / mass
                    + float(self._land_bxy_smooth[0])
                    + float(a_cor[0]),
                    a_t_sm * sin(ty_mid)
                    + float(aero_f[1]) / mass
                    + float(self._land_bxy_smooth[1])
                    + float(a_cor[1]),
                )
                p_lead_mpc = (
                    est.pos_enu_m[0]
                    + est.vel_enu_m_s[0] * tau_lead_mpc
                    + 0.5 * a_now_xy[0] * (tau_lead_mpc * tau_lead_mpc),
                    est.pos_enu_m[1]
                    + est.vel_enu_m_s[1] * tau_lead_mpc
                    + 0.5 * a_now_xy[1] * (tau_lead_mpc * tau_lead_mpc),
                )
                v_lead_mpc = (
                    est.vel_enu_m_s[0] + a_now_xy[0] * tau_lead_mpc,
                    est.vel_enu_m_s[1] + a_now_xy[1] * tau_lead_mpc,
                )
                # 2 Hz online N=12 log-mass GFOLD MPC targeting Vertical Body handoff (h=30 m);
                # when a severe EO/blackout leaves r_xy >= 4.0 m at h=40 m, keep MPC active down
                # to h=22 m (z_tgt=15 m) so terminal centering finishes smoothly inside MPC
                # instead of discontinuously jumping to v1 guidance!
                r_xy_pre = hypot(est.pos_enu_m[0], est.pos_enu_m[1])
                if (h_vref <= 40.0 and r_xy_pre < 4.0) or h_vref <= 22.0:
                    self._mpc_terminal_latched = True
                if (
                    h_vref > 22.0
                    and (not self._mpc_terminal_latched)
                    and est.time_s
                    >= self.online_mpc.last_solve_time_s
                    + self.online_mpc.cfg.replan_period_s
                    - 1e-9
                ):
                    a_aero_est = aero_f / mass + np.array(
                        [
                            float(self._land_bxy_smooth[0]),
                            float(self._land_bxy_smooth[1]),
                            float(self._land_bz_smooth),
                        ],
                        dtype=np.float64,
                    )
                    z_tgt_mpc = 30.0 if h_vref > 42.0 else max(14.0, h_vref - 12.0)
                    v_tgt_mpc = min(
                        sqrt(
                            v_td * v_td
                            + 2.0 * a_lo * z_tgt_mpc
                            + ((a_hi - a_lo) / h_shape) * z_tgt_mpc * z_tgt_mpc
                        ),
                        max(2.5, vz_down + 1.0),
                    )
                    # Cap MPC planned thrust at smooth 1-engine tracking envelope (~72% max)
                    # so GFOLD avoids cliff-dive bang-bang & leaves ample tracking headroom.
                    t_max_mpc = max(
                        t_min_1 + 25_000.0,
                        min(t_max_1, mass * (g_loc + a_hi + 1.0)),
                    )
                    self.online_mpc.solve(
                        time_s=est.time_s,
                        pos0_enu_m=(p_lead_mpc[0], p_lead_mpc[1], h_vref),
                        vel0_enu_m_s=(
                            v_lead_mpc[0],
                            v_lead_mpc[1],
                            min(est.vel_enu_m_s[2], -0.5),
                        ),
                        mass0_kg=mass,
                        dry_mass_kg=self.vehicle.booster_dry_mass_kg,
                        t_min_n=t_min_1,
                        t_max_n=t_max_mpc,
                        isp_eff_s=self.vehicle.propulsion.sea_level_isp_s,
                        gravity_m_s2=g_loc,
                        v_touchdown_m_s=v_tgt_mpc,
                        a_aero_now_enu_m_s2=(
                            float(a_aero_est[0]),
                            float(a_aero_est[1]),
                            float(a_aero_est[2]),
                        ),
                        ground_effect_coeff=(
                            self.gnc_v2.ground_effect_nominal_coeff
                            if self.gnc_v2.use_ground_effect_feedforward
                            else 0.0
                        ),
                        z_target_m=z_tgt_mpc,
                    )
                mpc_ref = self.online_mpc.sample_reference(est.time_s)

            v_xy_mag_now = hypot(est.vel_enu_m_s[0], est.vel_enu_m_s[1])
            r_xy_now = hypot(est.pos_enu_m[0], est.pos_enu_m[1])
            if self.online_mpc is not None and not is_abort:
                # CDR C6 unconditional safety floor handoff to Vertical Body (30 m capture gate);
                # keeps MPC smoothly active down to h=14..22 m only if r_xy >= 4.0 m after EO.
                tilt_now_deg = degrees(tilt_angle_rad(est.quat_wxyz))
                vb_trigger = (
                    (
                        h_vref <= 30.0
                        and tilt_now_deg < 2.5
                        and v_xy_mag_now < 3.5
                        and r_xy_now < 4.0
                    )
                    or (h_vref <= 14.0)
                    or (h_vref <= 36.0 and mpc_ref is None)
                )
            else:
                vb_trigger = h_vref <= 10.0 or (
                    h_vref <= s.landing_vertical_body_height_m and v_xy_mag_now <= 4.5
                )
            if vb_trigger:
                if not self.in_vertical_body:
                    self._vb_started_s = est.time_s
                self.in_vertical_body = True
            vertical_body = self.in_vertical_body
            if vertical_body:
                mpc_ref = None

            if vertical_body:
                lever = lever_trim
                t_rem = 2.0 * h_vref / max(vz_down + v_td, 1.0)
                bf0_w, _, _, _, _, _, _ = compute_body_aero_3d(
                    altitude_m=est.pos_enu_m[2],
                    vel_enu_m_s=est.vel_enu_m_s,
                    rot_wb=np.eye(3),
                    omega_body_rad_s=(0.0, 0.0, 0.0),
                    cg_station_m=cg,
                    atmo_lut=self.forecast_atmo,
                    wind_lut=self.forecast_wind,
                    time_s=est.time_s,
                    vehicle=self.vehicle,
                )
                a_wind_xy = bf0_w[0:2] / mass + self._land_bxy_smooth + a_cor[0:2]
                raw_wt = np.clip(-a_wind_xy / max(a_t_sm, 5.0), -radians(1.8), radians(1.8))
                self._vb_wind_trim_xy += (1.0 - exp(-s.control_dt_s / 0.85)) * (
                    raw_wt - self._vb_wind_trim_xy
                )

                p_xy = np.array([est.pos_enu_m[0], est.pos_enu_m[1]], dtype=np.float64)
                v_xy = np.array([est.vel_enu_m_s[0], est.vel_enu_m_s[1]], dtype=np.float64)
                r_xy = float(np.linalg.norm(p_xy))
                # Gentle off-pad centering lean (<= 2.40 deg, smoothly zeroed before final braking;
                # v2 CSI loop uses longer 2.4 s velocity lead & h > 5 m fade so body unwinds early)
                if self.gnc_v2.use_structural_filter:
                    p_stop_xy = p_xy + 2.40 * v_xy
                    r_stop = float(np.linalg.norm(p_stop_xy))
                    vb_mid_env = (
                        _smoothstep((h_vref - 3.2) / 5.5)
                        * _smoothstep((r_xy - 2.0) / 2.0)
                        * _smoothstep((r_stop - 1.5) / 2.0)
                    )
                else:
                    p_stop_xy = p_xy + 0.95 * v_xy
                    r_stop = float(np.linalg.norm(p_stop_xy))
                    vb_mid_env = (
                        _smoothstep((h_vref - 2.2) / 3.4)
                        * _smoothstep((r_xy - 2.0) / 2.0)
                        * _smoothstep((r_stop - 1.2) / 1.8)
                    )
                th_vb_mid = -radians(0.75) * p_stop_xy * vb_mid_env
                mid_nrm = float(np.linalg.norm(th_vb_mid))
                if mid_nrm > radians(2.40) and mid_nrm > 1e-12:
                    th_vb_mid *= radians(2.40) / mid_nrm
                # In v2 dual-tank configuration, the higher LOX-dominant CG (~9.1 m vs 7.3 m)
                # increases post-touchdown 4-leg rocking inertia; taper upwind body lean smoothly
                # before touchdown. Under M2 Dryden turbulence + CSI gains, pre-trim to the causal
                # 4-leg static wind-lean equilibrium (capped <= 1.45 deg) to eliminate the initial
                # downwind rocking step excitation while TVC+RCS holds lateral pad position.
                if self.gnc_v2.use_structural_filter:
                    wt_taper = _smoothstep((h_vref - 12.0) / 10.0)
                    gear = self.vehicle.landing_gear
                    arm_cp = (
                        self.vehicle.aerodynamics.pressure_center_descent_m - gear.foot_station_m
                    )
                    k_rot = max(
                        2.0 * gear.per_leg_stiffness_n_m * (gear.half_span_m**2)
                        - mass * g_loc * (cg - gear.foot_station_m),
                        250_000.0,
                    )
                    # Causal surface wind lateral force (forecast + damped MEKF wind offset so
                    # brief pre-touchdown Dryden lulls do not under-trim steady mean wind lean)
                    f_wind_xy = bf0_w[0:2] + 0.5 * mass * self._land_bxy_smooth
                    th_eq_xy = (arm_cp / k_rot) * f_wind_xy
                    eq_nrm = float(np.linalg.norm(th_eq_xy))
                    eq_cap = radians(1.45)
                    if eq_nrm > eq_cap and eq_nrm > 1e-12:
                        th_eq_xy *= eq_cap / eq_nrm
                    vb_dir_xy = (
                        self._vb_wind_trim_xy * wt_taper + th_eq_xy * (1.0 - wt_taper) + th_vb_mid
                    )
                elif self.gnc_v2.use_dual_tank_mass_props:
                    wt_taper = _smoothstep((h_vref - 2.5) / 6.5)
                    vb_dir_xy = self._vb_wind_trim_xy * wt_taper + th_vb_mid
                else:
                    wt_taper = 1.0
                    vb_dir_xy = self._vb_wind_trim_xy * wt_taper + th_vb_mid
                vb_nrm = float(np.linalg.norm(vb_dir_xy))
                if vb_nrm > radians(2.55) and vb_nrm > 1e-12:
                    vb_dir_xy *= radians(2.55) / vb_nrm

                ax_lim_rcs = (
                    0.90 * self.vehicle.propulsion.rcs_torque_limit_nm / max(lever * mass, 1.0)
                )
                if self.gnc_v2.use_structural_filter:
                    v_ground_safe = 0.10 + 0.65 * ax_lim_rcs * max(t_rem - 0.35, 0.0)
                    v_deck_cap = 0.08 + 3.2 * _smoothstep((h_vref - 0.5) / 1.8)
                    closing = min(
                        0.65 * r_xy,
                        sqrt(2.0 * 0.55 * ax_lim_rcs * r_xy),
                        v_ground_safe,
                        v_deck_cap,
                    )
                else:
                    v_ground_safe = 0.14 + 0.85 * ax_lim_rcs * max(t_rem - 0.25, 0.0)
                    v_deck_cap = 0.08 + 3.4 * _smoothstep((h_vref - 0.5) / 1.65)
                    closing = min(
                        0.82 * r_xy,
                        sqrt(2.0 * 0.85 * ax_lim_rcs * r_xy),
                        v_ground_safe,
                        v_deck_cap,
                    )
                des_v_xy = (
                    np.zeros(2, dtype=np.float64)
                    if is_abort
                    else ((-p_xy / r_xy) * closing if r_xy > 1e-4 else np.zeros(2))
                )
                if is_abort:
                    # Keep gentle wind-trim in ocean abort so surface wind cannot push vh > 0.8 m/s
                    vb_dir_xy = self._vb_wind_trim_xy * wt_taper
                k_vb = 3.2 + 0.8 * _smoothstep((h_vref - 1.5) / 4.0)
                a_xy_cmd = k_vb * (des_v_xy - v_xy)
                a_nrm = float(np.linalg.norm(a_xy_cmd))
                if a_nrm > 3.5:
                    a_xy_cmd *= 3.5 / a_nrm
                max_tilt = radians(s.landing_divert_max_tilt_deg)
            else:
                t_kin = 2.0 * h_vref / max(vz_down + v_td, 1.0)
                a_dec_eff = max(1.5, 0.5 * (a_lo + a_ff) + max(0.0, 0.55 * self._land_bz_smooth))
                t_dec = max(0.0, vz_down - v_td) / a_dec_eff
                t_rem_b = min(28.0, max(1.0, 0.5 * (t_kin + t_dec)))
                t_align = max(2.75, t_rem_b - 3.85)
                a_att_b = (
                    radians(1.40)
                    if self.gnc_v2.use_structural_filter
                    else radians(2.44 + 0.34 * _smoothstep((self._land_bz_smooth - 0.85) / 0.22))
                )

                # Quasi-static 2-axis TVC aerodynamic trim:
                #   tvc_tau = [lever*T*gx, -lever*T*gy, 0] cancels tau_aero_b_now
                #   -> gy_trim = +tau_y / (lever*T), gx_trim = -tau_x / (lever*T)
                denom_gt = max(lever_trim * mass * a_t_sm, 1.0)
                lim_gt = 0.85 * sin(max_gim_rad)
                gim_trim_x = asin(max(-lim_gt, min(lim_gt, float(tau_aero_b_now[1]) / denom_gt)))
                gim_trim_y = asin(max(-lim_gt, min(lim_gt, -float(tau_aero_b_now[0]) / denom_gt)))

                th_trim_xy = np.clip(
                    (-aero_f[0:2] / mass - self._land_bxy_smooth - a_cor[0:2]) / max(a_t_sm, 5.0),
                    -0.25,
                    0.25,
                )
                tx_rel = tilt_x - float(th_trim_xy[0])
                ty_rel = tilt_y - float(th_trim_xy[1])
                if tx_rel * gim_trim_x < 0.0:
                    tx_rel_eff = tx_rel + 0.80 * gim_trim_x
                    if tx_rel_eff * tx_rel < 0.0:
                        tx_rel_eff = 0.0
                else:
                    tx_rel_eff = tx_rel
                if ty_rel * gim_trim_y < 0.0:
                    ty_rel_eff = ty_rel + 0.80 * gim_trim_y
                    if ty_rel_eff * ty_rel < 0.0:
                        ty_rel_eff = 0.0
                else:
                    ty_rel_eff = ty_rel

                p_lead, v_lead = predict_zero_tilt_state_3d(
                    (est.pos_enu_m[0], est.pos_enu_m[1]),
                    (est.vel_enu_m_s[0], est.vel_enu_m_s[1]),
                    (tx_rel_eff, ty_rel_eff),
                    (rate_tx, rate_ty),
                    a_t_sm,
                    a_att_b,
                    x_horizon_cap_s=(3.0 if self.gnc_v2.use_structural_filter else 2.4),
                )
                r_err = max(hypot(est.pos_enu_m[0], est.pos_enu_m[1]), hypot(p_lead[0], p_lead[1]))
                v_err = hypot(v_lead[0], v_lead[1])
                tilt_frac = _smoothstep(
                    (h_vref - s.landing_vertical_body_height_m)
                    / max(s.landing_vertical_align_height_m - s.landing_vertical_body_height_m, 1.0)
                )
                tilt_floor = (
                    0.08
                    + 0.42 * _smoothstep((r_err - 1.5) / 7.5)
                    + 0.24 * _smoothstep((v_err - 0.6) / 1.5)
                )
                max_tilt = radians(s.landing_divert_max_tilt_deg) * max(tilt_floor, tilt_frac)
                if is_abort:
                    # Zero-tilt lead velocity nulling -> monotonic vh -> 0 with tilt < 1 deg!
                    a_xy_cmd = np.array([-0.85 * v_lead[0], -0.85 * v_lead[1]], dtype=np.float64)
                    max_tilt = radians(8.0) * _smoothstep((h_vref - 32.0) / 45.0)
                else:
                    a_xy_cmd = np.array(
                        [
                            -7.4 * p_lead[0] / (t_align * t_align) - 6.2 * v_lead[0] / t_align,
                            -7.4 * p_lead[1] / (t_align * t_align) - 6.2 * v_lead[1] / t_align,
                        ],
                        dtype=np.float64,
                    )
                max_lat = a_t_sm * sin(max_tilt)
                a_nrm = float(np.linalg.norm(a_xy_cmd))
                if a_nrm > max_lat and a_nrm > 1e-9:
                    a_xy_cmd *= max_lat / a_nrm

        # Assemble required 3D thrust vector (2 Hz GFOLD MPC upstream of v1 gimbal-trim decoupler)
        if mpc_ref is not None and (not vertical_body):
            p_ref, v_ref, a_ref = mpc_ref
            # Track attitude-lead state + exact 50 Hz crosswind/bias decoupling (zero wind drift!)
            p_err_xy = p_ref[0:2] - np.array([p_lead_mpc[0], p_lead_mpc[1]], dtype=np.float64)
            v_err_xy = v_ref[0:2] - np.array([v_lead_mpc[0], v_lead_mpc[1]], dtype=np.float64)
            a_xy_mpc = a_ref[0:2] + 0.35 * p_err_xy + 1.30 * v_err_xy
            prop_xy_mpc = a_xy_mpc - self._land_bxy_smooth - a_cor[0:2]
            f_xy_req = mass * prop_xy_mpc - aero_f[0:2]
            u_z_mpc = (
                float(a_ref[2])
                + 0.45 * (float(p_ref[2]) - h_vref)
                + 1.35 * (float(v_ref[2]) - est.vel_enu_m_s[2])
                - float(a_cor[2])
            )
            # Monotone Stage-B vertical-corridor guard: identical 75%/25% on nominal flights
            # (vz_down >= 0.88 * v_stage_b_cap), smoothly ramping Stage-B authority up to 100%
            # whenever receding-horizon MPC tries to over-brake vertically (eliminates hover-stall).
            u_z_v1 = max(
                t_min / mass,
                (az_net + g_loc - b_z - float(a_cor[2])) - float(aero_f[2]) / mass,
            )
            w_v1 = 0.25 + 0.75 * _smoothstep(
                (0.88 * v_stage_b_cap - vz_down) / max(0.22 * v_stage_b_cap, 1.0)
            )
            u_z = (1.0 - w_v1) * u_z_mpc + w_v1 * u_z_v1
            fz_req = max(t_min * 0.9, mass * u_z)
            r_err_mpc = max(r_xy_now, hypot(p_lead_mpc[0], p_lead_mpc[1]))
            max_tilt = max(
                radians(2.0) + radians(9.5) * _smoothstep((h_vref - 32.0) / 55.0),
                radians(2.0)
                + radians(5.5)
                * _smoothstep((r_err_mpc - 2.5) / 8.0)
                * _smoothstep((h_vref - 22.0) / 25.0),
            )
        else:
            prop_xy = a_xy_cmd - self._land_bxy_smooth - a_cor[0:2]
            if self.landing_single_engine and vz_down <= 0.78 * v_td:
                fz_req = t_min
            else:
                prop_z = az_net + g_loc - b_z - float(a_cor[2])
                fz_req = max(t_min * 0.9, mass * prop_z - float(aero_f[2]))
                if (
                    self.gnc_v2.use_ground_effect_feedforward
                    and self.landing_single_engine
                    and h < 18.0
                ):
                    h_noz_est = h + (
                        self.vehicle.geometry.nozzle_station_m
                        - self.vehicle.landing_gear.foot_station_m
                    )
                    ge_ff = ground_effect_nominal_lift_n(
                        nozzle_height_m=h_noz_est,
                        thrust_n=max(t_min, fz_req),
                        c_ge_nom=self.gnc_v2.ground_effect_nominal_coeff,
                    )
                    fz_req = max(t_min * 0.9, fz_req - ge_ff)
            f_xy_req = mass * prop_xy - aero_f[0:2]

        thrust_tilt_body_xy: tuple[float, float] | None = None
        if vertical_body:
            lever = max(cg - self.vehicle.geometry.nozzle_station_m, 1.0)
            vb_lat_scale = (
                _smoothstep((est.time_s - (self._vb_started_s or est.time_s)) / 0.45)
                if self.gnc_v2.use_structural_filter
                else 1.0
            )
            lat_lim = 0.85 * vb_lat_scale * self.vehicle.propulsion.rcs_torque_limit_nm / lever
            f_center = fz_req * np.array([sin(tilt_x), sin(tilt_y)], dtype=np.float64)
            df = f_xy_req - f_center
            df_nrm = float(np.linalg.norm(df))
            if df_nrm > lat_lim and df_nrm > 1e-9:
                df *= lat_lim / df_nrm
            f_xy_req = f_center + df
            f_req_w = np.array([float(f_xy_req[0]), float(f_xy_req[1]), fz_req], dtype=np.float64)
            f_req_b = r_wb.T @ f_req_w
            f_req_nrm = max(float(np.linalg.norm(f_req_b)), 1.0)
            thrust_tilt_body_xy = (
                asin(max(-0.5, min(0.5, float(f_req_b[0]) / f_req_nrm))),
                asin(max(-0.5, min(0.5, float(f_req_b[1]) / f_req_nrm))),
            )
            dir_des = np.array([float(vb_dir_xy[0]), float(vb_dir_xy[1]), 1.0], dtype=np.float64)
        else:
            f_xy_mag = float(np.linalg.norm(f_xy_req))
            max_f_xy = fz_req * sin(max_tilt) / max(cos(max_tilt), 0.5)
            if f_xy_mag > max_f_xy and f_xy_mag > 1e-9:
                f_xy_req *= max_f_xy / f_xy_mag
            dir_thrust = np.array(
                [float(f_xy_req[0]), float(f_xy_req[1]), fz_req], dtype=np.float64
            )
            # Decouple 2-axis aerodynamic TVC gimbal trim so actual (body + gimbal) thrust aligns
            # with `dir_thrust` without crosswind static bias (3D port of DM100 2D guidance).
            r_dir = quat_to_rotmat(pointing_dir_to_quat_wxyz(dir_thrust))
            _, btau_dir, _, _, _, _, _ = compute_body_aero_3d(
                altitude_m=est.pos_enu_m[2],
                vel_enu_m_s=est.vel_enu_m_s,
                rot_wb=r_dir,
                omega_body_rad_s=(0.0, 0.0, 0.0),
                cg_station_m=cg,
                atmo_lut=self.forecast_atmo,
                wind_lut=self.forecast_wind,
                time_s=est.time_s,
                vehicle=self.vehicle,
                drag_scale=self.cached_drag_scale,
            )
            denom_gd = max(lever_trim * mass * a_t_sm, 1.0)
            lim_gd = 0.85 * sin(max_gim_rad)
            g_trim_x = asin(max(-lim_gd, min(lim_gd, float(btau_dir[1]) / denom_gd)))
            g_trim_y = asin(max(-lim_gd, min(lim_gd, -float(btau_dir[0]) / denom_gd)))
            # Rotate thrust direction back into body centerline frame (subtract gimbal offset)
            t_unit = dir_thrust / max(float(np.linalg.norm(dir_thrust)), 1e-9)
            tx_b = atan2(float(t_unit[0]), max(float(t_unit[2]), 0.1)) - g_trim_x
            ty_b = atan2(float(t_unit[1]), max(float(t_unit[2]), 0.1)) - g_trim_y
            b_tilt_mag = hypot(tx_b, ty_b)
            if b_tilt_mag > max_tilt and b_tilt_mag > 1e-12:
                sc_b = max_tilt / b_tilt_mag
                tx_b *= sc_b
                ty_b *= sc_b
            dir_des = np.array(
                [sin(tx_b), sin(ty_b) * cos(tx_b), cos(tx_b) * cos(ty_b)],
                dtype=np.float64,
            )

        f_mag = sqrt(float(f_xy_req[0]) ** 2 + float(f_xy_req[1]) ** 2 + fz_req * fz_req)
        throttle = self.vehicle.propulsion.quantize_throttle(
            f_mag / max(full_t, 1.0), active_engines
        )

        meco3 = self.landing_single_engine and h_vref <= 0.25 and est.vel_enu_m_s[2] >= 0.05
        if (
            meco3
            and self.gnc_v2.use_strapdown_mekf
            and est.radar_foot_height_m is not None
            and est.radar_foot_height_m > 0.15
        ):
            meco3 = False
        if meco3:
            throttle = 0.0
            active_engines = 0
            thrust_tilt_body_xy = None

        lit_thr = 0.5 * self.vehicle.propulsion.throttle_bounds(1)[0]
        d_ret_lb = self._retro_dir(self._air_vel(est))
        el_lb = est.time_s - (self.landing_started_s or est.time_s)
        if est.engine_throttle < lit_thr:
            q_des = pointing_dir_to_quat_wxyz(d_ret_lb)
        elif self.gnc_v2.use_structural_filter and el_lb < 0.75:
            # Smoothly blend out of retrograde over 0.55 s so 3-engine divert step does not
            # stack on top of the 0.15 s atmospheric ignition nozzle side-load pulse!
            b_ign = _smoothstep((el_lb - 0.20) / 0.55)
            d_u = dir_des / max(float(np.linalg.norm(dir_des)), 1e-9)
            q_des = pointing_dir_to_quat_wxyz((1.0 - b_ign) * d_ret_lb + b_ign * d_u)
        else:
            q_des = pointing_dir_to_quat_wxyz(dir_des)

        land_rate_lim = 9.0
        if vertical_body and degrees(tilt_angle_rad(est.quat_wxyz)) < 2.9:
            if self.gnc_v2.use_structural_filter:
                land_rate_lim = 0.22 + 5.78 * _smoothstep((h_vref - 0.65) / 3.1)
            else:
                land_rate_lim = 0.25 + 5.75 * _smoothstep((h_vref - 0.65) / 3.1)
        vb_blend = (
            _smoothstep((est.time_s - (self._vb_started_s or est.time_s)) / 0.28)
            if vertical_body
            else 1.0
        )
        vb_acc = 2.30 + 0.50 * vb_blend
        if vertical_body:
            land_acc_lim = vb_acc
        elif self.gnc_v2.use_structural_filter:
            ign_acc_ramp = _smoothstep((el_lb - 0.30) / 0.55)
            # Ramp across 1-engine ignition spool-up [lit_thr -> lit_thr + 0.030] so single-engine
            # powered descent (thr ~ 0.055..0.14) receives full 2.80 deg/s^2 attitude authority.
            land_acc_lim = 1.35 + 1.45 * ign_acc_ramp * _smoothstep(
                (est.engine_throttle - lit_thr) / 0.030
            )
        else:
            land_acc_lim = 3.05

        gim, rcs, fins = self._attitude_allocate_3d(
            est,
            q_des,
            throttle,
            rate_limit_deg_s=land_rate_lim,
            accel_limit_deg_s2=land_acc_lim,
            allow_fin=(est.fin_extension >= 0.05 and h > 300.0),
            thrust_tilt_cmd_body_xy=thrust_tilt_body_xy,
            high_q_guard=False,
            accel_guard_deg_s2=land_acc_lim,
            use_xy_torque_bias=(not vertical_body) or self.gnc_v2.use_structural_filter,
            engine_mask_7=land_mask_7,
        )
        if est.engine_throttle < lit_thr:
            gim = (0.0, 0.0)

        mpc_used = self.online_mpc is not None and self.online_mpc.total_valid_solves > 0
        mpc_ms = self.online_mpc.max_solve_ms if self.online_mpc is not None else 0.0
        self.last_diag = GuidanceDiagnostics3D(
            phase="landing_burn",
            tilt_target_deg=degrees(tilt_angle_rad(q_des)),
            roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
            socp_solver_used=(
                mpc_used or (self.socp_solution is not None and self.socp_solution.feasible)
            ),
            socp_solver_status=(
                "mpc_optimal"
                if mpc_used
                else (self.socp_solution.status if self.socp_solution is not None else "unused")
            ),
            socp_solve_ms=(
                mpc_ms
                if mpc_used
                else (self.socp_solution.solve_ms if self.socp_solution is not None else 0.0)
            ),
            extra={
                "active_engines": active_engines,
                "vertical_body": vertical_body,
                "mpc_solves": self.online_mpc.total_solves if self.online_mpc is not None else 0,
                "mpc_valid_solves": (
                    self.online_mpc.total_valid_solves if self.online_mpc is not None else 0
                ),
                "mpc_max_iter": (
                    self.online_mpc.max_iterations if self.online_mpc is not None else 0
                ),
            },
        )
        return ActuatorCommand3D(
            throttle_cmd=throttle,
            active_engines=active_engines,
            gimbal_rad=gim,
            rcs_torque_nm=rcs,
            fin_deflection_rad=fins,
            deploy_fins=False,
            deploy_gear=(h <= s.gear_deploy_height_m),
            phase="landing_burn",
            engine_mask_7=(
                (False, False, False, False, False, False, False) if meco3 else land_mask_7
            ),
            mission_mode=(self.fdir.mission_mode if self.fdir is not None else "RTLS"),
            fdir_state=(
                self._fdir_status.fdir_state if self._fdir_status is not None else "NOMINAL"
            ),
        )

    def step(self, telemetry: TelemetryPacket3D) -> ActuatorCommand3D:
        """Execute one strictly causal 50 Hz 6DOF GNC cycle."""
        # Cleanly reset MEKF linear bias across stage-separation COM shift
        if self._was_upper_attached and not telemetry.upper_attached:
            self.estimator.reset_for_separation(
                np.asarray(telemetry.pos_enu_m, dtype=np.float64),
                np.asarray(telemetry.vel_enu_m_s, dtype=np.float64),
                time_s=telemetry.time_s,
            )
            self._was_upper_attached = False

        est = self.estimator.observe(telemetry)

        # M3 Strictly Causal FDIR update (Engine-Out & Grid-Fin Jam)
        if self.fdir is not None:
            mass_f = self._nominal_mass(est)
            _, _, rho_f = self.forecast_atmo.at(est.pos_enu_m[2])
            v_air_f = self._air_vel(est)
            q_f = 0.5 * rho_f * float(np.dot(v_air_f, v_air_f))
            self._fdir_status = self.fdir.update(
                pkt=telemetry,
                q_pa=q_f,
                vel_enu_m_s=est.vel_enu_m_s,
                pos_enu_m=est.pos_enu_m,
                accel_dist_world_m_s2=est.accel_bias_world_m_s2,
                torque_bias_body_nm=est.torque_bias_body_nm,
                mass_kg=mass_f,
            )
            eo_cnt = sum(1 for f in self.fdir.engine_failed if f)
            if eo_cnt > self._prev_eo_count:
                # Reset stale pre-isolation asymmetric disturbance torque bias & Matched-DOB SOS
                # state immediately upon symmetric engine reconfiguration (no pitch rebound!).
                self.estimator.reset_torque_bias_on_reconfig(upper_attached=est.upper_attached)
                est = replace(est, torque_bias_body_nm=(0.0, 0.0, est.torque_bias_body_nm[2]))
                self._prev_eo_count = eo_cnt
            fj_cnt = sum(1 for f in self.fdir.fin_jammed if f)
            if fj_cnt > self._prev_fj_count or (
                fj_cnt > 0
                and self.fdir.fin_jam_detected_time_s is not None
                and (est.time_s - self.fdir.fin_jam_detected_time_s) <= 0.75
            ):
                # Keep Roll DOB clean across the 0.75 s Fin-Jam runaway & healthy-fin slew
                # so resolver lag never charges wrong-sign roll bias into post-slew recovery!
                self.estimator.reset_torque_bias_on_reconfig(
                    upper_attached=est.upper_attached, reset_roll=True
                )
                est = replace(
                    est,
                    torque_bias_body_nm=(
                        est.torque_bias_body_nm[0],
                        est.torque_bias_body_nm[1],
                        0.0,
                    ),
                )
                self._prev_fj_count = fj_cnt

        def _finish(cmd_out: ActuatorCommand3D) -> ActuatorCommand3D:
            self.estimator.record_command(est.time_s, cmd_out)
            if self.fdir is not None:
                self.fdir.record_command(cmd_out)
            return cmd_out

        m_mode = self.fdir.mission_mode if self.fdir is not None else "RTLS"
        f_state = self._fdir_status.fdir_state if self._fdir_status is not None else "NOMINAL"

        # Phase 8: Touchdown Settle (immediate MECO + 5 s 4-leg upright settle)
        if est.touchdown_detected:
            self.phase = "touchdown_settle"
            if self._td_started_s is None:
                self._td_started_s = est.time_s
            # In v2 sloped-pad landings, brief cold-gas nitrogen RCS rate damping during the first
            # 3.5 s after MECO (strictly 0.0 during final 1.5 s passive standing check; main engine
            # 100% OFF!) damps 2.5-deg sloped-pad leg rocking (Falcon 9 ASDS standard practice).
            td_rcs = (0.0, 0.0, 0.0)
            dt_td = est.time_s - self._td_started_s
            if self.gnc_v2.use_online_mpc and dt_td <= 3.5 and est.rcs_propellant_kg > 0.0:
                fade_td = 1.0 - _smoothstep((dt_td - 3.0) / 0.5)
                k_td_damp = 900_000.0 * fade_td
                td_lim = min(45_000.0, self.vehicle.propulsion.rcs_torque_limit_nm) * fade_td
                td_rcs = (
                    max(-td_lim, min(td_lim, -k_td_damp * float(est.omega_body_rad_s[0]))),
                    max(-td_lim, min(td_lim, -k_td_damp * float(est.omega_body_rad_s[1]))),
                    0.0,
                )
            cmd = ActuatorCommand3D(
                throttle_cmd=0.0,
                active_engines=0,
                gimbal_rad=(0.0, 0.0),
                rcs_torque_nm=td_rcs,
                fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
                deploy_fins=True,
                deploy_gear=True,
                phase="touchdown_settle",
                mission_mode=m_mode,
                fdir_state=f_state,
            )
            mpc_used = self.online_mpc is not None and self.online_mpc.total_valid_solves > 0
            mpc_ms = self.online_mpc.max_solve_ms if self.online_mpc is not None else 0.0
            self.last_diag = GuidanceDiagnostics3D(
                phase="touchdown_settle",
                roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
                socp_solver_used=(
                    mpc_used or (self.socp_solution is not None and self.socp_solution.feasible)
                ),
                socp_solver_status=(
                    "mpc_optimal"
                    if mpc_used
                    else (self.socp_solution.status if self.socp_solution is not None else "unused")
                ),
                socp_solve_ms=(
                    mpc_ms
                    if mpc_used
                    else (self.socp_solution.solve_ms if self.socp_solution is not None else 0.0)
                ),
            )
            return _finish(cmd)

        # Phase 1: Ascent
        if est.upper_attached:
            self.phase = "ascent"
            cmd = self._ascent_guidance(est)
            return _finish(cmd)

        s = self.settings
        mass = self._nominal_mass(est)
        mp = self._mass_props(mass, est.gear_extension, False, est.rcs_propellant_kg)
        cg = mp.cg_station_m
        h = self._foot_height_m(est, cg)
        r_wb = quat_to_rotmat(est.quat_wxyz)
        _, _, q_pa, _, _, _, _ = compute_body_aero_3d(
            altitude_m=est.pos_enu_m[2],
            vel_enu_m_s=est.vel_enu_m_s,
            rot_wb=r_wb,
            omega_body_rad_s=est.omega_body_rad_s,
            cg_station_m=cg,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=est.time_s,
            vehicle=self.vehicle,
        )

        if self.phase == "ascent":
            self.phase = "flip"

        # Phase 2: Low-q 3-Axis RCS Flip
        if self.phase == "flip":
            if m_mode == "DOWNRANGE_SPLASHDOWN_ABORT":
                # Skip boostback completely in ocean abort mode -> align retrograde in coast
                self.phase = "coast"
            else:
                if self.flip_started_s is None:
                    if q_pa > s.flip_max_q_pa:
                        v_air = self._air_vel(est)
                        d_hold = (
                            v_air if est.vel_enu_m_s[2] > 0.0 else self._retro_dir(v_air, 260.0)
                        )
                        q_des = pointing_dir_to_quat_wxyz(d_hold)
                        gim, rcs, fins = self._attitude_allocate_3d(est, q_des, 0.0)
                        cmd = ActuatorCommand3D(
                            gimbal_rad=gim,
                            rcs_torque_nm=rcs,
                            fin_deflection_rad=fins,
                            phase="flip",
                            mission_mode=m_mode,
                            fdir_state=f_state,
                        )
                        return _finish(cmd)
                    self.flip_started_s = est.time_s
                    self._plan_boostback_3d(est)

                q_des = pointing_dir_to_quat_wxyz(
                    np.array([float(self.boost_dir_xy[0]), float(self.boost_dir_xy[1]), 0.0])
                )
                q_err = quat_multiply_wxyz(quat_conjugate_wxyz(est.quat_wxyz), q_des)
                err_ang_deg = degrees(
                    2.0 * atan2(float(np.linalg.norm(q_err[1:4])), abs(float(q_err[0])))
                )
                om_mag_deg = degrees(float(np.linalg.norm(est.omega_body_rad_s)))
                if err_ang_deg < 2.0 and om_mag_deg < 1.0:
                    self.phase = "boostback"
                    self.boost_started_s = est.time_s
                    self._plan_boostback_3d(est)
                else:
                    gim, rcs, fins = self._attitude_allocate_3d(
                        est,
                        q_des,
                        0.0,
                        rate_limit_deg_s=s.flip_rate_limit_deg_s,
                        accel_limit_deg_s2=s.flip_accel_limit_deg_s2,
                        high_q_guard=False,
                    )
                    cmd = ActuatorCommand3D(
                        gimbal_rad=gim,
                        rcs_torque_nm=rcs,
                        fin_deflection_rad=fins,
                        phase="flip",
                        mission_mode=m_mode,
                        fdir_state=f_state,
                    )
                    self.last_diag = GuidanceDiagnostics3D(
                        phase="flip",
                        tilt_target_deg=degrees(tilt_angle_rad(q_des)),
                        roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
                    )
                    return _finish(cmd)

        # Phase 3: 3-Engine 3D Boostback Burn (Restart #1)
        if self.phase == "boostback":
            if est.time_s >= self.boost_end_s - 1e-8:
                self.phase = "coast"
            else:
                if est.time_s >= self.next_replan_s and self.boost_end_s - est.time_s > 0.15:
                    self._plan_boostback_3d(est)
                if est.time_s >= self.boost_end_s - 1e-8:
                    self.phase = "coast"
                else:
                    q_des = pointing_dir_to_quat_wxyz(
                        np.array([float(self.boost_dir_xy[0]), float(self.boost_dir_xy[1]), 0.0])
                    )
                    _, p_pa, _ = self.forecast_atmo.at(est.pos_enu_m[2])
                    full_t, _ = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)
                    min_m = max(
                        self.vehicle.booster_dry_mass_kg - 1_000.0,
                        mass - 0.01 * self.vehicle.launch_mass_kg,
                    )
                    max_bb_t = s.boostback_max_load_g * G0 * min_m / 1.03
                    if self.fdir is not None:
                        bb_mask_7, bb_eng = self.fdir.select_3eng_mask()
                    else:
                        bb_mask_7, bb_eng = None, 3
                    lo3, _ = self.vehicle.propulsion.throttle_bounds(bb_eng)
                    bb_el = est.time_s - (self.boost_started_s or est.time_s)
                    ramp = _smoothstep((bb_el - 0.15) / 2.0)
                    bb_req = lo3 + ramp * max(0.0, s.boostback_throttle_total - lo3)
                    throttle = self.vehicle.propulsion.quantize_throttle(
                        min(bb_req, max_bb_t / max(full_t, 1.0)), bb_eng
                    )
                    gim, rcs, fins = self._attitude_allocate_3d(
                        est,
                        q_des,
                        throttle,
                        rate_limit_deg_s=6.0,
                        accel_limit_deg_s2=1.0,
                        high_q_guard=False,
                        use_xy_torque_bias=self.gnc_v2.use_structural_filter,
                        engine_mask_7=bb_mask_7,
                    )
                    cmd = ActuatorCommand3D(
                        throttle_cmd=throttle,
                        active_engines=bb_eng,
                        gimbal_rad=gim,
                        rcs_torque_nm=rcs,
                        fin_deflection_rad=fins,
                        phase="boostback",
                        engine_mask_7=bb_mask_7,
                        mission_mode=m_mode,
                        fdir_state=f_state,
                    )
                    self.last_diag = GuidanceDiagnostics3D(
                        phase="boostback",
                        tilt_target_deg=degrees(tilt_angle_rad(q_des)),
                        roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
                    )
                    return _finish(cmd)

        # Phase 4: Ballistic Coast & Grid-Fin Deployment
        if self.phase == "coast":
            deploy_fins = est.vel_enu_m_s[2] < 0.0 and est.pos_enu_m[2] <= s.fin_deploy_altitude_m
            if (
                est.vel_enu_m_s[2] < 0.0
                and est.pos_enu_m[2] < 68_000.0
                and not self.entry_burn_done
            ):
                _, _, rho_now = self.forecast_atmo.at(est.pos_enu_m[2])
                v_air = self._air_vel(est)
                qdot_now = compute_stagnation_heat_flux_w_m2(
                    rho_now, float(np.linalg.norm(v_air)), self.vehicle.geometry.body_radius_m
                )
                if q_pa >= s.entry_min_ignite_q_pa or qdot_now >= s.entry_min_ignite_heat_w_m2:
                    if (
                        self.cached_unpowered_q_pa is None
                        or est.time_s >= self.last_peak_q_time_s + 0.20 - 1e-9
                    ):
                        pq, pqd = self._predict_unpowered_peak_q_3d(
                            est.pos_enu_m, est.vel_enu_m_s, mass, est.time_s
                        )
                        self.cached_unpowered_q_pa = pq
                        self.cached_unpowered_qdot_w_m2 = pqd
                        self.last_peak_q_time_s = est.time_s
                    if (
                        self.cached_unpowered_q_pa > s.entry_q_trigger_pa
                        or self.cached_unpowered_qdot_w_m2 > s.entry_heat_trigger_w_m2
                    ):
                        self.phase = "entry_burn"
                        self.entry_burn_started_s = est.time_s
                elif est.pos_enu_m[2] < 22_000.0:
                    self.entry_burn_done = True
                    self.phase = "aero_glide"
                    self.glide_started_s = est.time_s

            if self.phase == "coast":
                q_des = pointing_dir_to_quat_wxyz(self._retro_dir(self._air_vel(est), 260.0))
                gim, rcs, fins = self._attitude_allocate_3d(
                    est,
                    q_des,
                    0.0,
                    allow_fin=deploy_fins,
                    rcs_deadband_nm=150.0 if q_pa < 100.0 else 0.0,
                )
                cmd = ActuatorCommand3D(
                    gimbal_rad=gim,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fins,
                    deploy_fins=deploy_fins,
                    phase="coast",
                    mission_mode=m_mode,
                    fdir_state=f_state,
                )
                self.last_diag = GuidanceDiagnostics3D(
                    phase="coast",
                    tilt_target_deg=degrees(tilt_angle_rad(q_des)),
                    roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
                    predicted_unpowered_peak_q_pa=self.cached_unpowered_q_pa,
                )
                return _finish(cmd)

        # Phase 5: Entry Burn (Restart #2)
        if self.phase == "entry_burn":
            if (
                self.cached_unpowered_q_pa is None
                or est.time_s >= self.last_peak_q_time_s + 0.20 - 1e-9
            ):
                pq, pqd = self._predict_unpowered_peak_q_3d(
                    est.pos_enu_m, est.vel_enu_m_s, mass, est.time_s
                )
                self.cached_unpowered_q_pa = pq
                self.cached_unpowered_qdot_w_m2 = pqd
                self.last_peak_q_time_s = est.time_s
            pq = self.cached_unpowered_q_pa
            pqd = self.cached_unpowered_qdot_w_m2
            if (pq <= s.entry_q_cutoff_pa and pqd <= s.entry_heat_cutoff_w_m2) or est.pos_enu_m[
                2
            ] <= 11_000.0:
                self.entry_burn_done = True
                self.phase = "aero_glide"
                self.glide_started_s = est.time_s
            else:
                v_air = self._air_vel(est)
                d_ret = self._retro_dir(v_air, 20.0)
                if est.time_s >= self.last_iip_time_s + 0.40 - 1e-9:
                    xi, yi, _ = self._predict_rtls_iip_3d(
                        est,
                        0.0,
                        self.boost_dir_xy,
                        include_entry_burn=False,
                        handoff_z_m=100.0,
                        dt=0.35,
                    )
                    dt_iip = est.time_s - self.last_iip_time_s
                    if 0.05 <= dt_iip <= 2.0:
                        raw_v_iip = np.clip(
                            np.array(
                                [
                                    (xi - self.cached_iip_xy_m[0]) / dt_iip,
                                    (yi - self.cached_iip_xy_m[1]) / dt_iip,
                                ],
                                dtype=np.float64,
                            ),
                            -250.0,
                            250.0,
                        )
                        self._eb_iip_vel_xy_m_s += (1.0 - exp(-dt_iip / 0.65)) * (
                            raw_v_iip - self._eb_iip_vel_xy_m_s
                        )
                    else:
                        self._eb_iip_vel_xy_m_s[:] = 0.0
                    self.cached_iip_xy_m = (xi, yi)
                    self.last_iip_time_s = est.time_s

                eb_el = est.time_s - (self.entry_burn_started_s or est.time_s)
                eb_ramp = _smoothstep(eb_el / 0.85)
                eb_lim = 9.0 if q_pa < 14_000.0 else (6.0 if q_pa < 24_000.0 else 3.5)
                err_xy = np.array([self.cached_iip_xy_m[0] + 150.0, self.cached_iip_xy_m[1]])
                if m_mode == "DOWNRANGE_SPLASHDOWN_ABORT":
                    raw_st = np.zeros(2, dtype=np.float64)
                    tau_eb_st = 0.70
                elif self.gnc_v2.use_structural_filter:
                    # CSI-certified softer booster loop (kp=0.55, kd=1.65) has ~3 s settling lag;
                    # use causal PD phase-lead on IIP velocity + circular 6 deg cap + earlier cutoff
                    # taper to eliminate entry-burn coning & slosh wall rebound at MECO-2.
                    eb_lim_v2 = min(eb_lim, 6.0)
                    q_taper = _smoothstep((pq - (s.entry_q_cutoff_pa + 1_500.0)) / 7_500.0)
                    err_lead = err_xy + 3.2 * self._eb_iip_vel_xy_m_s
                    raw_st = -0.012 * err_lead * q_taper * eb_ramp
                    st_nrm = float(np.linalg.norm(raw_st))
                    if st_nrm > eb_lim_v2 and st_nrm > 1e-9:
                        raw_st *= eb_lim_v2 / st_nrm
                    tau_eb_st = 0.70
                else:
                    q_taper = _smoothstep((pq - s.entry_q_cutoff_pa) / 6_500.0)
                    raw_st = np.clip(-0.040 * err_xy, -eb_lim, eb_lim) * q_taper * eb_ramp
                    tau_eb_st = 0.45
                self._eb_steer_xy_deg += (1.0 - exp(-s.control_dt_s / tau_eb_st)) * (
                    raw_st - self._eb_steer_xy_deg
                )
                d_des = d_ret + np.array(
                    [
                        radians(float(self._eb_steer_xy_deg[0])),
                        radians(float(self._eb_steer_xy_deg[1])),
                        0.0,
                    ]
                )
                q_des = pointing_dir_to_quat_wxyz(d_des)

                _, p_pa, _ = self.forecast_atmo.at(est.pos_enu_m[2])
                full_t, _ = compute_engine_thrust_3d(1.0, p_pa, self.vehicle)
                bf_w, _, _, _, _, _, _ = compute_body_aero_3d(
                    altitude_m=est.pos_enu_m[2],
                    vel_enu_m_s=est.vel_enu_m_s,
                    rot_wb=r_wb,
                    omega_body_rad_s=est.omega_body_rad_s,
                    cg_station_m=cg,
                    atmo_lut=self.forecast_atmo,
                    wind_lut=self.forecast_wind,
                    time_s=est.time_s,
                    vehicle=self.vehicle,
                )
                aero_mag = float(np.linalg.norm(bf_w))
                min_m = max(
                    self.vehicle.booster_dry_mass_kg - 1_000.0,
                    mass - 0.01 * self.vehicle.launch_mass_kg,
                )
                max_t = max(0.0, (s.entry_max_load_g * G0 * min_m - 1.15 * aero_mag) / 1.03)
                lo3, hi3 = self.vehicle.propulsion.throttle_bounds(3)
                raw_thr = min(hi3, max_t / max(full_t, 1.0))
                if q_pa > 12_000.0 and raw_thr < lo3:
                    self.entry_single_engine = True
                eb_mask_7: tuple[bool, bool, bool, bool, bool, bool, bool] | None = None
                if self.fdir is not None:
                    if self.entry_single_engine:
                        eb_mask_7, eb_eng = self.fdir.select_1eng_mask()
                    else:
                        eb_mask_7, eb_eng = self.fdir.select_3eng_mask()
                else:
                    eb_eng = 1 if self.entry_single_engine else 3
                lo_eb = self.vehicle.propulsion.throttle_bounds(eb_eng)[0]
                throttle = self.vehicle.propulsion.quantize_throttle(
                    lo_eb + eb_ramp * max(0.0, raw_thr - lo_eb), eb_eng
                )
                gim, rcs, fins = self._attitude_allocate_3d(
                    est,
                    q_des,
                    throttle,
                    rate_limit_deg_s=6.5,
                    accel_limit_deg_s2=1.6,
                    allow_fin=True,
                    accel_guard_deg_s2=(2.2 if self.gnc_v2.use_structural_filter else 3.5),
                    use_xy_torque_bias=(eb_el > 1.2),
                    engine_mask_7=eb_mask_7,
                )
                cmd = ActuatorCommand3D(
                    throttle_cmd=throttle,
                    active_engines=eb_eng,
                    gimbal_rad=gim,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fins,
                    deploy_fins=True,
                    phase="entry_burn",
                    engine_mask_7=eb_mask_7,
                    mission_mode=m_mode,
                    fdir_state=f_state,
                )
                self.last_diag = GuidanceDiagnostics3D(
                    phase="entry_burn",
                    tilt_target_deg=degrees(tilt_angle_rad(q_des)),
                    roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
                    predicted_unpowered_peak_q_pa=pq,
                )
                return _finish(cmd)

        # Phase 6: Unpowered 4-Grid-Fin 3D Aerodynamic Glide (0% Main Throttle!)
        if self.phase == "aero_glide":
            h_ignite = self._compute_hoverslam_ignition_height(est, mass)
            if h <= h_ignite:
                self.phase = "landing_burn"
                self.landing_started_s = est.time_s
            else:
                if est.time_s >= self.last_iip_time_s + s.glide_iip_period_s - 1e-9:
                    xi, yi, _ = self._predict_rtls_iip_3d(
                        est,
                        0.0,
                        self.boost_dir_xy,
                        include_entry_burn=False,
                        handoff_z_m=100.0,
                        dt=0.20,
                        use_nav_bias=True,
                    )
                    self.cached_iip_xy_m = (xi, yi)
                    self.last_iip_time_s = est.time_s

                v_air = self._air_vel(est)
                d_ret = self._retro_dir(v_air, 20.0)
                h_above = max(h - h_ignite, 150.0)
                k_gl = max(0.07, min(0.28, 0.11 * (5_000.0 / max(h_above, 1_500.0))))
                pre_taper = _smoothstep((h - (h_ignite + 280.0)) / 420.0)
                hi_z_cap = 6.0 + (s.glide_max_alpha_deg - 6.0) * (
                    1.0 - _smoothstep((est.pos_enu_m[2] - 19_000.0) / 8_000.0)
                )
                if self.gnc_v2.use_structural_filter:
                    # Two-stage aerodynamic AoA envelope (v2):
                    # - Subsonic low-q (q <= 9.0 kPa, z ~ 1..7 km): 9.5..10.1 deg authority for
                    #   post-blackout crossrange closure without high-q slosh-CG roll coupling.
                    # - Transonic/supersonic high-q (q >= 11.0 kPa, z ~ 8..25 km): 4.5 deg cap
                    #   suppresses q*alpha^2 roll torque in jet-shear/Dryden & fin-jam glide.
                    q_cap = min(
                        hi_z_cap,
                        s.glide_max_alpha_deg
                        - 6.5 * _smoothstep((q_pa - 3_500.0) / 5_500.0)
                        - 5.0 * _smoothstep((q_pa - 9_000.0) / 2_000.0),
                    )
                else:
                    q_cap = min(
                        hi_z_cap,
                        s.glide_max_alpha_deg - 7.5 * _smoothstep((q_pa - 3_500.0) / 6_500.0),
                    )
                max_a = q_cap * pre_taper

                # Causal 3D upwind drift lead compensating 24 s Hoverslam wind exposure
                # (scaled down smoothly in thin/low-drag air to avoid over-leading)
                glide_target_xy = np.zeros(2, dtype=np.float64)
                if est.pos_enu_m[2] < 14_000.0:
                    dense_w = max(0.0, min(1.3, (self.cached_drag_scale - 0.85) / 0.15))
                    w_gnd = self.forecast_wind.at(150.0, est.time_s)
                    w_mid = self.forecast_wind.at(2_000.0, est.time_s)
                    w_hi = self.forecast_wind.at(5_000.0, est.time_s)
                    for ax_i in range(2):
                        if w_gnd[ax_i] * w_mid[ax_i] > 0.0 and w_gnd[ax_i] * w_hi[ax_i] > 0.0:
                            w_low = 0.65 * w_gnd[ax_i] + 0.35 * w_mid[ax_i]
                            glide_target_xy[ax_i] = max(-55.0, min(55.0, -4.5 * w_low * dense_w))

                gl_ramp = _smoothstep((est.time_s - (self.glide_started_s or est.time_s)) / 2.5)
                iip_vec = np.array(self.cached_iip_xy_m, dtype=np.float64) - glide_target_xy
                raw_a_xy = (
                    np.zeros(2, dtype=np.float64)
                    if m_mode == "DOWNRANGE_SPLASHDOWN_ABORT"
                    else k_gl * iip_vec * gl_ramp
                )
                a_nrm = float(np.linalg.norm(raw_a_xy))
                if a_nrm > max_a and a_nrm > 1e-9:
                    raw_a_xy *= max_a / a_nrm
                # M3 Fin-Jam AoA Corridor Shift (CDR §3.4): keep jammed fin well inside
                # hinge limit while capping high-q body AoA (<= 4.5 deg at q >= 16 kPa -> 9.5 deg
                # at low q) so the single healthy fin on the jammed axis NEVER hits +/-20 deg stop!
                has_fj_gl = self.fdir is not None and any(self.fdir.fin_jammed)
                if has_fj_gl and self.fdir is not None and q_pa >= 400.0:
                    hi_q_w = _smoothstep((q_pa - 5_000.0) / 7_000.0)
                    body_lim_deg = 6.5 - 4.0 * hi_q_w
                    corridor_half_deg = 13.0 - 3.5 * hi_q_w
                    for j in range(4):
                        if self.fdir.fin_jammed[j]:
                            d_jam_deg = degrees(self.fdir.fin_jam_angles_rad[j])
                            # Fins 0,1 (+X/-X) see ay_ret (1); fins 2,3 (+Y/-Y) see ax_ret (0)
                            ax_j = 1 if j < 2 else 0
                            lo_c = max(-body_lim_deg, d_jam_deg - corridor_half_deg)
                            hi_c = min(body_lim_deg, d_jam_deg + corridor_half_deg)
                            if lo_c > hi_c:
                                mid_c = max(-body_lim_deg, min(body_lim_deg, d_jam_deg))
                                lo_c, hi_c = mid_c, mid_c
                            raw_a_xy[ax_j] = max(lo_c, min(hi_c, float(raw_a_xy[ax_j])))
                    fj_cap = 7.0 - 4.0 * hi_q_w
                    a_fj_nrm = float(np.linalg.norm(raw_a_xy))
                    if a_fj_nrm > fj_cap and a_fj_nrm > 1e-9:
                        raw_a_xy *= fj_cap / a_fj_nrm
                if est.engine_throttle > 0.01:
                    raw_a_xy[:] = 0.0

                tau_gl_a = (
                    0.55 if has_fj_gl else (0.50 if self.gnc_v2.use_structural_filter else 0.35)
                )
                self._glide_alpha_xy_deg += (1.0 - exp(-s.control_dt_s / tau_gl_a)) * (
                    raw_a_xy - self._glide_alpha_xy_deg
                )
                # Construct 3D target pointing vector by tilting retrograde axis in ENU X/Y
                a_x_r = radians(float(self._glide_alpha_xy_deg[0]))
                a_y_r = radians(float(self._glide_alpha_xy_deg[1]))
                d_des = d_ret + np.array([a_x_r, a_y_r, 0.0], dtype=np.float64)
                q_des = pointing_dir_to_quat_wxyz(d_des)

                deploy_fins_gl = h > h_ignite + 240.0
                hi_q_gl = (
                    _smoothstep((q_pa - 6_000.0) / 7_000.0)
                    if (self.gnc_v2.use_structural_filter and not has_fj_gl)
                    else 0.0
                )
                gl_acc_lim = 1.45 - 0.35 * hi_q_gl if self.gnc_v2.use_structural_filter else 2.2
                gl_acc_guard = 1.85 - 0.45 * hi_q_gl if self.gnc_v2.use_structural_filter else 2.3
                if has_fj_gl and (
                    abs(degrees(roll_angle_rad(est.quat_wxyz))) > 0.8
                    or abs(degrees(est.omega_body_rad_s[2])) > 1.5
                ):
                    el_fj_gl = (
                        est.time_s - self.fdir.fin_jam_detected_time_s
                        if (self.fdir is not None and self.fdir.fin_jam_detected_time_s is not None)
                        else 0.0
                    )
                    gl_acc_guard = max(gl_acc_guard, 2.9 if el_fj_gl > 1.00 else 2.25)
                gim, rcs, fins = self._attitude_allocate_3d(
                    est,
                    q_des,
                    0.0,
                    rate_limit_deg_s=9.0,
                    accel_limit_deg_s2=gl_acc_lim,
                    allow_fin=True,
                    rcs_deadband_nm=(
                        3_500.0
                        if has_fj_gl
                        else (5_000.0 if (q_pa > 1_000.0 and est.fin_extension >= 0.4) else 0.0)
                    ),
                    accel_guard_deg_s2=gl_acc_guard,
                    use_xy_torque_bias=True,
                )
                cmd = ActuatorCommand3D(
                    throttle_cmd=0.0,
                    active_engines=0,
                    gimbal_rad=gim,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fins,
                    deploy_fins=deploy_fins_gl,
                    deploy_gear=(h <= s.gear_deploy_height_m),
                    phase="aero_glide",
                    mission_mode=m_mode,
                    fdir_state=f_state,
                )
                self.last_diag = GuidanceDiagnostics3D(
                    phase="aero_glide",
                    tilt_target_deg=degrees(tilt_angle_rad(q_des)),
                    roll_deg=degrees(roll_angle_rad(est.quat_wxyz)),
                    predicted_iip_xy_m=self.cached_iip_xy_m,
                    hoverslam_ignition_height_m=h_ignite,
                )
                return _finish(cmd)

        # Phase 7: 3->1 Center-Engine 3D Hoverslam
        cmd = self._landing_burn_guidance(est, mass, cg, h)
        return _finish(cmd)

    def diagnostics(self) -> GuidanceDiagnostics3D:
        return self.last_diag
