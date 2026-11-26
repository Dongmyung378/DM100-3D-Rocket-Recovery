"""15-State Multiplicative EKF (MEKF) and causal actuator/torque-bias estimator.

Error state vector (15-DOF):
  dx = [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]^T

Strictly causal: consumes ONLY `TelemetryPacket3D`, forecast atmosphere/wind, and past
issued `ActuatorCommand3D`. Never receives hidden true wind gusts, scale errors, or true biases.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, radians, sqrt

import numpy as np
from numpy.typing import NDArray

from dm100_3d.config import (
    R_AIR,
    DM100VehicleConfig,
    GncV2Config,
    MassProperties3D,
    mass_properties_3d,
)
from dm100_3d.coords import (
    axis_angle_to_quat_wxyz,
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_to_rotmat,
)
from dm100_3d.interfaces import ActuatorCommand3D, TelemetryPacket3D
from dm100_3d.physics import (
    AtmosphereLUT,
    Wind3DLUT,
    compute_coriolis_and_centrifugal_3d,
    compute_engine_thrust_3d,
    compute_forces_and_torques_3d,
    srp_nominal_cp_shift_m,
    srp_nominal_drag_multiplier,
    wgs84_coriolis_and_transport_enu,
    wgs84_normal_gravity,
)
from dm100_3d.plant_mujoco import fast_gravity


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class SensorConfig3D:
    """Sensor noise, bias, and dropout envelope for 6DOF telemetry."""

    position_noise_m: float = 0.0
    velocity_noise_m_s: float = 0.0
    attitude_noise_deg: float = 0.0
    gyro_noise_deg_s: float = 0.0
    accel_noise_m_s2: float = 0.0
    gyro_bias_deg_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_bias_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dropout_rate: float = 0.0
    observation_delay_steps: int = 0
    action_delay_steps: int = 0


@dataclass(frozen=True, slots=True)
class EstimatedState3D:
    time_s: float
    pos_enu_m: tuple[float, float, float]
    vel_enu_m_s: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    omega_body_rad_s: tuple[float, float, float]
    accel_bias_world_m_s2: tuple[float, float, float]
    gyro_bias_body_rad_s: tuple[float, float, float]
    torque_bias_body_nm: tuple[float, float, float]
    main_propellant_kg: float
    rcs_propellant_kg: float
    gear_extension: float
    fin_deflection_rad: tuple[float, float, float, float]
    fin_extension: float
    gimbal_rad: tuple[float, float]
    engine_throttle: float
    upper_attached: bool
    on_pad: bool
    touchdown_detected: bool
    active_engines: int
    restarts_used: int


class CausalEstimator3D:
    """15-State MEKF + causal latency propagator + 3-axis external torque-bias estimator."""

    def __init__(
        self,
        *,
        vehicle: DM100VehicleConfig,
        forecast_atmo: AtmosphereLUT,
        forecast_wind: Wind3DLUT,
        latitude_deg: float = 37.0,
        coriolis_enabled: bool = True,
        planar_coriolis_only: bool = False,
        sensor_config: SensorConfig3D | None = None,
        control_dt_s: float = 0.02,
        gnc_v2: GncV2Config | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.forecast_atmo = forecast_atmo
        self.forecast_wind = forecast_wind
        self.latitude_deg = float(latitude_deg)
        self.coriolis_enabled = bool(coriolis_enabled)
        self.planar_coriolis_only = bool(planar_coriolis_only)
        self.sensor_config = sensor_config or SensorConfig3D()
        self.control_dt_s = float(control_dt_s)
        self.gnc_v2 = gnc_v2 or GncV2Config()

        # Nominal MEKF state
        self.initialized: bool = False
        self.last_meas_t: float = 0.0
        self.p: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.v: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.q: NDArray[np.float64] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.b_a: NDArray[np.float64] = np.zeros(3, dtype=np.float64)  # ENU accel residual bias
        self.b_g: NDArray[np.float64] = np.zeros(3, dtype=np.float64)  # body gyro bias
        self.P: NDArray[np.float64] = np.diag(
            [1.0] * 3 + [0.5] * 3 + [1e-3] * 3 + [0.5] * 3 + [1e-5] * 3
        )

        # Measurement noise floors
        sc = self.sensor_config
        sig_p = max(sc.position_noise_m, 0.05)
        sig_v = max(sc.velocity_noise_m_s, 0.03)
        sig_att = max(radians(sc.attitude_noise_deg), radians(0.015))
        self.R_meas = np.diag([sig_p**2] * 3 + [sig_v**2] * 3 + [sig_att**2] * 3)
        self.H_meas = np.zeros((9, 15), dtype=np.float64)
        self.H_meas[0:3, 0:3] = np.eye(3)
        self.H_meas[3:6, 3:6] = np.eye(3)
        self.H_meas[6:9, 6:9] = np.eye(3)

        self.command_history: deque[tuple[float, ActuatorCommand3D]] = deque(maxlen=64)
        self.torque_bias_body_nm: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._last_omega_filt: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._prev_ang_sample: tuple | None = None
        self._sep_time_s: float | None = None

    def record_command(self, time_s: float, cmd: ActuatorCommand3D) -> None:
        exec_end = time_s + (self.sensor_config.action_delay_steps + 1) * self.control_dt_s
        self.command_history.append((exec_end, cmd))

    def _nominal_mass(self, pkt: TelemetryPacket3D) -> float:
        return (
            self.vehicle.booster_dry_mass_kg
            + max(pkt.main_propellant_kg, 0.0)
            + max(pkt.rcs_propellant_kg, 0.0)
            + (self.vehicle.upper_stage.total_mass_kg if pkt.upper_attached else 0.0)
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

    def _nominal_srp_terms(
        self,
        altitude_m: float,
        vel_enu_m_s: NDArray[np.float64],
        q_wxyz: NDArray[np.float64],
        throttle: float,
        upper_attached: bool,
        time_s: float,
    ) -> tuple[float, float]:
        """Causal nominal SRP (drag_multiplier, cp_shift_m) for MEKF propagation & torque bias."""
        if (not self.gnc_v2.use_srp_iip_feedforward) or upper_attached or throttle <= 1e-6:
            return 1.0, 0.0
        t_k, p_pa, rho = self.forecast_atmo.at(altitude_m)
        wx, wy, wz = self.forecast_wind.at(altitude_m, time_s)
        v_rel = np.array(
            [float(vel_enu_m_s[0]) - wx, float(vel_enu_m_s[1]) - wy, float(vel_enu_m_s[2]) - wz],
            dtype=np.float64,
        )
        spd = float(np.linalg.norm(v_rel))
        if spd <= 1.0:
            return 1.0, 0.0
        q_pa = 0.5 * rho * spd * spd
        mach = spd / sqrt(1.4 * R_AIR * t_k)
        vz_b = float((quat_to_rotmat(q_wxyz).T @ v_rel)[2])
        t_nom, _ = compute_engine_thrust_3d(throttle, p_pa, self.vehicle)
        r_body = self.vehicle.geometry.body_radius_m
        drag_mult = srp_nominal_drag_multiplier(
            mach=mach,
            v_air_zb_m_s=vz_b,
            thrust_n=t_nom,
            dynamic_pressure_pa=q_pa,
            body_radius_m=r_body,
            eta_nom=self.gnc_v2.srp_nominal_eta,
            upper_attached=False,
        )
        cp_shift = srp_nominal_cp_shift_m(
            mach=mach,
            v_air_zb_m_s=vz_b,
            thrust_n=t_nom,
            dynamic_pressure_pa=q_pa,
            body_radius_m=r_body,
            upper_attached=False,
        )
        return drag_mult, cp_shift

    def _nominal_world_accel(
        self, pkt: TelemetryPacket3D, q_wxyz: NDArray[np.float64], omega_b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        mass = self._nominal_mass(pkt)
        mp = self._mass_props(mass, pkt.gear_extension, pkt.upper_attached, pkt.rcs_propellant_kg)
        thr_cl = max(0.0, min(1.0, pkt.engine_throttle))
        srp_drag_mult, srp_cp_shift = self._nominal_srp_terms(
            float(self.p[2]),
            self.v,
            q_wxyz,
            thr_cl,
            pkt.upper_attached,
            pkt.measurement_time_s,
        )
        ft = compute_forces_and_torques_3d(
            altitude_m=float(self.p[2]),
            vel_enu_m_s=self.v,
            quat_wxyz=q_wxyz,
            omega_body_rad_s=omega_b,
            cg_station_m=mp.cg_station_m,
            throttle_actual=thr_cl,
            gimbal_rad=pkt.gimbal_rad,
            rcs_torque_body_nm=(0.0, 0.0, 0.0),
            fin_deflection_rad=pkt.fin_deflection_rad,
            fin_extension=pkt.fin_extension,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=pkt.measurement_time_s,
            vehicle=self.vehicle,
            upper_attached=pkt.upper_attached,
            axial_drag_multiplier=srp_drag_mult,
            cp_shift_m=srp_cp_shift,
        )
        if self.gnc_v2.use_wgs84_gravity:
            g_loc = wgs84_normal_gravity(self.latitude_deg, float(self.p[2]))
            a_cor = wgs84_coriolis_and_transport_enu(
                altitude_m=float(self.p[2]),
                vel_enu_m_s=self.v,
                latitude_deg=self.latitude_deg,
                coriolis_enabled=self.coriolis_enabled,
                planar_coriolis_only=self.planar_coriolis_only,
            )
        else:
            g_loc = fast_gravity(float(self.p[2]))
            a_cor = compute_coriolis_and_centrifugal_3d(
                altitude_m=float(self.p[2]),
                vel_enu_m_s=self.v,
                latitude_deg=self.latitude_deg,
                coriolis_enabled=self.coriolis_enabled,
                planar_coriolis_only=self.planar_coriolis_only,
            )
        if pkt.on_pad and pkt.upper_attached and ft.force_world_n[2] < mass * g_loc:
            return np.zeros(3, dtype=np.float64)
        a_w = np.asarray(ft.force_world_n, dtype=np.float64) / mass + a_cor
        a_w[2] -= g_loc
        return a_w

    def _predict_actuators(
        self, pkt: TelemetryPacket3D
    ) -> tuple[tuple[float, float], float, tuple[float, float, float, float]]:
        gim = [float(pkt.gimbal_rad[0]), float(pkt.gimbal_rad[1])]
        thr = float(pkt.engine_throttle)
        fins = [float(v) for v in pkt.fin_deflection_rad]
        begin = pkt.measurement_time_s
        g_tau = self.vehicle.propulsion.gimbal_tau_s
        e_tau = self.vehicle.propulsion.nominal_engine_tau_s
        f_tau = self.vehicle.fins.actuator_time_constant_s

        for end, cmd in self.command_history:
            if end <= begin + 1e-12:
                continue
            dur = min(end, pkt.time_s) - begin
            if dur <= 0.0:
                continue
            thr = cmd.throttle_cmd + (thr - cmd.throttle_cmd) * exp(-dur / e_tau)
            for i in range(2):
                gim[i] = cmd.gimbal_rad[i] + (gim[i] - cmd.gimbal_rad[i]) * exp(-dur / g_tau)
            for i in range(4):
                fins[i] = cmd.fin_deflection_rad[i] + (fins[i] - cmd.fin_deflection_rad[i]) * exp(
                    -dur / f_tau
                )
            begin += dur
        return (
            (gim[0], gim[1]),
            max(0.0, min(1.0, thr)),
            (fins[0], fins[1], fins[2], fins[3]),
        )

    def observe(self, pkt: TelemetryPacket3D) -> EstimatedState3D:
        """Run one 15-state MEKF predict+update cycle and forward-propagate over sensor latency."""
        p_meas = np.asarray(pkt.pos_enu_m, dtype=np.float64)
        v_meas = np.asarray(pkt.vel_enu_m_s, dtype=np.float64)
        q_meas = normalize_quat_wxyz(pkt.quat_wxyz, canonical=True)
        om_meas = np.asarray(pkt.omega_body_rad_s, dtype=np.float64)

        if not self.initialized:
            self.p = p_meas.copy()
            self.v = v_meas.copy()
            self.q = q_meas.copy()
            self.last_meas_t = pkt.measurement_time_s
            self._last_omega_filt = om_meas.copy()
            self.initialized = True
        else:
            dt = pkt.measurement_time_s - self.last_meas_t
            if dt > 1e-9:
                # 1. Propagate nominal state with bias-corrected gyro and nominal model + b_a
                om_corr = om_meas - self.b_g
                a_nom = self._nominal_world_accel(pkt, self.q, om_corr)
                a_eff = a_nom + self.b_a
                self.p = self.p + self.v * dt + 0.5 * a_eff * (dt * dt)
                self.v = self.v + a_eff * dt
                dq = axis_angle_to_quat_wxyz(om_corr, float(np.linalg.norm(om_corr)) * dt)
                self.q = normalize_quat_wxyz(quat_multiply_wxyz(self.q, dq), canonical=True)

                # 2. Propagate 15x15 error-state covariance
                f_mat = np.eye(15, dtype=np.float64)
                f_mat[0:3, 3:6] = np.eye(3) * dt
                f_mat[3:6, 9:12] = np.eye(3) * dt
                f_mat[6:9, 6:9] = np.eye(3) - _skew(om_corr) * dt
                f_mat[6:9, 12:15] = -np.eye(3) * dt

                q_proc = np.diag(
                    [1e-4 * dt] * 3
                    + [0.25 * dt] * 3
                    + [2e-5 * dt] * 3
                    + [0.8 * dt] * 3
                    + [1e-6 * dt] * 3
                )
                self.P = f_mat @ self.P @ f_mat.T + q_proc
                self.last_meas_t = pkt.measurement_time_s

            # 3. Measurement update (skipped if packet was dropped)
            if not pkt.packet_dropped:
                dp_innov = p_meas - self.p
                dv_innov = v_meas - self.v
                dq_err = quat_multiply_wxyz(quat_conjugate_wxyz(self.q), q_meas)
                if dq_err[0] < 0.0:
                    dq_err = -dq_err
                dth_innov = 2.0 * dq_err[1:4]
                y_innov = np.concatenate([dp_innov, dv_innov, dth_innov])

                s_mat = self.H_meas @ self.P @ self.H_meas.T + self.R_meas
                k_gain = self.P @ self.H_meas.T @ np.linalg.inv(s_mat)
                dx = k_gain @ y_innov

                # Inject error state into nominal state (multiplicative quaternion update)
                self.p += dx[0:3]
                self.v += dx[3:6]
                dth = dx[6:9]
                dq_inj = axis_angle_to_quat_wxyz(dth, float(np.linalg.norm(dth)))
                self.q = normalize_quat_wxyz(quat_multiply_wxyz(self.q, dq_inj), canonical=True)
                self.b_a = np.clip(self.b_a + dx[9:12], -4.5, 4.5)
                self.b_g = np.clip(self.b_g + dx[12:15], -radians(0.5), radians(0.5))

                # Joseph stabilized covariance update
                i_kh = np.eye(15) - k_gain @ self.H_meas
                self.P = i_kh @ self.P @ i_kh.T + k_gain @ self.R_meas @ k_gain.T
                self.P = 0.5 * (self.P + self.P.T)

        # Low-pass filter bias-corrected angular rate for smooth derivative control
        om_now = om_meas - self.b_g
        alpha_om = 0.65 if self.sensor_config.gyro_noise_deg_s > 0.0 else 1.0
        self._last_omega_filt += alpha_om * (om_now - self._last_omega_filt)
        om_out = self._last_omega_filt

        # Causal 3-axis unmodeled disturbance torque observer (rejects external roll/aero torque)
        self._update_torque_bias_3d(pkt, self.q, om_out)

        # Forward-propagate over observation delay [measurement_time_s -> time_s]
        delay = max(0.0, pkt.time_s - pkt.measurement_time_s)
        if delay > 1e-9:
            a_pred = self._nominal_world_accel(pkt, self.q, om_out) + self.b_a
            p_now = self.p + self.v * delay + 0.5 * a_pred * (delay * delay)
            v_now = self.v + a_pred * delay
            dq_lat = axis_angle_to_quat_wxyz(om_out, float(np.linalg.norm(om_out)) * delay)
            q_now = normalize_quat_wxyz(quat_multiply_wxyz(self.q, dq_lat), canonical=True)
        else:
            p_now = self.p
            v_now = self.v
            q_now = self.q

        gim_now, thr_now, fins_now = self._predict_actuators(pkt)
        return EstimatedState3D(
            time_s=pkt.time_s,
            pos_enu_m=(float(p_now[0]), float(p_now[1]), float(p_now[2])),
            vel_enu_m_s=(float(v_now[0]), float(v_now[1]), float(v_now[2])),
            quat_wxyz=(float(q_now[0]), float(q_now[1]), float(q_now[2]), float(q_now[3])),
            omega_body_rad_s=(float(om_out[0]), float(om_out[1]), float(om_out[2])),
            accel_bias_world_m_s2=(float(self.b_a[0]), float(self.b_a[1]), float(self.b_a[2])),
            gyro_bias_body_rad_s=(float(self.b_g[0]), float(self.b_g[1]), float(self.b_g[2])),
            torque_bias_body_nm=(
                float(self.torque_bias_body_nm[0]),
                float(self.torque_bias_body_nm[1]),
                float(self.torque_bias_body_nm[2]),
            ),
            main_propellant_kg=pkt.main_propellant_kg,
            rcs_propellant_kg=pkt.rcs_propellant_kg,
            gear_extension=pkt.gear_extension,
            fin_deflection_rad=fins_now,
            fin_extension=pkt.fin_extension,
            gimbal_rad=gim_now,
            engine_throttle=thr_now,
            upper_attached=pkt.upper_attached,
            on_pad=pkt.on_pad,
            touchdown_detected=pkt.touchdown_detected,
            active_engines=pkt.active_engines,
            restarts_used=pkt.restarts_used,
        )

    def _update_torque_bias_3d(
        self, pkt: TelemetryPacket3D, q_wxyz: NDArray[np.float64], om_b: NDArray[np.float64]
    ) -> None:
        """Estimate unmodeled 3-axis external disturbance torque causally from angular rate."""
        if (
            pkt.upper_attached
            or pkt.on_pad
            or pkt.touchdown_detected
            or (self._sep_time_s is not None and pkt.measurement_time_s < self._sep_time_s + 0.35)
        ):
            self._prev_ang_sample = None
            self.torque_bias_body_nm[:] = 0.0
            return
        if pkt.packet_dropped:
            # Hold last estimated disturbance torque bias across transient packet dropouts
            return

        mass = self._nominal_mass(pkt)
        mp = self._mass_props(mass, pkt.gear_extension, False, pkt.rcs_propellant_kg)
        i_vec = np.asarray(mp.inertia_diag_kg_m2, dtype=np.float64)
        cur = (
            pkt.measurement_time_s,
            om_b.copy(),
            i_vec.copy(),
            q_wxyz.copy(),
            self.p.copy(),
            self.v.copy(),
            max(0.0, min(1.0, pkt.engine_throttle)),
            pkt.gimbal_rad,
            pkt.fin_deflection_rad,
            pkt.fin_extension,
        )
        prev = self._prev_ang_sample
        if prev is not None and pkt.measurement_time_s <= prev[0] + 1e-9:
            return
        self._prev_ang_sample = cur
        if prev is None:
            return

        dt = pkt.measurement_time_s - prev[0]
        if dt <= 1e-5 or dt > 0.25:
            return

        # Average executed RCS command over [prev_t, meas_t]
        rcs_avg = np.zeros(3, dtype=np.float64)
        if self.command_history and pkt.rcs_propellant_kg > 0.0:
            for end_t, cmd in reversed(self.command_history):
                if end_t - self.control_dt_s <= pkt.measurement_time_s + 1e-9:
                    rcs_avg = np.asarray(cmd.rcs_torque_nm, dtype=np.float64)
                    break

        om_mid = 0.5 * (prev[1] + om_b)
        q_aligned = q_wxyz if float(prev[3] @ q_wxyz) >= 0.0 else -q_wxyz
        q_mid = normalize_quat_wxyz(prev[3] + q_aligned, canonical=True)
        p_mid = 0.5 * (prev[4] + self.p)
        v_mid = 0.5 * (prev[5] + self.v)
        thr_mid = 0.5 * (prev[6] + cur[6])
        gim_mid = (0.5 * (prev[7][0] + cur[7][0]), 0.5 * (prev[7][1] + cur[7][1]))
        fin_mid = tuple(0.5 * (prev[8][k] + cur[8][k]) for k in range(4))
        fext_mid = 0.5 * (prev[9] + cur[9])

        t_mid = 0.5 * (prev[0] + pkt.measurement_time_s)
        srp_drag_mid, srp_cp_mid = self._nominal_srp_terms(
            float(p_mid[2]), v_mid, q_mid, thr_mid, False, t_mid
        )
        ft = compute_forces_and_torques_3d(
            altitude_m=float(p_mid[2]),
            vel_enu_m_s=v_mid,
            quat_wxyz=q_mid,
            omega_body_rad_s=om_mid,
            cg_station_m=mp.cg_station_m,
            throttle_actual=thr_mid,
            gimbal_rad=gim_mid,
            rcs_torque_body_nm=(float(rcs_avg[0]), float(rcs_avg[1]), float(rcs_avg[2])),
            fin_deflection_rad=(fin_mid[0], fin_mid[1], fin_mid[2], fin_mid[3]),
            fin_extension=fext_mid,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=t_mid,
            vehicle=self.vehicle,
            upper_attached=False,
            axial_drag_multiplier=srp_drag_mid,
            cp_shift_m=srp_cp_mid,
        )
        # 3-axis causal disturbance torque observer (mirrors DM100 2D estimator.py):
        #   - X/Y pitch/yaw: rejects unmodeled wind-error & gust aero moments (tau = 0.18 s)
        #   - Z roll: rejects external roll pulses with fast response (tau = 0.10 s)
        i_mid = 0.5 * (prev[2] + i_vec)
        d_l_dt = (i_vec * om_b - prev[2] * prev[1]) / dt + np.cross(om_mid, i_mid * om_mid)
        res_3 = d_l_dt - np.asarray(ft.torque_body_nm, dtype=np.float64)
        tau_xy = 0.26 if self.sensor_config.gyro_noise_deg_s > 0.0 else 0.14
        alphas = (
            1.0 - exp(-dt / tau_xy),
            1.0 - exp(-dt / tau_xy),
            1.0 - exp(-dt / 0.10),
        )
        bounds = (120_000.0, 120_000.0, 12_000.0)
        for ax_i in range(3):
            r_c = max(-bounds[ax_i], min(bounds[ax_i], float(res_3[ax_i])))
            upd = float(self.torque_bias_body_nm[ax_i]) + alphas[ax_i] * (
                r_c - float(self.torque_bias_body_nm[ax_i])
            )
            self.torque_bias_body_nm[ax_i] = max(-bounds[ax_i], min(bounds[ax_i], upd))

    def reset_for_separation(
        self, p_new: NDArray[np.float64], v_new: NDArray[np.float64], time_s: float = 0.0
    ) -> None:
        """Reset linear accel residual bias cleanly across the cold separation discontinuity."""
        self.p = p_new.copy()
        self.v = v_new.copy()
        self.b_a[:] = 0.0
        self.torque_bias_body_nm[:] = 0.0
        self._prev_ang_sample = None
        self._sep_time_s = time_s
        _ = quat_to_rotmat(self.q)
