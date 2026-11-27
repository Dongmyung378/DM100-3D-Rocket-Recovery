"""Realistic 500->50 Hz multi-sensor suite for DM100 3D v2 (M2, CDR Rev 3.1 §2.6).

Provides strictly causal, isolated-RNG sensor synthesis:
  - 500 Hz IMU synthesis (body specific force incl. ground reaction + flex contamination +
    axis non-orthogonality + scale error + bias + ARW/VRW + thrust vibration) passed through
    exact 2nd-order Butterworth anti-aliasing filter (f_aa = 18.0 Hz @ 500 Hz) -> 50 Hz.
  - GPS receiver with plasma-heat / altitude blackout suppression (`gps_valid = False`) and
    ZERO oracle quaternion leaks during flight (`attitude_valid = False` off pad).
  - Vacuum-coast Star Tracker (`q_pa < 1 Pa`, `h > 60 km`, `T == 0`) with 0.02 deg 1-sigma noise.
  - Hull-base mounted (`z_radar_b = 0.0 m`) slant-range radar altimeter (`h < 1,500 m`,
    `tilt <= 35 deg`, `sigma = 0.10 m`) intersecting either the finite 15 m tilted pad or
    flat z = 0 outer terrain.
  - Normalized 7-engine chamber pressure ratios (`P_c,meas / P_c,full`, 1st-order 30 ms lag +
    1.5% noise), fin resolver error (+/-0.10 deg, 1-step lag), and propellant slosh-gauge noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot, radians, sqrt
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from dm100_3d.coords import axis_angle_to_quat_wxyz, normalize_quat_wxyz, quat_multiply_wxyz


@dataclass(frozen=True, slots=True)
class SensorV2Settings:
    """Configuration for DM100 3D v2 realistic multi-sensor simulation."""

    imu_rate_hz: float = 500.0
    aa_cutoff_hz: float = 18.0
    aa_order: int = 2
    gyro_arw_scale: float = 1.0
    accel_vrw_scale: float = 1.0
    gyro_scale_err_ppm: float = 0.0
    accel_scale_err_ppm: float = 0.0
    gyro_nonortho_deg: float = 0.0
    accel_nonortho_deg: float = 0.0
    thrust_vib_rms_m_s2: float = 0.08
    gps_blackout_duration_s: float = 0.0
    gps_blackout_heat_flux_w_m2: float = 10_000.0
    gps_blackout_alt_min_m: float = 35_000.0
    gps_blackout_alt_max_m: float = 65_000.0
    star_tracker_noise_deg: float = 0.02
    radar_max_alt_m: float = 1_500.0
    radar_max_tilt_deg: float = 35.0
    radar_sigma_m: float = 0.10
    radar_alt_noise_scale: float = 1.0
    chamber_p_tau_s: float = 0.030
    chamber_p_noise_std: float = 0.015
    fin_resolver_noise_deg: float = 0.10
    prop_gauge_noise_kg: float = 25.0
    pad_radius_m: float = 15.0


def _nonortho_matrix(
    angle_deg: float, scale_ppm: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Construct fixed 3x3 scale + small off-diagonal axis misalignment calibration matrix."""
    s = 1.0 + float(scale_ppm) * 1e-6
    m = np.diag([s, s, s]).astype(np.float64)
    if abs(angle_deg) > 1e-9:
        sig = radians(abs(float(angle_deg)))
        eps = rng.normal(0.0, sig, size=6)
        m[0, 1], m[0, 2] = eps[0], eps[1]
        m[1, 0], m[1, 2] = eps[2], eps[3]
        m[2, 0], m[2, 1] = eps[4], eps[5]
    return m


def compute_base_slant_radar_range_m(
    *,
    pos_base_enu_m: ArrayLike,
    rot_wb: NDArray[np.float64],
    pad_normal_enu: ArrayLike,
    pad_radius_m: float = 15.0,
    upper_attached: bool = False,
    sloped_pad_enabled: bool = False,
) -> tuple[bool, float, float]:
    """Intersect `-z_b` ray from hull base `z_radar_b=0` with finite pad or z=0 outer plane.

    Returns `(valid_hit, true_slant_range_m, r_hit_xy_m)`.
    """
    p0 = np.asarray(pos_base_enu_m, dtype=np.float64).reshape(3)
    d = -rot_wb[:, 2]  # Ray along -z_b in ENU
    if float(d[2]) >= -0.25:  # Pointing upward or > ~75 deg off vertical
        return False, 0.0, 0.0

    # Check finite tilted landing pad first when post-separation and sloped pad enabled
    if (not upper_attached) and sloped_pad_enabled:
        n_pad = np.asarray(pad_normal_enu, dtype=np.float64).reshape(3)
        denom = float(n_pad @ d)
        if denom < -1e-6:
            t_pad = -float(n_pad @ p0) / denom
            if t_pad >= 0.0:
                p_hit = p0 + t_pad * d
                r_hit = hypot(float(p_hit[0]), float(p_hit[1]))
                if r_hit <= pad_radius_m:
                    return True, float(t_pad), float(r_hit)

    # Outer terrain / flat launchpad at exact z = 0.0 m
    t_flat = -float(p0[2]) / float(d[2])
    if t_flat < 0.0:
        return False, 0.0, 0.0
    p_hit_f = p0 + t_flat * d
    r_hit_f = hypot(float(p_hit_f[0]), float(p_hit_f[1]))
    return True, float(t_flat), float(r_hit_f)


class SensorSuiteV2:
    """Stateful 500 Hz anti-aliased IMU + GPS blackout + star tracker + base radar suite."""

    def __init__(
        self,
        *,
        sensor_seed: int,
        base_config: Any | None = None,
        settings: SensorV2Settings | None = None,
    ) -> None:
        self.base_config = base_config
        self.settings = settings or SensorV2Settings()
        cfg_pos_noise = float(getattr(base_config, "position_noise_m", 0.0))
        cfg_vel_noise = float(getattr(base_config, "velocity_noise_m_s", 0.0))
        cfg_att_noise = float(getattr(base_config, "attitude_noise_deg", 0.0))
        cfg_gyro_noise = float(getattr(base_config, "gyro_noise_deg_s", 0.0))
        cfg_acc_noise = float(getattr(base_config, "accel_noise_m_s2", 0.0))
        cfg_gyro_bias = tuple(getattr(base_config, "gyro_bias_deg_s", (0.0, 0.0, 0.0)))
        cfg_acc_bias = tuple(getattr(base_config, "accel_bias_m_s2", (0.0, 0.0, 0.0)))
        self._pos_noise_m = cfg_pos_noise
        self._vel_noise_m_s = cfg_vel_noise
        self._att_noise_deg = cfg_att_noise
        self._gyro_noise_deg_s = cfg_gyro_noise
        self._accel_noise_m_s2 = cfg_acc_noise
        ss = np.random.SeedSequence([int(sensor_seed), 70401])
        self._rng = np.random.default_rng(ss)

        # Exact 2nd-order Butterworth anti-aliasing filter at f_aa = 18 Hz (fs = 500 Hz)
        wn = float(self.settings.aa_cutoff_hz) / (0.5 * float(self.settings.imu_rate_hz))
        self._sos_aa = np.asarray(
            signal.butter(self.settings.aa_order, wn, btype="low", output="sos"), dtype=np.float64
        )
        # Single 2nd-order section scalar coefficients for ultra-fast 500 Hz DF-II-T filtering
        self._aa_b0 = float(self._sos_aa[0, 0])
        self._aa_b1 = float(self._sos_aa[0, 1])
        self._aa_b2 = float(self._sos_aa[0, 2])
        self._aa_a1 = float(self._sos_aa[0, 4])
        self._aa_a2 = float(self._sos_aa[0, 5])
        self._zi_unit = np.asarray(signal.sosfilt_zi(self._sos_aa), dtype=np.float64)
        self._aa_g_z0 = np.zeros(3, dtype=np.float64)
        self._aa_g_z1 = np.zeros(3, dtype=np.float64)
        self._aa_a_z0 = np.zeros(3, dtype=np.float64)
        self._aa_a_z1 = np.zeros(3, dtype=np.float64)
        self._imu_initialized = False
        self._last_gyro_aa = np.zeros(3, dtype=np.float64)
        self._last_accel_aa = np.array([0.0, 0.0, 9.80665], dtype=np.float64)

        # Fixed calibration misalignment & scale matrices (drawn once per flight)
        self._m_gyro = _nonortho_matrix(
            self.settings.gyro_nonortho_deg, self.settings.gyro_scale_err_ppm, self._rng
        )
        self._m_accel = _nonortho_matrix(
            self.settings.accel_nonortho_deg, self.settings.accel_scale_err_ppm, self._rng
        )
        self._gyro_bias_rad_s = np.array(
            [radians(float(v)) for v in cfg_gyro_bias], dtype=np.float64
        )
        self._accel_bias_m_s2 = np.asarray(cfg_acc_bias, dtype=np.float64)

        # GPS blackout timer state
        self._blackout_started_s: float | None = None
        self.gps_blackout_active: bool = False

        # Lagged chamber pressure & 1-step lagged fin resolver
        self._chamber_p_filt = np.zeros(7, dtype=np.float64)
        self._prev_fin_deflection: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._prop_gauge_filt_err: float = 0.0

    def step_imu_500hz(
        self,
        *,
        omega_true_body_rad_s: ArrayLike,
        accel_true_body_m_s2: ArrayLike,
        flex_omega_delta_rad_s: ArrayLike = (0.0, 0.0, 0.0),
        flex_accel_delta_m_s2: ArrayLike = (0.0, 0.0, 0.0),
        thrust_ratio: float = 0.0,
        dt_s: float = 0.002,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Synthesize one 500 Hz IMU sample and advance the 18 Hz Butterworth AA filter."""
        om = np.asarray(omega_true_body_rad_s, dtype=np.float64).reshape(3) + np.asarray(
            flex_omega_delta_rad_s, dtype=np.float64
        ).reshape(3)
        ac = np.asarray(accel_true_body_m_s2, dtype=np.float64).reshape(3) + np.asarray(
            flex_accel_delta_m_s2, dtype=np.float64
        ).reshape(3)

        # Scale noise PSD to 500 Hz so 18 Hz AA output matches configured 50 Hz noise floor
        s = self.settings
        bw_scale = sqrt(s.imu_rate_hz / (2.0 * max(s.aa_cutoff_hz, 1.0)))
        g_std = radians(self._gyro_noise_deg_s) * s.gyro_arw_scale * bw_scale
        a_std = self._accel_noise_m_s2 * s.accel_vrw_scale * bw_scale

        om_raw = self._m_gyro @ om + self._gyro_bias_rad_s
        if g_std > 0.0:
            om_raw = om_raw + self._rng.normal(0.0, g_std, size=3)

        ac_raw = self._m_accel @ ac + self._accel_bias_m_s2
        if a_std > 0.0:
            ac_raw = ac_raw + self._rng.normal(0.0, a_std, size=3)
        if thrust_ratio > 1e-3 and s.thrust_vib_rms_m_s2 > 0.0:
            ac_raw = ac_raw + self._rng.normal(
                0.0, s.thrust_vib_rms_m_s2 * min(1.0, thrust_ratio), size=3
            )

        if not self._imu_initialized:
            z0u = float(self._zi_unit[0, 0])
            z1u = float(self._zi_unit[0, 1])
            self._aa_g_z0 = z0u * om_raw
            self._aa_g_z1 = z1u * om_raw
            self._aa_a_z0 = z0u * ac_raw
            self._aa_a_z1 = z1u * ac_raw
            self._imu_initialized = True

        # Vectorized 3-axis DF-II-T biquad step (exact match to scipy.signal.sosfilt, ~50x faster)
        b0, b1, b2, a1, a2 = self._aa_b0, self._aa_b1, self._aa_b2, self._aa_a1, self._aa_a2
        om_aa = b0 * om_raw + self._aa_g_z0
        self._aa_g_z0 = b1 * om_raw - a1 * om_aa + self._aa_g_z1
        self._aa_g_z1 = b2 * om_raw - a2 * om_aa

        ac_aa = b0 * ac_raw + self._aa_a_z0
        self._aa_a_z0 = b1 * ac_raw - a1 * ac_aa + self._aa_a_z1
        self._aa_a_z1 = b2 * ac_raw - a2 * ac_aa

        self._last_gyro_aa = om_aa
        self._last_accel_aa = ac_aa
        _ = dt_s
        return om_aa, ac_aa

    def update_chamber_pressures(
        self, true_ratios_7: tuple[float, ...], dt_s: float
    ) -> tuple[float, float, float, float, float, float, float]:
        """Advance lagged normalized engine chamber pressure sensors (`P_c,meas / P_c,full`)."""
        alpha = 1.0 - exp(-max(dt_s, 1e-4) / max(self.settings.chamber_p_tau_s, 1e-3))
        tgt = np.asarray(true_ratios_7, dtype=np.float64).reshape(7)
        self._chamber_p_filt += alpha * (tgt - self._chamber_p_filt)
        noise = self._rng.normal(0.0, self.settings.chamber_p_noise_std, size=7)
        # Only add noise on non-zero / active pressures so off engines read clean ~0
        meas = np.where(
            self._chamber_p_filt > 1e-3, np.maximum(0.0, self._chamber_p_filt + noise), 0.0
        )
        return tuple(float(v) for v in meas)  # type: ignore[return-value]

    def sample_50hz(
        self,
        *,
        time_s: float,
        pos_com_enu_m: NDArray[np.float64],
        vel_com_enu_m_s: NDArray[np.float64],
        quat_wxyz: NDArray[np.float64],
        rot_wb: NDArray[np.float64],
        cg_station_m: float,
        dynamic_pressure_pa: float,
        heat_flux_w_m2: float,
        thrust_n: float,
        upper_attached: bool,
        on_pad: bool,
        pad_normal_enu: NDArray[np.float64],
        sloped_pad_enabled: bool,
        true_fin_deflection_rad: tuple[float, float, float, float],
        true_main_prop_kg: float,
        true_chamber_ratios_7: tuple[float, float, float, float, float, float, float],
        dt_ctrl_s: float = 0.02,
    ) -> dict[str, object]:
        """Produce all 50 Hz decimated v2 sensor channels (zero oracle leaks)."""
        s = self.settings
        alt_m = float(pos_com_enu_m[2])
        vz_m_s = float(vel_com_enu_m_s[2])

        # 1. GPS Plasma / Reentry Blackout state machine
        in_plasma = (not upper_attached) and (heat_flux_w_m2 >= s.gps_blackout_heat_flux_w_m2)
        in_alt_window = (
            (not upper_attached)
            and vz_m_s < -50.0
            and s.gps_blackout_alt_min_m <= alt_m <= s.gps_blackout_alt_max_m
        )
        if (
            s.gps_blackout_duration_s > 0.0
            and self._blackout_started_s is None
            and (in_plasma or in_alt_window)
        ):
            self._blackout_started_s = time_s

        blackout_timer = (
            self._blackout_started_s is not None
            and s.gps_blackout_duration_s > 0.0
            and (time_s - self._blackout_started_s) < s.gps_blackout_duration_s
        )
        self.gps_blackout_active = bool(in_plasma or blackout_timer)
        gps_valid = not self.gps_blackout_active

        if gps_valid:
            p_m = pos_com_enu_m + self._rng.normal(0.0, self._pos_noise_m, size=3)
            v_m = vel_com_enu_m_s + self._rng.normal(0.0, self._vel_noise_m_s, size=3)
            gps_pos = (float(p_m[0]), float(p_m[1]), float(p_m[2]))
            gps_vel = (float(v_m[0]), float(v_m[1]), float(v_m[2]))
        else:
            gps_pos = None
            gps_vel = None

        # 2. Causal attitude references: pad pre-launch alignment OR vacuum-coast star tracker ONLY
        att_valid = bool(on_pad)
        if att_valid:
            if self._att_noise_deg > 0.0:
                ang_n = self._rng.normal(0.0, radians(self._att_noise_deg), size=3)
                q_att = normalize_quat_wxyz(
                    quat_multiply_wxyz(
                        quat_wxyz, axis_angle_to_quat_wxyz(ang_n, float(np.linalg.norm(ang_n)))
                    ),
                    canonical=True,
                )
            else:
                q_att = quat_wxyz
            q_pkt = (float(q_att[0]), float(q_att[1]), float(q_att[2]), float(q_att[3]))
        else:
            # Never leak true quat_wxyz in flight when sensors_v2 is active!
            q_pkt = (1.0, 0.0, 0.0, 0.0)

        st_valid = (
            (not upper_attached)
            and (not on_pad)
            and dynamic_pressure_pa < 1.0
            and alt_m > 60_000.0
            and thrust_n <= 1.0
        )
        if st_valid:
            ang_st = self._rng.normal(0.0, radians(s.star_tracker_noise_deg), size=3)
            q_st = normalize_quat_wxyz(
                quat_multiply_wxyz(
                    quat_wxyz, axis_angle_to_quat_wxyz(ang_st, float(np.linalg.norm(ang_st)))
                ),
                canonical=True,
            )
            st_quat: tuple[float, float, float, float] | None = (
                float(q_st[0]),
                float(q_st[1]),
                float(q_st[2]),
                float(q_st[3]),
            )
        else:
            st_quat = None

        # 3. Hull-base mounted slant radar altimeter (z_radar_b = 0.0 m)
        cos_tilt = max(-1.0, min(1.0, float(rot_wb[2, 2])))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        radar_valid = False
        radar_range = 0.0
        if (not upper_attached) and alt_m < s.radar_max_alt_m and tilt_deg <= s.radar_max_tilt_deg:
            p_base = pos_com_enu_m - rot_wb @ np.array([0.0, 0.0, cg_station_m], dtype=np.float64)
            hit_ok, true_rng, _ = compute_base_slant_radar_range_m(
                pos_base_enu_m=p_base,
                rot_wb=rot_wb,
                pad_normal_enu=pad_normal_enu,
                pad_radius_m=s.pad_radius_m,
                upper_attached=upper_attached,
                sloped_pad_enabled=sloped_pad_enabled,
            )
            if hit_ok and true_rng <= s.radar_max_alt_m * 1.3:
                radar_valid = True
                sig_r = s.radar_sigma_m * s.radar_alt_noise_scale
                radar_range = max(0.0, true_rng + float(self._rng.normal(0.0, sig_r)))

        # 4. 1-step lagged fin resolver with +/-0.10 deg error
        f_sig = radians(s.fin_resolver_noise_deg)
        fin_meas = tuple(
            float(
                self._prev_fin_deflection[i]
                + (self._rng.normal(0.0, f_sig) if f_sig > 0.0 else 0.0)
            )
            for i in range(4)
        )
        self._prev_fin_deflection = true_fin_deflection_rad

        # 5. Propellant slosh-gauge noise (1st-order correlated +/-25 kg, clamped >= 0)
        if s.prop_gauge_noise_kg > 0.0 and true_main_prop_kg > 0.0:
            a_pg = exp(-dt_ctrl_s / 0.8)
            self._prop_gauge_filt_err = (
                a_pg * self._prop_gauge_filt_err
                + s.prop_gauge_noise_kg
                * sqrt(max(0.0, 1.0 - a_pg * a_pg))
                * float(self._rng.standard_normal())
            )
            prop_meas = max(
                0.0,
                true_main_prop_kg
                + max(
                    -2.5 * s.prop_gauge_noise_kg,
                    min(2.5 * s.prop_gauge_noise_kg, self._prop_gauge_filt_err),
                ),
            )
        else:
            prop_meas = true_main_prop_kg

        # 6. Normalized 7-engine chamber pressure ratios
        pc_ratios = self.update_chamber_pressures(true_chamber_ratios_7, dt_ctrl_s)

        return {
            "omega_body_rad_s": (
                float(self._last_gyro_aa[0]),
                float(self._last_gyro_aa[1]),
                float(self._last_gyro_aa[2]),
            ),
            "accel_body_m_s2": (
                float(self._last_accel_aa[0]),
                float(self._last_accel_aa[1]),
                float(self._last_accel_aa[2]),
            ),
            "gps_valid": gps_valid,
            "gps_pos_enu_m": gps_pos,
            "gps_vel_enu_m_s": gps_vel,
            "attitude_valid": att_valid,
            "quat_wxyz": q_pkt,
            "star_tracker_valid": st_valid,
            "star_tracker_quat_wxyz": st_quat,
            "radar_alt_valid": radar_valid,
            "radar_range_m": float(radar_range),
            "fin_deflection_rad": fin_meas,
            "main_propellant_kg": float(prop_meas),
            "engine_chamber_pressure_ratio_7": pc_ratios,
        }
