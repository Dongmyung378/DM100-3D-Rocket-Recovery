"""3D US Standard Atmosphere LUT and 3D wind/shear/gust field (East X, North Y, Vertical Z).

Matches DM100 2D atmosphere & wind identically when crosswind_y == 0 and gust_azimuth == 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, log, pi, sin

from dm100_3d.config import G0, R_AIR

PROFILE_TOP_M = 100_000.0

# 1976 US Standard Atmosphere base layers: (altitude_m, base_temp_K, lapse_K_per_m)
_STD_LAYERS: tuple[tuple[float, float, float], ...] = (
    (0.0, 288.15, -0.0065),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 0.0010),
    (32_000.0, 228.65, 0.0028),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -0.0028),
    (71_000.0, 214.65, -0.0020),
    (84_852.0, 186.946, 0.0),
)


@dataclass(frozen=True, slots=True)
class Atmosphere3DSettings:
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101_325.0
    latitude_deg: float = 37.0
    coriolis_enabled: bool = True
    planar_coriolis_only: bool = False
    density_scale: float = 1.0
    upper_air_density_scale: float = 1.0


def _upper_air_density_factor(altitude_m: float, upper_scale: float) -> float:
    """Smoothstep upper-air density multiplier for h > 10 km (zero double-counting with T0/P0)."""
    if upper_scale == 1.0 or altitude_m <= 10_000.0:
        return 1.0
    u = min(1.0, (altitude_m - 10_000.0) / 4_000.0)
    return 1.0 + (upper_scale - 1.0) * (u * u * (3.0 - 2.0 * u))


def standard_atmosphere(
    altitude_m: float,
    settings: Atmosphere3DSettings | None = None,
) -> tuple[float, float, float]:
    """Return (temperature_K, pressure_Pa, density_kg_m3) matching DM100 2D atmosphere."""
    s = settings or Atmosphere3DSettings()
    h = max(0.0, float(altitude_m))
    t_offset = s.sea_level_temperature_k - 288.15
    p_pa = s.sea_level_pressure_pa

    for idx, (h0, t0_std, lapse) in enumerate(_STD_LAYERS):
        t0 = max(120.0, t0_std + t_offset)
        h_next = _STD_LAYERS[idx + 1][0] if idx + 1 < len(_STD_LAYERS) else float("inf")
        if h <= h_next:
            dh = h - h0
            t_k = max(120.0, t0 + lapse * dh)
            if abs(lapse) < 1e-12:
                p_pa *= exp(-G0 * dh / (R_AIR * t0))
            else:
                p_pa *= (t_k / t0) ** (-G0 / (R_AIR * lapse))
            rho = (p_pa / (R_AIR * t_k)) * s.density_scale
            if s.upper_air_density_scale != 1.0:
                rho *= _upper_air_density_factor(h, s.upper_air_density_scale)
            return t_k, p_pa, rho
        dh_full = h_next - h0
        t_next = max(120.0, t0 + lapse * dh_full)
        if abs(lapse) < 1e-12:
            p_pa *= exp(-G0 * dh_full / (R_AIR * t0))
        else:
            p_pa *= (t_next / t0) ** (-G0 / (R_AIR * lapse))

    t_top = max(120.0, _STD_LAYERS[-1][1] + t_offset)
    rho = (p_pa / (R_AIR * t_top)) * s.density_scale
    if s.upper_air_density_scale != 1.0:
        rho *= _upper_air_density_factor(h, s.upper_air_density_scale)
    return t_top, p_pa, rho


class AtmosphereLUT:
    """Fast scalar 100 m piecewise log-pressure atmosphere lookup table."""

    __slots__ = ("_dz", "_inv_dz", "_max_h", "_n", "_t", "_log_p", "_log_rho", "_rho")

    def __init__(
        self,
        settings: Atmosphere3DSettings | None = None,
        step_m: float = 100.0,
    ) -> None:
        self._dz = float(step_m)
        self._inv_dz = 1.0 / self._dz
        self._max_h = PROFILE_TOP_M
        n = int(round(self._max_h / self._dz)) + 1
        self._n = n
        self._t = [0.0] * n
        self._log_p = [0.0] * n
        self._log_rho = [0.0] * n
        self._rho = [0.0] * n
        s = settings or Atmosphere3DSettings()
        for i in range(n):
            t_k, p_pa, rho = standard_atmosphere(i * self._dz, s)
            self._t[i] = t_k
            self._log_p[i] = log(max(p_pa, 1e-30))
            self._rho[i] = rho
            self._log_rho[i] = log(max(rho, 1e-30))

    def at(self, altitude_m: float) -> tuple[float, float, float]:
        if altitude_m <= 0.0:
            return self._t[0], exp(self._log_p[0]), self._rho[0]
        if altitude_m >= self._max_h:
            last = self._n - 1
            t_top = self._t[last]
            extra = altitude_m - self._max_h
            lp_top = self._log_p[last] - G0 * extra / (R_AIR * t_top)
            p_top = exp(lp_top)
            rho_top = self._rho[last] * exp(lp_top - self._log_p[last])
            return t_top, p_top, rho_top
        pos = altitude_m * self._inv_dz
        idx = int(pos)
        if idx >= self._n - 1:
            last = self._n - 1
            return self._t[last], exp(self._log_p[last]), self._rho[last]
        frac = pos - idx
        t_k = self._t[idx] + frac * (self._t[idx + 1] - self._t[idx])
        p_pa = exp(self._log_p[idx] + frac * (self._log_p[idx + 1] - self._log_p[idx]))
        rho = exp(self._log_rho[idx] + frac * (self._log_rho[idx + 1] - self._log_rho[idx]))
        return t_k, p_pa, rho


@dataclass(frozen=True, slots=True)
class Wind3DConfig:
    """3D wind profiles (East X, North Y, Vertical Z) + compact smoothstep 3D gust."""

    wind_x_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    wind_y_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    wind_z_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    forecast_wind_x_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    forecast_wind_y_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    gust_amplitude_m_s: float = 0.0
    gust_azimuth_rad: float = 0.0  # 0 = pure East (+X), pi/2 = pure North (+Y)
    gust_period_s: float = 16.0
    gust_phase_rad: float = 0.0
    gust_start_s: float = 0.0
    gust_duration_s: float = 100.0
    gust_altitude_center_m: float = 10_000.0
    gust_altitude_half_width_m: float = 4_000.0


class Wind3DLUT:
    """Fast scalar 3D wind evaluator `(wx, wy, wz)` for Plant (actual) and GNC (forecast)."""

    __slots__ = (
        "_zx",
        "_wx",
        "_zy",
        "_wy",
        "_zz",
        "_wz",
        "_gust_amp",
        "_gust_cos",
        "_gust_sin",
        "_gust_omega",
        "_gust_phase",
        "_gust_start",
        "_gust_end",
        "_gust_dur",
        "_gust_zc",
        "_gust_zhw",
        "_forecast",
    )

    def __init__(self, config: Wind3DConfig | None = None, *, forecast: bool = False) -> None:
        c = config or Wind3DConfig()
        px = c.forecast_wind_x_profile if forecast else c.wind_x_profile
        py = c.forecast_wind_y_profile if forecast else c.wind_y_profile
        pz = c.wind_z_profile
        self._zx = tuple(p[0] for p in px)
        self._wx = tuple(p[1] for p in px)
        self._zy = tuple(p[0] for p in py)
        self._wy = tuple(p[1] for p in py)
        self._zz = tuple(p[0] for p in pz)
        self._wz = tuple(p[1] for p in pz)
        self._forecast = forecast
        self._gust_amp = 0.0 if forecast else float(c.gust_amplitude_m_s)
        self._gust_cos = cos(c.gust_azimuth_rad)
        self._gust_sin = sin(c.gust_azimuth_rad)
        self._gust_omega = 2.0 * pi / max(c.gust_period_s, 1e-3)
        self._gust_phase = float(c.gust_phase_rad)
        self._gust_start = float(c.gust_start_s)
        self._gust_dur = max(float(c.gust_duration_s), 1e-3)
        self._gust_end = self._gust_start + self._gust_dur
        self._gust_zc = float(c.gust_altitude_center_m)
        self._gust_zhw = max(float(c.gust_altitude_half_width_m), 1e-3)

    @staticmethod
    def _interp(zs: tuple[float, ...], vs: tuple[float, ...], z: float) -> float:
        if z <= zs[0]:
            return vs[0]
        if z >= zs[-1]:
            return vs[-1]
        for i in range(len(zs) - 1):
            z1 = zs[i + 1]
            if z <= z1:
                z0 = zs[i]
                return vs[i] + (z - z0) / (z1 - z0) * (vs[i + 1] - vs[i])
        return vs[-1]

    @staticmethod
    def _smoothstep(u: float) -> float:
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return u * u * (3.0 - 2.0 * u)

    def at(self, altitude_m: float, time_s: float) -> tuple[float, float, float]:
        wx = self._interp(self._zx, self._wx, altitude_m)
        wy = self._interp(self._zy, self._wy, altitude_m)
        wz = 0.0 if self._forecast else self._interp(self._zz, self._wz, altitude_m)
        if self._gust_amp != 0.0 and self._gust_start < time_s < self._gust_end:
            dz = abs(altitude_m - self._gust_zc)
            if dz < self._gust_zhw:
                z_env = self._smoothstep(1.0 - dz / self._gust_zhw)
                t_unit = (time_s - self._gust_start) / self._gust_dur
                t_env = self._smoothstep(2.0 * min(t_unit, 1.0 - t_unit))
                g_val = (
                    self._gust_amp
                    * z_env
                    * t_env
                    * sin(self._gust_omega * (time_s - self._gust_start) + self._gust_phase)
                )
                wx += g_val * self._gust_cos
                wy += g_val * self._gust_sin
        return wx, wy, wz
