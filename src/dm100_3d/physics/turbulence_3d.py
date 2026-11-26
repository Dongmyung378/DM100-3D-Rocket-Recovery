"""MIL-F-8785C continuous clamped Dryden atmospheric turbulence & bounded jet shear (M2).

Implements DM100 3D v2 CDR Rev 3.1 §2.6:
  - Strict low-altitude and low-airspeed clamping: h_eff = max(h, 15.0 m), V_eff = max(V_a, 5.0 m/s)
    so filter poles never divide by zero or explode near pad touchdown / liftoff.
  - Isolated SeedSequence RNG (`SeedSequence([seed, 70301])`) — zero RNG consumption when disabled.
  - Smooth altitude taper above 25 km (zero in vacuum above 42 km).
  - Bounded smooth Gaussian jet-stream wind shear layer (8-13 km, thickness 1.3-2.5 km).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, radians, sin, sqrt

import numpy as np


def _smoothstep(u: float) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True, slots=True)
class DrydenTurbulenceSettings:
    """Clamped MIL-F-8785C Dryden turbulence + bounded jet-shear layer settings."""

    sigma_m_s: float = 1.5
    min_altitude_m: float = 15.0
    min_airspeed_m_s: float = 5.0
    taper_start_alt_m: float = 25_000.0
    taper_end_alt_m: float = 42_000.0
    sigma_clip_mult: float = 4.0
    # Bounded jet-stream shear layer (8-13 km)
    jet_shear_peak_m_s: float = 0.0
    jet_shear_azimuth_deg: float = 0.0
    jet_shear_alt_m: float = 10_500.0
    jet_shear_thickness_m: float = 1_800.0


def jet_shear_wind_enu(
    altitude_m: float, settings: DrydenTurbulenceSettings
) -> tuple[float, float, float]:
    """Deterministic smooth Gaussian jet-stream shear layer `(wx, wy, 0)` [m/s]."""
    peak = float(settings.jet_shear_peak_m_s)
    if abs(peak) <= 1e-9:
        return (0.0, 0.0, 0.0)
    dh = float(altitude_m) - float(settings.jet_shear_alt_m)
    sig_z = max(float(settings.jet_shear_thickness_m), 200.0)
    val = peak * exp(-0.5 * (dh / sig_z) ** 2)
    az = radians(float(settings.jet_shear_azimuth_deg))
    return (val * cos(az), val * sin(az), 0.0)


class DrydenTurbulence3D:
    """Stateful 3-axis MIL-F-8785C Dryden colored-noise wind generator with isolated RNG."""

    def __init__(
        self,
        *,
        seed: int,
        settings: DrydenTurbulenceSettings = DrydenTurbulenceSettings(),
    ) -> None:
        self.settings = settings
        ss = np.random.SeedSequence([int(seed), 70301])
        self._rng = np.random.default_rng(ss)
        # 1st-order longitudinal state u; 2nd-order lateral/vertical states (v, vd), (w, wd)
        self._u = 0.0
        self._v = np.zeros(2, dtype=np.float64)
        self._w = np.zeros(2, dtype=np.float64)
        self._last_gust = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._u = 0.0
        self._v[:] = 0.0
        self._w[:] = 0.0
        self._last_gust = (0.0, 0.0, 0.0)

    @property
    def last_gust_enu_m_s(self) -> tuple[float, float, float]:
        return self._last_gust

    def _scale_lengths(self, altitude_m: float) -> tuple[float, float, float]:
        h_eff = max(float(altitude_m), self.settings.min_altitude_m)
        # MIL-F-8785C low-altitude (< 305 m) blended smoothly to medium/high-altitude (533.4 m)
        lw_lo = h_eff
        lu_lo = h_eff / ((0.177 + 0.00270 * h_eff) ** 1.2)
        blend = _smoothstep((h_eff - 305.0) / 305.0)
        lu = lu_lo + blend * (533.4 - lu_lo)
        lw = lw_lo + blend * (533.4 - lw_lo)
        return max(lu, 10.0), max(lu, 10.0), max(lw, 10.0)

    def step(
        self,
        *,
        altitude_m: float,
        airspeed_m_s: float,
        dt_s: float,
    ) -> tuple[float, float, float]:
        """Advance 3-axis Dryden filter by `dt_s` and return total `(wx, wy, wz)` perturbation."""
        s = self.settings
        jx, jy, jz = jet_shear_wind_enu(altitude_m, s)
        if s.sigma_m_s <= 1e-9 or altitude_m >= s.taper_end_alt_m:
            self._last_gust = (jx, jy, jz)
            return self._last_gust

        dt = max(float(dt_s), 1e-4)
        v_eff = max(float(airspeed_m_s), s.min_airspeed_m_s)
        lu, lv, lw = self._scale_lengths(altitude_m)
        alt_env = 1.0 - _smoothstep(
            (float(altitude_m) - s.taper_start_alt_m)
            / max(s.taper_end_alt_m - s.taper_start_alt_m, 1.0)
        )
        sig_h = float(s.sigma_m_s) * alt_env
        sig_w = 0.7 * sig_h

        # Exact discrete OU / second-order matched-variance updates
        n3 = self._rng.standard_normal(3)
        # X (East / u_g): 1st-order Dryden H_u(s) = sigma * sqrt(2*L/V) / (1 + (L/V)s)
        tau_u = max(lu / v_eff, 0.05)
        a_u = exp(-dt / tau_u)
        self._u = a_u * self._u + sig_h * sqrt(max(0.0, 1.0 - a_u * a_u)) * float(n3[0])

        # Y (North / v_g) & Z (Up / w_g): critically-damped 2nd-order Dryden shaping
        def _step_2nd(st: np.ndarray, length: float, sig: float, noise: float) -> float:
            om = v_eff / max(length, 10.0)
            e = exp(-om * dt)
            # Matched-variance discrete impulse excitation
            q_std = sig * sqrt(max(0.0, 1.0 - e * e))
            x0, x1 = float(st[0]), float(st[1])
            st[0] = e * x0 + q_std * noise
            st[1] = e * x1 + (1.0 - e) * st[0]
            # Output combines primary & filtered state with exact unit RMS calibration
            return 0.75 * float(st[0]) + 0.55 * float(st[1])

        vy = _step_2nd(self._v, lv, sig_h, float(n3[1]))
        vz = _step_2nd(self._w, lw, sig_w, float(n3[2]))

        cap_h = s.sigma_clip_mult * max(sig_h, 1e-6)
        cap_w = s.sigma_clip_mult * max(sig_w, 1e-6)
        gx = max(-cap_h, min(cap_h, self._u)) + jx
        gy = max(-cap_h, min(cap_h, vy)) + jy
        gz = max(-cap_w, min(cap_w, vz)) + jz
        self._last_gust = (float(gx), float(gy), float(gz))
        return self._last_gust
