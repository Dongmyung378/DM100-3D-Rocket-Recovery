"""C1-smooth Supersonic Retropropulsion (SRP) plume-aero interference & buffeting model.

Based on NASA Langley / Korzun-Braun empirical correlations (CDR Rev 3.1):
  - C1 smoothstep gating over Mach in [0.95, 1.35] and retrograde axial flow [-80, -30] m/s.
  - Axial drag shielding: C_D,eff = C_D * (1 - w_srp * eta_srp(M) * C_T / (C_T + 1.4)),
    with eta_srp(M) in [0.25, 0.58] (25-58% shielding during supersonic entry burn).
  - Center-of-pressure shift: delta_z_cp = -1.4 m * w_srp * C_T / (C_T + 2.0).
  - Deterministic isolated-RNG 1-6 Hz bandpass buffeting torque bounded by C_{m,rms,peak}.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np


def _smoothstep(u: float) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True, slots=True)
class Srp3DSettings:
    mach_low: float = 0.95
    mach_high: float = 1.35
    retro_speed_low_m_s: float = 30.0
    retro_speed_high_m_s: float = 80.0
    eta_min: float = 0.25
    eta_max: float = 0.58
    ct_drag_knee: float = 1.4
    ct_cp_knee: float = 2.0
    max_cp_shift_m: float = -1.4
    shield_scale: float = 1.0
    buffet_enabled: bool = True
    cm_rms_peak: float = 0.0055
    buffet_f_low_hz: float = 1.0
    buffet_f_high_hz: float = 6.0


@dataclass(frozen=True, slots=True)
class Srp3DEffect:
    weight: float
    thrust_coeff_ct: float
    drag_multiplier: float
    delta_z_cp_m: float
    cm_rms: float
    buffet_torque_body_nm: tuple[float, float, float]


class SrpBuffetFilter:
    """2-axis deterministic 1-6 Hz bandpass filter on an isolated SeedSequence RNG."""

    def __init__(self, *, seed: int, f_low_hz: float = 1.0, f_high_hz: float = 6.0) -> None:
        ss = np.random.SeedSequence([int(seed), 70201])
        self._rng = np.random.default_rng(ss)
        self._f_low = float(f_low_hz)
        self._f_high = float(f_high_hz)
        # First-order low-pass + high-pass cascade per axis (X, Y)
        self._lp = np.zeros(2, dtype=np.float64)
        self._hp_prev_in = np.zeros(2, dtype=np.float64)
        self._hp_out = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self._lp[:] = 0.0
        self._hp_prev_in[:] = 0.0
        self._hp_out[:] = 0.0

    def step(self, dt_s: float) -> tuple[float, float]:
        """Advance unit-RMS 1-6 Hz bandpass noise vector `(xi_x, xi_y)` by `dt_s`."""
        dt = max(float(dt_s), 1e-4)
        w_hi = 2.0 * pi * self._f_high
        w_lo = 2.0 * pi * self._f_low
        a_lp = (w_hi * dt) / (1.0 + w_hi * dt)
        a_hp = 1.0 / (1.0 + w_lo * dt)

        # Unit-variance continuous-time scaled white excitation matched to bandpass RMS
        bw_hz = max(self._f_high - self._f_low, 0.5)
        gain_rms = sqrt((1.0 / (2.0 * dt)) / (0.5 * pi * bw_hz))
        raw = self._rng.standard_normal(2) * gain_rms

        # Low-pass at f_high, then high-pass at f_low
        self._lp += a_lp * (raw - self._lp)
        hp_new = a_hp * (self._hp_out + self._lp - self._hp_prev_in)
        self._hp_prev_in = self._lp.copy()
        self._hp_out = hp_new
        # Clip extreme >3.5-sigma stochastic outliers for bounded envelope integrity
        clipped = np.clip(self._hp_out, -3.5, 3.5)
        return float(clipped[0]), float(clipped[1])


def srp_gating_weight(
    mach: float,
    v_air_zb_m_s: float,
    thrust_n: float,
    *,
    upper_attached: bool = False,
    settings: Srp3DSettings = Srp3DSettings(),
) -> float:
    """Compute C1-smooth SRP activation weight in [0, 1]."""
    if upper_attached or thrust_n <= 1.0:
        return 0.0
    w_m = _smoothstep(
        (mach - settings.mach_low) / max(settings.mach_high - settings.mach_low, 1e-6)
    )
    retro_spd = -float(v_air_zb_m_s)
    w_v = _smoothstep(
        (retro_spd - settings.retro_speed_low_m_s)
        / max(settings.retro_speed_high_m_s - settings.retro_speed_low_m_s, 1e-6)
    )
    return w_m * w_v


def srp_nominal_drag_multiplier(
    *,
    mach: float,
    v_air_zb_m_s: float,
    thrust_n: float,
    dynamic_pressure_pa: float,
    body_radius_m: float,
    eta_nom: float = 0.40,
    upper_attached: bool = False,
) -> float:
    """Strictly causal nominal SRP axial drag multiplier for GNC entry-burn prediction."""
    if upper_attached or thrust_n <= 1.0 or dynamic_pressure_pa <= 1.0:
        return 1.0
    w = srp_gating_weight(mach, v_air_zb_m_s, thrust_n, upper_attached=upper_attached)
    if w <= 1e-9:
        return 1.0
    s_ref = pi * body_radius_m * body_radius_m
    c_t = thrust_n / max(dynamic_pressure_pa * s_ref, 1.0)
    mult = 1.0 - w * eta_nom * (c_t / (c_t + 1.4))
    return max(0.15, min(1.0, mult))


def compute_srp_effects(
    *,
    mach: float,
    v_air_zb_m_s: float,
    thrust_n: float,
    dynamic_pressure_pa: float,
    body_radius_m: float,
    dt_s: float = 0.01,
    upper_attached: bool = False,
    settings: Srp3DSettings = Srp3DSettings(),
    buffet_filter: SrpBuffetFilter | None = None,
) -> Srp3DEffect:
    """Evaluate C1-smooth SRP drag shielding, CP shift, and bounded 1-6 Hz buffeting torque."""
    w = srp_gating_weight(
        mach, v_air_zb_m_s, thrust_n, upper_attached=upper_attached, settings=settings
    )
    if w <= 1e-9 or dynamic_pressure_pa <= 1.0:
        return Srp3DEffect(
            weight=0.0,
            thrust_coeff_ct=0.0,
            drag_multiplier=1.0,
            delta_z_cp_m=0.0,
            cm_rms=0.0,
            buffet_torque_body_nm=(0.0, 0.0, 0.0),
        )

    s_ref = pi * body_radius_m * body_radius_m
    diam = 2.0 * body_radius_m
    c_t = thrust_n / max(dynamic_pressure_pa * s_ref, 1.0)

    eta_base = settings.eta_min + (settings.eta_max - settings.eta_min) * _smoothstep(
        (mach - 1.0) / 2.2
    )
    eta_eff = max(0.0, min(0.85, eta_base * settings.shield_scale))
    drag_mult = max(0.15, 1.0 - w * eta_eff * (c_t / (c_t + settings.ct_drag_knee)))
    dz_cp = settings.max_cp_shift_m * w * (c_t / (c_t + settings.ct_cp_knee))

    # Peak buffeting around C_T ~= 2.0, decaying cleanly at very low or very high C_T
    ct_shape = (4.0 * c_t) / ((c_t + 2.0) ** 2)
    cm_rms = settings.cm_rms_peak * ct_shape

    if settings.buffet_enabled and buffet_filter is not None and cm_rms > 1e-9:
        xi_x, xi_y = buffet_filter.step(dt_s)
        qsd = dynamic_pressure_pa * s_ref * diam
        amp = w * qsd * cm_rms
        buffet_tau = (amp * xi_x, amp * xi_y, 0.0)
    else:
        buffet_tau = (0.0, 0.0, 0.0)

    return Srp3DEffect(
        weight=float(w),
        thrust_coeff_ct=float(c_t),
        drag_multiplier=float(drag_mult),
        delta_z_cp_m=float(dz_cp),
        cm_rms=float(cm_rms),
        buffet_torque_body_nm=buffet_tau,
    )
