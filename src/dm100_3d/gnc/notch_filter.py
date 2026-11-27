"""Single-path 4th-order Elliptic (Cauer) CSI structural filter & Matched DOB clone (M2).

Implements DM100 3D v2 CDR Rev 3.1 §3.1:
  - Pitch / Yaw ONLY 4th-order Elliptic LPF (Rp = 0.5 dB, Rs = 42.0 dB):
      * Ascent stack (33 m): fc = 1.20 Hz
      * Booster alone (24 m): fc = 2.25 Hz
    Designed via pre-warped bilinear transform (`scipy.signal.ellip(..., output='sos')`) at 50 Hz.
  - Roll axis (body Z) strictly BYPASSED (0 flex coupling -> maximizes roll phase margin).
  - Bumpless steady-state initialization (`sosfilt_zi`) across stage separation.
  - Identical SOS filter instance (`create_matched_dob_filter`) applied to model torque inside
    the torque-bias observer (Matched DOB) to preserve exact rigid-body self-cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal


@dataclass(frozen=True, slots=True)
class StructuralFilterConfig:
    """Locked CDR Rev 3.1 single-path CSI structural filter specification."""

    sample_rate_hz: float = 50.0
    order: int = 4
    passband_ripple_db: float = 0.5
    stopband_atten_db: float = 42.0
    ascent_cutoff_hz: float = 1.20
    booster_cutoff_hz: float = 2.25


def design_elliptic_lpf_sos(
    cutoff_hz: float,
    *,
    sample_rate_hz: float = 50.0,
    order: int = 4,
    rp_db: float = 0.5,
    rs_db: float = 42.0,
) -> NDArray[np.float64]:
    """Design digital 4th-order Elliptic LPF in SOS form (0 dB passband ceiling, CDR Rev 3.1)."""
    wn = float(cutoff_hz) / (0.5 * float(sample_rate_hz))
    if not (0.0 < wn < 1.0):
        raise ValueError(f"Normalized cutoff {wn:.4f} out of (0, 1) for fc={cutoff_hz} Hz")
    return np.asarray(
        signal.ellip(order, rp_db, rs_db, wn, btype="low", analog=False, output="sos"),
        dtype=np.float64,
    )


def _sos_step_scalar(sos: NDArray[np.float64], x: float, zi: NDArray[np.float64]) -> float:
    """Fast scalar Direct-Form II Transposed SOS step (identical to scipy.signal.sosfilt)."""
    y = float(x)
    for k in range(sos.shape[0]):
        b0, b1, b2, _, a1, a2 = (
            float(sos[k, 0]),
            float(sos[k, 1]),
            float(sos[k, 2]),
            float(sos[k, 3]),
            float(sos[k, 4]),
            float(sos[k, 5]),
        )
        z0 = float(zi[k, 0])
        z1 = float(zi[k, 1])
        out = b0 * y + z0
        zi[k, 0] = b1 * y - a1 * out + z1
        zi[k, 1] = b2 * y - a2 * out
        y = out
    return y


class StructuralFilter3D:
    """Single-path Pitch/Yaw 4th-order Elliptic LPF with Roll bypass & bumpless regime switch."""

    def __init__(
        self,
        config: StructuralFilterConfig = StructuralFilterConfig(),
        *,
        upper_attached: bool = True,
    ) -> None:
        self.config = config
        self.sos_ascent = design_elliptic_lpf_sos(
            config.ascent_cutoff_hz,
            sample_rate_hz=config.sample_rate_hz,
            order=config.order,
            rp_db=config.passband_ripple_db,
            rs_db=config.stopband_atten_db,
        )
        self.sos_booster = design_elliptic_lpf_sos(
            config.booster_cutoff_hz,
            sample_rate_hz=config.sample_rate_hz,
            order=config.order,
            rp_db=config.passband_ripple_db,
            rs_db=config.stopband_atten_db,
        )
        self._zi_unit_ascent = np.asarray(signal.sosfilt_zi(self.sos_ascent), dtype=np.float64)
        self._zi_unit_booster = np.asarray(signal.sosfilt_zi(self.sos_booster), dtype=np.float64)
        self._upper_attached = bool(upper_attached)
        self._initialized = False
        self._zi_x = np.zeros_like(self._zi_unit_ascent)
        self._zi_y = np.zeros_like(self._zi_unit_ascent)

    @property
    def active_sos(self) -> NDArray[np.float64]:
        return self.sos_ascent if self._upper_attached else self.sos_booster

    @property
    def active_cutoff_hz(self) -> float:
        return (
            self.config.ascent_cutoff_hz if self._upper_attached else self.config.booster_cutoff_hz
        )

    def clone(self) -> StructuralFilter3D:
        """Return a fresh independent copy with identical SOS coefficients (for Matched DOB)."""
        return StructuralFilter3D(self.config, upper_attached=self._upper_attached)

    def reset(
        self,
        initial_xy: tuple[float, float] = (0.0, 0.0),
        *,
        upper_attached: bool | None = None,
    ) -> None:
        """Reset filter memory to exact DC equilibrium at `initial_xy`."""
        if upper_attached is not None:
            self._upper_attached = bool(upper_attached)
        zi_u = self._zi_unit_ascent if self._upper_attached else self._zi_unit_booster
        self._zi_x = zi_u * float(initial_xy[0])
        self._zi_y = zi_u * float(initial_xy[1])
        self._initialized = True

    def set_regime(
        self, *, upper_attached: bool, bumpless_xy: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """Switch between ascent (1.20 Hz) and booster (2.25 Hz) with bumpless DC continuity."""
        new_flag = bool(upper_attached)
        if new_flag != self._upper_attached:
            self._upper_attached = new_flag
            zi_u = self._zi_unit_ascent if self._upper_attached else self._zi_unit_booster
            self._zi_x = zi_u * float(bumpless_xy[0])
            self._zi_y = zi_u * float(bumpless_xy[1])
            self._initialized = True

    def filter_sample(
        self,
        vec3: ArrayLike,
        *,
        upper_attached: bool | None = None,
    ) -> NDArray[np.float64]:
        """Filter 3-vector `(x, y, z)`: apply Elliptic LPF to `(x, y)` and bypass `z` (Roll)."""
        v = np.asarray(vec3, dtype=np.float64).reshape(3)
        if upper_attached is not None and bool(upper_attached) != self._upper_attached:
            self.set_regime(
                upper_attached=bool(upper_attached), bumpless_xy=(float(v[0]), float(v[1]))
            )
        if not self._initialized:
            self.reset((float(v[0]), float(v[1])))

        sos = self.active_sos
        yx = _sos_step_scalar(sos, float(v[0]), self._zi_x)
        yy = _sos_step_scalar(sos, float(v[1]), self._zi_y)
        return np.array([yx, yy, float(v[2])], dtype=np.float64)
