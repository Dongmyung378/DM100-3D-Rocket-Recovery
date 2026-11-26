"""Post-separation landing ground-effect cushion lift & lateral tilt perturbation (CDR Rev 3.1).

Key invariants:
  - Strictly gated on `not upper_attached` (NEVER fires on the launchpad flame trench at liftoff!).
  - Cluster shroud effective diameter `D_eff = 1.56 m` (`2.4 * D_eff = 3.744 m`).
  - C1 smoothstep taper to zero over `h_noz in [12.0, 15.0] m`.
  - Smoothly transitions between tilted pad normal (`r_xy <= 14 m`) and horizontal outer/ocean
    surface `(0, 0, 1)` at `z = 0` (`r_xy >= 16 m`).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _smoothstep(u: float) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True, slots=True)
class GroundEffect3DSettings:
    c_ge: float = 0.07
    c_lat: float = 0.25
    d_eff_m: float = 1.56
    taper_start_m: float = 12.0
    taper_end_m: float = 15.0
    pad_blend_inner_m: float = 14.0
    pad_blend_outer_m: float = 16.0


@dataclass(frozen=True, slots=True)
class GroundEffect3DSample:
    force_world_n: tuple[float, float, float]
    torque_body_nm: tuple[float, float, float]
    nozzle_height_m: float
    cushion_force_n: float


def ground_effect_nominal_lift_n(
    *,
    nozzle_height_m: float,
    thrust_n: float,
    c_ge_nom: float = 0.06,
    settings: GroundEffect3DSettings = GroundEffect3DSettings(),
) -> float:
    """Strictly causal nominal upward cushion force [N] for landing Vertical Body feedforward."""
    if thrust_n <= 1.0 or nozzle_height_m >= settings.taper_end_m:
        return 0.0
    w_ge = 1.0 - _smoothstep(
        (nozzle_height_m - settings.taper_start_m)
        / max(settings.taper_end_m - settings.taper_start_m, 1e-6)
    )
    if w_ge <= 1e-9:
        return 0.0
    h_clamped = max(0.0, float(nozzle_height_m))
    return float(w_ge * thrust_n * c_ge_nom * exp(-h_clamped / (2.4 * settings.d_eff_m)))


def compute_ground_effect_3d(
    *,
    pos_com_enu_m: ArrayLike,
    rot_wb: NDArray[np.float64],
    cg_station_m: float,
    nozzle_station_m: float,
    thrust_n: float,
    upper_attached: bool,
    pad_normal_enu: ArrayLike = (0.0, 0.0, 1.0),
    pad_origin_enu_m: ArrayLike = (0.0, 0.0, 0.0),
    settings: GroundEffect3DSettings = GroundEffect3DSettings(),
) -> GroundEffect3DSample:
    """Evaluate landing exhaust ground-effect world force & body torque."""
    if upper_attached or thrust_n <= 1.0 or settings.c_ge <= 0.0:
        return GroundEffect3DSample(
            force_world_n=(0.0, 0.0, 0.0),
            torque_body_nm=(0.0, 0.0, 0.0),
            nozzle_height_m=float("inf"),
            cushion_force_n=0.0,
        )

    p_com = np.asarray(pos_com_enu_m, dtype=np.float64).reshape(3)
    r_noz_b = np.array([0.0, 0.0, float(nozzle_station_m) - float(cg_station_m)], dtype=np.float64)
    p_noz = p_com + rot_wb @ r_noz_b

    p_pad = np.asarray(pad_origin_enu_m, dtype=np.float64).reshape(3)
    n_pad = np.asarray(pad_normal_enu, dtype=np.float64).reshape(3)
    n_pad_norm = float(np.linalg.norm(n_pad))
    n_pad = n_pad / n_pad_norm if n_pad_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    n_flat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Smooth C1 transition from tilted finite pad (r <= 14 m) to flat outer terrain (r >= 16 m)
    r_xy = hypot(float(p_noz[0] - p_pad[0]), float(p_noz[1] - p_pad[1]))
    blend_outer = _smoothstep(
        (r_xy - settings.pad_blend_inner_m)
        / max(settings.pad_blend_outer_m - settings.pad_blend_inner_m, 1e-6)
    )
    n_eff = (1.0 - blend_outer) * n_pad + blend_outer * n_flat
    n_eff /= float(np.linalg.norm(n_eff))

    h_pad = float((p_noz - p_pad) @ n_pad)
    h_flat = float(p_noz[2])
    h_noz = (1.0 - blend_outer) * h_pad + blend_outer * h_flat

    if h_noz >= settings.taper_end_m:
        return GroundEffect3DSample(
            force_world_n=(0.0, 0.0, 0.0),
            torque_body_nm=(0.0, 0.0, 0.0),
            nozzle_height_m=h_noz,
            cushion_force_n=0.0,
        )

    w_ge = 1.0 - _smoothstep(
        (h_noz - settings.taper_start_m) / max(settings.taper_end_m - settings.taper_start_m, 1e-6)
    )
    f_mag = w_ge * thrust_n * settings.c_ge * exp(-max(0.0, h_noz) / (2.4 * settings.d_eff_m))

    zb_w = rot_wb[:, 2]
    zb_tan = zb_w - float(zb_w @ n_eff) * n_eff
    f_world = f_mag * (n_eff + settings.c_lat * zb_tan)
    f_body = rot_wb.T @ f_world
    tau_body = np.cross(r_noz_b, f_body)

    return GroundEffect3DSample(
        force_world_n=(float(f_world[0]), float(f_world[1]), float(f_world[2])),
        torque_body_nm=(float(tau_body[0]), float(tau_body[1]), float(tau_body[2])),
        nozzle_height_m=float(h_noz),
        cushion_force_n=float(f_mag),
    )
