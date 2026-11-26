"""50 Hz dm100_replay_3d_v1 exporter, loader, validator, and COM->base resolver.

Preserves 100% compatibility with DM100 2D golden replays (`calm_replay_50hz.json`,
`example_weather_replay_50hz.json`) while supporting native 3D hull-base coordinates
(`base_pos_enu_m`, `cg_station_m`) for drift-free Unreal Engine 5.4 rendering.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dm100_3d.config import DM100VehicleConfig, mass_properties_3d
from dm100_3d.coords import com_to_base_pos_enu
from dm100_3d.interfaces import ReplayFrame3D

REPLAY_SCHEMA_VERSION = "dm100_replay_3d_v1"
EXPECTED_COORDINATE_FRAME = "ENU_right_handed_SI"


def export_replay_json(
    frames: list[ReplayFrame3D],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    indent: int | None = None,
) -> Path:
    """Write a self-describing dm100_replay_3d_v1 JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "coordinate_frame": EXPECTED_COORDINATE_FRAME,
        "units": {"length": "m", "angle": "rad", "quaternion_order": "wxyz"},
        "ue5_conversion_note": (
            "Map ENU (x, y, z) [m] -> UE5 (x*100, -y*100, z*100) [cm]; "
            "R_UE5 = S @ R_ENU @ S^-1 with S = diag(1, -1, 1); "
            "q_UE5_xyzw = (-qx, +qy, -qz, +qw)."
        ),
        "frame_count": len(frames),
        "metadata": metadata or {},
        "frames": [frame.to_dict() for frame in frames],
    }
    text = (
        json.dumps(payload, separators=(",", ":"))
        if indent is None
        else json.dumps(payload, indent=indent)
    )
    path.write_text(text, encoding="utf-8")
    return path


def load_replay_json(path: str | Path) -> tuple[dict[str, Any], list[ReplayFrame3D]]:
    """Load and validate a dm100_replay_3d_v1 JSON file (2D legacy or 3D native)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_replay_payload(raw)
    frames = [ReplayFrame3D.from_dict(item) for item in raw["frames"]]
    meta = dict(raw.get("metadata", {}))
    meta["_schema_version"] = raw["schema_version"]
    meta["_coordinate_frame"] = raw["coordinate_frame"]
    return meta, frames


def validate_replay_payload(payload: dict[str, Any], *, quat_tol: float = 1e-4) -> None:
    """Validate schema headers, monotonic timestamps, finiteness, and unit quaternions."""
    if not isinstance(payload, dict):
        raise ValueError("Replay payload must be a JSON object")
    if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version: {payload.get('schema_version')!r} "
            f"(expected {REPLAY_SCHEMA_VERSION!r})"
        )
    if payload.get("coordinate_frame") != EXPECTED_COORDINATE_FRAME:
        raise ValueError(f"Unsupported coordinate_frame: {payload.get('coordinate_frame')!r}")
    units = payload.get("units", {})
    if units.get("quaternion_order") != "wxyz":
        raise ValueError("Replay quaternion_order must be 'wxyz'")

    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("Replay must contain at least one frame")
    if payload.get("frame_count") != len(frames):
        raise ValueError(
            f"frame_count mismatch: header={payload.get('frame_count')} vs actual={len(frames)}"
        )

    prev_t = -float("inf")
    for idx, frame in enumerate(frames):
        t = float(frame["time_s"])
        if not isfinite(t) or t < prev_t - 1e-9:
            raise ValueError(f"Non-monotonic time_s at frame {idx}: {t} < {prev_t}")
        prev_t = t

        for key, expected_len in (
            ("pos_enu_m", 3),
            ("vel_enu_m_s", 3),
            ("omega_body_rad_s", 3),
            ("quat_wxyz", 4),
            ("gimbal_rad", 2),
            ("rcs_torque_nm", 3),
            ("fin_deflection_rad", 4),
            ("leg_stroke_m", 4),
        ):
            vec = frame.get(key)
            if not isinstance(vec, list) or len(vec) != expected_len:
                raise ValueError(f"Frame {idx} field {key!r} must have length {expected_len}")
            if not all(isfinite(float(v)) for v in vec):
                raise ValueError(f"Frame {idx} field {key!r} contains non-finite value")

        q_norm = float(np.linalg.norm(np.asarray(frame["quat_wxyz"], dtype=np.float64)))
        if abs(q_norm - 1.0) > quat_tol:
            raise ValueError(f"Frame {idx} quat_wxyz norm {q_norm} deviates from 1.0")

        up_q = frame.get("upper_quat_wxyz")
        if up_q is not None:
            uq_norm = float(np.linalg.norm(np.asarray(up_q, dtype=np.float64)))
            if abs(uq_norm - 1.0) > quat_tol:
                raise ValueError(f"Frame {idx} upper_quat_wxyz norm {uq_norm} deviates from 1.0")


def resolve_frame_base_pos_enu(
    frame: ReplayFrame3D,
    *,
    vehicle: DM100VehicleConfig | None = None,
) -> NDArray[np.float64]:
    """Return geometric hull-base (station=0) ENU position [m] for UE 5.4 placement.

    Uses `frame.base_pos_enu_m` directly when recorded by MuJoCoPlant3D.
    For legacy 2D replays where `pos_enu_m` is instantaneous COM only, reconstructs
    `cg_station_m` from propellant, gear extension, and stage-attachment status.
    """
    if frame.base_pos_enu_m is not None:
        return np.asarray(frame.base_pos_enu_m, dtype=np.float64)
    if frame.cg_station_m is not None:
        cg = frame.cg_station_m
    else:
        cfg = vehicle or DM100VehicleConfig()
        tot_mass = (
            cfg.booster_dry_mass_kg
            + max(frame.main_propellant_kg, 0.0)
            + max(frame.rcs_propellant_kg, 0.0)
            + (cfg.upper_stage.total_mass_kg if frame.upper_attached else 0.0)
        )
        cg = mass_properties_3d(
            tot_mass,
            frame.gear_extension,
            vehicle=cfg,
            upper_attached=frame.upper_attached,
        ).cg_station_m
    return com_to_base_pos_enu(frame.pos_enu_m, frame.quat_wxyz, cg)
