"""Pure-Python ENU (m, wxyz) -> UE 5.4 LH (cm, FQuat / continuous FRotator) math helpers.

This module has ZERO dependency on `unreal` so it can be unit-tested in standard pytest
AND imported directly inside UE 5.4's embedded Python interpreter by `import_dm100_replay.py`.

Coordinate mapping (identical to `dm100_3d.coords`):
  - World reflection matrix: S = diag(1, -1, 1)  (ENU +Y North -> UE5 -Y)
  - Position:  p_ue5_cm = (x_m * 100, -y_m * 100, z_m * 100)
  - Rotation:  R_ue5 = S @ R_enu @ S^-1
  - Quat:      q_ue5_xyzw = (-qx, +qy, -qz, +qw)   (matches UE5 FQuat::RotateVector)
  - FRotator:  Exact port of UE5 `FQuat::Rotator()` (Pitch, Yaw, Roll in degrees),
               plus continuous branch-unwrapping along timelines for smooth Sequencer keys.
"""

from __future__ import annotations

from math import asin, atan2, degrees, radians, sqrt
from typing import Any

M_TO_CM: float = 100.0


def enu_pos_to_ue5_cm(
    pos_enu_m: tuple[float, float, float] | list[float],
) -> tuple[float, float, float]:
    """Convert right-handed ENU [m] -> left-handed UE 5.4 [cm]."""
    return (
        float(pos_enu_m[0]) * M_TO_CM,
        -float(pos_enu_m[1]) * M_TO_CM,
        float(pos_enu_m[2]) * M_TO_CM,
    )


def enu_quat_wxyz_to_ue5_xyzw(
    q_wxyz: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    """Convert canonical ENU unit quaternion (w, x, y, z) -> UE 5.4 FQuat (x, y, z, w)."""
    w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]))
    nrm = sqrt(w * w + x * x + y * y + z * z)
    if nrm < 1e-15:
        return (0.0, 0.0, 0.0, 1.0)
    w, x, y, z = w / nrm, x / nrm, y / nrm, z / nrm
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (-x, y, -z, w)


def ue5_quat_xyzw_to_rotator_deg(
    q_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Exact port of Unreal Engine 5.4 `FQuat::Rotator()` -> `(pitch_deg, yaw_deg, roll_deg)`.

    Matches UE5 `Engine/Source/Runtime/Core/Private/Math/UnrealMath.cpp`:
      - Pitch = rotation around +Y (nose up/down)
      - Yaw   = rotation around +Z
      - Roll  = rotation around +X
    """
    x, y, z, w = q_xyzw
    singularity_test = z * x - w * y
    yaw_y = 2.0 * (w * z + x * y)
    yaw_x = 1.0 - 2.0 * (y * y + z * z)

    singularity_threshold = 0.4999995
    yaw = degrees(atan2(yaw_y, yaw_x))
    if singularity_test < -singularity_threshold:
        pitch = -90.0
        roll = -yaw - 2.0 * degrees(atan2(x, w))
    elif singularity_test > singularity_threshold:
        pitch = 90.0
        roll = yaw - 2.0 * degrees(atan2(x, w))
    else:
        pitch = degrees(asin(max(-1.0, min(1.0, 2.0 * singularity_test))))
        roll = degrees(atan2(-2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))

    # Normalize into [-180, 180]
    roll = ((roll + 180.0) % 360.0) - 180.0
    pitch = ((pitch + 180.0) % 360.0) - 180.0
    yaw = ((yaw + 180.0) % 360.0) - 180.0
    return (pitch, yaw, roll)


def unwrap_euler_deg(
    prev_deg: tuple[float, float, float] | None,
    curr_deg: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Unwrap `(pitch, yaw, roll)` degrees relative to previous frame to avoid 360 deg jumps."""
    if prev_deg is None:
        return curr_deg
    out = []
    for p, c in zip(prev_deg, curr_deg, strict=True):
        diff = ((c - p + 180.0) % 360.0) - 180.0
        out.append(p + diff)
    return (out[0], out[1], out[2])


def resolve_booster_base_pos_enu(frame: dict[str, Any]) -> tuple[float, float, float]:
    """Extract drift-free hull-base ENU position (station = 0 m) from a replay frame."""
    base = frame.get("base_pos_enu_m")
    if base is not None and len(base) == 3:
        return (float(base[0]), float(base[1]), float(base[2]))
    # Legacy 2D fallback: reconstruct from COM + cg_station_m along body +z_b
    pos = frame["pos_enu_m"]
    q = frame["quat_wxyz"]
    cg = float(frame.get("cg_station_m") or 8.5)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    zb_x = 2.0 * (x * z + w * y)
    zb_y = 2.0 * (y * z - w * x)
    zb_z = 1.0 - 2.0 * (x * x + y * y)
    return (float(pos[0]) - cg * zb_x, float(pos[1]) - cg * zb_y, float(pos[2]) - cg * zb_z)


def compute_plume_scales(throttle: float, active_engines: int) -> list[tuple[float, float, float]]:
    """Compute visual scale `(sx, sy, sz)` for each of the 7 Methalox nozzles.

    Engine layout matches `generate_rocket_glb.py`:
      - Index 0      : Center engine (active for 7, 3, or 1 engine modes)
      - Index 1, 4   : Opposing inner pair active during 3-engine boostback / entry / hoverslam
      - Index 2,3,5,6: Remaining 4 outer engines active during 7-engine ascent
    """
    thr = max(0.0, float(throttle))
    eng = int(active_engines)
    if thr <= 1e-3 or eng <= 0:
        return [(0.001, 0.001, 0.001)] * 7

    if eng >= 7:
        active_set = {0, 1, 2, 3, 4, 5, 6}
        per_eng = min(1.0, thr)
    elif eng >= 3:
        active_set = {0, 1, 4}
        per_eng = min(1.0, thr * (7.0 / 3.0))
    else:
        active_set = {0}
        per_eng = min(1.0, thr * 7.0)

    scales: list[tuple[float, float, float]] = []
    for idx in range(7):
        if idx in active_set:
            rad_s = 0.65 + 0.55 * per_eng
            len_s = 0.45 + 1.65 * per_eng
            scales.append((rad_s, rad_s, len_s))
        else:
            scales.append((0.001, 0.001, 0.001))
    return scales


def compute_highlight_windows(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Auto-select dramatic mission highlight segments for 1080p MRQ reel rendering."""
    if not frames:
        return []
    t_end = float(frames[-1]["time_s"])
    t_sep = None
    t_entry = None
    t_glide = None
    t_land = None
    t_td = None
    for f in frames:
        ph = str(f.get("phase", ""))
        t = float(f["time_s"])
        if t_sep is None and not bool(f.get("upper_attached", True)):
            t_sep = t
        if t_entry is None and ph == "entry_burn":
            t_entry = t
        if t_glide is None and ph == "aero_glide":
            t_glide = t
        if t_land is None and ph == "landing_burn":
            t_land = t
        if t_td is None and ph == "touchdown_settle":
            t_td = t

    windows: list[dict[str, Any]] = [
        {"name": "01_liftoff", "camera": "Cam_LaunchPad", "start_s": 0.0, "end_s": min(14.0, t_end)}
    ]
    if t_sep is not None:
        windows.append(
            {
                "name": "02_meco_sep_flip",
                "camera": "Cam_Chase_3D",
                "start_s": max(0.0, t_sep - 3.0),
                "end_s": min(t_end, t_sep + 18.0),
            }
        )
    if t_entry is not None:
        windows.append(
            {
                "name": "03_entry_burn_fin_cam",
                "camera": "Cam_Onboard_Fin",
                "start_s": max(0.0, t_entry - 2.0),
                "end_s": min(t_end, t_entry + 14.0),
            }
        )
    if t_land is not None:
        td_ref = t_td if t_td is not None else t_end
        t_cut = max(t_land + 2.0, td_ref - 12.0)
        windows.append(
            {
                "name": "04_hoverslam_approach",
                "camera": "Cam_Chase_3D",
                "start_s": max(0.0, t_land - 2.0),
                "end_s": t_cut,
            }
        )
        windows.append(
            {
                "name": "05_touchdown_settle",
                "camera": "Cam_Pad_Hero",
                "start_s": t_cut,
                "end_s": t_end,
            }
        )
    return windows


def _rotator_to_mat_ue5(pitch_deg: float, yaw_deg: float, roll_deg: float) -> list[list[float]]:
    """Reference UE5 FRotationMatrix from (Pitch, Yaw, Roll) degrees for verification tests."""
    from math import cos, sin

    sp, cp = sin(radians(pitch_deg)), cos(radians(pitch_deg))
    sy, cy = sin(radians(yaw_deg)), cos(radians(yaw_deg))
    sr, cr = sin(radians(roll_deg)), cos(radians(roll_deg))
    # UE5 FRotationMatrix::Make (columns are transformed basis vectors X, Y, Z)
    return [
        [cp * cy, sr * sp * cy - cr * sy, -(cr * sp * cy + sr * sy)],
        [cp * sy, sr * sp * sy + cr * cy, cy * sr - cr * sp * sy],
        [sp, -sr * cp, cr * cp],
    ]
