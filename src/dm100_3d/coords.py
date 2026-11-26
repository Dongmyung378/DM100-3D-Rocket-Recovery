"""Coordinate frames, quaternion algebra, and ENU (m) <-> UE 5.4 LH (cm) transforms.

Conventions:
  - World frame: Right-handed ENU (+X East/downrange, +Y North/crossrange, +Z Up).
  - Body frame: Right-handed (+z_b base->nose, +x_b downrange when upright, +y_b crossrange).
  - Quaternions: Unit quaternions in scalar-first (w, x, y, z) order representing
    body-to-world rotation R_wb = R(q) such that v_world = R_wb @ v_body.
  - Unreal Engine 5.4: Left-handed (+X forward/East, +Y right/South = -North, +Z Up), cm.
    Mapping uses the reflection matrix S = diag(1, -1, 1):
      p_ue5_cm = 100.0 * (S @ p_enu_m)
      R_ue5    = S @ R_enu @ S^{-1}
      q_ue5_xyzw = (-q_x, +q_y, -q_z, +q_w)  (matches FQuat::RotateVector identically)
"""

from __future__ import annotations

from math import acos, atan2, cos, isfinite, sin, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

REFLECTION_ENU_TO_UE5 = np.diag([1.0, -1.0, 1.0])
M_TO_CM = 100.0


def normalize_quat_wxyz(q: ArrayLike, *, canonical: bool = False) -> NDArray[np.float64]:
    """Normalize a (w, x, y, z) quaternion to unit length."""
    arr = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(arr))
    if not isfinite(norm) or norm < 1e-15:
        raise ValueError("Quaternion norm must be finite and strictly positive")
    unit = arr / norm
    if canonical and unit[0] < 0.0:
        unit = -unit
    return unit


def quat_conjugate_wxyz(q: ArrayLike) -> NDArray[np.float64]:
    """Conjugate (inverse for unit quaternion) of (w, x, y, z)."""
    arr = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([arr[0], -arr[1], -arr[2], -arr[3]], dtype=np.float64)


def quat_multiply_wxyz(q1: ArrayLike, q2: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product q1 (*) q2 in (w, x, y, z) order."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def axis_angle_to_quat_wxyz(axis: ArrayLike, angle_rad: float) -> NDArray[np.float64]:
    """Construct a unit quaternion (w, x, y, z) from rotation axis and angle."""
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u = ax / norm
    half = 0.5 * float(angle_rad)
    s = sin(half)
    return np.array([cos(half), u[0] * s, u[1] * s, u[2] * s], dtype=np.float64)


def pitch_to_quat_wxyz(theta_rad: float) -> tuple[float, float, float, float]:
    """2D planar pitch (+theta around ENU +Y tilts +Z toward +X) -> unit quaternion (w,x,y,z)."""
    half = 0.5 * float(theta_rad)
    return (cos(half), 0.0, sin(half), 0.0)


def quat_to_rotmat(q: ArrayLike) -> NDArray[np.float64]:
    """Convert unit quaternion (w, x, y, z) to 3x3 body-to-world rotation matrix R_wb."""
    w, x, y, z = normalize_quat_wxyz(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(rot: ArrayLike) -> NDArray[np.float64]:
    """Convert a proper 3x3 rotation matrix to canonical unit quaternion (w, x, y, z)."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = 2.0 * sqrt(tr + 1.0)
        q = np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2.0 * sqrt(max(1e-18, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
        q = np.array(
            [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif r[1, 1] >= r[2, 2]:
        s = 2.0 * sqrt(max(1e-18, 1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
        q = np.array(
            [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = 2.0 * sqrt(max(1e-18, 1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
        q = np.array(
            [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return normalize_quat_wxyz(q, canonical=True)


def rotate_body_to_world(q_wxyz: ArrayLike, vec_body: ArrayLike) -> NDArray[np.float64]:
    """Rotate a 3D vector from body frame into ENU world frame."""
    return quat_to_rotmat(q_wxyz) @ np.asarray(vec_body, dtype=np.float64).reshape(3)


def rotate_world_to_body(q_wxyz: ArrayLike, vec_world: ArrayLike) -> NDArray[np.float64]:
    """Rotate a 3D vector from ENU world frame into body frame."""
    return quat_to_rotmat(q_wxyz).T @ np.asarray(vec_world, dtype=np.float64).reshape(3)


def tilt_angle_rad(q_wxyz: ArrayLike) -> float:
    """Angle [rad] between body +z_b axis and world vertical +Z."""
    r = quat_to_rotmat(q_wxyz)
    cos_tilt = max(-1.0, min(1.0, float(r[2, 2])))
    return acos(cos_tilt)


def roll_angle_rad(q_wxyz: ArrayLike) -> float:
    """Singularity-free body roll about +z_b relative to the vertical-plane zero-roll frame.

    Measures the rotation angle of body +y_b away from the zero-roll reference axis
    (the projection of world +Y orthogonal to body +z_b), so pitching through +/-90 deg
    during horizontal boostback reports 0 deg roll instead of an atan2 180 deg flip.
    """
    r = quat_to_rotmat(q_wxyz)
    zb = r[:, 2]
    yb = r[:, 1]
    y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(zb @ y_ref)) > 0.98:
        y_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y0 = y_ref - float(y_ref @ zb) * zb
    y0 /= float(np.linalg.norm(y0))
    x0 = np.cross(y0, zb)
    return atan2(float(-yb @ x0), float(yb @ y0))


def com_to_base_pos_enu(
    pos_com_enu_m: ArrayLike,
    quat_wxyz: ArrayLike,
    cg_station_m: float,
) -> NDArray[np.float64]:
    """Convert instantaneous COM position in ENU to geometric hull-base (station=0) ENU position."""
    com = np.asarray(pos_com_enu_m, dtype=np.float64).reshape(3)
    axis_world = quat_to_rotmat(quat_wxyz)[:, 2]
    return com - float(cg_station_m) * axis_world


def base_to_com_pos_enu(
    pos_base_enu_m: ArrayLike,
    quat_wxyz: ArrayLike,
    cg_station_m: float,
) -> NDArray[np.float64]:
    """Convert geometric hull-base (station=0) ENU position to instantaneous COM ENU position."""
    base = np.asarray(pos_base_enu_m, dtype=np.float64).reshape(3)
    axis_world = quat_to_rotmat(quat_wxyz)[:, 2]
    return base + float(cg_station_m) * axis_world


def enu_pos_m_to_ue5_cm(pos_enu_m: ArrayLike) -> NDArray[np.float64]:
    """Convert right-handed ENU position [m] to UE 5.4 left-handed position [cm]."""
    p = np.asarray(pos_enu_m, dtype=np.float64).reshape(3)
    return np.array([p[0] * M_TO_CM, -p[1] * M_TO_CM, p[2] * M_TO_CM], dtype=np.float64)


def ue5_cm_to_enu_pos_m(pos_ue5_cm: ArrayLike) -> NDArray[np.float64]:
    """Inverse of enu_pos_m_to_ue5_cm: UE 5.4 [cm] -> right-handed ENU [m]."""
    p = np.asarray(pos_ue5_cm, dtype=np.float64).reshape(3)
    return np.array([p[0] / M_TO_CM, -p[1] / M_TO_CM, p[2] / M_TO_CM], dtype=np.float64)


def enu_rotmat_to_ue5_rotmat(rot_enu: ArrayLike) -> NDArray[np.float64]:
    """Transform ENU rotation matrix into UE 5.4 frame via R_UE5 = S @ R_ENU @ S^{-1}."""
    r = np.asarray(rot_enu, dtype=np.float64).reshape(3, 3)
    return REFLECTION_ENU_TO_UE5 @ r @ REFLECTION_ENU_TO_UE5


def enu_quat_wxyz_to_ue5_quat_xyzw(q_enu_wxyz: ArrayLike) -> NDArray[np.float64]:
    """Convert ENU unit quaternion (w,x,y,z) to UE 5.4 FQuat (x,y,z,w).

    Algebraically identical to extracting a quaternion from R_UE5 = S @ R_ENU @ S^{-1}:
      (w, x, y, z)_ENU -> (-x, +y, -z, +w)_UE5_xyzw.
    """
    w, x, y, z = normalize_quat_wxyz(q_enu_wxyz, canonical=True)
    return np.array([-x, y, -z, w], dtype=np.float64)


def ue5_quat_xyzw_to_enu_quat_wxyz(q_ue5_xyzw: ArrayLike) -> NDArray[np.float64]:
    """Inverse of enu_quat_wxyz_to_ue5_quat_xyzw: UE 5.4 (x,y,z,w) -> ENU (w,x,y,z)."""
    qx, qy, qz, qw = np.asarray(q_ue5_xyzw, dtype=np.float64).reshape(4)
    return normalize_quat_wxyz(np.array([qw, -qx, qy, -qz], dtype=np.float64), canonical=True)


def ue5_quat_xyzw_rotate_vector(q_ue5_xyzw: ArrayLike, vec_ue5: ArrayLike) -> NDArray[np.float64]:
    """Exact mirror of UE 5.4 FQuat::RotateVector(V) for unit verification."""
    qx, qy, qz, qw = np.asarray(q_ue5_xyzw, dtype=np.float64).reshape(4)
    q_vec = np.array([qx, qy, qz], dtype=np.float64)
    v = np.asarray(vec_ue5, dtype=np.float64).reshape(3)
    t = 2.0 * np.cross(q_vec, v)
    return v + qw * t + np.cross(q_vec, t)
