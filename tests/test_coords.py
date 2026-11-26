"""Unit tests for ENU <-> Body <-> UE 5.4 coordinate and quaternion contracts."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from dm100_3d.coords import (
    REFLECTION_ENU_TO_UE5,
    axis_angle_to_quat_wxyz,
    base_to_com_pos_enu,
    com_to_base_pos_enu,
    enu_pos_m_to_ue5_cm,
    enu_quat_wxyz_to_ue5_quat_xyzw,
    enu_rotmat_to_ue5_rotmat,
    normalize_quat_wxyz,
    pitch_to_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_to_rotmat,
    roll_angle_rad,
    rotate_body_to_world,
    rotate_world_to_body,
    rotmat_to_quat_wxyz,
    tilt_angle_rad,
    ue5_cm_to_enu_pos_m,
    ue5_quat_xyzw_rotate_vector,
    ue5_quat_xyzw_to_enu_quat_wxyz,
)


@pytest.mark.parametrize(
    ("axis", "angle_deg"),
    [
        ((1.0, 0.0, 0.0), 35.0),
        ((1.0, 0.0, 0.0), -120.0),
        ((0.0, 1.0, 0.0), 42.0),
        ((0.0, 1.0, 0.0), -145.0),
        ((0.0, 0.0, 1.0), 60.0),
        ((0.0, 0.0, 1.0), -175.0),
        ((1.0, -2.0, 0.5), 113.0),
    ],
)
def test_six_axis_and_general_enu_to_ue5_consistency(
    axis: tuple[float, float, float], angle_deg: float
) -> None:
    """R_UE5 = S R_ENU S^-1 must agree identically with FQuat::RotateVector on q_UE5."""
    angle_rad = angle_deg * pi / 180.0
    q_enu = axis_angle_to_quat_wxyz(axis, angle_rad)
    r_enu = quat_to_rotmat(q_enu)

    # Proper orthogonal rotation in ENU and UE5
    assert np.allclose(r_enu.T @ r_enu, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r_enu), 1.0, atol=1e-12)

    r_ue5 = enu_rotmat_to_ue5_rotmat(r_enu)
    assert np.allclose(r_ue5.T @ r_ue5, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r_ue5), 1.0, atol=1e-12)

    q_ue5_xyzw = enu_quat_wxyz_to_ue5_quat_xyzw(q_enu)
    # Extract quaternion directly from R_UE5 and compare against closed-form (-x, +y, -z, +w)
    q_ue5_from_mat_wxyz = rotmat_to_quat_wxyz(r_ue5)
    q_ue5_from_mat_xyzw = np.array(
        [
            q_ue5_from_mat_wxyz[1],
            q_ue5_from_mat_wxyz[2],
            q_ue5_from_mat_wxyz[3],
            q_ue5_from_mat_wxyz[0],
        ]
    )
    assert np.allclose(q_ue5_xyzw, q_ue5_from_mat_xyzw, atol=1e-12) or np.allclose(
        q_ue5_xyzw, -q_ue5_from_mat_xyzw, atol=1e-12
    )

    # Verify vector rotation equivalence across all 3 basis vectors + arbitrary vector
    for v_body_enu in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.2, -0.7, 2.4]),
    ):
        v_world_enu = rotate_body_to_world(q_enu, v_body_enu)
        v_world_ue5_expected = REFLECTION_ENU_TO_UE5 @ v_world_enu
        v_body_ue5 = REFLECTION_ENU_TO_UE5 @ v_body_enu
        v_world_ue5_mat = r_ue5 @ v_body_ue5
        v_world_ue5_quat = ue5_quat_xyzw_rotate_vector(q_ue5_xyzw, v_body_ue5)
        assert np.allclose(v_world_ue5_mat, v_world_ue5_expected, atol=1e-12)
        assert np.allclose(v_world_ue5_quat, v_world_ue5_expected, atol=1e-12)

    # Exact round-trip UE5 -> ENU
    q_back = ue5_quat_xyzw_to_enu_quat_wxyz(q_ue5_xyzw)
    q_canon = normalize_quat_wxyz(q_enu, canonical=True)
    assert np.allclose(q_back, q_canon, atol=1e-12)


def test_2d_pitch_convention_matches_dm100_2d() -> None:
    """+theta around ENU +Y tilts nose (+z_b) toward +X (East/downrange)."""
    theta = 30.0 * pi / 180.0
    q = pitch_to_quat_wxyz(theta)
    nose_world = rotate_body_to_world(q, (0.0, 0.0, 1.0))
    assert np.allclose(nose_world, [0.5, 0.0, sqrt(3.0) / 2.0], atol=1e-12)
    assert np.isclose(tilt_angle_rad(q), theta, atol=1e-12)
    assert np.isclose(roll_angle_rad(q), 0.0, atol=1e-12)

    # In UE5, +Z still tilts toward +X identically
    q_ue5 = enu_quat_wxyz_to_ue5_quat_xyzw(q)
    nose_ue5 = ue5_quat_xyzw_rotate_vector(q_ue5, (0.0, 0.0, 1.0))
    assert np.allclose(nose_ue5, [0.5, 0.0, sqrt(3.0) / 2.0], atol=1e-12)


def test_position_and_com_base_roundtrip() -> None:
    p_enu = np.array([1234.5, -678.25, 42000.0])
    p_ue5 = enu_pos_m_to_ue5_cm(p_enu)
    assert np.allclose(p_ue5, [123450.0, 67825.0, 4200000.0], atol=1e-9)
    assert np.allclose(ue5_cm_to_enu_pos_m(p_ue5), p_enu, atol=1e-12)

    q = axis_angle_to_quat_wxyz((0.2, 0.9, -0.3), 0.45)
    cg_m = 13.2926
    base = com_to_base_pos_enu(p_enu, q, cg_m)
    com_back = base_to_com_pos_enu(base, q, cg_m)
    assert np.allclose(com_back, p_enu, atol=1e-12)
    assert np.isclose(np.linalg.norm(p_enu - base), cg_m, atol=1e-12)


def test_quaternion_algebra_identities() -> None:
    q1 = axis_angle_to_quat_wxyz((1.0, 2.0, 3.0), 0.7)
    q2 = axis_angle_to_quat_wxyz((-0.5, 1.0, 0.2), -1.1)
    q12 = quat_multiply_wxyz(q1, q2)
    assert np.allclose(quat_to_rotmat(q12), quat_to_rotmat(q1) @ quat_to_rotmat(q2), atol=1e-12)

    q_id = quat_multiply_wxyz(q1, quat_conjugate_wxyz(q1))
    assert np.allclose(q_id, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    vec = np.array([3.1, -4.2, 5.3])
    assert np.allclose(rotate_world_to_body(q1, rotate_body_to_world(q1, vec)), vec, atol=1e-12)
