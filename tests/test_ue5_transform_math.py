"""Unit tests verifying UE 5.4 FQuat::Rotator() math, unwrapping, and plume scaling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dm100_3d.coords import (
    axis_angle_to_quat_wxyz,
    enu_pos_m_to_ue5_cm,
    enu_quat_wxyz_to_ue5_quat_xyzw,
    enu_rotmat_to_ue5_rotmat,
    quat_to_rotmat,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATH_SCRIPT = REPO_ROOT / "ue5" / "scripts" / "ue5_transform_math.py"


def _load_math():
    spec = importlib.util.spec_from_file_location("ue5_transform_math", MATH_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ue5_rotator_matches_reflection_matrix_across_3d_orientations() -> None:
    mod = _load_math()
    rng = np.random.default_rng(378)

    for _ in range(40):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-np.pi, np.pi))
        q_enu = axis_angle_to_quat_wxyz(axis, angle)

        # Must agree with dm100_3d.coords
        q_ue5_a = np.asarray(mod.enu_quat_wxyz_to_ue5_xyzw(q_enu))
        q_ue5_b = enu_quat_wxyz_to_ue5_quat_xyzw(q_enu)
        assert np.allclose(q_ue5_a, q_ue5_b, atol=1e-12)

        # FQuat::Rotator() reconstructed UE5 matrix must equal S @ R_ENU @ S^-1
        pitch, yaw, roll = mod.ue5_quat_xyzw_to_rotator_deg(tuple(q_ue5_a))
        r_from_rotator = np.asarray(mod._rotator_to_mat_ue5(pitch, yaw, roll))
        r_expected = enu_rotmat_to_ue5_rotmat(quat_to_rotmat(q_enu))
        assert np.allclose(r_from_rotator, r_expected, atol=1e-6)


def test_ue5_pos_and_plume_and_highlight_windows() -> None:
    mod = _load_math()
    p_enu = (12.5, -4.0, 110_000.0)
    assert np.allclose(mod.enu_pos_to_ue5_cm(p_enu), enu_pos_m_to_ue5_cm(p_enu), atol=1e-9)

    # 7 -> 3 -> 1 -> 0 plume activation check
    s7 = mod.compute_plume_scales(1.0, 7)
    s3 = mod.compute_plume_scales(3.0 / 7.0, 3)
    s1 = mod.compute_plume_scales(1.0 / 7.0, 1)
    s0 = mod.compute_plume_scales(0.0, 0)
    assert sum(1 for s in s7 if s[2] > 0.1) == 7
    assert sum(1 for s in s3 if s[2] > 0.1) == 3
    assert sum(1 for s in s1 if s[2] > 0.1) == 1
    assert sum(1 for s in s0 if s[2] > 0.1) == 0

    # Continuous Euler unwrapping eliminates +/-180 deg branch flips
    u = mod.unwrap_euler_deg((10.0, 178.0, -179.0), (11.0, -179.0, 179.0))
    assert u[1] == pytest.approx(181.0, abs=1e-9)
    assert u[2] == pytest.approx(-181.0, abs=1e-9)
