"""Unit tests for dm100_replay_3d_v1 export/load/validation and 2D golden replay ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dm100_3d.coords import enu_pos_m_to_ue5_cm, enu_quat_wxyz_to_ue5_quat_xyzw
from dm100_3d.interfaces import ReplayFrame3D
from dm100_3d.replay import (
    REPLAY_SCHEMA_VERSION,
    export_replay_json,
    load_replay_json,
    resolve_frame_base_pos_enu,
)

DM100_2D_REPLAY_DIR = Path(r"d:\Projects\DM100-2D-Rocket-Recovery\docs\results\replays")


def test_replay_roundtrip_with_native_3d_fields(tmp_path: Path) -> None:
    frame0 = ReplayFrame3D(
        time_s=0.0,
        phase="ascent",
        pos_enu_m=(0.0, 0.0, 13.4926),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        vel_enu_m_s=(0.0, 0.0, 0.0),
        omega_body_rad_s=(0.0, 0.0, 0.0),
        throttle=1.0,
        active_engines=7,
        gimbal_rad=(0.0, 0.0),
        rcs_torque_nm=(0.0, 0.0, 0.0),
        fin_extension=0.0,
        fin_deflection_rad=(0.0, 0.0, 0.0, 0.0),
        gear_extension=0.0,
        leg_stroke_m=(0.0, 0.0, 0.0, 0.0),
        dynamic_pressure_pa=0.0,
        mach=0.0,
        alpha_deg=0.0,
        main_propellant_kg=69_500.0,
        rcs_propellant_kg=300.0,
        upper_attached=True,
        base_pos_enu_m=(0.0, 0.0, 0.2),
        cg_station_m=13.2926,
        heat_flux_w_m2=0.0,
    )
    out = export_replay_json([frame0], tmp_path / "sample.json", metadata={"scenario": "unit"})
    meta, loaded = load_replay_json(out)
    assert meta["_schema_version"] == REPLAY_SCHEMA_VERSION
    assert meta["scenario"] == "unit"
    assert len(loaded) == 1
    assert loaded[0] == frame0
    assert np.allclose(resolve_frame_base_pos_enu(loaded[0]), [0.0, 0.0, 0.2], atol=1e-12)


@pytest.mark.parametrize(
    ("filename", "pad_base_tol_m"),
    [
        ("calm_replay_50hz.json", 5e-3),
        # Legacy 2D weather replay includes hidden initial_mass_scale perturbation (~1 cm CG shift)
        ("example_weather_replay_50hz.json", 0.02),
    ],
)
def test_load_dm100_2d_golden_replays_and_resolve_hull_base(
    filename: str, pad_base_tol_m: float
) -> None:
    replay_path = DM100_2D_REPLAY_DIR / filename
    if not replay_path.exists():
        pytest.skip(f"2D golden replay not found at {replay_path}")

    meta, frames = load_replay_json(replay_path)
    assert meta["_schema_version"] == REPLAY_SCHEMA_VERSION
    assert meta["requirements_passed"] is True
    assert len(frames) > 10_000

    # Frame 0 is on the launchpad: hull base sits at z = -nozzle_station_m = +0.2 m
    base0 = resolve_frame_base_pos_enu(frames[0])
    assert np.isclose(base0[2], 0.2, atol=pad_base_tol_m)

    # Final settled frame: 4-leg upright rest should put hull base ~0.6-0.8 m above ground
    base_end = resolve_frame_base_pos_enu(frames[-1])
    assert 0.4 < base_end[2] < 1.0

    # Verify UE5 transform runs cleanly on first and last frames
    p_ue5_end = enu_pos_m_to_ue5_cm(base_end)
    q_ue5_end = enu_quat_wxyz_to_ue5_quat_xyzw(frames[-1].quat_wxyz)
    assert np.all(np.isfinite(p_ue5_end))
    assert np.isclose(np.linalg.norm(q_ue5_end), 1.0, atol=1e-12)
