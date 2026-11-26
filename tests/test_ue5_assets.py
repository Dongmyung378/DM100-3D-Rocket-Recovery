"""Unit tests verifying exact DM100 3D procedural GLB mesh geometry and pivots for UE 5.4."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import trimesh

from dm100_3d.config import DM100VehicleConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "ue5" / "scripts" / "generate_rocket_glb.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_rocket_glb", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_procedural_glb_assets_match_dm100_dimensions(tmp_path: Path) -> None:
    mod = _load_generator()
    cfg = DM100VehicleConfig()
    manifest = mod.generate_all_assets(tmp_path, vehicle=cfg)

    assert set(manifest.keys()) == {
        "dm100_booster_hull",
        "dm100_upper_stage",
        "dm100_grid_fin",
        "dm100_landing_leg",
        "dm100_landing_pad",
    }

    # Booster hull: nozzle rim at -0.20 m, interstage top at +24.0 m, diameter ~3.1 m
    b_min = manifest["dm100_booster_hull"]["bounds_zup_m"]["min"]
    b_max = manifest["dm100_booster_hull"]["bounds_zup_m"]["max"]
    assert b_min[2] == pytest.approx(cfg.geometry.nozzle_station_m, abs=1e-3)
    assert b_max[2] == pytest.approx(cfg.geometry.body_length_m, abs=1e-3)
    assert b_max[0] - b_min[0] == pytest.approx(2.0 * cfg.geometry.body_radius_m, abs=0.35)

    # Upper stage: fairing apex at +9.0 m, vacuum bell tucked below z = 0
    u_min = manifest["dm100_upper_stage"]["bounds_zup_m"]["min"]
    u_max = manifest["dm100_upper_stage"]["bounds_zup_m"]["max"]
    assert u_max[2] == pytest.approx(cfg.geometry.upper_length_m, abs=1e-3)
    assert u_min[2] < 0.0

    # Grid fin: 1.20 m span (+X), 1.25 m chord (Z)
    f_ext = manifest["dm100_grid_fin"]["bounds_zup_m"]["extents"]
    assert f_ext[0] == pytest.approx(1.25, abs=0.10)
    assert f_ext[2] == pytest.approx(1.28, abs=0.05)

    # Landing leg: reaches radial delta 2.45 m and vertical delta -2.20 m
    l_min = manifest["dm100_landing_leg"]["bounds_zup_m"]["min"]
    l_max = manifest["dm100_landing_leg"]["bounds_zup_m"]["max"]
    assert l_min[2] == pytest.approx(
        cfg.landing_gear.foot_station_m - cfg.landing_gear.anchor_station_m, abs=1e-3
    )
    assert l_max[0] > cfg.landing_gear.half_span_m - cfg.geometry.body_radius_m

    # Verify every GLB reloads cleanly via trimesh as a non-empty glTF scene
    for name, info in manifest.items():
        glb_path = Path(info["path"])
        assert glb_path.exists() and glb_path.stat().st_size > 1_000
        reloaded = trimesh.load(glb_path, force="scene")
        assert len(reloaded.geometry) >= 1, f"{name}.glb had no geometry on reload"
