"""Lightweight unit tests for Phase 4 3D validation suites, sharding, and Kaggle merge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dm100_3d.validation_suites import get_validation_suite, select_shard

KAGGLE_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "kaggle"


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validation_suites_and_exhaustive_disjoint_sharding() -> None:
    edge = get_validation_suite("edge-3d")
    holdout = get_validation_suite("holdout-3d")
    full = get_validation_suite("full-3d")

    assert len(edge) == 32
    assert len(holdout) == 64
    assert len(full) == 96
    assert len({c.name for c in full}) == 96

    # 4-way sharding must be disjoint and exhaustive
    shards = [select_shard(full, f"{k}/4") for k in range(4)]
    assert all(len(s) == 24 for s in shards)
    seen = [c.name for s in shards for c in s]
    assert len(seen) == 96
    assert set(seen) == {c.name for c in full}


def test_kaggle_shard_merge_and_script_generation(tmp_path: Path) -> None:
    collect_mod = _load_script_module(
        "collect_kaggle_results", KAGGLE_SCRIPTS_DIR / "collect_kaggle_results.py"
    )
    push_mod = _load_script_module(
        "push_kaggle_shards", KAGGLE_SCRIPTS_DIR / "push_kaggle_shards.py"
    )
    merge_shard_reports = collect_mod.merge_shard_reports
    build_shard_script = push_mod.build_shard_script

    s0 = tmp_path / "results_s0.json"
    s1 = tmp_path / "results_s1.json"
    s0.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scenario": "edge3d-01-calm-nominal",
                        "outcome": "landed",
                        "requirements_passed": True,
                        "failed_criteria": [],
                        "position_error_m": 0.01,
                        "peak_roll_deg": 0.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    s1.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scenario": "edge3d-02-max-east-wind",
                        "outcome": "landed",
                        "requirements_passed": True,
                        "failed_criteria": [],
                        "position_error_m": 0.15,
                        "peak_roll_deg": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    merged = merge_shard_reports([s0, s1], suite="edge-3d")
    assert merged["total_cases"] == 2
    assert merged["passed_cases"] == 2
    assert merged["pass_rate"] == 1.0
    assert merged["max_position_error_m"] == 0.15

    script_txt = build_shard_script(
        suite="edge-3d",
        shard_idx=2,
        num_shards=4,
        repo_url="https://github.com/Dongmyung378/DM100-3D-Rocket-Recovery.git",
        repo_ref="main",
    )
    assert "os.environ['DM100_SHARD'] = '2/4'" in script_txt
    assert "os.environ['DM100_WORKERS'] = '4'" in script_txt
