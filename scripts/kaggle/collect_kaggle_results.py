"""Pull and merge sharded Kaggle CPU validation outputs + frozen 50 Hz 3D replays.

Usage:
  python scripts/kaggle/collect_kaggle_results.py --username <user> --suite edge-3d --shards 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _resolve_kaggle_cli() -> str:
    py_dir = Path(sys.executable).resolve().parent
    for name in ("kaggle.exe", "kaggle"):
        cand = py_dir / name
        if cand.exists():
            return str(cand)
    found = shutil.which("kaggle")
    if found:
        return found
    raise FileNotFoundError("Kaggle CLI executable not found in active Python environment or PATH.")


def _resolve_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        user = str(data.get("username", "")).strip()
        if user:
            return user
    raise SystemExit("Kaggle username not provided and ~/.kaggle/kaggle.json not found.")


def merge_shard_reports(shard_paths: list[Path], *, suite: str) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    for p in shard_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            by_name[str(r["scenario"])] = r

    ordered = [by_name[k] for k in sorted(by_name)]
    total = len(ordered)
    passed = sum(1 for r in ordered if r.get("requirements_passed"))
    failed = [
        {"scenario": r["scenario"], "outcome": r["outcome"], "failed": r["failed_criteria"]}
        for r in ordered
        if not r.get("requirements_passed")
    ]

    def _max_f(key: str) -> float | None:
        vals = [float(r[key]) for r in ordered if r.get(key) is not None]
        return max(vals) if vals else None

    return {
        "schema": "dm100_3d_validation_merged_v1",
        "suite": suite,
        "shards_merged": len(shard_paths),
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": (passed / total) if total > 0 else 0.0,
        "failed_cases": failed,
        "max_position_error_m": _max_f("position_error_m"),
        "max_vertical_foot_speed_m_s": _max_f("vertical_foot_speed_m_s"),
        "max_horizontal_foot_speed_m_s": _max_f("horizontal_foot_speed_m_s"),
        "max_tilt_deg": _max_f("tilt_deg"),
        "max_dynamic_pressure_pa": _max_f("peak_dynamic_pressure_pa"),
        "max_heat_flux_w_m2": _max_f("peak_heat_flux_w_m2"),
        "max_angular_rate_deg_s": _max_f("peak_angular_rate_deg_s"),
        "max_angular_acceleration_deg_s2": _max_f("peak_angular_acceleration_deg_s2"),
        "max_roll_deg": _max_f("peak_roll_deg"),
        "max_fin_hinge_nm": _max_f("peak_fin_hinge_nm"),
        "results": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and merge Kaggle 3D validation outputs.")
    parser.add_argument(
        "--username",
        default=None,
        help="Kaggle username (defaults to username in ~/.kaggle/kaggle.json)",
    )
    parser.add_argument("--suite", default="edge-3d", choices=["edge-3d", "holdout-3d", "full-3d"])
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--status-only", action="store_true", help="Only query shard statuses")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/kaggle_results"))
    parser.add_argument("--replays-dir", type=Path, default=Path("artifacts/replays"))
    args = parser.parse_args()

    username = _resolve_username(args.username)
    kaggle_exe = _resolve_kaggle_cli()

    if args.status_only:
        for idx in range(args.shards):
            slug = f"{username}/dm100-3d-{args.suite}-shard-{idx}-{args.shards}"
            subprocess.run([kaggle_exe, "kernels", "status", slug], check=False)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.replays_dir.mkdir(parents=True, exist_ok=True)
    shard_jsons: list[Path] = []

    for idx in range(args.shards):
        slug = f"{username}/dm100-3d-{args.suite}-shard-{idx}-{args.shards}"
        s_out = args.out_dir / f"shard_{idx}"
        s_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [kaggle_exe, "kernels", "output", slug, "-p", str(s_out), "--force"],
            check=True,
        )
        shard_jsons.extend(sorted(s_out.rglob("results_*.json")))
        for rep in s_out.rglob("*_50hz.json"):
            shutil.copy2(rep, args.replays_dir / rep.name)

    merged = merge_shard_reports(shard_jsons, suite=args.suite)
    merged_path = args.out_dir / f"validation_3d_{args.suite}_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"Merged {merged['passed_cases']}/{merged['total_cases']} "
        f"({100.0 * merged['pass_rate']:.1f}%) -> {merged_path}"
    )


if __name__ == "__main__":
    main()
