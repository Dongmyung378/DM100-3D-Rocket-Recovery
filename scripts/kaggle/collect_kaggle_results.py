"""Pull and merge sharded Kaggle CPU validation outputs + frozen 50 Hz 3D replays.

Uses strict suite/tag-namespaced directories and exact shard filename matching so stale
outputs from earlier suites or shard counts can NEVER contaminate merged reports.

Usage:
  python scripts/kaggle/collect_kaggle_results.py --suite edge-3d --shards 4 --tag m1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SUPPORTED_SUITES = (
    "edge-3d",
    "holdout-3d",
    "full-3d",
    "edge-v2",
    "mc-1000",
    "fault-mc-160",
    "envelope-boundary",
)


def _sanitize_slug(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if len(cleaned) > 50:
        cleaned = cleaned[:50].rstrip("-")
    return cleaned


def _make_shard_slug(*, suite: str, shard_idx: int, num_shards: int, tag: str | None = None) -> str:
    base = f"dm100-3d-{suite}-{tag}" if tag else f"dm100-3d-{suite}"
    return _sanitize_slug(f"{base}-s{shard_idx}-{num_shards}")


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


def merge_shard_reports(
    shard_paths: list[Path],
    *,
    suite: str,
    expected_shards: int | None = None,
    expected_cases: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate and merge shard JSON reports with strict suite/duplicate/count checks."""
    if not shard_paths:
        raise ValueError(f"No shard JSON files found for suite {suite!r}")
    if expected_shards is not None and len(shard_paths) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} shard files for {suite!r}, found {len(shard_paths)}"
        )

    by_name: dict[str, dict[str, Any]] = {}
    seen_shards: set[str] = set()
    presets_seen: set[tuple[str, str]] = set()
    for p in shard_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        file_suite = data.get("suite")
        if file_suite is not None and str(file_suite) != suite:
            raise ValueError(f"Contaminated shard {p}: suite={file_suite!r} != expected {suite!r}")
        if strict and file_suite is None:
            raise ValueError(f"Missing 'suite' metadata in shard {p}")
        shard_id = data.get("shard")
        if shard_id is not None:
            shard_key = str(shard_id)
            if shard_key in seen_shards:
                raise ValueError(f"Duplicate shard {shard_key!r} encountered in {p}")
            seen_shards.add(shard_key)
        elif strict:
            raise ValueError(f"Missing 'shard' metadata in shard {p}")
        presets_seen.add(
            (
                str(data.get("physics_v2_preset", "legacy")),
                str(data.get("gnc_v2_preset", "legacy")),
            )
        )
        for r in data.get("results", []):
            name = str(r["scenario"])
            if name in by_name:
                raise ValueError(f"Duplicate scenario {name!r} across shards (in {p})")
            by_name[name] = r

    if len(presets_seen) > 1:
        raise ValueError(f"Inconsistent v2 presets across shards: {sorted(presets_seen)}")

    ordered = [by_name[k] for k in sorted(by_name)]
    total = len(ordered)
    if expected_cases is not None and total != expected_cases:
        raise ValueError(
            f"Merged case count mismatch for {suite!r}: got {total}, expected {expected_cases}"
        )
    passed = sum(1 for r in ordered if r.get("requirements_passed"))
    failed = [
        {
            "scenario": r["scenario"],
            "outcome": r["outcome"],
            "failed": r.get("failed_criteria", []),
        }
        for r in ordered
        if not r.get("requirements_passed")
    ]

    def _max_f(key: str) -> float | None:
        vals = [float(r[key]) for r in ordered if r.get(key) is not None]
        return max(vals) if vals else None

    phys_preset, gnc_preset = next(iter(presets_seen))
    is_v2 = phys_preset != "legacy" or gnc_preset != "legacy"
    return {
        "schema": "dm100_3d_validation_merged_v2" if is_v2 else "dm100_3d_validation_merged_v1",
        "suite": suite,
        "physics_v2_preset": phys_preset,
        "gnc_v2_preset": gnc_preset,
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
    parser.add_argument("--suite", default="edge-3d", choices=list(SUPPORTED_SUITES))
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--tag", default=None, help="Run namespace tag matching push_kaggle_shards")
    parser.add_argument("--legacy-slug", action="store_true", help="Use v1 '-shard-' slug format")
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--status-only", action="store_true", help="Only query shard statuses")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/kaggle_results"))
    parser.add_argument("--replays-dir", type=Path, default=Path("artifacts/replays"))
    args = parser.parse_args()

    username = _resolve_username(args.username)
    kaggle_exe = _resolve_kaggle_cli()

    def _slug_for(idx: int) -> str:
        if args.legacy_slug:
            return f"{username}/dm100-3d-{args.suite}-shard-{idx}-{args.shards}"
        slug = _make_shard_slug(
            suite=args.suite, shard_idx=idx, num_shards=args.shards, tag=args.tag
        )
        return f"{username}/{slug}"

    if args.status_only:
        for idx in range(args.shards):
            subprocess.run([kaggle_exe, "kernels", "status", _slug_for(idx)], check=False)
        return

    run_ns = f"{args.suite}_{args.tag}" if args.tag else args.suite
    run_out_dir = args.out_dir / run_ns
    run_out_dir.mkdir(parents=True, exist_ok=True)
    args.replays_dir.mkdir(parents=True, exist_ok=True)
    shard_jsons: list[Path] = []

    for idx in range(args.shards):
        slug = _slug_for(idx)
        s_out = run_out_dir / f"shard_{idx}"
        s_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [kaggle_exe, "kernels", "output", slug, "-p", str(s_out), "--force"],
            check=True,
        )
        exact_name = f"results_{args.suite}_{idx}_of_{args.shards}.json"
        matches = sorted(s_out.rglob(exact_name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 '{exact_name}' under {s_out}, found {len(matches)}: {matches}"
            )
        shard_jsons.append(matches[0])
        for rep in s_out.rglob("*_50hz.json"):
            shutil.copy2(rep, args.replays_dir / rep.name)

    merged = merge_shard_reports(
        shard_jsons,
        suite=args.suite,
        expected_shards=args.shards,
        expected_cases=args.expected_cases,
        strict=True,
    )
    merged_path = run_out_dir / f"validation_3d_{run_ns}_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"Merged {merged['passed_cases']}/{merged['total_cases']} "
        f"({100.0 * merged['pass_rate']:.1f}%) -> {merged_path}"
    )


if __name__ == "__main__":
    main()
