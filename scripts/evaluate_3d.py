"""Headless sharded 3D validation runner with crash-safe JSONL resume checkpointing.

Designed for multi-kernel Kaggle CPU offload (`--workers 4 --shard k/N --resume ...`)
and optionally exports frozen 50 Hz `dm100_replay_3d_v1` JSON replays for verified hero runs.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from dm100_3d.replay import export_replay_json
from dm100_3d.sim_3d import Scenario3D, SimConfig3D, run_flight_3d
from dm100_3d.validation_suites import get_validation_suite, select_shard

HERO_REPLAY_CASES = frozenset(
    {
        "edge3d-01-calm-nominal",
        "edge3d-29-combined-crosswind-roll-imu",
    }
)


def _run_one(
    payload: tuple[Scenario3D, str | None],
) -> dict[str, Any]:
    scen, replay_dir_str = payload
    want_replay = replay_dir_str is not None and scen.name in HERO_REPLAY_CASES
    t0 = perf_counter()
    summary, frames = run_flight_3d(
        scen,
        sim_config=SimConfig3D(
            record_replay=want_replay,
            seed=scen.seed,
            sensor_seed=scen.seed + 1000,
        ),
    )
    summary["wall_time_s"] = perf_counter() - t0
    if want_replay and replay_dir_str is not None:
        out_path = Path(replay_dir_str) / f"{scen.name}_50hz.json"
        export_replay_json(frames, out_path, metadata=summary)
        summary["replay_exported_path"] = str(out_path)
    return summary


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        done[str(item["scenario"])] = item
    return done


def summarize_results(results: list[dict[str, Any]], *, suite: str, shard: str) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("requirements_passed"))
    failed_cases = [
        {"scenario": r["scenario"], "outcome": r["outcome"], "failed": r["failed_criteria"]}
        for r in results
        if not r.get("requirements_passed")
    ]

    def _max_field(key: str) -> float | None:
        vals = [float(r[key]) for r in results if r.get(key) is not None]
        return max(vals) if vals else None

    return {
        "schema": "dm100_3d_validation_v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "suite": suite,
        "shard": shard,
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": (passed / total) if total > 0 else 0.0,
        "failed_cases": failed_cases,
        "max_position_error_m": _max_field("position_error_m"),
        "max_vertical_foot_speed_m_s": _max_field("vertical_foot_speed_m_s"),
        "max_horizontal_foot_speed_m_s": _max_field("horizontal_foot_speed_m_s"),
        "max_tilt_deg": _max_field("tilt_deg"),
        "max_dynamic_pressure_pa": _max_field("peak_dynamic_pressure_pa"),
        "max_heat_flux_w_m2": _max_field("peak_heat_flux_w_m2"),
        "max_angular_rate_deg_s": _max_field("peak_angular_rate_deg_s"),
        "max_angular_acceleration_deg_s2": _max_field("peak_angular_acceleration_deg_s2"),
        "max_roll_deg": _max_field("peak_roll_deg"),
        "max_fin_hinge_nm": _max_field("peak_fin_hinge_nm"),
        "results": sorted(results, key=lambda r: str(r["scenario"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sharded DM100 3D headless validation suite.")
    parser.add_argument("--suite", default="edge-3d", choices=["edge-3d", "holdout-3d", "full-3d"])
    parser.add_argument("--shard", default="0/1", help="Shard spec '<index>/<total>'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None, help="JSONL checkpoint path")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--export-replays-dir", type=Path, default=None)
    args = parser.parse_args()

    all_cases = get_validation_suite(args.suite)
    shard_cases = select_shard(all_cases, args.shard)
    completed = load_checkpoint(args.resume) if args.resume is not None else {}
    pending = [c for c in shard_cases if c.name not in completed]

    print(
        f"[DM100-3D] suite={args.suite} shard={args.shard} "
        f"total={len(shard_cases)} resumed={len(completed)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )

    t0 = perf_counter()
    replay_dir_str = str(args.export_replays_dir) if args.export_replays_dir is not None else None
    if args.resume is not None:
        args.resume.parent.mkdir(parents=True, exist_ok=True)

    if pending:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(_run_one, (c, replay_dir_str)): c.name for c in pending}
            for fut in as_completed(futs):
                res = fut.result()
                completed[str(res["scenario"])] = res
                if args.resume is not None:
                    with args.resume.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(res, separators=(",", ":")) + "\n")
                status = "PASS" if res["requirements_passed"] else f"FAIL {res['failed_criteria']}"
                print(
                    f"  [{len(completed)}/{len(shard_cases)}] {res['scenario']}: {status} "
                    f"(pos={res.get('position_error_m')}, roll={res.get('peak_roll_deg'):.2f} deg, "
                    f"t={res['wall_time_s']:.1f}s)",
                    flush=True,
                )

    ordered = [completed[c.name] for c in shard_cases if c.name in completed]
    report = summarize_results(ordered, suite=args.suite, shard=args.shard)
    report["wall_time_total_s"] = perf_counter() - t0
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[DM100-3D] DONE {report['passed_cases']}/{report['total_cases']} "
        f"({100.0 * report['pass_rate']:.1f}%) -> {args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
