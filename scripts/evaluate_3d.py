"""Headless sharded 3D validation runner with crash-safe JSONL resume checkpointing.

Designed for multi-kernel Kaggle CPU offload (`--workers 4 --shard k/N --resume ...`)
and optionally exports frozen 50 Hz `dm100_replay_3d_v1`/`v2` JSON replays for explicit allowlists.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np  # noqa: E402

from dm100_3d.config import GncV2Config, PhysicsV2Config  # noqa: E402
from dm100_3d.replay import export_replay_json  # noqa: E402
from dm100_3d.sim_3d import Scenario3D, SimConfig3D, run_flight_3d  # noqa: E402
from dm100_3d.validation_suites import get_validation_suite, select_shard  # noqa: E402

HERO_REPLAY_CASES = frozenset(
    {
        "edge3d-01-calm-nominal",
        "edge3d-29-combined-crosswind-roll-imu",
    }
)

SUPPORTED_SUITES = (
    "edge-3d",
    "holdout-3d",
    "full-3d",
    "edge-v2",
    "mc-1000",
    "fault-mc-160",
    "envelope-boundary",
)


def _resolve_physics_v2(preset: str) -> PhysicsV2Config:
    if preset == "legacy":
        return PhysicsV2Config()
    if preset == "m1":
        return PhysicsV2Config.m1_preset()
    if preset == "m2":
        return PhysicsV2Config.m2_preset()
    if preset == "full":
        return PhysicsV2Config.full_v2()
    raise ValueError(f"Unknown physics v2 preset: {preset!r}")


def _resolve_gnc_v2(preset: str) -> GncV2Config:
    if preset == "legacy":
        return GncV2Config()
    if preset == "m1":
        return GncV2Config.m1_preset()
    if preset == "m2":
        return GncV2Config.m2_preset()
    if preset == "full":
        return GncV2Config.full_v2()
    raise ValueError(f"Unknown gnc v2 preset: {preset!r}")


def _derive_sensor_seed(case_seed: int, *, physics_preset: str, case_index: int) -> int:
    if physics_preset == "legacy":
        return case_seed + 1000
    ss = np.random.SeedSequence([int(case_seed), int(case_index), 202601])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _run_one(
    payload: tuple[Scenario3D, int, str | None, frozenset[str], str, str],
) -> dict[str, Any]:
    scen, case_idx, replay_dir_str, replay_allowlist, phys_preset, gnc_preset = payload
    want_replay = replay_dir_str is not None and scen.name in replay_allowlist
    t0 = perf_counter()
    try:
        phys_v2 = _resolve_physics_v2(phys_preset)
        gnc_v2 = _resolve_gnc_v2(gnc_preset)
        sensor_seed = _derive_sensor_seed(
            scen.seed, physics_preset=phys_preset, case_index=case_idx
        )
        summary, frames = run_flight_3d(
            scen,
            sim_config=SimConfig3D(
                record_replay=want_replay,
                seed=scen.seed,
                sensor_seed=sensor_seed,
                physics_v2=phys_v2,
                gnc_v2=gnc_v2,
            ),
        )
        summary["wall_time_s"] = perf_counter() - t0
        summary["physics_v2_preset"] = phys_preset
        summary["gnc_v2_preset"] = gnc_preset
        if want_replay and replay_dir_str is not None:
            out_path = Path(replay_dir_str) / f"{scen.name}_50hz.json"
            export_replay_json(frames, out_path, metadata=summary)
            summary["replay_exported_path"] = str(out_path)
        return summary
    except Exception as exc:  # pragma: no cover - crash guard for cloud shards
        return {
            "version": "dm100_3d_mujoco",
            "scenario": scen.name,
            "outcome": "worker_exception",
            "requirements_passed": False,
            "criteria": {},
            "failed_criteria": ["worker_exception"],
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
            "wall_time_s": perf_counter() - t0,
            "physics_v2_preset": phys_preset,
            "gnc_v2_preset": gnc_preset,
        }


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


def summarize_results(
    results: list[dict[str, Any]],
    *,
    suite: str,
    shard: str,
    physics_v2_preset: str = "legacy",
    gnc_v2_preset: str = "legacy",
) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("requirements_passed"))
    failed_cases = [
        {"scenario": r["scenario"], "outcome": r["outcome"], "failed": r.get("failed_criteria", [])}
        for r in results
        if not r.get("requirements_passed")
    ]

    def _max_field(key: str) -> float | None:
        vals = [float(r[key]) for r in results if r.get(key) is not None]
        return max(vals) if vals else None

    return {
        "schema": "dm100_3d_validation_v2"
        if (physics_v2_preset != "legacy" or gnc_v2_preset != "legacy")
        else "dm100_3d_validation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "shard": shard,
        "physics_v2_preset": physics_v2_preset,
        "gnc_v2_preset": gnc_v2_preset,
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
    parser.add_argument("--suite", default="edge-3d", choices=list(SUPPORTED_SUITES))
    parser.add_argument("--shard", default="0/1", help="Shard spec '<index>/<total>'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None, help="JSONL checkpoint path")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--export-replays-dir", type=Path, default=None)
    parser.add_argument(
        "--replay-cases",
        default=None,
        help="Comma-separated scenario allowlist for 50 Hz replay export (defaults to hero cases)",
    )
    parser.add_argument(
        "--physics-v2-preset",
        default="legacy",
        choices=["legacy", "m1", "m2", "full"],
        help="PhysicsV2Config preset ('legacy', 'm1', 'm2', or 'full')",
    )
    parser.add_argument(
        "--gnc-v2-preset",
        default="legacy",
        choices=["legacy", "m1", "m2", "full"],
        help="GncV2Config preset ('legacy', 'm1', 'm2', or 'full')",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=900.0,
        help="Per-case future timeout [s] to guard cloud shard hangs",
    )
    args = parser.parse_args()

    all_cases = get_validation_suite(args.suite)
    shard_cases = select_shard(all_cases, args.shard)
    completed = load_checkpoint(args.resume) if args.resume is not None else {}
    pending = [(idx, c) for idx, c in enumerate(shard_cases) if c.name not in completed]

    replay_allowlist = (
        frozenset(s.strip() for s in args.replay_cases.split(",") if s.strip())
        if args.replay_cases is not None
        else HERO_REPLAY_CASES
    )

    print(
        f"[DM100-3D] suite={args.suite} shard={args.shard} "
        f"phys_v2={args.physics_v2_preset} gnc_v2={args.gnc_v2_preset} "
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
            futs = {
                pool.submit(
                    _run_one,
                    (
                        c,
                        idx,
                        replay_dir_str,
                        replay_allowlist,
                        args.physics_v2_preset,
                        args.gnc_v2_preset,
                    ),
                ): c.name
                for idx, c in pending
            }
            for fut in as_completed(futs):
                scen_name = futs[fut]
                try:
                    res = fut.result(timeout=args.case_timeout_s)
                except Exception as exc:
                    res = {
                        "version": "dm100_3d_mujoco",
                        "scenario": scen_name,
                        "outcome": "worker_future_error",
                        "requirements_passed": False,
                        "criteria": {},
                        "failed_criteria": ["worker_future_error"],
                        "error": repr(exc),
                        "wall_time_s": 0.0,
                        "physics_v2_preset": args.physics_v2_preset,
                        "gnc_v2_preset": args.gnc_v2_preset,
                    }
                completed[str(res["scenario"])] = res
                if args.resume is not None:
                    with args.resume.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(res, separators=(",", ":")) + "\n")
                status = (
                    "PASS"
                    if res.get("requirements_passed")
                    else f"FAIL {res.get('failed_criteria')}"
                )
                pos_err = res.get("position_error_m")
                pos_str = f"{pos_err:.3f}m" if isinstance(pos_err, (int, float)) else str(pos_err)
                roll_deg = res.get("peak_roll_deg")
                roll_str = (
                    f"{roll_deg:.2f} deg" if isinstance(roll_deg, (int, float)) else str(roll_deg)
                )
                print(
                    f"  [{len(completed)}/{len(shard_cases)}] {res['scenario']}: {status} "
                    f"(pos={pos_str}, roll={roll_str}, t={res.get('wall_time_s', 0.0):.1f}s)",
                    flush=True,
                )

    ordered = [completed[c.name] for c in shard_cases if c.name in completed]
    report = summarize_results(
        ordered,
        suite=args.suite,
        shard=args.shard,
        physics_v2_preset=args.physics_v2_preset,
        gnc_v2_preset=args.gnc_v2_preset,
    )
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
