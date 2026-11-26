"""Kaggle CPU sharded runner script for DM100 3D headless validation suites.

Runs on free Kaggle CPU kernels (4 vCPU, 30 GB RAM, Internet ON, GPU OFF, 0 GPU quota).
Supports Public GitHub checkout (branch, tag, OR raw commit SHA) and pinned lockfile installs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get(
    "DM100_REPO_URL",
    "https://github.com/Dongmyung378/DM100-3D-Rocket-Recovery.git",
)
REPO_REF = os.environ.get("DM100_REPO_REF", "main")
SUITE = os.environ.get("DM100_SUITE", "edge-3d")
SHARD = os.environ.get("DM100_SHARD", "0/4")
WORKERS = os.environ.get("DM100_WORKERS", "4")
PHYSICS_V2_PRESET = os.environ.get("DM100_PHYSICS_V2_PRESET", "legacy")
GNC_V2_PRESET = os.environ.get("DM100_GNC_V2_PRESET", "legacy")
REPLAY_CASES = os.environ.get("DM100_REPLAY_CASES", "")

WORK_DIR = Path("/kaggle/working")
DATASET_SRC = Path("/kaggle/input/dm100-3d-src")
REPO_DIR = Path("/tmp/DM100-3D-Rocket-Recovery")

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _checkout_repo(repo_url: str, repo_ref: str, dest: Path) -> None:
    """Checkout branch, tag, or raw commit SHA cleanly using git fetch + FETCH_HEAD."""
    if dest.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    _run(["git", "init"], cwd=dest)
    _run(["git", "remote", "add", "origin", repo_url], cwd=dest)
    _run(["git", "fetch", "--depth", "1", "origin", repo_ref], cwd=dest)
    _run(["git", "checkout", "FETCH_HEAD"], cwd=dest)


def main() -> None:
    if DATASET_SRC.exists():
        src_root = DATASET_SRC
    else:
        _checkout_repo(REPO_URL, REPO_REF, REPO_DIR)
        src_root = REPO_DIR

    lockfile = src_root / "requirements-kaggle.lock"
    if lockfile.exists():
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "-r",
                str(lockfile),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "--no-deps",
                "-e",
                str(src_root),
            ]
        )
    else:
        _run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-e", str(src_root)])

    out_dir = WORK_DIR / "dm100_3d_outputs"
    replays_dir = out_dir / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = SHARD.replace("/", "_of_")
    eval_cmd = [
        sys.executable,
        str(src_root / "scripts" / "evaluate_3d.py"),
        "--suite",
        SUITE,
        "--shard",
        SHARD,
        "--workers",
        WORKERS,
        "--physics-v2-preset",
        PHYSICS_V2_PRESET,
        "--gnc-v2-preset",
        GNC_V2_PRESET,
        "--resume",
        str(out_dir / f"checkpoint_{SUITE}_{shard_tag}.jsonl"),
        "--out-json",
        str(out_dir / f"results_{SUITE}_{shard_tag}.json"),
        "--export-replays-dir",
        str(replays_dir),
    ]
    if REPLAY_CASES.strip():
        eval_cmd.extend(["--replay-cases", REPLAY_CASES.strip()])
    _run(eval_cmd, cwd=src_root)


if __name__ == "__main__":
    main()
