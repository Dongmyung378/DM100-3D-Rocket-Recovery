"""Push N parallel Kaggle CPU validation shards (4 vCPU each, 0 GPU quota).

Usage:
  python scripts/kaggle/push_kaggle_shards.py --suite edge-3d --shards 4 --physics-v2-preset m1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "run_shard_template.py"

SUPPORTED_SUITES = (
    "edge-3d",
    "holdout-3d",
    "full-3d",
    "edge-v2",
    "mc-1000",
    "fault-mc-160",
    "envelope-boundary",
)


def sanitize_slug(raw: str) -> str:
    """Enforce Kaggle slug hygiene: lowercase ASCII alphanumerics + single hyphens, max 50 chars."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if len(cleaned) > 50:
        cleaned = cleaned[:50].rstrip("-")
    if not cleaned:
        raise ValueError(f"Empty Kaggle slug after sanitization: {raw!r}")
    return cleaned


def make_shard_slug(*, suite: str, shard_idx: int, num_shards: int, tag: str | None = None) -> str:
    base = f"dm100-3d-{suite}-{tag}" if tag else f"dm100-3d-{suite}"
    return sanitize_slug(f"{base}-s{shard_idx}-{num_shards}")


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
    raise SystemExit(
        "Kaggle username not provided and ~/.kaggle/kaggle.json not found. "
        "Download kaggle.json from Kaggle Settings -> API -> Create New Token."
    )


def build_shard_script(
    *,
    suite: str,
    shard_idx: int,
    num_shards: int,
    repo_url: str,
    repo_ref: str,
    physics_v2_preset: str = "legacy",
    gnc_v2_preset: str = "legacy",
    replay_cases: str = "",
) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    env_block = (
        "\nimport os as _dm100_os\n"
        f"_dm100_os.environ['DM100_SUITE'] = {suite!r}\n"
        f"_dm100_os.environ['DM100_SHARD'] = '{shard_idx}/{num_shards}'\n"
        f"_dm100_os.environ['DM100_REPO_URL'] = {repo_url!r}\n"
        f"_dm100_os.environ['DM100_REPO_REF'] = {repo_ref!r}\n"
        f"_dm100_os.environ['DM100_PHYSICS_V2_PRESET'] = {physics_v2_preset!r}\n"
        f"_dm100_os.environ['DM100_GNC_V2_PRESET'] = {gnc_v2_preset!r}\n"
        f"_dm100_os.environ['DM100_REPLAY_CASES'] = {replay_cases!r}\n"
        "_dm100_os.environ['DM100_WORKERS'] = '4'\n"
    )
    future_line = "from __future__ import annotations\n"
    if future_line in src:
        return src.replace(future_line, future_line + env_block, 1)
    return env_block + "\n" + src


def select_shard_indices(
    num_shards: int,
    *,
    explicit_indices: str | None = None,
    batch: str | None = None,
) -> list[int]:
    if explicit_indices is not None:
        indices = sorted({int(tok.strip()) for tok in explicit_indices.split(",") if tok.strip()})
        for idx in indices:
            if not (0 <= idx < num_shards):
                raise ValueError(f"Shard index {idx} out of range [0, {num_shards})")
        return indices
    if batch is not None:
        parts = batch.strip().split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid --batch {batch!r}; expected '<batch_idx>/<num_batches>'")
        b_idx, b_tot = int(parts[0]), int(parts[1])
        if b_tot <= 0 or not (0 <= b_idx < b_tot):
            raise ValueError(f"Invalid batch range {b_idx}/{b_tot}")
        per_batch = (num_shards + b_tot - 1) // b_tot
        start = b_idx * per_batch
        end = min(num_shards, start + per_batch)
        return list(range(start, end))
    return list(range(num_shards))


def _push_with_retry(kaggle_exe: str, sdir: Path, *, max_retries: int = 3) -> None:
    for attempt in range(1, max_retries + 1):
        res = subprocess.run([kaggle_exe, "kernels", "push", "-p", str(sdir)], check=False)
        if res.returncode == 0:
            return
        if attempt < max_retries:
            wait_s = 5 * attempt
            print(
                f"  Push retry {attempt}/{max_retries} for {sdir.name} after {wait_s}s...",
                flush=True,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to push Kaggle kernel after {max_retries} attempts: {sdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push sharded DM100 3D validation to Kaggle CPU.")
    parser.add_argument(
        "--username",
        default=None,
        help="Kaggle username (defaults to username in ~/.kaggle/kaggle.json)",
    )
    parser.add_argument("--suite", default="edge-3d", choices=list(SUPPORTED_SUITES))
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--tag", default=None, help="Optional run namespace tag (e.g., m1, m2)")
    parser.add_argument(
        "--shard-indices",
        default=None,
        help="Optional comma-separated shard indices to push (e.g. '0,1,2,3')",
    )
    parser.add_argument(
        "--batch",
        default=None,
        help="Optional batch slice '<batch_idx>/<num_batches>' for wave submission",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Advisory concurrency guard warning when pushing more than N shards at once",
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/Dongmyung378/DM100-3D-Rocket-Recovery.git",
    )
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument(
        "--physics-v2-preset",
        default="legacy",
        choices=["legacy", "m1", "m2", "full"],
    )
    parser.add_argument(
        "--gnc-v2-preset",
        default="legacy",
        choices=["legacy", "m1", "m2", "full"],
    )
    parser.add_argument(
        "--replay-cases",
        default="",
        help="Optional comma-separated scenario allowlist for 50 Hz replay export",
    )
    parser.add_argument("--dataset-source", default=None, help="Optional private dataset slug")
    parser.add_argument("--stage-dir", type=Path, default=Path("artifacts/kaggle_stage"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    username = _resolve_username(args.username)
    kaggle_exe = None if args.dry_run else _resolve_kaggle_cli()
    indices = select_shard_indices(
        args.shards, explicit_indices=args.shard_indices, batch=args.batch
    )
    if len(indices) > args.max_concurrent and not args.dry_run:
        print(
            f"[WARN] Submitting {len(indices)} shards > --max-concurrent={args.max_concurrent}; "
            "consider using --batch or --shard-indices to respect Kaggle concurrency limits.",
            flush=True,
        )

    for idx in indices:
        slug = make_shard_slug(
            suite=args.suite, shard_idx=idx, num_shards=args.shards, tag=args.tag
        )
        sdir = args.stage_dir / slug
        sdir.mkdir(parents=True, exist_ok=True)
        script_path = sdir / "run_shard.py"
        script_path.write_text(
            build_shard_script(
                suite=args.suite,
                shard_idx=idx,
                num_shards=args.shards,
                repo_url=args.repo_url,
                repo_ref=args.repo_ref,
                physics_v2_preset=args.physics_v2_preset,
                gnc_v2_preset=args.gnc_v2_preset,
                replay_cases=args.replay_cases,
            ),
            encoding="utf-8",
        )
        title_tag = f" {args.tag.upper()}" if args.tag else ""
        meta = {
            "id": f"{username}/{slug}",
            "title": f"DM100 3D {args.suite}{title_tag} S{idx}-{args.shards}",
            "code_file": "run_shard.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [args.dataset_source] if args.dataset_source else [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (sdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Prepared shard {idx}/{args.shards} ({slug}) -> {sdir}")
        if not args.dry_run:
            assert kaggle_exe is not None
            _push_with_retry(kaggle_exe, sdir)


if __name__ == "__main__":
    main()
