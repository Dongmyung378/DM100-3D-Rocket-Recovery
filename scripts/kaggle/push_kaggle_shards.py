"""Push N parallel Kaggle CPU validation shards (4 vCPU each, 0 GPU quota).

Usage:
  python scripts/kaggle/push_kaggle_shards.py --username <kaggle_user> --suite edge-3d --shards 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "run_shard_template.py"


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
) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    env_block = (
        "\nimport os as _dm100_os\n"
        f"_dm100_os.environ['DM100_SUITE'] = {suite!r}\n"
        f"_dm100_os.environ['DM100_SHARD'] = '{shard_idx}/{num_shards}'\n"
        f"_dm100_os.environ['DM100_REPO_URL'] = {repo_url!r}\n"
        f"_dm100_os.environ['DM100_REPO_REF'] = {repo_ref!r}\n"
        "_dm100_os.environ['DM100_WORKERS'] = '4'\n"
    )
    future_line = "from __future__ import annotations\n"
    if future_line in src:
        return src.replace(future_line, future_line + env_block, 1)
    return env_block + "\n" + src


def main() -> None:
    parser = argparse.ArgumentParser(description="Push sharded DM100 3D validation to Kaggle CPU.")
    parser.add_argument(
        "--username",
        default=None,
        help="Kaggle username (defaults to username in ~/.kaggle/kaggle.json)",
    )
    parser.add_argument("--suite", default="edge-3d", choices=["edge-3d", "holdout-3d", "full-3d"])
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--repo-url",
        default="https://github.com/Dongmyung378/DM100-3D-Rocket-Recovery.git",
    )
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--dataset-source", default=None, help="Optional private dataset slug")
    parser.add_argument("--stage-dir", type=Path, default=Path("artifacts/kaggle_stage"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    username = _resolve_username(args.username)
    kaggle_exe = None if args.dry_run else _resolve_kaggle_cli()

    for idx in range(args.shards):
        slug = f"dm100-3d-{args.suite}-s{idx}of{args.shards}"
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
            ),
            encoding="utf-8",
        )
        meta = {
            "id": f"{username}/{slug}",
            "title": f"DM100 3D {args.suite} Shard {idx}/{args.shards}",
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
        print(f"Prepared shard {idx}/{args.shards} -> {sdir}")
        if not args.dry_run:
            assert kaggle_exe is not None
            subprocess.run([kaggle_exe, "kernels", "push", "-p", str(sdir)], check=True)


if __name__ == "__main__":
    main()
