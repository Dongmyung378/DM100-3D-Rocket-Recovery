"""Push N parallel Kaggle CPU validation shards (4 vCPU each, 0 GPU quota).

Usage:
  python scripts/kaggle/push_kaggle_shards.py --username <kaggle_user> --suite edge-3d --shards 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "run_shard_template.py"


def build_shard_script(
    *,
    suite: str,
    shard_idx: int,
    num_shards: int,
    repo_url: str,
    repo_ref: str,
) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    header = (
        "import os\n"
        f"os.environ['DM100_SUITE'] = {suite!r}\n"
        f"os.environ['DM100_SHARD'] = '{shard_idx}/{num_shards}'\n"
        f"os.environ['DM100_REPO_URL'] = {repo_url!r}\n"
        f"os.environ['DM100_REPO_REF'] = {repo_ref!r}\n"
        "os.environ['DM100_WORKERS'] = '4'\n\n"
    )
    return header + src


def main() -> None:
    parser = argparse.ArgumentParser(description="Push sharded DM100 3D validation to Kaggle CPU.")
    parser.add_argument("--username", required=True, help="Kaggle username")
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
            "id": f"{args.username}/{slug}",
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
            subprocess.run(["kaggle", "kernels", "push", "-p", str(sdir)], check=True)


if __name__ == "__main__":
    main()
