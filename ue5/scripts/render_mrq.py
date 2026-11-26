"""Offline 1080p Movie Render Queue (MRQ) renderer & ffmpeg muxer for DM100 3D in UE 5.4.

Hardware target: RTX 3060 Laptop (6 GB VRAM)
  - Resolution: 1920x1080 (1080p), 30 fps (or 50 fps) offline PNG sequence
  - VRAM guard: `r.Streaming.PoolSize 2048`, classic CSM shadows, zero real-time Lumen RT spikes
  - Multi-cam highlight windows: Liftoff, MECO/Separation/Flip, Entry Burn Onboard Fin-Cam,
    Hoverslam Approach, and 4-Leg Touchdown + 5 s Settle.

Usage inside UE 5.4 (headless MRQ render via DM100RuntimeMRQExecutor):
  UnrealEditor-Cmd.exe ue5/DM100Cinematics/DM100Cinematics.uproject ^
      /Game/DM100/Maps/DM100_RTLS_Level -game ^
      -MoviePipelineLocalExecutorClass=/Script/MovieRenderPipelineCore.MoviePipelinePythonHostExecutor
      -ExecutorPythonClass=/Engine/PythonTypes.DM100RuntimeMRQExecutor ^
      -LevelSequence=/Game/DM100/Sequences/LS_DM100_Calm_3D ^
      -OutDir=ue5/renders/calm -StartFrame=0 -EndFrame=300 -ResX=1920 -ResY=1080 -FPS=30

Usage outside UE 5.4 (mux rendered PNG frames into H.264 MP4 via ffmpeg):
  python ue5/scripts/render_mrq.py --mux-only --out-dir ue5/renders/calm --fps 30
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def mux_pngs_to_mp4(frames_dir: Path, output_mp4: Path, *, fps: int = 30) -> Path:
    """Assemble rendered 1080p PNG sequence into a web-compatible H.264 MP4 via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH; install ffmpeg to mux PNG renders.")
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    # Write ffmpeg concat list so arbitrary highlight frame numbering joins seamlessly
    list_file = frames_dir / "_ffmpeg_frames.txt"
    dt = 1.0 / float(max(fps, 1))
    lines = [f"file '{p.name}'\nduration {dt:.6f}" for p in pngs]
    lines.append(f"file '{pngs[-1].name}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)
    return output_mp4


def run_mrq_inside_ue5(
    *,
    level_path: str,
    sequence_path: str,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    spatial_samples: int = 4,
) -> None:
    """Configure and execute an offline 1080p Movie Render Queue job inside UE 5.4."""
    import unreal  # type: ignore[import-not-found]

    out_dir.mkdir(parents=True, exist_ok=True)
    queue_sub = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = queue_sub.get_queue()
    queue.delete_all_jobs()

    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = f"DM100_{start_frame}_{end_frame}"
    job.map = unreal.SoftObjectPath(level_path)
    job.sequence = unreal.SoftObjectPath(sequence_path)

    cfg = job.get_configuration()
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

    out_setting = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out_setting.output_directory = unreal.DirectoryPath(str(out_dir))
    out_setting.file_name_format = "dm100_{frame_number}"
    out_setting.output_resolution = unreal.IntPoint(int(width), int(height))
    out_setting.use_custom_frame_rate = True
    out_setting.output_frame_rate = unreal.FrameRate(int(fps), 1)
    out_setting.use_custom_playback_range = True
    out_setting.custom_start_frame = int(start_frame)
    out_setting.custom_end_frame = int(end_frame)
    out_setting.zero_pad_frame_numbers = 6

    aa_setting = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa_setting.spatial_sample_count = int(spatial_samples)
    aa_setting.temporal_sample_count = 1
    aa_setting.override_anti_aliasing = True
    aa_setting.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE

    cvar_setting = cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for k_cvar, v_cvar in (
        ("r.Streaming.PoolSize", 2048.0),
        ("r.MotionBlurQuality", 0.0),
        ("r.ToneMapper.Sharpen", 0.5),
        ("r.Shadow.CSM.MaxCascades", 4.0),
    ):
        cvar_setting.add_or_update_console_variable(k_cvar, v_cvar)

    executor = unreal.MoviePipelinePIEExecutor()

    def _on_finished(_exec: object, success: bool) -> None:
        unreal.log(f"[DM100-MRQ] Finished render success={success} -> {out_dir}")

    executor.on_executor_finished_delegate.add_callable(_on_finished)
    queue_sub.render_queue_with_executor_instance(executor)


def main() -> None:
    parser = argparse.ArgumentParser(description="DM100 3D UE 5.4 offline 1080p MRQ & ffmpeg tool")
    parser.add_argument("--level", default="/Game/DM100/Maps/DM100_RTLS_Level")
    parser.add_argument(
        "--sequence", default="/Game/DM100/Sequences/LS_edge3d-01-calm-nominal_50hz"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("ue5/renders/dm100_1080p"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=700)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--spatial-samples", type=int, default=4)
    parser.add_argument("--mux-only", action="store_true", help="Only mux existing PNGs via ffmpeg")
    parser.add_argument("--mp4", type=Path, default=None, help="Output MP4 path when --mux-only")
    args, _ = parser.parse_known_args()

    if args.mux_only:
        mp4 = args.mp4 or (args.out_dir / "dm100_3d_showcase_1080p.mp4")
        out = mux_pngs_to_mp4(args.out_dir, mp4, fps=args.fps)
        print(f"[OK] Muxed 1080p MP4 -> {out}")
        return

    try:
        import unreal  # noqa: F401
    except ImportError:
        print(
            "Run this script inside UE 5.4 (UnrealEditor-Cmd.exe) to execute MRQ, "
            "or pass --mux-only to encode rendered PNG frames with ffmpeg.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    run_mrq_inside_ue5(
        level_path=args.level,
        sequence_path=args.sequence,
        out_dir=args.out_dir.resolve(),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        width=args.width,
        height=args.height,
        fps=args.fps,
        spatial_samples=args.spatial_samples,
    )


if __name__ == "__main__":
    main()
