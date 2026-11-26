"""Headless UE 5.4 Movie Render Queue runtime executor for DM100 3D 1080p offline renders.

Registered on engine startup via `Content/Python/init_unreal.py` and invoked with
`-game`, `-MoviePipelineLocalExecutorClass`, and `-ExecutorPythonClass`:
  UnrealEditor-Cmd.exe DM100Cinematics.uproject /Game/DM100/Maps/DM100_RTLS_Level -game ^
      -LevelSequence=/Game/DM100/Sequences/LS_DM100_Calm_3D ^
      -OutDir=ue5/renders/calm -StartFrame=0 -EndFrame=300 -ResX=1920 -ResY=1080 -FPS=30
"""

from __future__ import annotations

import unreal  # type: ignore[import-not-found]


@unreal.uclass()
class DM100RuntimeMRQExecutor(unreal.MoviePipelinePythonHostExecutor):
    active_pipeline = unreal.uproperty(unreal.MoviePipeline)

    def _post_init(self) -> None:
        self.active_pipeline = None

    @unreal.ufunction(override=True)
    def execute_delayed(self, in_pipeline_queue) -> None:
        _tokens, _switches, params = unreal.SystemLibrary.parse_command_line(
            unreal.SystemLibrary.get_command_line()
        )
        seq_path = params.get("LevelSequence", "/Game/DM100/Sequences/LS_DM100_Calm_3D")
        out_dir = params.get("OutDir", "d:/Projects/DM100-3D-Rocket-Recovery/ue5/renders/dm100")
        start_f = int(params.get("StartFrame", "0"))
        end_f = int(params.get("EndFrame", "150"))
        res_x = int(params.get("ResX", "1920"))
        res_y = int(params.get("ResY", "1080"))
        fps = int(params.get("FPS", "30"))
        samples = int(params.get("Samples", "2"))
        prefix = params.get("Prefix", "dm100")

        unreal.log(
            f"[DM100-MRQ] Starting offline render: seq={seq_path}, frames={start_f}..{end_f}, "
            f"{res_x}x{res_y}@{fps}fps, samples={samples} -> {out_dir}"
        )

        self.pipeline_queue = unreal.new_object(unreal.MoviePipelineQueue, outer=self)
        job = self.pipeline_queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.job_name = f"DM100_{prefix}_{start_f}_{end_f}"
        job.sequence = unreal.SoftObjectPath(seq_path)

        cfg = job.get_configuration()
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        out_set = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        out_set.output_directory = unreal.DirectoryPath(out_dir)
        out_set.file_name_format = f"{prefix}_{{frame_number}}"
        out_set.output_resolution = unreal.IntPoint(res_x, res_y)
        out_set.use_custom_frame_rate = True
        out_set.output_frame_rate = unreal.FrameRate(fps, 1)
        out_set.use_custom_playback_range = True
        out_set.custom_start_frame = start_f
        out_set.custom_end_frame = end_f
        out_set.zero_pad_frame_numbers = 6

        aa_set = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa_set.spatial_sample_count = samples
        aa_set.temporal_sample_count = 1
        aa_set.override_anti_aliasing = True
        aa_set.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE

        cvar_set = cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        for k_cvar, v_cvar in (
            ("r.Streaming.PoolSize", 2048.0),
            ("r.MotionBlurQuality", 0.0),
            ("r.ToneMapper.Sharpen", 0.5),
            ("r.Shadow.CSM.MaxCascades", 4.0),
            ("r.SkyAtmosphere.FastSkyLUT", 1.0),
        ):
            cvar_set.add_or_update_console_variable(k_cvar, v_cvar)

        cfg.initialize_transient_settings()

        self.active_pipeline = unreal.new_object(
            self.target_pipeline_class,
            outer=self.get_last_loaded_world(),
            base_type=unreal.MoviePipeline,
        )
        self.active_pipeline.on_movie_pipeline_finished_delegate.add_function_unique(
            self, "on_movie_pipeline_finished"
        )
        self.active_pipeline.initialize(job)

    @unreal.ufunction(override=True)
    def on_begin_frame(self) -> None:
        super().on_begin_frame()

    @unreal.ufunction(override=True)
    def on_map_load(self, in_world) -> None:
        pass

    @unreal.ufunction(override=True)
    def is_rendering(self) -> bool:
        return self.active_pipeline is not None

    @unreal.ufunction(ret=None, params=[unreal.MoviePipeline, bool])
    def on_movie_pipeline_finished(self, in_pipeline, b_success: bool) -> None:
        unreal.log(f"[DM100-MRQ] Finished offline render success={b_success}")
        self.active_pipeline = None
        self.on_executor_finished_impl()
