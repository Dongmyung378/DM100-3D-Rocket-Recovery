"""Unreal Engine 5.4 Python Importer: 50 Hz `dm100_replay_3d_v1` JSON -> LWC Level & Level Sequence.

Run headless via UnrealEditor-Cmd.exe (or from UE 5.4 Editor Output Log -> Python):
  $env:DM100_REPLAY_PATH="artifacts/replays/edge3d-01-calm-nominal_50hz.json"
  UnrealEditor-Cmd.exe ue5/DM100Cinematics/DM100Cinematics.uproject ^
      -ExecutePythonScript="ue5/scripts/import_dm100_replay.py" ^
      -unattended -nop4 -nosplash -NullRHI

Features:
  1. Imports exact-dimension procedural GLB rocket & pad meshes (`ue5/assets/meshes/*.glb`).
  2. Creates procedural PBR/emissive materials (Methalox plume, cold-gas RCS puff, reentry glow).
  3. Builds an LWC launch/landing level (`/Game/DM100/Maps/DM100_RTLS_Level`)
     with `SkyAtmosphere`, `DirectionalLight`, 10 m landing pad, articulated 1st-stage booster
     (4 grid fins, 4 landing legs, 7 Methalox plumes, 4 RCS puffs, aft thermal glow),
     separating Stage 2 upper stage (25 s post-sep orbital burn), and 4 CineCameras.
  4. Writes a 50 fps `LevelSequence` with exact ENU RH (m, wxyz) -> UE5 LH (cm, FQuat/FRotator)
     transform keys, quaternion interpolation, 7->3->1 engine cluster plume animation,
     4-fin & 4-leg deployment/stroke articulation, and multi-camera Cut Track.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Ensure sibling pure-Python math helper is importable inside UE 5.4 embedded Python
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ue5_transform_math import (  # noqa: E402
    compute_highlight_windows,
    compute_plume_scales,
    enu_pos_to_ue5_cm,
    enu_quat_wxyz_to_ue5_xyzw,
    resolve_booster_base_pos_enu,
    ue5_quat_xyzw_to_rotator_deg,
    unwrap_euler_deg,
)

try:
    import unreal
except ImportError as exc:  # pragma: no cover - only executed inside UE 5.4 Python
    raise SystemExit(
        "This script must be executed inside Unreal Engine 5.4 Python "
        "(UnrealEditor-Cmd.exe -ExecutePythonScript=...)."
    ) from exc


LEVEL_PATH = "/Game/DM100/Maps/DM100_RTLS_Level"
MESH_DEST = "/Game/DM100/Meshes"
MAT_DEST = "/Game/DM100/Materials"
SEQ_DEST = "/Game/DM100/Sequences"

# 7 Methalox nozzle local XY positions in UE5 cm (matches generate_rocket_glb.py, r = 92 cm)
NOZZLE_POS_CM: list[tuple[float, float, float]] = [(0.0, 0.0, -25.0)] + [
    (
        92.0 * math.cos(math.radians(60.0 * k)),
        -92.0 * math.sin(math.radians(60.0 * k)),
        -25.0,
    )
    for k in range(6)
]

# 4 Grid-fin mounts at z = 22.4 m, r = 1.55 m (ENU +X, -X, +Y, -Y -> UE5 +X, -X, -Y, +Y)
FIN_MOUNTS_CM: list[tuple[tuple[float, float, float], float]] = [
    ((155.0, 0.0, 2240.0), 0.0),
    ((-155.0, 0.0, 2240.0), 180.0),
    ((0.0, -155.0, 2240.0), -90.0),
    ((0.0, 155.0, 2240.0), 90.0),
]

# 4 Landing-leg mounts at z = 1.2 m, r = 1.55 m (ENU +X, -X, +Y, -Y -> UE5 +X, -X, -Y, +Y)
LEG_MOUNTS_CM: list[tuple[tuple[float, float, float], float]] = [
    ((155.0, 0.0, 120.0), 0.0),
    ((-155.0, 0.0, 120.0), 180.0),
    ((0.0, -155.0, 120.0), -90.0),
    ((0.0, 155.0, 120.0), 90.0),
]


def _log(msg: str) -> None:
    unreal.log(f"[DM100-UE5] {msg}")


def _import_glb_meshes(meshes_dir: Path) -> dict[str, list[Any]]:
    """Import procedural GLB meshes into per-asset subfolders and return all StaticMesh parts."""
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    mesh_names = [
        "dm100_booster_hull",
        "dm100_upper_stage",
        "dm100_grid_fin",
        "dm100_landing_leg",
        "dm100_landing_pad",
    ]
    tasks = []
    for name in mesh_names:
        src = meshes_dir / f"{name}.glb"
        if not src.exists():
            raise FileNotFoundError(
                f"Missing mesh {src}. Run `python ue5/scripts/generate_rocket_glb.py` first."
            )
        task = unreal.AssetImportTask()
        task.filename = str(src)
        task.destination_path = f"{MESH_DEST}/{name}"
        task.destination_name = name
        task.replace_existing = True
        task.automated = True
        task.save = True
        tasks.append(task)

    asset_tools.import_asset_tasks(tasks)

    loaded: dict[str, list[Any]] = {}
    for name in mesh_names:
        sub_dir = f"{MESH_DEST}/{name}"
        part_meshes: list[Any] = []
        for asset_path in sorted(unreal.EditorAssetLibrary.list_assets(sub_dir, recursive=True)):
            obj = unreal.EditorAssetLibrary.load_asset(asset_path)
            if isinstance(obj, unreal.StaticMesh):
                part_meshes.append(obj)
        if not part_meshes:
            _log(f"WARNING: No StaticMesh parts found under {sub_dir}; falling back to Cylinder")
            part_meshes = [
                unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cylinder.Cylinder")
            ]
        loaded[name] = part_meshes
        _log(f"Loaded '{name}': {len(part_meshes)} StaticMesh parts from {sub_dir}")
    return loaded


def _create_emissive_material(
    mat_name: str,
    base_rgb: tuple[float, float, float],
    emissive_rgb: tuple[float, float, float],
    roughness: float = 0.35,
) -> Any:
    """Create or load a lightweight procedural UE 5.4 material (RTX 3060 VRAM friendly)."""
    full_path = f"{MAT_DEST}/{mat_name}"
    if unreal.EditorAssetLibrary.does_asset_exist(full_path):
        mat = unreal.EditorAssetLibrary.load_asset(full_path)
        if mat is not None:
            return mat

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat_factory = unreal.MaterialFactoryNew()
    mat = asset_tools.create_asset(mat_name, MAT_DEST, unreal.Material, mat_factory)
    mel = unreal.MaterialEditingLibrary

    c_base = mel.create_material_expression(
        mat, unreal.MaterialExpressionConstant3Vector, -350, -100
    )
    c_base.constant = unreal.LinearColor(base_rgb[0], base_rgb[1], base_rgb[2], 1.0)
    mel.connect_material_property(c_base, "", unreal.MaterialProperty.MP_BASE_COLOR)

    c_em = mel.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -350, 120)
    c_em.constant = unreal.LinearColor(emissive_rgb[0], emissive_rgb[1], emissive_rgb[2], 1.0)
    mel.connect_material_property(c_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    c_rough = mel.create_material_expression(mat, unreal.MaterialExpressionConstant, -350, 280)
    c_rough.r = float(roughness)
    mel.connect_material_property(c_rough, "", unreal.MaterialProperty.MP_ROUGHNESS)

    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(full_path)
    return mat


def _make_rotator(pitch: float, yaw: float, roll: float) -> Any:
    """Construct UE 5.4 Rotator with explicit kwargs (UE Python order is roll, pitch, yaw)."""
    return unreal.Rotator(roll=float(roll), pitch=float(pitch), yaw=float(yaw))


def _spawn_mesh_actor(
    actor_sub: Any,
    label: str,
    mesh: Any,
    loc_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    material: Any | None = None,
    movable: bool = True,
) -> Any:
    actor = actor_sub.spawn_actor_from_class(
        unreal.StaticMeshActor,
        unreal.Vector(*loc_cm),
        _make_rotator(rot_deg[0], rot_deg[1], rot_deg[2]),
    )
    actor.set_actor_label(label)
    smc = actor.static_mesh_component
    smc.set_mobility(
        unreal.ComponentMobility.MOVABLE if movable else unreal.ComponentMobility.STATIC
    )
    smc.set_static_mesh(mesh)
    if material is not None:
        smc.set_material(0, material)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    return actor


def _get_transform_channels(section: Any) -> dict[str, Any]:
    """Return mapping of channel short-name -> MovieSceneScriptingDoubleChannel."""
    channels = section.get_all_channels()
    by_name: dict[str, Any] = {}
    ordered_names = [
        "Location.X",
        "Location.Y",
        "Location.Z",
        "Rotation.X",
        "Rotation.Y",
        "Rotation.Z",
        "Scale.X",
        "Scale.Y",
        "Scale.Z",
    ]
    for idx, ch in enumerate(channels):
        cname = str(ch.get_name())
        matched = False
        for key in ordered_names:
            if key in cname:
                by_name[key] = ch
                matched = True
                break
        if not matched and idx < len(ordered_names):
            by_name[ordered_names[idx]] = ch
    return by_name


def _spawn_assembly_actor(
    actor_sub: Any,
    label: str,
    part_meshes: list[Any],
    loc_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    movable: bool = True,
) -> Any:
    """Spawn a multi-part GLB assembly preserving every sub-part's PBR material and pivot."""
    root = _spawn_mesh_actor(
        actor_sub,
        label,
        part_meshes[0],
        loc_cm=loc_cm,
        rot_deg=rot_deg,
        movable=movable,
    )
    for idx, pm in enumerate(part_meshes[1:], start=1):
        child = _spawn_mesh_actor(
            actor_sub,
            f"{label}_part_{idx:02d}",
            pm,
            loc_cm=(0.0, 0.0, 0.0),
            rot_deg=(0.0, 0.0, 0.0),
            movable=movable,
        )
        child.attach_to_actor(
            root,
            "",
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            False,
        )
    return root


def _add_transform_section(
    seq: Any,
    actor: Any,
    end_frame: int,
    *,
    use_quat: bool = True,
    init_loc: tuple[float, float, float] | None = None,
    init_rot_pyr: tuple[float, float, float] | None = None,
    init_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[Any, dict[str, Any]]:
    binding = seq.add_possessable(actor)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(0, end_frame)
    if use_quat and hasattr(section, "set_use_quaternion_interpolation"):
        section.set_use_quaternion_interpolation(True)
    elif use_quat and hasattr(section, "set_editor_property"):
        with contextlib.suppress(Exception):
            section.set_editor_property("use_quaternion_interpolation", True)
    channels = _get_transform_channels(section)

    # Seed every channel default from actor's initial relative transform so unkeyed channels
    # NEVER collapse to (0, 0, 0) during Sequencer playback!
    if init_loc is None:
        rc = actor.root_component
        rel_l = rc.get_editor_property("relative_location")
        init_loc = (float(rel_l.x), float(rel_l.y), float(rel_l.z))
    if init_rot_pyr is None:
        rc = actor.root_component
        rel_r = rc.get_editor_property("relative_rotation")
        init_rot_pyr = (float(rel_r.pitch), float(rel_r.yaw), float(rel_r.roll))
    defaults = {
        "Location.X": init_loc[0],
        "Location.Y": init_loc[1],
        "Location.Z": init_loc[2],
        # Sequencer Rotation channels: X=Roll, Y=Pitch, Z=Yaw
        "Rotation.X": init_rot_pyr[2],
        "Rotation.Y": init_rot_pyr[0],
        "Rotation.Z": init_rot_pyr[1],
        "Scale.X": init_scale[0],
        "Scale.Y": init_scale[1],
        "Scale.Z": init_scale[2],
    }
    for key, val in defaults.items():
        if key in channels:
            channels[key].set_default(float(val))
    return binding, channels


def _key_vec3(
    channels: dict[str, Any],
    prefix: str,
    frame_num: Any,
    xyz: tuple[float, float, float],
) -> None:
    channels[f"{prefix}.X"].add_key(frame_num, float(xyz[0]))
    channels[f"{prefix}.Y"].add_key(frame_num, float(xyz[1]))
    channels[f"{prefix}.Z"].add_key(frame_num, float(xyz[2]))


def _look_at_rotator_deg(
    cam_pos_cm: tuple[float, float, float],
    target_pos_cm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute UE5 (Pitch, Yaw, Roll) degrees so +X camera forward looks at `target_pos_cm`."""
    dx = target_pos_cm[0] - cam_pos_cm[0]
    dy = target_pos_cm[1] - cam_pos_cm[1]
    dz = target_pos_cm[2] - cam_pos_cm[2]
    horiz = math.hypot(dx, dy)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, max(horiz, 1e-3)))
    return (pitch, yaw, 0.0)


def build_dm100_ue5_scene_and_sequence(
    replay_path: Path,
    meshes_dir: Path,
    *,
    sequence_name: str | None = None,
    downsample: int = 1,
) -> dict[str, Any]:
    """Import DM100 50 Hz 3D replay into UE 5.4 Level + LevelSequence."""
    _log(f"Loading 50 Hz 3D replay: {replay_path}")
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    all_frames: list[dict[str, Any]] = payload["frames"]
    step_n = max(1, int(downsample))
    frames = all_frames[::step_n]
    if frames[-1]["time_s"] != all_frames[-1]["time_s"]:
        frames.append(all_frames[-1])

    fps = 50
    end_frame = int(round(float(frames[-1]["time_s"]) * fps)) + 10
    seq_name = sequence_name or f"LS_{replay_path.stem}"
    _log(
        f"Replay '{replay_path.name}': {len(all_frames)} raw frames, keying {len(frames)} samples "
        f"(0..{frames[-1]['time_s']:.2f} s, end_frame={end_frame} @ {fps} fps)"
    )

    # 1. Import meshes & build procedural VFX materials
    meshes = _import_glb_meshes(meshes_dir)
    engine_cone = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cone.Cone")
    engine_plane = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Plane.Plane")
    engine_sphere = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Sphere.Sphere")

    mat_plume = _create_emissive_material(
        "M_DM100_Plume_Methalox", (0.65, 0.85, 1.0), (22.0, 38.0, 95.0), roughness=0.15
    )
    mat_vac_plume = _create_emissive_material(
        "M_DM100_Plume_Vacuum", (0.70, 0.80, 1.0), (16.0, 26.0, 75.0), roughness=0.15
    )
    mat_rcs = _create_emissive_material(
        "M_DM100_RCS_Puff", (0.88, 0.94, 1.0), (6.0, 8.5, 14.0), roughness=0.45
    )
    mat_heat = _create_emissive_material(
        "M_DM100_Reentry_Glow", (1.0, 0.35, 0.08), (35.0, 9.0, 1.5), roughness=0.25
    )
    mat_ground = _create_emissive_material(
        "M_DM100_Ground_Complex", (0.045, 0.052, 0.048), (0.0, 0.0, 0.0), roughness=0.95
    )

    # 2. Create fresh LWC Level (delete existing level first so no ghost actors remain!)
    level_sub = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(LEVEL_PATH):
        unreal.EditorAssetLibrary.delete_asset(LEVEL_PATH)
    level_sub.new_level(LEVEL_PATH)
    # Destroy any leftover actors in the newly created template world
    for old_actor in list(actor_sub.get_all_level_actors()):
        actor_sub.destroy_actor(old_actor)

    # Atmospheric & planetary lighting (sea level -> 110 km curved Earth sky transition)
    sun = actor_sub.spawn_actor_from_class(
        unreal.DirectionalLight, unreal.Vector(0.0, 0.0, 5000.0), _make_rotator(-28.0, -52.0, 0.0)
    )
    sun.set_actor_label("DM100_SunLight")
    sun_comp = sun.light_component
    sun_comp.set_intensity(6.5)
    if hasattr(sun_comp, "set_atmosphere_sun_light"):
        sun_comp.set_atmosphere_sun_light(True)

    sky_atmo = actor_sub.spawn_actor_from_class(
        unreal.SkyAtmosphere, unreal.Vector(0.0, 0.0, 0.0), _make_rotator(0.0, 0.0, 0.0)
    )
    sky_atmo.set_actor_label("DM100_SkyAtmosphere")

    sky_light = actor_sub.spawn_actor_from_class(
        unreal.SkyLight, unreal.Vector(0.0, 0.0, 3000.0), _make_rotator(0.0, 0.0, 0.0)
    )
    sky_light.set_actor_label("DM100_SkyLight")
    sky_light.light_component.set_editor_property("real_time_capture", True)
    sky_light.light_component.set_intensity(1.35)

    fog = actor_sub.spawn_actor_from_class(
        unreal.ExponentialHeightFog, unreal.Vector(0.0, 0.0, 0.0), _make_rotator(0.0, 0.0, 0.0)
    )
    fog.set_actor_label("DM100_HeightFog")
    fog.component.set_fog_density(0.0020)

    # Unbound PostProcessVolume for crisp aerospace exposure & vibrant Methalox bloom
    ppv = actor_sub.spawn_actor_from_class(
        unreal.PostProcessVolume, unreal.Vector(0.0, 0.0, 0.0), _make_rotator(0.0, 0.0, 0.0)
    )
    ppv.set_actor_label("DM100_PostProcess")
    ppv.set_editor_property("unbound", True)
    pps = ppv.get_editor_property("settings")
    pps.set_editor_property("override_auto_exposure_bias", True)
    pps.set_editor_property("auto_exposure_bias", -0.85)
    pps.set_editor_property("override_bloom_intensity", True)
    pps.set_editor_property("bloom_intensity", 0.80)
    ppv.set_editor_property("settings", pps)

    # Coastal/desert launch complex ground plane (4 km x 4 km) + 10 m concrete landing pad
    _spawn_mesh_actor(
        actor_sub,
        "DM100_Ground_Plane",
        engine_plane,
        loc_cm=(0.0, 0.0, -35.0),
        scale=(4000.0, 4000.0, 1.0),
        material=mat_ground,
        movable=False,
    )
    _spawn_assembly_actor(
        actor_sub,
        "DM100_LandingPad",
        meshes["dm100_landing_pad"],
        loc_cm=(0.0, 0.0, 0.0),
        movable=False,
    )

    # 3. Spawn DM100 1st-Stage Booster & articulated children
    booster = _spawn_assembly_actor(
        actor_sub,
        "DM100_Booster",
        meshes["dm100_booster_hull"],
        loc_cm=(0.0, 0.0, 20.0),
    )

    fin_actors = []
    for idx, (loc_cm, yaw_deg) in enumerate(FIN_MOUNTS_CM):
        fa = _spawn_assembly_actor(
            actor_sub,
            f"DM100_GridFin_{idx}",
            meshes["dm100_grid_fin"],
            loc_cm=loc_cm,
            rot_deg=(0.0, yaw_deg, 0.0),
        )
        fa.attach_to_actor(
            booster,
            "",
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            False,
        )
        fin_actors.append((fa, yaw_deg))

    leg_actors = []
    for idx, (loc_cm, yaw_deg) in enumerate(LEG_MOUNTS_CM):
        la = _spawn_assembly_actor(
            actor_sub,
            f"DM100_LandingLeg_{idx}",
            meshes["dm100_landing_leg"],
            loc_cm=loc_cm,
            rot_deg=(0.0, yaw_deg, 0.0),
        )
        la.attach_to_actor(
            booster,
            "",
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            False,
        )
        leg_actors.append((la, loc_cm, yaw_deg))

    # 7 Methalox exhaust plumes (inverted cones pointing -Z below each nozzle)
    plume_actors = []
    for idx, (nx, ny, nz) in enumerate(NOZZLE_POS_CM):
        pa = _spawn_mesh_actor(
            actor_sub,
            f"DM100_Plume_{idx}",
            engine_cone,
            loc_cm=(nx, ny, nz - 450.0),
            rot_deg=(180.0, 0.0, 0.0),
            scale=(0.001, 0.001, 0.001),
            material=mat_plume,
        )
        pa.attach_to_actor(
            booster,
            "",
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            False,
        )
        plume_actors.append(pa)

    # 4 Cold-gas RCS thruster puffs at z = 22.4 m
    rcs_actors = []
    for idx, az_deg in enumerate((45.0, 135.0, 225.0, 315.0)):
        az = math.radians(az_deg)
        ra = _spawn_mesh_actor(
            actor_sub,
            f"DM100_RCS_Puff_{idx}",
            engine_cone,
            loc_cm=(190.0 * math.cos(az), -190.0 * math.sin(az), 2240.0),
            rot_deg=(90.0, -az_deg, 0.0),
            scale=(0.001, 0.001, 0.001),
            material=mat_rcs,
        )
        ra.attach_to_actor(
            booster,
            "",
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            unreal.AttachmentRule.KEEP_RELATIVE,
            False,
        )
        rcs_actors.append(ra)

    # Aft reentry thermal plasma glow
    heat_glow = _spawn_mesh_actor(
        actor_sub,
        "DM100_Reentry_Glow",
        engine_sphere,
        loc_cm=(0.0, 0.0, -35.0),
        scale=(0.001, 0.001, 0.001),
        material=mat_heat,
    )
    heat_glow.attach_to_actor(
        booster,
        "",
        unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE,
        False,
    )

    # Stage 2 Upper Stage + vacuum engine plume (independent actor keyed from booster -> upper_pos)
    upper_stage = _spawn_assembly_actor(
        actor_sub,
        "DM100_UpperStage",
        meshes["dm100_upper_stage"],
        loc_cm=(0.0, 0.0, 2420.0),
    )
    upper_plume = _spawn_mesh_actor(
        actor_sub,
        "DM100_UpperStage_Plume",
        engine_cone,
        loc_cm=(0.0, 0.0, -480.0),
        rot_deg=(180.0, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001),
        material=mat_vac_plume,
    )
    upper_plume.attach_to_actor(
        upper_stage,
        "",
        unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE,
        False,
    )

    # 4. Spawn 4 CineCameras (LaunchPad, Onboard Fin-Cam, Chase 3D, Pad Hero)
    def _spawn_cine_cam(
        label: str,
        loc: tuple[float, float, float],
        rot_pyr: tuple[float, float, float],
        focal_mm: float,
    ) -> Any:
        cam = actor_sub.spawn_actor_from_class(
            unreal.CineCameraActor,
            unreal.Vector(*loc),
            _make_rotator(rot_pyr[0], rot_pyr[1], rot_pyr[2]),
        )
        cam.set_actor_label(label)
        cam.get_cine_camera_component().current_focal_length = float(focal_mm)
        cam.get_cine_camera_component().focus_settings.focus_method = (
            unreal.CameraFocusMethod.DISABLE
        )
        return cam

    cam_lp_pos = (6200.0, 7200.0, 850.0)
    cam_hero_pos = (4200.0, 3600.0, 1050.0)
    cam_launchpad = _spawn_cine_cam(
        "Cam_LaunchPad", cam_lp_pos, _look_at_rotator_deg(cam_lp_pos, (0.0, 0.0, 1200.0)), 22.0
    )
    cam_pad_hero = _spawn_cine_cam(
        "Cam_Pad_Hero", cam_hero_pos, _look_at_rotator_deg(cam_hero_pos, (0.0, 0.0, 1100.0)), 22.0
    )
    cam_chase = _spawn_cine_cam(
        "Cam_Chase_3D", (-5200.0, -4600.0, 2600.0), (-15.0, 40.0, 0.0), 24.0
    )
    cam_onboard = _spawn_cine_cam(
        "Cam_Onboard_Fin", (245.0, -245.0, 2380.0), (-75.0, 135.0, 0.0), 16.0
    )
    cam_onboard.attach_to_actor(
        booster,
        "",
        unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE,
        False,
    )

    # Verify imported booster world bounds in UE5 log (expected Z height ~2420 cm)
    b_orig, b_ext = booster.get_actor_bounds(False)
    _log(
        f"Booster actor bounds: origin=({b_orig.x:.1f},{b_orig.y:.1f},{b_orig.z:.1f}) cm, "
        f"half_extents=({b_ext.x:.1f},{b_ext.y:.1f},{b_ext.z:.1f}) cm"
    )

    level_sub.save_current_level()

    # 5. Create 50 fps Level Sequence & populate all animation tracks
    seq_asset_path = f"{SEQ_DEST}/{seq_name}"
    if unreal.EditorAssetLibrary.does_asset_exist(seq_asset_path):
        unreal.EditorAssetLibrary.delete_asset(seq_asset_path)
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    seq = asset_tools.create_asset(
        seq_name, SEQ_DEST, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
    )
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    seq.set_tick_resolution(unreal.FrameRate(5000, 1))
    seq.set_playback_start(0)
    seq.set_playback_end(end_frame)

    _, ch_booster = _add_transform_section(seq, booster, end_frame, use_quat=True)
    _, ch_upper = _add_transform_section(seq, upper_stage, end_frame, use_quat=True)
    _, ch_upper_plume = _add_transform_section(seq, upper_plume, end_frame, use_quat=False)
    ch_fins = [
        _add_transform_section(seq, fa, end_frame, use_quat=False, init_loc=loc_cm)[1]
        for (fa, _), (loc_cm, _) in zip(fin_actors, FIN_MOUNTS_CM, strict=True)
    ]
    ch_legs = [
        _add_transform_section(seq, la, end_frame, use_quat=False, init_loc=loc_cm)[1]
        for la, loc_cm, _ in leg_actors
    ]
    ch_plumes = [
        _add_transform_section(
            seq,
            pa,
            end_frame,
            use_quat=False,
            init_loc=(nx, ny, nz - 450.0),
            init_rot_pyr=(180.0, 0.0, 0.0),
            init_scale=(0.001, 0.001, 0.001),
        )[1]
        for pa, (nx, ny, nz) in zip(plume_actors, NOZZLE_POS_CM, strict=True)
    ]
    ch_rcs = [
        _add_transform_section(
            seq,
            ra,
            end_frame,
            use_quat=False,
            init_scale=(0.001, 0.001, 0.001),
        )[1]
        for ra in rcs_actors
    ]
    _, ch_heat = _add_transform_section(
        seq, heat_glow, end_frame, use_quat=False, init_scale=(0.001, 0.001, 0.001)
    )
    b_lp, ch_cam_lp = _add_transform_section(
        seq, cam_launchpad, end_frame, use_quat=False, init_loc=cam_lp_pos
    )
    b_hero, ch_cam_hero = _add_transform_section(
        seq, cam_pad_hero, end_frame, use_quat=False, init_loc=cam_hero_pos
    )
    b_chase, ch_cam_chase = _add_transform_section(seq, cam_chase, end_frame, use_quat=False)
    b_onboard = seq.add_possessable(cam_onboard)

    # Bind all 4 cameras for Camera Cut Track (no duplicate possessables)
    cam_bindings = {
        "Cam_LaunchPad": b_lp,
        "Cam_Onboard_Fin": b_onboard,
        "Cam_Chase_3D": b_chase,
        "Cam_Pad_Hero": b_hero,
    }

    prev_b_rot: tuple[float, float, float] | None = None
    prev_u_rot: tuple[float, float, float] | None = None
    prev_lp_rot: tuple[float, float, float] | None = None
    prev_hero_rot: tuple[float, float, float] | None = None
    prev_chase_rot: tuple[float, float, float] | None = None
    sep_time_s: float | None = None

    for f in frames:
        t_s = float(f["time_s"])
        fnum = unreal.FrameNumber(int(round(t_s * fps)))

        # Booster hull-base pose in UE5 LH cm
        base_enu = resolve_booster_base_pos_enu(f)
        b_pos_cm = enu_pos_to_ue5_cm(base_enu)
        b_q_ue5 = enu_quat_wxyz_to_ue5_xyzw(f["quat_wxyz"])
        b_rot_raw = ue5_quat_xyzw_to_rotator_deg(b_q_ue5)  # (pitch, yaw, roll)
        b_rot = unwrap_euler_deg(prev_b_rot, b_rot_raw)
        prev_b_rot = b_rot
        _key_vec3(ch_booster, "Location", fnum, b_pos_cm)
        # Sequencer Rotation channels: X=Roll, Y=Pitch, Z=Yaw
        _key_vec3(ch_booster, "Rotation", fnum, (b_rot[2], b_rot[0], b_rot[1]))

        # Stage 2 upper stage (rigidly at z=+24.0 m during ascent, then independent 25 s trajectory)
        upper_attached = bool(f.get("upper_attached", True))
        if not upper_attached and sep_time_s is None:
            sep_time_s = t_s
        up_pos_enu = f.get("upper_pos_enu_m")
        up_q_wxyz = f.get("upper_quat_wxyz")
        if upper_attached or up_pos_enu is None or up_q_wxyz is None:
            # Compute booster nose position = base + 24.0 m * z_b
            q = f["quat_wxyz"]
            w, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            zb = (
                2.0 * (qx * qz + w * qy),
                2.0 * (qy * qz - w * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            )
            nose_enu = (
                base_enu[0] + 24.0 * zb[0],
                base_enu[1] + 24.0 * zb[1],
                base_enu[2] + 24.0 * zb[2],
            )
            u_pos_cm = enu_pos_to_ue5_cm(nose_enu)
            u_rot = b_rot
            u_plume_s = (0.001, 0.001, 0.001)
        else:
            u_pos_cm = enu_pos_to_ue5_cm(up_pos_enu)
            u_rot_raw = ue5_quat_xyzw_to_rotator_deg(enu_quat_wxyz_to_ue5_xyzw(up_q_wxyz))
            u_rot = unwrap_euler_deg(prev_u_rot, u_rot_raw)
            # Vacuum engine ignites 4.0 s after separation
            dt_sep = t_s - (sep_time_s or t_s)
            u_plume_s = (2.4, 2.4, 8.5) if dt_sep >= 4.0 else (0.001, 0.001, 0.001)
        prev_u_rot = u_rot
        _key_vec3(ch_upper, "Location", fnum, u_pos_cm)
        _key_vec3(ch_upper, "Rotation", fnum, (u_rot[2], u_rot[0], u_rot[1]))
        _key_vec3(ch_upper_plume, "Scale", fnum, u_plume_s)

        # 4 Grid fins: fold angle (90 deg stowed -> 0 deg deployed) + per-fin deflection
        fin_ext = max(0.0, min(1.0, float(f.get("fin_extension", 0.0))))
        fold_pitch = 90.0 * (1.0 - fin_ext)
        fin_defs = f.get("fin_deflection_rad", [0.0, 0.0, 0.0, 0.0])
        for idx, ((_, yaw_deg), ch_f) in enumerate(zip(fin_actors, ch_fins, strict=True)):
            def_deg = math.degrees(float(fin_defs[idx]))
            # Roll = deflection around radial hinge +X, Pitch = fold stowage, Yaw = mount azimuth
            _key_vec3(ch_f, "Rotation", fnum, (def_deg, fold_pitch, yaw_deg))

        # 4 Landing legs: stow flush against hull when retracted (scale smoothly out on deploy)
        gear_ext = max(0.0, min(1.0, float(f.get("gear_extension", 0.0))))
        leg_fold_pitch = 115.0 * (1.0 - gear_ext)
        leg_s = 0.001 if gear_ext < 0.01 else (0.35 + 0.65 * gear_ext)
        strokes = f.get("leg_stroke_m", [0.0, 0.0, 0.0, 0.0])
        for idx, ((_, loc_cm, yaw_deg), ch_l) in enumerate(zip(leg_actors, ch_legs, strict=True)):
            stroke_cm = float(strokes[idx]) * 100.0
            _key_vec3(ch_l, "Location", fnum, (loc_cm[0], loc_cm[1], loc_cm[2] + stroke_cm))
            _key_vec3(ch_l, "Rotation", fnum, (0.0, leg_fold_pitch, yaw_deg))
            _key_vec3(ch_l, "Scale", fnum, (leg_s, leg_s, leg_s))

        # 7 -> 3 -> 1 Methalox exhaust plumes (keep cone base anchored inside nozzle at z=-25 cm)
        plume_scales = compute_plume_scales(
            float(f.get("throttle", 0.0)), int(f.get("active_engines", 0))
        )
        for (nx, ny, nz), ps, ch_p in zip(NOZZLE_POS_CM, plume_scales, ch_plumes, strict=True):
            sz_p = ps[2] * 7.5
            _key_vec3(ch_p, "Location", fnum, (nx, ny, nz - 50.0 * sz_p))
            _key_vec3(ch_p, "Scale", fnum, (ps[0] * 1.15, ps[1] * 1.15, sz_p))

        # 4 Cold-gas RCS puffs driven by |rcs_torque_nm|
        rcs_t = f.get("rcs_torque_nm", [0.0, 0.0, 0.0])
        rcs_mag = math.sqrt(sum(float(v) * float(v) for v in rcs_t))
        rcs_s = min(1.8, (rcs_mag / 30_000.0) * 1.6) if rcs_mag > 150.0 else 0.001
        for ch_r in ch_rcs:
            _key_vec3(ch_r, "Scale", fnum, (rcs_s * 0.45, rcs_s * 0.45, rcs_s * 2.2))

        # Reentry thermal heating glow driven by heat_flux_w_m2 (0.0 scale when cold)
        qdot = float(f.get("heat_flux_w_m2", 0.0))
        heat_s = min(3.8, ((qdot - 4_000.0) / 14_000.0) * 3.4) if qdot > 4_000.0 else 0.0
        _key_vec3(ch_heat, "Scale", fnum, (heat_s, heat_s, heat_s * 0.45))

        # Smooth camera aim tracking on booster mid-hull (+11 m above base)
        target_cm = (b_pos_cm[0], b_pos_cm[1], b_pos_cm[2] + 1100.0)

        lp_rot = unwrap_euler_deg(prev_lp_rot, _look_at_rotator_deg(cam_lp_pos, target_cm))
        prev_lp_rot = lp_rot
        _key_vec3(ch_cam_lp, "Rotation", fnum, (lp_rot[2], lp_rot[0], lp_rot[1]))

        hero_rot = unwrap_euler_deg(prev_hero_rot, _look_at_rotator_deg(cam_hero_pos, target_cm))
        prev_hero_rot = hero_rot
        _key_vec3(ch_cam_hero, "Rotation", fnum, (hero_rot[2], hero_rot[0], hero_rot[1]))

        chase_pos = (b_pos_cm[0] - 5200.0, b_pos_cm[1] - 4600.0, max(500.0, b_pos_cm[2] + 1600.0))
        chase_rot = unwrap_euler_deg(prev_chase_rot, _look_at_rotator_deg(chase_pos, target_cm))
        prev_chase_rot = chase_rot
        _key_vec3(ch_cam_chase, "Location", fnum, chase_pos)
        _key_vec3(ch_cam_chase, "Rotation", fnum, (chase_rot[2], chase_rot[0], chase_rot[1]))

    # Add Multi-Camera Cut Track for mission highlights
    windows = compute_highlight_windows(all_frames)
    cam_cut_track = (
        seq.add_track(unreal.MovieSceneCameraCutTrack)
        if hasattr(seq, "add_track")
        else seq.add_master_track(unreal.MovieSceneCameraCutTrack)
    )
    for win in windows:
        sec = cam_cut_track.add_section()
        sec.set_range(int(round(win["start_s"] * fps)), int(round(win["end_s"] * fps)))
        cam_bind = cam_bindings.get(win["camera"])
        if cam_bind is not None:
            sec.set_camera_binding_id(seq.get_binding_id(cam_bind))

    unreal.EditorAssetLibrary.save_asset(seq_asset_path)
    level_sub.save_current_level()

    manifest = {
        "level_path": LEVEL_PATH,
        "sequence_path": seq_asset_path,
        "replay_path": str(replay_path),
        "fps": fps,
        "frame_count": len(frames),
        "end_frame": end_frame,
        "highlight_windows": windows,
    }
    _log(f"SUCCESS: Built UE 5.4 Level + LevelSequence -> {seq_asset_path}")
    print("DM100_UE5_IMPORT_MANIFEST=" + json.dumps(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Import DM100 50 Hz 3D replay into UE 5.4")
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path(os.environ["DM100_REPLAY_PATH"])
        if "DM100_REPLAY_PATH" in os.environ
        else None,
        help="Path to *_50hz.json replay (or set DM100_REPLAY_PATH env var)",
    )
    parser.add_argument(
        "--meshes-dir",
        type=Path,
        default=_SCRIPT_DIR.parent / "assets" / "meshes",
        help="Directory containing generated dm100_*.glb meshes",
    )
    parser.add_argument(
        "--sequence-name",
        default=os.environ.get("DM100_SEQUENCE_NAME"),
        help="Output LevelSequence asset name",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=int(os.environ.get("DM100_DOWNSAMPLE", "1")),
        help="Frame stride (1 = full 50 Hz)",
    )
    args, _ = parser.parse_known_args()
    if args.replay is None:
        raise SystemExit("Pass --replay <path> or set DM100_REPLAY_PATH.")
    build_dm100_ue5_scene_and_sequence(
        args.replay.resolve(),
        args.meshes_dir.resolve(),
        sequence_name=args.sequence_name,
        downsample=args.downsample,
    )


if __name__ == "__main__":
    main()
