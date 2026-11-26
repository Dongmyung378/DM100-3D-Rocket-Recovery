# DM100 3D — 6DOF MuJoCo Rocket Recovery & Unreal Engine 5.4 Cinematics

6DOF extension of [DM100-2D-Rocket-Recovery](https://github.com/Dongmyung378/DM100-2D-Rocket-Recovery): a 100 t Methalox two-stage launch vehicle placing 1,800 kg into a 700 ± 15 km circular orbit while returning the first-stage booster via RTLS, unpowered 4-grid-fin crossrange glide, 3→1 minimum-throttle Hoverslam, and 4-leg upright touchdown stability.

## Architecture

- **Stage 2 (Physics & Causal 6DOF GNC)**:
  - `float64` RK4 MuJoCo 6DOF plant (`booster` + 4 hinged grid fins + 4 telescoping landing legs) with 50 Hz dynamic mass/COM/3-axis inertia updates (`mj_setConst`).
  - Pure 3D external forces (`xfrc_applied`): 3D body aero, independent 4-fin aero & hinge limits, 7→3→1 Methalox cluster, 2-axis conical TVC (`±6°`), 3-axis RCS, 2-DOF spherical-pendulum slosh, closed helium pressurization, and 3D spherical-Earth Stage 2 orbital insertion + 25 s post-separation trajectory.
  - Causal 6DOF GNC (`TelemetryPacket3D -> ActuatorCommand3D`): 15-State MEKF, 3×9 priority control allocation (4 fins → 2-axis TVC → 3-axis RCS), and Clarabel 3D GFOLD SOCP with analytic 3D zero-tilt fallback.
- **Headless Large-Scale Validation (Kaggle CPU Sharding)**:
  - OS-independent headless package with built-in `--shard k/N` and JSONL `--resume` checkpointing for multi-kernel free Kaggle CPU offload.
- **Stage 3 (Unreal Engine 5.4 Offline Cinematics)**:
  - Consumes frozen 50 Hz `dm100_replay_3d_v1` JSON replays (`ENU RH m + wxyz` → `UE5 LH cm` via reflection matrix $S = \operatorname{diag}(1, -1, 1)$), Python `trimesh` procedural GLB meshes, Niagara VFX, and 1080p Movie Render Queue rendering.

## Quick Verification (Phase 0)

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
```
