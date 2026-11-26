# DM100 3D Unreal Engine 5.4 오프라인 시네마틱 파이프라인 (`Stage 3`)

본 문서는 동결된 50 Hz `dm100_replay_3d_v1` JSON 리플레이를 **Unreal Engine 5.4**로 자동 임포트하고, **RTX 3060 Laptop (6 GB VRAM)** 환경에 최적화된 **1080p 오프라인 Movie Render Queue (MRQ)** 영상으로 렌더링하는 전체 파이프라인을 설명합니다.

---

## 1. 하드웨어 예산 및 렌더링 전략 (RTX 3060 6 GB VRAM)

실시간 4K Lumen/가상 섀도우 맵(VSM) 대신 고고도 LWC(0 ~ 110 km) 행성 스케일에 안정적이고 VRAM 한계 내에서 고품질을 보장하는 오프라인 설정을 적용했습니다([DefaultEngine.ini](file:///d:/Projects/DM100-3D-Rocket-Recovery/ue5/DM100Cinematics/Config/DefaultEngine.ini)).

- **Large World Coordinates (LWC)**: `bEnableLargeWorlds=True` — 해수면(`z = 0`)부터 110 km 원지점(`11,000,000 cm`)까지 `double` 정밀도 유지.
- **VRAM 보호**: `r.Streaming.PoolSize=2048`, `r.DynamicGlobalIlluminationMethod=0`, `r.Shadow.Virtual.Enable=0` (클래식 CSM 4-캐스케이드), `r.GenerateMeshDistanceFields=False`.
- **행성 대기 조명**: `SkyAtmosphere` + `DirectionalLight` (Atmosphere Sun Light) + 실시간 `SkyLight` — 해수면 푸른 하늘에서 고도 60~110 km 검은 우주 및 지구 곡면 지평선으로 자연 전환.

---

## 2. 실치수 프로시저럴 GLB 로켓 메시 (5종)

[generate_rocket_glb.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/ue5/scripts/generate_rocket_glb.py)가 `DM100VehicleConfig`에서 치수를 직접 읽어 5종의 PBR glTF 2.0(`.glb`) 에셋을 생성하며, [test_ue5_assets.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/tests/test_ue5_assets.py)가 바운딩 박스와 피벗 정밀도를 검증합니다.

| 에셋 파일 | 실치수 & 구성 | 피벗 기준 |
| --- | --- | --- |
| `dm100_booster_hull.glb` | 24.0 m × 3.1 m, 7기 메탈록스 노즐(`z = -0.2 m`), 카본 인터스테이지, 4개 RCS 포드 | 기체 바닥 중심 `(0, 0, 0)` |
| `dm100_upper_stage.glb` | 9.0 m 2단 탱크 + 오자이브 페어링 + 진공 노즐 | 1-2단 분리면 `(0, 0, 0)` |
| `dm100_grid_fin.glb` | `1.20 m × 1.25 m` (`1.50 m²`) 티타늄 4×4 와플 격자핀 | 동체 힌지 루트 `(0, 0, 0)` |
| `dm100_landing_leg.glb` | 카본 A-프레임 + 유압 스트럿 + 발판 (`r: 1.55 → 4.0 m`, `z: +1.2 → -1.0 m`) | 동체 앵커 힌지 `(0, 0, 0)` |
| `dm100_landing_pad.glb` | 반경 `10.0 m` 콘크리트 착륙 패드 + `2.0 m` 목표 불스아이 + 방폭 에이프런 | 패드 중심 `(0, 0, 0)` |

```bash
python ue5/scripts/generate_rocket_glb.py --out ue5/assets/meshes
```

---

## 3. ENU RH(m) → UE 5.4 LH(cm) 좌표 변환 검증

[ue5_transform_math.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/ue5/scripts/ue5_transform_math.py)는 UE5 외부에서도 `pytest`로 단위 검증되는 순수 파이썬 수학 모듈입니다([test_ue5_transform_math.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/tests/test_ue5_transform_math.py)).

- 반사 행렬 $S = \operatorname{diag}(1, -1, 1)$ 적용: `pos_cm = (100*x, -100*y, 100*z)`, `q_ue5_xyzw = (-qx, +qy, -qz, +qw)`.
- **무드리프트 기체 바닥 추적**: 리플레이의 `base_pos_enu_m`을 직접 사용해 연료 소모 중 COM 이동에 의한 시각적 드리프트 제거.
- **짐벌락 없는 180° 플립**: Sequencer 트랜스폼 섹션에 `use_quaternion_interpolation = True` 활성화 + 연속 오일러 언래핑(`unwrap_euler_deg`) 동시 적용.

---

## 4. 50 Hz 리플레이 자동 임포트 & 1080p MRQ 렌더링

### Step 1 — 50 Hz 리플레이 → LWC 레벨 & Level Sequence 자동 생성
[import_dm100_replay.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/ue5/scripts/import_dm100_replay.py)는 50개 멀티파트 메시 임포트, 절차적 VFX 머티리얼(`7 → 3 → 1` 메탈록스 플룸, RCS 퍼프, 재진입 열광), 4개 `CineCameraActor`(발사대/온보드 핀캠/3D 추적/착륙패드 클로즈업), 5구간 카메라 컷 트랙을 자동으로 구성합니다.

```powershell
$env:DM100_REPLAY_PATH = "d:\Projects\DM100-3D-Rocket-Recovery\artifacts\replays\edge3d-01-calm-nominal_50hz.json"
$env:DM100_SEQUENCE_NAME = "LS_DM100_Calm_3D"
& "D:\Epic Games\UE_5.4\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
    "d:\Projects\DM100-3D-Rocket-Recovery\ue5\DM100Cinematics\DM100Cinematics.uproject" `
    -ExecutePythonScript="d:\Projects\DM100-3D-Rocket-Recovery\ue5\scripts\import_dm100_replay.py" `
    -unattended -nop4 -nosplash -NullRHI
```

### Step 2 — 오프라인 1080p Movie Render Queue 렌더링 & ffmpeg MP4 합성
커스텀 런타임 익스큐터([dm100_mrq_executor.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/ue5/DM100Cinematics/Content/Python/dm100_mrq_executor.py))로 헤드리스 PNG 시퀀스를 렌더링한 뒤, [render_mrq.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/ue5/scripts/render_mrq.py)의 `--mux-only` 모드로 `ffmpeg` H.264 MP4를 생성합니다.

```powershell
& "D:\Epic Games\UE_5.4\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
    "d:\Projects\DM100-3D-Rocket-Recovery\ue5\DM100Cinematics\DM100Cinematics.uproject" `
    /Game/DM100/Maps/DM100_RTLS_Level -game `
    -MoviePipelineLocalExecutorClass=/Script/MovieRenderPipelineCore.MoviePipelinePythonHostExecutor `
    -ExecutorPythonClass=/Engine/PythonTypes.DM100RuntimeMRQExecutor `
    -LevelSequence=/Game/DM100/Sequences/LS_DM100_Calm_3D `
    -OutDir="d:\Projects\DM100-3D-Rocket-Recovery\ue5\renders\calm" `
    -StartFrame=0 -EndFrame=350 -ResX=1920 -ResY=1080 -FPS=30 -Samples=4 `
    -windowed -resx=1280 -resy=720 -unattended -nop4 -nosplash

python ue5/scripts/render_mrq.py --mux-only --out-dir ue5/renders/calm --fps 30
```
