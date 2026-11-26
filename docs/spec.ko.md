# DM100 3D 시스템 명세서 (6DOF MuJoCo 물리 + 인과적 6DOF GNC)

본 문서는 [DM100-2D-Rocket-Recovery](https://github.com/Dongmyung378/DM100-2D-Rocket-Recovery)의 2D 평면 기준 설계를 **완전 6자유도(6DOF) 3D 강체·다물체 시뮬레이션**으로 확장한 **DM100 3D**의 기체, 물리 모델, 인과성 격리 계약, 6DOF GNC 알고리즘 및 안전 기준을 정의합니다.

---

## 1. 기준 기체 및 질량·추진 사양

모든 치수와 추진 예산은 DM100 2D 사양을 100% 계승하며, 3축 대각 관성 모멘트 $\operatorname{diag}(I_{xx}, I_{yy}, I_{zz})$와 독립 4핀·4착륙다리로 확장했습니다([config.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/config.py)).

| 구분 | 항목 | 사양 | 비고 |
| --- | --- | --- | --- |
| **총 이륙 질량** | 1단 + 2단 + 탑재체 | **100,000 kg (100 t)** | 1단 76,000 kg + 2단 24,000 kg |
| **1단 부스터** | 건조 / 메탈록스 / RCS | 6,200 / 69,500 / 300 kg | 길이 24.0 m, 직경 3.1 m (`r = 1.55 m`) |
| **2단 및 탑재체** | 건조 / 추진제 / 탑재체 | 1,550 / 20,650 / 1,800 kg | 9.0 m + 페어링, 목표 궤도 `700 ± 15 km` 원궤도 |
| **1단 주엔진** | 7기 메탈록스 클러스터 | 1,550 kN (SL) / 1,693 kN (vac) | `7 → 3 → 1` 클러스터, 최소 스로틀 36%, 재점화 ≤ 3회 |
| **2축 TVC** | 원추형 짐벌 구속 | `±6.0°` 원형 원뿔, 60 deg/s | 1차 지연 `τ = 0.06 s`, 비대칭 모서리 과추력 방지 |
| **3축 RCS** | 냉가스 N₂ 자세제어 | 피치/요 `±100 kN·m`, 롤 `±18 kN·m` | `Isp = 260 s`, `z = 22.4 m` |
| **4개 격자핀** | 독립 힌지 공력제어 | `@ 22.4 m`, 총 6.0 m² (핀당 1.5 m²) | `±20°`, 60 deg/s, 핀당 힌지 한계 `7,500 N·m` |
| **4개 착륙다리** | 텔레스코핑 완충 스트럿 | 반폭 `4.0 m`, 패드 반경 `10.0 m` | 다리당 `k = 80 kN/m`, `c = 22.5 kN·s/m` (`4×80 = 2×160`), 최대 스트로크 `0.90 m` |

---

## 2. 좌표계 및 2D–3D 등가성 계약

### 2.1 좌표계 정의 ([coords.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/coords.py))
- **월드 좌표계 (ENU, 오른손 SI)**: `+X` 동쪽(Downrange), `+Y` 북쪽(Crossrange), `+Z` 상방(Up), 단위 미터(`m`).
- **기체 좌표계 (Body, 오른손)**: `+z_b` 노즐→노즈 축, 직립 시 `+x_b = +X`, `+y_b = +Y`.
- **자세 표현**: 스칼라 우선 단위 쿼터니언 `quat_wxyz = (w, x, y, z)` (`v_world = R(q) @ v_body`).
- **UE 5.4 변환 (왼손, `cm`)**: 반사 행렬 $S = \operatorname{diag}(1, -1, 1)$ 기반 통일 변환:
  - 위치: $\mathbf{p}_{\text{UE5}} = 100 \cdot S\mathbf{p}_{\text{ENU}} = (100x, -100y, 100z)$
  - 회전 행렬: $R_{\text{UE5}} = S R_{\text{ENU}} S^{-1}$
  - 쿼터니언: `q_ue5_xyzw = (-qx, +qy, -qz, +qw)` (`FQuat::RotateVector` 대수적 동일성 단위 테스트 검증 완료).

### 2.2 2D–3D 평면 등가성 게이트 (Plane Equivalence Gate)
횡풍 `wy = 0`, 롤 `0`, 요 `0`, 대칭 피치 핀 명령 조건에서 3D 물리 모듈(`forces_3d.py`, `mass_properties_3d`)의 `(Fx, Fz, Tau_y, q, Mach, α, 4-leg vertical stiffness)`가 DM100 2D 기준 모델과 일치하며 모든 면외 성분(`Fy, Tau_x, Tau_z`)이 `0`임을 자동화 단위 테스트([test_plane_equivalence_and_physics.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/tests/test_plane_equivalence_and_physics.py))로 보증합니다.

---

## 3. MuJoCo 6DOF 물리 플랜트 (`Stage 2`)

- **파라메트릭 MJCF + `float64` RK4 적분기** ([builder.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/mjcf/builder.py), [plant_mujoco.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/plant_mujoco.py)):
  - 루트 바디 원점을 **기하학적 기체 바닥(`station = 0.0 m`)**에 고정하여 연료 소모에 따른 `body_ipos = [0, 0, cg_m]` 이동 시 접촉 지오메트리나 시각 메시가 드리프트하지 않습니다.
  - 50 Hz 제어 스텝마다 `mj_setConst`로 실시간 질량, COM, 3축 대각 관성을 갱신하며, 4개 착륙다리 발판 질량(각 50 kg)과 평행축 관성을 정확히 차감해 합성 관성이 `mass_properties_3d()`와 기계 정밀도로 일치합니다.
  - 고도별 접촉 구역(`h <= 30 m`) 진입 시 적분 스텝을 `0.01 s → 0.002 s`로 자동 전환해 강성 착륙 접촉(`k = 80 kN/m`, `c = 22.5 kN·s/m`, 마찰계수 `μ = 0.75`)과 5초 무추력 직립 안정성을 수치 발산 없이 해석합니다.
- **순수 3D 외력 (`xfrc_applied`)**:
  - 3D 동체 공력(천음속 항력 상승, 세장체 양력, 3축 감쇠) + 독립 4-격자핀 공력 및 힌지 모멘트([forces_3d.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/physics/forces_3d.py)).
  - 3D 코리올리 및 구형 지구 원심력(적도~남극 위도 지원).
  - **2-DOF 주기적 구형 진자 슬로시** ([slosh_3d.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/physics/slosh_3d.py)): 심플렉틱 중점 적분 및 $2\pi$-주기적 구형 포텐셜로 무중력 코스트 구간 적분 발산(windup)을 원천 차단합니다.
  - 폐쇄형 헬륨 가압 피드 및 재점화 준비 상태([feed_3d.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/physics/feed_3d.py)).
  - **3D 단 분리 및 2단 궤도 투입** ([orbit_3d.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/physics/orbit_3d.py)): 운동량 보존 콜드 분리 후 구형 지구 3D 중력·지구 자전 부스트를 포함한 `700 ± 15 km` 원궤도 투입 판정과 함께 분리 후 **25초간 50 Hz(1,251 프레임)** 2단 점화·이탈 궤적을 리플레이에 기록합니다.

---

## 4. 인과적 6DOF GNC 아키텍처

플랜트(`PlantProtocol3D`)와 제어기(`ControllerProtocol3D`)는 `TelemetryPacket3D → ActuatorCommand3D`로 엄격히 격리되어 있으며([interfaces.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/interfaces.py)), 제어기는 숨겨진 밀도/항력/추력/질량/관성 분산과 실제 돌풍을 절대 참조하지 않습니다.

1. **15-State Multiplicative EKF (`CausalEstimator3D`)** ([mekf.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/gnc/mekf.py)):
   - 상태 벡터: 위치(3), 속도(3), 자세 오차(3), 자이로 바이어스(3), 가속도 외란 바이어스(3) — 패킷 누락(최대 8%), 관측/명령 지연, 센서 잡음 하에서 공분산 SPD 유지.
   - 공력 구간(`entry_burn`, `aero_glide`, `landing_burn`)에서 중점 잔차 기반 3축 외란 토크 바이어스 관측기를 인과적으로 구동해 예보 오차 및 돌풍 모멘트를 상쇄합니다.
2. **3×9 우선순위 제어 할당 (`PriorityControlAllocator3D`)** ([allocation_3d.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/gnc/allocation_3d.py)):
   - 우선순위: **4개 독립 격자핀(차동 롤 최우선 → 집단 피치/요, 운전점 유한차분 선형화 및 7,500 N·m 힌지 제한)** → **2축 원추형 TVC(`±6°` 원형 제한)** → **3축 RCS(관성 비례 연속 데드밴드 및 3D 유클리드 각가속도 안전구 구속)**.
3. **Clarabel 3D GFOLD SOCP & 8단계 RTLS 유도 (`RtlsController3D`)** ([socp_gfold.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/gnc/socp_gfold.py), [guidance_3d.py](file:///d:/Projects/DM100-3D-Rocket-Recovery/src/dm100_3d/gnc/guidance_3d.py)):
   - `ascent` → `flip` → `boostback` (3D 2축 IIP 예측-수정) → `coast` (58 km 핀 전개) → `entry_burn` (동압·열유속 가드) → `aero_glide` (**0% 주추력** 4핀 3D 활공, 온라인 `drag_scale` 추정 및 역풍 리드) → `landing_burn` (`3 → 1` 호버슬램 + 3D TVC 공력 짐벌 트림 분리 + Clarabel 3D GFOLD SOCP / 3D 제로-틸트 폴백 + 말기 수직 자세 패드 센터링) → `touchdown_settle` (즉시 MECO + 5초 4다리 직립 검증).

---

## 5. 19개 합격 기준 (Flight Safety & Mission Criteria)

| 영역 | 기준 항목 | 한계치 |
| --- | --- | --- |
| **구조·열 하중** | 동압 / 정체점 열유속 / 상승 `q·α` / 고동압 `α` | `≤ 42 kPa` / `≤ 80 kW/m²` / `≤ 150 kPa·deg` / `≤ 5.0°` |
| **가속도·자세** | 상승 축 하중 / 회수 하중 / 각속도 / 각가속도 | `≤ 4.5 g` / `≤ 5.5 g` / `≤ 15.0 deg/s` / `≤ 5.0 deg/s²` |
| **구동기·추진** | TVC 짐벌 / 핀 힌지 / 피드 잔량 / 슬로시 영역 | `≤ 6.0°` / `≤ 7,500 N·m` / 고갈 없음 / 영역 내 |
| **임무·궤도** | 1-2단 분리 게이트 / 2단 궤도 투입 | `60 ± 16 km`, `≥ 1,600 m/s` / `700 ± 15 km` 원궤도 |
| **착륙·안정성** | 위치 오차 / 수직·수평 속도 / 기울기 / 접촉 각속도 | `≤ 2.0 m` / `≤ 1.5 m/s`, `≤ 0.5 m/s` / `≤ 3.0°` / `≤ 0.5 deg/s` |
| **4다리 접촉** | 스트로크 / 다리 하중 / 접촉 하중 / 5초 엔진정지 직립 | `≤ 0.90 m` / `≤ 250 kN` / `≤ 4.0 g` / 4다리 접지 & `≤ 5°` |
