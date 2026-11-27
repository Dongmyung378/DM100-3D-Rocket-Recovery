"""Strictly causal 6DOF interfaces bridging MuJoCo Plant, 6DOF GNC, and UE 5.4 Replay.

Causality contract (inherited from DM100 2D):
  - TelemetryPacket3D carries ONLY sensor-observable / delayed / noisy measurements.
  - Hidden plant truth (true wind gusts, thrust/drag/mass/inertia scales, internal
    slosh states, true IMU biases) MUST NEVER appear in TelemetryPacket3D.
  - ReplayFrame3D preserves full dm100_replay_3d_v1 compatibility while adding
    optional hull-base geometry fields (`base_pos_enu_m`, `cg_station_m`) so
    UE 5.4 meshes never drift vertically as propellant burns or stages separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TelemetryPacket3D:
    """Strictly causal 6DOF telemetry delivered from Plant/Sensors to the GNC core."""

    time_s: float
    measurement_time_s: float
    pos_enu_m: tuple[float, float, float]
    vel_enu_m_s: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    omega_body_rad_s: tuple[float, float, float]
    accel_body_m_s2: tuple[float, float, float]
    main_propellant_kg: float
    rcs_propellant_kg: float
    gear_extension: float
    fin_deflection_rad: tuple[float, float, float, float]
    fin_extension: float
    gimbal_rad: tuple[float, float]
    engine_throttle: float
    upper_attached: bool
    on_pad: bool
    touchdown_detected: bool
    active_engines: int
    restarts_used: int
    packet_dropped: bool = False
    # Optional DM100 3D v2 causal multi-sensor telemetry channels (v1 bit-exact defaults)
    gps_valid: bool = True
    gps_pos_enu_m: tuple[float, float, float] | None = None
    gps_vel_enu_m_s: tuple[float, float, float] | None = None
    attitude_valid: bool = True
    star_tracker_valid: bool = False
    star_tracker_quat_wxyz: tuple[float, float, float, float] | None = None
    radar_alt_valid: bool = False
    radar_range_m: float = 0.0
    engine_chamber_pressure_ratio_7: (
        tuple[float, float, float, float, float, float, float] | None
    ) = None


@dataclass(frozen=True, slots=True)
class ActuatorCommand3D:
    """6DOF actuator commands issued by the GNC core to the MuJoCo Plant."""

    throttle_cmd: float = 0.0
    active_engines: int = 0
    gimbal_rad: tuple[float, float] = (0.0, 0.0)
    rcs_torque_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fin_deflection_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    deploy_fins: bool = False
    deploy_gear: bool = False
    phase: str = "ascent"
    engine_mask_7: tuple[bool, bool, bool, bool, bool, bool, bool] | None = None
    engine_throttles_7: tuple[float, float, float, float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class ForceTorqueSample3D:
    """World-frame forces [N] and world/body torques [N*m] produced by 3D external forces."""

    force_world_n: tuple[float, float, float]
    torque_body_nm: tuple[float, float, float]
    aero_force_world_n: tuple[float, float, float]
    aero_torque_body_nm: tuple[float, float, float]
    fin_force_world_n: tuple[float, float, float]
    fin_torque_body_nm: tuple[float, float, float]
    fin_hinge_nm: tuple[float, float, float, float]
    thrust_n: float
    main_flow_kg_s: float
    rcs_flow_kg_s: float
    dynamic_pressure_pa: float
    heat_flux_w_m2: float
    mach: float
    total_alpha_deg: float
    aero_roll_deg: float


@dataclass(frozen=True, slots=True)
class ContactSample3D:
    """Post-touchdown 4-leg 3D contact loads, strokes, and hull clearance state."""

    force_world_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_body_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    leg_normal_n: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    leg_stroke_m: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    leg_in_contact: tuple[bool, bool, bool, bool] = (False, False, False, False)
    any_contact: bool = False
    body_ground_collision: bool = False
    contact_load_g: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplayFrame3D:
    """50 Hz dm100_replay_3d_v1 frame (right-handed ENU meters + unit quaternion wxyz)."""

    time_s: float
    phase: str
    pos_enu_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    vel_enu_m_s: tuple[float, float, float]
    omega_body_rad_s: tuple[float, float, float]
    throttle: float
    active_engines: int
    gimbal_rad: tuple[float, float]
    rcs_torque_nm: tuple[float, float, float]
    fin_extension: float
    fin_deflection_rad: tuple[float, float, float, float]
    gear_extension: float
    leg_stroke_m: tuple[float, float, float, float]
    dynamic_pressure_pa: float
    mach: float
    alpha_deg: float
    main_propellant_kg: float
    rcs_propellant_kg: float
    upper_attached: bool
    upper_pos_enu_m: tuple[float, float, float] | None = None
    upper_quat_wxyz: tuple[float, float, float, float] | None = None
    base_pos_enu_m: tuple[float, float, float] | None = None
    cg_station_m: float | None = None
    heat_flux_w_m2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "time_s": float(self.time_s),
            "phase": self.phase,
            "pos_enu_m": [float(v) for v in self.pos_enu_m],
            "quat_wxyz": [float(v) for v in self.quat_wxyz],
            "vel_enu_m_s": [float(v) for v in self.vel_enu_m_s],
            "omega_body_rad_s": [float(v) for v in self.omega_body_rad_s],
            "throttle": float(self.throttle),
            "active_engines": int(self.active_engines),
            "gimbal_rad": [float(v) for v in self.gimbal_rad],
            "rcs_torque_nm": [float(v) for v in self.rcs_torque_nm],
            "fin_extension": float(self.fin_extension),
            "fin_deflection_rad": [float(v) for v in self.fin_deflection_rad],
            "gear_extension": float(self.gear_extension),
            "leg_stroke_m": [float(v) for v in self.leg_stroke_m],
            "dynamic_pressure_pa": float(self.dynamic_pressure_pa),
            "mach": float(self.mach),
            "alpha_deg": float(self.alpha_deg),
            "main_propellant_kg": float(self.main_propellant_kg),
            "rcs_propellant_kg": float(self.rcs_propellant_kg),
            "upper_attached": bool(self.upper_attached),
            "upper_pos_enu_m": (
                [float(v) for v in self.upper_pos_enu_m]
                if self.upper_pos_enu_m is not None
                else None
            ),
            "upper_quat_wxyz": (
                [float(v) for v in self.upper_quat_wxyz]
                if self.upper_quat_wxyz is not None
                else None
            ),
        }
        if self.base_pos_enu_m is not None:
            payload["base_pos_enu_m"] = [float(v) for v in self.base_pos_enu_m]
        if self.cg_station_m is not None:
            payload["cg_station_m"] = float(self.cg_station_m)
        if self.heat_flux_w_m2 is not None:
            payload["heat_flux_w_m2"] = float(self.heat_flux_w_m2)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReplayFrame3D:
        def _t3(val: Any) -> tuple[float, float, float]:
            return (float(val[0]), float(val[1]), float(val[2]))

        def _t4(val: Any) -> tuple[float, float, float, float]:
            return (float(val[0]), float(val[1]), float(val[2]), float(val[3]))

        def _t2(val: Any) -> tuple[float, float]:
            return (float(val[0]), float(val[1]))

        return cls(
            time_s=float(raw["time_s"]),
            phase=str(raw["phase"]),
            pos_enu_m=_t3(raw["pos_enu_m"]),
            quat_wxyz=_t4(raw["quat_wxyz"]),
            vel_enu_m_s=_t3(raw["vel_enu_m_s"]),
            omega_body_rad_s=_t3(raw["omega_body_rad_s"]),
            throttle=float(raw["throttle"]),
            active_engines=int(raw["active_engines"]),
            gimbal_rad=_t2(raw["gimbal_rad"]),
            rcs_torque_nm=_t3(raw["rcs_torque_nm"]),
            fin_extension=float(raw["fin_extension"]),
            fin_deflection_rad=_t4(raw["fin_deflection_rad"]),
            gear_extension=float(raw["gear_extension"]),
            leg_stroke_m=_t4(raw["leg_stroke_m"]),
            dynamic_pressure_pa=float(raw["dynamic_pressure_pa"]),
            mach=float(raw["mach"]),
            alpha_deg=float(raw["alpha_deg"]),
            main_propellant_kg=float(raw["main_propellant_kg"]),
            rcs_propellant_kg=float(raw["rcs_propellant_kg"]),
            upper_attached=bool(raw["upper_attached"]),
            upper_pos_enu_m=_t3(raw["upper_pos_enu_m"])
            if raw.get("upper_pos_enu_m") is not None
            else None,
            upper_quat_wxyz=_t4(raw["upper_quat_wxyz"])
            if raw.get("upper_quat_wxyz") is not None
            else None,
            base_pos_enu_m=_t3(raw["base_pos_enu_m"])
            if raw.get("base_pos_enu_m") is not None
            else None,
            cg_station_m=float(raw["cg_station_m"])
            if raw.get("cg_station_m") is not None
            else None,
            heat_flux_w_m2=float(raw["heat_flux_w_m2"])
            if raw.get("heat_flux_w_m2") is not None
            else None,
        )


@dataclass(slots=True)
class GuidanceDiagnostics3D:
    """Optional diagnostic telemetry exported by the 6DOF GNC core."""

    phase: str = "ascent"
    tilt_target_deg: float = 0.0
    roll_deg: float = 0.0
    predicted_iip_xy_m: tuple[float, float] | None = None
    predicted_unpowered_peak_q_pa: float | None = None
    hoverslam_ignition_height_m: float | None = None
    socp_solver_used: bool = False
    socp_solver_status: str = "unused"
    socp_solve_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class ControllerProtocol3D(Protocol):
    """Engine-agnostic 6DOF GNC interface."""

    def step(self, telemetry: TelemetryPacket3D) -> ActuatorCommand3D: ...

    def diagnostics(self) -> GuidanceDiagnostics3D: ...


class PlantProtocol3D(Protocol):
    """Physics backend interface implemented by MuJoCoPlant3D."""

    def telemetry(self) -> TelemetryPacket3D: ...

    def step(self, command: ActuatorCommand3D, duration_s: float) -> None: ...

    @property
    def done(self) -> bool: ...
