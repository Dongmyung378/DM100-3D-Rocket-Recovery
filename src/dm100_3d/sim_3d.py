"""Closed-loop 6DOF MuJoCo Plant (`MuJoCoPlant3D`) and RTLS Flight runner (`RtlsFlight3D`).

Couples:
  - MuJoCoRocketBody (float64 RK4, 50 Hz mj_setConst dynamic 3-axis inertia, 4-leg contact)
  - Pure 3D external forces (`forces_3d.py`) + 3D Coriolis + 2-DOF spherical slosh (`slosh_3d.py`)
  - Closed helium feed & restart readiness (`feed_3d.py`)
  - Momentum-conserving 3D cold separation + 700 km Stage 2 orbit & 25 s departure (`orbit_3d.py`)
  - Causal sensor noise, bias, delay queues, and deterministic packet dropouts
  - 50 Hz `dm100_replay_3d_v1` recording (with `base_pos_enu_m`, `cg_station_m`, and Stage 2)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import degrees, exp, hypot, isfinite, radians

import numpy as np

from dm100_3d.config import (
    G0,
    R_AIR,
    DM100VehicleConfig,
    GncV2Config,
    PhysicsV2Config,
    mass_properties_3d,
)
from dm100_3d.coords import (
    axis_angle_to_quat_wxyz,
    base_to_com_pos_enu,
    normalize_quat_wxyz,
    quat_multiply_wxyz,
    quat_to_rotmat,
    roll_angle_rad,
    tilt_angle_rad,
)
from dm100_3d.gnc.guidance_3d import Guidance3DSettings, RtlsController3D
from dm100_3d.gnc.mekf import SensorConfig3D
from dm100_3d.interfaces import (
    ActuatorCommand3D,
    ContactSample3D,
    ControllerProtocol3D,
    ReplayFrame3D,
    TelemetryPacket3D,
)
from dm100_3d.mjcf.builder import LEG_UNIT_DIRS
from dm100_3d.physics import (
    Atmosphere3DSettings,
    AtmosphereLUT,
    DualTankSloshSettings,
    DualTankSloshState,
    FeedSettings,
    FeedState,
    GroundEffect3DSettings,
    IgnitionSettings,
    Separation3DSettings,
    Slosh3DSettings,
    Slosh3DState,
    Srp3DSettings,
    SrpBuffetFilter,
    UpperStageOrbitResult,
    Wind3DConfig,
    Wind3DLUT,
    compute_coriolis_and_centrifugal_3d,
    compute_engine_thrust_3d,
    compute_forces_and_torques_3d,
    compute_ground_effect_3d,
    compute_srp_effects,
    dual_slosh_3d_response,
    dual_slosh_body_offset_3d,
    evaluate_upper_stage_orbit_3d,
    feed_rates,
    ignition_ready,
    initial_feed_state,
    separation_event_3d,
    slosh_3d_response,
    slosh_body_offset_3d,
    wgs84_coriolis_and_transport_enu,
    wgs84_normal_gravity,
)
from dm100_3d.plant_mujoco import MuJoCoRocketBody, fast_gravity


@dataclass(frozen=True, slots=True)
class Disturbances3D:
    """Hidden healthy-model plant dispersions & optional 3D roll pulse disturbance."""

    thrust_scale: float = 1.0
    drag_scale: float = 1.0
    initial_mass_scale: float = 1.0
    inertia_scale: float = 1.0
    engine_lag_s: float = 0.0
    roll_disturbance_torque_nm: float = 0.0
    roll_disturbance_start_s: float = 260.0
    roll_disturbance_duration_s: float = 6.0


@dataclass(frozen=True, slots=True)
class SimConfig3D:
    integration_dt_s: float = 0.01
    control_dt_s: float = 0.02
    contact_dt_s: float = 0.002
    deadline_s: float = 680.0
    replay_hz: float = 50.0
    record_replay: bool = True
    seed: int = 378
    sensor_seed: int = 1378
    physics_v2: PhysicsV2Config = field(default_factory=PhysicsV2Config)
    gnc_v2: GncV2Config = field(default_factory=GncV2Config)


@dataclass(frozen=True, slots=True)
class Scenario3D:
    """Self-contained specification for one 6DOF DM100 3D mission."""

    name: str = "calm_3d"
    atmosphere: Atmosphere3DSettings = field(default_factory=Atmosphere3DSettings)
    wind: Wind3DConfig = field(default_factory=Wind3DConfig)
    disturbances: Disturbances3D = field(default_factory=Disturbances3D)
    sensors: SensorConfig3D = field(default_factory=SensorConfig3D)
    seed: int = 378


class MuJoCoPlant3D:
    """Full 6DOF MuJoCo rocket plant implementing `PlantProtocol3D`."""

    def __init__(
        self,
        *,
        vehicle: DM100VehicleConfig | None = None,
        scenario: Scenario3D | None = None,
        sim_config: SimConfig3D | None = None,
    ) -> None:
        self.vehicle = vehicle or DM100VehicleConfig()
        self.scenario = scenario or Scenario3D()
        self.config = sim_config or SimConfig3D(
            seed=self.scenario.seed, sensor_seed=self.scenario.seed + 1000
        )

        self.physics_v2 = self.config.physics_v2
        pv2 = self.physics_v2

        self.atmo_lut = AtmosphereLUT(self.scenario.atmosphere, step_m=100.0)
        self.wind_lut = Wind3DLUT(self.scenario.wind, forecast=False)
        self.feed_settings = (
            FeedSettings.dual_tank_v2(pv2.dual_tank_geometry)
            if pv2.dual_tank_slosh
            else FeedSettings()
        )
        self.ignition_settings = IgnitionSettings()
        self.slosh_settings = Slosh3DSettings()
        self.dual_slosh_settings = DualTankSloshSettings(
            geometry=pv2.dual_tank_geometry,
            damping_scale=pv2.slosh_baffle_damping_scale,
        )
        self.srp_settings = Srp3DSettings(
            shield_scale=pv2.srp_shield_scale,
            buffet_enabled=pv2.srp_buffet_enabled,
            cm_rms_peak=pv2.srp_cm_rms_peak,
        )
        self.srp_buffet_filter = (
            SrpBuffetFilter(seed=self.config.seed) if pv2.srp_interference else None
        )
        self.ground_effect_settings = GroundEffect3DSettings(
            c_ge=pv2.ground_effect_coeff,
            c_lat=pv2.ground_effect_lat_coeff,
        )
        self.separation_settings = Separation3DSettings()

        dist = self.scenario.disturbances
        self.airframe_mass_offset_kg = (dist.initial_mass_scale - 1.0) * self.vehicle.launch_mass_kg
        if pv2.booster_dry_mass_scale != 1.0:
            self.airframe_mass_offset_kg += (
                pv2.booster_dry_mass_scale - 1.0
            ) * self.vehicle.booster_dry_mass_kg
        self.actual_dry_mass_kg = self.vehicle.booster_dry_mass_kg + self.airframe_mass_offset_kg
        self.engine_tau_s = max(self.vehicle.propulsion.nominal_engine_tau_s, dist.engine_lag_s)

        # Dynamic plant state
        self.main_prop_kg: float = self.vehicle.propellant.main_propellant_kg
        self.rcs_prop_kg: float = self.vehicle.propellant.rcs_propellant_kg
        self.feed_state: FeedState = initial_feed_state(self.main_prop_kg, self.feed_settings)
        self.slosh_state: Slosh3DState = Slosh3DState()
        self.dual_slosh_state: DualTankSloshState = DualTankSloshState()
        self.peak_srp_drag_reduction: float = 0.0
        self.peak_ground_effect_n: float = 0.0
        self.peak_dual_slosh_force_n: float = 0.0

        self.gear_extension: float = 0.0
        self.fin_extension: float = 0.0
        self.fin_deflection: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.gimbal: list[float] = [0.0, 0.0]
        self.engine_throttle: float = 0.0
        self.active_engines: int = 0
        self.restarts_used: int = 0
        self._engine_lit_before: bool = False
        self._ignition_requested: bool = False
        self._ignition_ready_s: float = float("inf")

        self.upper_attached: bool = True
        self.on_pad: bool = True
        self.time_s: float = 0.0
        self.phase: str = "ascent"
        self.outcome: str = "running"

        # Initialize MuJoCo body on launchpad (nozzle tip at z = 0)
        self.body = MuJoCoRocketBody(
            self.vehicle,
            timestep_s=self.config.integration_dt_s,
            gear_deployed=False,
            upper_attached=True,
        )
        mp0 = self.body.update_mass_properties(
            self._total_mass(),
            0.0,
            upper_attached=True,
            airframe_mass_offset_kg=self.airframe_mass_offset_kg,
            inertia_scale=dist.inertia_scale,
            dual_tank=pv2.dual_tank_slosh,
            rcs_propellant_kg=self.rcs_prop_kg if pv2.dual_tank_slosh else None,
            dual_tank_geom=pv2.dual_tank_geometry if pv2.dual_tank_slosh else None,
            struts_active=False,
        )
        p0 = base_to_com_pos_enu(
            (0.0, 0.0, -self.vehicle.geometry.nozzle_station_m),
            (1.0, 0.0, 0.0, 0.0),
            mp0.cg_station_m,
        )
        self.body.set_pose_from_com(p0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        # Sensor RNG & delay queues
        self.sensor_rng = np.random.default_rng(self.config.sensor_seed)
        sc = self.scenario.sensors
        self.obs_queue: deque[TelemetryPacket3D] = deque(maxlen=sc.observation_delay_steps + 1)
        self.action_queue: deque[ActuatorCommand3D] = deque(
            [ActuatorCommand3D()] * (sc.action_delay_steps + 1),
            maxlen=sc.action_delay_steps + 1,
        )
        self._last_packet: TelemetryPacket3D | None = None
        self._last_accel_body: np.ndarray = np.array([0.0, 0.0, G0], dtype=np.float64)
        self._last_contact = ContactSample3D()

        # Extrema & mission audit records
        self.max_q_pa: float = 0.0
        self.max_heat_flux_w_m2: float = 0.0
        self.max_q_alpha_kpa_deg: float = 0.0
        self.max_ascent_high_q_alpha_deg: float = 0.0
        self.max_ascent_axial_g: float = 0.0
        self.max_recovery_g: float = 0.0
        self.max_rate_deg_s: float = 0.0
        self.max_accel_deg_s2: float = 0.0
        self.max_roll_deg: float = 0.0
        self.max_gimbal_deg: float = 0.0
        self.max_fin_hinge_nm: float = 0.0
        self.max_leg_stroke_m: float = 0.0
        self.max_leg_force_n: float = 0.0
        self.max_contact_load_g: float = 0.0
        self.body_ground_collision: bool = False
        self.post_contact_max_tilt_deg: float = 0.0
        self.standing_stable: bool | None = None
        self.observed_max_q_passed: bool = False
        self._peak_ascent_q: float = 0.0
        self.feed_inventory_ok: bool = True
        self.slosh_domain_ok: bool = True
        self._prev_omega_b: np.ndarray = np.zeros(3, dtype=np.float64)

        self.separation_time_s: float | None = None
        self.separation_record: dict | None = None
        self.upper_orbit_result: UpperStageOrbitResult | None = None
        self.touchdown_time_s: float | None = None
        self.first_contact_record: dict | None = None
        self.phase_stats: dict[str, dict] = {}
        self.replay_frames: list[ReplayFrame3D] = []
        self._next_replay_s: float = 0.0
        self._replay_dt_s: float = 1.0 / max(self.config.replay_hz, 1.0)
        self._last_mass_update_s: float = -1.0
        self._last_struts_on: bool = False

    def _total_mass(self) -> float:
        return (
            self.actual_dry_mass_kg
            + max(self.main_prop_kg, 0.0)
            + max(self.rcs_prop_kg, 0.0)
            + (self.vehicle.upper_stage.total_mass_kg if self.upper_attached else 0.0)
        )

    @property
    def done(self) -> bool:
        return self.outcome != "running"

    def telemetry(self) -> TelemetryPacket3D:
        """Build strictly causal noisy/delayed TelemetryPacket3D (zero hidden truth)."""
        sc = self.scenario.sensors
        p_com, v_com, q, om_b = self.body.get_com_state()

        dropped = (
            self._last_packet is not None
            and sc.dropout_rate > 0.0
            and float(self.sensor_rng.random()) < sc.dropout_rate
        )
        if dropped and self._last_packet is not None:
            pkt = TelemetryPacket3D(
                time_s=self.time_s,
                measurement_time_s=self._last_packet.measurement_time_s,
                pos_enu_m=self._last_packet.pos_enu_m,
                vel_enu_m_s=self._last_packet.vel_enu_m_s,
                quat_wxyz=self._last_packet.quat_wxyz,
                omega_body_rad_s=self._last_packet.omega_body_rad_s,
                accel_body_m_s2=self._last_packet.accel_body_m_s2,
                main_propellant_kg=self.main_prop_kg,
                rcs_propellant_kg=self.rcs_prop_kg,
                gear_extension=self.gear_extension,
                fin_deflection_rad=(
                    self.fin_deflection[0],
                    self.fin_deflection[1],
                    self.fin_deflection[2],
                    self.fin_deflection[3],
                ),
                fin_extension=self.fin_extension,
                gimbal_rad=(self.gimbal[0], self.gimbal[1]),
                engine_throttle=self.engine_throttle,
                upper_attached=self.upper_attached,
                on_pad=self.on_pad,
                touchdown_detected=self.touchdown_time_s is not None,
                active_engines=self.active_engines,
                restarts_used=self.restarts_used,
                packet_dropped=True,
            )
        else:
            p_m = p_com + self.sensor_rng.normal(0.0, sc.position_noise_m, size=3)
            v_m = v_com + self.sensor_rng.normal(0.0, sc.velocity_noise_m_s, size=3)
            if sc.attitude_noise_deg > 0.0:
                ang_n = self.sensor_rng.normal(0.0, radians(sc.attitude_noise_deg), size=3)
                dq = axis_angle_to_quat_wxyz(ang_n, float(np.linalg.norm(ang_n)))
                q_m = normalize_quat_wxyz(quat_multiply_wxyz(q, dq), canonical=True)
            else:
                q_m = q
            gb = np.array([radians(v) for v in sc.gyro_bias_deg_s], dtype=np.float64)
            om_m = om_b + gb + self.sensor_rng.normal(0.0, radians(sc.gyro_noise_deg_s), size=3)
            ab = np.asarray(sc.accel_bias_m_s2, dtype=np.float64)
            acc_m = (
                self._last_accel_body
                + ab
                + self.sensor_rng.normal(0.0, sc.accel_noise_m_s2, size=3)
            )

            pkt = TelemetryPacket3D(
                time_s=self.time_s,
                measurement_time_s=self.time_s,
                pos_enu_m=(float(p_m[0]), float(p_m[1]), float(p_m[2])),
                vel_enu_m_s=(float(v_m[0]), float(v_m[1]), float(v_m[2])),
                quat_wxyz=(float(q_m[0]), float(q_m[1]), float(q_m[2]), float(q_m[3])),
                omega_body_rad_s=(float(om_m[0]), float(om_m[1]), float(om_m[2])),
                accel_body_m_s2=(float(acc_m[0]), float(acc_m[1]), float(acc_m[2])),
                main_propellant_kg=self.main_prop_kg,
                rcs_propellant_kg=self.rcs_prop_kg,
                gear_extension=self.gear_extension,
                fin_deflection_rad=(
                    self.fin_deflection[0],
                    self.fin_deflection[1],
                    self.fin_deflection[2],
                    self.fin_deflection[3],
                ),
                fin_extension=self.fin_extension,
                gimbal_rad=(self.gimbal[0], self.gimbal[1]),
                engine_throttle=self.engine_throttle,
                upper_attached=self.upper_attached,
                on_pad=self.on_pad,
                touchdown_detected=self.touchdown_time_s is not None,
                active_engines=self.active_engines,
                restarts_used=self.restarts_used,
                packet_dropped=False,
            )
            self._last_packet = pkt

        self.obs_queue.append(pkt)
        delayed = self.obs_queue[0]
        return TelemetryPacket3D(
            time_s=self.time_s,
            measurement_time_s=delayed.measurement_time_s,
            pos_enu_m=delayed.pos_enu_m,
            vel_enu_m_s=delayed.vel_enu_m_s,
            quat_wxyz=delayed.quat_wxyz,
            omega_body_rad_s=delayed.omega_body_rad_s,
            accel_body_m_s2=delayed.accel_body_m_s2,
            main_propellant_kg=delayed.main_propellant_kg,
            rcs_propellant_kg=delayed.rcs_propellant_kg,
            gear_extension=delayed.gear_extension,
            fin_deflection_rad=delayed.fin_deflection_rad,
            fin_extension=delayed.fin_extension,
            gimbal_rad=delayed.gimbal_rad,
            engine_throttle=delayed.engine_throttle,
            upper_attached=self.upper_attached,
            on_pad=self.on_pad,
            touchdown_detected=self.touchdown_time_s is not None,
            active_engines=self.active_engines,
            restarts_used=self.restarts_used,
            packet_dropped=delayed.packet_dropped,
        )

    def _upper_stage_pose_at(
        self, time_s: float
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float, float] | None]:
        if self.upper_attached or self.separation_time_s is None or self.upper_orbit_result is None:
            return None, None
        dt_sep = time_s - self.separation_time_s
        samples = self.upper_orbit_result.replay_samples
        if dt_sep < 0.0 or not samples or dt_sep > samples[-1].time_since_sep_s + 1e-6:
            return None, None
        idx = min(len(samples) - 1, int(round(dt_sep * self.config.replay_hz)))
        return samples[idx].pos_enu_m, samples[idx].quat_wxyz

    def _check_separation(self, q_pa: float, alpha_deg: float) -> None:
        if not self.upper_attached:
            return
        if q_pa > self._peak_ascent_q:
            self._peak_ascent_q = q_pa
        elif self._peak_ascent_q > 10_000.0 and q_pa < 0.85 * self._peak_ascent_q:
            self.observed_max_q_passed = True

        ag = self.vehicle.ascent_gate
        p_com, v_com, q_wxyz, om_b = self.body.get_com_state()
        spd = float(np.linalg.norm(v_com))
        checks = {
            "altitude": abs(float(p_com[2]) - ag.separation_altitude_m)
            <= ag.separation_tolerance_m,
            "speed": spd >= ag.separation_min_speed_m_s,
            "dynamic_pressure": q_pa <= ag.separation_max_q_pa,
            "angular_rate": degrees(float(np.linalg.norm(om_b))) <= ag.separation_max_rate_deg_s,
            "alpha": abs(alpha_deg) <= ag.separation_max_alpha_deg,
            "main_reserve": self.main_prop_kg >= ag.separation_min_main_propellant_kg,
            "main_engine_off": self.engine_throttle <= 0.01,
            "after_observed_max_q": self.observed_max_q_passed,
        }
        if not all(checks.values()):
            if float(p_com[2]) > ag.separation_altitude_m + ag.separation_tolerance_m:
                self.outcome = "separation_gate_missed"
            return

        # Execute 6DOF momentum-conserving cold stage separation
        dist = self.scenario.disturbances
        pv2 = self.physics_v2
        tot_m = self._total_mass()
        u_m = self.vehicle.upper_stage.total_mass_kg
        b_m = tot_m - u_m
        mp_b = mass_properties_3d(
            b_m,
            self.gear_extension,
            vehicle=self.vehicle,
            upper_attached=False,
            airframe_mass_offset_kg=self.airframe_mass_offset_kg,
            inertia_scale=dist.inertia_scale,
            dual_tank=pv2.dual_tank_slosh,
            rcs_propellant_kg=self.rcs_prop_kg if pv2.dual_tank_slosh else None,
            dual_tank_geom=pv2.dual_tank_geometry if pv2.dual_tank_slosh else None,
        )
        r = self.vehicle.geometry.body_radius_m
        u_l = self.vehicle.geometry.upper_length_m
        u_i_lat = u_m * (u_l * u_l / 12.0 + r * r / 4.0)
        u_i_roll = 0.5 * u_m * r * r

        imp = separation_event_3d(
            stack_pos_com_enu_m=p_com,
            stack_vel_com_enu_m_s=v_com,
            quat_wxyz=q_wxyz,
            omega_body_rad_s=om_b,
            booster_mass_kg=b_m,
            upper_mass_kg=u_m,
            booster_cg_station_m=mp_b.cg_station_m,
            upper_cg_station_m=self.vehicle.geometry.upper_station_m,
            booster_inertia_diag_kg_m2=mp_b.inertia_diag_kg_m2,
            upper_inertia_diag_kg_m2=(u_i_lat, u_i_lat, u_i_roll),
            settings=self.separation_settings,
        )

        self.upper_attached = False
        self.separation_time_s = self.time_s
        self.body.update_mass_properties(
            b_m,
            self.gear_extension,
            upper_attached=False,
            airframe_mass_offset_kg=self.airframe_mass_offset_kg,
            inertia_scale=dist.inertia_scale,
            dual_tank=pv2.dual_tank_slosh,
            rcs_propellant_kg=self.rcs_prop_kg if pv2.dual_tank_slosh else None,
            dual_tank_geom=pv2.dual_tank_geometry if pv2.dual_tank_slosh else None,
            struts_active=False,
        )
        self.body.set_pose_from_com(
            imp["booster_pos_enu_m"],
            imp["booster_vel_enu_m_s"],
            q_wxyz,
            imp["booster_omega_body_rad_s"],
            fin_deflections_rad=(
                self.fin_deflection[0],
                self.fin_deflection[1],
                self.fin_deflection[2],
                self.fin_deflection[3],
            ),
        )
        up_p = np.asarray(imp["upper_pos_enu_m"], dtype=np.float64)
        up_v = np.asarray(imp["upper_vel_enu_m_s"], dtype=np.float64)
        self.upper_orbit_result = evaluate_upper_stage_orbit_3d(
            pos_sep_enu_m=(float(up_p[0]), float(up_p[1]), float(up_p[2])),
            vel_sep_enu_m_s=(float(up_v[0]), float(up_v[1]), float(up_v[2])),
            latitude_deg=self.scenario.atmosphere.latitude_deg,
            settings=self.vehicle.upper_stage,
            replay_hz=self.config.replay_hz,
            use_wgs84_j2=pv2.earth_wgs84_j2,
        )
        self.separation_record = {
            "time_s": self.time_s,
            "altitude_m": float(imp["booster_pos_enu_m"][2]),
            "speed_m_s": float(np.linalg.norm(imp["booster_vel_enu_m_s"])),
            "main_propellant_kg": self.main_prop_kg,
            "checks_before_separation": checks,
            "linear_momentum_residual_kg_m_s": float(imp["linear_momentum_residual_kg_m_s"]),
        }

    def _record_first_contact(self) -> None:
        p_com, v_com, q, om_b = self.body.get_com_state()
        r_wb = quat_to_rotmat(q)
        cg = self.body.current_mass_props.cg_station_m
        gear = self.vehicle.landing_gear

        # Evaluate 4 footpad positions and velocities
        foot_positions = []
        foot_v_vert_max = 0.0
        foot_v_horiz_max = 0.0
        for ux, uy in LEG_UNIT_DIRS:
            r_f_b = np.array(
                [ux * gear.half_span_m, uy * gear.half_span_m, gear.foot_station_m - cg]
            )
            p_f = p_com + r_wb @ r_f_b
            v_f = v_com + r_wb @ np.cross(om_b, r_f_b)
            foot_positions.append(p_f)
            foot_v_vert_max = max(foot_v_vert_max, max(0.0, -float(v_f[2])))
            foot_v_horiz_max = max(foot_v_horiz_max, hypot(float(v_f[0]), float(v_f[1])))

        pos_err = hypot(float(p_com[0]), float(p_com[1]))
        tilt_deg = degrees(tilt_angle_rad(q))
        rate_deg_s = degrees(float(np.linalg.norm(om_b)))
        feet_on_pad = all(
            hypot(float(fp[0]), float(fp[1])) <= gear.pad_radius_m for fp in foot_positions
        )

        lim = self.vehicle.safety
        criteria = {
            "position": pos_err <= lim.max_position_error_m,
            "vertical_foot_speed": foot_v_vert_max <= lim.max_vertical_foot_speed_m_s,
            "horizontal_foot_speed": foot_v_horiz_max <= lim.max_horizontal_contact_speed_m_s,
            "tilt": tilt_deg <= lim.max_contact_tilt_deg,
            "angular_rate": rate_deg_s <= lim.max_contact_rate_deg_s,
            "main_propellant": self.main_prop_kg > 0.0,
            "rcs_propellant": self.rcs_prop_kg >= 0.0,
            "gear_deployed": self.gear_extension >= 0.99,
            "feet_on_pad": feet_on_pad,
        }
        self.first_contact_record = {
            "time_s": self.time_s,
            "pos_xy_m": (float(p_com[0]), float(p_com[1])),
            "position_error_m": pos_err,
            "vertical_foot_speed_m_s": foot_v_vert_max,
            "horizontal_foot_speed_m_s": foot_v_horiz_max,
            "tilt_deg": tilt_deg,
            "contact_rate_deg_s": rate_deg_s,
            "criteria": criteria,
        }

    def step(self, command: ActuatorCommand3D, duration_s: float) -> None:
        """Advance the 6DOF MuJoCo plant by one control interval `duration_s` (typically 0.02 s)."""
        if self.done:
            return
        self.action_queue.append(command)
        cmd = self.action_queue[0]
        self.phase = cmd.phase

        # Enforce engine restart readiness, ignition delay (0.2 s), and max 3 restarts
        prop = self.vehicle.propulsion
        req_thr = prop.quantize_throttle(cmd.throttle_cmd, cmd.active_engines)
        if self.touchdown_time_s is not None:
            req_thr = 0.0
            cmd_engines = 0
        else:
            cmd_engines = cmd.active_engines if req_thr > 0.0 else 0

        if req_thr > 0.0 and self.main_prop_kg > 0.0:
            if not self._engine_lit_before:
                self._engine_lit_before = True
                self.active_engines = cmd_engines
            elif self.active_engines == 0:
                if not self._ignition_requested and self.restarts_used < prop.max_restarts:
                    fr_chk = feed_rates(
                        self.main_prop_kg, 0.0, 0.0, self.feed_state, self.feed_settings
                    )
                    if ignition_ready(
                        self.feed_state, fr_chk.tank_pressure_pa, self.ignition_settings
                    ):
                        self._ignition_requested = True
                        self._ignition_ready_s = self.time_s + self.ignition_settings.delay_s
                if self._ignition_requested and self.time_s >= self._ignition_ready_s - 1e-9:
                    self.restarts_used += 1
                    self.active_engines = cmd_engines
                    self._ignition_requested = False
                    self._ignition_ready_s = float("inf")
                else:
                    req_thr = 0.0
            else:
                self.active_engines = cmd_engines
        else:
            self.active_engines = 0
            self._ignition_requested = False
            self._ignition_ready_s = float("inf")
            req_thr = 0.0

        # Choose integration dt: 0.002 s in landing contact zone (h < 25 m), else 0.01 s
        p_c0, _, _, _ = self.body.get_com_state()
        in_contact_zone = (not self.upper_attached) and (float(p_c0[2]) < 35.0)
        sub_dt = self.config.contact_dt_s if in_contact_zone else self.config.integration_dt_s
        n_sub = max(1, int(round(duration_s / sub_dt)))
        dt = duration_s / n_sub

        # Dynamic mass/COM/3-axis inertia update via scratch MjData (10 Hz or on gear/strut change)
        dist = self.scenario.disturbances
        pv2 = self.physics_v2
        struts_on = (
            (not self.upper_attached) and (self.gear_extension >= 0.85) and (float(p_c0[2]) < 45.0)
        )
        if (
            self.time_s >= self._last_mass_update_s + 0.10 - 1e-9
            or struts_on != self._last_struts_on
            or (0.0 < self.gear_extension < 1.0)
        ):
            mp = self.body.update_mass_properties(
                self._total_mass(),
                self.gear_extension,
                upper_attached=self.upper_attached,
                airframe_mass_offset_kg=self.airframe_mass_offset_kg,
                inertia_scale=dist.inertia_scale,
                dual_tank=pv2.dual_tank_slosh,
                rcs_propellant_kg=self.rcs_prop_kg if pv2.dual_tank_slosh else None,
                dual_tank_geom=pv2.dual_tank_geometry if pv2.dual_tank_slosh else None,
                struts_active=struts_on,
            )
            self._last_mass_update_s = self.time_s
            self._last_struts_on = struts_on
        else:
            mp = self.body.current_mass_props

        f_rate = radians(self.vehicle.fins.deflection_rate_deg_s)
        f_tau = self.vehicle.fins.actuator_time_constant_s
        f_lim = radians(self.vehicle.fins.deflection_limit_deg)
        g_rate = radians(prop.gimbal_rate_deg_s)
        g_tau = prop.gimbal_tau_s
        g_lim = radians(prop.gimbal_limit_deg)

        last_ft = None
        for _ in range(n_sub):
            # 1. Update actuator first-order + rate-limited states
            self.engine_throttle += (req_thr - self.engine_throttle) * (
                1.0 - exp(-dt / self.engine_tau_s)
            )
            if self.main_prop_kg <= 0.0 or self.touchdown_time_s is not None:
                self.engine_throttle = 0.0

            for i in range(2):
                dg = cmd.gimbal_rad[i] - self.gimbal[i]
                step_g = max(-g_rate * dt, min(g_rate * dt, dg * (1.0 - exp(-dt / g_tau))))
                self.gimbal[i] = max(-g_lim, min(g_lim, self.gimbal[i] + step_g))
            g_mag = hypot(self.gimbal[0], self.gimbal[1])
            if g_mag > g_lim and g_mag > 1e-12:
                self.gimbal[0] *= g_lim / g_mag
                self.gimbal[1] *= g_lim / g_mag

            for i in range(4):
                df = cmd.fin_deflection_rad[i] - self.fin_deflection[i]
                step_f = max(-f_rate * dt, min(f_rate * dt, df * (1.0 - exp(-dt / f_tau))))
                self.fin_deflection[i] = max(-f_lim, min(f_lim, self.fin_deflection[i] + step_f))

            if cmd.deploy_fins and not self.upper_attached:
                self.fin_extension = min(
                    1.0, self.fin_extension + dt / self.vehicle.fins.deployment_time_s
                )
            elif not cmd.deploy_fins and self.fin_extension > 0.0:
                self.fin_extension = max(
                    0.0, self.fin_extension - dt / self.vehicle.fins.retraction_time_s
                )
            if cmd.deploy_gear and not self.upper_attached:
                self.gear_extension = min(
                    1.0, self.gear_extension + dt / self.vehicle.landing_gear.deployment_time_s
                )

            self.body.sync_kinematic_fins(
                (
                    self.fin_deflection[0],
                    self.fin_deflection[1],
                    self.fin_deflection[2],
                    self.fin_deflection[3],
                )
            )

            # 2. Evaluate 3D external forces, feed limiting, slosh, and optional roll disturbance
            p_com, v_com, q_wxyz, om_b = self.body.get_com_state()
            r_wb = quat_to_rotmat(q_wxyz)
            tot_m = self._total_mass()
            _, p_pa, _ = self.atmo_lut.at(float(p_com[2]))
            _, dem_flow = compute_engine_thrust_3d(
                self.engine_throttle, p_pa, self.vehicle, thrust_scale=dist.thrust_scale
            )
            ax_proper = float(self._last_accel_body[2])
            fr = feed_rates(
                self.main_prop_kg, dem_flow, ax_proper, self.feed_state, self.feed_settings
            )
            if not fr.inventory_valid:
                self.feed_inventory_ok = False
            delivered_ratio = fr.delivered_flow_kg_s / dem_flow if dem_flow > 1e-6 else 1.0

            rcs_cmd = (
                cmd.rcs_torque_nm
                if (self.rcs_prop_kg > 0.0 and self.touchdown_time_s is None)
                else (0.0, 0.0, 0.0)
            )
            if pv2.dual_tank_slosh:
                b_off = dual_slosh_body_offset_3d(
                    tot_m, self.main_prop_kg, self.dual_slosh_state, self.dual_slosh_settings
                )
            else:
                b_off = slosh_body_offset_3d(
                    tot_m, self.main_prop_kg, self.slosh_state, self.slosh_settings
                )

            srp_drag_mult = 1.0
            srp_cp_shift = 0.0
            srp_buffet_tau = (0.0, 0.0, 0.0)
            if pv2.srp_interference and (not self.upper_attached) and self.engine_throttle > 1e-6:
                t_k_s, p_pa_s, rho_s = self.atmo_lut.at(float(p_com[2]))
                wx_s, wy_s, wz_s = self.wind_lut.at(float(p_com[2]), self.time_s)
                v_rel_s = np.array(
                    [float(v_com[0]) - wx_s, float(v_com[1]) - wy_s, float(v_com[2]) - wz_s],
                    dtype=np.float64,
                )
                spd_s = float(np.linalg.norm(v_rel_s))
                if spd_s > 1.0:
                    q_s = 0.5 * rho_s * spd_s * spd_s
                    mach_s = spd_s / np.sqrt(1.4 * R_AIR * t_k_s)
                    vz_b_s = float((r_wb.T @ v_rel_s)[2])
                    t_est_s, _ = compute_engine_thrust_3d(
                        self.engine_throttle, p_pa_s, self.vehicle, thrust_scale=dist.thrust_scale
                    )
                    srp_eff = compute_srp_effects(
                        mach=float(mach_s),
                        v_air_zb_m_s=vz_b_s,
                        thrust_n=t_est_s * delivered_ratio,
                        dynamic_pressure_pa=q_s,
                        body_radius_m=self.vehicle.geometry.body_radius_m,
                        dt_s=dt,
                        upper_attached=self.upper_attached,
                        settings=self.srp_settings,
                        buffet_filter=self.srp_buffet_filter,
                    )
                    srp_drag_mult = srp_eff.drag_multiplier
                    srp_cp_shift = srp_eff.delta_z_cp_m
                    srp_buffet_tau = srp_eff.buffet_torque_body_nm
                    self.peak_srp_drag_reduction = max(
                        self.peak_srp_drag_reduction, 1.0 - srp_drag_mult
                    )

            ft = compute_forces_and_torques_3d(
                altitude_m=float(p_com[2]),
                vel_enu_m_s=v_com,
                quat_wxyz=q_wxyz,
                omega_body_rad_s=om_b,
                cg_station_m=mp.cg_station_m,
                throttle_actual=self.engine_throttle,
                gimbal_rad=(self.gimbal[0], self.gimbal[1]),
                rcs_torque_body_nm=rcs_cmd,
                fin_deflection_rad=(
                    self.fin_deflection[0],
                    self.fin_deflection[1],
                    self.fin_deflection[2],
                    self.fin_deflection[3],
                ),
                fin_extension=self.fin_extension,
                atmo_lut=self.atmo_lut,
                wind_lut=self.wind_lut,
                time_s=self.time_s,
                vehicle=self.vehicle,
                upper_attached=self.upper_attached,
                thrust_scale=dist.thrust_scale,
                drag_scale=dist.drag_scale,
                delivered_ratio=delivered_ratio,
                body_offset_body_m=b_off,
                axial_drag_multiplier=srp_drag_mult,
                cp_shift_m=srp_cp_shift,
            )
            last_ft = ft

            # Optional external 3D roll disturbance pulse + SRP buffeting + ground effect
            tau_b = np.asarray(ft.torque_body_nm, dtype=np.float64)
            if srp_buffet_tau != (0.0, 0.0, 0.0):
                tau_b = tau_b + np.asarray(srp_buffet_tau, dtype=np.float64)
            if (
                dist.roll_disturbance_torque_nm != 0.0
                and dist.roll_disturbance_start_s
                <= self.time_s
                <= dist.roll_disturbance_start_s + dist.roll_disturbance_duration_s
            ):
                t_in = self.time_s - dist.roll_disturbance_start_s
                t_left = dist.roll_disturbance_duration_s - t_in
                u_env = max(0.0, min(1.0, min(t_in, t_left) / 0.5))
                env = u_env * u_env * (3.0 - 2.0 * u_env)
                tau_b[2] += dist.roll_disturbance_torque_nm * env

            f_w_ext = np.asarray(ft.force_world_n, dtype=np.float64)
            if (
                pv2.ground_effect
                and (not self.upper_attached)
                and ft.thrust_n > 1.0
                and float(p_com[2]) < 35.0
            ):
                ge = compute_ground_effect_3d(
                    pos_com_enu_m=p_com,
                    rot_wb=r_wb,
                    cg_station_m=mp.cg_station_m,
                    nozzle_station_m=self.vehicle.geometry.nozzle_station_m,
                    thrust_n=ft.thrust_n,
                    upper_attached=self.upper_attached,
                    settings=self.ground_effect_settings,
                )
                if ge.cushion_force_n > 0.0:
                    f_w_ext = f_w_ext + np.asarray(ge.force_world_n, dtype=np.float64)
                    tau_b = tau_b + np.asarray(ge.torque_body_nm, dtype=np.float64)
                    self.peak_ground_effect_n = max(self.peak_ground_effect_n, ge.cushion_force_n)

            # Slosh response (v2 dual-tank 3x3 coupled vs legacy v1 single-tank)
            a_proper_b = r_wb.T @ (f_w_ext / max(tot_m, 1.0))
            self._last_accel_body = a_proper_b
            if pv2.dual_tank_slosh:
                dsl = dual_slosh_3d_response(
                    total_mass_kg=tot_m,
                    rigid_inertia_diag_kg_m2=mp.inertia_diag_kg_m2,
                    main_propellant_kg=self.main_prop_kg,
                    cg_station_m=mp.cg_station_m,
                    proper_accel_body_m_s2=a_proper_b,
                    external_torque_body_nm=tau_b,
                    omega_body_rad_s=om_b,
                    state=self.dual_slosh_state,
                    settings=self.dual_slosh_settings,
                )
                if not dsl.valid:
                    self.slosh_domain_ok = False
                self.peak_dual_slosh_force_n = max(
                    self.peak_dual_slosh_force_n, dsl.constraint_force_n
                )
                tau_b = tau_b + np.asarray(dsl.slosh_torque_body_nm, dtype=np.float64)

                # Integrate both LOX and CH4 spherical pendulums (symplectic midpoint)
                s_lox = self.dual_slosh_state.lox
                lox_qxd = s_lox.qxd_m_s + dsl.lox_qdd_m_s2[0] * dt
                lox_qyd = s_lox.qyd_m_s + dsl.lox_qdd_m_s2[1] * dt
                s_ch4 = self.dual_slosh_state.ch4
                ch4_qxd = s_ch4.qxd_m_s + dsl.ch4_qdd_m_s2[0] * dt
                ch4_qyd = s_ch4.qyd_m_s + dsl.ch4_qdd_m_s2[1] * dt
                self.dual_slosh_state = DualTankSloshState(
                    lox=Slosh3DState(
                        qx_m=s_lox.qx_m + 0.5 * (s_lox.qxd_m_s + lox_qxd) * dt,
                        qy_m=s_lox.qy_m + 0.5 * (s_lox.qyd_m_s + lox_qyd) * dt,
                        qxd_m_s=lox_qxd,
                        qyd_m_s=lox_qyd,
                    ),
                    ch4=Slosh3DState(
                        qx_m=s_ch4.qx_m + 0.5 * (s_ch4.qxd_m_s + ch4_qxd) * dt,
                        qy_m=s_ch4.qy_m + 0.5 * (s_ch4.qyd_m_s + ch4_qyd) * dt,
                        qxd_m_s=ch4_qxd,
                        qyd_m_s=ch4_qyd,
                    ),
                )
            else:
                tank_st = self.vehicle.geometry.tank_bottom_m + 0.5 * mp.fuel_height_m
                sl_resp = slosh_3d_response(
                    total_mass_kg=tot_m,
                    rigid_inertia_diag_kg_m2=mp.inertia_diag_kg_m2,
                    main_propellant_kg=self.main_prop_kg,
                    station_offset_m=tank_st - mp.cg_station_m,
                    proper_accel_body_m_s2=a_proper_b,
                    external_torque_body_nm=tau_b,
                    omega_body_rad_s=om_b,
                    state=self.slosh_state,
                    settings=self.slosh_settings,
                )
                if not sl_resp.valid:
                    self.slosh_domain_ok = False
                tau_b = tau_b + np.asarray(sl_resp.slosh_torque_body_nm, dtype=np.float64)

                # Integrate slosh (symplectic midpoint)
                qxd_new = self.slosh_state.qxd_m_s + sl_resp.qdd_m_s2[0] * dt
                qyd_new = self.slosh_state.qyd_m_s + sl_resp.qdd_m_s2[1] * dt
                self.slosh_state = Slosh3DState(
                    qx_m=self.slosh_state.qx_m + 0.5 * (self.slosh_state.qxd_m_s + qxd_new) * dt,
                    qy_m=self.slosh_state.qy_m + 0.5 * (self.slosh_state.qyd_m_s + qyd_new) * dt,
                    qxd_m_s=qxd_new,
                    qyd_m_s=qyd_new,
                )

            self.main_prop_kg = max(0.0, self.main_prop_kg - fr.delivered_flow_kg_s * dt)
            self.rcs_prop_kg = max(0.0, self.rcs_prop_kg - ft.rcs_flow_kg_s * dt)
            self.feed_state = FeedState(
                buffer_kg=max(
                    0.0,
                    min(
                        self.main_prop_kg,
                        self.feed_state.buffer_kg + fr.buffer_derivative_kg_s * dt,
                    ),
                ),
                acquisition=max(
                    0.0,
                    min(1.0, self.feed_state.acquisition + fr.acquisition_derivative_s_inv * dt),
                ),
                tank_gas_kg=max(
                    0.0, self.feed_state.tank_gas_kg + fr.tank_gas_derivative_kg_s * dt
                ),
                bottle_gas_kg=max(
                    0.0, self.feed_state.bottle_gas_kg + fr.bottle_gas_derivative_kg_s * dt
                ),
            )

            # Pad hold-down before TWR > 1 & Coriolis / gravity
            if pv2.earth_wgs84_j2:
                a_cor = wgs84_coriolis_and_transport_enu(
                    altitude_m=float(p_com[2]),
                    vel_enu_m_s=v_com,
                    latitude_deg=self.scenario.atmosphere.latitude_deg,
                    coriolis_enabled=self.scenario.atmosphere.coriolis_enabled,
                    planar_coriolis_only=self.scenario.atmosphere.planar_coriolis_only,
                )
                g_step = wgs84_normal_gravity(
                    self.scenario.atmosphere.latitude_deg, float(p_com[2])
                )
            else:
                a_cor = compute_coriolis_and_centrifugal_3d(
                    altitude_m=float(p_com[2]),
                    vel_enu_m_s=v_com,
                    latitude_deg=self.scenario.atmosphere.latitude_deg,
                    coriolis_enabled=self.scenario.atmosphere.coriolis_enabled,
                    planar_coriolis_only=self.scenario.atmosphere.planar_coriolis_only,
                )
                g_step = None

            if self.on_pad and self.upper_attached:
                g_pad = g_step if g_step is not None else fast_gravity(float(p_com[2]))
                if f_w_ext[2] > tot_m * g_pad:
                    self.on_pad = False
                else:
                    self.time_s += dt
                    continue

            f_tot_w = f_w_ext + tot_m * a_cor
            tau_w = r_wb @ tau_b
            self.body.step_physics(
                dt_s=dt,
                force_world_n=f_tot_w,
                torque_world_nm=tau_w,
                altitude_for_gravity_m=float(p_com[2]),
                gravity_m_s2=g_step,
            )
            self.time_s += dt

            # 3. Monitor contacts, extrema, separation, and post-touchdown stability
            p_after, v_after, q_after, om_after = self.body.get_com_state()
            if not np.all(np.isfinite(p_after)) or not isfinite(float(q_after[0])):
                self.outcome = "numerical_divergence"
                return

            in_ground_contact = self.touchdown_time_s is not None or (
                in_contact_zone
                and (
                    self.body.sample_contact().any_contact
                    or float(p_after[2]) - mp.cg_station_m
                    <= -self.vehicle.landing_gear.foot_station_m + 0.05
                )
            )
            self.max_q_pa = max(self.max_q_pa, ft.dynamic_pressure_pa)
            self.max_heat_flux_w_m2 = max(self.max_heat_flux_w_m2, ft.heat_flux_w_m2)
            if self.upper_attached:
                qa = (ft.dynamic_pressure_pa / 1_000.0) * abs(ft.total_alpha_deg)
                self.max_q_alpha_kpa_deg = max(self.max_q_alpha_kpa_deg, qa)
                if ft.dynamic_pressure_pa >= 10_000.0:
                    self.max_ascent_high_q_alpha_deg = max(
                        self.max_ascent_high_q_alpha_deg, abs(ft.total_alpha_deg)
                    )
                self.max_ascent_axial_g = max(
                    self.max_ascent_axial_g, float(np.linalg.norm(a_proper_b)) / G0
                )
            elif not in_ground_contact:
                self.max_recovery_g = max(
                    self.max_recovery_g, float(np.linalg.norm(a_proper_b)) / G0
                )
                self.max_roll_deg = max(self.max_roll_deg, abs(degrees(roll_angle_rad(q_after))))

            ang_acc = degrees(float(np.linalg.norm((om_after - self._prev_omega_b) / dt)))
            if not in_ground_contact:
                self.max_rate_deg_s = max(
                    self.max_rate_deg_s, degrees(float(np.linalg.norm(om_after)))
                )
                if self.separation_time_s is None or self.time_s > self.separation_time_s + 0.20:
                    self.max_accel_deg_s2 = max(self.max_accel_deg_s2, ang_acc)
            self._prev_omega_b = om_after.copy()
            self.max_gimbal_deg = max(
                self.max_gimbal_deg, degrees(hypot(self.gimbal[0], self.gimbal[1]))
            )
            self.max_fin_hinge_nm = max(self.max_fin_hinge_nm, max(ft.fin_hinge_nm))

            self._check_separation(ft.dynamic_pressure_pa, ft.total_alpha_deg)
            if self.done:
                return

            if in_contact_zone:
                cs = self.body.sample_contact()
                self._last_contact = cs
                if cs.body_ground_collision:
                    self.body_ground_collision = True
                    self.standing_stable = False
                    self.outcome = "body_ground_collision"
                    return
                if cs.any_contact:
                    if self.touchdown_time_s is None:
                        self.touchdown_time_s = self.time_s
                        self._record_first_contact()
                    self.max_leg_stroke_m = max(self.max_leg_stroke_m, max(cs.leg_stroke_m))
                    self.max_leg_force_n = max(self.max_leg_force_n, max(cs.leg_normal_n))
                    self.max_contact_load_g = max(self.max_contact_load_g, cs.contact_load_g)

                if self.touchdown_time_s is not None:
                    tilt_now = degrees(tilt_angle_rad(q_after))
                    self.post_contact_max_tilt_deg = max(self.post_contact_max_tilt_deg, tilt_now)
                    if tilt_now > 35.0:
                        self.standing_stable = False
                        self.outcome = "tip_over"
                        return
                    if (
                        self.time_s - self.touchdown_time_s
                        >= self.vehicle.landing_gear.settle_duration_s
                    ):
                        gear = self.vehicle.landing_gear
                        spd = float(np.linalg.norm(v_after))
                        rate = degrees(float(np.linalg.norm(om_after)))
                        legs_down = sum(1 for c in cs.leg_in_contact if c)
                        self.standing_stable = (
                            not self.body_ground_collision
                            and self.engine_throttle <= 1e-3
                            and tilt_now <= gear.max_standing_tilt_deg
                            and spd <= gear.max_standing_speed_m_s
                            and rate <= gear.max_standing_rate_deg_s
                            and legs_down == 4
                        )
                        self.outcome = "landed" if self.standing_stable else "standing_unstable"
                        return

            if self.time_s >= self.config.deadline_s:
                self.outcome = "timeout"
                return

        # Track per-phase stats & record 50 Hz replay frame
        ps = self.phase_stats.setdefault(
            self.phase,
            {
                "start_s": self.time_s - duration_s,
                "end_s": self.time_s,
                "max_throttle": 0.0,
                "max_throttle_cmd": 0.0,
                "steps": 0,
            },
        )
        ps["end_s"] = self.time_s
        ps["max_throttle"] = max(ps["max_throttle"], self.engine_throttle)
        ps["max_throttle_cmd"] = max(ps["max_throttle_cmd"], cmd.throttle_cmd)
        ps["steps"] += 1

        if (
            self.config.record_replay
            and self.time_s >= self._next_replay_s - 1e-9
            and last_ft is not None
        ):
            p_base, q_b = self.body.get_base_pose()
            p_com, v_com, _, om_b = self.body.get_com_state()
            up_pos, up_quat = self._upper_stage_pose_at(self.time_s)
            self.replay_frames.append(
                ReplayFrame3D(
                    time_s=self.time_s,
                    phase=self.phase,
                    pos_enu_m=(float(p_com[0]), float(p_com[1]), float(p_com[2])),
                    quat_wxyz=(float(q_b[0]), float(q_b[1]), float(q_b[2]), float(q_b[3])),
                    vel_enu_m_s=(float(v_com[0]), float(v_com[1]), float(v_com[2])),
                    omega_body_rad_s=(float(om_b[0]), float(om_b[1]), float(om_b[2])),
                    throttle=self.engine_throttle,
                    active_engines=self.active_engines,
                    gimbal_rad=(self.gimbal[0], self.gimbal[1]),
                    rcs_torque_nm=cmd.rcs_torque_nm,
                    fin_extension=self.fin_extension,
                    fin_deflection_rad=(
                        self.fin_deflection[0],
                        self.fin_deflection[1],
                        self.fin_deflection[2],
                        self.fin_deflection[3],
                    ),
                    gear_extension=self.gear_extension,
                    leg_stroke_m=self._last_contact.leg_stroke_m,
                    dynamic_pressure_pa=last_ft.dynamic_pressure_pa,
                    mach=last_ft.mach,
                    alpha_deg=last_ft.total_alpha_deg,
                    main_propellant_kg=self.main_prop_kg,
                    rcs_propellant_kg=self.rcs_prop_kg,
                    upper_attached=self.upper_attached,
                    upper_pos_enu_m=up_pos,
                    upper_quat_wxyz=up_quat,
                    base_pos_enu_m=(float(p_base[0]), float(p_base[1]), float(p_base[2])),
                    cg_station_m=self.body.current_mass_props.cg_station_m,
                    heat_flux_w_m2=last_ft.heat_flux_w_m2,
                )
            )
            self._next_replay_s += self._replay_dt_s

    def summary(self) -> dict:
        s = self.vehicle.safety
        gear = self.vehicle.landing_gear
        eps = 1e-6
        sep_ok = self.separation_record is not None and all(
            self.separation_record["checks_before_separation"].values()
        )
        touch_ok = self.first_contact_record is not None and all(
            self.first_contact_record["criteria"].values()
        )
        criteria = {
            "dynamic_pressure": self.max_q_pa <= s.max_q_pa + eps,
            "entry_heat_flux": self.max_heat_flux_w_m2 <= s.max_heat_flux_w_m2 + eps,
            "ascent_q_alpha": self.max_q_alpha_kpa_deg <= s.max_q_alpha_kpa_deg + eps,
            "ascent_high_q_alpha": self.max_ascent_high_q_alpha_deg <= s.max_high_q_alpha_deg + eps,
            "ascent_axial_g": self.max_ascent_axial_g <= s.max_ascent_axial_g + eps,
            "flight_load_g": self.max_recovery_g <= s.max_recovery_flight_g + eps,
            "angular_rate": self.max_rate_deg_s <= s.max_angular_rate_deg_s + eps,
            "angular_acceleration": self.max_accel_deg_s2
            <= s.max_angular_acceleration_deg_s2 + eps,
            "gimbal": self.max_gimbal_deg <= self.vehicle.propulsion.gimbal_limit_deg + eps,
            "fin_hinge": self.max_fin_hinge_nm <= self.vehicle.fins.per_fin_hinge_limit_nm + eps,
            "feed_inventory": self.feed_inventory_ok,
            "slosh_domain": self.slosh_domain_ok,
            "separation": sep_ok,
            "upper_stage_orbit": bool(
                self.upper_orbit_result and self.upper_orbit_result.orbit_inserted
            ),
            "touchdown": touch_ok,
            "leg_stroke": self.max_leg_stroke_m <= gear.max_leg_stroke_m + eps,
            "leg_force": self.max_leg_force_n <= gear.max_leg_force_n + eps,
            "contact_load_g": self.max_contact_load_g <= gear.max_contact_load_g + eps,
            "engine_off_standing": bool(self.standing_stable),
        }
        failed = [k for k, v in criteria.items() if not v]
        passed = (self.outcome == "landed") and (len(failed) == 0)
        fc = self.first_contact_record or {}
        _, v_end, q_end, _ = self.body.get_com_state()
        return {
            "version": "dm100_3d_mujoco",
            "scenario": self.scenario.name,
            "outcome": self.outcome,
            "requirements_passed": passed,
            "criteria": criteria,
            "failed_criteria": failed,
            "flight_time_s": self.time_s,
            "touchdown_time_s": self.touchdown_time_s,
            "restarts_used": self.restarts_used,
            "main_remaining_kg": self.main_prop_kg,
            "rcs_remaining_kg": self.rcs_prop_kg,
            "position_error_m": fc.get("position_error_m"),
            "pos_xy_m": fc.get("pos_xy_m"),
            "vertical_foot_speed_m_s": fc.get("vertical_foot_speed_m_s"),
            "horizontal_foot_speed_m_s": fc.get("horizontal_foot_speed_m_s"),
            "tilt_deg": fc.get("tilt_deg"),
            "contact_rate_deg_s": fc.get("contact_rate_deg_s"),
            "peak_dynamic_pressure_pa": self.max_q_pa,
            "peak_heat_flux_w_m2": self.max_heat_flux_w_m2,
            "peak_ascent_axial_g": self.max_ascent_axial_g,
            "peak_recovery_g": self.max_recovery_g,
            "peak_angular_rate_deg_s": self.max_rate_deg_s,
            "peak_angular_acceleration_deg_s2": self.max_accel_deg_s2,
            "peak_roll_deg": self.max_roll_deg,
            "peak_fin_hinge_nm": self.max_fin_hinge_nm,
            "post_contact": {
                "max_leg_stroke_m": self.max_leg_stroke_m,
                "max_leg_force_n": self.max_leg_force_n,
                "max_contact_load_g": self.max_contact_load_g,
                "max_tilt_deg": self.post_contact_max_tilt_deg,
                "standing_stable": self.standing_stable,
                "final_speed_m_s": float(np.linalg.norm(v_end)),
                "final_tilt_deg": degrees(tilt_angle_rad(q_end)),
            },
            "upper_stage_orbit": (
                {
                    "orbit_inserted": self.upper_orbit_result.orbit_inserted,
                    "final_apogee_km": self.upper_orbit_result.final_apogee_km,
                    "mean_radius_altitude_km": self.upper_orbit_result.mean_radius_altitude_km,
                    "mean_eccentricity": self.upper_orbit_result.mean_eccentricity,
                    "upper_propellant_remaining_kg": (
                        self.upper_orbit_result.upper_propellant_remaining_kg
                    ),
                    "replay_samples_count": len(self.upper_orbit_result.replay_samples),
                }
                if self.upper_orbit_result is not None
                else None
            ),
            "physics_v2_metrics": (
                {
                    "peak_srp_drag_reduction": self.peak_srp_drag_reduction,
                    "peak_ground_effect_n": self.peak_ground_effect_n,
                    "peak_dual_slosh_force_n": self.peak_dual_slosh_force_n,
                }
                if self.physics_v2.any_enabled()
                else None
            ),
            "phases": self.phase_stats,
            "replay_frame_count": len(self.replay_frames),
        }


def run_flight_3d(
    scenario: Scenario3D | None = None,
    *,
    vehicle: DM100VehicleConfig | None = None,
    sim_config: SimConfig3D | None = None,
    guidance_settings: Guidance3DSettings | None = None,
    controller: ControllerProtocol3D | None = None,
) -> tuple[dict, list[ReplayFrame3D]]:
    """Run one complete closed-loop 6DOF DM100 3D flight and return (summary, replay_frames)."""
    cfg = vehicle or DM100VehicleConfig()
    scen = scenario or Scenario3D()
    scfg = sim_config or SimConfig3D(seed=scen.seed, sensor_seed=scen.seed + 1000)
    plant = MuJoCoPlant3D(vehicle=cfg, scenario=scen, sim_config=scfg)
    ctrl: ControllerProtocol3D = controller or RtlsController3D(
        vehicle=cfg,
        atmo_settings=scen.atmosphere,
        wind_config=scen.wind,
        sensor_config=scen.sensors,
        guidance_settings=guidance_settings,
        gnc_v2=scfg.gnc_v2,
    )

    dt_ctrl = scfg.control_dt_s
    while not plant.done:
        pkt = plant.telemetry()
        cmd = ctrl.step(pkt)
        plant.step(cmd, dt_ctrl)

    summary = plant.summary()
    diag = ctrl.diagnostics()
    summary["socp_solver_used"] = diag.socp_solver_used
    summary["socp_solver_status"] = diag.socp_solver_status
    summary["socp_solve_ms"] = diag.socp_solve_ms
    return summary, plant.replay_frames
