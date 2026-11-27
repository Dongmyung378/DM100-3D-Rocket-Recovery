"""DM100 3D vehicle specification, YAML configuration loader, and 3D mass-inertia properties.

All mass, geometry, propulsion, aerodynamic, grid-fin, upper-stage, and safety parameters
inherit directly from DM100 2D (`DM100-2D-Rocket-Recovery`) and extend cleanly to 6DOF
(including 3-axis inertia tensor diag(Ixx, Iyy, Izz) and 4 independent landing legs/fins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import Any

import yaml

G0 = 9.80665
R_AIR = 287.05287
EARTH_RADIUS_M = 6_371_000.0
EARTH_MU_M3_S2 = 3.986004418e14
OMEGA_EARTH = 7.292115e-5
EARTH_EQUATORIAL_SPEED_M_S = 465.101
SUTTON_GRAVES_EARTH_K = 1.7415e-4


@dataclass(frozen=True, slots=True)
class VehicleGeometry:
    body_length_m: float = 24.0
    body_radius_m: float = 1.55
    engine_mass_kg: float = 1_050.0
    engine_station_m: float = 0.8
    airframe_mass_kg: float = 3_750.0
    airframe_station_m: float = 11.5
    gear_mass_kg: float = 800.0
    gear_retracted_station_m: float = 2.0
    gear_deployed_station_m: float = 0.3
    tank_bottom_m: float = 3.0
    tank_length_m: float = 15.0
    propellant_density_kg_m3: float = 830.0
    nozzle_station_m: float = -0.2
    nozzle_radius_m: float = 0.78
    rcs_station_m: float = 22.4
    rcs_arm_m: float = 9.5
    upper_station_m: float = 28.0
    upper_length_m: float = 9.0


@dataclass(frozen=True, slots=True)
class PropellantBudget:
    main_propellant_kg: float = 69_500.0
    rcs_propellant_kg: float = 300.0


@dataclass(frozen=True, slots=True)
class PropulsionConfig:
    engine_count: int = 7
    sea_level_thrust_n: float = 1_550_000.0
    sea_level_isp_s: float = 325.0
    nozzle_exit_area_m2: float = 1.4113
    min_engine_throttle: float = 0.36
    max_restarts: int = 3
    gimbal_limit_deg: float = 6.0
    gimbal_rate_deg_s: float = 60.0
    gimbal_tau_s: float = 0.06
    nominal_engine_tau_s: float = 0.04
    rcs_isp_s: float = 260.0
    rcs_torque_limit_nm: float = 100_000.0
    rcs_roll_torque_limit_nm: float = 18_000.0

    def throttle_bounds(self, active_engines: int) -> tuple[float, float]:
        if active_engines <= 0:
            return (0.0, 0.0)
        frac = min(active_engines, self.engine_count) / self.engine_count
        return (self.min_engine_throttle * frac, frac)

    def quantize_throttle(self, requested_total: float, active_engines: int) -> float:
        if active_engines <= 0 or requested_total <= 1e-6:
            return 0.0
        low, high = self.throttle_bounds(active_engines)
        return max(low, min(high, requested_total))


@dataclass(frozen=True, slots=True)
class AeroConfig:
    axial_cd_ascent: float = 0.40
    axial_cd_descent: float = 1.65
    normal_cd: float = 0.80
    slender_lift_slope: float = 2.0
    pressure_center_ascent_m: float = 11.5
    pressure_center_descent_m: float = 11.0
    rotation_damping: float = 0.30
    roll_damping: float = 0.08
    transonic_drag_rise: float = 0.20


@dataclass(frozen=True, slots=True)
class GridFinConfig:
    count: int = 4
    surface_mass_kg: float = 360.0
    actuator_mass_kg: float = 160.0
    deployment_mass_kg: float = 60.0
    electrical_mass_kg: float = 20.0
    station_m: float = 22.4
    mean_mass_radius_m: float = 1.55
    radial_center_m: float = 2.15
    installed_area_m2: float = 6.0
    per_fin_area_m2: float = 1.5
    lift_slope_rad_inv: float = 2.8
    max_lift_coefficient: float = 1.0
    base_drag_coefficient: float = 0.35
    induced_drag_factor: float = 0.50
    deflection_limit_deg: float = 20.0
    deflection_rate_deg_s: float = 60.0
    actuator_time_constant_s: float = 0.08
    deployment_time_s: float = 2.0
    retraction_time_s: float = 2.0
    control_min_q_pa: float = 80.0
    control_max_alpha_deg: float = 25.0
    model_max_mach: float = 8.0
    hinge_arm_m: float = 0.12
    per_fin_hinge_limit_nm: float = 7_500.0

    @property
    def mass_kg(self) -> float:
        return (
            self.surface_mass_kg
            + self.actuator_mass_kg
            + self.deployment_mass_kg
            + self.electrical_mass_kg
        )


@dataclass(frozen=True, slots=True)
class LandingGearConfig:
    """4-leg 3D telescoping landing gear matching 2D vertical stiffness (4*80 = 2*160 kN/m)."""

    leg_count: int = 4
    half_span_m: float = 4.0
    foot_station_m: float = -1.0
    anchor_station_m: float = 1.2
    deployment_time_s: float = 2.0
    pad_radius_m: float = 10.0
    per_leg_stiffness_n_m: float = 80_000.0
    per_leg_damping_n_s_m: float = 22_500.0
    friction_coefficient: float = 0.75
    friction_velocity_scale_m_s: float = 0.05
    max_leg_stroke_m: float = 0.90
    max_leg_force_n: float = 250_000.0
    max_contact_load_g: float = 4.0
    settle_duration_s: float = 5.0
    max_standing_tilt_deg: float = 5.0
    max_standing_speed_m_s: float = 0.08
    max_standing_rate_deg_s: float = 0.5


@dataclass(frozen=True, slots=True)
class UpperStageConfig:
    dry_mass_kg: float = 1_550.0
    propellant_mass_kg: float = 20_650.0
    payload_mass_kg: float = 1_800.0
    vacuum_thrust_n: float = 240_000.0
    vacuum_isp_s: float = 380.0
    target_altitude_m: float = 700_000.0
    altitude_tolerance_m: float = 15_000.0
    ignition_coast_s: float = 4.0
    replay_post_sep_duration_s: float = 25.0

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg + self.payload_mass_kg


@dataclass(frozen=True, slots=True)
class FlightSafetyLimits:
    max_q_pa: float = 42_000.0
    max_heat_flux_w_m2: float = 80_000.0
    max_q_alpha_kpa_deg: float = 150.0
    max_high_q_alpha_deg: float = 5.0
    max_angular_rate_deg_s: float = 15.0
    max_angular_acceleration_deg_s2: float = 5.0
    max_ascent_axial_g: float = 4.5
    max_recovery_flight_g: float = 5.5
    max_position_error_m: float = 2.0
    max_horizontal_contact_speed_m_s: float = 0.5
    max_vertical_foot_speed_m_s: float = 1.5
    max_contact_tilt_deg: float = 3.0
    max_contact_rate_deg_s: float = 0.5


@dataclass(frozen=True, slots=True)
class AscentGateConfig:
    separation_altitude_m: float = 60_000.0
    separation_tolerance_m: float = 16_000.0
    separation_min_main_propellant_kg: float = 12_300.0
    separation_min_speed_m_s: float = 1_600.0
    separation_max_q_pa: float = 3_000.0
    separation_max_rate_deg_s: float = 5.0
    separation_max_alpha_deg: float = 5.0


@dataclass(frozen=True, slots=True)
class DM100VehicleConfig:
    root_seed: int = 378
    vehicle_name: str = "DM100-3D"
    geometry: VehicleGeometry = field(default_factory=VehicleGeometry)
    propellant: PropellantBudget = field(default_factory=PropellantBudget)
    propulsion: PropulsionConfig = field(default_factory=PropulsionConfig)
    aerodynamics: AeroConfig = field(default_factory=AeroConfig)
    fins: GridFinConfig = field(default_factory=GridFinConfig)
    landing_gear: LandingGearConfig = field(default_factory=LandingGearConfig)
    upper_stage: UpperStageConfig = field(default_factory=UpperStageConfig)
    safety: FlightSafetyLimits = field(default_factory=FlightSafetyLimits)
    ascent_gate: AscentGateConfig = field(default_factory=AscentGateConfig)

    @property
    def booster_dry_mass_kg(self) -> float:
        return (
            self.geometry.engine_mass_kg
            + self.geometry.airframe_mass_kg
            + self.geometry.gear_mass_kg
            + self.fins.mass_kg
        )

    @property
    def booster_wet_mass_kg(self) -> float:
        return (
            self.booster_dry_mass_kg
            + self.propellant.main_propellant_kg
            + self.propellant.rcs_propellant_kg
        )

    @property
    def launch_mass_kg(self) -> float:
        return self.booster_wet_mass_kg + self.upper_stage.total_mass_kg


@dataclass(frozen=True, slots=True)
class DualTankGeometry:
    """Common-bulkhead LOX (upper) + CH4 (lower) dual-tank specification (CDR Rev 3.1)."""

    tank_inner_radius_m: float = 1.50
    of_ratio: float = 3.6
    lox_density_kg_m3: float = 1_141.0
    ch4_density_kg_m3: float = 422.0
    ch4_bottom_m: float = 2.60
    lox_bottom_m: float = 9.20
    ch4_tank_volume_m3: float = 39.0
    lox_tank_volume_m3: float = 52.0
    common_bulkhead_dome_depth_m: float = 1.061
    ring_baffle_zeta0: float = 0.080
    ring_baffle_kb: float = 0.08
    ring_baffle_eps_rad: float = 1e-3

    @property
    def total_tank_volume_m3(self) -> float:
        return self.ch4_tank_volume_m3 + self.lox_tank_volume_m3

    @property
    def lox_mass_fraction(self) -> float:
        return self.of_ratio / (1.0 + self.of_ratio)

    @property
    def bulk_propellant_density_kg_m3(self) -> float:
        f_lox = self.lox_mass_fraction
        f_ch4 = 1.0 - f_lox
        return 1.0 / (f_lox / self.lox_density_kg_m3 + f_ch4 / self.ch4_density_kg_m3)


@dataclass(frozen=True, slots=True)
class PhysicsV2Config:
    """Independent toggle flags for all DM100 3D v2 high-fidelity physics modules.

    All booleans default to `False` so legacy DM100 3D v1.0 behavior is reproduced bit-for-bit.
    """

    # M1 (Phase 7A)
    earth_wgs84_j2: bool = False
    dual_tank_slosh: bool = False
    srp_interference: bool = False
    ground_effect: bool = False

    # M2/M3 (Phase 7B+)
    flex_body: bool = False
    dryden_turbulence: bool = False
    sloped_pad: bool = False
    sensors_v2: bool = False
    faults_v2: bool = False

    # Dispersion / model parameters (neutral defaults)
    srp_shield_scale: float = 1.0
    srp_buffet_enabled: bool = True
    srp_cm_rms_peak: float = 0.0055
    ground_effect_coeff: float = 0.07
    ground_effect_lat_coeff: float = 0.25
    slosh_baffle_damping_scale: float = 1.0
    booster_dry_mass_scale: float = 1.0
    dual_tank_geometry: DualTankGeometry = field(default_factory=DualTankGeometry)

    # M2 flex-body, Dryden/jet turbulence, sloped pad, and sensor v2 parameters
    flex_freq_mode1_scale: float = 1.0
    flex_freq_mode2_scale: float = 1.0
    flex_damping_scale: float = 1.0
    dryden_sigma_m_s: float = 1.5
    jet_shear_peak_m_s: float = 0.0
    jet_shear_azimuth_deg: float = 0.0
    jet_shear_alt_m: float = 10_500.0
    jet_shear_thickness_m: float = 1_800.0
    pad_slope_deg: float = 0.0
    pad_slope_azimuth_deg: float = 0.0
    pad_radius_m: float = 15.0
    leg_stiffness_scale_4: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    gps_blackout_duration_s: float = 0.0
    gps_blackout_heat_flux_w_m2: float = 10_000.0
    gps_blackout_alt_min_m: float = 35_000.0
    gps_blackout_alt_max_m: float = 65_000.0
    gyro_arw_scale: float = 1.0
    accel_vrw_scale: float = 1.0
    gyro_scale_err_ppm: float = 0.0
    accel_scale_err_ppm: float = 0.0
    gyro_nonortho_deg: float = 0.0
    accel_nonortho_deg: float = 0.0
    radar_alt_noise_scale: float = 1.0
    fin_resolver_noise_deg: float = 0.10
    prop_gauge_noise_kg: float = 25.0
    # M3 7-engine cluster dispersions, nozzle side-loads, and injected EO / Fin-Jam faults
    engine_thrust_bias_7: tuple[float, float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    engine_ign_delay_s_7: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    side_load_peak_n: float = 2_000.0
    engine_out_index: int | None = None
    engine_out_time_s: float | None = None
    engine_out_phase: str | None = None
    engine_out_phase_elapsed_s: float = 0.0
    engine_out_spindown_tau_s: float = 0.25
    fin_jam_index: int | None = None
    fin_jam_time_s: float | None = None
    fin_jam_phase: str | None = None
    fin_jam_phase_elapsed_s: float = 0.0
    fin_jam_angle_deg: float | None = None

    def any_enabled(self) -> bool:
        return any(
            (
                self.earth_wgs84_j2,
                self.dual_tank_slosh,
                self.srp_interference,
                self.ground_effect,
                self.flex_body,
                self.dryden_turbulence,
                self.sloped_pad,
                self.sensors_v2,
                self.faults_v2,
            )
        )

    @classmethod
    def m1_preset(cls) -> PhysicsV2Config:
        """Enable all Phase 7A (M1) high-fidelity physics modules."""
        return cls(
            earth_wgs84_j2=True,
            dual_tank_slosh=True,
            srp_interference=True,
            ground_effect=True,
        )

    @classmethod
    def m2_preset(cls) -> PhysicsV2Config:
        """Enable Phase 7A+7B (M1+M2): flex, Dryden, finite sloped pad, and 500->50 Hz sensors."""
        return cls(
            earth_wgs84_j2=True,
            dual_tank_slosh=True,
            srp_interference=True,
            ground_effect=True,
            flex_body=True,
            dryden_turbulence=True,
            sloped_pad=True,
            sensors_v2=True,
        )

    @classmethod
    def full_v2(cls) -> PhysicsV2Config:
        """Enable the full DM100 3D v2 high-fidelity physics stack."""
        return cls(
            earth_wgs84_j2=True,
            dual_tank_slosh=True,
            srp_interference=True,
            ground_effect=True,
            flex_body=True,
            dryden_turbulence=True,
            sloped_pad=True,
            sensors_v2=True,
            faults_v2=True,
        )


@dataclass(frozen=True, slots=True)
class GncV2Config:
    """Paired GNC v2 toggle flags (all `False` by default for bit-exact v1 legacy equivalence)."""

    # M1 causal physics alignment
    use_wgs84_gravity: bool = False
    use_dual_tank_mass_props: bool = False
    use_srp_iip_feedforward: bool = False
    use_ground_effect_feedforward: bool = False

    # M2/M3 GNC upgrades
    use_structural_filter: bool = False
    use_strapdown_mekf: bool = False
    use_online_mpc: bool = False
    use_fdir_v2: bool = False

    # Strictly causal nominal feedforward constants (never see hidden plant dispersions)
    srp_nominal_eta: float = 0.40
    ground_effect_nominal_coeff: float = 0.06
    dual_tank_geometry: DualTankGeometry = field(default_factory=DualTankGeometry)

    @classmethod
    def m1_preset(cls) -> GncV2Config:
        """Enable causal M1 physics models inside GNC/MEKF."""
        return cls(
            use_wgs84_gravity=True,
            use_dual_tank_mass_props=True,
            use_srp_iip_feedforward=True,
            use_ground_effect_feedforward=True,
        )

    @classmethod
    def m2_preset(cls) -> GncV2Config:
        """Enable causal M1+M2 GNC: Elliptic CSI filter, Matched DOB, and Strapdown MEKF v2."""
        return cls(
            use_wgs84_gravity=True,
            use_dual_tank_mass_props=True,
            use_srp_iip_feedforward=True,
            use_ground_effect_feedforward=True,
            use_structural_filter=True,
            use_strapdown_mekf=True,
        )

    @classmethod
    def full_v2(cls) -> GncV2Config:
        """Enable all DM100 3D v2 GNC upgrades."""
        return cls(
            use_wgs84_gravity=True,
            use_dual_tank_mass_props=True,
            use_srp_iip_feedforward=True,
            use_ground_effect_feedforward=True,
            use_structural_filter=True,
            use_strapdown_mekf=True,
            use_online_mpc=True,
            use_fdir_v2=True,
        )


@dataclass(frozen=True, slots=True)
class MassProperties3D:
    total_mass_kg: float
    cg_station_m: float
    inertia_diag_kg_m2: tuple[float, float, float]
    fuel_height_m: float


def dual_tank_propellant_levels(
    main_propellant_kg: float,
    geom: DualTankGeometry | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return `(m_lox, h_lox, cg_lox, m_ch4, h_ch4, cg_ch4)` for common-bulkhead LOX/CH4 tanks."""
    dtg = geom or DualTankGeometry()
    main = max(0.0, float(main_propellant_kg))
    area = pi * dtg.tank_inner_radius_m * dtg.tank_inner_radius_m
    m_lox = main * dtg.lox_mass_fraction
    m_ch4 = main - m_lox
    h_lox = m_lox / (dtg.lox_density_kg_m3 * area)
    h_ch4 = m_ch4 / (dtg.ch4_density_kg_m3 * area)
    cg_lox = dtg.lox_bottom_m + 0.5 * h_lox
    cg_ch4 = dtg.ch4_bottom_m + 0.5 * h_ch4
    return (m_lox, h_lox, cg_lox, m_ch4, h_ch4, cg_ch4)


def mass_properties_3d(
    total_mass_kg: float,
    gear_extension: float,
    *,
    vehicle: DM100VehicleConfig | None = None,
    upper_attached: bool = False,
    airframe_mass_offset_kg: float = 0.0,
    inertia_scale: float = 1.0,
    inertia_roll_scale: float | None = None,
    dual_tank: bool = False,
    rcs_propellant_kg: float | None = None,
    dual_tank_geom: DualTankGeometry | None = None,
) -> MassProperties3D:
    """Compute exact 3D mass, COM station, and diagonal inertia (Ixx, Iyy, Izz).

    When `dual_tank=False` (default), reproduces DM100 3D v1.0 (and DM100 2D in XZ) bit-for-bit.
    When `dual_tank=True`, separates upper LOX (`z=9.20 m`) and lower CH4 (`z=2.60 m`) tanks plus
    dedicated RCS propellant at `rcs_station_m=22.4 m`.
    """
    cfg = vehicle or DM100VehicleConfig()
    g = cfg.geometry
    gear = cfg.landing_gear
    fins = cfg.fins
    upper_mass = cfg.upper_stage.total_mass_kg
    roll_sc = inertia_scale if inertia_roll_scale is None else inertia_roll_scale

    airframe_mass = g.airframe_mass_kg + airframe_mass_offset_kg
    fixed_dry = (
        g.engine_mass_kg
        + airframe_mass
        + g.gear_mass_kg
        + fins.mass_kg
        + (upper_mass if upper_attached else 0.0)
    )

    if dual_tank:
        dtg = dual_tank_geom or DualTankGeometry()
        avail_prop = max(0.0, total_mass_kg - fixed_dry)
        if rcs_propellant_kg is None:
            rcs_mass = min(cfg.propellant.rcs_propellant_kg, avail_prop)
        else:
            rcs_mass = max(0.0, min(float(rcs_propellant_kg), avail_prop))
        main_prop = max(0.0, avail_prop - rcs_mass)

        r = g.body_radius_m
        r2 = r * r
        r_in = dtg.tank_inner_radius_m
        r_in2 = r_in * r_in
        m_lox, h_lox, st_lox, m_ch4, h_ch4, st_ch4 = dual_tank_propellant_levels(main_prop, dtg)

        dep = max(0.0, min(1.0, gear_extension))
        gear_st = (1.0 - dep) * g.gear_retracted_station_m + dep * g.gear_deployed_station_m
        gear_r = (1.0 - dep) * r + dep * (0.5 * gear.half_span_m)

        m_eng = g.engine_mass_kg
        s_eng = g.engine_station_m
        i_eng = m_eng * ((2.0 * s_eng) ** 2 / 12.0 + g.nozzle_radius_m**2 / 4.0)
        izz_eng = 0.5 * m_eng * (g.nozzle_radius_m**2)

        s_air = g.airframe_station_m
        i_air = airframe_mass * (g.body_length_m**2 / 12.0 + r2 / 4.0)
        izz_air = airframe_mass * r2

        m_gear = g.gear_mass_kg
        i_gear = m_gear * (gear_r**2)
        izz_gear = m_gear * (gear_r**2)

        i_lox = m_lox * (h_lox * h_lox / 12.0 + r_in2 / 4.0)
        izz_lox = 0.5 * m_lox * r_in2
        i_ch4 = m_ch4 * (h_ch4 * h_ch4 / 12.0 + r_in2 / 4.0)
        izz_ch4 = 0.5 * m_ch4 * r_in2

        s_rcs = g.rcs_station_m
        i_rcs = rcs_mass * (0.5 * r2)
        izz_rcs = rcs_mass * r2

        m_fin = fins.mass_kg
        i_fin = m_fin * (fins.mean_mass_radius_m**2)
        izz_fin = m_fin * (fins.mean_mass_radius_m**2)

        tot = m_eng + airframe_mass + m_gear + m_lox + m_ch4 + rcs_mass + m_fin
        mom = (
            m_eng * s_eng
            + airframe_mass * s_air
            + m_gear * gear_st
            + m_lox * st_lox
            + m_ch4 * st_ch4
            + rcs_mass * s_rcs
            + m_fin * fins.station_m
        )
        if upper_attached:
            tot += upper_mass
            mom += upper_mass * g.upper_station_m
        cg = mom / tot

        i_lat = (
            i_eng
            + m_eng * (s_eng - cg) ** 2
            + i_air
            + airframe_mass * (s_air - cg) ** 2
            + i_gear
            + m_gear * (gear_st - cg) ** 2
            + i_lox
            + m_lox * (st_lox - cg) ** 2
            + i_ch4
            + m_ch4 * (st_ch4 - cg) ** 2
            + i_rcs
            + rcs_mass * (s_rcs - cg) ** 2
            + i_fin
            + m_fin * (fins.station_m - cg) ** 2
        )
        izz_booster = izz_eng + izz_air + izz_gear + izz_lox + izz_ch4 + izz_rcs + izz_fin

        if upper_attached:
            i_up = upper_mass * (g.upper_length_m**2 / 12.0 + r2 / 4.0)
            izz_up = 0.5 * upper_mass * r2
            booster_tot = tot - upper_mass
            booster_cg = (mom - upper_mass * g.upper_station_m) / booster_tot
            booster_i = (
                i_eng
                + m_eng * (s_eng - booster_cg) ** 2
                + i_air
                + airframe_mass * (s_air - booster_cg) ** 2
                + i_gear
                + m_gear * (gear_st - booster_cg) ** 2
                + i_lox
                + m_lox * (st_lox - booster_cg) ** 2
                + i_ch4
                + m_ch4 * (st_ch4 - booster_cg) ** 2
                + i_rcs
                + rcs_mass * (s_rcs - booster_cg) ** 2
                + i_fin
                + m_fin * (fins.station_m - booster_cg) ** 2
            )
            i_lat = (
                i_lat
                + (inertia_scale - 1.0) * booster_i
                + i_up
                + upper_mass * (g.upper_station_m - cg) ** 2
            )
            i_roll = roll_sc * izz_booster + izz_up
        else:
            i_lat *= inertia_scale
            i_roll = roll_sc * izz_booster

        return MassProperties3D(
            total_mass_kg=float(tot),
            cg_station_m=float(cg),
            inertia_diag_kg_m2=(float(i_lat), float(i_lat), float(i_roll)),
            fuel_height_m=float(h_lox + h_ch4),
        )

    # Exact legacy v1 homogeneous-tank branch (bit-for-bit untouched)
    fuel = max(0.0, total_mass_kg - fixed_dry)

    r = g.body_radius_m
    r2 = r * r
    fuel_h = fuel / (g.propellant_density_kg_m3 * pi * r2)
    fuel_st = g.tank_bottom_m + 0.5 * fuel_h
    dep = max(0.0, min(1.0, gear_extension))
    gear_st = (1.0 - dep) * g.gear_retracted_station_m + dep * g.gear_deployed_station_m
    gear_r = (1.0 - dep) * r + dep * (0.5 * gear.half_span_m)

    m_eng = g.engine_mass_kg
    s_eng = g.engine_station_m
    i_eng = m_eng * ((2.0 * s_eng) ** 2 / 12.0 + g.nozzle_radius_m**2 / 4.0)
    izz_eng = 0.5 * m_eng * (g.nozzle_radius_m**2)

    s_air = g.airframe_station_m
    i_air = airframe_mass * (g.body_length_m**2 / 12.0 + r2 / 4.0)
    izz_air = airframe_mass * r2  # thin-walled cylindrical shell

    m_gear = g.gear_mass_kg
    i_gear = m_gear * (gear_r**2)
    izz_gear = m_gear * (gear_r**2)

    i_fuel = fuel * (fuel_h * fuel_h / 12.0 + r2 / 4.0)
    izz_fuel = 0.5 * fuel * r2  # solid propellant column

    m_fin = fins.mass_kg
    i_fin = m_fin * (fins.mean_mass_radius_m**2)
    izz_fin = m_fin * (fins.mean_mass_radius_m**2)

    tot = m_eng + airframe_mass + m_gear + fuel + m_fin
    mom = (
        m_eng * s_eng
        + airframe_mass * s_air
        + m_gear * gear_st
        + fuel * fuel_st
        + m_fin * fins.station_m
    )
    if upper_attached:
        tot += upper_mass
        mom += upper_mass * g.upper_station_m
    cg = mom / tot

    i_lat = (
        i_eng
        + m_eng * (s_eng - cg) ** 2
        + i_air
        + airframe_mass * (s_air - cg) ** 2
        + i_gear
        + m_gear * (gear_st - cg) ** 2
        + i_fuel
        + fuel * (fuel_st - cg) ** 2
        + i_fin
        + m_fin * (fins.station_m - cg) ** 2
    )
    izz_booster = izz_eng + izz_air + izz_gear + izz_fuel + izz_fin

    if upper_attached:
        i_up = upper_mass * (g.upper_length_m**2 / 12.0 + r2 / 4.0)
        izz_up = 0.5 * upper_mass * r2
        booster_tot = tot - upper_mass
        booster_cg = (mom - upper_mass * g.upper_station_m) / booster_tot
        booster_i = (
            i_eng
            + m_eng * (s_eng - booster_cg) ** 2
            + i_air
            + airframe_mass * (s_air - booster_cg) ** 2
            + i_gear
            + m_gear * (gear_st - booster_cg) ** 2
            + i_fuel
            + fuel * (fuel_st - booster_cg) ** 2
            + i_fin
            + m_fin * (fins.station_m - booster_cg) ** 2
        )
        i_lat = (
            i_lat
            + (inertia_scale - 1.0) * booster_i
            + i_up
            + upper_mass * (g.upper_station_m - cg) ** 2
        )
        i_roll = roll_sc * izz_booster + izz_up
    else:
        i_lat *= inertia_scale
        i_roll = roll_sc * izz_booster

    return MassProperties3D(
        total_mass_kg=float(tot),
        cg_station_m=float(cg),
        inertia_diag_kg_m2=(float(i_lat), float(i_lat), float(i_roll)),
        fuel_height_m=float(fuel_h),
    )


def load_vehicle_config(path: str | Path | None = None) -> DM100VehicleConfig:
    """Load DM100VehicleConfig from YAML (defaults to configs/vehicle_dm100.yaml)."""
    if path is None:
        return DM100VehicleConfig()
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Vehicle config YAML must have schema_version: 1")
    return DM100VehicleConfig(
        root_seed=int(raw.get("root_seed", 378)),
        vehicle_name=str(raw.get("vehicle_name", "DM100-3D")),
        geometry=VehicleGeometry(**raw.get("geometry", {})),
        propellant=PropellantBudget(**raw.get("propellant", {})),
        propulsion=PropulsionConfig(**raw.get("propulsion", {})),
        aerodynamics=AeroConfig(**raw.get("aerodynamics", {})),
        fins=GridFinConfig(**raw.get("fins", {})),
        landing_gear=LandingGearConfig(**raw.get("landing_gear", {})),
        upper_stage=UpperStageConfig(**raw.get("upper_stage", {})),
        safety=FlightSafetyLimits(**raw.get("safety", {})),
        ascent_gate=AscentGateConfig(**raw.get("ascent_gate", {})),
    )
