"""Deterministic 3D 6DOF validation suites (edge-3d, holdout-3d, full-3d) for Kaggle offload.

Covers the full 3D operating envelope:
  - East/West/North/South winds (+/-8 m/s ground, +/-20 m/s aloft)
  - 3D directional wind shear (orthogonal & opposing X/Y altitude profiles)
  - 3D directional gusts (ascent Max-Q, entry burn, aero glide, terminal landing)
  - External 3D roll torque disturbances (in coast & glide)
  - Hidden mass/thrust/drag/density/inertia dispersions
  - Equator (max Coriolis) and South Pole (zero Earth rotation boost)
  - Noisy 15-State MEKF + gyro/accel bias + delays + packet dropouts
"""

from __future__ import annotations

from dataclasses import replace
from math import cos, pi, radians, sin

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from dm100_3d.config import G0, PhysicsV2Config
from dm100_3d.gnc.mekf import SensorConfig3D
from dm100_3d.physics import Atmosphere3DSettings, Wind3DConfig
from dm100_3d.sim_3d import Disturbances3D, Scenario3D


def _prof(
    ground: float, aloft: float, err: float = 0.0
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
]:
    actual = ((0.0, ground), (50.0, ground), (10_000.0, aloft), (100_000.0, aloft))
    fc_g = max(-8.0, min(8.0, ground + err))
    fc_a = max(-20.0, min(20.0, aloft + err))
    forecast = ((0.0, fc_g), (50.0, fc_g), (10_000.0, fc_a), (100_000.0, fc_a))
    return actual, forecast


def _prof_unclipped(
    ground: float, aloft: float, err: float = 0.0
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
]:
    """Unclipped actual & forecast wind profile for untruncated Normal M4 campaigns."""
    actual = ((0.0, ground), (50.0, ground), (10_000.0, aloft), (100_000.0, aloft))
    forecast = (
        (0.0, ground + err),
        (50.0, ground + err),
        (10_000.0, aloft + err),
        (100_000.0, aloft + err),
    )
    return actual, forecast


def build_edge_3d_suite() -> list[Scenario3D]:
    """32 adversarial 3D boundary & combined worst-corner scenarios."""
    cases: list[Scenario3D] = []

    def add(
        name: str,
        *,
        gx: float = 0.0,
        ax: float = 0.0,
        ex: float = 0.0,
        gy: float = 0.0,
        ay: float = 0.0,
        ey: float = 0.0,
        wz: float = 0.0,
        gust_amp: float = 0.0,
        gust_az: float = 0.0,
        gust_start: float = 280.0,
        gust_alt: float = 8_000.0,
        lat: float = 37.0,
        rho_scale: float = 1.0,
        temp_k: float = 288.15,
        press_pa: float = 101_325.0,
        thrust_scale: float = 1.0,
        drag_scale: float = 1.0,
        mass_scale: float = 1.0,
        inertia_scale: float = 1.0,
        engine_lag: float = 0.0,
        roll_nm: float = 0.0,
        roll_start: float = 320.0,
        sensors: SensorConfig3D | None = None,
        seed: int = 378,
    ) -> None:
        wx_a, wx_f = _prof(gx, ax, ex)
        wy_a, wy_f = _prof(gy, ay, ey)
        cases.append(
            Scenario3D(
                name=name,
                atmosphere=Atmosphere3DSettings(
                    sea_level_temperature_k=temp_k,
                    sea_level_pressure_pa=press_pa,
                    latitude_deg=lat,
                    coriolis_enabled=True,
                    density_scale=rho_scale,
                ),
                wind=Wind3DConfig(
                    wind_x_profile=wx_a,
                    wind_y_profile=wy_a,
                    wind_z_profile=((0.0, wz), (100_000.0, wz)),
                    forecast_wind_x_profile=wx_f,
                    forecast_wind_y_profile=wy_f,
                    gust_amplitude_m_s=gust_amp,
                    gust_azimuth_rad=gust_az,
                    gust_start_s=gust_start,
                    gust_duration_s=100.0,
                    gust_altitude_center_m=gust_alt,
                    gust_altitude_half_width_m=4_000.0,
                ),
                disturbances=Disturbances3D(
                    thrust_scale=thrust_scale,
                    drag_scale=drag_scale,
                    initial_mass_scale=mass_scale,
                    inertia_scale=inertia_scale,
                    engine_lag_s=engine_lag,
                    roll_disturbance_torque_nm=roll_nm,
                    roll_disturbance_start_s=roll_start,
                    roll_disturbance_duration_s=4.5,
                ),
                sensors=sensors or SensorConfig3D(),
                seed=seed,
            )
        )

    nom_imu = SensorConfig3D(
        position_noise_m=0.5,
        velocity_noise_m_s=0.15,
        attitude_noise_deg=0.05,
        gyro_noise_deg_s=0.05,
        gyro_bias_deg_s=(0.02, -0.02, 0.02),
        dropout_rate=0.05,
    )
    harsh_imu = SensorConfig3D(
        position_noise_m=1.0,
        velocity_noise_m_s=0.25,
        attitude_noise_deg=0.08,
        gyro_noise_deg_s=0.06,
        gyro_bias_deg_s=(0.03, -0.03, 0.03),
        dropout_rate=0.08,
        observation_delay_steps=1,
        action_delay_steps=1,
    )

    # Hero & cardinal wind axes
    add("edge3d-01-calm-nominal")
    add("edge3d-02-max-east-wind", gx=8.0, ax=18.0)
    add("edge3d-03-max-west-wind", gx=-8.0, ax=-18.0)
    add("edge3d-04-max-north-crosswind", gy=8.0, ay=18.0)
    add("edge3d-05-max-south-crosswind", gy=-8.0, ay=-18.0)
    add("edge3d-06-diagonal-ne-wind", gx=6.0, ax=14.0, gy=6.0, ay=14.0)
    add("edge3d-07-diagonal-sw-wind", gx=-6.0, ax=-14.0, gy=-6.0, ay=-14.0)

    # 3D directional wind shears
    add("edge3d-08-shear-east-to-north", gx=7.0, ax=-12.0, gy=-6.0, ay=16.0)
    add("edge3d-09-shear-west-to-south", gx=-7.0, ax=12.0, gy=6.0, ay=-16.0)
    add("edge3d-10-opposing-cross-shear", gx=6.0, ax=-16.0, gy=-7.0, ay=16.0, ex=1.5, ey=-1.5)
    add("edge3d-11-updraft-crosswind", gx=4.0, ax=10.0, gy=5.0, ay=12.0, wz=2.5, ex=1.5, ey=1.5)
    add(
        "edge3d-12-downdraft-crosswind",
        gx=-4.0,
        ax=-10.0,
        gy=-5.0,
        ay=-12.0,
        wz=-2.5,
        ex=-1.5,
        ey=-1.5,
    )

    # 3D directional gusts across flight phases
    add(
        "edge3d-13-gust-ascent-maxq",
        gx=4.0,
        ax=10.0,
        gy=4.0,
        ay=10.0,
        gust_amp=4.5,
        gust_az=0.8,
        gust_start=30.0,
        gust_alt=10_000.0,
    )
    add(
        "edge3d-14-gust-entry-burn",
        gx=-4.0,
        ax=-12.0,
        gy=4.0,
        ay=12.0,
        gust_amp=4.5,
        gust_az=1.3,
        gust_start=390.0,
        gust_alt=22_000.0,
    )
    add(
        "edge3d-15-gust-aero-glide",
        gx=5.0,
        ax=12.0,
        gy=-5.0,
        ay=-12.0,
        gust_amp=4.5,
        gust_az=2.1,
        gust_start=450.0,
        gust_alt=8_000.0,
    )
    add(
        "edge3d-16-gust-terminal-pad",
        gx=-4.0,
        ax=-10.0,
        gy=4.0,
        ay=10.0,
        gust_amp=3.5,
        gust_az=1.57,
        gust_start=510.0,
        gust_alt=1_500.0,
    )

    # Roll disturbances (vacuum coast + high-q glide)
    add("edge3d-17-roll-pulse-coast-pos", roll_nm=2_500.0, roll_start=320.0)
    add("edge3d-18-roll-pulse-coast-neg", roll_nm=-2_500.0, roll_start=320.0)
    add("edge3d-19-roll-pulse-glide-pos", gy=5.0, ay=12.0, roll_nm=2_500.0, roll_start=465.0)
    add("edge3d-20-roll-pulse-glide-neg", gy=-5.0, ay=-12.0, roll_nm=-2_500.0, roll_start=465.0)

    # Propulsion, mass, drag, density, and latitude extremes
    add(
        "edge3d-21-heavy-low-thrust",
        mass_scale=1.008,
        thrust_scale=0.975,
        rho_scale=1.06,
        gy=4.0,
        ay=10.0,
    )
    add(
        "edge3d-22-light-high-thrust",
        mass_scale=0.992,
        thrust_scale=1.025,
        rho_scale=0.94,
        gy=-4.0,
        ay=-10.0,
    )
    add(
        "edge3d-23-maxq-cold-dense",
        temp_k=273.15,
        press_pa=104_500.0,
        rho_scale=1.08,
        drag_scale=0.90,
        gx=5.0,
        ax=14.0,
    )
    add(
        "edge3d-24-thin-hot-low-drag",
        temp_k=308.15,
        press_pa=96_000.0,
        rho_scale=0.92,
        drag_scale=0.88,
        gy=5.0,
        ay=12.0,
    )
    add("edge3d-25-equator-coriolis", lat=0.0, gx=5.0, ax=12.0, gy=5.0, ay=12.0)
    add("edge3d-26-south-pole-zero-rot", lat=-89.0, gx=-4.0, ax=-10.0, gy=4.0, ay=10.0)

    # Sensor noise, bias, dropout, and combined 3D nightmares
    add(
        "edge3d-27-noisy-mekf-dropouts",
        gx=4.0,
        ax=10.0,
        gy=-4.0,
        ay=-10.0,
        sensors=nom_imu,
        seed=401,
    )
    add(
        "edge3d-28-harsh-imu-delays",
        gx=-4.0,
        ax=-10.0,
        gy=4.0,
        ay=10.0,
        engine_lag=0.04,
        sensors=harsh_imu,
        seed=402,
    )
    add(
        "edge3d-29-combined-crosswind-roll-imu",
        gx=5.0,
        ax=12.0,
        ex=-1.0,
        gy=-4.0,
        ay=10.0,
        ey=1.0,
        gust_amp=3.0,
        gust_az=1.1,
        gust_start=280.0,
        rho_scale=1.03,
        thrust_scale=0.99,
        drag_scale=1.04,
        roll_nm=2_500.0,
        sensors=nom_imu,
        seed=403,
    )
    add(
        "edge3d-30-combined-heavy-shear-nightmare",
        gx=-6.0,
        ax=14.0,
        ex=1.2,
        gy=6.0,
        ay=-14.0,
        ey=-1.2,
        mass_scale=1.006,
        thrust_scale=0.98,
        inertia_scale=1.02,
        rho_scale=1.04,
        roll_nm=-2_000.0,
        sensors=nom_imu,
        seed=404,
    )
    add(
        "edge3d-31-combined-light-thin-nightmare",
        gx=6.0,
        ax=-14.0,
        ex=-1.2,
        gy=-6.0,
        ay=14.0,
        ey=1.2,
        mass_scale=0.994,
        thrust_scale=1.02,
        drag_scale=0.90,
        rho_scale=0.95,
        roll_nm=2_000.0,
        sensors=nom_imu,
        seed=405,
    )
    add(
        "edge3d-32-combined-pole-gust-roll",
        lat=-75.0,
        gx=5.0,
        ax=12.0,
        gy=-5.0,
        ay=-12.0,
        gust_amp=3.5,
        gust_az=2.4,
        gust_start=440.0,
        gust_alt=7_000.0,
        roll_nm=-2_200.0,
        roll_start=310.0,
        sensors=nom_imu,
        seed=406,
    )
    return cases


def build_holdout_3d_suite(count: int = 64, *, root_seed: int = 378) -> list[Scenario3D]:
    """Deterministic seeded 3D Monte Carlo holdout suite."""
    rng = np.random.default_rng(root_seed + 9_000)
    cases: list[Scenario3D] = []
    for idx in range(count):
        gx = float(rng.uniform(-6.5, 6.5))
        ax = float(rng.uniform(-16.0, 16.0))
        ex = float(rng.uniform(-1.5, 1.5))
        gy = float(rng.uniform(-6.5, 6.5))
        ay = float(rng.uniform(-16.0, 16.0))
        ey = float(rng.uniform(-1.5, 1.5))
        wz = float(rng.uniform(-2.0, 2.0))
        wx_a, wx_f = _prof(gx, ax, ex)
        wy_a, wy_f = _prof(gy, ay, ey)

        use_imu = idx % 2 == 1
        sensors = (
            SensorConfig3D(
                position_noise_m=float(rng.uniform(0.2, 0.8)),
                velocity_noise_m_s=float(rng.uniform(0.05, 0.22)),
                attitude_noise_deg=float(rng.uniform(0.02, 0.06)),
                gyro_noise_deg_s=float(rng.uniform(0.02, 0.05)),
                gyro_bias_deg_s=(
                    float(rng.uniform(-0.025, 0.025)),
                    float(rng.uniform(-0.025, 0.025)),
                    float(rng.uniform(-0.025, 0.025)),
                ),
                dropout_rate=float(rng.uniform(0.0, 0.06)),
                observation_delay_steps=int(rng.integers(0, 2)),
            )
            if use_imu
            else SensorConfig3D()
        )
        cases.append(
            Scenario3D(
                name=f"holdout3d-{idx + 1:03d}",
                atmosphere=Atmosphere3DSettings(
                    sea_level_temperature_k=float(rng.uniform(276.0, 305.0)),
                    sea_level_pressure_pa=float(rng.uniform(97_000.0, 104_000.0)),
                    latitude_deg=float(rng.uniform(-75.0, 75.0)),
                    coriolis_enabled=True,
                    density_scale=float(rng.uniform(0.94, 1.06)),
                ),
                wind=Wind3DConfig(
                    wind_x_profile=wx_a,
                    wind_y_profile=wy_a,
                    wind_z_profile=((0.0, wz), (100_000.0, wz)),
                    forecast_wind_x_profile=wx_f,
                    forecast_wind_y_profile=wy_f,
                    gust_amplitude_m_s=float(rng.uniform(0.0, 4.0)),
                    gust_azimuth_rad=float(rng.uniform(0.0, 2.0 * pi)),
                    gust_start_s=float(rng.uniform(25.0, 480.0)),
                    gust_duration_s=90.0,
                    gust_altitude_center_m=float(rng.uniform(2_000.0, 20_000.0)),
                ),
                disturbances=Disturbances3D(
                    thrust_scale=float(rng.uniform(0.98, 1.02)),
                    drag_scale=float(rng.uniform(0.90, 1.10)),
                    initial_mass_scale=float(rng.uniform(0.993, 1.007)),
                    inertia_scale=float(rng.uniform(0.98, 1.02)),
                    engine_lag_s=float(rng.uniform(0.0, 0.03)),
                    roll_disturbance_torque_nm=float(rng.uniform(-2_200.0, 2_200.0))
                    if (idx % 3 == 0)
                    else 0.0,
                    roll_disturbance_start_s=float(rng.uniform(260.0, 460.0)),
                    roll_disturbance_duration_s=4.0,
                ),
                sensors=sensors,
                seed=root_seed + 1_000 + idx,
            )
        )
    return cases


def build_edge_v2_suite() -> list[Scenario3D]:
    """48-case Tier-2 v2 boundary & fault suite: 32 v1 edge cases + 16 v2 extreme/fault cases."""
    cases = build_edge_3d_suite()
    base_v2 = PhysicsV2Config.full_v2()

    nom_imu = SensorConfig3D(
        position_noise_m=0.5,
        velocity_noise_m_s=0.15,
        attitude_noise_deg=0.05,
        gyro_noise_deg_s=0.05,
        gyro_bias_deg_s=(0.02, -0.02, 0.02),
        dropout_rate=0.05,
    )
    harsh_imu = SensorConfig3D(
        position_noise_m=1.0,
        velocity_noise_m_s=0.25,
        attitude_noise_deg=0.08,
        gyro_noise_deg_s=0.06,
        gyro_bias_deg_s=(0.03, -0.03, 0.03),
        dropout_rate=0.08,
        observation_delay_steps=1,
        action_delay_steps=1,
    )

    def add_v2(
        name: str,
        *,
        gx: float = 0.0,
        ax: float = 0.0,
        ex: float = 0.0,
        gy: float = 0.0,
        ay: float = 0.0,
        ey: float = 0.0,
        lat: float = 37.0,
        rho_scale: float = 1.0,
        temp_k: float = 288.15,
        press_pa: float = 101_325.0,
        thrust_scale: float = 1.0,
        drag_scale: float = 1.0,
        sensors: SensorConfig3D | None = None,
        seed: int = 500,
        expected_mission_mode: str = "RTLS",
        **pv2_overrides: object,
    ) -> None:
        wx_a, wx_f = _prof(gx, ax, ex)
        wy_a, wy_f = _prof(gy, ay, ey)
        pv2 = replace(base_v2, **pv2_overrides)  # type: ignore[arg-type]
        cases.append(
            Scenario3D(
                name=name,
                atmosphere=Atmosphere3DSettings(
                    sea_level_temperature_k=temp_k,
                    sea_level_pressure_pa=press_pa,
                    latitude_deg=lat,
                    coriolis_enabled=True,
                    density_scale=rho_scale,
                ),
                wind=Wind3DConfig(
                    wind_x_profile=wx_a,
                    wind_y_profile=wy_a,
                    wind_z_profile=((0.0, 0.0), (100_000.0, 0.0)),
                    forecast_wind_x_profile=wx_f,
                    forecast_wind_y_profile=wy_f,
                ),
                disturbances=Disturbances3D(
                    thrust_scale=thrust_scale,
                    drag_scale=drag_scale,
                ),
                sensors=sensors or nom_imu,
                seed=seed,
                physics_v2=pv2,
                expected_mission_mode=expected_mission_mode,
            )
        )

    # 8 extreme v2 environment/structure/sensor/pole cases
    add_v2(
        "edgev2-33-flex-minus10-maxq",
        gx=5.0,
        ax=14.0,
        gy=4.0,
        ay=10.0,
        temp_k=275.15,
        press_pa=104_000.0,
        rho_scale=1.06,
        flex_freq_mode1_scale=0.90,
        flex_freq_mode2_scale=0.90,
        seed=501,
    )
    add_v2(
        "edgev2-34-flex-plus10-dryden",
        gx=-5.0,
        ax=12.0,
        gy=5.0,
        ay=-12.0,
        flex_freq_mode1_scale=1.10,
        flex_freq_mode2_scale=1.10,
        dryden_sigma_m_s=2.5,
        seed=502,
    )
    add_v2(
        "edgev2-35-srp-max-shield-buffet",
        temp_k=305.15,
        press_pa=97_000.0,
        rho_scale=0.94,
        drag_scale=0.90,
        gy=5.0,
        ay=12.0,
        srp_shield_scale=1.15,
        srp_cm_rms_peak=0.0075,
        seed=503,
    )
    add_v2(
        "edgev2-36-blackout-30s-harsh",
        gx=4.0,
        ax=10.0,
        gy=5.0,
        ay=12.0,
        sensors=harsh_imu,
        gps_blackout_duration_s=30.0,
        seed=504,
    )
    add_v2(
        "edgev2-37-sloped-pad-2p5-east",
        gx=6.0,
        ax=14.0,
        pad_slope_deg=2.5,
        pad_slope_azimuth_deg=0.0,
        seed=505,
    )
    add_v2(
        "edgev2-38-sloped-pad-2p5-diag-legs",
        gx=-5.0,
        ax=-12.0,
        gy=5.0,
        ay=12.0,
        pad_slope_deg=2.5,
        pad_slope_azimuth_deg=135.0,
        leg_stiffness_scale_4=(0.95, 1.05, 0.96, 1.04),
        seed=506,
    )
    # CDR §2.6: mean 18 m/s jet reflected in forecast profile + 2.0 m/s unforecast residual +
    # 2.5 m/s continuous Dryden turbulence (20.0 m/s total true jet-stream peak at 10.5 km).
    wx39_a = ((0.0, 5.0), (5_000.0, 8.0), (10_500.0, 19.0), (16_000.0, 10.0), (100_000.0, 4.0))
    wx39_f = ((0.0, 5.0), (5_000.0, 8.0), (10_500.0, 19.0), (16_000.0, 10.0), (100_000.0, 4.0))
    wy39_a = ((0.0, -5.0), (5_000.0, -8.0), (10_500.0, 5.6), (16_000.0, -10.0), (100_000.0, -4.0))
    wy39_f = ((0.0, -5.0), (5_000.0, -8.0), (10_500.0, 5.6), (16_000.0, -10.0), (100_000.0, -4.0))
    cases.append(
        Scenario3D(
            name="edgev2-39-severe-dryden-jet-shear",
            atmosphere=Atmosphere3DSettings(latitude_deg=37.0, coriolis_enabled=True),
            wind=Wind3DConfig(
                wind_x_profile=wx39_a,
                wind_y_profile=wy39_a,
                wind_z_profile=((0.0, 0.0), (100_000.0, 0.0)),
                forecast_wind_x_profile=wx39_f,
                forecast_wind_y_profile=wy39_f,
            ),
            disturbances=Disturbances3D(),
            sensors=nom_imu,
            seed=507,
            physics_v2=replace(
                base_v2,
                dryden_sigma_m_s=2.5,
                jet_shear_peak_m_s=2.0,
                jet_shear_azimuth_deg=60.0,
            ),
            expected_mission_mode="RTLS",
        )
    )
    add_v2(
        "edgev2-40-north-pole-89deg-j2",
        lat=89.0,
        gx=4.0,
        ax=10.0,
        gy=-4.0,
        ay=-10.0,
        pad_slope_deg=1.5,
        pad_slope_azimuth_deg=45.0,
        seed=508,
    )

    # 4 Engine-Out FDIR cases (early Abort, late RTLS, entry burn, landing burn)
    add_v2(
        "edgev2-41-eo-ascent-early-abort",
        gx=3.0,
        ax=8.0,
        engine_out_index=1,
        engine_out_time_s=35.0,
        engine_out_spindown_tau_s=0.25,
        expected_mission_mode="DOWNRANGE_SPLASHDOWN_ABORT",
        seed=509,
    )
    add_v2(
        "edgev2-42-eo-ascent-late-rtls",
        gx=4.0,
        ax=10.0,
        gy=-3.0,
        ay=-8.0,
        engine_out_index=2,
        engine_out_time_s=92.0,
        engine_out_spindown_tau_s=0.25,
        expected_mission_mode="RTLS",
        seed=510,
    )
    add_v2(
        "edgev2-43-eo-entry-burn",
        gx=-4.0,
        ax=-10.0,
        gy=4.0,
        ay=10.0,
        engine_out_index=1,
        engine_out_phase="entry_burn",
        engine_out_phase_elapsed_s=4.0,
        engine_out_spindown_tau_s=0.25,
        seed=511,
    )
    add_v2(
        "edgev2-44-eo-landing-burn",
        gx=4.0,
        ax=10.0,
        gy=4.0,
        ay=10.0,
        engine_out_index=1,
        engine_out_phase="landing_burn",
        engine_out_phase_elapsed_s=1.0,
        engine_out_spindown_tau_s=0.25,
        seed=512,
    )

    # 4 Grid-Fin Jam FDIR cases (+/-6..10 deg across pitch/yaw fins during glide)
    add_v2(
        "edgev2-45-fin-jam-pos6-glide",
        gx=4.0,
        ax=10.0,
        fin_jam_index=2,
        fin_jam_phase="aero_glide",
        fin_jam_phase_elapsed_s=5.0,
        fin_jam_angle_deg=6.0,
        seed=513,
    )
    add_v2(
        "edgev2-46-fin-jam-neg8-glide",
        gy=5.0,
        ay=12.0,
        fin_jam_index=0,
        fin_jam_phase="aero_glide",
        fin_jam_phase_elapsed_s=5.0,
        fin_jam_angle_deg=-8.0,
        seed=514,
    )
    add_v2(
        "edgev2-47-fin-jam-pos10-glide",
        gx=-4.0,
        ax=-10.0,
        gy=4.0,
        ay=10.0,
        fin_jam_index=3,
        fin_jam_phase="aero_glide",
        fin_jam_phase_elapsed_s=6.0,
        fin_jam_angle_deg=10.0,
        seed=515,
    )
    add_v2(
        "edgev2-48-fin-jam-neg10-slope-bo",
        gx=4.0,
        ax=10.0,
        gy=-4.0,
        ay=-10.0,
        fin_jam_index=1,
        fin_jam_phase="aero_glide",
        fin_jam_phase_elapsed_s=6.0,
        fin_jam_angle_deg=-10.0,
        pad_slope_deg=2.0,
        pad_slope_azimuth_deg=90.0,
        gps_blackout_duration_s=20.0,
        seed=516,
    )
    return cases


# ---------------------------------------------------------------------------
# Phase 9 (M4) Suites: mc-1000 (54-scalar 10x100 Replicated LHS),
# fault-mc-160 (4x40 stratified FDIR), and envelope-boundary (4x5x5 = 100)
# ---------------------------------------------------------------------------

MC_1000_SCALAR_NAMES: tuple[str, ...] = (
    # 1-10: Weather / Atmosphere (10)
    "wind_ground",
    "wind_aloft_jet",
    "wind_forecast_err",
    "wind_vertical",
    "gust_amp",
    "wind_azimuth_deg",
    "dryden_sigma_m_s",
    "surface_temp_offset_k",
    "surface_pressure_offset_pa",
    "upper_air_density_scale",
    # 11-29: Vehicle / Propulsion / Structure / Fluid (19)
    "booster_dry_mass_scale",
    "propellant_load_scale",
    "inertia_lat_scale",
    "inertia_roll_scale",
    "engine_thrust_bias_0",
    "engine_thrust_bias_1",
    "engine_thrust_bias_2",
    "engine_thrust_bias_3",
    "engine_thrust_bias_4",
    "engine_thrust_bias_5",
    "engine_thrust_bias_6",
    "thrust_global_scale",
    "drag_scale",
    "engine_ignition_delay_s",
    "flex_freq_mode1_scale",
    "flex_freq_mode2_scale",
    "slosh_baffle_damping_scale",
    "srp_shield_scale",
    "ground_effect_scale",
    # 30-54: Navigation / Sensors / Comm / Pad / Legs (25)
    "gps_pos_noise_scale",
    "gps_vel_noise_scale",
    "radar_alt_noise_scale",
    "gyro_bias_x_deg_s",
    "gyro_bias_y_deg_s",
    "gyro_bias_z_deg_s",
    "accel_bias_x_mg",
    "accel_bias_y_mg",
    "accel_bias_z_mg",
    "gyro_arw_scale",
    "accel_vrw_scale",
    "gyro_scale_err_ppm",
    "accel_scale_err_ppm",
    "gyro_nonortho_deg",
    "accel_nonortho_deg",
    "packet_drop_prob",
    "comm_delay_steps",
    "gps_blackout_duration_s",
    "roll_disturbance_pulse_nm",
    "pad_slope_deg",
    "pad_slope_azimuth_deg",
    "leg_stiffness_scale_0",
    "leg_stiffness_scale_1",
    "leg_stiffness_scale_2",
    "leg_stiffness_scale_3",
)


def _lhs_uniform(n: int, d: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate exact (n, d) Latin Hypercube uniform strata in (0, 1)."""
    u = np.empty((n, d), dtype=np.float64)
    eps = 1e-6
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.uniform(eps, 1.0 - eps, size=n)) / float(n)
    return u


def _iman_conover_pair(
    driver: NDArray[np.float64],
    target_marginal: NDArray[np.float64],
    rho: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Reorder `target_marginal` via Iman-Conover van der Waerden scores for exact correlation."""
    n = len(driver)
    # Van der Waerden normal scores of the driver variable
    order_d = np.argsort(driver, kind="mergesort")
    ranks_d = np.empty(n, dtype=np.float64)
    ranks_d[order_d] = np.arange(1, n + 1, dtype=np.float64)
    s_d = norm.ppf(ranks_d / (n + 1.0))
    s_d = (s_d - float(np.mean(s_d))) / max(float(np.std(s_d)), 1e-12)

    # Independent normal score vector orthogonalized against s_d
    z_ind = norm.ppf((rng.permutation(n) + 0.5) / float(n))
    z_ind = z_ind - float(np.mean(z_ind))
    z_orth = z_ind - (float(np.dot(z_ind, s_d)) / max(float(np.dot(s_d, s_d)), 1e-12)) * s_d
    z_orth = z_orth / max(float(np.std(z_orth)), 1e-12)

    score = rho * s_d + float(np.sqrt(max(0.0, 1.0 - rho * rho))) * z_orth
    sorted_target = np.sort(target_marginal)
    reordered = np.empty(n, dtype=np.float64)
    reordered[np.argsort(score, kind="mergesort")] = sorted_target
    return reordered


def sample_mc_1000_lhs_matrix(
    *,
    n_replicates: int = 10,
    per_replicate: int = 100,
    root_seed: int = 202604,
) -> NDArray[np.float64]:
    """Generate the exact CDR §4.2 54-scalar `10 x 100` Replicated LHS matrix.

    Preserves exact untransformed/untruncated marginals (with physical floors where specified)
    and induces Iman-Conover rank correlations:
      - rho(dryden_sigma, |wind_ground|) = +0.65
      - rho(T0, P0)                      = -0.35
      - rho(I_lat, dry_mass)             = +0.70
      - rho(I_roll, dry_mass)            = +0.70
      - rho(f1, f2)                      = +0.80
    """
    n_dim = len(MC_1000_SCALAR_NAMES)
    assert n_dim == 54
    root_ss = np.random.SeedSequence(int(root_seed))
    rep_seeds = root_ss.spawn(n_replicates)
    blocks: list[NDArray[np.float64]] = []

    for r_ss in rep_seeds:
        rng = np.random.default_rng(r_ss)
        u = _lhs_uniform(per_replicate, n_dim, rng)
        x = np.empty((per_replicate, n_dim), dtype=np.float64)

        # 1-5: Weather / Wind
        x[:, 0] = norm.ppf(u[:, 0]) * 2.67  # wind_ground [m/s]
        x[:, 1] = norm.ppf(u[:, 1]) * 6.67  # wind_aloft_jet [m/s]
        x[:, 2] = norm.ppf(u[:, 2]) * 0.67  # wind_forecast_err [m/s]
        x[:, 3] = norm.ppf(u[:, 3]) * 0.83  # wind_vertical [m/s]
        x[:, 4] = norm.ppf(0.5 + 0.5 * u[:, 4]) * 1.67  # gust_amp Half-Normal >= 0

        # 6-7: Wind azimuth & Dryden sigma (rho=+0.65 with |wind_ground|)
        x[:, 5] = u[:, 5] * 360.0
        dryden_raw = 1.0 + 1.5 * u[:, 6]
        x[:, 6] = _iman_conover_pair(np.abs(x[:, 0]), dryden_raw, 0.65, rng)

        # 8-10: Surface T0, P0 (rho=-0.35), and Upper-Air density (h > 10 km only)
        x[:, 7] = norm.ppf(u[:, 7]) * 6.0  # surface_temp_offset_k
        p0_raw = norm.ppf(u[:, 8]) * 1_200.0  # surface_pressure_offset_pa
        x[:, 8] = _iman_conover_pair(x[:, 7], p0_raw, -0.35, rng)
        x[:, 9] = np.maximum(0.85, 1.0 + norm.ppf(u[:, 9]) * 0.012)

        # 11-14: Dry mass, propellant load, I_lat & I_roll (rho=+0.70 with dry_mass)
        x[:, 10] = np.maximum(0.90, 1.0 + norm.ppf(u[:, 10]) * 0.00333)
        x[:, 11] = np.maximum(0.95, 1.0 + norm.ppf(u[:, 11]) * 0.001)
        i_lat_raw = np.maximum(0.85, 1.0 + norm.ppf(u[:, 12]) * 0.01)
        i_roll_raw = np.maximum(0.85, 1.0 + norm.ppf(u[:, 13]) * 0.01)
        x[:, 12] = _iman_conover_pair(x[:, 10], i_lat_raw, 0.70, rng)
        x[:, 13] = _iman_conover_pair(x[:, 10], i_roll_raw, 0.70, rng)

        # 15-21: Per-engine thrust bias (7 independent scalars)
        for eng_i in range(7):
            x[:, 14 + eng_i] = np.maximum(0.90, 1.0 + norm.ppf(u[:, 14 + eng_i]) * 0.005)

        # 22-24: Global thrust, drag, engine ignition/lag delay
        x[:, 21] = np.maximum(0.90, 1.0 + norm.ppf(u[:, 21]) * 0.00833)
        x[:, 22] = np.maximum(0.70, 1.0 + norm.ppf(u[:, 22]) * 0.0333)
        x[:, 23] = u[:, 23] * 0.040  # U[0.0, 0.040] s

        # 25-29: Flex f1/f2 (rho=+0.80), slosh baffle, SRP shield, ground effect
        x[:, 24] = np.maximum(0.70, 1.0 + norm.ppf(u[:, 24]) * 0.0333)
        f2_raw = np.maximum(0.70, 1.0 + norm.ppf(u[:, 25]) * 0.0333)
        x[:, 25] = _iman_conover_pair(x[:, 24], f2_raw, 0.80, rng)
        x[:, 26] = np.maximum(0.70, 1.0 + norm.ppf(u[:, 26]) * 0.0667)
        x[:, 27] = np.maximum(0.50, 1.0 + norm.ppf(u[:, 27]) * 0.050)
        x[:, 28] = np.maximum(0.40, 1.0 + norm.ppf(u[:, 28]) * 0.0833)

        # 30-32: GPS pos/vel & radar altimeter noise scales
        x[:, 29] = np.maximum(0.50, 1.0 + norm.ppf(u[:, 29]) * 0.15)
        x[:, 30] = np.maximum(0.50, 1.0 + norm.ppf(u[:, 30]) * 0.15)
        x[:, 31] = np.maximum(0.50, 1.0 + norm.ppf(u[:, 31]) * 0.15)

        # 33-38: 3-axis gyro bias [deg/s] & 3-axis accel bias [mg]
        for ax_i in range(3):
            x[:, 32 + ax_i] = norm.ppf(u[:, 32 + ax_i]) * 0.010
            x[:, 35 + ax_i] = norm.ppf(u[:, 35 + ax_i]) * 5.0

        # 39-44: ARW/VRW, scale error ppm, non-orthogonality deg
        x[:, 38] = np.maximum(0.50, 1.0 + norm.ppf(u[:, 38]) * 0.15)
        x[:, 39] = np.maximum(0.50, 1.0 + norm.ppf(u[:, 39]) * 0.15)
        x[:, 40] = norm.ppf(u[:, 40]) * 66.7
        x[:, 41] = norm.ppf(u[:, 41]) * 66.7
        x[:, 42] = norm.ppf(u[:, 42]) * 0.0167
        x[:, 43] = norm.ppf(u[:, 43]) * 0.0167

        # 45-48: Packet dropout, discrete comm delay {0,1,2}, GPS blackout, roll pulse
        x[:, 44] = u[:, 44] * 0.08
        x[:, 45] = np.floor(u[:, 45] * 3.0).clip(0.0, 2.0)
        x[:, 46] = u[:, 46] * 30.0
        x[:, 47] = norm.ppf(u[:, 47]) * 833.0

        # 49-54: Pad slope, slope azimuth, 4 independent leg stiffness scales
        x[:, 48] = u[:, 48] * 2.5
        x[:, 49] = u[:, 49] * 360.0
        for leg_i in range(4):
            x[:, 50 + leg_i] = np.maximum(0.75, 1.0 + norm.ppf(u[:, 50 + leg_i]) * 0.0167)

        blocks.append(x)

    return np.vstack(blocks)


def build_mc_1000_suite(
    *,
    n_replicates: int = 10,
    per_replicate: int = 100,
    root_seed: int = 202604,
) -> list[Scenario3D]:
    """1,000-case Tier-3A `mc-1000` suite (54-scalar `10 x 100` Replicated LHS, CDR §4.2)."""
    mat = sample_mc_1000_lhs_matrix(
        n_replicates=n_replicates, per_replicate=per_replicate, root_seed=root_seed
    )
    aux_rng = np.random.default_rng(np.random.SeedSequence([int(root_seed), 99101]))
    base_v2 = PhysicsV2Config.full_v2()
    cases: list[Scenario3D] = []

    for global_idx in range(mat.shape[0]):
        rep_id = global_idx // per_replicate
        rep_case = global_idx % per_replicate
        row = mat[global_idx]

        az_rad = radians(float(row[5]))
        ca, sa = cos(az_rad), sin(az_rad)
        wg = float(row[0])
        wa = float(row[1])
        we = float(row[2])
        wz = float(row[3])
        wx_a, wx_f = _prof_unclipped(wg * ca, wa * ca, we * ca)
        wy_a, wy_f = _prof_unclipped(wg * sa, wa * sa, we * sa)

        comm_steps = int(round(float(row[45])))
        obs_delay = 1 if comm_steps >= 1 else 0
        act_delay = 1 if comm_steps >= 2 else 0

        gust_az = float(aux_rng.uniform(0.0, 2.0 * pi))
        gust_start = float(aux_rng.uniform(25.0, 480.0))
        gust_alt = float(aux_rng.uniform(2_000.0, 20_000.0))
        roll_start = float(aux_rng.uniform(260.0, 470.0))

        pv2 = replace(
            base_v2,
            booster_dry_mass_scale=float(row[10]),
            engine_thrust_bias_7=(
                float(row[14]),
                float(row[15]),
                float(row[16]),
                float(row[17]),
                float(row[18]),
                float(row[19]),
                float(row[20]),
            ),
            flex_freq_mode1_scale=float(row[24]),
            flex_freq_mode2_scale=float(row[25]),
            slosh_baffle_damping_scale=float(row[26]),
            srp_shield_scale=float(row[27]),
            ground_effect_coeff=0.07 * float(row[28]),
            dryden_sigma_m_s=float(row[6]),
            radar_alt_noise_scale=float(row[31]),
            gyro_arw_scale=float(row[38]),
            accel_vrw_scale=float(row[39]),
            gyro_scale_err_ppm=float(row[40]),
            accel_scale_err_ppm=float(row[41]),
            gyro_nonortho_deg=float(row[42]),
            accel_nonortho_deg=float(row[43]),
            gps_blackout_duration_s=float(row[46]),
            pad_slope_deg=float(row[48]),
            pad_slope_azimuth_deg=float(row[49]),
            leg_stiffness_scale_4=(
                float(row[50]),
                float(row[51]),
                float(row[52]),
                float(row[53]),
            ),
        )

        sensors = SensorConfig3D(
            position_noise_m=0.5 * float(row[29]),
            velocity_noise_m_s=0.15 * float(row[30]),
            attitude_noise_deg=0.05,
            gyro_noise_deg_s=0.05,
            accel_noise_m_s2=0.05,
            gyro_bias_deg_s=(float(row[32]), float(row[33]), float(row[34])),
            accel_bias_m_s2=(
                float(row[35]) * 1e-3 * G0,
                float(row[36]) * 1e-3 * G0,
                float(row[37]) * 1e-3 * G0,
            ),
            dropout_rate=float(row[44]),
            observation_delay_steps=obs_delay,
            action_delay_steps=act_delay,
        )

        cases.append(
            Scenario3D(
                name=f"mc1000-r{rep_id:02d}-c{rep_case:03d}",
                atmosphere=Atmosphere3DSettings(
                    sea_level_temperature_k=288.15 + float(row[7]),
                    sea_level_pressure_pa=101_325.0 + float(row[8]),
                    latitude_deg=37.0,
                    coriolis_enabled=True,
                    density_scale=1.0,
                    upper_air_density_scale=float(row[9]),
                ),
                wind=Wind3DConfig(
                    wind_x_profile=wx_a,
                    wind_y_profile=wy_a,
                    wind_z_profile=((0.0, wz), (100_000.0, wz)),
                    forecast_wind_x_profile=wx_f,
                    forecast_wind_y_profile=wy_f,
                    gust_amplitude_m_s=float(row[4]),
                    gust_azimuth_rad=gust_az,
                    gust_start_s=gust_start,
                    gust_duration_s=90.0,
                    gust_altitude_center_m=gust_alt,
                ),
                disturbances=Disturbances3D(
                    thrust_scale=float(row[21]),
                    drag_scale=float(row[22]),
                    initial_mass_scale=1.0,
                    inertia_scale=float(row[12]),
                    inertia_roll_scale=float(row[13]),
                    propellant_load_scale=float(row[11]),
                    engine_lag_s=float(row[23]),
                    roll_disturbance_torque_nm=float(row[47]),
                    roll_disturbance_start_s=roll_start,
                    roll_disturbance_duration_s=4.0,
                ),
                sensors=sensors,
                seed=root_seed + 10_000 + global_idx,
                physics_v2=pv2,
                expected_mission_mode="RTLS",
                suite_metadata={
                    "suite": "mc-1000",
                    "replicate_id": rep_id,
                    "replicate_case": rep_case,
                },
            )
        )
    return cases


def build_fault_mc_160_suite(*, per_stratum: int = 40, root_seed: int = 202605) -> list[Scenario3D]:
    """160-case Tier-3B stratified Fault Monte Carlo suite (4 strata x 40 cases, CDR §4.1)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(root_seed), 40160]))
    base_v2 = PhysicsV2Config.full_v2()
    cases: list[Scenario3D] = []

    def _draw_nominal_env() -> tuple[
        Atmosphere3DSettings, Wind3DConfig, Disturbances3D, SensorConfig3D, dict[str, object]
    ]:
        az = float(rng.uniform(0.0, 2.0 * pi))
        wg = float(np.clip(rng.normal(0.0, 2.5), -6.5, 6.5))
        wa = float(np.clip(rng.normal(0.0, 6.0), -15.0, 15.0))
        we = float(np.clip(rng.normal(0.0, 0.5), -1.2, 1.2))
        wz = float(np.clip(rng.normal(0.0, 0.6), -1.5, 1.5))
        wx_a, wx_f = _prof_unclipped(wg * cos(az), wa * cos(az), we * cos(az))
        wy_a, wy_f = _prof_unclipped(wg * sin(az), wa * sin(az), we * sin(az))
        atmo = Atmosphere3DSettings(
            sea_level_temperature_k=288.15 + float(np.clip(rng.normal(0.0, 5.0), -12.0, 12.0)),
            sea_level_pressure_pa=101_325.0 + float(np.clip(rng.normal(0.0, 1_000.0), -2500, 2500)),
            latitude_deg=37.0,
            coriolis_enabled=True,
            upper_air_density_scale=float(np.clip(rng.normal(1.0, 0.010), 0.96, 1.04)),
        )
        wind = Wind3DConfig(
            wind_x_profile=wx_a,
            wind_y_profile=wy_a,
            wind_z_profile=((0.0, wz), (100_000.0, wz)),
            forecast_wind_x_profile=wx_f,
            forecast_wind_y_profile=wy_f,
        )
        dist = Disturbances3D(
            thrust_scale=float(np.clip(rng.normal(1.0, 0.007), 0.98, 1.02)),
            drag_scale=float(np.clip(rng.normal(1.0, 0.030), 0.92, 1.08)),
            inertia_scale=float(np.clip(rng.normal(1.0, 0.008), 0.98, 1.02)),
            inertia_roll_scale=float(np.clip(rng.normal(1.0, 0.008), 0.98, 1.02)),
            propellant_load_scale=float(np.clip(rng.normal(1.0, 0.0008), 0.997, 1.003)),
            engine_lag_s=float(rng.uniform(0.0, 0.030)),
        )
        sensors = SensorConfig3D(
            position_noise_m=float(rng.uniform(0.35, 0.70)),
            velocity_noise_m_s=float(rng.uniform(0.10, 0.20)),
            attitude_noise_deg=0.05,
            gyro_noise_deg_s=0.05,
            accel_noise_m_s2=0.05,
            gyro_bias_deg_s=(
                float(np.clip(rng.normal(0.0, 0.008), -0.025, 0.025)),
                float(np.clip(rng.normal(0.0, 0.008), -0.025, 0.025)),
                float(np.clip(rng.normal(0.0, 0.008), -0.025, 0.025)),
            ),
            accel_bias_m_s2=(
                float(np.clip(rng.normal(0.0, 4.0), -12.0, 12.0)) * 1e-3 * G0,
                float(np.clip(rng.normal(0.0, 4.0), -12.0, 12.0)) * 1e-3 * G0,
                float(np.clip(rng.normal(0.0, 4.0), -12.0, 12.0)) * 1e-3 * G0,
            ),
            dropout_rate=float(rng.uniform(0.0, 0.06)),
            observation_delay_steps=int(rng.integers(0, 2)),
            action_delay_steps=0,
        )
        pv2_common: dict[str, object] = {
            "booster_dry_mass_scale": float(np.clip(rng.normal(1.0, 0.003), 0.992, 1.008)),
            "engine_thrust_bias_7": tuple(
                float(np.clip(rng.normal(1.0, 0.004), 0.988, 1.012)) for _ in range(7)
            ),
            "flex_freq_mode1_scale": float(np.clip(rng.normal(1.0, 0.025), 0.93, 1.07)),
            "flex_freq_mode2_scale": float(np.clip(rng.normal(1.0, 0.025), 0.93, 1.07)),
            "slosh_baffle_damping_scale": float(np.clip(rng.normal(1.0, 0.05), 0.85, 1.15)),
            "srp_shield_scale": float(np.clip(rng.normal(1.0, 0.04), 0.90, 1.10)),
            "dryden_sigma_m_s": float(rng.uniform(1.0, 2.2)),
            "gps_blackout_duration_s": float(rng.uniform(0.0, 20.0)),
            "pad_slope_deg": float(rng.uniform(0.0, 2.0)),
            "pad_slope_azimuth_deg": float(rng.uniform(0.0, 360.0)),
            "leg_stiffness_scale_4": tuple(
                float(np.clip(rng.normal(1.0, 0.015), 0.96, 1.04)) for _ in range(4)
            ),
        }
        return atmo, wind, dist, sensors, pv2_common

    case_seq = 0

    # Stratum 1: Early-Ascent EO -> DOWNRANGE_SPLASHDOWN_ABORT window (t_eo in [20, 52] s)
    for k in range(per_stratum):
        atmo, wind, dist, sensors, pv2_c = _draw_nominal_env()
        eng_idx = int(rng.integers(1, 7))
        t_eo = float(rng.uniform(20.0, 52.0))
        tau_eo = float(rng.uniform(0.20, 0.30))
        pv2 = replace(
            base_v2,
            **pv2_c,  # type: ignore[arg-type]
            engine_out_index=eng_idx,
            engine_out_time_s=t_eo,
            engine_out_spindown_tau_s=tau_eo,
        )
        cases.append(
            Scenario3D(
                name=f"faultmc-s1-eo-abort-{k + 1:02d}",
                atmosphere=atmo,
                wind=wind,
                disturbances=dist,
                sensors=sensors,
                seed=root_seed + 1_000 + case_seq,
                physics_v2=pv2,
                expected_mission_mode=None,  # Fair causal mission_mode scoring near boundary
                suite_metadata={
                    "suite": "fault-mc-160",
                    "stratum": "stratum_1_eo_ascent_abort",
                    "stratum_index": 1,
                    "fault_kind": "engine_out_ascent_early",
                    "engine_out_index": eng_idx,
                    "engine_out_time_s": t_eo,
                },
            )
        )
        case_seq += 1

    # Stratum 2: Late-Ascent EO -> RTLS window (t_eo in [72, 108] s)
    for k in range(per_stratum):
        atmo, wind, dist, sensors, pv2_c = _draw_nominal_env()
        eng_idx = int(rng.integers(1, 7))
        t_eo = float(rng.uniform(72.0, 108.0))
        tau_eo = float(rng.uniform(0.20, 0.30))
        pv2 = replace(
            base_v2,
            **pv2_c,  # type: ignore[arg-type]
            engine_out_index=eng_idx,
            engine_out_time_s=t_eo,
            engine_out_spindown_tau_s=tau_eo,
        )
        cases.append(
            Scenario3D(
                name=f"faultmc-s2-eo-rtls-{k + 1:02d}",
                atmosphere=atmo,
                wind=wind,
                disturbances=dist,
                sensors=sensors,
                seed=root_seed + 1_000 + case_seq,
                physics_v2=pv2,
                expected_mission_mode=None,  # Fair causal mission_mode scoring near boundary
                suite_metadata={
                    "suite": "fault-mc-160",
                    "stratum": "stratum_2_eo_ascent_rtls",
                    "stratum_index": 2,
                    "fault_kind": "engine_out_ascent_late",
                    "engine_out_index": eng_idx,
                    "engine_out_time_s": t_eo,
                },
            )
        )
        case_seq += 1

    # Stratum 3: Post-Separation 3-Engine EO (boostback / entry_burn / landing_burn)
    # Active default 3-engine triplet is [0, 1, 4]; inject outer engine 1 or 4 failure.
    n_bb = per_stratum // 4
    n_eb = (per_stratum - n_bb) // 2
    for k in range(per_stratum):
        atmo, wind, dist, sensors, pv2_c = _draw_nominal_env()
        eng_idx = 1 if (k % 2 == 0) else 4
        tau_eo = float(rng.uniform(0.20, 0.30))
        if k < n_bb:
            eo_phase = "boostback"
            eo_el = float(rng.uniform(3.0, 20.0))
        elif k < n_bb + n_eb:
            eo_phase = "entry_burn"
            eo_el = float(rng.uniform(2.0, 12.0))
        else:
            eo_phase = "landing_burn"
            eo_el = float(rng.uniform(0.5, 2.5))
        pv2 = replace(
            base_v2,
            **pv2_c,  # type: ignore[arg-type]
            engine_out_index=eng_idx,
            engine_out_phase=eo_phase,
            engine_out_phase_elapsed_s=eo_el,
            engine_out_spindown_tau_s=tau_eo,
        )
        cases.append(
            Scenario3D(
                name=f"faultmc-s3-eo-postsep-{k + 1:02d}",
                atmosphere=atmo,
                wind=wind,
                disturbances=dist,
                sensors=sensors,
                seed=root_seed + 1_000 + case_seq,
                physics_v2=pv2,
                expected_mission_mode="RTLS",
                suite_metadata={
                    "suite": "fault-mc-160",
                    "stratum": "stratum_3_eo_postsep_3to2",
                    "stratum_index": 3,
                    "fault_kind": f"engine_out_{eo_phase}",
                    "engine_out_index": eng_idx,
                    "engine_out_phase": eo_phase,
                    "engine_out_phase_elapsed_s": eo_el,
                },
            )
        )
        case_seq += 1

    # Stratum 4: Aero-Glide Grid-Fin Jam (+/- 6..10 deg across all 4 fins)
    for k in range(per_stratum):
        atmo, wind, dist, sensors, pv2_c = _draw_nominal_env()
        fin_idx = k % 4
        sign_j = 1.0 if (k % 2 == 0) else -1.0
        jam_deg = sign_j * float(rng.uniform(6.0, 10.0))
        jam_el = float(rng.uniform(3.0, 35.0))
        pv2 = replace(
            base_v2,
            **pv2_c,  # type: ignore[arg-type]
            fin_jam_index=fin_idx,
            fin_jam_phase="aero_glide",
            fin_jam_phase_elapsed_s=jam_el,
            fin_jam_angle_deg=jam_deg,
        )
        cases.append(
            Scenario3D(
                name=f"faultmc-s4-fin-jam-{k + 1:02d}",
                atmosphere=atmo,
                wind=wind,
                disturbances=dist,
                sensors=sensors,
                seed=root_seed + 1_000 + case_seq,
                physics_v2=pv2,
                expected_mission_mode="RTLS",
                suite_metadata={
                    "suite": "fault-mc-160",
                    "stratum": "stratum_4_fin_jam_glide",
                    "stratum_index": 4,
                    "fault_kind": "fin_jam_aero_glide",
                    "fin_jam_index": fin_idx,
                    "fin_jam_angle_deg": jam_deg,
                    "fin_jam_phase_elapsed_s": jam_el,
                },
            )
        )
        case_seq += 1

    return cases


def build_envelope_boundary_suite() -> list[Scenario3D]:
    """100-case Tier-4 deterministic failure-boundary suite (4 x 5x5 grids, CDR §4.1)."""
    base_v2 = PhysicsV2Config.full_v2()
    nom_imu = SensorConfig3D(
        position_noise_m=0.5,
        velocity_noise_m_s=0.15,
        attitude_noise_deg=0.05,
        gyro_noise_deg_s=0.05,
        accel_noise_m_s2=0.05,
        gyro_bias_deg_s=(0.02, -0.02, 0.02),
        dropout_rate=0.05,
    )
    cases: list[Scenario3D] = []
    seed_seq = 80_000

    # Grid 1: Coupled wind v_g=[8,11,14,17,20] (v_a=2*v_g) x pad slope [2.0,3.5,5.0,6.5,8.0] deg
    g1_vg = (8.0, 11.0, 14.0, 17.0, 20.0)
    g1_slope = (2.0, 3.5, 5.0, 6.5, 8.0)
    for r, sl_deg in enumerate(g1_slope):
        for c, vg in enumerate(g1_vg):
            va = 2.0 * vg
            wx_a, wx_f = _prof_unclipped(vg, va, 0.0)
            wy_a, wy_f = _prof_unclipped(0.0, 0.0, 0.0)
            cases.append(
                Scenario3D(
                    name=f"env-g1-wind-slope-r{r}c{c}",
                    atmosphere=Atmosphere3DSettings(latitude_deg=37.0, coriolis_enabled=True),
                    wind=Wind3DConfig(
                        wind_x_profile=wx_a,
                        wind_y_profile=wy_a,
                        wind_z_profile=((0.0, 0.0), (100_000.0, 0.0)),
                        forecast_wind_x_profile=wx_f,
                        forecast_wind_y_profile=wy_f,
                    ),
                    disturbances=Disturbances3D(),
                    sensors=nom_imu,
                    seed=seed_seq,
                    physics_v2=replace(base_v2, pad_slope_deg=sl_deg, pad_slope_azimuth_deg=0.0),
                    expected_mission_mode="RTLS",
                    suite_metadata={
                        "suite": "envelope-boundary",
                        "grid_id": "g1_wind_slope",
                        "grid_title": "Coupled Wind vs Pad Slope",
                        "row": r,
                        "col": c,
                        "x_label": "Ground Wind v_g [m/s] (v_a=2v_g)",
                        "y_label": "Pad Slope [deg]",
                        "x_val": vg,
                        "y_val": sl_deg,
                    },
                )
            )
            seed_seq += 1

    # Grid 2: Fin-Jam [6,9,12,15,18] deg x glide crosswind [10,15,20,25,30] m/s
    g2_jam = (6.0, 9.0, 12.0, 15.0, 18.0)
    g2_wind = (10.0, 15.0, 20.0, 25.0, 30.0)
    for r, va in enumerate(g2_wind):
        for c, jam_deg in enumerate(g2_jam):
            vg = 0.4 * va
            wx_a, wx_f = _prof_unclipped(0.0, 0.0, 0.0)
            wy_a, wy_f = _prof_unclipped(vg, va, 0.0)
            cases.append(
                Scenario3D(
                    name=f"env-g2-finjam-wind-r{r}c{c}",
                    atmosphere=Atmosphere3DSettings(latitude_deg=37.0, coriolis_enabled=True),
                    wind=Wind3DConfig(
                        wind_x_profile=wx_a,
                        wind_y_profile=wy_a,
                        wind_z_profile=((0.0, 0.0), (100_000.0, 0.0)),
                        forecast_wind_x_profile=wx_f,
                        forecast_wind_y_profile=wy_f,
                    ),
                    disturbances=Disturbances3D(),
                    sensors=nom_imu,
                    seed=seed_seq,
                    physics_v2=replace(
                        base_v2,
                        fin_jam_index=2,
                        fin_jam_phase="aero_glide",
                        fin_jam_phase_elapsed_s=5.0,
                        fin_jam_angle_deg=jam_deg,
                    ),
                    expected_mission_mode="RTLS",
                    suite_metadata={
                        "suite": "envelope-boundary",
                        "grid_id": "g2_finjam_wind",
                        "grid_title": "Fin-Jam Angle vs Glide Crosswind",
                        "row": r,
                        "col": c,
                        "x_label": "Fin-Jam Angle [deg]",
                        "y_label": "Glide Crosswind v_a [m/s]",
                        "x_val": jam_deg,
                        "y_val": va,
                    },
                )
            )
            seed_seq += 1

    # Grid 3: Forced reentry GPS blackout [30,50,70,95,120] s x gyro bias [0.03..0.25] deg/s
    g3_bo = (30.0, 50.0, 70.0, 95.0, 120.0)
    g3_gb = (0.03, 0.06, 0.10, 0.15, 0.25)
    for r, gb in enumerate(g3_gb):
        for c, bo_s in enumerate(g3_bo):
            wx_a, wx_f = _prof_unclipped(4.0, 10.0, 0.0)
            wy_a, wy_f = _prof_unclipped(4.0, 10.0, 0.0)
            sens = SensorConfig3D(
                position_noise_m=0.8,
                velocity_noise_m_s=0.20,
                attitude_noise_deg=0.06,
                gyro_noise_deg_s=0.06,
                accel_noise_m_s2=0.06,
                gyro_bias_deg_s=(gb, -gb, gb),
                dropout_rate=0.06,
                observation_delay_steps=1,
                action_delay_steps=0,
            )
            cases.append(
                Scenario3D(
                    name=f"env-g3-blackout-gyro-r{r}c{c}",
                    atmosphere=Atmosphere3DSettings(latitude_deg=37.0, coriolis_enabled=True),
                    wind=Wind3DConfig(
                        wind_x_profile=wx_a,
                        wind_y_profile=wy_a,
                        wind_z_profile=((0.0, 0.0), (100_000.0, 0.0)),
                        forecast_wind_x_profile=wx_f,
                        forecast_wind_y_profile=wy_f,
                    ),
                    disturbances=Disturbances3D(),
                    sensors=sens,
                    seed=seed_seq,
                    physics_v2=replace(base_v2, gps_blackout_duration_s=bo_s),
                    expected_mission_mode="RTLS",
                    suite_metadata={
                        "suite": "envelope-boundary",
                        "grid_id": "g3_blackout_gyro",
                        "grid_title": "GPS Blackout Duration vs Gyro Bias",
                        "row": r,
                        "col": c,
                        "x_label": "Forced GPS Blackout [s]",
                        "y_label": "Gyro Bias [deg/s]",
                        "x_val": bo_s,
                        "y_val": gb,
                    },
                )
            )
            seed_seq += 1

    # Grid 4: Downward flex frequency error [-10,-14,-18,-22,-26]% x Dryden [2.0,3.5,5.0,6.5,8.0]
    g4_flex_pct = (-10.0, -14.0, -18.0, -22.0, -26.0)
    g4_dryden = (2.0, 3.5, 5.0, 6.5, 8.0)
    for r, sig_d in enumerate(g4_dryden):
        for c, f_pct in enumerate(g4_flex_pct):
            f_sc = 1.0 + f_pct / 100.0
            wx_a, wx_f = _prof_unclipped(4.0, 10.0, 0.0)
            wy_a, wy_f = _prof_unclipped(-4.0, -10.0, 0.0)
            cases.append(
                Scenario3D(
                    name=f"env-g4-flex-dryden-r{r}c{c}",
                    atmosphere=Atmosphere3DSettings(latitude_deg=37.0, coriolis_enabled=True),
                    wind=Wind3DConfig(
                        wind_x_profile=wx_a,
                        wind_y_profile=wy_a,
                        wind_z_profile=((0.0, 0.0), (100_000.0, 0.0)),
                        forecast_wind_x_profile=wx_f,
                        forecast_wind_y_profile=wy_f,
                    ),
                    disturbances=Disturbances3D(),
                    sensors=nom_imu,
                    seed=seed_seq,
                    physics_v2=replace(
                        base_v2,
                        flex_freq_mode1_scale=f_sc,
                        flex_freq_mode2_scale=f_sc,
                        dryden_sigma_m_s=sig_d,
                    ),
                    expected_mission_mode="RTLS",
                    suite_metadata={
                        "suite": "envelope-boundary",
                        "grid_id": "g4_flex_dryden",
                        "grid_title": "Flex Frequency Error vs Dryden Turbulence",
                        "row": r,
                        "col": c,
                        "x_label": "Flex Frequency Error [%]",
                        "y_label": "Dryden Turbulence sigma [m/s]",
                        "x_val": f_pct,
                        "y_val": sig_d,
                    },
                )
            )
            seed_seq += 1

    return cases


def get_validation_suite(name: str) -> list[Scenario3D]:
    """Return scenarios for all v1 & v2 suites."""
    if name == "edge-3d":
        return build_edge_3d_suite()
    if name == "holdout-3d":
        return build_holdout_3d_suite(64)
    if name == "full-3d":
        return build_edge_3d_suite() + build_holdout_3d_suite(64)
    if name == "edge-v2":
        return build_edge_v2_suite()
    if name == "mc-1000":
        return build_mc_1000_suite()
    if name == "fault-mc-160":
        return build_fault_mc_160_suite()
    if name == "envelope-boundary":
        return build_envelope_boundary_suite()
    raise ValueError(
        f"Unknown suite {name!r}; expected 'edge-3d', 'holdout-3d', 'full-3d', "
        "'edge-v2', 'mc-1000', 'fault-mc-160', or 'envelope-boundary'"
    )


def select_shard(cases: list[Scenario3D], shard_spec: str) -> list[Scenario3D]:
    """Select deterministic round-robin shard `index/total` (0 <= index < total)."""
    parts = shard_spec.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard spec {shard_spec!r}; expected '<index>/<total>'")
    idx, total = int(parts[0]), int(parts[1])
    if total <= 0 or not (0 <= idx < total):
        raise ValueError(f"Invalid shard range {idx}/{total}")
    return [case for pos, case in enumerate(cases) if pos % total == idx]
