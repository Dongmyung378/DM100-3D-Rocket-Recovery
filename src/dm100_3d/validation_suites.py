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
from math import pi

import numpy as np

from dm100_3d.config import PhysicsV2Config
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


def get_validation_suite(name: str) -> list[Scenario3D]:
    """Return scenarios for 'edge-3d' (32), 'holdout-3d' (64), 'full-3d' (96), 'edge-v2' (48)."""
    if name == "edge-3d":
        return build_edge_3d_suite()
    if name == "holdout-3d":
        return build_holdout_3d_suite(64)
    if name == "full-3d":
        return build_edge_3d_suite() + build_holdout_3d_suite(64)
    if name == "edge-v2":
        return build_edge_v2_suite()
    raise ValueError(
        f"Unknown suite {name!r}; expected 'edge-3d', 'holdout-3d', 'full-3d', or 'edge-v2'"
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
