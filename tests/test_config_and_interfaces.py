"""Unit tests for DM100 3D vehicle config, 3D inertia tensor, and causal interfaces."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from dm100_3d.config import DM100VehicleConfig, load_vehicle_config, mass_properties_3d
from dm100_3d.interfaces import TelemetryPacket3D

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_vehicle_config_matches_dm100_mass_budget() -> None:
    cfg = load_vehicle_config(ROOT / "configs" / "vehicle_dm100.yaml")
    default_cfg = DM100VehicleConfig()
    assert cfg == default_cfg
    assert np.isclose(cfg.booster_dry_mass_kg, 6_200.0)
    assert np.isclose(cfg.booster_wet_mass_kg, 76_000.0)
    assert np.isclose(cfg.upper_stage.total_mass_kg, 24_000.0)
    assert np.isclose(cfg.launch_mass_kg, 100_000.0)
    # 4 legs x 80 kN/m == 2 planar legs x 160 kN/m
    assert np.isclose(
        cfg.landing_gear.leg_count * cfg.landing_gear.per_leg_stiffness_n_m, 320_000.0
    )
    assert np.isclose(cfg.landing_gear.leg_count * cfg.landing_gear.per_leg_damping_n_s_m, 90_000.0)


def test_mass_properties_3d_matches_2d_launchpad_cg_and_roll_inertia() -> None:
    cfg = DM100VehicleConfig()
    mp_launch = mass_properties_3d(cfg.launch_mass_kg, 0.0, vehicle=cfg, upper_attached=True)
    # Matches DM100 2D calm_replay_50hz.json frame 0: z_com = cg0 - nozzle_station_m (-0.2)
    z_com_pad = mp_launch.cg_station_m - cfg.geometry.nozzle_station_m
    assert np.isclose(z_com_pad, 13.492616121843149, atol=1e-9)

    ixx, iyy, izz = mp_launch.inertia_diag_kg_m2
    assert np.isclose(ixx, iyy)
    assert izz > 0.0
    # Slender rocket roll inertia must be much smaller than pitch/yaw inertia
    assert izz < 0.05 * ixx

    # Landing booster (nearly dry, gear deployed, upper detached)
    mp_land = mass_properties_3d(8_800.0, 1.0, vehicle=cfg, upper_attached=False)
    assert 5.0 < mp_land.cg_station_m < 12.0
    assert mp_land.inertia_diag_kg_m2[2] < 0.05 * mp_land.inertia_diag_kg_m2[0]


def test_telemetry_packet_strict_causality() -> None:
    """TelemetryPacket3D must never expose hidden plant truth fields."""
    names = {f.name for f in fields(TelemetryPacket3D)}
    forbidden = {
        "true_wind",
        "wind_gust",
        "density_scale",
        "drag_scale",
        "thrust_scale",
        "mass_scale",
        "inertia_scale",
        "slosh_state",
        "true_bias",
    }
    assert names.isdisjoint(forbidden)
