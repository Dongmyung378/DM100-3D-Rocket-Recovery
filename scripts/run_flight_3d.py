"""Run a single 6DOF DM100 3D RTLS mission and optionally export 50 Hz replay JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from dm100_3d.gnc.mekf import SensorConfig3D
from dm100_3d.physics import Atmosphere3DSettings, Wind3DConfig
from dm100_3d.replay import export_replay_json
from dm100_3d.sim_3d import Disturbances3D, Scenario3D, SimConfig3D, run_flight_3d


def build_named_scenario(name: str) -> Scenario3D:
    if name == "calm":
        return Scenario3D(name="calm_3d")
    if name == "crosswind-3d":
        w_x = ((0.0, 5.0), (50.0, 5.0), (10_000.0, 14.0), (100_000.0, 14.0))
        w_y = ((0.0, -4.5), (50.0, -4.5), (10_000.0, 12.0), (100_000.0, 12.0))
        return Scenario3D(
            name="crosswind_3d",
            atmosphere=Atmosphere3DSettings(latitude_deg=37.0, density_scale=1.03),
            wind=Wind3DConfig(
                wind_x_profile=w_x,
                wind_y_profile=w_y,
                forecast_wind_x_profile=(
                    (0.0, 4.0),
                    (50.0, 4.0),
                    (10_000.0, 13.0),
                    (100_000.0, 13.0),
                ),
                forecast_wind_y_profile=(
                    (0.0, -3.5),
                    (50.0, -3.5),
                    (10_000.0, 11.0),
                    (100_000.0, 11.0),
                ),
                gust_amplitude_m_s=3.5,
                gust_azimuth_rad=0.9,
                gust_start_s=280.0,
                gust_duration_s=120.0,
                gust_altitude_center_m=8_000.0,
            ),
            disturbances=Disturbances3D(
                thrust_scale=0.99,
                drag_scale=1.04,
                roll_disturbance_torque_nm=2_500.0,
                roll_disturbance_start_s=320.0,
                roll_disturbance_duration_s=5.0,
            ),
            sensors=SensorConfig3D(
                position_noise_m=0.5,
                velocity_noise_m_s=0.15,
                attitude_noise_deg=0.05,
                gyro_noise_deg_s=0.05,
                gyro_bias_deg_s=(0.02, -0.02, 0.02),
                dropout_rate=0.04,
            ),
        )
    raise ValueError(f"Unknown scenario {name!r}; choose 'calm' or 'crosswind-3d'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single DM100 3D MuJoCo 6DOF RTLS flight.")
    parser.add_argument("--scenario", default="calm", choices=["calm", "crosswind-3d"])
    parser.add_argument("--export-replay", type=Path, default=None)
    parser.add_argument("--out-summary", type=Path, default=None)
    args = parser.parse_args()

    t0 = perf_counter()
    scen = build_named_scenario(args.scenario)
    summary, frames = run_flight_3d(
        scen,
        sim_config=SimConfig3D(
            record_replay=args.export_replay is not None,
            seed=scen.seed,
            sensor_seed=scen.seed + 1000,
        ),
    )
    elapsed = perf_counter() - t0
    summary["wall_time_s"] = elapsed

    print(json.dumps(summary, indent=2))
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.export_replay is not None:
        export_replay_json(frames, args.export_replay, metadata=summary)
        print(f"Exported {len(frames)} frames to {args.export_replay}")


if __name__ == "__main__":
    main()
