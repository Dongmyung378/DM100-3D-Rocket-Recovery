"""3D external forces, atmosphere/wind, slosh, propellant feed, and Stage 2 orbit modules."""

from __future__ import annotations

from dm100_3d.physics.atmosphere_3d import (
    Atmosphere3DSettings,
    AtmosphereLUT,
    Wind3DConfig,
    Wind3DLUT,
    standard_atmosphere,
)
from dm100_3d.physics.feed_3d import (
    FeedRates,
    FeedSettings,
    FeedState,
    IgnitionSettings,
    Separation3DSettings,
    feed_rates,
    ignition_ready,
    initial_feed_state,
    separation_event_3d,
)
from dm100_3d.physics.forces_3d import (
    compute_body_aero_3d,
    compute_coriolis_and_centrifugal_3d,
    compute_engine_thrust_3d,
    compute_fins_aero_3d,
    compute_forces_and_torques_3d,
    compute_stagnation_heat_flux_w_m2,
)
from dm100_3d.physics.orbit_3d import (
    UpperStageOrbitResult,
    UpperStageTrajectorySample,
    evaluate_upper_stage_orbit_3d,
)
from dm100_3d.physics.slosh_3d import (
    Slosh3DResponse,
    Slosh3DSettings,
    Slosh3DState,
    slosh_3d_response,
    slosh_body_offset_3d,
)

__all__ = [
    "Atmosphere3DSettings",
    "AtmosphereLUT",
    "FeedRates",
    "FeedSettings",
    "FeedState",
    "IgnitionSettings",
    "Separation3DSettings",
    "Slosh3DResponse",
    "Slosh3DSettings",
    "Slosh3DState",
    "UpperStageOrbitResult",
    "UpperStageTrajectorySample",
    "Wind3DConfig",
    "Wind3DLUT",
    "compute_body_aero_3d",
    "compute_coriolis_and_centrifugal_3d",
    "compute_engine_thrust_3d",
    "compute_fins_aero_3d",
    "compute_forces_and_torques_3d",
    "compute_stagnation_heat_flux_w_m2",
    "evaluate_upper_stage_orbit_3d",
    "feed_rates",
    "ignition_ready",
    "initial_feed_state",
    "separation_event_3d",
    "slosh_3d_response",
    "slosh_body_offset_3d",
    "standard_atmosphere",
]
