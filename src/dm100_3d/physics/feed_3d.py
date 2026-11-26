"""Propellant feed buffer, closed helium pressurization, and 3D cold stage separation.

Directly inherits DM100 2D's conserved 1,500 kg liquid buffer + 75 kg closed helium
inventory (`vehicle_systems.py`) and extends cold separation to full 6DOF momentum conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, radians, sin, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dm100_3d.config import DualTankGeometry
from dm100_3d.coords import quat_to_rotmat


@dataclass(frozen=True, slots=True)
class FeedSettings:
    main_tank_volume_m3: float = 15.0 * pi * 1.55**2
    liquid_density_kg_m3: float = 830.0
    buffer_capacity_kg: float = 1_500.0
    maximum_flow_kg_s: float = 650.0
    buffer_fill_time_s: float = 0.5
    flow_emptying_time_s: float = 0.05
    acquisition_time_s: float = 0.35
    acquisition_loss_time_s: float = 8.0
    settled_acceleration_m_s2: float = 0.3
    regulated_pressure_pa: float = 350_000.0
    minimum_pressure_pa: float = 220_000.0
    bottle_volume_m3: float = 1.5
    helium_total_mass_kg: float = 75.0
    gas_temperature_k: float = 290.0
    helium_specific_gas_constant_j_kg_k: float = 2_077.1
    regulator_time_s: float = 0.1
    regulator_maximum_flow_kg_s: float = 0.6
    regulator_head_pa: float = 50_000.0

    @classmethod
    def dual_tank_v2(cls, geom: DualTankGeometry | None = None) -> FeedSettings:
        """Combined dual-tank (CH4 39 m^3 + LOX 52 m^3 = 91 m^3, +9% ullage) feed settings."""
        dtg = geom or DualTankGeometry()
        return cls(
            main_tank_volume_m3=dtg.total_tank_volume_m3,
            liquid_density_kg_m3=dtg.bulk_propellant_density_kg_m3,
        )


@dataclass(frozen=True, slots=True)
class FeedState:
    buffer_kg: float
    acquisition: float
    tank_gas_kg: float
    bottle_gas_kg: float


@dataclass(frozen=True, slots=True)
class FeedRates:
    delivered_flow_kg_s: float
    pickup_flow_kg_s: float
    buffer_derivative_kg_s: float
    acquisition_derivative_s_inv: float
    tank_gas_derivative_kg_s: float
    bottle_gas_derivative_kg_s: float
    tank_pressure_pa: float
    bottle_pressure_pa: float
    demand_shortfall_kg_s: float
    inventory_valid: bool


@dataclass(frozen=True, slots=True)
class IgnitionSettings:
    delay_s: float = 0.2
    minimum_buffer_kg: float = 350.0
    minimum_pressure_pa: float = 280_000.0


@dataclass(frozen=True, slots=True)
class Separation3DSettings:
    relative_axial_speed_m_s: float = 0.5
    application_station_m: float = 20.0
    lateral_offset_x_m: float = 0.002
    lateral_offset_y_m: float = 0.0
    impulse_misalignment_deg: float = 0.0
    effective_duration_s: float = 0.15


def initial_feed_state(
    main_propellant_kg: float,
    settings: FeedSettings = FeedSettings(),
) -> FeedState:
    ullage = settings.main_tank_volume_m3 - main_propellant_kg / settings.liquid_density_kg_m3
    if ullage <= 0.0:
        raise ValueError("Main tank requires positive ullage volume")
    gas_rt = settings.helium_specific_gas_constant_j_kg_k * settings.gas_temperature_k
    tank_gas = settings.regulated_pressure_pa * ullage / gas_rt
    bottle_gas = settings.helium_total_mass_kg - tank_gas
    if bottle_gas <= 0.0:
        raise ValueError("Insufficient helium inventory for initial ullage pressure")
    return FeedState(
        buffer_kg=min(main_propellant_kg, settings.buffer_capacity_kg),
        acquisition=1.0,
        tank_gas_kg=tank_gas,
        bottle_gas_kg=bottle_gas,
    )


def feed_rates(
    main_propellant_kg: float,
    demanded_flow_kg_s: float,
    axial_proper_acceleration_m_s2: float,
    state: FeedState,
    settings: FeedSettings = FeedSettings(),
) -> FeedRates:
    """Continuous feed & closed helium pressurization rates (exact DM100 2D parity)."""
    main = max(0.0, main_propellant_kg)
    buf = min(main, max(0.0, state.buffer_kg))
    acq = min(1.0, max(0.0, state.acquisition))
    ullage = max(1e-9, settings.main_tank_volume_m3 - main / settings.liquid_density_kg_m3)
    gas_rt = settings.helium_specific_gas_constant_j_kg_k * settings.gas_temperature_k
    tank_p = max(0.0, state.tank_gas_kg) * gas_rt / ullage
    bottle_p = max(0.0, state.bottle_gas_kg) * gas_rt / settings.bottle_volume_m3

    p_margin = min(
        1.0,
        max(
            0.0,
            (tank_p - settings.minimum_pressure_pa)
            / (settings.regulated_pressure_pa - settings.minimum_pressure_pa),
        ),
    )
    p_flow_lim = settings.maximum_flow_kg_s * sqrt(p_margin)
    demand = max(0.0, demanded_flow_kg_s)
    desired_pickup = (
        demand + max(0.0, settings.buffer_capacity_kg - buf) / settings.buffer_fill_time_s
    )
    pickup = min(
        acq * p_flow_lim,
        desired_pickup,
        max(0.0, main - buf) / settings.flow_emptying_time_s,
    )
    delivered = min(demand, p_flow_lim, pickup + buf / settings.flow_emptying_time_s)
    settled = min(
        1.0, max(0.0, axial_proper_acceleration_m_s2 / settings.settled_acceleration_m_s2)
    )
    acq_tau = settings.acquisition_time_s if settled > acq else settings.acquisition_loss_time_s

    target_gas = settings.regulated_pressure_pa * ullage / gas_rt
    gas_ff = settings.regulated_pressure_pa * delivered / (settings.liquid_density_kg_m3 * gas_rt)
    req_gas = max(0.0, (target_gas - state.tank_gas_kg) / settings.regulator_time_s + gas_ff)
    reg_frac = min(1.0, max(0.0, (bottle_p - tank_p) / settings.regulator_head_pa))
    xfer = min(
        req_gas,
        reg_frac * settings.regulator_maximum_flow_kg_s,
        max(0.0, state.bottle_gas_kg) / settings.regulator_time_s,
    )
    valid = (
        isfinite(main_propellant_kg)
        and main_propellant_kg >= -1e-6
        and -1e-6 <= state.buffer_kg <= min(main, settings.buffer_capacity_kg) + 1e-6
        and -1e-8 <= state.acquisition <= 1.0 + 1e-8
        and state.tank_gas_kg >= -1e-8
        and state.bottle_gas_kg >= -1e-8
        and abs(state.tank_gas_kg + state.bottle_gas_kg - settings.helium_total_mass_kg) <= 1e-6
    )
    return FeedRates(
        delivered_flow_kg_s=delivered,
        pickup_flow_kg_s=pickup,
        buffer_derivative_kg_s=pickup - delivered,
        acquisition_derivative_s_inv=(settled - acq) / acq_tau,
        tank_gas_derivative_kg_s=xfer,
        bottle_gas_derivative_kg_s=-xfer,
        tank_pressure_pa=tank_p,
        bottle_pressure_pa=bottle_p,
        demand_shortfall_kg_s=demand - delivered,
        inventory_valid=bool(valid),
    )


def ignition_ready(
    state: FeedState,
    tank_pressure_pa: float,
    settings: IgnitionSettings = IgnitionSettings(),
) -> bool:
    return bool(
        state.buffer_kg >= settings.minimum_buffer_kg
        and tank_pressure_pa >= settings.minimum_pressure_pa
    )


def separation_event_3d(
    *,
    stack_pos_com_enu_m: ArrayLike,
    stack_vel_com_enu_m_s: ArrayLike,
    quat_wxyz: ArrayLike,
    omega_body_rad_s: ArrayLike,
    booster_mass_kg: float,
    upper_mass_kg: float,
    booster_cg_station_m: float,
    upper_cg_station_m: float,
    booster_inertia_diag_kg_m2: tuple[float, float, float],
    upper_inertia_diag_kg_m2: tuple[float, float, float],
    settings: Separation3DSettings = Separation3DSettings(),
) -> dict[str, NDArray[np.float64] | float]:
    """Momentum-conserving 6DOF cold pusher separation at the interstage interface."""
    p_stack = np.asarray(stack_pos_com_enu_m, dtype=np.float64).reshape(3)
    v_stack = np.asarray(stack_vel_com_enu_m_s, dtype=np.float64).reshape(3)
    om_b = np.asarray(omega_body_rad_s, dtype=np.float64).reshape(3)
    rot_wb = quat_to_rotmat(quat_wxyz)

    m_tot = booster_mass_kg + upper_mass_kg
    stack_cg = (booster_mass_kg * booster_cg_station_m + upper_mass_kg * upper_cg_station_m) / m_tot

    r_b_rel = np.array([0.0, 0.0, booster_cg_station_m - stack_cg], dtype=np.float64)
    r_u_rel = np.array([0.0, 0.0, upper_cg_station_m - stack_cg], dtype=np.float64)

    p_b = p_stack + rot_wb @ r_b_rel
    p_u = p_stack + rot_wb @ r_u_rel
    v_b_pre = v_stack + rot_wb @ np.cross(om_b, r_b_rel)
    v_u_pre = v_stack + rot_wb @ np.cross(om_b, r_u_rel)

    # Reduced mass & equal/opposite impulse along body +z_b (with optional tiny misalignment)
    m_red = (booster_mass_kg * upper_mass_kg) / m_tot
    j_mag = m_red * settings.relative_axial_speed_m_s
    mis = radians(settings.impulse_misalignment_deg)
    j_body_on_upper = j_mag * np.array([sin(mis), 0.0, sqrt(max(0.0, 1.0 - sin(mis) ** 2))])
    j_world_on_upper = rot_wb @ j_body_on_upper

    v_u_post = v_u_pre + j_world_on_upper / upper_mass_kg
    v_b_post = v_b_pre - j_world_on_upper / booster_mass_kg

    # Interface application point in body frame relative to each stage COM
    r_app_b = np.array(
        [
            settings.lateral_offset_x_m,
            settings.lateral_offset_y_m,
            settings.application_station_m - booster_cg_station_m,
        ],
        dtype=np.float64,
    )
    r_app_u = np.array(
        [
            settings.lateral_offset_x_m,
            settings.lateral_offset_y_m,
            settings.application_station_m - upper_cg_station_m,
        ],
        dtype=np.float64,
    )

    ib = np.asarray(booster_inertia_diag_kg_m2, dtype=np.float64)
    iu = np.asarray(upper_inertia_diag_kg_m2, dtype=np.float64)
    d_om_b = np.cross(r_app_b, -j_body_on_upper) / ib
    d_om_u = np.cross(r_app_u, j_body_on_upper) / iu
    om_b_post = om_b + d_om_b
    om_u_post = om_b + d_om_u

    # Audit linear momentum conservation
    p_lin_pre = m_tot * v_stack
    p_lin_post = booster_mass_kg * v_b_post + upper_mass_kg * v_u_post
    lin_res = float(np.linalg.norm(p_lin_post - p_lin_pre))

    return {
        "booster_pos_enu_m": p_b,
        "booster_vel_enu_m_s": v_b_post,
        "booster_omega_body_rad_s": om_b_post,
        "upper_pos_enu_m": p_u,
        "upper_vel_enu_m_s": v_u_post,
        "upper_omega_body_rad_s": om_u_post,
        "linear_momentum_residual_kg_m_s": lin_res,
        "pulse_accel_rad_s2": float(np.linalg.norm(d_om_b) / settings.effective_duration_s),
    }
