"""3D spherical-Earth Stage 2 700 km circular orbit evaluator + 25 s post-sep 3D trajectory.

Extends DM100 2D `orbit.py` to 3D separation state `(x, y, z, vx, vy, vz)` while:
  1. Preserving bit-for-bit parity with 2D orbital altitude & propellant budget when vy == 0.
  2. Recording a 50 Hz 3D ENU trajectory (`upper_pos_enu_m`, `upper_quat_wxyz`) for the
     configured post-separation window (`25.0 s`, including 4.0 s coast + Stage 2 vacuum
     engine 240 kN ignition and departure) for Unreal Engine 5.4 separation cinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, pi, radians, sin, sqrt

import numpy as np

from dm100_3d.config import (
    EARTH_EQUATORIAL_SPEED_M_S,
    EARTH_MU_M3_S2,
    EARTH_RADIUS_M,
    G0,
    UpperStageConfig,
)
from dm100_3d.coords import axis_angle_to_quat_wxyz, quat_multiply_wxyz


@dataclass(frozen=True, slots=True)
class UpperStageTrajectorySample:
    time_since_sep_s: float
    pos_enu_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    engine_lit: bool


@dataclass(frozen=True, slots=True)
class UpperStageOrbitResult:
    orbit_inserted: bool
    target_altitude_km: float
    payload_mass_kg: float
    burn1_duration_s: float
    transfer_perigee_km: float | None
    transfer_apogee_km: float | None
    final_perigee_km: float | None
    final_apogee_km: float | None
    circular_speed_m_s: float | None
    circularization_dv_m_s: float | None
    upper_propellant_remaining_kg: float
    replay_samples: tuple[UpperStageTrajectorySample, ...]


def _heading_pitch_to_quat_wxyz(
    heading_from_east_rad: float, pitch_from_horiz_rad: float
) -> tuple[float, float, float, float]:
    """Build ENU unit quaternion with body +z_b pointing along (heading, elevation=pitch)."""
    # Tilt +z_b toward +X by (pi/2 - pitch_from_horiz), then yaw around +Z by heading
    tilt_from_vert = 0.5 * pi - pitch_from_horiz_rad
    q_pitch = axis_angle_to_quat_wxyz((0.0, 1.0, 0.0), tilt_from_vert)
    q_yaw = axis_angle_to_quat_wxyz((0.0, 0.0, 1.0), heading_from_east_rad)
    q = quat_multiply_wxyz(q_yaw, q_pitch)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def evaluate_upper_stage_orbit_3d(
    *,
    pos_sep_enu_m: tuple[float, float, float],
    vel_sep_enu_m_s: tuple[float, float, float],
    latitude_deg: float = 37.0,
    settings: UpperStageConfig | None = None,
    replay_hz: float = 50.0,
) -> UpperStageOrbitResult:
    """Evaluate 3D 2-burn 700 km circular orbit insertion and record 25 s separation trajectory."""
    s = settings or UpperStageConfig()
    x_sep, y_sep, z_sep = (
        float(pos_sep_enu_m[0]),
        float(pos_sep_enu_m[1]),
        float(pos_sep_enu_m[2]),
    )
    vx_sep, vy_sep, vz_sep = (
        float(vel_sep_enu_m_s[0]),
        float(vel_sep_enu_m_s[1]),
        float(vel_sep_enu_m_s[2]),
    )

    # 1. Record local-ENU 50 Hz visual trajectory over [0, replay_post_sep_duration_s]
    rec_dt = 1.0 / max(replay_hz, 1.0)
    n_rec = int(round(s.replay_post_sep_duration_s / rec_dt))
    p_rec = np.array([x_sep, y_sep, z_sep], dtype=np.float64)
    v_rec = np.array([vx_sep, vy_sep, vz_sep], dtype=np.float64)
    m_prop_rec = s.propellant_mass_kg
    m_inert = s.dry_mass_kg + s.payload_mass_kg
    mdot = s.vacuum_thrust_n / (s.vacuum_isp_s * G0)
    horiz_spd0 = hypot(vx_sep, vy_sep)
    heading_rad = atan2(vy_sep, vx_sep) if horiz_spd0 > 1e-3 else 0.0
    hx, hy = cos(heading_rad), sin(heading_rad)

    samples: list[UpperStageTrajectorySample] = []
    for step_idx in range(n_rec + 1):
        t_rel = step_idx * rec_dt
        lit = t_rel >= s.ignition_coast_s and m_prop_rec > 0.0
        burn_t = max(0.0, t_rel - s.ignition_coast_s)
        pitch_rad = radians(max(0.0, 14.0 - 0.06 * burn_t))
        if not lit:
            v_h = hypot(float(v_rec[0]), float(v_rec[1]))
            q_now = _heading_pitch_to_quat_wxyz(heading_rad, atan2(float(v_rec[2]), max(v_h, 1.0)))
        else:
            q_now = _heading_pitch_to_quat_wxyz(heading_rad, pitch_rad)

        samples.append(
            UpperStageTrajectorySample(
                time_since_sep_s=t_rel,
                pos_enu_m=(float(p_rec[0]), float(p_rec[1]), float(p_rec[2])),
                quat_wxyz=q_now,
                engine_lit=lit,
            )
        )
        if step_idx == n_rec:
            break

        r_loc = EARTH_RADIUS_M + max(float(p_rec[2]), 0.0)
        g_loc = EARTH_MU_M3_S2 / (r_loc * r_loc)
        a_cent = (float(v_rec[0]) ** 2 + float(v_rec[1]) ** 2) / r_loc
        ax = 0.0
        ay = 0.0
        az = -g_loc + a_cent
        if lit:
            mass_rec = m_inert + m_prop_rec
            a_thr = s.vacuum_thrust_n / mass_rec
            ax += a_thr * cos(pitch_rad) * hx
            ay += a_thr * cos(pitch_rad) * hy
            az += a_thr * sin(pitch_rad)
            m_prop_rec = max(0.0, m_prop_rec - mdot * rec_dt)
        p_rec += v_rec * rec_dt + 0.5 * np.array([ax, ay, az]) * (rec_dt * rec_dt)
        v_rec += np.array([ax, ay, az]) * rec_dt

    # 2. Exact spherical-Earth 2-burn orbital insertion check (matches DM100 2D orbit.py when vy=0)
    r = EARTH_RADIUS_M + max(z_sep, 0.0)
    v_earth = EARTH_EQUATORIAL_SPEED_M_S * cos(radians(latitude_deg))
    vt_east = vx_sep + v_earth
    vt = hypot(vt_east, vy_sep)
    vr = vz_sep

    dt_coast = 0.2
    for _ in range(int(round(s.ignition_coast_s / dt_coast))):
        g_loc = EARTH_MU_M3_S2 / (r * r)
        ar = -g_loc + (vt * vt) / r
        at = -(vr * vt) / r
        vr += ar * dt_coast
        vt += at * dt_coast
        r += vr * dt_coast

    m_prop = s.propellant_mass_kg
    r_target = EARTH_RADIUS_M + s.target_altitude_m
    dt = 0.2
    burn1_s = 0.0

    while m_prop > 0.0 and burn1_s < 460.0:
        v2 = vr * vr + vt * vt
        eps = 0.5 * v2 - EARTH_MU_M3_S2 / r
        h_ang = r * vt
        step_dt = dt
        if eps < 0.0:
            a_orb = -EARTH_MU_M3_S2 / (2.0 * eps)
            e = sqrt(max(0.0, 1.0 + 2.0 * eps * h_ang * h_ang / (EARTH_MU_M3_S2**2)))
            ra = a_orb * (1.0 + e)
            if ra >= r_target:
                break
            if ra > r_target - 120_000.0:
                step_dt = 0.02
        else:
            break

        mass = m_inert + m_prop
        pitch_rad = radians(max(0.0, 14.0 - 0.06 * burn1_s))
        ar = (s.vacuum_thrust_n * sin(pitch_rad)) / mass - EARTH_MU_M3_S2 / (r * r) + (vt * vt) / r
        at = (s.vacuum_thrust_n * cos(pitch_rad)) / mass - (vr * vt) / r
        vr += ar * step_dt
        vt += at * step_dt
        r += vr * step_dt
        m_prop = max(0.0, m_prop - mdot * step_dt)
        burn1_s += step_dt

    v_end = hypot(vr, vt)
    eps = 0.5 * v_end * v_end - EARTH_MU_M3_S2 / r
    h_ang = r * vt
    target_km = s.target_altitude_m / 1000.0
    if eps >= 0.0 or m_prop <= 0.0:
        return UpperStageOrbitResult(
            orbit_inserted=False,
            target_altitude_km=target_km,
            payload_mass_kg=s.payload_mass_kg,
            burn1_duration_s=burn1_s,
            transfer_perigee_km=None,
            transfer_apogee_km=None,
            final_perigee_km=None,
            final_apogee_km=None,
            circular_speed_m_s=None,
            circularization_dv_m_s=None,
            upper_propellant_remaining_kg=0.0,
            replay_samples=tuple(samples),
        )

    a_orb = -EARTH_MU_M3_S2 / (2.0 * eps)
    e = sqrt(max(0.0, 1.0 + 2.0 * eps * h_ang * h_ang / (EARTH_MU_M3_S2**2)))
    ra = a_orb * (1.0 + e)
    rp = a_orb * (1.0 - e)

    v_at_ra = sqrt(max(0.0, EARTH_MU_M3_S2 * (2.0 / ra - 1.0 / a_orb)))
    v_circ = sqrt(EARTH_MU_M3_S2 / ra)
    dv_circ = max(0.0, v_circ - v_at_ra)
    m_before = m_inert + m_prop
    m_after = m_before / exp(dv_circ / (s.vacuum_isp_s * G0))
    prop_rem = m_after - m_inert
    alt_km = (ra - EARTH_RADIUS_M) / 1000.0
    tol_km = s.altitude_tolerance_m / 1000.0
    ok = (prop_rem > 0.0) and (abs(alt_km - target_km) <= tol_km)

    return UpperStageOrbitResult(
        orbit_inserted=bool(ok),
        target_altitude_km=target_km,
        payload_mass_kg=s.payload_mass_kg,
        burn1_duration_s=burn1_s,
        transfer_perigee_km=(rp - EARTH_RADIUS_M) / 1000.0,
        transfer_apogee_km=alt_km,
        final_perigee_km=alt_km if ok else (rp - EARTH_RADIUS_M) / 1000.0,
        final_apogee_km=alt_km,
        circular_speed_m_s=v_circ,
        circularization_dv_m_s=dv_circ,
        upper_propellant_remaining_kg=max(0.0, prop_rem),
        replay_samples=tuple(samples),
    )
