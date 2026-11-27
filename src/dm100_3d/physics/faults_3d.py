"""7-engine individual cluster dynamics, nozzle side-loads, and Engine-Out / Fin-Jam faults (M3).

Implements DM100 3D v2 CDR Rev 3.1 §2.6:
  - 7-engine nozzle layout: engine 0 at center (0, 0), engines 1..6 on 60 deg ring at r = 0.92 m
    (outer nozzle rim = 0.92 + 0.253 = 1.173 m, giving 0.749 m clearance on 2.5 deg sloped pad).
  - Individual per-engine thrust bias (+/-1.5%), per-engine ignition delay (+/-15 ms), and
    individual first-order throttle/spindown integration (`tau_fail = 0.20..0.30 s` on EO).
  - Atmospheric-pressure-scaled ignition nozzle side-load pulse:
    F_side = F_peak * (p_amb / 101,325 Pa) * sin(pi * dt / 0.15 s), F_peak in [1.5, 3.0] kN.
  - Asymmetric thrust-offset body moment: sum_i (r_eng_i - r_cg) x T_i_vec.
  - Grid-fin jam injection: locks fin `j` at instantaneous angle or after rate-limited runaway
    to target `fin_jam_angle_deg` (`+/-6..10 deg`).
  - Strictly isolated RNG (`SeedSequence([seed, 70501])`) — zero RNG consumption when disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, pi, radians, sin

import numpy as np

from dm100_3d.config import DM100VehicleConfig
from dm100_3d.physics.forces_3d import compute_engine_thrust_3d

ENGINE_RING_RADIUS_M: float = 0.92
ENGINE_OUTER_RIM_RADIUS_M: float = 1.173

# Exact body-frame (x_b, y_b) nozzle centers [m]: 0=center, 1..6=60 deg*(i-1) @ r=0.92 m
ENGINE_NOZZLE_XY_M: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (ENGINE_RING_RADIUS_M * cos(radians(0.0)), ENGINE_RING_RADIUS_M * sin(radians(0.0))),
    (ENGINE_RING_RADIUS_M * cos(radians(60.0)), ENGINE_RING_RADIUS_M * sin(radians(60.0))),
    (ENGINE_RING_RADIUS_M * cos(radians(120.0)), ENGINE_RING_RADIUS_M * sin(radians(120.0))),
    (ENGINE_RING_RADIUS_M * cos(radians(180.0)), ENGINE_RING_RADIUS_M * sin(radians(180.0))),
    (ENGINE_RING_RADIUS_M * cos(radians(240.0)), ENGINE_RING_RADIUS_M * sin(radians(240.0))),
    (ENGINE_RING_RADIUS_M * cos(radians(300.0)), ENGINE_RING_RADIUS_M * sin(radians(300.0))),
)

# Symmetric 3-engine triplets (center + opposite outer pair): exact zero net lateral offset
SYMMETRIC_3ENG_MASKS: tuple[tuple[bool, bool, bool, bool, bool, bool, bool], ...] = (
    (True, True, False, False, True, False, False),  # [0, 1, 4]
    (True, False, True, False, False, True, False),  # [0, 2, 5]
    (True, False, False, True, False, False, True),  # [0, 3, 6]
)


def default_engine_mask_for_count(active_engines: int) -> tuple[bool, ...]:
    """Return canonical symmetric 7-engine boolean mask for 0, 1, 3, or 7 engines."""
    if active_engines >= 7:
        return (True, True, True, True, True, True, True)
    if active_engines >= 3:
        return SYMMETRIC_3ENG_MASKS[0]
    if active_engines >= 1:
        return (True, False, False, False, False, False, False)
    return (False, False, False, False, False, False, False)


@dataclass(frozen=True, slots=True)
class FaultSettings3D:
    """Configuration for 7-engine cluster dispersions, side-loads, and injected faults."""

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
    side_load_duration_s: float = 0.15
    # Optional Engine-Out fault injection
    engine_out_index: int | None = None
    engine_out_time_s: float | None = None
    engine_out_phase: str | None = None
    engine_out_phase_elapsed_s: float = 0.0
    engine_out_spindown_tau_s: float = 0.25
    # Optional Grid-Fin Jam fault injection
    fin_jam_index: int | None = None
    fin_jam_time_s: float | None = None
    fin_jam_phase: str | None = None
    fin_jam_phase_elapsed_s: float = 0.0
    fin_jam_angle_deg: float | None = None


@dataclass(frozen=True, slots=True)
class ClusterStepOutput3D:
    """Per-substep output from the 7-engine cluster & fault physics module."""

    engine_throttles_7: tuple[float, float, float, float, float, float, float]
    total_throttle_equiv: float
    total_thrust_n: float
    total_main_flow_kg_s: float
    asymmetric_torque_body_nm: tuple[float, float, float]
    side_load_force_body_n: tuple[float, float, float]
    side_load_torque_body_nm: tuple[float, float, float]
    chamber_pressure_ratios_7: tuple[float, float, float, float, float, float, float]
    engine_failed_7: tuple[bool, bool, bool, bool, bool, bool, bool]
    fin_jammed_4: tuple[bool, bool, bool, bool]
    fault_active_now: bool
    fault_onset_time_s: float | None


class EngineClusterFaultModel3D:
    """Stateful 7-engine cluster & fin-jam fault plant with isolated SeedSequence RNG."""

    def __init__(
        self,
        *,
        seed: int,
        vehicle: DM100VehicleConfig,
        settings: FaultSettings3D | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.settings = settings or FaultSettings3D()
        ss = np.random.SeedSequence([int(seed), 70501])
        self._rng = np.random.default_rng(ss)

        self.engine_throttles = np.zeros(7, dtype=np.float64)
        self.engine_failed = [False] * 7
        self._eng_cmd_on_time: list[float | None] = [None] * 7
        self._eng_ign_pulse_start: list[float | None] = [None] * 7
        self._eng_ign_batch_n: list[int] = [1] * 7
        self._eng_pulse_az: list[float] = [
            float(self._rng.uniform(0.0, 2.0 * pi)) for _ in range(7)
        ]
        self._prev_mask = [False] * 7

        # Fin-jam state
        self.fin_jammed = [False, False, False, False]
        self._fin_jam_target_rad: list[float | None] = [None, None, None, None]
        self._fin_jam_locked = [False, False, False, False]

        # Phase tracking for phase-relative fault injection
        self._phase_entry_time: dict[str, float] = {}
        self.engine_out_triggered_time_s: float | None = None
        self.fin_jam_triggered_time_s: float | None = None

    @property
    def first_fault_time_s(self) -> float | None:
        times = [
            t
            for t in (self.engine_out_triggered_time_s, self.fin_jam_triggered_time_s)
            if t is not None
        ]
        return min(times) if times else None

    def _should_trigger(
        self,
        *,
        time_s: float,
        phase: str,
        trig_time_s: float | None,
        trig_phase: str | None,
        trig_phase_elapsed_s: float,
    ) -> bool:
        if trig_time_s is not None and time_s >= trig_time_s - 1e-9:
            return True
        if trig_phase is not None and phase == trig_phase:
            t0 = self._phase_entry_time.get(phase)
            if t0 is not None and (time_s - t0) >= trig_phase_elapsed_s - 1e-9:
                return True
        return False

    def step_cluster(
        self,
        *,
        time_s: float,
        dt_s: float,
        phase: str,
        req_total_throttle: float,
        active_engines: int,
        engine_mask_7: tuple[bool, ...] | None,
        engine_throttles_cmd_7: tuple[float, ...] | None,
        engine_tau_s: float,
        ambient_pressure_pa: float,
        cg_station_m: float,
        gimbal_rad: tuple[float, float],
        thrust_scale: float,
        delivered_ratio: float,
        main_prop_kg: float,
        touchdown_detected: bool,
        body_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> ClusterStepOutput3D:
        """Advance 7 individual engine throttles, side-loads, and asymmetric EO moments."""
        s = self.settings
        if phase not in self._phase_entry_time:
            self._phase_entry_time[phase] = time_s

        # 1. Check Engine-Out fault trigger
        if (
            s.engine_out_index is not None
            and 0 <= s.engine_out_index < 7
            and not self.engine_failed[s.engine_out_index]
            and self._should_trigger(
                time_s=time_s,
                phase=phase,
                trig_time_s=s.engine_out_time_s,
                trig_phase=s.engine_out_phase,
                trig_phase_elapsed_s=s.engine_out_phase_elapsed_s,
            )
        ):
            self.engine_failed[s.engine_out_index] = True
            self.engine_out_triggered_time_s = time_s

        # 2. Resolve per-engine commanded normalized throttle in [0, 1]
        if touchdown_detected or main_prop_kg <= 0.0 or req_total_throttle <= 1e-6:
            mask = (False,) * 7
            cmd_per = [0.0] * 7
        else:
            if engine_mask_7 is not None and len(engine_mask_7) == 7:
                mask = tuple(bool(v) for v in engine_mask_7)
            else:
                mask = default_engine_mask_for_count(active_engines)
            n_on = sum(1 for v in mask if v)
            if engine_throttles_cmd_7 is not None and len(engine_throttles_cmd_7) == 7:
                cmd_per = [
                    max(0.0, min(1.0, float(engine_throttles_cmd_7[i]))) if mask[i] else 0.0
                    for i in range(7)
                ]
            elif n_on > 0:
                # Total throttle is normalized to 7 engines: per-engine = req_total * 7 / n_on
                per_val = max(0.0, min(1.0, float(req_total_throttle) * (7.0 / n_on)))
                cmd_per = [per_val if mask[i] else 0.0 for i in range(7)]
            else:
                cmd_per = [0.0] * 7

        # 3. Integrate each engine's throttle with individual delay and EO turbopump spindown
        tau_nom = max(float(engine_tau_s), 1e-3)
        tau_fail = max(float(s.engine_out_spindown_tau_s), 0.05)
        alpha_nom = 1.0 - exp(-dt_s / tau_nom)
        alpha_fail = 1.0 - exp(-dt_s / tau_fail)
        min_ign = min(s.engine_ign_delay_s_7)

        n_new_ign = sum(1 for j in range(7) if mask[j] and not self._prev_mask[j])
        for i in range(7):
            if mask[i] and not self._prev_mask[i]:
                # Shift per-engine delay >= 0 so fastest engine starts immediately
                self._eng_cmd_on_time[i] = time_s + max(
                    0.0, float(s.engine_ign_delay_s_7[i]) - min_ign
                )
                self._eng_ign_pulse_start[i] = None
                self._eng_ign_batch_n[i] = max(1, n_new_ign)
            elif not mask[i]:
                self._eng_cmd_on_time[i] = None
                self._eng_ign_pulse_start[i] = None
                self._eng_ign_batch_n[i] = 1

            if self.engine_failed[i]:
                # Realistic turbopump spindown to zero thrust
                self.engine_throttles[i] += (0.0 - self.engine_throttles[i]) * alpha_fail
                if self.engine_throttles[i] < 1e-5:
                    self.engine_throttles[i] = 0.0
            elif mask[i]:
                t_start = self._eng_cmd_on_time[i]
                if t_start is None or time_s >= t_start - 1e-9:
                    if self._eng_ign_pulse_start[i] is None and self.engine_throttles[i] < 0.05:
                        self._eng_ign_pulse_start[i] = time_s
                    self.engine_throttles[i] += (cmd_per[i] - self.engine_throttles[i]) * alpha_nom
            else:
                self.engine_throttles[i] += (0.0 - self.engine_throttles[i]) * alpha_nom
                if self.engine_throttles[i] < 1e-6:
                    self.engine_throttles[i] = 0.0

            self._prev_mask[i] = mask[i]

        if main_prop_kg <= 0.0 or touchdown_detected:
            self.engine_throttles[:] = 0.0

        # 4. Compute exact per-engine thrust, flow, asymmetric CG moment, and side-load pulse
        full_t_7, full_flow_7 = compute_engine_thrust_3d(
            1.0, ambient_pressure_pa, self.vehicle, thrust_scale=thrust_scale
        )
        t_one_full = (full_t_7 / 7.0) * delivered_ratio
        flow_one_full = (full_flow_7 / 7.0) * delivered_ratio

        # Conical 2-axis gimbal unit vector in body frame (matches forces_3d.py)
        g_lim = radians(self.vehicle.propulsion.gimbal_limit_deg)
        gy, gx = float(gimbal_rad[0]), float(gimbal_rad[1])
        g_mag = hypot(gy, gx)
        if g_mag > g_lim and g_mag > 1e-12:
            sc_g = g_lim / g_mag
            gy *= sc_g
            gx *= sc_g
            g_mag = g_lim
        if g_mag > 1e-12:
            s_g = sin(g_mag) / g_mag
            u_gim = np.array([gy * s_g, gx * s_g, cos(g_mag)], dtype=np.float64)
        else:
            u_gim = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        total_thrust = 0.0
        total_flow = 0.0
        eff_thr_sum = 0.0
        ox, oy, oz = body_offset_body_m
        z_noz_rel = self.vehicle.geometry.nozzle_station_m - cg_station_m + oz

        # Asymmetric torque = sum_i(r_eng_i x T_i) - center(r_noz x sum T_i),
        # so forces_3d.py computes the standard center TVC torque and we add pure offset moment.
        asym_tau = np.zeros(3, dtype=np.float64)
        side_f = np.zeros(3, dtype=np.float64)
        side_tau = np.zeros(3, dtype=np.float64)
        ratios_7: list[float] = []

        p_ratio_amb = max(0.0, min(1.2, ambient_pressure_pa / 101_325.0))
        dur_sl = max(float(s.side_load_duration_s), 0.02)

        for i in range(7):
            bias_i = float(s.engine_thrust_bias_7[i])
            thr_i = float(self.engine_throttles[i])
            eff_i = thr_i * bias_i
            eff_thr_sum += eff_i
            t_i = eff_i * t_one_full
            total_thrust += t_i
            total_flow += eff_i * flow_one_full
            # Normalized chamber pressure ratio P_c,i / P_c,full (with thrust bias & spindown)
            ratios_7.append(max(0.0, eff_i * delivered_ratio))

            nx_i, ny_i = ENGINE_NOZZLE_XY_M[i]
            if t_i > 1e-3 and (nx_i != 0.0 or ny_i != 0.0):
                # Pure lateral offset arm dr_i = (nx_i, ny_i, 0) x (t_i * u_gim)
                t_vec = t_i * u_gim
                asym_tau[0] += ny_i * t_vec[2]
                asym_tau[1] += -nx_i * t_vec[2]
                asym_tau[2] += nx_i * t_vec[1] - ny_i * t_vec[0]

            # Atmospheric-pressure-scaled ignition nozzle side-load pulse (0.15 s half-sine)
            t_ign0 = self._eng_ign_pulse_start[i]
            if (
                t_ign0 is not None
                and s.side_load_peak_n > 0.0
                and p_ratio_amb > 1e-4
                and not self.engine_failed[i]
            ):
                dt_ign = time_s - t_ign0
                if 0.0 <= dt_ign <= dur_sl:
                    # Cluster side-load peak (CDR §2.6: 1.5-3.0 kN) distributed across
                    # simultaneously igniting engines so net resultant stays <= side_load_peak_n
                    f_sl = (
                        (s.side_load_peak_n / max(1, self._eng_ign_batch_n[i]))
                        * p_ratio_amb
                        * sin(pi * dt_ign / dur_sl)
                    )
                    az = self._eng_pulse_az[i]
                    fx_sl = f_sl * cos(az)
                    fy_sl = f_sl * sin(az)
                    side_f[0] += fx_sl
                    side_f[1] += fy_sl
                    r_sl = np.array([nx_i + ox, ny_i + oy, z_noz_rel], dtype=np.float64)
                    side_tau += np.cross(r_sl, np.array([fx_sl, fy_sl, 0.0], dtype=np.float64))

        return ClusterStepOutput3D(
            engine_throttles_7=tuple(float(v) for v in self.engine_throttles),  # type: ignore[arg-type]
            total_throttle_equiv=float(eff_thr_sum / 7.0),
            total_thrust_n=float(total_thrust),
            total_main_flow_kg_s=float(total_flow),
            asymmetric_torque_body_nm=(float(asym_tau[0]), float(asym_tau[1]), float(asym_tau[2])),
            side_load_force_body_n=(float(side_f[0]), float(side_f[1]), float(side_f[2])),
            side_load_torque_body_nm=(float(side_tau[0]), float(side_tau[1]), float(side_tau[2])),
            chamber_pressure_ratios_7=tuple(ratios_7),  # type: ignore[arg-type]
            engine_failed_7=tuple(self.engine_failed),  # type: ignore[arg-type]
            fin_jammed_4=tuple(self.fin_jammed),  # type: ignore[arg-type]
            fault_active_now=any(self.engine_failed) or any(self.fin_jammed),
            fault_onset_time_s=self.first_fault_time_s,
        )

    def apply_fin_jam_dynamics(
        self,
        *,
        time_s: float,
        dt_s: float,
        phase: str,
        fin_deflection: list[float],
        cmd_fin_deflection_rad: tuple[float, float, float, float],
        fin_rate_rad_s: float,
        fin_tau_s: float,
        fin_limit_rad: float,
        dynamic_pressure_pa: float = 0.0,
    ) -> list[float]:
        """Update 4 grid-fin deflections, enforcing Fin-Jam trigger / runaway / lock."""
        s = self.settings
        if phase not in self._phase_entry_time:
            self._phase_entry_time[phase] = time_s

        if (
            s.fin_jam_index is not None
            and 0 <= s.fin_jam_index < 4
            and not self.fin_jammed[s.fin_jam_index]
            and self._should_trigger(
                time_s=time_s,
                phase=phase,
                trig_time_s=s.fin_jam_time_s,
                trig_phase=s.fin_jam_phase,
                trig_phase_elapsed_s=s.fin_jam_phase_elapsed_s,
            )
        ):
            j = s.fin_jam_index
            self.fin_jammed[j] = True
            self.fin_jam_triggered_time_s = time_s
            if s.fin_jam_angle_deg is None:
                self._fin_jam_target_rad[j] = float(fin_deflection[j])
                self._fin_jam_locked[j] = True
            else:
                tgt = max(-fin_limit_rad, min(fin_limit_rad, radians(float(s.fin_jam_angle_deg))))
                self._fin_jam_target_rad[j] = tgt
                self._fin_jam_locked[j] = False

        # Physical hydraulic runaway rate derating under aerodynamic hinge load
        # (60 deg/s unloaded at q=0; ~24 deg/s at 13 kPa; ~15 deg/s at peak 25 kPa supersonic q)
        jam_rate_eff = fin_rate_rad_s / (1.0 + max(0.0, float(dynamic_pressure_pa)) / 8_500.0)
        out = list(fin_deflection)
        for i in range(4):
            if self.fin_jammed[i]:
                tgt = self._fin_jam_target_rad[i]
                if tgt is None or self._fin_jam_locked[i]:
                    continue
                # Rate-limited actuator runaway until reaching jam stop, then hard-locked
                diff = tgt - out[i]
                max_step = jam_rate_eff * dt_s
                if abs(diff) <= max_step + 1e-9:
                    out[i] = tgt
                    self._fin_jam_locked[i] = True
                else:
                    out[i] += max_step if diff > 0.0 else -max_step
            else:
                df = cmd_fin_deflection_rad[i] - out[i]
                step_f = max(
                    -fin_rate_rad_s * dt_s,
                    min(fin_rate_rad_s * dt_s, df * (1.0 - exp(-dt_s / fin_tau_s))),
                )
                out[i] = max(-fin_limit_rad, min(fin_limit_rad, out[i] + step_f))
        return out
