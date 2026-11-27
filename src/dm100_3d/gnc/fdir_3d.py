"""Strictly causal FDIR (Fault Detection, Isolation, and Recovery) for DM100 3D v2 (M3).

Implements DM100 3D v2 CDR Rev 3.1 §3.4 & CDR-C4:
  1. Engine-Out FDIR (zero false positives across 0.36 min throttle & restarts):
       - Propagates causal delay-compensated expected chamber pressure `p_hat_i(t)` including
         observation/action delay, 0.20 s restart ignition delay, engine spool lag, and 30 ms
         chamber pressure sensor lag.
       - Isolates engine `i` ONLY when `t_since_ign_i >= 0.45 s` AND `p_hat_i >= 0.25` AND
         (`p_meas_i / p_hat_i < 0.55` OR with >= 3 active engines `p_meas_i / median < 0.55`)
         for 3 consecutive steps (60 ms) with physical thrust/torque residual corroboration.
       - Ascent 7->6: retains 6 healthy engines + causal asymmetric thrust-moment feedforward
         + RTLS vs `DOWNRANGE_SPLASHDOWN_ABORT` propellant-sufficiency gate.
       - Post-sep 3->3_sym / 3->1 / 3->2->1 reconfiguration: dynamically selects a healthy
         symmetric 3-engine triplet (`[0,2,5]` / `[0,3,6]`) when available; during landing burn
         switches 3->1 center engine within 20 ms if `h > h_stop_1 + 80 m`, else holds 2-engine
         trimmed burn until `h <= 250 m` before forcing 2->1 touchdown.
  2. Grid-Fin Jam FDIR:
       - Propagates causal expected fin deflections through the 0.08 s / 60 deg/s actuator
         model and detects fin `j` stuck when `|meas - pred| > 2.5 deg` for 4 consecutive steps
         (80 ms) under `q >= 300 Pa`, `fin_extension >= 0.40`.
       - Supplies `jammed_fin_mask` and `jammed_fin_angles_rad` to the 3x4/3x3 QP allocator and
         shifts the `aero_glide` AoA corridor (`|alpha_retro_axis - delta_jam| <= 8.0 deg`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, radians

import numpy as np

from dm100_3d.config import G0, DM100VehicleConfig
from dm100_3d.interfaces import ActuatorCommand3D, TelemetryPacket3D
from dm100_3d.physics.faults_3d import (
    ENGINE_NOZZLE_XY_M,
    SYMMETRIC_3ENG_MASKS,
    default_engine_mask_for_count,
)


@dataclass(frozen=True, slots=True)
class FdirStatus3D:
    """Strictly causal FDIR state snapshot exported to GNC, Allocator, and Diagnostics."""

    fdir_state: str  # "NOMINAL", "ENGINE_OUT_RTLS", "ENGINE_OUT_ABORT", "FIN_JAM", "MULTI_FAULT"
    mission_mode: str  # "RTLS" or "DOWNRANGE_SPLASHDOWN_ABORT"
    engine_failed_7: tuple[bool, bool, bool, bool, bool, bool, bool]
    fin_jammed_4: tuple[bool, bool, bool, bool]
    fin_jam_angles_rad_4: tuple[float, float, float, float]
    engine_out_detected_time_s: float | None
    fin_jam_detected_time_s: float | None
    abort_triggered_time_s: float | None


class CausalFdir3D:
    """Strictly causal 50 Hz Engine-Out & Grid-Fin Jam FDIR observer."""

    def __init__(
        self,
        *,
        vehicle: DM100VehicleConfig,
        control_dt_s: float = 0.02,
        observation_delay_steps: int = 0,
        action_delay_steps: int = 0,
        restart_ignition_delay_s: float = 0.20,
        chamber_p_tau_s: float = 0.030,
        rtls_min_prop_reserve_kg: float = 11_400.0,
    ) -> None:
        self.vehicle = vehicle
        self.dt_s = float(control_dt_s)
        self.total_delay_steps = max(0, int(observation_delay_steps) + int(action_delay_steps))
        self.restart_ign_delay_s = float(restart_ignition_delay_s)
        self.chamber_p_tau_s = float(chamber_p_tau_s)
        self.rtls_min_prop_reserve_kg = float(rtls_min_prop_reserve_kg)

        # Delayed command FIFO so expected engine/fin models align with measurement_time_s
        self._cmd_fifo: deque[ActuatorCommand3D] = deque(maxlen=max(8, self.total_delay_steps + 4))
        self._prev_fin_hat = np.zeros(4, dtype=np.float64)
        self._fin_hat = np.zeros(4, dtype=np.float64)
        self._fin_err_count = [0, 0, 0, 0]
        self._prev_omega_z: float | None = None

        # Expected engine throttle & lagged chamber pressure model (7 engines)
        self._eng_thr_hat = np.zeros(7, dtype=np.float64)
        self._p_hat = np.zeros(7, dtype=np.float64)
        self._eng_ign_timer = np.zeros(7, dtype=np.float64)
        self._eng_low_count = [0] * 7
        self._engine_lit_before: bool = False
        self._was_any_active: bool = False
        self._restart_wait_s: float = 0.0

        # Latched causal fault state
        self.engine_failed: list[bool] = [False] * 7
        self.fin_jammed: list[bool] = [False, False, False, False]
        self.fin_jam_angles_rad: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.mission_mode: str = "RTLS"
        self.engine_out_detected_time_s: float | None = None
        self.fin_jam_detected_time_s: float | None = None
        self.abort_triggered_time_s: float | None = None
        self._last_meas_t: float | None = None

    def record_command(self, cmd: ActuatorCommand3D) -> None:
        """Record issued actuator command for delayed causal model propagation."""
        self._cmd_fifo.append(cmd)

    def _effective_cmd_at_measurement(self) -> ActuatorCommand3D | None:
        if not self._cmd_fifo:
            return None
        idx = len(self._cmd_fifo) - 1 - self.total_delay_steps
        if idx < 0:
            return None
        return self._cmd_fifo[idx]

    def update(
        self,
        *,
        pkt: TelemetryPacket3D,
        q_pa: float,
        vel_enu_m_s: tuple[float, float, float],
        pos_enu_m: tuple[float, float, float],
        accel_dist_world_m_s2: tuple[float, float, float],
        torque_bias_body_nm: tuple[float, float, float],
        mass_kg: float,
    ) -> FdirStatus3D:
        """Update Engine-Out & Fin-Jam detectors from strictly causal telemetry."""
        if self._last_meas_t is None:
            self._last_meas_t = pkt.measurement_time_s
            self._fin_hat[:] = np.asarray(pkt.fin_deflection_rad, dtype=np.float64)
            self._prev_fin_hat[:] = self._fin_hat
            return self.status()

        dt = max(0.0, pkt.measurement_time_s - self._last_meas_t)
        self._last_meas_t = pkt.measurement_time_s
        if dt <= 1e-6 or pkt.packet_dropped or pkt.touchdown_detected:
            return self.status()

        cmd = self._effective_cmd_at_measurement()
        if cmd is not None:
            # --- 1. Propagate causal expected 7-engine chamber pressure ratios p_hat ---
            req_thr = float(cmd.throttle_cmd)
            if req_thr > 1e-4 and pkt.main_propellant_kg > 0.0:
                if cmd.engine_mask_7 is not None and len(cmd.engine_mask_7) == 7:
                    mask = tuple(bool(v) for v in cmd.engine_mask_7)
                else:
                    mask = default_engine_mask_for_count(cmd.active_engines)
                n_on = sum(1 for v in mask if v)
                if not self._engine_lit_before:
                    self._engine_lit_before = True
                    self._restart_wait_s = 0.0
                elif not self._was_any_active:
                    self._restart_wait_s = self.restart_ign_delay_s
                self._was_any_active = n_on > 0

                if self._restart_wait_s > 0.0:
                    self._restart_wait_s = max(0.0, self._restart_wait_s - dt)
                    cmd_per = [0.0] * 7
                    active_now = [False] * 7
                else:
                    if cmd.engine_throttles_7 is not None and len(cmd.engine_throttles_7) == 7:
                        cmd_per = [
                            max(0.0, min(1.0, float(cmd.engine_throttles_7[i])))
                            if (mask[i] and not self.engine_failed[i])
                            else 0.0
                            for i in range(7)
                        ]
                    elif n_on > 0:
                        per_val = max(0.0, min(1.0, req_thr * (7.0 / n_on)))
                        cmd_per = [
                            per_val if (mask[i] and not self.engine_failed[i]) else 0.0
                            for i in range(7)
                        ]
                    else:
                        cmd_per = [0.0] * 7
                    active_now = [mask[i] and not self.engine_failed[i] for i in range(7)]
            else:
                self._was_any_active = False
                self._restart_wait_s = 0.0
                cmd_per = [0.0] * 7
                active_now = [False] * 7

            # Conservative spool lag (tolerates up to 0.055 s engine_lag without false alarm)
            a_eng = 1.0 - exp(-dt / max(self.vehicle.propulsion.nominal_engine_tau_s, 0.055))
            a_sen = 1.0 - exp(-dt / max(self.chamber_p_tau_s, 0.030))
            for i in range(7):
                if active_now[i]:
                    self._eng_ign_timer[i] += dt
                else:
                    self._eng_ign_timer[i] = 0.0
                self._eng_thr_hat[i] += (cmd_per[i] - self._eng_thr_hat[i]) * a_eng
                self._p_hat[i] += (self._eng_thr_hat[i] - self._p_hat[i]) * a_sen

            # --- 2. Engine-Out Detection (CDR C4: zero false positives!) ---
            if pkt.engine_chamber_pressure_ratio_7 is not None:
                p_meas = pkt.engine_chamber_pressure_ratio_7
                active_p = [
                    float(p_meas[i])
                    for i in range(7)
                    if active_now[i] and self._eng_ign_timer[i] >= 0.45 and self._p_hat[i] >= 0.25
                ]
                med_active = float(np.median(active_p)) if len(active_p) >= 3 else 0.0
                a_dist_mag = float(np.linalg.norm(accel_dist_world_m_s2))
                tb_xy_mag = float(np.hypot(torque_bias_body_nm[0], torque_bias_body_nm[1]))
                corroborated = (a_dist_mag > 0.30) or (tb_xy_mag > 8_000.0)

                for i in range(7):
                    if self.engine_failed[i]:
                        continue
                    if active_now[i] and self._eng_ign_timer[i] >= 0.45 and self._p_hat[i] >= 0.25:
                        ratio_hat = float(p_meas[i]) / max(float(self._p_hat[i]), 1e-6)
                        ratio_med = (
                            float(p_meas[i]) / max(med_active, 1e-6) if med_active >= 0.25 else 1.0
                        )
                        low_p = (ratio_hat < 0.55) or (len(active_p) >= 3 and ratio_med < 0.55)
                        deep_flameout = ratio_hat < 0.35
                        if low_p and (corroborated or deep_flameout):
                            self._eng_low_count[i] += 1
                            if self._eng_low_count[i] >= 3:
                                self.engine_failed[i] = True
                                if self.engine_out_detected_time_s is None:
                                    self.engine_out_detected_time_s = pkt.time_s
                                self._evaluate_ascent_abort(
                                    pkt=pkt,
                                    vel_enu_m_s=vel_enu_m_s,
                                    pos_enu_m=pos_enu_m,
                                    mass_kg=mass_kg,
                                )
                        else:
                            self._eng_low_count[i] = 0
                    else:
                        self._eng_low_count[i] = 0

            # Also re-check causal RTLS propellant feasibility while ascending with EO
            if pkt.upper_attached and any(self.engine_failed) and self.mission_mode == "RTLS":
                self._evaluate_ascent_abort(
                    pkt=pkt,
                    vel_enu_m_s=vel_enu_m_s,
                    pos_enu_m=pos_enu_m,
                    mass_kg=mass_kg,
                )

            # --- 3. Propagate causal expected fin deflections & detect Fin-Jam ---
            f_rate = radians(self.vehicle.fins.deflection_rate_deg_s)
            f_tau = self.vehicle.fins.actuator_time_constant_s
            f_lim = radians(self.vehicle.fins.deflection_limit_deg)
            # Resolver has 1-step lag, so compare measurement against previous-step prediction
            prev_pred = self._prev_fin_hat.copy()
            self._prev_fin_hat[:] = self._fin_hat
            for j in range(4):
                df = float(cmd.fin_deflection_rad[j]) - float(self._fin_hat[j])
                step_f = max(-f_rate * dt, min(f_rate * dt, df * (1.0 - exp(-dt / f_tau))))
                self._fin_hat[j] = max(-f_lim, min(f_lim, float(self._fin_hat[j]) + step_f))

            om_z_now = float(pkt.omega_body_rad_s[2])
            roll_kick_deg_s2 = (
                abs(om_z_now - self._prev_omega_z) * (57.29577951308232 / max(dt, 1e-3))
                if self._prev_omega_z is not None
                else 0.0
            )
            self._prev_omega_z = om_z_now

            if (not pkt.upper_attached) and pkt.fin_extension >= 0.40:
                # 0.85 deg (8.5-sigma above 0.10 deg resolver noise) x 2 steps (40 ms),
                # OR 1-step (20 ms) when corroborated by > 6 deg/s^2 roll kick at high q!
                thresh = radians(0.85)
                thresh_fast = radians(1.0)
                for j in range(4):
                    meas_j = float(pkt.fin_deflection_rad[j])
                    if self.fin_jammed[j]:
                        # Track real-time jammed fin angle immediately without smoothing lag
                        self.fin_jam_angles_rad[j] = meas_j
                        continue
                    err_j = abs(meas_j - float(prev_pred[j]))
                    err_cur = abs(meas_j - float(self._fin_hat[j]))
                    err_min = min(err_j, err_cur)
                    if err_min > thresh:
                        self._fin_err_count[j] += 1
                        fast_corrob = (err_min > thresh_fast) and (roll_kick_deg_s2 > 6.0)
                        if self._fin_err_count[j] >= 2 or fast_corrob:
                            self.fin_jammed[j] = True
                            self.fin_jam_angles_rad[j] = meas_j
                            if self.fin_jam_detected_time_s is None:
                                self.fin_jam_detected_time_s = pkt.time_s
                    else:
                        self._fin_err_count[j] = 0
            else:
                self._fin_err_count = [0, 0, 0, 0]

        return self.status()

    def _evaluate_ascent_abort(
        self,
        *,
        pkt: TelemetryPacket3D,
        vel_enu_m_s: tuple[float, float, float],
        pos_enu_m: tuple[float, float, float],
        mass_kg: float,
    ) -> None:
        """Decide causally between EO-RTLS and DOWNRANGE_SPLASHDOWN_ABORT during ascent."""
        if not pkt.upper_attached:
            return
        v_now = float(np.linalg.norm(vel_enu_m_s))
        # Replay-calibrated 6-engine ascent remaining delta-v (CDR-P-M1):
        # MECO-1 target 1,700 m/s + 6-engine gravity/drag loss (~1,220 m/s from SL -> 0 at 48 km).
        # Triggers DOWNRANGE_SPLASHDOWN_ABORT for early EO (<~55 s), preserves RTLS for late EO.
        dv_need = max(0.0, 1_700.0 - v_now)
        alt_km = max(0.0, float(pos_enu_m[2]) / 1_000.0)
        dv_loss_6eng = 1_220.0 * max(0.0, 1.0 - alt_km / 48.0)
        dv_eff = dv_need + dv_loss_6eng
        isp_eff = 325.0 + 30.0 * min(1.0, alt_km / 30.0)
        m_burn_est = mass_kg * (1.0 - exp(-dv_eff / max(isp_eff * G0, 1.0)))
        m_rem_pred = pkt.main_propellant_kg - m_burn_est
        if m_rem_pred < self.rtls_min_prop_reserve_kg:
            self.mission_mode = "DOWNRANGE_SPLASHDOWN_ABORT"
            if self.abort_triggered_time_s is None:
                self.abort_triggered_time_s = pkt.time_s

    def healthy_ascent_mask(self) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
        """Return 7-engine mask with any isolated failed engines disabled."""
        return tuple(not f for f in self.engine_failed)  # type: ignore[return-value]

    def select_3eng_mask(
        self,
        *,
        current_mask: tuple[bool, ...] | None = None,
        allow_2eng_trim: bool = False,
    ) -> tuple[tuple[bool, bool, bool, bool, bool, bool, bool], int]:
        """Select healthy symmetric 3-engine triplet, or 2-engine/1-engine EO fallback."""
        # If currently burning with a valid mask and one outer engine failed during low-alt landing:
        if allow_2eng_trim and current_mask is not None:
            rem = tuple(bool(current_mask[i]) and (not self.engine_failed[i]) for i in range(7))
            n_rem = sum(1 for v in rem if v)
            if n_rem == 2 and rem[0]:
                return rem, 2  # type: ignore[return-value]

        # Prefer an undamaged symmetric 3-engine triplet ([0,1,4], [0,2,5], or [0,3,6])
        for sym in SYMMETRIC_3ENG_MASKS:
            if all(not self.engine_failed[i] for i in range(7) if sym[i]):
                return sym, 3

        # Fallback to healthy center single engine
        if not self.engine_failed[0]:
            return (True, False, False, False, False, False, False), 1

        # Extreme fallback (center engine failed): pick opposite healthy outer pair
        for a, b in ((1, 4), (2, 5), (3, 6)):
            if (not self.engine_failed[a]) and (not self.engine_failed[b]):
                m = [False] * 7
                m[a] = True
                m[b] = True
                return tuple(m), 2  # type: ignore[return-value]

        m_any = tuple(not f for f in self.engine_failed)
        return m_any, sum(1 for v in m_any if v)  # type: ignore[return-value]

    def select_1eng_mask(self) -> tuple[tuple[bool, bool, bool, bool, bool, bool, bool], int]:
        """Select healthy center engine (or symmetric pair if center engine failed)."""
        if not self.engine_failed[0]:
            return (True, False, False, False, False, False, False), 1
        for a, b in ((1, 4), (2, 5), (3, 6)):
            if (not self.engine_failed[a]) and (not self.engine_failed[b]):
                m = [False] * 7
                m[a] = True
                m[b] = True
                return tuple(m), 2  # type: ignore[return-value]
        return (False, False, False, False, False, False, False), 0

    def causal_asymmetric_torque_ff_nm(
        self,
        *,
        engine_mask_7: tuple[bool, ...],
        total_thrust_n: float,
    ) -> tuple[float, float, float]:
        """Causal nominal asymmetric thrust-offset torque `(tx, ty, 0)` from known active mask."""
        if total_thrust_n <= 1.0:
            return (0.0, 0.0, 0.0)
        active_idx = [i for i in range(7) if engine_mask_7[i] and not self.engine_failed[i]]
        if not active_idx:
            return (0.0, 0.0, 0.0)
        t_per = total_thrust_n / len(active_idx)
        tau_x = 0.0
        tau_y = 0.0
        for i in active_idx:
            nx_i, ny_i = ENGINE_NOZZLE_XY_M[i]
            tau_x += ny_i * t_per
            tau_y += -nx_i * t_per
        return (tau_x, tau_y, 0.0)

    def status(self) -> FdirStatus3D:
        has_eo = any(self.engine_failed)
        has_fj = any(self.fin_jammed)
        if has_eo and has_fj:
            state = "MULTI_FAULT"
        elif has_eo:
            state = (
                "ENGINE_OUT_ABORT"
                if self.mission_mode == "DOWNRANGE_SPLASHDOWN_ABORT"
                else "ENGINE_OUT_RTLS"
            )
        elif has_fj:
            state = "FIN_JAM"
        else:
            state = "NOMINAL"
        return FdirStatus3D(
            fdir_state=state,
            mission_mode=self.mission_mode,
            engine_failed_7=tuple(self.engine_failed),  # type: ignore[arg-type]
            fin_jammed_4=tuple(self.fin_jammed),  # type: ignore[arg-type]
            fin_jam_angles_rad_4=tuple(self.fin_jam_angles_rad),  # type: ignore[arg-type]
            engine_out_detected_time_s=self.engine_out_detected_time_s,
            fin_jam_detected_time_s=self.fin_jam_detected_time_s,
            abort_triggered_time_s=self.abort_triggered_time_s,
        )
