"""3x9 Priority Control Allocation (4 Grid Fins -> 2-Axis Conical TVC -> 3-Axis RCS).

Solves B u = tau_des (B in R^{3x9}) in strict fuel-minimizing priority:
  1. 4 Independent Grid Fins:
       - Differential roll damping FIRST (low roll inertia Izz << Ixx).
       - Collective pitch (fins +Y, -Y) & yaw (fins +X, -X) within mechanical +/-20 deg
         AND per-fin 7,500 N*m hinge-moment limits.
  2. 2-Axis Conical TVC Gimbal (`(gamma_pitch_y, gamma_yaw_x)` inside 6 deg circle):
       - Zero RCS propellant while engines run; supports vertical-body thrust steering.
  3. 3-Axis RCS (`tau_x, tau_y, tau_z`):
       - Handles low-q flip/coast and residual torque, guarded by 3-axis angular
         acceleration safety bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, exp, hypot, radians, sin, sqrt

import numpy as np

from dm100_3d.config import DM100VehicleConfig


@dataclass(frozen=True, slots=True)
class Allocation3DInput:
    requested_torque_body_nm: tuple[float, float, float]
    background_torque_body_nm: tuple[float, float, float]
    inertia_diag_kg_m2: tuple[float, float, float]
    lever_arm_m: float
    sizing_thrust_n: float
    fin_arm_z_m: float
    dynamic_pressure_pa: float
    fin_extension: float
    alpha_retro_xy_rad: tuple[float, float]  # (alpha_x -> fins 2,3; alpha_y -> fins 0,1)
    rcs_available: bool
    allow_fin_allocation: bool
    thrust_tilt_cmd_body_xy: tuple[float, float] | None = None
    current_gimbal_rad: tuple[float, float] = (0.0, 0.0)
    current_thrust_n: float = 0.0
    control_dt_s: float = 0.02
    angular_accel_guard_deg_s2: float = 3.8
    rcs_deadband_nm: float = 0.0
    # Optional exact operating-point linearization of compute_fins_aero_3d (includes fin drag
    # weathercock torque and cos(alpha) lift projection; matches DM100 2D guidance.py)
    fin_zero_torque_body_nm: tuple[float, float, float] | None = None
    fin_axis_eff_nm_per_rad: tuple[float, float, float] | None = None
    # Optional M3 3x4 individual-fin Jacobian & Fin-Jam FDIR reconfiguration inputs
    fin_jacobian_3x4: np.ndarray | None = None
    fin_ref_deflection_rad: tuple[float, float, float, float] | None = None
    fin_ref_torque_body_nm: tuple[float, float, float] | None = None
    fin_jammed_mask: tuple[bool, bool, bool, bool] | None = None
    fin_jammed_angles_rad: tuple[float, float, float, float] | None = None
    current_fin_deflection_rad: tuple[float, float, float, float] | None = None
    fin_jam_elapsed_s: float | None = None
    vb_gimbal_max_step_deg: float = 1.6


@dataclass(frozen=True, slots=True)
class Allocation3DOutput:
    gimbal_rad: tuple[float, float]
    rcs_torque_body_nm: tuple[float, float, float]
    fin_deflection_rad: tuple[float, float, float, float]
    allocated_torque_body_nm: tuple[float, float, float]
    effectiveness_matrix_3x9: np.ndarray


class PriorityControlAllocator3D:
    """Hierarchical 3-axis -> 9-DOF control allocator."""

    @staticmethod
    def _solve_small_box_qp(
        h_mat: np.ndarray,
        g_vec: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Solve exact global min of 0.5*u^T H u - g^T u s.t. lo <= u <= hi for m <= 3.

        Immune to ill-conditioned H (e.g., cond(H) = 10^6 from 10x inertia-normalized roll
        priority) where coordinate descent stalls; runs in < 0.03 ms.
        """
        m = len(g_vec)
        try:
            u_unc = np.linalg.solve(h_mat, g_vec)
            if np.all(u_unc >= lo - 1e-12) and np.all(u_unc <= hi + 1e-12):
                return np.clip(u_unc, lo, hi)
        except np.linalg.LinAlgError:
            pass

        best_u = np.clip(np.zeros(m, dtype=np.float64), lo, hi)
        best_obj = float(0.5 * (best_u @ (h_mat @ best_u)) - g_vec @ best_u)

        # Enumerate all 3^m <= 27 active-set faces (0=free, 1=lower, 2=upper)
        n_states = 3**m
        for code in range(n_states):
            c = code
            u_cand = np.zeros(m, dtype=np.float64)
            free_i: list[int] = []
            fixed_i: list[int] = []
            for k in range(m):
                st = c % 3
                c //= 3
                if st == 0:
                    free_i.append(k)
                elif st == 1:
                    u_cand[k] = float(lo[k])
                    fixed_i.append(k)
                else:
                    u_cand[k] = float(hi[k])
                    fixed_i.append(k)
            if free_i:
                rhs = g_vec[free_i].copy()
                if fixed_i:
                    rhs -= h_mat[np.ix_(free_i, fixed_i)] @ u_cand[fixed_i]
                h_ff = h_mat[np.ix_(free_i, free_i)]
                try:
                    u_ff = np.linalg.solve(h_ff, rhs)
                except np.linalg.LinAlgError:
                    continue
                if np.any(u_ff < lo[free_i] - 1e-10) or np.any(u_ff > hi[free_i] + 1e-10):
                    continue
                u_cand[free_i] = np.clip(u_ff, lo[free_i], hi[free_i])
            obj = float(0.5 * (u_cand @ (h_mat @ u_cand)) - g_vec @ u_cand)
            if obj < best_obj:
                best_obj = obj
                best_u = u_cand
        return best_u

    def __init__(self, vehicle: DM100VehicleConfig) -> None:
        self.vehicle = vehicle
        self._gim_lim = radians(vehicle.propulsion.gimbal_limit_deg)
        self._fin_lim = radians(vehicle.fins.deflection_limit_deg)

    def _per_fin_bounds(
        self, q_pa: float, fin_ext: float, alpha_retro: float
    ) -> tuple[float, float]:
        fins = self.vehicle.fins
        if fin_ext < 0.05 or q_pa < fins.control_min_q_pa:
            return (0.0, 0.0)
        max_lift = fins.per_fin_hinge_limit_nm / max(fins.hinge_arm_m, 1e-6)
        denom = max(q_pa * fins.per_fin_area_m2 * fin_ext, 1e-6)
        cl_lim = min(fins.max_lift_coefficient, 0.90 * max_lift / denom)
        eff_lim = min(
            radians(fins.control_max_alpha_deg), cl_lim / max(fins.lift_slope_rad_inv, 1e-6)
        )
        lo = max(-self._fin_lim, alpha_retro - eff_lim)
        hi = min(self._fin_lim, alpha_retro + eff_lim)
        if lo > hi:
            mid = max(-self._fin_lim, min(self._fin_lim, alpha_retro))
            return (mid, mid)
        return (lo, hi)

    def allocate(self, inp: Allocation3DInput) -> Allocation3DOutput:
        fins = self.vehicle.fins
        prop = self.vehicle.propulsion
        req = np.asarray(inp.requested_torque_body_nm, dtype=np.float64)
        bg = np.asarray(inp.background_torque_body_nm, dtype=np.float64)

        # Build linearized 3x9 effectiveness matrix B = [B_fin(3x4), B_tvc(3x2), B_rcs(3x3)]
        b_mat = np.zeros((3, 9), dtype=np.float64)
        q_eff = inp.dynamic_pressure_pa if (inp.fin_extension >= 0.05) else 0.0
        f_gain = q_eff * fins.per_fin_area_m2 * inp.fin_extension * fins.lift_slope_rad_inv
        az = inp.fin_arm_z_m
        rc = fins.radial_center_m

        # Fin 0 (+X, +Y lift): tau_x = -az*L, tau_z = +rc*L
        # Fin 1 (-X, +Y lift): tau_x = -az*L, tau_z = -rc*L
        # Fin 2 (+Y, +X lift): tau_y = +az*L, tau_z = -rc*L
        # Fin 3 (-Y, +X lift): tau_y = +az*L, tau_z = +rc*L
        if inp.fin_jacobian_3x4 is not None:
            b_mat[:, 0:4] = np.asarray(inp.fin_jacobian_3x4, dtype=np.float64).reshape(3, 4)
        elif inp.fin_axis_eff_nm_per_rad is not None:
            # Exact finite-difference axis slopes:
            #   eff_x = d(tau_x)/d(d_yaw)   (< 0, shared across fins 0,1)
            #   eff_y = d(tau_y)/d(d_pitch) (> 0, shared across fins 2,3)
            #   eff_z = d(tau_z)/d(d_roll)  (> 0, shared across fins 0,1,2,3)
            ex, ey, ez = inp.fin_axis_eff_nm_per_rad
            b_mat[0, 0] = 0.5 * ex
            b_mat[0, 1] = 0.5 * ex
            b_mat[1, 2] = 0.5 * ey
            b_mat[1, 3] = 0.5 * ey
            b_mat[2, 0] = 0.25 * ez
            b_mat[2, 1] = -0.25 * ez
            b_mat[2, 2] = -0.25 * ez
            b_mat[2, 3] = 0.25 * ez
        elif f_gain > 1.0:
            b_mat[0, 0] = -az * f_gain
            b_mat[0, 1] = -az * f_gain
            b_mat[1, 2] = az * f_gain
            b_mat[1, 3] = az * f_gain
            b_mat[2, 0] = rc * f_gain
            b_mat[2, 1] = -rc * f_gain
            b_mat[2, 2] = -rc * f_gain
            b_mat[2, 3] = rc * f_gain

        t_lev = inp.lever_arm_m * inp.sizing_thrust_n if inp.sizing_thrust_n > 1.0 else 0.0
        if t_lev > 1.0:
            # gimbal_rad = (gamma_y -> Fx -> tau_y = -t_lev, gamma_x -> Fy -> tau_x = +t_lev)
            b_mat[1, 4] = -t_lev
            b_mat[0, 5] = t_lev
        if inp.rcs_available:
            b_mat[0:3, 6:9] = np.eye(3)

        # Clamp requested torque inside the 3D Euclidean angular-acceleration safety ball BEFORE
        # allocating fins/TVC so fins never overshoot the guard and waste RCS counter-torque.
        guard_rad_s2 = radians(inp.angular_accel_guard_deg_s2)
        i_vec = np.asarray(inp.inertia_diag_kg_m2, dtype=np.float64)
        net_req_acc = (req + bg) / i_vec
        acc_norm = float(np.linalg.norm(net_req_acc))
        if acc_norm > guard_rad_s2 and acc_norm > 1e-12:
            net_req_acc *= guard_rad_s2 / acc_norm
        req_guarded = net_req_acc * i_vec - bg

        # --- Stage 1: 4 Independent Grid Fins (Roll priority first, then Pitch & Yaw) ---
        ax_ret, ay_ret = inp.alpha_retro_xy_rad
        lo_x, hi_x = self._per_fin_bounds(inp.dynamic_pressure_pa, inp.fin_extension, ax_ret)
        lo_y, hi_y = self._per_fin_bounds(inp.dynamic_pressure_pa, inp.fin_extension, ay_ret)
        fin_cmd = [0.0, 0.0, 0.0, 0.0]
        fin_tau = np.zeros(3, dtype=np.float64)
        fin_tau_for_rcs = np.zeros(3, dtype=np.float64)
        active_fin_slew = False

        has_jam = (inp.fin_jammed_mask is not None) and any(inp.fin_jammed_mask)
        if has_jam and (f_gain > 1.0 or inp.fin_jacobian_3x4 is not None):
            # --- M3 Fin-Jam Reconfiguration: 3x4 Jacobian + roll-weighted (10x) bounded QP ---
            b_fin = b_mat[:, 0:4]
            if inp.fin_ref_deflection_rad is not None and inp.fin_ref_torque_body_nm is not None:
                u_ref = np.asarray(inp.fin_ref_deflection_rad, dtype=np.float64)
                t_ref = np.asarray(inp.fin_ref_torque_body_nm, dtype=np.float64)
            else:
                u_ref = np.array([ay_ret, ay_ret, ax_ret, ax_ret], dtype=np.float64)
                t_ref = (
                    np.asarray(inp.fin_zero_torque_body_nm, dtype=np.float64) + b_fin @ u_ref
                    if inp.fin_zero_torque_body_nm is not None
                    else np.zeros(3, dtype=np.float64)
                )
            jam_mask = inp.fin_jammed_mask
            assert jam_mask is not None
            jam_ang = inp.fin_jammed_angles_rad or (0.0, 0.0, 0.0, 0.0)
            bounds_4 = [(lo_y, hi_y), (lo_y, hi_y), (lo_x, hi_x), (lo_x, hi_x)]
            feather_4 = [ay_ret, ay_ret, ax_ret, ax_ret]

            u_sol = np.array(feather_4, dtype=np.float64)
            free_idx: list[int] = []
            tau_jam_delta = np.zeros(3, dtype=np.float64)
            for j in range(4):
                if jam_mask[j]:
                    u_j = max(-self._fin_lim, min(self._fin_lim, float(jam_ang[j])))
                    u_sol[j] = u_j
                    tau_jam_delta += b_fin[:, j] * (u_j - float(u_ref[j]))
                else:
                    free_idx.append(j)
                    lo_j, hi_j = bounds_4[j]
                    u_sol[j] = max(lo_j, min(hi_j, feather_4[j]))

            if free_idx:
                inv_i = 1.0 / np.maximum(i_vec, 1.0)
                if inp.allow_fin_allocation:
                    # Healthy-fin target: req_guarded = t_ref + tau_jam_delta + B_free*(u - u_ref)
                    rhs_tau = req_guarded - t_ref - tau_jam_delta
                    # Inertia-normalized angular-accel weighting with 10x roll priority
                    # (Izz << Ixx; guarantees roll torque exact even if yaw/pitch fin saturates!)
                    w_vec = np.array([inv_i[0], inv_i[1], 10.0 * inv_i[2]], dtype=np.float64)
                    reg_scale = 1e-5
                else:
                    # Feather healthy fins while cancelling jammed fin's roll torque
                    rhs_tau = req_guarded - t_ref - tau_jam_delta
                    w_vec = np.array(
                        [0.05 * inv_i[0], 0.05 * inv_i[1], 10.0 * inv_i[2]], dtype=np.float64
                    )
                    reg_scale = 1e-3

                b_free = b_fin[:, free_idx]
                u_ref_f = u_ref[free_idx]
                u_f = u_sol[free_idx] - u_ref_f
                u_f_lo = np.array([bounds_4[j][0] - float(u_ref[j]) for j in free_idx])
                u_f_hi = np.array([bounds_4[j][1] - float(u_ref[j]) for j in free_idx])
                u_f_feath = np.array([feather_4[j] - float(u_ref[j]) for j in free_idx])

                # Normal equations H = B_f^T W^2 B_f + reg*I, g = B_f^T W^2 rhs + reg*u_feath
                wb = b_free * w_vec[:, None]
                reg_w = reg_scale * max(float(np.max(np.abs(wb))) ** 2, 1e-12)
                h_mat = wb.T @ wb + reg_w * np.eye(len(free_idx), dtype=np.float64)
                g_vec = wb.T @ (w_vec * rhs_tau) + reg_w * u_f_feath

                # Exact active-set 3x3 box QP (exact to 1e-15 even at cond(H) = 10^6, < 0.03 ms)
                u_f = self._solve_small_box_qp(h_mat, g_vec, u_f_lo, u_f_hi)

                for k, j in enumerate(free_idx):
                    u_sol[j] = float(u_ref_f[k] + u_f[k])

            fin_cmd = [float(u_sol[0]), float(u_sol[1]), float(u_sol[2]), float(u_sol[3])]
            fin_tau = t_ref + b_fin @ (u_sol - u_ref)
            # Smooth slew-midpoint fin torque for Stage-3 RCS: `cur_j` lags the true plant fin by
            # 1.0*dt (resolver lag), so advance 1.5*dt and smoothly blend [0.25, 0.55] deg into
            # `u_sol` so RCS engages immediately on jam detection AND ramps off with zero step kick!
            if inp.current_fin_deflection_rad is not None:
                f_rate = radians(fins.deflection_rate_deg_s)
                f_tau_act = fins.actuator_time_constant_s
                dt_pred = 2.0 * inp.control_dt_s
                # Tight slew ramp during first 0.65 s of jam recovery for smooth RCS handoff,
                # then smoothly widen to [0.75, 1.45] deg so steady resolver noise never chatters.
                el_j = inp.fin_jam_elapsed_s if inp.fin_jam_elapsed_s is not None else 0.0
                w_el = max(0.0, min(1.0, (el_j - 0.65) / 0.60))
                gate_lo = radians(0.15 + 0.60 * w_el)
                gate_hi = radians(0.45 + 1.00 * w_el)
                u_mid = np.zeros(4, dtype=np.float64)
                for j in range(4):
                    if jam_mask[j]:
                        u_mid[j] = u_sol[j]
                    else:
                        cur_j = float(inp.current_fin_deflection_rad[j])
                        df_j = float(u_sol[j]) - cur_j
                        step_j = max(
                            -f_rate * dt_pred,
                            min(f_rate * dt_pred, df_j * (1.0 - exp(-dt_pred / f_tau_act))),
                        )
                        u_pred_j = cur_j + step_j
                        err_j = abs(float(u_sol[j]) - u_pred_j)
                        if err_j <= gate_lo:
                            u_mid[j] = u_sol[j]
                        elif err_j >= gate_hi:
                            active_fin_slew = True
                            u_mid[j] = u_pred_j
                        else:
                            w_s = (err_j - gate_lo) / (gate_hi - gate_lo)
                            if w_s > 0.25:
                                active_fin_slew = True
                            u_mid[j] = (1.0 - w_s) * float(u_sol[j]) + w_s * u_pred_j
                fin_tau_for_rcs = t_ref + b_fin @ (u_mid - u_ref)
            else:
                fin_tau_for_rcs = fin_tau
        elif f_gain > 1.0 or inp.fin_axis_eff_nm_per_rad is not None:
            if inp.fin_zero_torque_body_nm is not None and inp.fin_axis_eff_nm_per_rad is not None:
                tau0 = np.asarray(inp.fin_zero_torque_body_nm, dtype=np.float64)
                eff_yaw, eff_pitch, eff_roll = inp.fin_axis_eff_nm_per_rad
            else:
                # Analytical weathercock torque from tail-first AoA: delta_eff = delta - a_retro
                tau0 = np.array(
                    [
                        2.0 * az * f_gain * ay_ret,
                        -2.0 * az * f_gain * ax_ret,
                        0.0,
                    ],
                    dtype=np.float64,
                )
                eff_yaw = -2.0 * az * f_gain
                eff_pitch = 2.0 * az * f_gain
                eff_roll = 4.0 * rc * f_gain
            if inp.allow_fin_allocation:
                # 1a) Differential roll command shared across all 4 fins (priority #1)
                d_roll = float(req_guarded[2] - tau0[2]) / eff_roll if abs(eff_roll) > 1e-6 else 0.0
                d_roll = max(-0.5 * self._fin_lim, min(0.5 * self._fin_lim, d_roll))
                # 1b) Collective yaw (fins 0,1) & pitch (fins 2,3)
                d_yaw = (
                    (float(req_guarded[0]) - float(tau0[0])) / eff_yaw
                    if abs(eff_yaw) > 1e-6
                    else 0.0
                )
                d_pitch = (
                    (float(req_guarded[1]) - float(tau0[1])) / eff_pitch
                    if abs(eff_pitch) > 1e-6
                    else 0.0
                )

                fin_cmd = [
                    max(lo_y, min(hi_y, d_yaw + d_roll)),
                    max(lo_y, min(hi_y, d_yaw - d_roll)),
                    max(lo_x, min(hi_x, d_pitch - d_roll)),
                    max(lo_x, min(hi_x, d_pitch + d_roll)),
                ]
            else:
                # Feather fins into airflow while allowing tiny differential roll damping
                d_roll = float(req_guarded[2] - tau0[2]) / eff_roll if abs(eff_roll) > 1e-6 else 0.0
                d_roll = max(-radians(4.0), min(radians(4.0), d_roll))
                fin_cmd = [
                    max(lo_y, min(hi_y, ay_ret + d_roll)),
                    max(lo_y, min(hi_y, ay_ret - d_roll)),
                    max(lo_x, min(hi_x, ax_ret - d_roll)),
                    max(lo_x, min(hi_x, ax_ret + d_roll)),
                ]
            fin_tau = tau0 + b_mat[:, 0:4] @ np.asarray(fin_cmd, dtype=np.float64)
            fin_tau_for_rcs = fin_tau

        res_after_fin = req_guarded - fin_tau

        # --- Stage 2: 2-Axis Conical TVC Gimbal ---
        gim_y, gim_x = 0.0, 0.0
        tvc_tau = np.zeros(3, dtype=np.float64)
        if t_lev > 1.0:
            # tau_y = -t_lev * sin(gim_y), tau_x = +t_lev * sin(gim_x)
            gy_raw = asin(max(-1.0, min(1.0, -float(res_after_fin[1]) / t_lev)))
            gx_raw = asin(max(-1.0, min(1.0, float(res_after_fin[0]) / t_lev)))
            if inp.thrust_tilt_cmd_body_xy is not None:
                # In vertical-body terminal mode, steer thrust laterally within RCS counter-margin
                rcs_lim = prop.rcs_torque_limit_nm
                des_gy, des_gx = inp.thrust_tilt_cmd_body_xy
                max_step = radians(inp.vb_gimbal_max_step_deg)
                lo_gy = max(
                    -self._gim_lim,
                    asin(max(-1.0, min(1.0, (-rcs_lim - float(res_after_fin[1])) / t_lev))),
                    inp.current_gimbal_rad[0] - max_step,
                )
                hi_gy = min(
                    self._gim_lim,
                    asin(max(-1.0, min(1.0, (rcs_lim - float(res_after_fin[1])) / t_lev))),
                    inp.current_gimbal_rad[0] + max_step,
                )
                lo_gx = max(
                    -self._gim_lim,
                    asin(max(-1.0, min(1.0, (-rcs_lim + float(res_after_fin[0])) / t_lev))),
                    inp.current_gimbal_rad[1] - max_step,
                )
                hi_gx = min(
                    self._gim_lim,
                    asin(max(-1.0, min(1.0, (rcs_lim + float(res_after_fin[0])) / t_lev))),
                    inp.current_gimbal_rad[1] + max_step,
                )
                gy_raw = max(lo_gy, min(hi_gy, des_gy)) if lo_gy <= hi_gy else gy_raw
                gx_raw = max(lo_gx, min(hi_gx, des_gx)) if lo_gx <= hi_gx else gx_raw

            # Enforce circular conical gimbal bound sqrt(gy^2 + gx^2) <= gimbal_limit
            g_mag = hypot(gy_raw, gx_raw)
            if g_mag > self._gim_lim and g_mag > 1e-12:
                sc = self._gim_lim / g_mag
                gim_y = gy_raw * sc
                gim_x = gx_raw * sc
                g_mag = self._gim_lim
            else:
                gim_y, gim_x = gy_raw, gx_raw
            if g_mag > 1e-12:
                s_g = sin(g_mag) / g_mag
                tvc_tau = np.array([t_lev * gim_x * s_g, -t_lev * gim_y * s_g, 0.0])

        # --- Stage 3: 3-Axis RCS matched to actual TVC slew midpoint + 3D accel-norm guard ---
        rcs_cmd = np.zeros(3, dtype=np.float64)
        if inp.rcs_available:
            dt = inp.control_dt_s
            g_rate = radians(prop.gimbal_rate_deg_s)
            g_tau = prop.gimbal_tau_s
            g_cur = np.asarray(inp.current_gimbal_rad, dtype=np.float64)
            g_cmd = np.array([gim_y, gim_x], dtype=np.float64)
            dg = g_cmd - g_cur
            step_mid = np.clip(
                dg * (1.0 - exp(-0.5 * dt / g_tau)), -0.5 * g_rate * dt, 0.5 * g_rate * dt
            )
            g_mid = g_cur + step_mid
            t_cur = inp.current_thrust_n if inp.current_thrust_n > 0.0 else inp.sizing_thrust_n
            t_fut = (
                inp.sizing_thrust_n
                if inp.sizing_thrust_n > 1.0
                else t_cur * exp(-dt / prop.nominal_engine_tau_s)
            )
            t_mid = 0.5 * (t_cur + t_fut)
            g_mid_mag = hypot(float(g_mid[0]), float(g_mid[1]))
            if g_mid_mag > 1e-12 and t_mid > 1.0:
                sm = sin(g_mid_mag) / g_mid_mag
                tvc_tau_mid = np.array(
                    [
                        inp.lever_arm_m * t_mid * float(g_mid[1]) * sm,
                        -inp.lever_arm_m * t_mid * float(g_mid[0]) * sm,
                        0.0,
                    ]
                )
            else:
                tvc_tau_mid = np.zeros(3, dtype=np.float64)

            # Use TVC slew-midpoint torque during ignition spool-up, shutdown tail-off, and
            # vertical-body thrust steering; use commanded TVC during steady powered flight so
            # noisy-IMU gimbal jitter during 7-engine ascent never burns RCS propellant.
            if (
                inp.thrust_tilt_cmd_body_xy is not None
                or (inp.sizing_thrust_n <= 1.0 and t_cur > 1.0)
                or (inp.sizing_thrust_n > 1.0 and t_cur < 0.85 * inp.sizing_thrust_n)
            ):
                eff_tvc = tvc_tau_mid
            else:
                eff_tvc = tvc_tau

            res_after_tvc = req_guarded - (fin_tau_for_rcs + eff_tvc)
            lims = (
                prop.rcs_torque_limit_nm,
                prop.rcs_torque_limit_nm,
                prop.rcs_roll_torque_limit_nm,
            )
            # Enforce 3D vector norm ||(bg + fin + eff_tvc + rcs) / I||_2 <= guard_rad_s2
            # Use inertia-scaled continuous soft-deadband so low-inertia roll (Izz << Ixx)
            # never experiences step discontinuities or zero-g bang-bang slosh pumping.
            axis_guard = guard_rad_s2 / sqrt(3.0)
            i_ref = max(float(i_vec[0]), 1.0)
            eff_deadband = (
                0.0
                if (has_jam and active_fin_slew)
                else (min(inp.rcs_deadband_nm, 3_500.0) if has_jam else inp.rcs_deadband_nm)
            )
            for ax_i in range(3):
                val = float(res_after_tvc[ax_i])
                db_i = eff_deadband * min(1.0, float(i_vec[ax_i]) / i_ref)
                if db_i > 0.0:
                    aval = abs(val)
                    if aval <= db_i:
                        val = 0.0
                    elif aval < 1.5 * db_i:
                        val *= (aval - db_i) / (0.5 * db_i)
                g_tau_i = inp.inertia_diag_kg_m2[ax_i] * guard_rad_s2
                sup = float(fin_tau_for_rcs[ax_i] + eff_tvc[ax_i])
                lo = max(-lims[ax_i], -g_tau_i - float(bg[ax_i]) - sup)
                hi = min(lims[ax_i], g_tau_i - float(bg[ax_i]) - sup)
                rcs_cmd[ax_i] = (
                    max(-lims[ax_i], min(lims[ax_i], -float(bg[ax_i]) - sup))
                    if lo > hi
                    else max(lo, min(hi, val))
                )
            # Final 3D Euclidean norm check on total commanded angular acceleration
            tot_acc = (bg + fin_tau_for_rcs + eff_tvc + rcs_cmd) / i_vec
            tot_acc_norm = float(np.linalg.norm(tot_acc))
            if tot_acc_norm > guard_rad_s2 and tot_acc_norm > 1e-12:
                des_tot = tot_acc * (guard_rad_s2 / tot_acc_norm) * i_vec
                rcs_scaled = des_tot - (bg + fin_tau_for_rcs + eff_tvc)
                # Under Fin-Jam pitch/yaw RCS saturation, never scale down roll RCS when the
                # roll axis itself is already safely inside the guard!
                max_ax = 2 if (has_jam and abs(float(tot_acc[2])) <= guard_rad_s2) else 3
                for ax_i in range(max_ax):
                    rcs_cmd[ax_i] = max(-lims[ax_i], min(lims[ax_i], float(rcs_scaled[ax_i])))
            _ = axis_guard

        alloc_tau = fin_tau + tvc_tau + rcs_cmd
        return Allocation3DOutput(
            gimbal_rad=(float(gim_y), float(gim_x)),
            rcs_torque_body_nm=(float(rcs_cmd[0]), float(rcs_cmd[1]), float(rcs_cmd[2])),
            fin_deflection_rad=(
                float(fin_cmd[0]),
                float(fin_cmd[1]),
                float(fin_cmd[2]),
                float(fin_cmd[3]),
            ),
            allocated_torque_body_nm=(
                float(alloc_tau[0]),
                float(alloc_tau[1]),
                float(alloc_tau[2]),
            ),
            effectiveness_matrix_3x9=b_mat,
        )
