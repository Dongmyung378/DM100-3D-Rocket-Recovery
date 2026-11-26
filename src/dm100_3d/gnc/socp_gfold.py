"""Clarabel-based 3D GFOLD SOCP terminal hoverslam planner + 3D zero-tilt analytic backup.

Directly builds Clarabel sparse CSC matrices (zero CVXPY overhead, ~1-3 ms per solve)
for the 3D lossless-convexified minimum-fuel powered descent problem:
  minimize   sum_k sigma_k * dt
  subject to
    3D double-integrator dynamics with gravity g = (0, 0, -g0)
    ||(a_x, a_y, a_z)||_2 <= sigma_k                    (4D thrust-magnitude SOC)
    a_min <= sigma_k <= a_max                           (non-hovering Hoverslam bounds)
    ||(a_x, a_y)||_2 <= tan(theta_lim(k)) * a_z         (3D tilt & terminal verticality cone)
    ||(x_k, y_k)||_2 <= tan(gamma_gs) * z_k + r_margin  (3D glide-slope cone)
    p_0 = (x0, y0, z0), v_0 = (vx0, vy0, vz0) -> p_N = (0, 0, 0), v_N = (0, 0, -v_td)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin, sqrt, tan
from time import perf_counter

import clarabel
import numpy as np
from scipy import sparse


@dataclass(frozen=True, slots=True)
class Gfold3DProblem:
    pos0_enu_m: tuple[float, float, float]
    vel0_enu_m_s: tuple[float, float, float]
    a_min_m_s2: float
    a_max_m_s2: float
    gravity_m_s2: float = 9.80665
    v_touchdown_m_s: float = 0.65
    max_tilt_deg: float = 18.0
    final_tilt_deg: float = 2.0
    final_vertical_fraction: float = 0.25
    glide_slope_max_deg: float = 65.0
    pad_margin_m: float = 2.0
    n_steps: int = 16


@dataclass(frozen=True, slots=True)
class Gfold3DSolution:
    feasible: bool
    status: str
    flight_time_s: float
    dt_s: float
    solve_ms: float
    times_s: tuple[float, ...]
    pos_enu_m: tuple[tuple[float, float, float], ...]
    vel_enu_m_s: tuple[tuple[float, float, float], ...]
    accel_enu_m_s2: tuple[tuple[float, float, float], ...]
    fuel_proxy_dv_m_s: float


def _solve_fixed_tf_socp(prob: Gfold3DProblem, tf: float) -> Gfold3DSolution | None:
    """Solve the 3D GFOLD SOCP for a fixed burn duration `tf` using raw Clarabel API."""
    t0 = perf_counter()
    n = prob.n_steps
    dt = tf / n
    g = prob.gravity_m_s2
    half_dt2 = 0.5 * dt * dt

    # Variables:
    #   states k = 0..n: 6 per node [x, y, z, vx, vy, vz] -> 6*(n+1)
    #   controls k = 0..n-1: 4 per step [ax, ay, az, sigma] -> 4*n
    n_state = 6 * (n + 1)
    n_vars = n_state + 4 * n

    def sp(k: int, axis: int) -> int:
        return 6 * k + axis

    def sv(k: int, axis: int) -> int:
        return 6 * k + 3 + axis

    def ua(k: int, axis: int) -> int:
        return n_state + 4 * k + axis

    def usig(k: int) -> int:
        return n_state + 4 * k + 3

    # Objective: min sum(sigma_k * dt) + tiny quadratic regularizer on controls for smoothness
    q_obj = np.zeros(n_vars, dtype=np.float64)
    p_rows: list[int] = []
    p_cols: list[int] = []
    p_vals: list[float] = []
    for k in range(n):
        q_obj[usig(k)] = dt
        for ax in range(3):
            idx = ua(k, ax)
            p_rows.append(idx)
            p_cols.append(idx)
            p_vals.append(1e-4 * dt)
    p_mat = sparse.csc_matrix((p_vals, (p_rows, p_cols)), shape=(n_vars, n_vars))

    # Build Clarabel constraints: A x + s = b, s in K
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_list: list[float] = []
    cones: list[object] = []
    row = 0

    def add_row(entries: list[tuple[int, float]], rhs: float) -> None:
        nonlocal row
        for c_idx, val in entries:
            rows.append(row)
            cols.append(c_idx)
            vals.append(val)
        b_list.append(rhs)
        row += 1

    # --- 1. ZeroConeT: Initial state (6) + Dynamics (6*n) + Terminal state (6) ---
    n_zero_start = row
    x0, y0, z0 = prob.pos0_enu_m
    vx0, vy0, vz0 = prob.vel0_enu_m_s
    for i, v_init in enumerate((x0, y0, z0, vx0, vy0, vz0)):
        add_row([(i, 1.0)], v_init)

    for k in range(n):
        for ax in range(3):
            g_ax = -g if ax == 2 else 0.0
            # p_{k+1} - p_k - dt*v_k - 0.5*dt^2*a_k = 0.5*dt^2*g_ax
            add_row(
                [
                    (sp(k + 1, ax), 1.0),
                    (sp(k, ax), -1.0),
                    (sv(k, ax), -dt),
                    (ua(k, ax), -half_dt2),
                ],
                half_dt2 * g_ax,
            )
            # v_{k+1} - v_k - dt*a_k = dt*g_ax
            add_row(
                [
                    (sv(k + 1, ax), 1.0),
                    (sv(k, ax), -1.0),
                    (ua(k, ax), -dt),
                ],
                dt * g_ax,
            )

    for i, v_fin in enumerate((0.0, 0.0, 0.0, 0.0, 0.0, -abs(prob.v_touchdown_m_s))):
        idx = sp(n, i) if i < 3 else sv(n, i - 3)
        add_row([(idx, 1.0)], v_fin)
    cones.append(clarabel.ZeroConeT(row - n_zero_start))

    # --- 2. NonnegativeConeT: z_k >= 0, vz_k <= 0, a_min <= sigma_k <= a_max, az_k >= 0 ---
    n_nn_start = row
    for k in range(1, n):
        # -z_k + s = 0 => z_k >= 0
        add_row([(sp(k, 2), -1.0)], 0.0)
        # vz_k + s = 0 => vz_k <= 0
        add_row([(sv(k, 2), 1.0)], 0.0)
    for k in range(n):
        # -sigma_k + s = -a_min => sigma_k >= a_min
        add_row([(usig(k), -1.0)], -prob.a_min_m_s2)
        # sigma_k + s = a_max => sigma_k <= a_max
        add_row([(usig(k), 1.0)], prob.a_max_m_s2)
        # -az_k + s = -a_min*cos(max_tilt) => az_k >= a_min*cos(max_tilt)
        add_row([(ua(k, 2), -1.0)], -prob.a_min_m_s2 * cos(radians(prob.max_tilt_deg)))
    cones.append(clarabel.NonnegativeConeT(row - n_nn_start))

    # --- 3. SecondOrderConeT(4): ||(ax_k, ay_k, az_k)||_2 <= sigma_k ---
    # s = b - A x = (sigma_k, ax_k, ay_k, az_k) in Q_4
    for k in range(n):
        add_row([(usig(k), -1.0)], 0.0)
        add_row([(ua(k, 0), -1.0)], 0.0)
        add_row([(ua(k, 1), -1.0)], 0.0)
        add_row([(ua(k, 2), -1.0)], 0.0)
        cones.append(clarabel.SecondOrderConeT(4))

    # --- 4. SecondOrderConeT(3): ||(ax_k, ay_k)||_2 <= tan(theta_lim) * az_k ---
    tan_max = tan(radians(prob.max_tilt_deg))
    tan_fin = tan(radians(prob.final_tilt_deg))
    fin_start = int(round((1.0 - prob.final_vertical_fraction) * n))
    for k in range(n):
        t_lim = tan_fin if k >= fin_start else tan_max
        add_row([(ua(k, 2), -t_lim)], 0.0)
        add_row([(ua(k, 0), -1.0)], 0.0)
        add_row([(ua(k, 1), -1.0)], 0.0)
        cones.append(clarabel.SecondOrderConeT(3))

    # --- 5. SecondOrderConeT(3): ||(x_k, y_k)||_2 <= tan(gamma_gs) * z_k + pad_margin ---
    tan_gs = tan(radians(prob.glide_slope_max_deg))
    for k in range(1, n):
        add_row([(sp(k, 2), -tan_gs)], prob.pad_margin_m)
        add_row([(sp(k, 0), -1.0)], 0.0)
        add_row([(sp(k, 1), -1.0)], 0.0)
        cones.append(clarabel.SecondOrderConeT(3))

    a_mat = sparse.csc_matrix((vals, (rows, cols)), shape=(row, n_vars))
    b_vec = np.asarray(b_list, dtype=np.float64)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 60
    settings.tol_gap_abs = 1e-5
    settings.tol_gap_rel = 1e-5
    settings.tol_feas = 1e-5

    solver = clarabel.DefaultSolver(p_mat, q_obj, a_mat, b_vec, cones, settings)
    sol = solver.solve()
    elapsed_ms = (perf_counter() - t0) * 1000.0

    status_str = str(sol.status)
    if "Solved" not in status_str or sol.x is None:
        return None

    x_opt = np.asarray(sol.x, dtype=np.float64)
    times = tuple(k * dt for k in range(n + 1))
    positions = tuple(
        (float(x_opt[sp(k, 0)]), float(x_opt[sp(k, 1)]), float(x_opt[sp(k, 2)]))
        for k in range(n + 1)
    )
    velocities = tuple(
        (float(x_opt[sv(k, 0)]), float(x_opt[sv(k, 1)]), float(x_opt[sv(k, 2)]))
        for k in range(n + 1)
    )
    accels = tuple(
        (float(x_opt[ua(k, 0)]), float(x_opt[ua(k, 1)]), float(x_opt[ua(k, 2)])) for k in range(n)
    )
    dv = float(sum(x_opt[usig(k)] for k in range(n)) * dt)
    return Gfold3DSolution(
        feasible=True,
        status="optimal",
        flight_time_s=tf,
        dt_s=dt,
        solve_ms=elapsed_ms,
        times_s=times,
        pos_enu_m=positions,
        vel_enu_m_s=velocities,
        accel_enu_m_s2=accels,
        fuel_proxy_dv_m_s=dv,
    )


def solve_gfold_3d_socp(prob: Gfold3DProblem) -> Gfold3DSolution:
    """Search flight duration tf for minimum-fuel feasible 3D GFOLD SOCP trajectory."""
    t_start = perf_counter()
    v0 = max(-prob.vel0_enu_m_s[2], 1.0)
    z0 = max(prob.pos0_enu_m[2], 1.0)
    a_net_max = max(prob.a_max_m_s2 - prob.gravity_m_s2, 1.0)
    a_net_min = max(prob.a_min_m_s2 - prob.gravity_m_s2, 0.3)

    tf_min = max(1.0, min(v0 / a_net_max, 2.0 * z0 / (v0 + prob.v_touchdown_m_s)) * 0.70)
    tf_max = max(tf_min + 2.0, min(60.0, max(v0 / a_net_min, 2.0 * z0 / (0.5 * v0 + 1.0)) * 1.45))

    best: Gfold3DSolution | None = None
    best_tf = 0.0
    # Fast 7-point coarse bracket + 2-point local refinement (~10-18 ms total)
    grid = np.linspace(tf_min, tf_max, 7)
    for tf in grid:
        sol = _solve_fixed_tf_socp(prob, float(tf))
        if sol is not None and (best is None or sol.fuel_proxy_dv_m_s < best.fuel_proxy_dv_m_s):
            best = sol
            best_tf = float(tf)

    if best is not None:
        step = (tf_max - tf_min) / 12.0
        for tf_ref in (max(tf_min, best_tf - step), min(tf_max, best_tf + step)):
            sol = _solve_fixed_tf_socp(prob, float(tf_ref))
            if sol is not None and sol.fuel_proxy_dv_m_s < best.fuel_proxy_dv_m_s:
                best = sol

    total_ms = (perf_counter() - t_start) * 1000.0
    if best is None:
        return Gfold3DSolution(
            feasible=False,
            status="infeasible",
            flight_time_s=0.0,
            dt_s=0.0,
            solve_ms=total_ms,
            times_s=(),
            pos_enu_m=(),
            vel_enu_m_s=(),
            accel_enu_m_s2=(),
            fuel_proxy_dv_m_s=float("inf"),
        )
    return Gfold3DSolution(
        feasible=best.feasible,
        status=best.status,
        flight_time_s=best.flight_time_s,
        dt_s=best.dt_s,
        solve_ms=total_ms,
        times_s=best.times_s,
        pos_enu_m=best.pos_enu_m,
        vel_enu_m_s=best.vel_enu_m_s,
        accel_enu_m_s2=best.accel_enu_m_s2,
        fuel_proxy_dv_m_s=best.fuel_proxy_dv_m_s,
    )


def _predict_zero_tilt_1d(
    pos: float,
    vel: float,
    theta: float,
    omega: float,
    g_eff: float,
    a_att: float,
    *,
    x_horizon_cap_s: float = 3.2,
) -> tuple[float, float]:
    """1D analytic zero-tilt unwind predictor (exact port of DM100 2D)."""
    a_eff = max(a_att * 0.88, 1e-4)
    if theta * omega > 0.0:
        t1 = abs(omega) / a_eff
        th_pk = theta + 0.5 * omega * t1
        dv1 = g_eff * sin(0.5 * (theta + th_pk)) * t1
        t2 = 2.0 * sqrt(abs(th_pk) / a_eff)
        dv2 = g_eff * sin(0.5 * th_pk) * t2
        t_tot = t1 + t2
        dv = dv1 + dv2
    else:
        stop_ang = 0.5 * omega * omega / a_eff
        if stop_ang <= abs(theta):
            v_mid = sqrt(0.5 * omega * omega + a_eff * abs(theta))
            t_tot = max(0.0, (2.0 * v_mid - abs(omega)) / a_eff)
            dv = g_eff * sin(0.45 * theta) * t_tot
        else:
            t1 = abs(omega) / a_eff
            th_over = theta + 0.5 * omega * t1
            dv1 = g_eff * sin(0.5 * (theta + th_over)) * t1
            t2 = 2.0 * sqrt(abs(th_over) / a_eff)
            dv2 = g_eff * sin(0.5 * th_over) * t2
            t_tot = t1 + t2
            dv = dv1 + dv2
    v_zero = vel + dv
    p_zero = pos + 0.45 * vel + 0.5 * dv * min(t_tot, x_horizon_cap_s)
    return p_zero, v_zero


def predict_zero_tilt_state_3d(
    pos_xy_m: tuple[float, float],
    vel_xy_m_s: tuple[float, float],
    tilt_xy_rad: tuple[float, float],
    tilt_rate_xy_rad_s: tuple[float, float],
    thrust_accel_m_s2: float,
    att_accel_rad_s2: float,
    *,
    x_horizon_cap_s: float = 3.2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """3D analytic zero-tilt backup predictor for (X, Y) horizontal divert phase lead."""
    x_z, vx_z = _predict_zero_tilt_1d(
        pos_xy_m[0],
        vel_xy_m_s[0],
        tilt_xy_rad[0],
        tilt_rate_xy_rad_s[0],
        thrust_accel_m_s2,
        att_accel_rad_s2,
        x_horizon_cap_s=x_horizon_cap_s,
    )
    y_z, vy_z = _predict_zero_tilt_1d(
        pos_xy_m[1],
        vel_xy_m_s[1],
        tilt_xy_rad[1],
        tilt_rate_xy_rad_s[1],
        thrust_accel_m_s2,
        att_accel_rad_s2,
        x_horizon_cap_s=x_horizon_cap_s,
    )
    _ = hypot(x_z, y_z)
    return (x_z, y_z), (vx_z, vy_z)
