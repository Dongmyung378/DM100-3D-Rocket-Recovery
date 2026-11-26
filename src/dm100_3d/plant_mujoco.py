"""MuJoCo 6DOF rocket body mechanics, dynamic inertia updates, and 4-leg contact dynamics.

Phase 1 core provides:
  - Exact composite mass/COM/3-axis inertia updates (`mj_setConst`) subtracting child
    leg/fin masses so composite rigid-body inertia matches `mass_properties_3d()` exactly.
  - Altitude-varying gravity via `model.opt.gravity[2] = -fast_gravity(z)`.
  - Smooth landing-leg deployment + telescoping strut spring-damper activation.
  - Exact 4-leg 3D contact force/stroke/hull-collision readout and 5 s upright standing check.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, hypot, isfinite

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from dm100_3d.config import (
    EARTH_RADIUS_M,
    G0,
    DM100VehicleConfig,
    DualTankGeometry,
    MassProperties3D,
    mass_properties_3d,
)
from dm100_3d.coords import (
    base_to_com_pos_enu,
    com_to_base_pos_enu,
    normalize_quat_wxyz,
    quat_to_rotmat,
    tilt_angle_rad,
)
from dm100_3d.interfaces import ContactSample3D
from dm100_3d.mjcf.builder import (
    FIN_BODY_INERTIA_KG_M2,
    FIN_BODY_MASS_KG,
    FIN_NAMES,
    FIN_UNIT_DIRS,
    FOOT_BODY_INERTIA_KG_M2,
    FOOT_BODY_MASS_KG,
    FOOT_RADIUS_M,
    LEG_NAMES,
    LEG_UNIT_DIRS,
    compile_dm100_model,
)


def fast_gravity(altitude_m: float) -> float:
    """Spherical-Earth altitude-varying gravitational acceleration [m/s^2]."""
    h = altitude_m if altitude_m > 0.0 else 0.0
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + h)
    return G0 * ratio * ratio


@dataclass(frozen=True, slots=True)
class DropTestResult:
    """Summary of a 3D MuJoCo 4-leg touchdown and 5 s upright settling test."""

    outcome: str
    standing_stable: bool
    touchdown_time_s: float | None
    final_time_s: float
    max_leg_stroke_m: float
    max_leg_force_n: float
    max_contact_load_g: float
    max_tilt_deg: float
    final_tilt_deg: float
    final_speed_m_s: float
    final_rate_deg_s: float
    body_ground_collision: bool
    legs_in_contact_final: tuple[bool, bool, bool, bool]
    final_leg_strokes_m: tuple[float, float, float, float]


class MuJoCoRocketBody:
    """Low-level 6DOF MuJoCo vehicle wrapper managing dynamic inertia and 4-leg contact."""

    def __init__(
        self,
        vehicle: DM100VehicleConfig | None = None,
        *,
        timestep_s: float = 0.01,
        gear_deployed: bool = False,
        upper_attached: bool = True,
    ) -> None:
        self.vehicle = vehicle or DM100VehicleConfig()
        self.model, self.data = compile_dm100_model(
            self.vehicle,
            timestep_s=timestep_s,
            gear_deployed=gear_deployed,
            upper_attached=upper_attached,
        )

        # Resolve MuJoCo element IDs once
        self.booster_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "booster")
        self.ground_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_plane")
        self.hull_gids = frozenset(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("booster_hull", "booster_nozzle")
        )
        self.upper_vis_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "upper_stage_visual"
        )

        self.leg_bids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in LEG_NAMES
        )
        self.leg_gids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_foot")
            for name in LEG_NAMES
        )
        self.leg_jids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_slide")
            for name in LEG_NAMES
        )
        self.leg_qpos_adrs = tuple(int(self.model.jnt_qposadr[j]) for j in self.leg_jids)
        self.leg_dof_adrs = tuple(int(self.model.jnt_dofadr[j]) for j in self.leg_jids)

        self.fin_bids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FIN_NAMES
        )
        self.fin_jids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_hinge")
            for name in FIN_NAMES
        )
        self.fin_qpos_adrs = tuple(int(self.model.jnt_qposadr[j]) for j in self.fin_jids)
        self.fin_dof_adrs = tuple(int(self.model.jnt_dofadr[j]) for j in self.fin_jids)

        self._leg_gid_to_idx = {gid: idx for idx, gid in enumerate(self.leg_gids)}
        self._wrench_buf = np.zeros(6, dtype=np.float64)
        # Dedicated scratch MjData for mj_setConst so live simulation state is never reset
        self._const_data = mujoco.MjData(self.model)

        self.gear_extension: float = 1.0 if gear_deployed else 0.0
        self.upper_attached: bool = upper_attached
        self.struts_active: bool = gear_deployed
        init_mass = (
            self.vehicle.launch_mass_kg if upper_attached else self.vehicle.booster_wet_mass_kg
        )
        self.current_mass_props: MassProperties3D = self.update_mass_properties(
            init_mass,
            self.gear_extension,
            upper_attached=upper_attached,
            struts_active=gear_deployed,
        )

    def update_mass_properties(
        self,
        total_mass_kg: float,
        gear_extension: float,
        *,
        upper_attached: bool = False,
        airframe_mass_offset_kg: float = 0.0,
        inertia_scale: float = 1.0,
        inertia_roll_scale: float | None = None,
        dual_tank: bool = False,
        rcs_propellant_kg: float | None = None,
        dual_tank_geom: DualTankGeometry | None = None,
        struts_active: bool | None = None,
    ) -> MassProperties3D:
        """Update booster mass, COM, 3-axis inertia, and leg deployment via `mj_setConst`.

        Subtracts exact child leg & fin mass/moments/parallel-axis inertia so the
        composite MuJoCo multibody matches `mass_properties_3d()` to machine precision.
        """
        cfg = self.vehicle
        g = cfg.geometry
        gear = cfg.landing_gear
        fins = cfg.fins
        dep = max(0.0, min(1.0, float(gear_extension)))
        self.gear_extension = dep
        self.upper_attached = upper_attached

        mp = mass_properties_3d(
            total_mass_kg,
            dep,
            vehicle=cfg,
            upper_attached=upper_attached,
            airframe_mass_offset_kg=airframe_mass_offset_kg,
            inertia_scale=inertia_scale,
            inertia_roll_scale=inertia_roll_scale,
            dual_tank=dual_tank,
            rcs_propellant_kg=rcs_propellant_kg,
            dual_tank_geom=dual_tank_geom,
        )

        # Update leg mount positions and contact/spring activation
        active = self.struts_active if struts_active is None else bool(struts_active)
        if dep < 0.5:
            active = False
        self.struts_active = active

        span = g.body_radius_m + dep * (gear.half_span_m - g.body_radius_m)
        foot_z = (1.0 - dep) * g.gear_retracted_station_m + dep * (
            gear.foot_station_m + FOOT_RADIUS_M
        )

        child_mass = 0.0
        child_z_mom = 0.0
        child_ixx = 0.0
        child_iyy = 0.0
        child_izz = 0.0
        cg_tot = mp.cg_station_m

        for i, (bid, gid, jid, dof_adr) in enumerate(
            zip(self.leg_bids, self.leg_gids, self.leg_jids, self.leg_dof_adrs, strict=True)
        ):
            ux, uy = LEG_UNIT_DIRS[i]
            lx, ly, lz = ux * span, uy * span, foot_z
            self.model.body_pos[bid] = (lx, ly, lz)
            self.model.geom_contype[gid] = 4 if active else 0
            self.model.geom_conaffinity[gid] = 1 if active else 0
            self.model.jnt_range[jid] = (0.0, gear.max_leg_stroke_m if active else 1e-6)
            self.model.jnt_stiffness[jid] = gear.per_leg_stiffness_n_m if active else 0.0
            self.model.dof_damping[dof_adr] = gear.per_leg_damping_n_s_m if active else 0.0

            dz = lz - cg_tot
            child_mass += FOOT_BODY_MASS_KG
            child_z_mom += FOOT_BODY_MASS_KG * lz
            child_ixx += FOOT_BODY_INERTIA_KG_M2 + FOOT_BODY_MASS_KG * (ly * ly + dz * dz)
            child_iyy += FOOT_BODY_INERTIA_KG_M2 + FOOT_BODY_MASS_KG * (lx * lx + dz * dz)
            child_izz += FOOT_BODY_INERTIA_KG_M2 + FOOT_BODY_MASS_KG * (lx * lx + ly * ly)

        for i, _bid in enumerate(self.fin_bids):
            ux, uy = FIN_UNIT_DIRS[i]
            fx, fy, fz = ux * g.body_radius_m, uy * g.body_radius_m, fins.station_m
            dz = fz - cg_tot
            child_mass += FIN_BODY_MASS_KG
            child_z_mom += FIN_BODY_MASS_KG * fz
            child_ixx += FIN_BODY_INERTIA_KG_M2 + FIN_BODY_MASS_KG * (fy * fy + dz * dz)
            child_iyy += FIN_BODY_INERTIA_KG_M2 + FIN_BODY_MASS_KG * (fx * fx + dz * dz)
            child_izz += FIN_BODY_INERTIA_KG_M2 + FIN_BODY_MASS_KG * (fx * fx + fy * fy)

        m_booster = mp.total_mass_kg - child_mass
        if m_booster <= 1.0:
            raise ValueError(f"Invalid booster residual mass: {m_booster}")
        cg_booster = (mp.total_mass_kg * cg_tot - child_z_mom) / m_booster
        dz_b = cg_booster - cg_tot

        ixx_b = mp.inertia_diag_kg_m2[0] - child_ixx - m_booster * dz_b * dz_b
        iyy_b = mp.inertia_diag_kg_m2[1] - child_iyy - m_booster * dz_b * dz_b
        izz_b = mp.inertia_diag_kg_m2[2] - child_izz
        if min(ixx_b, iyy_b, izz_b) <= 1.0:
            raise ValueError(
                f"Non-positive booster residual inertia: ({ixx_b:.2f}, {iyy_b:.2f}, {izz_b:.2f})"
            )

        self.model.body_mass[self.booster_bid] = m_booster
        self.model.body_ipos[self.booster_bid] = (0.0, 0.0, cg_booster)
        self.model.body_inertia[self.booster_bid] = (ixx_b, iyy_b, izz_b)
        self.model.geom_rgba[self.upper_vis_gid, 3] = 0.85 if upper_attached else 0.0

        mujoco.mj_setConst(self.model, self._const_data)
        self.current_mass_props = mp
        return mp

    def set_pose_from_com(
        self,
        pos_com_enu_m: ArrayLike,
        vel_com_enu_m_s: ArrayLike,
        quat_wxyz: ArrayLike,
        omega_body_rad_s: ArrayLike,
        *,
        fin_deflections_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Initialize rigid-body state from composite COM position/velocity and attitude."""
        q = normalize_quat_wxyz(quat_wxyz, canonical=True)
        r_wb = quat_to_rotmat(q)
        omega_b = np.asarray(omega_body_rad_s, dtype=np.float64).reshape(3)
        v_com = np.asarray(vel_com_enu_m_s, dtype=np.float64).reshape(3)
        cg_m = self.current_mass_props.cg_station_m

        # Hull base (joint origin) position & velocity from rigid-body kinematics:
        #   p_base = p_com - R * [0, 0, cg]
        #   v_base = v_com - R * (omega_b x [0, 0, cg])
        p_base = com_to_base_pos_enu(pos_com_enu_m, q, cg_m)
        r_cg_b = np.array([0.0, 0.0, cg_m], dtype=np.float64)
        v_base = v_com - r_wb @ np.cross(omega_b, r_cg_b)

        self.data.qpos[0:3] = p_base
        self.data.qpos[3:7] = q
        self.data.qvel[0:3] = v_base
        self.data.qvel[3:6] = omega_b

        for idx, (qadr, dadr) in enumerate(zip(self.fin_qpos_adrs, self.fin_dof_adrs, strict=True)):
            self.data.qpos[qadr] = float(fin_deflections_rad[idx])
            self.data.qvel[dadr] = 0.0
        for qadr, dadr in zip(self.leg_qpos_adrs, self.leg_dof_adrs, strict=True):
            self.data.qpos[qadr] = 0.0
            self.data.qvel[dadr] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def sync_kinematic_fins(
        self,
        fin_deflections_rad: tuple[float, float, float, float],
        fin_rates_rad_s: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Set hinge positions/velocities of the 4 visual grid fins."""
        for i, (qadr, dadr) in enumerate(zip(self.fin_qpos_adrs, self.fin_dof_adrs, strict=True)):
            self.data.qpos[qadr] = float(fin_deflections_rad[i])
            self.data.qvel[dadr] = float(fin_rates_rad_s[i])

    def lock_stowed_legs(self) -> None:
        """Hold stowed leg slide joints at zero when struts are inactive."""
        if not self.struts_active:
            for qadr, dadr in zip(self.leg_qpos_adrs, self.leg_dof_adrs, strict=True):
                self.data.qpos[qadr] = 0.0
                self.data.qvel[dadr] = 0.0

    def get_base_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (base_pos_enu_m, quat_wxyz) of the booster geometric hull base."""
        return (
            np.array(self.data.qpos[0:3], dtype=np.float64),
            normalize_quat_wxyz(self.data.qpos[3:7], canonical=True),
        )

    def get_com_state(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (pos_com_enu_m, vel_com_enu_m_s, quat_wxyz, omega_body_rad_s)."""
        p_base, q = self.get_base_pose()
        v_base = np.array(self.data.qvel[0:3], dtype=np.float64)
        omega_b = np.array(self.data.qvel[3:6], dtype=np.float64)
        cg_m = self.current_mass_props.cg_station_m
        p_com = base_to_com_pos_enu(p_base, q, cg_m)
        r_wb = quat_to_rotmat(q)
        v_com = v_base + r_wb @ np.cross(omega_b, np.array([0.0, 0.0, cg_m], dtype=np.float64))
        return p_com, v_com, q, omega_b

    def measure_composite_mass_properties(
        self,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Audit actual composite (total_mass, com_body_m, inertia_body_3x3) from MuJoCo model."""
        body_ids = (self.booster_bid, *self.leg_bids, *self.fin_bids)
        total_mass = float(sum(self.model.body_mass[b] for b in body_ids))
        com_b = np.zeros(3, dtype=np.float64)
        for b in body_ids:
            m_b = float(self.model.body_mass[b])
            p_b = (
                np.array(self.model.body_ipos[b], dtype=np.float64)
                if b == self.booster_bid
                else np.array(self.model.body_pos[b] + self.model.body_ipos[b], dtype=np.float64)
            )
            com_b += m_b * p_b
        com_b /= total_mass

        inertia = np.zeros((3, 3), dtype=np.float64)
        for b in body_ids:
            m_b = float(self.model.body_mass[b])
            p_b = (
                np.array(self.model.body_ipos[b], dtype=np.float64)
                if b == self.booster_bid
                else np.array(self.model.body_pos[b] + self.model.body_ipos[b], dtype=np.float64)
            )
            r = p_b - com_b
            inertia += np.diag(self.model.body_inertia[b])
            inertia += m_b * (float(r @ r) * np.eye(3) - np.outer(r, r))
        return total_mass, com_b, inertia

    def sample_contact(self) -> ContactSample3D:
        """Read 4-leg ground normal forces, strut strokes, hull collisions, and contact load [g]."""
        leg_normals = [0.0, 0.0, 0.0, 0.0]
        leg_contact = [False, False, False, False]
        hull_collision = False
        net_force_world = np.zeros(3, dtype=np.float64)
        net_torque_world = np.zeros(3, dtype=np.float64)
        p_com, _, q, _ = self.get_com_state()

        for c_idx in range(int(self.data.ncon)):
            con = self.data.contact[c_idx]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self.ground_gid:
                other = g2
                sign = 1.0
            elif g2 == self.ground_gid:
                other = g1
                sign = -1.0
            else:
                continue

            if other in self.hull_gids:
                hull_collision = True

            mujoco.mj_contactForce(self.model, self.data, c_idx, self._wrench_buf)
            # Contact frame rows: [normal, tangent1, tangent2] pointing from geom1 -> geom2
            c_frame = con.frame.reshape(3, 3)
            f_world = sign * (c_frame.T @ self._wrench_buf[:3])
            net_force_world += f_world
            arm = np.asarray(con.pos, dtype=np.float64) - p_com
            net_torque_world += np.cross(arm, f_world)

            leg_idx = self._leg_gid_to_idx.get(other)
            if leg_idx is not None:
                fn = max(0.0, float(self._wrench_buf[0]))
                leg_normals[leg_idx] += fn
                if fn > 1.0 or float(con.dist) <= 1e-4:
                    leg_contact[leg_idx] = True

        strokes = tuple(max(0.0, float(self.data.qpos[adr])) for adr in self.leg_qpos_adrs)
        any_contact = any(leg_contact) or hull_collision
        r_wb = quat_to_rotmat(q)
        torque_body = r_wb.T @ net_torque_world

        # Measure net ground reaction load normalized by total booster weight [g]
        mass = max(self.current_mass_props.total_mass_kg, 1.0)
        load_g = float(np.linalg.norm(net_force_world)) / (mass * G0) if any_contact else 0.0

        return ContactSample3D(
            force_world_n=(
                float(net_force_world[0]),
                float(net_force_world[1]),
                float(net_force_world[2]),
            ),
            torque_body_nm=(float(torque_body[0]), float(torque_body[1]), float(torque_body[2])),
            leg_normal_n=(leg_normals[0], leg_normals[1], leg_normals[2], leg_normals[3]),
            leg_stroke_m=(strokes[0], strokes[1], strokes[2], strokes[3]),
            leg_in_contact=(leg_contact[0], leg_contact[1], leg_contact[2], leg_contact[3]),
            any_contact=any_contact,
            body_ground_collision=hull_collision,
            contact_load_g=load_g,
        )

    def step_physics(
        self,
        *,
        dt_s: float,
        force_world_n: ArrayLike = (0.0, 0.0, 0.0),
        torque_world_nm: ArrayLike = (0.0, 0.0, 0.0),
        altitude_for_gravity_m: float | None = None,
        gravity_m_s2: float | None = None,
    ) -> None:
        """Advance MuJoCo RK4 physics by one step `dt_s` with world force/torque at COM."""
        self.model.opt.timestep = float(dt_s)
        if gravity_m_s2 is not None:
            g_mag = float(gravity_m_s2)
        else:
            z_grav = (
                float(self.data.xipos[self.booster_bid, 2])
                if altitude_for_gravity_m is None
                else float(altitude_for_gravity_m)
            )
            g_mag = fast_gravity(z_grav)
        self.model.opt.gravity[0] = 0.0
        self.model.opt.gravity[1] = 0.0
        self.model.opt.gravity[2] = -g_mag

        self.data.xfrc_applied[:] = 0.0
        f_w = np.asarray(force_world_n, dtype=np.float64).reshape(3)
        t_w = np.asarray(torque_world_nm, dtype=np.float64).reshape(3)
        # Shift wrench from composite COM to booster_bid COM (xipos) so pure COM forces
        # create zero parasitic torque from the small offset between booster_bid and composite COM
        p_com, _, _, _ = self.get_com_state()
        p_booster_com = np.asarray(self.data.xipos[self.booster_bid], dtype=np.float64)
        t_at_xipos = t_w + np.cross(p_com - p_booster_com, f_w)
        self.data.xfrc_applied[self.booster_bid, 0:3] = f_w
        self.data.xfrc_applied[self.booster_bid, 3:6] = t_at_xipos

        mujoco.mj_step(self.model, self.data)
        self.lock_stowed_legs()


def run_4leg_drop_test(
    *,
    mass_kg: float = 8_800.0,
    initial_foot_clearance_m: float = 0.15,
    vel_enu_m_s: tuple[float, float, float] = (0.0, 0.0, -1.5),
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    omega_body_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact_dt_s: float = 0.002,
    vehicle: DM100VehicleConfig | None = None,
) -> DropTestResult:
    """Run an engine-off 3D 4-leg touchdown and 5 s upright standing stability drop test."""
    cfg = vehicle or DM100VehicleConfig()
    gear = cfg.landing_gear
    body = MuJoCoRocketBody(cfg, timestep_s=contact_dt_s, gear_deployed=True, upper_attached=False)
    mp = body.update_mass_properties(mass_kg, 1.0, upper_attached=False, struts_active=True)

    # Place lowest footpad bottom at `initial_foot_clearance_m` above z = 0
    q = normalize_quat_wxyz(quat_wxyz, canonical=True)
    r_wb = quat_to_rotmat(q)
    min_foot_z_rel = min(
        float(
            (r_wb @ np.array([ux * gear.half_span_m, uy * gear.half_span_m, gear.foot_station_m]))[
                2
            ]
        )
        for ux, uy in LEG_UNIT_DIRS
    )
    base_z = initial_foot_clearance_m - min_foot_z_rel
    p_com0 = base_to_com_pos_enu((0.0, 0.0, base_z), q, mp.cg_station_m)
    body.set_pose_from_com(p_com0, vel_enu_m_s, q, omega_body_rad_s)

    t = 0.0
    t_td: float | None = None
    max_stroke = 0.0
    max_leg_f = 0.0
    max_load_g = 0.0
    max_tilt = 0.0
    hull_hit = False
    last_contact = ContactSample3D()
    max_sim_s = gear.settle_duration_s + 5.0

    while t < max_sim_s:
        body.step_physics(dt_s=contact_dt_s)
        t += contact_dt_s
        cs = body.sample_contact()
        last_contact = cs
        _, v_com, q_now, om_now = body.get_com_state()
        if not np.all(np.isfinite(v_com)) or not isfinite(float(q_now[0])):
            return DropTestResult(
                outcome="numerical_divergence",
                standing_stable=False,
                touchdown_time_s=t_td,
                final_time_s=t,
                max_leg_stroke_m=max_stroke,
                max_leg_force_n=max_leg_f,
                max_contact_load_g=max_load_g,
                max_tilt_deg=max_tilt,
                final_tilt_deg=999.0,
                final_speed_m_s=999.0,
                final_rate_deg_s=999.0,
                body_ground_collision=True,
                legs_in_contact_final=cs.leg_in_contact,
                final_leg_strokes_m=cs.leg_stroke_m,
            )

        tilt_deg = degrees(tilt_angle_rad(q_now))
        max_tilt = max(max_tilt, tilt_deg)
        max_stroke = max(max_stroke, max(cs.leg_stroke_m))
        max_leg_f = max(max_leg_f, max(cs.leg_normal_n))
        max_load_g = max(max_load_g, cs.contact_load_g)
        if cs.body_ground_collision:
            hull_hit = True
        if t_td is None and cs.any_contact:
            t_td = t

        if hull_hit:
            return DropTestResult(
                outcome="body_ground_collision",
                standing_stable=False,
                touchdown_time_s=t_td,
                final_time_s=t,
                max_leg_stroke_m=max_stroke,
                max_leg_force_n=max_leg_f,
                max_contact_load_g=max_load_g,
                max_tilt_deg=max_tilt,
                final_tilt_deg=tilt_deg,
                final_speed_m_s=float(np.linalg.norm(v_com)),
                final_rate_deg_s=degrees(float(np.linalg.norm(om_now))),
                body_ground_collision=True,
                legs_in_contact_final=cs.leg_in_contact,
                final_leg_strokes_m=cs.leg_stroke_m,
            )
        if tilt_deg > 35.0:
            return DropTestResult(
                outcome="tip_over",
                standing_stable=False,
                touchdown_time_s=t_td,
                final_time_s=t,
                max_leg_stroke_m=max_stroke,
                max_leg_force_n=max_leg_f,
                max_contact_load_g=max_load_g,
                max_tilt_deg=max_tilt,
                final_tilt_deg=tilt_deg,
                final_speed_m_s=float(np.linalg.norm(v_com)),
                final_rate_deg_s=degrees(float(np.linalg.norm(om_now))),
                body_ground_collision=hull_hit,
                legs_in_contact_final=cs.leg_in_contact,
                final_leg_strokes_m=cs.leg_stroke_m,
            )
        if t_td is not None and (t - t_td) >= gear.settle_duration_s:
            final_spd = float(np.linalg.norm(v_com))
            final_rate = degrees(float(np.linalg.norm(om_now)))
            legs_down = sum(1 for c in cs.leg_in_contact if c)
            stable = (
                not hull_hit
                and max_stroke <= gear.max_leg_stroke_m
                and max_leg_f <= gear.max_leg_force_n
                and max_load_g <= gear.max_contact_load_g
                and tilt_deg <= gear.max_standing_tilt_deg
                and final_spd <= gear.max_standing_speed_m_s
                and final_rate <= gear.max_standing_rate_deg_s
                and legs_down == 4
            )
            return DropTestResult(
                outcome="landed" if stable else "standing_unstable",
                standing_stable=stable,
                touchdown_time_s=t_td,
                final_time_s=t,
                max_leg_stroke_m=max_stroke,
                max_leg_force_n=max_leg_f,
                max_contact_load_g=max_load_g,
                max_tilt_deg=max_tilt,
                final_tilt_deg=tilt_deg,
                final_speed_m_s=final_spd,
                final_rate_deg_s=final_rate,
                body_ground_collision=hull_hit,
                legs_in_contact_final=cs.leg_in_contact,
                final_leg_strokes_m=cs.leg_stroke_m,
            )

    _, v_com, q_now, om_now = body.get_com_state()
    return DropTestResult(
        outcome="no_touchdown_timeout",
        standing_stable=False,
        touchdown_time_s=t_td,
        final_time_s=t,
        max_leg_stroke_m=max_stroke,
        max_leg_force_n=max_leg_f,
        max_contact_load_g=max_load_g,
        max_tilt_deg=max_tilt,
        final_tilt_deg=degrees(tilt_angle_rad(q_now)),
        final_speed_m_s=hypot(float(v_com[0]), hypot(float(v_com[1]), float(v_com[2]))),
        final_rate_deg_s=degrees(float(np.linalg.norm(om_now))),
        body_ground_collision=hull_hit,
        legs_in_contact_final=last_contact.leg_in_contact,
        final_leg_strokes_m=last_contact.leg_stroke_m,
    )
