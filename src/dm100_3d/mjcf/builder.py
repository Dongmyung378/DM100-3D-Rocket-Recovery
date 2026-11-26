"""Parametric MJCF generator for DM100 3D (1st-stage booster + 4 fins + 4 landing legs).

Design choices for exact 2D-to-3D equivalence and RK4 numerical stability:
  - `booster` root body frame sits at geometric hull base (`station = 0.0 m`) so
    shifting `body_ipos = [0, 0, cg_m]` as fuel burns never teleports visual/contact geoms.
  - Gravity is driven per-step via `model.opt.gravity[2] = -g(z)` (altitude-varying).
  - Each of the 4 telescoping landing-leg footpads carries `FOOT_BODY_MASS_KG = 50 kg`
    (total 200 kg out of 800 kg gear mass; at half_span=4.0 m this matches the exact
    deployed gear roll inertia `800 * 2.0^2 = 3,200 kg*m^2` while staying RK4-stable
    with `k = 80 kN/m`, `c = 22.5 kN*s/m` at `contact_dt <= 0.002 s`).
  - Dynamic mass/COM/inertia updates subtract exact child mass & parallel-axis inertia
    so the composite multibody system matches `mass_properties_3d()` to machine precision.
"""

from __future__ import annotations

from math import radians
from xml.etree import ElementTree as ET

import mujoco

from dm100_3d.config import DM100VehicleConfig, mass_properties_3d

LEG_NAMES: tuple[str, str, str, str] = ("leg_px", "leg_mx", "leg_py", "leg_my")
FIN_NAMES: tuple[str, str, str, str] = ("fin_px", "fin_mx", "fin_py", "fin_my")
LEG_UNIT_DIRS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)
FIN_UNIT_DIRS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)

FOOT_BODY_MASS_KG: float = 50.0
FOOT_BODY_INERTIA_KG_M2: float = 0.5
FOOT_RADIUS_M: float = 0.15
FIN_BODY_MASS_KG: float = 1e-3
FIN_BODY_INERTIA_KG_M2: float = 1e-3


def build_dm100_mjcf_xml(
    vehicle: DM100VehicleConfig | None = None,
    *,
    timestep_s: float = 0.01,
    gear_deployed: bool = False,
    upper_attached: bool = True,
) -> str:
    """Generate parametric float64 RK4 MJCF XML string for DM100 3D."""
    cfg = vehicle or DM100VehicleConfig()
    g = cfg.geometry
    gear = cfg.landing_gear
    fins = cfg.fins

    init_mass = cfg.launch_mass_kg if upper_attached else cfg.booster_wet_mass_kg
    dep = 1.0 if gear_deployed else 0.0
    mp = mass_properties_3d(init_mass, dep, vehicle=cfg, upper_attached=upper_attached)

    root = ET.Element("mujoco", model="dm100_3d")

    ET.SubElement(root, "compiler", angle="radian", coordinate="local", inertiafromgeom="false")
    ET.SubElement(
        root,
        "option",
        integrator="RK4",
        timestep=f"{timestep_s:.6f}",
        gravity="0 0 0",
        density="0",
        viscosity="0",
        solver="Newton",
        iterations="60",
        tolerance="1e-10",
    )
    ET.SubElement(root, "size", njmax="200", nconmax="64")

    # Visual assets for MuJoCo viewer clarity
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid_tex",
        type="2d",
        builtin="checker",
        rgb1="0.18 0.22 0.28",
        rgb2="0.22 0.26 0.32",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="ground_mat",
        texture="grid_tex",
        texrepeat="40 40",
        reflectance="0.05",
    )
    ET.SubElement(asset, "material", name="pad_mat", rgba="0.12 0.20 0.35 1")
    ET.SubElement(asset, "material", name="hull_mat", rgba="0.92 0.94 0.96 1")
    ET.SubElement(asset, "material", name="dark_mat", rgba="0.12 0.14 0.18 1")
    ET.SubElement(asset, "material", name="fin_mat", rgba="0.30 0.34 0.40 1")
    ET.SubElement(asset, "material", name="leg_mat", rgba="0.18 0.20 0.24 1")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="30 -40 80", dir="-0.3 0.4 -0.85", diffuse="0.9 0.9 0.95")

    # Ground plane (bit 0 = ground)
    ET.SubElement(
        world,
        "geom",
        name="ground_plane",
        type="plane",
        pos="0 0 0",
        size="500 500 0.1",
        material="ground_mat",
        contype="1",
        conaffinity="7",
        friction=f"{gear.friction_coefficient:.4f} 0.005 0.0001",
        solref="0.006 1.0",
        solimp="0.95 0.99 0.001",
    )
    # Visual-only 10 m radius landing pad ring
    ET.SubElement(
        world,
        "geom",
        name="landing_pad_visual",
        type="cylinder",
        pos="0 0 -0.01",
        size=f"{gear.pad_radius_m:.3f} 0.01",
        material="pad_mat",
        contype="0",
        conaffinity="0",
    )

    # Booster root body: origin at hull base (station = 0.0 m)
    init_base_z = -g.nozzle_station_m
    booster = ET.SubElement(world, "body", name="booster", pos=f"0 0 {init_base_z:.6f}")
    ET.SubElement(booster, "freejoint", name="booster_free")

    # Initial inertial placeholder (exact composite subtraction applied immediately after compile)
    ET.SubElement(
        booster,
        "inertial",
        pos=f"0 0 {mp.cg_station_m:.6f}",
        mass=f"{mp.total_mass_kg:.6f}",
        diaginertia=(
            f"{mp.inertia_diag_kg_m2[0]:.6f} "
            f"{mp.inertia_diag_kg_m2[1]:.6f} "
            f"{mp.inertia_diag_kg_m2[2]:.6f}"
        ),
    )

    # Booster hull and nozzle geoms (contype=2, conaffinity=1 -> collides with ground only)
    ET.SubElement(
        booster,
        "geom",
        name="booster_hull",
        type="cylinder",
        fromto=f"0 0 0 0 0 {g.body_length_m:.4f}",
        size=f"{g.body_radius_m:.4f}",
        material="hull_mat",
        contype="2",
        conaffinity="1",
        friction="0.6 0.005 0.0001",
        solref="0.01 1.0",
    )
    ET.SubElement(
        booster,
        "geom",
        name="booster_nozzle",
        type="cylinder",
        fromto=f"0 0 {g.nozzle_station_m:.4f} 0 0 0",
        size=f"{g.nozzle_radius_m:.4f}",
        material="dark_mat",
        contype="2",
        conaffinity="1",
        friction="0.6 0.005 0.0001",
        solref="0.01 1.0",
    )
    # Visual-only upper stage fairing (toggled via alpha in viewer when detached)
    ET.SubElement(
        booster,
        "geom",
        name="upper_stage_visual",
        type="cylinder",
        fromto=f"0 0 {g.body_length_m:.4f} 0 0 {g.body_length_m + g.upper_length_m:.4f}",
        size=f"{g.body_radius_m:.4f}",
        rgba="0.86 0.88 0.92 0.85" if upper_attached else "0.86 0.88 0.92 0.0",
        contype="0",
        conaffinity="0",
    )

    # 4 Independent hinged grid fins at z = 22.4 m (+X, -X, +Y, -Y)
    fin_lim_rad = radians(fins.deflection_limit_deg)
    for name, (ux, uy) in zip(FIN_NAMES, FIN_UNIT_DIRS, strict=True):
        root_x = ux * g.body_radius_m
        root_y = uy * g.body_radius_m
        fin_body = ET.SubElement(
            booster,
            "body",
            name=name,
            pos=f"{root_x:.4f} {root_y:.4f} {fins.station_m:.4f}",
        )
        ET.SubElement(
            fin_body,
            "inertial",
            pos="0 0 0",
            mass=f"{FIN_BODY_MASS_KG:.6f}",
            diaginertia=(
                f"{FIN_BODY_INERTIA_KG_M2:.6f} "
                f"{FIN_BODY_INERTIA_KG_M2:.6f} "
                f"{FIN_BODY_INERTIA_KG_M2:.6f}"
            ),
        )
        # Hinge rotates around the fin's radial mounting axis (ux, uy, 0)
        ET.SubElement(
            fin_body,
            "joint",
            name=f"{name}_hinge",
            type="hinge",
            axis=f"{ux:.4f} {uy:.4f} 0",
            limited="true",
            range=f"{-fin_lim_rad:.6f} {fin_lim_rad:.6f}",
            damping="0.0",
        )
        # Visual fin paddle (approx 1.2 m span x 1.25 m chord = 1.5 m^2)
        cx = ux * 0.60
        cy = uy * 0.60
        sx = 0.60 if abs(ux) > 0.5 else 0.05
        sy = 0.60 if abs(uy) > 0.5 else 0.05
        ET.SubElement(
            fin_body,
            "geom",
            name=f"{name}_geom",
            type="box",
            pos=f"{cx:.4f} {cy:.4f} 0",
            size=f"{sx:.4f} {sy:.4f} 0.625",
            material="fin_mat",
            contype="0",
            conaffinity="0",
        )

    # 4 Telescoping landing-leg struts (+X, -X, +Y, -Y)
    span = g.body_radius_m + dep * (gear.half_span_m - g.body_radius_m)
    foot_center_z = (1.0 - dep) * g.gear_retracted_station_m + dep * (
        gear.foot_station_m + FOOT_RADIUS_M
    )
    leg_contype = "4" if gear_deployed else "0"
    leg_conaffinity = "1" if gear_deployed else "0"
    k_leg = gear.per_leg_stiffness_n_m if gear_deployed else 0.0
    c_leg = gear.per_leg_damping_n_s_m if gear_deployed else 0.0

    for name, (ux, uy) in zip(LEG_NAMES, LEG_UNIT_DIRS, strict=True):
        leg_x = ux * span
        leg_y = uy * span
        leg_body = ET.SubElement(
            booster,
            "body",
            name=name,
            pos=f"{leg_x:.4f} {leg_y:.4f} {foot_center_z:.4f}",
        )
        ET.SubElement(
            leg_body,
            "inertial",
            pos="0 0 0",
            mass=f"{FOOT_BODY_MASS_KG:.6f}",
            diaginertia=(
                f"{FOOT_BODY_INERTIA_KG_M2:.6f} "
                f"{FOOT_BODY_INERTIA_KG_M2:.6f} "
                f"{FOOT_BODY_INERTIA_KG_M2:.6f}"
            ),
        )
        # Telescoping upward slide axis (+z_b): positive qpos == upward strut compression stroke
        ET.SubElement(
            leg_body,
            "joint",
            name=f"{name}_slide",
            type="slide",
            axis="0 0 1",
            limited="true",
            range=f"0.0 {gear.max_leg_stroke_m:.4f}",
            stiffness=f"{k_leg:.4f}",
            damping=f"{c_leg:.4f}",
            springref="0.0",
            solreflimit="0.006 1.0",
            solimplimit="0.95 0.99 0.001",
        )
        # Footpad contact sphere (bottom sits at foot_center_z - FOOT_RADIUS_M == foot_station_m)
        ET.SubElement(
            leg_body,
            "geom",
            name=f"{name}_foot",
            type="sphere",
            pos="0 0 0",
            size=f"{FOOT_RADIUS_M:.4f}",
            material="leg_mat",
            contype=leg_contype,
            conaffinity=leg_conaffinity,
            friction=f"{gear.friction_coefficient:.4f} 0.005 0.0001",
            solref="0.006 1.0",
            solimp="0.95 0.99 0.001",
        )
        # Visual strut shaft linking footpad toward booster anchor
        ET.SubElement(
            leg_body,
            "geom",
            name=f"{name}_strut_vis",
            type="capsule",
            fromto=f"0 0 0 {-ux * 1.2:.4f} {-uy * 1.2:.4f} 1.8",
            size="0.09",
            material="leg_mat",
            contype="0",
            conaffinity="0",
        )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def compile_dm100_model(
    vehicle: DM100VehicleConfig | None = None,
    *,
    timestep_s: float = 0.01,
    gear_deployed: bool = False,
    upper_attached: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build and compile the DM100 3D MuJoCo model and data pair."""
    xml = build_dm100_mjcf_xml(
        vehicle,
        timestep_s=timestep_s,
        gear_deployed=gear_deployed,
        upper_attached=upper_attached,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
