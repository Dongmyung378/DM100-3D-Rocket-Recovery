"""Generate exact-dimension procedural 3D meshes (GLB / glTF 2.0) for DM100 3D in UE 5.4.

All dimensions are read directly from `DM100VehicleConfig` so Unreal Engine 5.4 visuals
match the MuJoCo 6DOF physics & contact geometry to the millimeter:
  - `dm100_booster_hull.glb` : 24.0 m x 3.1 m booster, 7 Methalox bell nozzles (rim at z = -0.2 m),
                               carbon interstage (20.8..24.0 m), 4 RCS pods, hull-base pivot.
  - `dm100_upper_stage.glb`  : 9.0 m Stage 2 + ogive payload fairing + vacuum bell nozzle,
                               pivot at Stage 2 interface (z = 0 -> attaches at booster z = 24.0 m).
  - `dm100_grid_fin.glb`     : 1.20 m x 1.25 m (1.50 m^2) titanium waffle-lattice grid fin,
                               pivot at hinge root (mounts at r = 1.55 m, z = 22.4 m).
  - `dm100_landing_leg.glb`  : Carbon A-frame landing leg spanning r = 1.55 m -> 4.0 m,
                               z = +1.2 m -> -1.0 m, pivot at hull root hinge.
  - `dm100_landing_pad.glb`  : 10.0 m radius target landing pad with 2.0 m bullseye & blast apron.

Coordinate convention:
  - Internal construction is done in right-handed ENU/Z-up meters (+Z up, +X East, +Y North).
  - Before glTF 2.0 (.glb) export, vertices are converted from Z-up to standard glTF Y-up
    via `(x, y, z)_zup -> (x, z, -y)_gltf` (proper rotation R_x(-90 deg), det = +1), which
    Unreal Engine 5.4's glTF/Interchange importer maps cleanly back to UE5 Z-up (cm).
"""

from __future__ import annotations

import argparse
import json
from math import cos, pi, radians, sin, sqrt
from pathlib import Path

import numpy as np
import trimesh

from dm100_3d.config import DM100VehicleConfig

# Proper rotation from Z-up (+X East, +Y North, +Z Up) to glTF Y-up (+X, +Y=Up, +Z=-North)
_T_ZUP_TO_GLTF_YUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _paint(
    mesh: trimesh.Trimesh,
    rgba: tuple[int, int, int, int],
    *,
    metallic: float = 0.15,
    roughness: float = 0.55,
) -> trimesh.Trimesh:
    """Assign a clean PBR material (glTF metallic-roughness) to a trimesh."""
    m = mesh.copy()
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[c / 255.0 for c in rgba],
        metallicFactor=float(metallic),
        roughnessFactor=float(roughness),
    )
    m.visual = trimesh.visual.TextureVisuals(material=mat)
    return m


def _cylinder_z(
    radius: float,
    z0: float,
    z1: float,
    *,
    sections: int = 48,
    cx: float = 0.0,
    cy: float = 0.0,
) -> trimesh.Trimesh:
    h = abs(z1 - z0)
    mesh = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    mesh.apply_translation([cx, cy, 0.5 * (z0 + z1)])
    return mesh


def _frustum_z(
    r0: float,
    r1: float,
    z0: float,
    z1: float,
    *,
    sections: int = 36,
    cx: float = 0.0,
    cy: float = 0.0,
) -> trimesh.Trimesh:
    """Create a closed frustum (truncated cone) between z0 (radius r0) and z1 (radius r1)."""
    angles = np.linspace(0.0, 2.0 * pi, sections, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)
    verts = []
    # Bottom ring (0..sections-1), top ring (sections..2*sections-1), bottom center, top center
    for i in range(sections):
        verts.append([cx + r0 * ca[i], cy + r0 * sa[i], z0])
    for i in range(sections):
        verts.append([cx + r1 * ca[i], cy + r1 * sa[i], z1])
    b_center = 2 * sections
    t_center = 2 * sections + 1
    verts.append([cx, cy, z0])
    verts.append([cx, cy, z1])

    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        b0, b1 = i, j
        t0, t1 = sections + i, sections + j
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        faces.append([b_center, b1, b0])
        faces.append([t_center, t0, t1])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces))
    trimesh.repair.fix_normals(mesh)
    return mesh


def _ogive_nosecone_z(
    radius: float,
    z0: float,
    z1: float,
    *,
    rings: int = 16,
    sections: int = 48,
) -> trimesh.Trimesh:
    """Create a smooth tangent-ogive payload fairing nosecone from z0 to z1."""
    length = z1 - z0
    angles = np.linspace(0.0, 2.0 * pi, sections, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)
    verts = []
    # Rings from i=0 (base) to i=rings-1 (near tip), plus apex and base center
    for k in range(rings):
        u = k / float(rings)
        z = z0 + u * length
        # Smooth power-ogive profile (exact r=radius at u=0, r->0 smoothly at u=1)
        r = radius * (1.0 - u**1.65) ** 0.72
        for i in range(sections):
            verts.append([r * ca[i], r * sa[i], z])
    apex_idx = rings * sections
    base_idx = apex_idx + 1
    verts.append([0.0, 0.0, z1])
    verts.append([0.0, 0.0, z0])

    faces = []
    for k in range(rings - 1):
        row0 = k * sections
        row1 = (k + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            faces.append([row0 + i, row0 + j, row1 + j])
            faces.append([row0 + i, row1 + j, row1 + i])
    top_row = (rings - 1) * sections
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([top_row + i, top_row + j, apex_idx])
        faces.append([base_idx, j, i])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces))
    trimesh.repair.fix_normals(mesh)
    return mesh


def _box(
    extents: tuple[float, float, float],
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _to_gltf_scene(parts: dict[str, trimesh.Trimesh]) -> trimesh.Scene:
    """Convert named Z-up meshes into a glTF 2.0 Y-up scene."""
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        m = mesh.copy()
        m.apply_transform(_T_ZUP_TO_GLTF_YUP)
        scene.add_geometry(m, node_name=name, geom_name=name)
    return scene


def build_booster_hull_parts(cfg: DM100VehicleConfig) -> dict[str, trimesh.Trimesh]:
    """Build 1st-stage booster hull (pivot at hull base z = 0.0 m)."""
    g = cfg.geometry
    r = g.body_radius_m
    h_tot = g.body_length_m  # 24.0 m
    z_inter = h_tot - 3.2  # 20.8 m -> 3.2 m black carbon interstage

    white_paint = (236, 239, 244, 255)
    soot_skirt = (46, 50, 58, 255)
    carbon_black = (30, 34, 42, 255)
    nozzle_metal = (68, 72, 82, 255)
    accent_band = (22, 68, 128, 255)
    rcs_metal = (150, 158, 168, 255)

    parts: dict[str, trimesh.Trimesh] = {}

    # Lower thermal protection skirt (0.0 -> 2.2 m)
    parts["aft_thermal_skirt"] = _paint(
        _cylinder_z(r * 1.005, 0.0, 2.2), soot_skirt, metallic=0.20, roughness=0.72
    )
    # Main LOX/CH4 white tank fuselage (2.2 -> 20.8 m)
    parts["main_tank_hull"] = _paint(
        _cylinder_z(r, 2.2, z_inter), white_paint, metallic=0.08, roughness=0.42
    )
    # Identification band near upper LOX dome (18.8 -> 19.6 m)
    parts["dm100_accent_band"] = _paint(
        _cylinder_z(r * 1.004, 18.8, 19.6), accent_band, metallic=0.12, roughness=0.38
    )
    # Carbon-composite interstage (20.8 -> 24.0 m)
    parts["interstage_hull"] = _paint(
        _cylinder_z(r * 1.003, z_inter, h_tot), carbon_black, metallic=0.18, roughness=0.58
    )
    # Cable/COPV raceways along +/-45 deg azimuths
    for idx, az_deg in enumerate((45.0, 225.0)):
        az = radians(az_deg)
        rx, ry = (r + 0.03) * cos(az), (r + 0.03) * sin(az)
        raceway = _box((0.14, 0.14, 18.0), (0.0, 0.0, 11.4))
        rot = trimesh.transformations.rotation_matrix(az, [0.0, 0.0, 1.0])
        raceway.apply_transform(rot)
        raceway.apply_translation([rx, ry, 0.0])
        parts[f"raceway_{idx}"] = _paint(raceway, carbon_black, metallic=0.15, roughness=0.55)

    # 4 Grid-fin hydraulic actuator blisters at z = 22.4 m (+X, -X, +Y, -Y)
    fin_z = cfg.fins.station_m
    for idx, (ux, uy) in enumerate(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))):
        housing = _box((0.36, 0.36, 0.85), (ux * (r - 0.02), uy * (r - 0.02), fin_z))
        parts[f"fin_actuator_{idx}"] = _paint(housing, rcs_metal, metallic=0.65, roughness=0.35)

    # 4 Cold-gas N2 RCS thruster pods at z = 22.4 m (45, 135, 225, 315 deg)
    for idx, az_deg in enumerate((45.0, 135.0, 225.0, 315.0)):
        az = radians(az_deg)
        pod = _box((0.28, 0.22, 0.42), (0.0, 0.0, 0.0))
        pod.apply_transform(trimesh.transformations.rotation_matrix(az, [0.0, 0.0, 1.0]))
        pod.apply_translation([(r + 0.06) * cos(az), (r + 0.06) * sin(az), g.rcs_station_m])
        parts[f"rcs_pod_{idx}"] = _paint(pod, rcs_metal, metallic=0.72, roughness=0.32)

    # Octaweb aft bulkhead dome
    parts["engine_bulkhead"] = _paint(
        _cylinder_z(r * 0.96, 0.0, 0.28), carbon_black, metallic=0.45, roughness=0.50
    )

    # 7 Methalox bell nozzles (1 center + 6 outer ring), rim exactly at nozzle_station_m = -0.20 m
    z_rim = g.nozzle_station_m  # -0.20 m
    z_throat = 0.25
    nozzle_xy = [(0.0, 0.0)] + [
        (0.92 * cos(radians(60.0 * k)), 0.92 * sin(radians(60.0 * k))) for k in range(6)
    ]
    for idx, (nx, ny) in enumerate(nozzle_xy):
        r_exit = 0.34 if idx == 0 else 0.31
        bell = _frustum_z(r_exit, 0.14, z_rim, z_throat, sections=28, cx=nx, cy=ny)
        parts[f"methalox_bell_{idx}"] = _paint(bell, nozzle_metal, metallic=0.82, roughness=0.30)

    return parts


def build_upper_stage_parts(cfg: DM100VehicleConfig) -> dict[str, trimesh.Trimesh]:
    """Build Stage 2 tank + vacuum bell + ogive payload fairing (pivot at z = 0 interface)."""
    g = cfg.geometry
    r = g.body_radius_m
    h_up = g.upper_length_m  # 9.0 m

    white_paint = (238, 241, 246, 255)
    carbon_ring = (34, 38, 46, 255)
    vac_bell = (76, 80, 92, 255)

    parts: dict[str, trimesh.Trimesh] = {}
    parts["stage2_sep_ring"] = _paint(
        _cylinder_z(r * 1.002, 0.0, 0.35), carbon_ring, metallic=0.25, roughness=0.50
    )
    parts["stage2_tank"] = _paint(
        _cylinder_z(r, 0.35, 4.4), white_paint, metallic=0.10, roughness=0.42
    )
    parts["fairing_barrel"] = _paint(
        _cylinder_z(r * 1.015, 4.4, 6.0), white_paint, metallic=0.06, roughness=0.38
    )
    parts["fairing_ogive"] = _paint(
        _ogive_nosecone_z(r * 1.015, 6.0, h_up), white_paint, metallic=0.06, roughness=0.38
    )
    # Single Methalox vacuum nozzle tucked inside interstage (-1.15 .. 0.05 m)
    parts["stage2_vac_bell"] = _paint(
        _frustum_z(0.68, 0.22, -1.15, 0.05, sections=36), vac_bell, metallic=0.85, roughness=0.28
    )
    return parts


def build_grid_fin_parts(cfg: DM100VehicleConfig) -> dict[str, trimesh.Trimesh]:
    """Build one titanium waffle-lattice grid fin (1.20 m span x 1.25 m chord = 1.50 m^2).

    Local frame:
      - Pivot `(0, 0, 0)` at the hull root hinge axis.
      - Fin extends radially along `+X` (`x in [0, 1.20 m]`), thickness along `Y` (`0.10 m`),
        chord along `Z` (`z in [-0.625, +0.625 m]`).
      - Deflection rotates around local `+X` (radial hinge axis).
    """
    _ = cfg
    span = 1.20
    chord = 1.25
    depth = 0.12
    wall = 0.032
    titanium = (78, 84, 94, 255)

    parts: dict[str, trimesh.Trimesh] = {}
    # Root shaft
    shaft = trimesh.creation.cylinder(radius=0.065, height=0.22, sections=20)
    shaft.apply_transform(trimesh.transformations.rotation_matrix(pi / 2.0, [0.0, 1.0, 0.0]))
    shaft.apply_translation([0.05, 0.0, 0.0])
    parts["fin_shaft"] = _paint(shaft, titanium, metallic=0.80, roughness=0.35)

    # Outer rectangular frame
    parts["frame_root"] = _paint(
        _box((wall, depth, chord), (0.12, 0.0, 0.0)), titanium, metallic=0.78, roughness=0.36
    )
    parts["frame_tip"] = _paint(
        _box((wall, depth, chord), (span, 0.0, 0.0)), titanium, metallic=0.78, roughness=0.36
    )
    parts["frame_top"] = _paint(
        _box((span - 0.12, depth, wall), (0.5 * (span + 0.12), 0.0, 0.5 * chord)),
        titanium,
        metallic=0.78,
        roughness=0.36,
    )
    parts["frame_bot"] = _paint(
        _box((span - 0.12, depth, wall), (0.5 * (span + 0.12), 0.0, -0.5 * chord)),
        titanium,
        metallic=0.78,
        roughness=0.36,
    )

    # 4x4 internal waffle ribs (distinctive hypersonic titanium grid-fin silhouette)
    rib = 0.020
    for i, frac in enumerate((0.25, 0.50, 0.75)):
        x_pos = 0.12 + frac * (span - 0.12)
        parts[f"rib_z_{i}"] = _paint(
            _box((rib, depth * 0.92, chord), (x_pos, 0.0, 0.0)),
            titanium,
            metallic=0.78,
            roughness=0.38,
        )
    for j, frac in enumerate((-0.25, 0.0, 0.25)):
        z_pos = frac * chord
        parts[f"rib_x_{j}"] = _paint(
            _box((span - 0.12, depth * 0.92, rib), (0.5 * (span + 0.12), 0.0, z_pos)),
            titanium,
            metallic=0.78,
            roughness=0.38,
        )
    return parts


def build_landing_leg_parts(cfg: DM100VehicleConfig) -> dict[str, trimesh.Trimesh]:
    """Build one deployed carbon-fiber A-frame landing leg (pivot at hull anchor z=1.2 m, r=1.55 m).

    In deployed pose (rotation 0 deg around local +Y):
      - Root hinge at `(0, 0, 0)` (mounts on hull at `(r=1.55 m, z=1.2 m)`).
      - Footpad bottom reaches `x = half_span_m - body_radius_m = +2.45 m`,
        `z = foot_station_m - anchor_station_m = -2.20 m` (so world z = -1.0 m when base z = 0).
    """
    g = cfg.geometry
    gear = cfg.landing_gear
    dx = gear.half_span_m - g.body_radius_m  # 4.0 - 1.55 = 2.45 m
    dz = gear.foot_station_m - gear.anchor_station_m  # -1.0 - 1.2 = -2.20 m
    length = sqrt(dx * dx + dz * dz)

    carbon = (34, 38, 46, 255)
    piston_steel = (165, 172, 182, 255)
    pad_dark = (28, 30, 36, 255)

    def _strut_between(p0: np.ndarray, p1: np.ndarray, radius: float) -> trimesh.Trimesh:
        vec = p1 - p0
        seg_len = float(np.linalg.norm(vec))
        cyl = trimesh.creation.cylinder(radius=radius, height=seg_len, sections=16)
        u = vec / max(seg_len, 1e-9)
        z_ax = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_ax, u)
        dot = float(np.clip(np.dot(z_ax, u), -1.0, 1.0))
        if float(np.linalg.norm(cross)) > 1e-9:
            ang = float(np.arccos(dot))
            rot = trimesh.transformations.rotation_matrix(ang, cross)
            cyl.apply_transform(rot)
        elif dot < 0.0:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(pi, [1.0, 0.0, 0.0]))
        cyl.apply_translation(0.5 * (p0 + p1))
        return cyl

    parts: dict[str, trimesh.Trimesh] = {}
    p_foot = np.array([dx, 0.0, dz + 0.14], dtype=np.float64)
    p_root_l = np.array([0.0, -0.34, 0.0], dtype=np.float64)
    p_root_r = np.array([0.0, 0.34, 0.0], dtype=np.float64)
    p_upper_brace = np.array([0.0, 0.0, 2.6], dtype=np.float64)
    p_mid = 0.58 * p_foot

    # Main A-frame twin carbon beams + upper telescoping pneumatic strut
    parts["leg_arm_left"] = _paint(
        _strut_between(p_root_l, p_foot, 0.11), carbon, metallic=0.22, roughness=0.52
    )
    parts["leg_arm_right"] = _paint(
        _strut_between(p_root_r, p_foot, 0.11), carbon, metallic=0.22, roughness=0.52
    )
    parts["leg_telescoping_brace"] = _paint(
        _strut_between(p_upper_brace, p_mid, 0.085), piston_steel, metallic=0.82, roughness=0.25
    )
    # Footpad disk (bottom exactly at z = dz = -2.20 m)
    parts["leg_footpad"] = _paint(
        _cylinder_z(0.32, dz, dz + 0.12, sections=20, cx=dx, cy=0.0),
        pad_dark,
        metallic=0.35,
        roughness=0.68,
    )
    _ = length
    return parts


def build_landing_pad_parts(cfg: DM100VehicleConfig) -> dict[str, trimesh.Trimesh]:
    """Build 10.0 m radius DM100 landing pad with 2.0 m target bullseye & blast apron."""
    r_pad = cfg.landing_gear.pad_radius_m  # 10.0 m
    r_bull = cfg.safety.max_position_error_m  # 2.0 m

    asphalt = (52, 56, 64, 255)
    concrete = (118, 122, 128, 255)
    white_ring = (232, 236, 240, 255)
    amber_ring = (230, 172, 34, 255)

    parts: dict[str, trimesh.Trimesh] = {}
    parts["blast_apron"] = _paint(
        _cylinder_z(r_pad + 8.0, -0.30, -0.04, sections=64), asphalt, metallic=0.02, roughness=0.92
    )
    parts["concrete_pad"] = _paint(
        _cylinder_z(r_pad, -0.04, 0.0, sections=64), concrete, metallic=0.02, roughness=0.86
    )
    parts["outer_safety_ring"] = _paint(
        _cylinder_z(r_pad - 0.35, 0.0, 0.015, sections=64),
        amber_ring,
        metallic=0.05,
        roughness=0.70,
    )
    parts["inner_concrete"] = _paint(
        _cylinder_z(r_pad - 0.85, 0.0, 0.018, sections=64), concrete, metallic=0.02, roughness=0.86
    )
    parts["bullseye_ring"] = _paint(
        _cylinder_z(r_bull, 0.0, 0.022, sections=48), white_ring, metallic=0.05, roughness=0.65
    )
    parts["bullseye_center"] = _paint(
        _cylinder_z(r_bull - 0.30, 0.0, 0.025, sections=48), asphalt, metallic=0.02, roughness=0.88
    )
    # Crosshair H-markings along +/-X and +/-Y
    parts["cross_x"] = _paint(
        _box((1.4, 0.22, 0.012), (0.0, 0.0, 0.028)), white_ring, metallic=0.05, roughness=0.65
    )
    parts["cross_y"] = _paint(
        _box((0.22, 1.4, 0.012), (0.0, 0.0, 0.028)), white_ring, metallic=0.05, roughness=0.65
    )
    return parts


def generate_all_assets(
    output_dir: str | Path,
    vehicle: DM100VehicleConfig | None = None,
) -> dict[str, dict[str, object]]:
    """Generate all DM100 3D procedural GLB meshes and return an inspection manifest."""
    cfg = vehicle or DM100VehicleConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    builders = {
        "dm100_booster_hull": build_booster_hull_parts,
        "dm100_upper_stage": build_upper_stage_parts,
        "dm100_grid_fin": build_grid_fin_parts,
        "dm100_landing_leg": build_landing_leg_parts,
        "dm100_landing_pad": build_landing_pad_parts,
    }

    manifest: dict[str, dict[str, object]] = {}
    for asset_name, builder in builders.items():
        parts_zup = builder(cfg)
        # Measure exact Z-up bounds before glTF Y-up conversion
        combined_zup = trimesh.util.concatenate(list(parts_zup.values()))
        bounds_zup = combined_zup.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]
        extents_zup = bounds_zup[1] - bounds_zup[0]

        scene = _to_gltf_scene(parts_zup)
        glb_path = out / f"{asset_name}.glb"
        glb_bytes = scene.export(file_type="glb")
        glb_path.write_bytes(glb_bytes)

        manifest[asset_name] = {
            "path": str(glb_path),
            "bytes": len(glb_bytes),
            "part_count": len(parts_zup),
            "vertex_count": int(len(combined_zup.vertices)),
            "face_count": int(len(combined_zup.faces)),
            "bounds_zup_m": {
                "min": [round(float(v), 4) for v in bounds_zup[0]],
                "max": [round(float(v), 4) for v in bounds_zup[1]],
                "extents": [round(float(v), 4) for v in extents_zup],
            },
        }

    (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DM100 3D procedural GLB meshes for UE 5.4"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ue5/assets/meshes"),
        help="Output directory for generated .glb meshes",
    )
    args = parser.parse_args()
    manifest = generate_all_assets(args.out)
    for name, info in manifest.items():
        ext = info["bounds_zup_m"]["extents"]
        print(
            f"[OK] {name}.glb ({info['bytes']:,} bytes, {info['face_count']:,} tris, "
            f"extents_zup={ext} m)"
        )


if __name__ == "__main__":
    main()
