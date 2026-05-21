#!/usr/bin/env python3
"""Step 5.3 — URDF Generation.

Build a URDF for the LEGO quadruped from the gravity-aligned, contact-
optimized robot mesh. Mass/inertia defaults assume a plastic LEGO body
(ABS, ~1050 kg/m^3); friction defaults are tuned for rubber-on-tile.

The base URDF wraps the full body as a single rigid link — exactly what
Step 6 (object-centric 6D tracking with FoundationPose) expects. If a leg
configuration is provided via --legs_json, four hip + knee revolute
joints are added so the residual RL policy in Step 7 has a useful action
space; otherwise the file is a single-link rigid body and the residual
controller will degenerate gracefully to torso-only tracking.
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

from gravity_align import load_mesh


PLASTIC_DENSITY_KG_PER_M3 = 1050.0  # ABS plastic, the LEGO standard
DEFAULT_LEG_MASS_FRACTION = 0.05    # share of total mass per leg link
DEFAULT_FRICTION = 0.9              # rubber-style tire on tile
DEFAULT_RESTITUTION = 0.05


def _estimate_inertia(mesh: trimesh.Trimesh, mass: float) -> np.ndarray:
    """Return a 3x3 inertia tensor about the mesh COM."""
    if mesh.volume > 1e-9 and mesh.is_watertight:
        density = mass / mesh.volume
        mesh = mesh.copy()
        mesh.density = density
        return np.asarray(mesh.moment_inertia)
    # Fallback: solid box with the AABB extents.
    extents = mesh.extents
    lx, ly, lz = (max(float(e), 1e-3) for e in extents)
    ixx = mass * (ly * ly + lz * lz) / 12.0
    iyy = mass * (lx * lx + lz * lz) / 12.0
    izz = mass * (lx * lx + ly * ly) / 12.0
    return np.diag([ixx, iyy, izz])


def _estimate_mass(mesh: trimesh.Trimesh, mass_override: float | None) -> float:
    if mass_override is not None:
        return float(mass_override)
    if mesh.is_watertight and mesh.volume > 1e-9:
        return float(mesh.volume * PLASTIC_DENSITY_KG_PER_M3)
    # Approximate as a solid AABB box of plastic.
    return float(np.prod(mesh.extents) * PLASTIC_DENSITY_KG_PER_M3 * 0.5)


def _inertial_block(parent: ET.Element, mass: float, com: Iterable[float], inertia: np.ndarray):
    inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{c:.6f}" for c in com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia[0, 0]:.8f}",
        ixy=f"{inertia[0, 1]:.8f}",
        ixz=f"{inertia[0, 2]:.8f}",
        iyy=f"{inertia[1, 1]:.8f}",
        iyz=f"{inertia[1, 2]:.8f}",
        izz=f"{inertia[2, 2]:.8f}",
    )


def _visual_collision(parent: ET.Element, mesh_filename: str, friction: float, restitution: float):
    for tag in ("visual", "collision"):
        node = ET.SubElement(parent, tag)
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "mesh", filename=mesh_filename)
    contact = ET.SubElement(parent, "contact")
    ET.SubElement(contact, "lateral_friction", value=f"{friction}")
    ET.SubElement(contact, "rolling_friction", value="0.001")
    ET.SubElement(contact, "spinning_friction", value="0.001")
    ET.SubElement(contact, "restitution", value=f"{restitution}")


def _add_leg(
    robot: ET.Element,
    leg_name: str,
    hip_xyz: Iterable[float],
    upper_length: float,
    lower_length: float,
    upper_radius: float,
    lower_radius: float,
    mass_per_link: float,
):
    """Two-link revolute leg: hip (pitch) + knee (pitch)."""
    # ───── upper link ─────
    upper = ET.SubElement(robot, "link", name=f"{leg_name}_upper")
    iner = mass_per_link * (3 * upper_radius ** 2 + upper_length ** 2) / 12.0
    _inertial_block(
        upper,
        mass_per_link,
        (0.0, 0.0, -upper_length / 2.0),
        np.diag([iner, iner, 0.5 * mass_per_link * upper_radius ** 2]),
    )
    for tag in ("visual", "collision"):
        node = ET.SubElement(upper, tag)
        ET.SubElement(node, "origin", xyz=f"0 0 {-upper_length/2:.4f}", rpy="0 0 0")
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "cylinder", radius=f"{upper_radius}", length=f"{upper_length}")

    hip = ET.SubElement(robot, "joint", name=f"{leg_name}_hip", type="revolute")
    ET.SubElement(hip, "parent", link="base_link")
    ET.SubElement(hip, "child", link=f"{leg_name}_upper")
    ET.SubElement(hip, "origin",
                  xyz=" ".join(f"{c:.4f}" for c in hip_xyz),
                  rpy="0 0 0")
    ET.SubElement(hip, "axis", xyz="0 1 0")
    ET.SubElement(hip, "limit", lower="-1.2", upper="1.2", effort="3.0", velocity="6.0")
    ET.SubElement(hip, "dynamics", damping="0.05", friction="0.01")

    # ───── lower link ─────
    lower = ET.SubElement(robot, "link", name=f"{leg_name}_lower")
    iner_l = mass_per_link * (3 * lower_radius ** 2 + lower_length ** 2) / 12.0
    _inertial_block(
        lower,
        mass_per_link,
        (0.0, 0.0, -lower_length / 2.0),
        np.diag([iner_l, iner_l, 0.5 * mass_per_link * lower_radius ** 2]),
    )
    for tag in ("visual", "collision"):
        node = ET.SubElement(lower, tag)
        ET.SubElement(node, "origin", xyz=f"0 0 {-lower_length/2:.4f}", rpy="0 0 0")
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "cylinder", radius=f"{lower_radius}", length=f"{lower_length}")

    knee = ET.SubElement(robot, "joint", name=f"{leg_name}_knee", type="revolute")
    ET.SubElement(knee, "parent", link=f"{leg_name}_upper")
    ET.SubElement(knee, "child", link=f"{leg_name}_lower")
    ET.SubElement(knee, "origin", xyz=f"0 0 {-upper_length:.4f}", rpy="0 0 0")
    ET.SubElement(knee, "axis", xyz="0 1 0")
    ET.SubElement(knee, "limit", lower="-2.4", upper="0.0", effort="3.0", velocity="6.0")
    ET.SubElement(knee, "dynamics", damping="0.05", friction="0.01")


def _default_leg_layout(robot_mesh: trimesh.Trimesh) -> dict:
    """Place 4 hips at the corners of the torso AABB top face."""
    lo, hi = robot_mesh.bounds
    x_front, x_back = float(hi[0]) * 0.6, float(lo[0]) * 0.6
    y_left, y_right = float(hi[1]) * 0.6, float(lo[1]) * 0.6
    z_hip = float(lo[2])
    return {
        "FL": [x_front, y_left,  z_hip],
        "FR": [x_front, y_right, z_hip],
        "RL": [x_back,  y_left,  z_hip],
        "RR": [x_back,  y_right, z_hip],
    }


def generate_urdf(
    robot_mesh_path: str | Path,
    output_dir: str | Path,
    robot_name: str = "lego_dog",
    mass_override: float | None = None,
    friction: float = DEFAULT_FRICTION,
    restitution: float = DEFAULT_RESTITUTION,
    legs: dict | None = None,
    leg_geometry: dict | None = None,
) -> Path:
    """Write `<output_dir>/<robot_name>.urdf` plus a mesh copy alongside it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(robot_mesh_path)
    mesh_dst = output_dir / f"{robot_name}.obj"
    if Path(robot_mesh_path).resolve() != mesh_dst.resolve():
        shutil.copy2(robot_mesh_path, mesh_dst)

    mass = _estimate_mass(mesh, mass_override)
    inertia = _estimate_inertia(mesh, mass)
    com = mesh.center_mass if mesh.is_watertight else mesh.centroid

    robot = ET.Element("robot", name=robot_name)

    base = ET.SubElement(robot, "link", name="base_link")
    _inertial_block(base, mass, com, inertia)
    _visual_collision(base, mesh_dst.name, friction, restitution)

    if legs:
        geom = leg_geometry or {
            "upper_length": 0.04,
            "lower_length": 0.04,
            "upper_radius": 0.006,
            "lower_radius": 0.005,
        }
        per_leg_mass = mass * DEFAULT_LEG_MASS_FRACTION
        for name, hip_xyz in legs.items():
            _add_leg(
                robot,
                leg_name=name,
                hip_xyz=hip_xyz,
                upper_length=geom["upper_length"],
                lower_length=geom["lower_length"],
                upper_radius=geom["upper_radius"],
                lower_radius=geom["lower_radius"],
                mass_per_link=per_leg_mass,
            )

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    urdf_path = output_dir / f"{robot_name}.urdf"
    tree.write(urdf_path, xml_declaration=True, encoding="utf-8")

    summary = {
        "urdf": str(urdf_path),
        "robot_mesh": str(mesh_dst),
        "mass_kg": mass,
        "inertia_diag": np.diag(inertia).tolist(),
        "com": list(map(float, com)),
        "friction": friction,
        "restitution": restitution,
        "legs": list((legs or {}).keys()),
    }
    print(f"[urdf] mass = {mass:.4f} kg  inertia diag = {summary['inertia_diag']}")
    print(f"[urdf] wrote {urdf_path}")
    return urdf_path, summary


def main():
    ap = argparse.ArgumentParser(description="Step 5.3 — URDF generation")
    ap.add_argument("--robot_mesh", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--robot_name", default="lego_dog")
    ap.add_argument("--mass", type=float, default=None,
                    help="Override mass (kg). If unset, estimate from mesh volume * plastic density.")
    ap.add_argument("--friction", type=float, default=DEFAULT_FRICTION)
    ap.add_argument("--restitution", type=float, default=DEFAULT_RESTITUTION)
    ap.add_argument("--legs_json", default=None,
                    help="Optional path to JSON of {leg_name: [x,y,z]} hip mounts.")
    args = ap.parse_args()

    legs = None
    if args.legs_json:
        legs = json.loads(Path(args.legs_json).read_text())
    else:
        # If the mesh has clear quadruped extents, auto-place 4 legs.
        mesh = load_mesh(args.robot_mesh)
        legs = _default_leg_layout(mesh)

    generate_urdf(
        robot_mesh_path=args.robot_mesh,
        output_dir=args.output_dir,
        robot_name=args.robot_name,
        mass_override=args.mass,
        friction=args.friction,
        restitution=args.restitution,
        legs=legs,
    )


if __name__ == "__main__":
    main()
