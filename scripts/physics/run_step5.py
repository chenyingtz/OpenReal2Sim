#!/usr/bin/env python3
"""Step 5 — Physical Grounding & Alignment Optimization (orchestrator).

Runs all three sub-steps in order:
  5.1  Gravity Plane Alignment   (RANSAC, R_grav → +Z)
  5.2  Contact / Penetration Opt (SDF-based foot seating)
  5.3  URDF Generation           (mass / inertia / friction)

Usage:
    python scripts/physics/run_step5.py \
        --robot_mesh   assets/lego_dog.obj \
        --terrain_mesh assets/terrain.obj \
        --output_dir   assets/refined/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_urdf import _default_leg_layout, generate_urdf
from gravity_align import load_mesh
from optimize_contacts import run_step5_alignment


def main():
    ap = argparse.ArgumentParser(description="Step 5 — full pipeline")
    ap.add_argument("--robot_mesh", required=True)
    ap.add_argument("--terrain_mesh", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--robot_name", default="lego_dog")
    ap.add_argument("--clearance", type=float, default=0.002)
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--friction", type=float, default=0.9)
    ap.add_argument("--restitution", type=float, default=0.05)
    ap.add_argument("--legs_json", default=None)
    ap.add_argument("--no_legs", action="store_true",
                    help="Emit a single-link rigid-body URDF (object-centric).")
    ap.add_argument("--skip_gravity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 5.1 + 5.2 ────────────────────────────────────────────────
    report = run_step5_alignment(
        robot_path=args.robot_mesh,
        terrain_path=args.terrain_mesh,
        output_dir=out_dir,
        clearance=args.clearance,
        do_gravity_align=not args.skip_gravity,
        seed=args.seed,
    )
    refined_robot_path = report["robot_mesh_refined"]

    # ── 5.3 ──────────────────────────────────────────────────────
    legs = None
    if not args.no_legs:
        if args.legs_json:
            legs = json.loads(Path(args.legs_json).read_text())
        else:
            legs = _default_leg_layout(load_mesh(refined_robot_path))

    urdf_path, urdf_summary = generate_urdf(
        robot_mesh_path=refined_robot_path,
        output_dir=out_dir,
        robot_name=args.robot_name,
        mass_override=args.mass,
        friction=args.friction,
        restitution=args.restitution,
        legs=legs,
    )

    report["urdf"] = urdf_summary
    with open(out_dir / "step5_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[step5] done — report at {out_dir / 'step5_report.json'}")


if __name__ == "__main__":
    main()
