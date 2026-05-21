#!/usr/bin/env python3
"""Visualize a trained residual-PPO policy.

Loads `policy_final.pt` (or any checkpoint), rolls the policy out in the
MuJoCo environment with offscreen rendering, and writes:
    <out_dir>/rollout.mp4       side-view video of the simulated robot
    <out_dir>/tracking.png      sim torso pose vs reference trajectory plot
    <out_dir>/rollout_log.csv   per-step pose, reward, tracking error

Headless Linux note: if the offscreen renderer fails to initialize, set
the OpenGL backend before running:
    MUJOCO_GL=egl python3 scripts/training/visualize_policy.py ...

Usage:
    python3 scripts/training/visualize_policy.py \
        --config     configs/lego_dog_locomotion.yaml \
        --checkpoint runs/lego_dog_residual/policy_final.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_residual_ppo import GaussianPolicy, LegoDogEnv, _load_cfg, _pick_device


def _make_tracking_camera(distance: float, azimuth: float, elevation: float
                          ) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera = getattr(mujoco, "mjv_defaultFreeCamera", None)
    if mujoco.mjv_defaultFreeCamera is not None:
        mujoco.mjv_defaultFreeCamera(mujoco.MjModel(), cam)  # type: ignore
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = [0.0, 0.0, 0.0]
    return cam


def _try_render(model, height: int, width: int):
    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as e:
        msg = (
            f"MuJoCo Renderer failed to initialize: {e}\n"
            f"If you are on a headless Linux server, set the GL backend "
            f"before running:\n"
            f"    MUJOCO_GL=egl python3 scripts/training/visualize_policy.py ...\n"
            f"On macOS the default backend should work; ensure you are not "
            f"running inside a context that lacks OpenGL access."
        )
        raise RuntimeError(msg) from e


def rollout(
    policy: GaussianPolicy, env: LegoDogEnv, n_steps: int,
    deterministic: bool, baseline_only: bool, device: str,
):
    obs = env.reset()
    rows = []
    cum_reward = 0.0
    for t in range(n_steps):
        if baseline_only:
            action = np.zeros(env.act_dim, dtype=np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic:
                    _, mean = policy(obs_t)
                    action = mean.squeeze(0).cpu().numpy()
                else:
                    action_t, _ = policy.act(obs_t)
                    action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, done, info = env.step(action)
        cum_reward += reward

        pos, quat, _, _ = env._base_pose()
        ref_pos, ref_quat = env._ref_at(env.t_sim)

        rows.append({
            "step": t,
            "t_sim": env.t_sim,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "ref_x": ref_pos[0], "ref_y": ref_pos[1], "ref_z": ref_pos[2],
            "pos_err": float(np.linalg.norm(pos - ref_pos)),
            "tilt_deg": info.get("tilt_deg", float("nan")),
            "rot_err_deg": info.get("rot_err_deg", float("nan")),
            "reward": reward,
            "done": int(done),
        })

        obs = next_obs
        if done:
            break

    return rows, cum_reward


def _write_mp4(frames, out_path: Path, fps: float):
    """Encode frames as MP4, with explicit FFMPEG plugin selection.

    Without the explicit `format="FFMPEG"`, imageio's auto-detection silently
    falls back to TIFF when `imageio-ffmpeg` isn't installed — which then
    rejects the `fps` / `codec` kwargs with a confusing TypeError. Forcing
    FFMPEG raises a clear ImportError instead.
    """
    import imageio.v2 as iio
    try:
        iio.mimsave(str(out_path), frames, format="FFMPEG",
                    fps=fps, codec="libx264", quality=8)
        return out_path
    except (ImportError, ValueError, Exception) as e:
        # Plugin not available — write PNG frames instead so the user still
        # has visual evidence.
        png_dir = out_path.with_suffix("")
        png_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            iio.imwrite(str(png_dir / f"frame_{i:05d}.png"), frame)
        raise RuntimeError(
            f"MP4 encoding failed ({e}). Wrote {len(frames)} PNG frames to "
            f"{png_dir}/ instead. To enable MP4 output run:\n"
            f"    pip install imageio-ffmpeg\n"
            f"You can stitch the PNGs into an MP4 manually with:\n"
            f"    ffmpeg -framerate {fps} -i {png_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {out_path}"
        ) from e


def render_video(env: LegoDogEnv, policy: GaussianPolicy, rows, out_path: Path,
                 width: int, height: int, fps: float, deterministic: bool,
                 baseline_only: bool, device: str,
                 distance: float, azimuth: float, elevation: float):
    """Second pass that re-rolls the (deterministic) policy and captures frames."""
    renderer = _try_render(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.fixedcamid = -1
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    obs = env.reset()
    frames = []
    for t in range(len(rows)):
        if baseline_only:
            action = np.zeros(env.act_dim, dtype=np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic:
                    _, mean = policy(obs_t)
                    action = mean.squeeze(0).cpu().numpy()
                else:
                    action_t, _ = policy.act(obs_t)
                    action = action_t.squeeze(0).cpu().numpy()
        obs, _, _, _ = env.step(action)

        pos, *_ = env._base_pose()
        cam.lookat[0], cam.lookat[1], cam.lookat[2] = float(pos[0]), float(pos[1]), float(pos[2])
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render())

    _write_mp4(frames, out_path, fps)
    return out_path, len(frames)


def write_csv(rows, path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_plot(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t_sim"] for r in rows])
    sim = np.array([[r["pos_x"], r["pos_y"], r["pos_z"]] for r in rows])
    ref = np.array([[r["ref_x"], r["ref_y"], r["ref_z"]] for r in rows])
    err = np.array([r["pos_err"] for r in rows])
    rew = np.array([r["reward"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, name in enumerate(["x", "y", "z"]):
        axes[0].plot(t, sim[:, i], label=f"sim {name}")
        axes[0].plot(t, ref[:, i], "--", label=f"ref {name}", alpha=0.55)
    axes[0].set_ylabel("torso position (m)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, err)
    axes[1].set_ylabel("|p_cc − p_o| (m)")
    axes[1].axhline(0.08, color="r", linestyle=":", alpha=0.4, label="σ_p (reward knee)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, rew)
    axes[2].set_ylabel("step reward")
    axes[2].set_xlabel("simulation time (s)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize a trained residual PPO policy")
    ap.add_argument("--config", required=True,
                    help="Path to lego_dog_locomotion.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="Default: <output.log_dir>/policy_final.pt")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <output.log_dir>/visualization/")
    ap.add_argument("--n_steps", type=int, default=300,
                    help="Number of control steps to roll out")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True,
                    help="Use the policy mean (default; stable for visualization)")
    ap.add_argument("--stochastic", dest="deterministic", action="store_false",
                    help="Sample from the policy distribution")
    ap.add_argument("--baseline_only", action="store_true",
                    help="Skip the policy; render the pure trot baseline for comparison")
    ap.add_argument("--no_video", action="store_true", help="Skip MP4 rendering")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--cam_distance", type=float, default=0.6)
    ap.add_argument("--cam_azimuth", type=float, default=90.0)
    ap.add_argument("--cam_elevation", type=float, default=-25.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))
    device = _pick_device(args.device)

    log_dir = Path(cfg.output["log_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (log_dir / "policy_final.pt")
    out_dir = Path(args.out_dir) if args.out_dir else (log_dir / "visualization")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = LegoDogEnv(cfg, seed=int(cfg.training["seed"]))
    policy = GaussianPolicy(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden=cfg.policy["hidden_sizes"],
        activation=cfg.policy["activation"],
        log_std_init=cfg.policy["log_std_init"],
        residual_scale=env.residual_scale,
    ).to(device)
    if not args.baseline_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Pass --checkpoint explicitly "
                f"or pass --baseline_only to view the trot baseline."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        print(f"[viz] loaded checkpoint {ckpt_path}")
    else:
        print(f"[viz] baseline-only rollout (no policy loaded)")

    # First pass: collect logs without rendering (cheap).
    rows, cum_reward = rollout(
        policy=policy, env=env,
        n_steps=args.n_steps, deterministic=args.deterministic,
        baseline_only=args.baseline_only, device=device,
    )
    print(f"[viz] {len(rows)} steps  cum_reward={cum_reward:+.3f}  "
          f"mean_pos_err={np.mean([r['pos_err'] for r in rows]):.4f} m")

    csv_path = out_dir / "rollout_log.csv"
    write_csv(rows, csv_path)
    print(f"[viz] csv  -> {csv_path}")

    try:
        plot_path = out_dir / "tracking.png"
        write_plot(rows, plot_path)
        print(f"[viz] plot -> {plot_path}")
    except ImportError as e:
        print(f"[viz] WARN: matplotlib unavailable, skipping plot ({e})")

    if not args.no_video:
        try:
            video_path = out_dir / ("rollout_baseline.mp4" if args.baseline_only
                                    else "rollout.mp4")
            written, n_frames = render_video(
                env, policy, rows, video_path,
                width=args.width, height=args.height, fps=args.fps,
                deterministic=args.deterministic,
                baseline_only=args.baseline_only,
                device=device,
                distance=args.cam_distance,
                azimuth=args.cam_azimuth,
                elevation=args.cam_elevation,
            )
            print(f"[viz] mp4  -> {written}  ({n_frames} frames @ {args.fps:.0f} fps)")
        except ImportError as e:
            print(f"[viz] WARN: imageio unavailable, skipping mp4 ({e})")
        except RuntimeError as e:
            print(f"[viz] WARN: rendering failed: {e}")


if __name__ == "__main__":
    main()
