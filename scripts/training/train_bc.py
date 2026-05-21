#!/usr/bin/env python3
"""Behavior Cloning baseline (comparison against residual / direct PPO).

Trains a student policy to imitate an expert's per-step action via supervised
MSE loss on (obs, expert_action) pairs. The expert is either:
  - a trained PPO checkpoint (`--expert_checkpoint path/to/policy_final.pt`),
    in which case BC distills that expert; the student inherits the expert's
    action_mode (residual or direct), or
  - the open-loop trot (`--expert trot`), which is a useful sanity check:
    can the student learn the trot exactly from demonstrations? If not, the
    student's MLP is mis-specified.

Optionally runs DAgger for `--dagger_iters` rounds to address compounding-
error distribution shift: in each round, roll out the *student* in the env,
label every visited state with the expert's action, aggregate into the
dataset, and re-train. This typically halves test-time error vs pure BC.

CLI:
    python scripts/training/train_bc.py \
        --config            configs/lego_dog_locomotion.yaml \
        --expert_checkpoint runs/lego_dog_residual/policy_final.pt \
        --n_demo_episodes   50

Output: a checkpoint in `<config.output.log_dir>_bc/` loadable by
visualize_policy.py with no extra flags.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_residual_ppo import (
    GaussianPolicy, LegoDogEnv, TrotController, _load_cfg, _pick_device,
)


# ───────────────────────── Expert wrappers ─────────────────────────

class _CheckpointExpert:
    """Wraps a trained PPO policy for deterministic action queries."""

    def __init__(self, policy: GaussianPolicy, device: str):
        self.policy = policy
        self.device = device

    def action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            _, mean = self.policy(obs_t)
            return mean.squeeze(0).cpu().numpy()


class _TrotExpert:
    """Wraps the open-loop trot controller as a BC expert. Action is the
    residual (which is zero by definition — the policy should output zero)."""

    def __init__(self, action_mode: str):
        # In residual mode the trot is the baseline so the "demonstration"
        # is action = 0. In direct mode the demonstration is the trot's
        # joint targets normalized to the policy's [-1, +1] output range.
        self.action_mode = action_mode

    def action(self, obs: np.ndarray, env: LegoDogEnv) -> np.ndarray:
        if self.action_mode == "residual":
            return np.zeros(env.act_dim, dtype=np.float32)
        baseline = env.trot(env.t_sim)
        normalized = (baseline - env._direct_center) / np.maximum(env._direct_half_range, 1e-6)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# ───────────────────────── Rollout collection ─────────────────────────

def _query_expert(expert, env: LegoDogEnv, obs: np.ndarray) -> np.ndarray:
    if isinstance(expert, _TrotExpert):
        return expert.action(obs, env)
    return expert.action(obs)


def collect_demonstrations(
    expert, env: LegoDogEnv, n_episodes: int, seed: int = 0,
    use_student: Optional[GaussianPolicy] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Roll out `expert` (or `use_student` if given) and record (obs, expert_action).

    When `use_student` is set, the *student* drives the env but every state
    is labeled by the expert — i.e. DAgger.
    """
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []
    ep_rewards, ep_lengths = [], []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_r, ep_l = 0.0, 0
        done = False
        while not done:
            expert_action = _query_expert(expert, env, obs)
            obs_list.append(obs.astype(np.float32))
            act_list.append(expert_action.astype(np.float32))

            if use_student is None:
                step_action = expert_action
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    _, mean = use_student(obs_t)
                    step_action = mean.squeeze(0).cpu().numpy()
                # Small exploration noise so DAgger sees a wider state distrib.
                step_action = step_action + rng.normal(scale=0.02, size=step_action.shape)

            obs, reward, done, _ = env.step(step_action)
            ep_r += reward
            ep_l += 1

        ep_rewards.append(ep_r)
        ep_lengths.append(ep_l)

    stats = {
        "episodes": n_episodes,
        "mean_ep_reward": float(np.mean(ep_rewards)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "total_pairs": len(obs_list),
    }
    return np.array(obs_list), np.array(act_list), stats


# ───────────────────────── Supervised training loop ─────────────────────────

def train_supervised(
    student: GaussianPolicy, obs: np.ndarray, act: np.ndarray,
    epochs: int, batch_size: int, lr: float, device: str,
    val_split: float = 0.1, log_every: int = 5,
) -> dict:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    n = len(obs_t)
    n_val = int(n * val_split)
    perm = np.random.permutation(n)
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long, device=device)
    train_idx = torch.as_tensor(perm[n_val:], dtype=torch.long, device=device)

    opt = torch.optim.Adam(student.mean_net.parameters(), lr=lr)
    last_train, last_val = float("nan"), float("nan")
    for ep in range(epochs):
        student.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        losses = []
        for start in range(0, len(order), batch_size):
            mb = order[start:start + batch_size]
            _, pred_mean = student(obs_t[mb])
            loss = F.mse_loss(pred_mean, act_t[mb])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.mean_net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        last_train = float(np.mean(losses))

        if n_val > 0:
            student.eval()
            with torch.no_grad():
                _, val_pred = student(obs_t[val_idx])
                last_val = float(F.mse_loss(val_pred, act_t[val_idx]).item())

        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"[bc] epoch {ep+1:3d}/{epochs}  "
                  f"train_mse={last_train:.5f}  val_mse={last_val:.5f}")

    return {"train_mse": last_train, "val_mse": last_val}


# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Behavior Cloning (with optional DAgger)")
    ap.add_argument("--config", required=True,
                    help="Path to lego_dog_locomotion.yaml")
    ap.add_argument("--expert_checkpoint", default=None,
                    help="Trained PPO policy_final.pt to clone. Mutually "
                         "exclusive with --expert trot.")
    ap.add_argument("--expert", choices=["checkpoint", "trot"], default="checkpoint",
                    help="If 'trot' use the open-loop trot as the demonstrator "
                         "(no checkpoint needed). Default 'checkpoint'.")
    ap.add_argument("--action_mode", choices=["residual", "direct"], default=None,
                    help="Override the env's action mode. Default: inherit from "
                         "the expert checkpoint, else 'residual' for trot.")
    ap.add_argument("--n_demo_episodes", type=int, default=50)
    ap.add_argument("--dagger_iters", type=int, default=3,
                    help="DAgger iterations (0 disables DAgger; pure BC)")
    ap.add_argument("--dagger_episodes_per_iter", type=int, default=20)
    ap.add_argument("--bc_epochs", type=int, default=60,
                    help="Supervised epochs per training phase (initial + each DAgger round)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log_dir_suffix", default="_bc",
                    help="Suffix appended to config.output.log_dir")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))
    device = _pick_device(args.device or cfg.training.get("device", "auto"))
    torch.manual_seed(int(cfg.training["seed"]))
    np.random.seed(int(cfg.training["seed"]))

    # Re-route log_dir.
    base_log = Path(cfg.output["log_dir"])
    log_dir = base_log.parent / (base_log.name + args.log_dir_suffix)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.output["log_dir"] = str(log_dir)

    # Decide action_mode.
    if args.action_mode is not None:
        action_mode = args.action_mode
    elif args.expert == "trot":
        action_mode = "residual"
    elif args.expert_checkpoint is not None:
        expert_ckpt = torch.load(args.expert_checkpoint, map_location=device,
                                 weights_only=False)
        action_mode = expert_ckpt.get("action_mode", "residual")
    else:
        action_mode = "residual"

    env = LegoDogEnv(cfg, seed=int(cfg.training["seed"]), action_mode=action_mode)
    print(f"[bc] obs_dim={env.obs_dim}  act_dim={env.act_dim}  "
          f"action_mode={action_mode}  device={device}")

    # Build student.
    student = GaussianPolicy(
        obs_dim=env.obs_dim, act_dim=env.act_dim,
        hidden=cfg.policy["hidden_sizes"],
        activation=cfg.policy["activation"],
        log_std_init=cfg.policy["log_std_init"],
        residual_scale=env.policy_act_scale,
    ).to(device)

    # Build expert.
    if args.expert == "trot":
        expert = _TrotExpert(action_mode=action_mode)
        print(f"[bc] expert: open-loop trot ({action_mode} mode)")
    else:
        if not args.expert_checkpoint:
            raise ValueError("--expert checkpoint requires --expert_checkpoint")
        expert_policy = GaussianPolicy(
            obs_dim=env.obs_dim, act_dim=env.act_dim,
            hidden=cfg.policy["hidden_sizes"],
            activation=cfg.policy["activation"],
            log_std_init=cfg.policy["log_std_init"],
            residual_scale=env.policy_act_scale,
        ).to(device)
        ckpt = torch.load(args.expert_checkpoint, map_location=device,
                          weights_only=False)
        expert_policy.load_state_dict(ckpt["policy"])
        expert_policy.eval()
        expert = _CheckpointExpert(expert_policy, device=device)
        print(f"[bc] expert: {args.expert_checkpoint} ({ckpt.get('method', '?')})")

    # ── Phase 1: collect initial demonstrations from the expert ──
    t0 = time.time()
    print(f"[bc] collecting {args.n_demo_episodes} demonstration episodes ...")
    obs_buf, act_buf, demo_stats = collect_demonstrations(
        expert, env, n_episodes=args.n_demo_episodes, seed=0, device=device,
    )
    print(f"[bc] demos: {demo_stats['total_pairs']} pairs  "
          f"mean_ep_R={demo_stats['mean_ep_reward']:+.3f}  "
          f"mean_ep_len={demo_stats['mean_ep_length']:.1f}")

    # ── Phase 2: supervised training on the demonstrations ──
    print(f"[bc] initial supervised training ({args.bc_epochs} epochs)")
    metrics = train_supervised(
        student, obs_buf, act_buf,
        epochs=args.bc_epochs, batch_size=args.batch_size, lr=args.lr,
        device=device,
    )

    # ── Phase 3: DAgger ──
    for it in range(args.dagger_iters):
        print(f"[bc] DAgger iter {it+1}/{args.dagger_iters} — "
              f"rolling out student and labeling with expert")
        new_obs, new_act, stats = collect_demonstrations(
            expert, env,
            n_episodes=args.dagger_episodes_per_iter,
            seed=100 + it,
            use_student=student,
            device=device,
        )
        print(f"[bc] new pairs: {stats['total_pairs']}  "
              f"student_mean_ep_R={stats['mean_ep_reward']:+.3f}  "
              f"student_mean_ep_len={stats['mean_ep_length']:.1f}")
        obs_buf = np.concatenate([obs_buf, new_obs], axis=0)
        act_buf = np.concatenate([act_buf, new_act], axis=0)
        metrics = train_supervised(
            student, obs_buf, act_buf,
            epochs=args.bc_epochs, batch_size=args.batch_size, lr=args.lr,
            device=device,
        )

    # ── Save checkpoint in the same shape as PPO checkpoints ──
    save_path = log_dir / "policy_final.pt"
    torch.save({
        "policy": student.state_dict(),
        "value": None,
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "residual_scale": env.residual_scale,
        "policy_act_scale": env.policy_act_scale,
        "action_mode": env.action_mode,
        "method": "bc",
        "bc_meta": {
            "expert": args.expert,
            "expert_checkpoint": args.expert_checkpoint,
            "n_demo_episodes": args.n_demo_episodes,
            "dagger_iters": args.dagger_iters,
            "final_train_mse": metrics["train_mse"],
            "final_val_mse": metrics["val_mse"],
            "total_pairs": int(len(obs_buf)),
        },
    }, save_path)
    print(f"[bc] checkpoint -> {save_path}")
    print(f"[bc] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
