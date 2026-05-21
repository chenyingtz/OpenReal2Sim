#!/usr/bin/env python3
"""Direct PPO baseline (comparison against residual PPO).

Same env, same PPO trainer, same reward function as residual PPO — but the
policy emits absolute normalized joint commands in [-1, +1] instead of an
additive residual on the open-loop trot. The env maps each joint's [-1, +1]
output onto its [q_lo, q_hi] range. The trot baseline is NOT added.

This is the standard "RL from scratch" baseline; it tests how much of the
residual approach's sample efficiency comes from the open-loop gait prior.
Expect this to need ~5-10x more timesteps than residual PPO to reach the
same tracking error, and to be more sensitive to reward shaping.

CLI matches train_residual_ppo.py:

    python scripts/training/train_direct_ppo.py \
        --config configs/lego_dog_locomotion.yaml

Default log_dir is `<config.output.log_dir>_direct/` so direct and residual
runs do not overwrite each other's checkpoints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_residual_ppo import LegoDogEnv, PPOTrainer, _load_cfg, _pick_device


def main():
    ap = argparse.ArgumentParser(description="Direct PPO (no-trot) baseline")
    ap.add_argument("--config", required=True,
                    help="Path to lego_dog_locomotion.yaml")
    ap.add_argument("--total_timesteps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log_dir_suffix", default="_direct",
                    help="Suffix appended to config.output.log_dir so direct "
                         "and residual runs don't share a checkpoint dir.")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))
    device = _pick_device(args.device or cfg.training.get("device", "auto"))

    # Re-route the log directory so this run doesn't clobber residual checkpoints.
    base_log = Path(cfg.output["log_dir"])
    cfg.output["log_dir"] = str(base_log.parent / (base_log.name + args.log_dir_suffix))

    env = LegoDogEnv(cfg, seed=int(cfg.training["seed"]), action_mode="direct")
    print(f"[direct-ppo] obs_dim={env.obs_dim}  act_dim={env.act_dim}  "
          f"policy_act_scale={env.policy_act_scale}  device={device}")
    print(f"[direct-ppo] log_dir={cfg.output['log_dir']}")

    trainer = PPOTrainer(env, cfg, device=device)
    trainer.train(total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()
