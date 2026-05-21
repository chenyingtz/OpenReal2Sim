#!/usr/bin/env python3
"""Step 7.3 — Residual PPO Trainer.

Implements the residual-RL formulation from the PDF:
    a_t = a_t^base + pi_theta(o_t)
where
    a_t^base       : open-loop diagonal trot (sinusoidal joint commands)
    pi_theta(o_t)  : Gaussian MLP policy outputting Delta a_t, bounded by
                     residual_scale, learned via PPO
The tracking reward follows the PDF formula:
    R_track = exp(-||p_cc - p_o||^2 / sigma_p^2)
            + exp(-Dist(q_cc, q_o)^2 / sigma_q^2)
plus torque + orientation regularization.

CLI matches the PDF:
    python scripts/training/train_residual_ppo.py \
        --config configs/lego_dog_locomotion.yaml

The script expects to find a `scene.yaml` produced by
`scripts/exports/export_to_isaac.py`. It locates it from
`<scene.urdf>/../isaac_export/scene.yaml` (resolving relative paths in the
config against the OpenReal2Sim root) and falls back to building paths
directly from the config if the export step was skipped.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ───────────────────────── Config dataclass ─────────────────────────

@dataclass
class TrainCfg:
    scene_yaml: Optional[Path]
    urdf: Path
    mjcf: Optional[Path]
    trajectory_npz: Path
    frame_zero_pose: Dict
    fps: float
    base_link: str
    joint_names: List[str]
    spawn_height_offset: float
    kp: float
    kd: float
    torque_clamp: float
    joint_armature: float
    dt: float
    control_dt: float
    gravity: List[float]
    gait: Dict
    policy: Dict
    reward: Dict
    training: Dict
    episode: Dict
    output: Dict


def _load_cfg(cfg_path: Path) -> TrainCfg:
    cfg_path = Path(cfg_path).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Look up the exported scene.yaml from the URDF location.
    repo_root = cfg_path.parent.parent  # configs/ → OpenReal2Sim/
    urdf_cfg = Path(cfg["scene"]["urdf"])
    urdf_path = urdf_cfg if urdf_cfg.is_absolute() else (repo_root / urdf_cfg)
    scene_yaml_path = urdf_path.parent / "isaac_export" / "scene.yaml"
    scene = {}
    if scene_yaml_path.exists():
        with open(scene_yaml_path) as f:
            scene = yaml.safe_load(f) or {}

    traj_cfg = Path(cfg["scene"]["trajectory"])
    traj_json = traj_cfg if traj_cfg.is_absolute() else (repo_root / traj_cfg)

    return TrainCfg(
        scene_yaml=scene_yaml_path if scene else None,
        urdf=Path(scene.get("urdf", urdf_path)),
        mjcf=Path(scene["mjcf"]) if scene.get("mjcf") else None,
        trajectory_npz=Path(scene["trajectory_npz"]) if scene.get("trajectory_npz")
                       else Path(str(traj_json).replace(".json", ".npz")),
        frame_zero_pose=scene.get("frame_zero_pose", {"position": [0,0,0], "quaternion": [0,0,0,1]}),
        fps=float(scene.get("fps", 30.0)),
        base_link=cfg["robot"]["base_link"],
        joint_names=cfg["robot"]["joint_names"],
        spawn_height_offset=float(cfg["robot"]["spawn_height_offset"]),
        kp=float(cfg["robot"]["pd"]["kp"]),
        kd=float(cfg["robot"]["pd"]["kd"]),
        torque_clamp=float(cfg["robot"]["pd"].get("torque_clamp", 1e3)),
        joint_armature=float(cfg["robot"].get("joint_armature", 0.0)),
        dt=float(cfg["scene"]["dt"]),
        control_dt=float(cfg["scene"]["control_dt"]),
        gravity=list(cfg["scene"]["gravity"]),
        gait=cfg["baseline_gait"],
        policy=cfg["policy"],
        reward=cfg["reward"],
        training=cfg["training"],
        episode=cfg["episode"],
        output=cfg["output"],
    )


# ───────────────────────── Trot controller ─────────────────────────

class TrotController:
    """Open-loop diagonal trot. Returns target joint positions in joint_names order."""

    def __init__(self, joint_names: List[str], gait: Dict):
        self.joint_names = joint_names
        self.cycle = float(gait["cycle_seconds"])
        self.hip_amp = float(gait["hip_amplitude_rad"])
        self.knee_amp = float(gait["knee_amplitude_rad"])
        self.hip_off = float(gait["hip_offset_rad"])
        self.knee_off = float(gait["knee_offset_rad"])
        phase = gait["phase_offsets"]
        self.phase = np.array([
            phase[name.split("_")[0]] for name in joint_names
        ], dtype=np.float64)
        self.is_knee = np.array(["knee" in n for n in joint_names], dtype=bool)
        self.is_hip = ~self.is_knee

    def __call__(self, t: float) -> np.ndarray:
        omega = 2.0 * np.pi / self.cycle
        s = np.sin(omega * t + 2.0 * np.pi * self.phase)
        out = np.zeros(len(self.joint_names), dtype=np.float64)
        out[self.is_hip] = self.hip_off + self.hip_amp * s[self.is_hip]
        # Knees pump in counter-phase to hips on the same leg, raising the foot
        # during swing and tucking during stance. Squared-sine keeps the knee
        # always bent (no hyperextension).
        out[self.is_knee] = self.knee_off - self.knee_amp * np.maximum(s[self.is_knee], 0.0)
        return out


# ───────────────────────── Quaternion utilities ─────────────────────────

def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic distance between two unit quaternions (xyzw order), in radians."""
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# ───────────────────────── MuJoCo environment ─────────────────────────

class LegoDogEnv:
    """Single-instance MuJoCo env.

    Two action modes are supported, used by the three training scripts:
      - "residual"  (default, train_residual_ppo.py): action is the additive
                    correction Delta a on top of the open-loop trot baseline.
                    Output bounded to +-residual_scale (~0.2 rad).
      - "direct"    (train_direct_ppo.py and train_bc.py cloning a direct
                    expert): action is a normalized joint command in [-1, +1]
                    that the env maps onto the per-joint range [q_lo, q_hi].
                    No trot baseline is added.
    """

    def __init__(self, cfg: TrainCfg, seed: int = 0, action_mode: str = "residual"):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        if action_mode not in ("residual", "direct"):
            raise ValueError(f"Unknown action_mode: {action_mode}")
        self.action_mode = action_mode

        # URDF is the source of truth; MuJoCo's MJCF parser does not accept
        # URDF directly (its <include> requires MJCF), so we load via MjSpec
        # which auto-detects URDF, then programmatically attach a free joint
        # to the base body (URDF has no floating-base concept) and a ground
        # plane so the robot can stand on something.
        spec = mujoco.MjSpec.from_file(str(cfg.urdf))
        base_body = spec.body(cfg.base_link)
        if base_body is None:
            raise RuntimeError(f"base_link '{cfg.base_link}' not found in URDF")
        has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in base_body.joints)
        if not has_free:
            base_body.add_freejoint(name="root")
        # Effective-inertia floor on every hinge joint. Tiny LEGO-scale leg
        # links (~3e-7 kg m^2) cause MuJoCo's integrator to blow up under any
        # nontrivial torque; armature adds a rotor inertia that costs nothing
        # in the real world (LEGO joints have negligible rotor) but stabilizes
        # the integrator at dt=5ms.
        if cfg.joint_armature > 0:
            for j in spec.joints:
                if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                    j.armature = cfg.joint_armature
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5.0, 5.0, 0.1],
            rgba=[0.7, 0.7, 0.7, 1.0],
            friction=[0.9, 0.005, 0.0001],
        )
        self.model = spec.compile()
        self.model.opt.timestep = cfg.dt
        self.model.opt.gravity[:] = cfg.gravity
        self.data = mujoco.MjData(self.model)

        self._build_index()
        self.trot = TrotController(cfg.joint_names, cfg.gait)
        self.n_joints = len(cfg.joint_names)

        # Reference trajectory (already recentered at frame 0 by the exporter).
        z = np.load(cfg.trajectory_npz)
        self.ref_times = z["times"].astype(np.float64)
        self.ref_pos = z["positions"].astype(np.float64)
        self.ref_quat = z["quaternions"].astype(np.float64)  # xyzw
        self.ref_T = len(self.ref_times)
        self.ref_duration = float(self.ref_times[-1] - self.ref_times[0])

        self.physics_steps_per_control = max(1, int(round(cfg.control_dt / cfg.dt)))
        self.max_steps = int(cfg.episode["max_steps"])

        # Action dims / bounds
        self.residual_scale = float(cfg.policy["residual_scale"])
        self.act_dim = self.n_joints
        if self.action_mode == "residual":
            # Policy output goes through tanh*residual_scale (e.g. +-0.2 rad)
            self.policy_act_scale = self.residual_scale
            self._direct_center = None
            self._direct_half_range = None
        else:  # direct
            # Policy output is in [-1, +1] (tanh-squashed); the env maps each
            # joint independently to its [q_lo, q_hi] range so the policy can
            # reach the full joint travel.
            self.policy_act_scale = 1.0
            self._direct_center = 0.5 * (self.q_lo + self.q_hi)
            self._direct_half_range = 0.5 * (self.q_hi - self.q_lo)

        # Cache observation dim — n_joints * 2 (q, qd) + 13 base (3 pos, 4 quat,
        # 3 linvel, 3 angvel) + 2 phase (sin, cos) + 7 next-frame target (3
        # delta_pos + 4 ref quat).
        self.obs_dim = self.n_joints * 2 + 13 + 2 + 7

        # Episode bookkeeping
        self.t_sim = 0.0
        self.step_count = 0
        self.cum_reward = 0.0

    def _build_index(self):
        m = self.model
        # joint and actuator IDs
        self.joint_ids: List[int] = []
        for name in self.cfg.joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint '{name}' not found in MuJoCo model "
                                   f"loaded from {self.cfg.urdf}")
            self.joint_ids.append(jid)
        self.qpos_addrs = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_addrs = np.array([m.jnt_dofadr[j] for j in self.joint_ids])

        # base body
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.cfg.base_link)
        if bid < 0:
            raise RuntimeError(f"base_link '{self.cfg.base_link}' not found")
        self.base_body_id = bid

        # joint limits
        lo = m.jnt_range[self.joint_ids, 0]
        hi = m.jnt_range[self.joint_ids, 1]
        m_margin = float(self.cfg.episode.get("joint_limit_safety_margin", 0.1))
        # If a joint has no limits (range = 0,0), use ±π.
        has_limit = m.jnt_limited[self.joint_ids].astype(bool)
        self.q_lo = np.where(has_limit, lo + m_margin, -np.pi)
        self.q_hi = np.where(has_limit, hi - m_margin, np.pi)

    # ───────── Pose / observation helpers ─────────

    def _base_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d = self.data
        pos = np.asarray(d.xpos[self.base_body_id], dtype=np.float64).copy()
        quat_wxyz = np.asarray(d.xquat[self.base_body_id], dtype=np.float64).copy()
        quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
        # cvel: 6D spatial velocity (angular, linear) of the body
        cvel = np.asarray(d.cvel[self.base_body_id], dtype=np.float64).copy()
        ang_vel = cvel[:3]
        lin_vel = cvel[3:6]
        return pos, quat_xyzw, lin_vel, ang_vel

    def _ref_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        # Clamp t to [0, ref_duration]; nearest-neighbor index into the trajectory.
        if self.ref_T == 1:
            return self.ref_pos[0], self.ref_quat[0]
        idx = int(np.clip(round(t * (self.ref_T - 1) / max(self.ref_duration, 1e-6)),
                          0, self.ref_T - 1))
        return self.ref_pos[idx], self.ref_quat[idx]

    def _get_obs(self) -> np.ndarray:
        d = self.data
        q = d.qpos[self.qpos_addrs]
        qd = d.qvel[self.qvel_addrs]
        pos, quat, linv, angv = self._base_pose()
        phase = 2.0 * np.pi * (self.t_sim / max(self.trot.cycle, 1e-6))
        ref_pos_next, ref_quat_next = self._ref_at(self.t_sim + self.cfg.control_dt)
        delta_pos = ref_pos_next - pos
        return np.concatenate([
            q, qd, pos, quat, linv, angv,
            np.array([np.sin(phase), np.cos(phase)]),
            delta_pos, ref_quat_next,
        ]).astype(np.float32)

    # ───────── Episode lifecycle ─────────

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        # Spawn at the trajectory's frame-0 position (already recentered to 0,0
        # in x,y by the exporter; z stays as-is).
        spawn_z = float(self.ref_pos[0, 2] + self.cfg.spawn_height_offset)
        free_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if free_jid >= 0:
            qadr = self.model.jnt_qposadr[free_jid]
            self.data.qpos[qadr:qadr+3] = [0.0, 0.0, spawn_z]
            # MuJoCo free-joint quat is (w, x, y, z).
            self.data.qpos[qadr+3:qadr+7] = quat_xyzw_to_wxyz(self.ref_quat[0])
        else:
            # Loaded URDF without a free-joint floating base — that's still OK
            # for learning, but the robot won't translate. Warn once.
            if not getattr(self, "_warned_floating", False):
                print("[env] WARNING: no free 'root' joint found — robot is "
                      "rigidly attached to the world. Position tracking will "
                      "be degenerate. Run scripts/exports/export_to_isaac.py "
                      "so the MJCF wrapper attaches a free joint.")
                self._warned_floating = True

        # Neutral joint pose at gait phase 0
        baseline_q = self.trot(0.0)
        self.data.qpos[self.qpos_addrs] = baseline_q
        mujoco.mj_forward(self.model, self.data)

        self.t_sim = 0.0
        self.step_count = 0
        self.cum_reward = 0.0
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64),
                         -self.policy_act_scale, self.policy_act_scale)
        if self.action_mode == "residual":
            baseline = self.trot(self.t_sim)
            target = np.clip(baseline + action, self.q_lo, self.q_hi)
        else:  # direct
            target = np.clip(
                self._direct_center + self._direct_half_range * action,
                self.q_lo, self.q_hi,
            )

        # PD control over the joints (no actuators required — write torques).
        # Clamp each joint torque to a per-joint limit so a single noisy
        # action cannot drive the integrator into NaN territory.
        kp, kd = self.cfg.kp, self.cfg.kd
        tau_lim = self.cfg.torque_clamp
        total_torque = np.zeros(self.n_joints, dtype=np.float64)
        for _ in range(self.physics_steps_per_control):
            q = self.data.qpos[self.qpos_addrs]
            qd = self.data.qvel[self.qvel_addrs]
            tau = np.clip(kp * (target - q) - kd * qd, -tau_lim, tau_lim)
            self.data.qfrc_applied[self.qvel_addrs] = tau
            mujoco.mj_step(self.model, self.data)
            total_torque += tau
        mean_tau = total_torque / self.physics_steps_per_control

        self.t_sim += self.cfg.control_dt
        self.step_count += 1

        # Reward + termination
        reward, info = self._reward(mean_tau)
        self.cum_reward += reward
        done = self._terminated()
        if self.step_count >= self.max_steps:
            done = True
            info["truncated"] = True
        if self.t_sim >= self.ref_duration:
            done = True
            info["trajectory_complete"] = True
        info["reward"] = reward
        info["t_sim"] = self.t_sim
        info["cum_reward"] = self.cum_reward
        return self._get_obs(), reward, done, info

    def _reward(self, tau: np.ndarray) -> Tuple[float, dict]:
        pos, quat, _, _ = self._base_pose()
        ref_pos, ref_quat = self._ref_at(self.t_sim)
        rw = self.cfg.reward

        sigma_p = float(rw["position_sigma"])
        sigma_q = float(rw["rotation_sigma"])
        pos_err = float(np.linalg.norm(pos - ref_pos))
        rot_err = quat_distance(quat, ref_quat)
        r_track_pos = float(np.exp(-(pos_err ** 2) / (sigma_p ** 2)))
        r_track_rot = float(np.exp(-(rot_err ** 2) / (sigma_q ** 2)))

        # Tilt = angle between body +Z and world +Z.
        body_z = self._body_z_in_world(quat)
        tilt = float(np.arccos(np.clip(body_z[2], -1.0, 1.0)))

        torque_pen = float(rw["torque_penalty"]) * float((tau ** 2).sum())
        orient_pen = float(rw["orientation_penalty"]) * (tilt ** 2)
        alive = float(rw["alive_bonus"])

        reward = r_track_pos + r_track_rot - torque_pen - orient_pen + alive
        info = {
            "r_track_pos": r_track_pos,
            "r_track_rot": r_track_rot,
            "pos_err": pos_err,
            "rot_err_deg": float(np.degrees(rot_err)),
            "tilt_deg": float(np.degrees(tilt)),
            "torque_pen": torque_pen,
            "orient_pen": orient_pen,
        }
        return reward, info

    @staticmethod
    def _body_z_in_world(quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quat_xyzw
        # Third column of the rotation matrix corresponding to the quaternion.
        return np.array([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ])

    def _terminated(self) -> bool:
        pos, quat, _, _ = self._base_pose()
        body_z = self._body_z_in_world(quat)
        tilt = float(np.arccos(np.clip(body_z[2], -1.0, 1.0)))
        z_min = float(self.cfg.episode["early_termination"]["base_z_min"])
        tilt_max = float(self.cfg.episode["early_termination"]["base_tilt_max_rad"])
        return bool(pos[2] < z_min or tilt > tilt_max)


# ───────────────────────── Policy + value networks ─────────────────────────

def _mlp(sizes: List[int], activation: str) -> nn.Sequential:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int],
                 activation: str, log_std_init: float, residual_scale: float):
        super().__init__()
        self.mean_net = _mlp([obs_dim] + list(hidden) + [act_dim], activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.act_scale = residual_scale

    def forward(self, obs: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        mean = self.mean_net(obs)
        # Squash mean into the residual range with a tanh so the trot remains
        # the dominant baseline at init.
        mean = torch.tanh(mean) * self.act_scale
        std = torch.exp(self.log_std).clamp(min=1e-4, max=1.0)
        dist = torch.distributions.Normal(mean, std)
        return dist, mean

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, mean = self.forward(obs)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: List[int], activation: str):
        super().__init__()
        self.net = _mlp([obs_dim] + list(hidden) + [1], activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ───────────────────────── PPO rollout buffer ─────────────────────────

@dataclass
class RolloutBuffer:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        T = len(self.rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values


# ───────────────────────── PPO trainer ─────────────────────────

class PPOTrainer:
    def __init__(self, env: LegoDogEnv, cfg: TrainCfg, device: str = "cpu"):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        torch.manual_seed(int(cfg.training["seed"]))

        self.policy = GaussianPolicy(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            hidden=cfg.policy["hidden_sizes"],
            activation=cfg.policy["activation"],
            log_std_init=cfg.policy["log_std_init"],
            residual_scale=env.policy_act_scale,
        ).to(self.device)
        self.value = ValueNet(
            obs_dim=env.obs_dim,
            hidden=cfg.policy["hidden_sizes"],
            activation=cfg.policy["activation"],
        ).to(self.device)
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.opt = torch.optim.Adam(params, lr=float(cfg.training["learning_rate"]))

        self.log_dir = Path(cfg.output["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def collect_rollout(self, n_steps: int, obs: np.ndarray) -> Tuple[RolloutBuffer, np.ndarray, dict]:
        obs_buf = np.zeros((n_steps, self.env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, self.env.act_dim), dtype=np.float32)
        logp_buf = np.zeros(n_steps, dtype=np.float32)
        rew_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)

        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        running_reward = 0.0
        running_length = 0

        for t in range(n_steps):
            obs_buf[t] = obs
            obs_t = self._to_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t = self.policy.act(obs_t)
                value_t = self.value(obs_t).squeeze(0)
            action = action_t.squeeze(0).cpu().numpy()
            act_buf[t] = action
            logp_buf[t] = logp_t.item()
            val_buf[t] = value_t.item()

            next_obs, reward, done, _ = self.env.step(action)
            rew_buf[t] = reward
            done_buf[t] = float(done)
            running_reward += reward
            running_length += 1
            if done:
                ep_rewards.append(running_reward)
                ep_lengths.append(running_length)
                running_reward, running_length = 0.0, 0
                next_obs = self.env.reset()
            obs = next_obs

        with torch.no_grad():
            last_value = self.value(self._to_tensor(obs).unsqueeze(0)).item()
        buf = RolloutBuffer(
            obs=obs_buf, actions=act_buf, log_probs=logp_buf,
            rewards=rew_buf, dones=done_buf, values=val_buf,
        )
        buf.compute_gae(
            last_value=last_value,
            gamma=float(self.cfg.training["discount"]),
            lam=float(self.cfg.training["gae_lambda"]),
        )
        stats = {
            "mean_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
            "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
            "episodes": len(ep_rewards),
        }
        return buf, obs, stats

    def update(self, buf: RolloutBuffer) -> dict:
        cfg = self.cfg.training
        n = len(buf.rewards)
        adv = (buf.advantages - buf.advantages.mean()) / (buf.advantages.std() + 1e-8)

        obs_t = self._to_tensor(buf.obs)
        act_t = self._to_tensor(buf.actions)
        old_logp = self._to_tensor(buf.log_probs)
        adv_t = self._to_tensor(adv)
        ret_t = self._to_tensor(buf.returns)

        clip_range = float(cfg["clip_range"])
        ent_coef = float(cfg["entropy_coef"])
        vf_coef = float(cfg["value_coef"])
        max_grad = float(cfg["max_grad_norm"])
        mbs = int(cfg["minibatch_size"])
        n_epochs = int(cfg["num_epochs"])

        total_loss = total_pi = total_vf = total_ent = 0.0
        kl_sum = 0.0
        steps = 0

        idx = np.arange(n)
        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mbs):
                mb = idx[start:start + mbs]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                dist, _ = self.policy(obs_t[mb_t])
                new_logp = dist.log_prob(act_t[mb_t]).sum(-1)
                ratio = (new_logp - old_logp[mb_t]).exp()
                surr1 = ratio * adv_t[mb_t]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t[mb_t]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_pred = self.value(obs_t[mb_t])
                vf_loss = F.mse_loss(v_pred, ret_t[mb_t])
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + vf_coef * vf_loss - ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    max_grad,
                )
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (old_logp[mb_t] - new_logp).mean().item()
                total_loss += float(loss.item())
                total_pi += float(pi_loss.item())
                total_vf += float(vf_loss.item())
                total_ent += float(ent.item())
                kl_sum += approx_kl
                steps += 1

        return {
            "loss": total_loss / max(1, steps),
            "pi_loss": total_pi / max(1, steps),
            "vf_loss": total_vf / max(1, steps),
            "entropy": total_ent / max(1, steps),
            "approx_kl": kl_sum / max(1, steps),
        }

    def train(self, total_timesteps: Optional[int] = None):
        cfg = self.cfg.training
        total_timesteps = total_timesteps or int(cfg["total_timesteps"])
        rollout_len = int(cfg["rollout_length"])
        checkpoint_every = int(self.cfg.output["checkpoint_every"])

        obs = self.env.reset()
        steps_done = 0
        next_checkpoint = checkpoint_every
        t0 = time.time()
        iter_idx = 0

        while steps_done < total_timesteps:
            buf, obs, stats = self.collect_rollout(rollout_len, obs)
            update_stats = self.update(buf)
            steps_done += rollout_len
            iter_idx += 1
            sps = steps_done / max(time.time() - t0, 1e-6)
            print(f"[ppo] iter {iter_idx:4d}  steps={steps_done:7d}  "
                  f"sps={sps:6.1f}  "
                  f"mean_ep_R={stats['mean_ep_reward']:+7.3f}  "
                  f"pi_loss={update_stats['pi_loss']:+.4f}  "
                  f"vf_loss={update_stats['vf_loss']:.4f}  "
                  f"ent={update_stats['entropy']:.3f}  "
                  f"kl={update_stats['approx_kl']:+.4f}")
            if steps_done >= next_checkpoint:
                self.save(self.log_dir / f"policy_{steps_done}.pt")
                next_checkpoint += checkpoint_every

        self.save(self.log_dir / "policy_final.pt")
        print(f"[ppo] done in {time.time() - t0:.1f}s")

    def save(self, path: Path):
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "obs_dim": self.env.obs_dim,
            "act_dim": self.env.act_dim,
            "residual_scale": self.env.residual_scale,
            "policy_act_scale": self.env.policy_act_scale,
            "action_mode": self.env.action_mode,
            "method": "ppo_" + self.env.action_mode,
        }, path)
        print(f"[ppo] checkpoint -> {path}")


# ───────────────────────── Main ─────────────────────────

def _pick_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def main():
    ap = argparse.ArgumentParser(description="Step 7.3 — Residual PPO training")
    ap.add_argument("--config", required=True,
                    help="Path to lego_dog_locomotion.yaml")
    ap.add_argument("--total_timesteps", type=int, default=None,
                    help="Override the YAML's training.total_timesteps")
    ap.add_argument("--device", default=None,
                    help="Override training.device ('cpu' / 'cuda' / 'mps')")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))
    device = _pick_device(args.device or cfg.training.get("device", "auto"))

    if cfg.scene_yaml is None:
        print(f"[ppo] WARNING: no scene.yaml found at "
              f"{cfg.urdf.parent / 'isaac_export' / 'scene.yaml'} — run "
              f"scripts/exports/export_to_isaac.py first for a fully "
              f"configured scene. Proceeding with raw URDF.")

    env = LegoDogEnv(cfg, seed=int(cfg.training["seed"]))
    print(f"[ppo] obs_dim={env.obs_dim}  act_dim={env.act_dim}  "
          f"physics_substeps={env.physics_steps_per_control}  device={device}")

    trainer = PPOTrainer(env, cfg, device=device)
    trainer.train(total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()
