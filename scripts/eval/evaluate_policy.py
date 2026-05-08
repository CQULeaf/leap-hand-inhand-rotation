#!/usr/bin/env python3

"""Evaluate stage1/stage2 cylinder-rotation policies and export metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate stage1/stage2 LEAP cylinder-rotation policies.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument(
    "--policy-type",
    type=str,
    required=True,
    choices=("stage1", "stage2"),
    help="Which policy family to evaluate.",
)
parser.add_argument("--stage1-checkpoint", type=str, default="", help="Path to the stage1 checkpoint.")
parser.add_argument("--stage2-checkpoint", type=str, default="", help="Path to the stage2 checkpoint.")
parser.add_argument("--stage1-cfg", type=str, required=True, help="Path to the stage1 rl_games yaml config.")
parser.add_argument(
    "--adapt-encoder-type",
    type=str,
    default="auto",
    choices=("auto", "tconv", "flatten_mlp", "gru"),
    help="Stage2 encoder type. 'auto' reads checkpoint metadata when available.",
)
parser.add_argument("--adapt-hist-len", type=int, default=None, help="Override stage2 adaptation history length.")
parser.add_argument("--latent-dim", type=int, default=None, help="Override stage2 latent dimension.")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--num-episodes", type=int, default=256, help="Number of episodes to evaluate.")
parser.add_argument(
    "--eval-preset",
    type=str,
    default="id",
    choices=("fixed", "id", "ood"),
    help="Evaluation preset: fixed single-condition, in-distribution randomization, or broader OOD randomization.",
)
parser.add_argument(
    "--fixed-object-scale",
    type=float,
    default=float("nan"),
    help="Optional fixed object scale override. Applied after the eval preset.",
)
parser.add_argument(
    "--fixed-object-mass",
    type=float,
    default=float("nan"),
    help="Optional fixed object mass override in kg. Applied after the eval preset.",
)
parser.add_argument(
    "--fixed-object-friction",
    type=float,
    default=float("nan"),
    help="Optional fixed object friction override. Applied after the eval preset.",
)
parser.add_argument(
    "--fixed-object-com",
    type=float,
    default=float("nan"),
    help="Optional fixed CoM offset applied identically to x/y/z. Applied after the eval preset.",
)
parser.add_argument(
    "--success-rotation-threshold",
    type=float,
    default=2.0 * math.pi,
    help="Minimum net rotation angle (rad) required for a timeout episode to count as success.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="",
    help="Optional output directory. Defaults to logs/evaluation/leap_hand_cylinder_rotation/<timestamp>.",
)
parser.add_argument("--run-name", type=str, default="", help="Optional run name used when constructing the output path.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.utils.math import quat_conjugate, quat_mul
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import LEAP_Isaaclab.tasks  # noqa: F401

from LEAP_Isaaclab.tasks.leap_hand_cylinder_rotation.cylinder_rotation_env import quat_to_axis_angle
from LEAP_Isaaclab.utils.hora_adaptation import HoraAdaptPolicy


def apply_eval_preset(env_cfg, preset: str) -> None:
    """Apply evaluation-specific randomization settings in a stable, low-overhead way."""

    # Always disable ADR during quantitative evaluation to keep the task definition stable.
    if hasattr(env_cfg, "enable_adr"):
        env_cfg.enable_adr = False

    if preset == "fixed":
        for attr_name in ("randomize_priv_mass", "randomize_priv_friction", "randomize_priv_com"):
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, False)
        if hasattr(env_cfg, "object_scale_lower"):
            env_cfg.object_scale_lower = 1.0
        if hasattr(env_cfg, "object_scale_upper"):
            env_cfg.object_scale_upper = 1.0
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "object_scale_size"):
            env_cfg.events.object_scale_size.params["scale_range"] = (1.0, 1.0)
        return

    # ID uses the current task defaults. We only make sure randomization is enabled.
    if preset == "id":
        for attr_name, value in (
            ("randomize_priv_mass", True),
            ("randomize_priv_friction", True),
            ("randomize_priv_com", True),
        ):
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, value)
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "object_scale_size"):
            env_cfg.events.object_scale_size.params["scale_range"] = (
                getattr(env_cfg, "object_scale_lower", 0.85),
                getattr(env_cfg, "object_scale_upper", 1.15),
            )
        return

    # OOD broadens the randomization range modestly while keeping the task solvable and grasp assumptions reasonable.
    if preset == "ood":
        for attr_name, value in (
            ("randomize_priv_mass", True),
            ("randomize_priv_friction", True),
            ("randomize_priv_com", True),
        ):
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, value)

        if hasattr(env_cfg, "object_scale_lower"):
            env_cfg.object_scale_lower = 0.80
        if hasattr(env_cfg, "object_scale_upper"):
            env_cfg.object_scale_upper = 1.20
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "object_scale_size"):
            env_cfg.events.object_scale_size.params["scale_range"] = (0.80, 1.20)

        if hasattr(env_cfg, "priv_mass_lower"):
            env_cfg.priv_mass_lower = 0.02
        if hasattr(env_cfg, "priv_mass_upper"):
            env_cfg.priv_mass_upper = 0.24
        if hasattr(env_cfg, "priv_friction_lower"):
            env_cfg.priv_friction_lower = 0.2
        if hasattr(env_cfg, "priv_friction_upper"):
            env_cfg.priv_friction_upper = 3.5
        if hasattr(env_cfg, "priv_com_lower"):
            env_cfg.priv_com_lower = -0.015
        if hasattr(env_cfg, "priv_com_upper"):
            env_cfg.priv_com_upper = 0.015
        return

    raise ValueError(f"Unsupported eval preset: {preset}")


def apply_fixed_overrides(env_cfg) -> None:
    if args_cli.fixed_object_scale == args_cli.fixed_object_scale:
        env_cfg.object_scale_lower = args_cli.fixed_object_scale
        env_cfg.object_scale_upper = args_cli.fixed_object_scale
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "object_scale_size"):
            env_cfg.events.object_scale_size.params["scale_range"] = (
                args_cli.fixed_object_scale,
                args_cli.fixed_object_scale,
            )

    if args_cli.fixed_object_mass == args_cli.fixed_object_mass:
        env_cfg.randomize_priv_mass = True
        env_cfg.priv_mass_lower = args_cli.fixed_object_mass
        env_cfg.priv_mass_upper = args_cli.fixed_object_mass

    if args_cli.fixed_object_friction == args_cli.fixed_object_friction:
        env_cfg.randomize_priv_friction = True
        env_cfg.priv_friction_lower = args_cli.fixed_object_friction
        env_cfg.priv_friction_upper = args_cli.fixed_object_friction

    if args_cli.fixed_object_com == args_cli.fixed_object_com:
        env_cfg.randomize_priv_com = True
        env_cfg.priv_com_lower = args_cli.fixed_object_com
        env_cfg.priv_com_upper = args_cli.fixed_object_com


def load_stage2_metadata() -> dict:
    if args_cli.policy_type != "stage2" or not args_cli.stage2_checkpoint:
        return {}
    try:
        weights = torch.load(args_cli.stage2_checkpoint, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Could not read stage2 metadata from {args_cli.stage2_checkpoint}: {exc}")
        return {}
    metadata = weights.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def build_output_dir() -> str:
    if args_cli.output_dir:
        return os.path.abspath(args_cli.output_dir)
    run_name = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.abspath(os.path.join("logs", "evaluation", "leap_hand_cylinder_rotation", run_name))


def load_policy(env, device: str, stage2_metadata: dict) -> HoraAdaptPolicy:
    encoder_type = args_cli.adapt_encoder_type
    if encoder_type == "auto":
        encoder_type = stage2_metadata.get("adapt_encoder_type", "tconv")
    latent_dim = args_cli.latent_dim
    if latent_dim is None:
        latent_dim = stage2_metadata.get("latent_dim")
    policy = HoraAdaptPolicy(
        stage1_cfg_path=args_cli.stage1_cfg,
        num_actions=env.unwrapped.single_action_space.shape[0],
        policy_obs_dim=env.unwrapped.single_observation_space["policy"].shape[0],
        priv_obs_dim=env.unwrapped.single_observation_space["critic"].shape[0],
        proprio_hist_shape=(env.unwrapped.cfg.prop_hist_len, 32),
        adapt_encoder_type=encoder_type,
        latent_dim=latent_dim,
        device=device,
    )
    if args_cli.policy_type == "stage1":
        if not args_cli.stage1_checkpoint:
            raise ValueError("--stage1-checkpoint is required when --policy-type stage1.")
        policy.load_stage1_checkpoint(args_cli.stage1_checkpoint)
    else:
        if not args_cli.stage2_checkpoint:
            raise ValueError("--stage2-checkpoint is required when --policy-type stage2.")
        policy.load_stage2_checkpoint(args_cli.stage2_checkpoint)
    policy.assert_feedforward_stage1()
    policy.freeze_stage1()
    policy.sa_mean_std.eval()
    policy.adapt_tconv.eval()
    return policy


@torch.inference_mode()
def compute_actions(policy: HoraAdaptPolicy, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    if args_cli.policy_type == "stage1":
        priv_obs = obs_dict["critic"].to(policy.device_name)
        policy_obs = obs_dict["policy"].to(policy.device_name)
        teacher_latent = policy.teacher_latent(priv_obs)
        mu = policy.actor_mu_from_latent(policy_obs, teacher_latent, priv_obs)
        return torch.clamp(mu, -1.0, 1.0)
    batch = policy.stage2_act_inference(
        obs_dict["policy"].to(policy.device_name),
        obs_dict["proprio_hist"].to(policy.device_name),
    )
    return batch.actions


def mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def build_summary(rows: list[dict[str, object]], output_dir: str) -> dict[str, object]:
    successes = [1.0 if bool(row["success"]) else 0.0 for row in rows]
    falls = [1.0 if row["termination_type"] == "fall" else 0.0 for row in rows]
    timeouts = [1.0 if row["termination_type"] == "timeout" else 0.0 for row in rows]
    mean_ttf, std_ttf = mean_and_std([float(row["ttf"]) for row in rows])
    mean_rotr, std_rotr = mean_and_std([float(row["mean_rotation_reward"]) for row in rows])
    mean_rot, std_rot = mean_and_std([float(row["net_rotation_angle"]) for row in rows])
    mean_turns, std_turns = mean_and_std([float(row["net_rotation_turns"]) for row in rows])
    mean_spin, std_spin = mean_and_std([float(row["mean_projected_angular_velocity"]) for row in rows])
    mean_linvel, std_linvel = mean_and_std([float(row["mean_object_linear_velocity"]) for row in rows])
    mean_torque_l1, std_torque_l1 = mean_and_std([float(row["mean_command_torque_l1"]) for row in rows])
    mean_action, std_action = mean_and_std([float(row["mean_action_l2"]) for row in rows])
    latent_values = [float(row["mean_latent_mse"]) for row in rows if row["mean_latent_mse"] == row["mean_latent_mse"]]
    action_gap_values = [float(row["mean_teacher_student_action_l2"]) for row in rows if row["mean_teacher_student_action_l2"] == row["mean_teacher_student_action_l2"]]
    mean_latent_mse, std_latent_mse = mean_and_std(latent_values)
    mean_action_gap, std_action_gap = mean_and_std(action_gap_values)

    summary = {
        "policy_type": args_cli.policy_type,
        "task": args_cli.task,
        "eval_preset": args_cli.eval_preset,
        "num_episodes": len(rows),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "success_rotation_threshold": args_cli.success_rotation_threshold,
        "fixed_object_scale": None if args_cli.fixed_object_scale != args_cli.fixed_object_scale else args_cli.fixed_object_scale,
        "fixed_object_mass": None if args_cli.fixed_object_mass != args_cli.fixed_object_mass else args_cli.fixed_object_mass,
        "fixed_object_friction": None if args_cli.fixed_object_friction != args_cli.fixed_object_friction else args_cli.fixed_object_friction,
        "fixed_object_com": None if args_cli.fixed_object_com != args_cli.fixed_object_com else args_cli.fixed_object_com,
        "stage1_cfg": os.path.abspath(args_cli.stage1_cfg),
        "stage1_checkpoint": os.path.abspath(args_cli.stage1_checkpoint) if args_cli.stage1_checkpoint else "",
        "stage2_checkpoint": os.path.abspath(args_cli.stage2_checkpoint) if args_cli.stage2_checkpoint else "",
        "adapt_encoder_type": args_cli.adapt_encoder_type,
        "adapt_hist_len": args_cli.adapt_hist_len,
        "latent_dim": args_cli.latent_dim,
        "success_rate": sum(successes) / max(len(successes), 1),
        "fall_rate": sum(falls) / max(len(falls), 1),
        "timeout_rate": sum(timeouts) / max(len(timeouts), 1),
        "ttf_mean": mean_ttf,
        "ttf_std": std_ttf,
        "rotation_reward_mean": mean_rotr,
        "rotation_reward_std": std_rotr,
        "net_rotation_angle_mean": mean_rot,
        "net_rotation_angle_std": std_rot,
        "net_rotation_turns_mean": mean_turns,
        "net_rotation_turns_std": std_turns,
        "mean_projected_angular_velocity_mean": mean_spin,
        "mean_projected_angular_velocity_std": std_spin,
        "mean_object_linear_velocity_mean": mean_linvel,
        "mean_object_linear_velocity_std": std_linvel,
        "mean_command_torque_l1_mean": mean_torque_l1,
        "mean_command_torque_l1_std": std_torque_l1,
        "mean_action_l2_mean": mean_action,
        "mean_action_l2_std": std_action,
        "mean_latent_mse_mean": mean_latent_mse,
        "mean_latent_mse_std": std_latent_mse,
        "mean_teacher_student_action_l2_mean": mean_action_gap,
        "mean_teacher_student_action_l2_std": std_action_gap,
        "output_dir": output_dir,
    }
    return summary


def export_results(rows: list[dict[str, object]], summary: dict[str, object], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "episodes.csv")
    json_path = os.path.join(output_dir, "summary.json")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_summary(summary: dict[str, object]) -> None:
    print("[eval] completed")
    print(f"  policy_type: {summary['policy_type']}")
    print(f"  eval_preset: {summary['eval_preset']}")
    print(f"  num_episodes: {summary['num_episodes']}")
    if summary["fixed_object_scale"] is not None:
        print(f"  fixed_object_scale: {summary['fixed_object_scale']:.4f}")
    if summary["fixed_object_mass"] is not None:
        print(f"  fixed_object_mass: {summary['fixed_object_mass']:.4f}")
    if summary["fixed_object_friction"] is not None:
        print(f"  fixed_object_friction: {summary['fixed_object_friction']:.4f}")
    if summary["fixed_object_com"] is not None:
        print(f"  fixed_object_com: {summary['fixed_object_com']:.4f}")
    print(f"  success_rate: {summary['success_rate']:.4f}")
    print(f"  fall_rate: {summary['fall_rate']:.4f}")
    print(f"  timeout_rate: {summary['timeout_rate']:.4f}")
    print(f"  ttf_mean: {summary['ttf_mean']:.4f}")
    print(f"  rotation_reward_mean: {summary['rotation_reward_mean']:.4f}")
    print(f"  net_rotation_turns_mean: {summary['net_rotation_turns_mean']:.4f}")
    print(f"  mean_projected_angular_velocity_mean: {summary['mean_projected_angular_velocity_mean']:.4f}")
    print(f"  mean_object_linear_velocity_mean: {summary['mean_object_linear_velocity_mean']:.4f}")
    print(f"  mean_command_torque_l1_mean: {summary['mean_command_torque_l1_mean']:.4f}")
    if summary["mean_latent_mse_mean"] == summary["mean_latent_mse_mean"]:
        print(f"  mean_latent_mse_mean: {summary['mean_latent_mse_mean']:.6f}")
    if summary["mean_teacher_student_action_l2_mean"] == summary["mean_teacher_student_action_l2_mean"]:
        print(f"  mean_teacher_student_action_l2_mean: {summary['mean_teacher_student_action_l2_mean']:.6f}")
    print(f"  output_dir: {summary['output_dir']}")


def main():
    stage2_metadata = load_stage2_metadata()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    apply_eval_preset(env_cfg, args_cli.eval_preset)
    apply_fixed_overrides(env_cfg)
    adapt_hist_len = args_cli.adapt_hist_len or stage2_metadata.get("adapt_hist_len")
    if adapt_hist_len is not None:
        env_cfg.prop_hist_len = int(adapt_hist_len)
    env = gym.make(args_cli.task, cfg=env_cfg)

    policy = load_policy(env, args_cli.device, stage2_metadata)
    output_dir = build_output_dir()

    obs_dict, _ = env.reset()
    unwrapped = env.unwrapped
    dt = float(unwrapped.step_dt)
    num_envs = int(unwrapped.num_envs)

    ep_reward = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_steps = torch.zeros(num_envs, dtype=torch.int64, device=unwrapped.device)
    ep_net_rotation = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_rotation_reward_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_projected_spin_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_object_linvel_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_torque_l1_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_action_l2_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_latent_mse_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)
    ep_action_gap_l2_sum = torch.zeros(num_envs, dtype=torch.float64, device=unwrapped.device)

    ep_scale = unwrapped.object_scale_buf[:, 0].detach().clone().to(dtype=torch.float64)
    ep_mass = unwrapped.object_mass_buf[:, 0].detach().clone().to(dtype=torch.float64)
    ep_friction = unwrapped.object_friction_buf[:, 0].detach().clone().to(dtype=torch.float64)
    ep_com = unwrapped.object_com_buf.detach().clone().to(dtype=torch.float64)

    rows: list[dict[str, object]] = []
    total_steps = 0

    while simulation_app.is_running() and len(rows) < args_cli.num_episodes:
        with torch.inference_mode():
            current_obs_dict = {key: value.clone() for key, value in obs_dict.items()}
            actions = compute_actions(policy, obs_dict)
            obs_dict, reward, terminated, truncated, _ = env.step(actions)

        total_steps += 1
        reward = reward.to(unwrapped.device, dtype=torch.float64)
        done = (terminated | truncated).to(unwrapped.device)

        quat_delta = quat_mul(unwrapped.object_rot, quat_conjugate(unwrapped.object_rot_prev))
        projected_step_angle = (quat_to_axis_angle(quat_delta) * unwrapped.rot_axis_buf).sum(-1).to(dtype=torch.float64)
        projected_spin = projected_step_angle / dt
        rotation_reward = torch.clamp(
            projected_spin,
            min=float(unwrapped.cfg.rot_reward_clip_min),
            max=float(unwrapped.cfg.rot_reward_clip_max),
        )
        object_linvel = torch.norm((unwrapped.object_pos - unwrapped.object_pos_prev) / dt, p=1, dim=-1).to(dtype=torch.float64)
        command_torque_l1 = torch.norm(unwrapped.hand.data.computed_torque, p=1, dim=-1).to(dtype=torch.float64)
        action_l2 = torch.norm(actions, p=2, dim=-1).to(dtype=torch.float64)
        latent_mse = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=unwrapped.device)
        action_gap_l2 = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=unwrapped.device)
        if args_cli.policy_type == "stage2":
            priv_obs = current_obs_dict["critic"].to(policy.device_name)
            policy_obs = current_obs_dict["policy"].to(policy.device_name)
            teacher_latent = policy.teacher_latent(priv_obs)
            teacher_actions = torch.clamp(policy.actor_mu_from_latent(policy_obs, teacher_latent, priv_obs), -1.0, 1.0)
            student_latent = policy.student_latent(current_obs_dict["proprio_hist"].to(policy.device_name))
            latent_mse = ((student_latent - teacher_latent) ** 2).mean(dim=-1).to(dtype=torch.float64)
            action_gap_l2 = torch.norm(actions - teacher_actions, p=2, dim=-1).to(dtype=torch.float64)

        ep_reward += reward
        ep_steps += 1
        ep_net_rotation += projected_step_angle
        ep_rotation_reward_sum += rotation_reward
        ep_projected_spin_sum += projected_spin
        ep_object_linvel_sum += object_linvel
        ep_torque_l1_sum += command_torque_l1
        ep_action_l2_sum += action_l2
        ep_latent_mse_sum = torch.where(torch.isnan(latent_mse), ep_latent_mse_sum, ep_latent_mse_sum + latent_mse)
        ep_action_gap_l2_sum = torch.where(torch.isnan(action_gap_l2), ep_action_gap_l2_sum, ep_action_gap_l2_sum + action_gap_l2)

        done_indices = done.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            for env_id in done_indices.tolist():
                steps = max(int(ep_steps[env_id].item()), 1)
                ttf = steps * dt
                net_rotation_angle = float(ep_net_rotation[env_id].item())
                termination_type = "unknown"
                if bool(unwrapped.last_out_of_reach[env_id].item()):
                    termination_type = "fall"
                elif bool(unwrapped.last_time_out[env_id].item()):
                    termination_type = "timeout"

                success = (
                    termination_type == "timeout"
                    and net_rotation_angle >= args_cli.success_rotation_threshold
                )

                rows.append(
                    {
                        "episode_index": len(rows),
                        "policy_type": args_cli.policy_type,
                        "eval_preset": args_cli.eval_preset,
                        "seed": args_cli.seed,
                        "env_id": env_id,
                        "steps": steps,
                        "ttf": ttf,
                        "episode_reward": float(ep_reward[env_id].item()),
                        "mean_rotation_reward": float((ep_rotation_reward_sum[env_id] / steps).item()),
                        "net_rotation_angle": net_rotation_angle,
                        "net_rotation_turns": net_rotation_angle / (2.0 * math.pi),
                        "mean_projected_angular_velocity": float((ep_projected_spin_sum[env_id] / steps).item()),
                        "mean_object_linear_velocity": float((ep_object_linvel_sum[env_id] / steps).item()),
                        "mean_command_torque_l1": float((ep_torque_l1_sum[env_id] / steps).item()),
                        "mean_action_l2": float((ep_action_l2_sum[env_id] / steps).item()),
                        "mean_latent_mse": (
                            float((ep_latent_mse_sum[env_id] / steps).item()) if args_cli.policy_type == "stage2" else float("nan")
                        ),
                        "mean_teacher_student_action_l2": (
                            float((ep_action_gap_l2_sum[env_id] / steps).item()) if args_cli.policy_type == "stage2" else float("nan")
                        ),
                        "success": bool(success),
                        "termination_type": termination_type,
                        "object_scale": float(ep_scale[env_id].item()),
                        "object_mass": float(ep_mass[env_id].item()),
                        "object_friction": float(ep_friction[env_id].item()),
                        "object_com_x": float(ep_com[env_id, 0].item()),
                        "object_com_y": float(ep_com[env_id, 1].item()),
                        "object_com_z": float(ep_com[env_id, 2].item()),
                        "requested_fixed_object_scale": (
                            float(args_cli.fixed_object_scale) if args_cli.fixed_object_scale == args_cli.fixed_object_scale else float("nan")
                        ),
                        "requested_fixed_object_mass": (
                            float(args_cli.fixed_object_mass) if args_cli.fixed_object_mass == args_cli.fixed_object_mass else float("nan")
                        ),
                        "requested_fixed_object_friction": (
                            float(args_cli.fixed_object_friction) if args_cli.fixed_object_friction == args_cli.fixed_object_friction else float("nan")
                        ),
                        "requested_fixed_object_com": (
                            float(args_cli.fixed_object_com) if args_cli.fixed_object_com == args_cli.fixed_object_com else float("nan")
                        ),
                    }
                )

            ep_reward[done_indices] = 0.0
            ep_steps[done_indices] = 0
            ep_net_rotation[done_indices] = 0.0
            ep_rotation_reward_sum[done_indices] = 0.0
            ep_projected_spin_sum[done_indices] = 0.0
            ep_object_linvel_sum[done_indices] = 0.0
            ep_torque_l1_sum[done_indices] = 0.0
            ep_action_l2_sum[done_indices] = 0.0
            ep_latent_mse_sum[done_indices] = 0.0
            ep_action_gap_l2_sum[done_indices] = 0.0

            ep_scale[done_indices] = unwrapped.object_scale_buf[done_indices, 0].detach().clone().to(dtype=torch.float64)
            ep_mass[done_indices] = unwrapped.object_mass_buf[done_indices, 0].detach().clone().to(dtype=torch.float64)
            ep_friction[done_indices] = unwrapped.object_friction_buf[done_indices, 0].detach().clone().to(dtype=torch.float64)
            ep_com[done_indices] = unwrapped.object_com_buf[done_indices].detach().clone().to(dtype=torch.float64)

    env.close()

    if not rows:
        raise RuntimeError("No finished episodes were collected. Try increasing num_episodes or checking the environment configuration.")

    rows = rows[: args_cli.num_episodes]
    summary = build_summary(rows, output_dir)
    export_results(rows, summary, output_dir)
    print_summary(summary)


if __name__ == "__main__":
    main()
    simulation_app.close()
