#!/usr/bin/env python3

"""Train the HORA-style Stage2 adaptation module for cylinder rotation."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--stage1-cfg", type=str, required=True)
parser.add_argument("--stage1-checkpoint", type=str, required=True)
parser.add_argument("--stage2-checkpoint", type=str, default="")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=3072)
parser.add_argument("--max-steps", type=int, default=5_000_000)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--action-loss-weight", type=float, default=0.0)
parser.add_argument("--adapt-encoder-type", choices=("tconv", "flatten_mlp", "gru"), default="tconv")
parser.add_argument("--adapt-hist-len", type=int, default=None)
parser.add_argument("--latent-dim", type=int, default=None)
parser.add_argument("--save-every", type=int, default=500_000)
parser.add_argument("--log-every", type=int, default=5_000)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import LEAP_Isaaclab.tasks  # noqa: F401

from LEAP_Isaaclab.utils.hora_adaptation import HoraAdaptPolicy


def build_log_dir() -> str:
    run_name = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.abspath(os.path.join("logs", "hora_stage2", "leap_hand_cylinder_rotation", run_name))


def save_checkpoint(policy: HoraAdaptPolicy, path: str, step: int, loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    policy.save_stage2_checkpoint(
        path,
        metadata={
            "global_step": step,
            "loss": loss,
            "stage1_checkpoint": os.path.abspath(args_cli.stage1_checkpoint),
            "action_loss_weight": args_cli.action_loss_weight,
            "learning_rate": args_cli.learning_rate,
        },
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.adapt_hist_len is not None:
        env_cfg.prop_hist_len = int(args_cli.adapt_hist_len)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    policy = HoraAdaptPolicy(
        stage1_cfg_path=args_cli.stage1_cfg,
        num_actions=unwrapped.single_action_space.shape[0],
        policy_obs_dim=unwrapped.single_observation_space["policy"].shape[0],
        priv_obs_dim=unwrapped.single_observation_space["critic"].shape[0],
        proprio_hist_shape=(unwrapped.cfg.prop_hist_len, 32),
        adapt_encoder_type=args_cli.adapt_encoder_type,
        latent_dim=args_cli.latent_dim,
        device=args_cli.device,
    )
    policy.load_stage1_checkpoint(args_cli.stage1_checkpoint)
    if args_cli.stage2_checkpoint:
        policy.load_stage2_checkpoint(args_cli.stage2_checkpoint)
    policy.assert_feedforward_stage1()
    policy.freeze_stage1()
    policy.model.eval()
    policy.adapt_tconv.train()
    policy.sa_mean_std.train()

    optimizer = torch.optim.Adam(policy.adapt_tconv.parameters(), lr=args_cli.learning_rate)
    log_dir = build_log_dir()
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
    nn_dir = os.path.join(log_dir, "nn")
    model_last = os.path.join(nn_dir, "model_last.pt")
    model_best = os.path.join(nn_dir, "model_best.pt")

    obs_dict, _ = env.reset()
    global_step = 0
    next_log = args_cli.log_every
    next_save = args_cli.save_every
    best_loss = float("inf")
    running_loss = 0.0
    running_latent = 0.0
    running_action = 0.0
    running_batches = 0

    while simulation_app.is_running() and global_step < args_cli.max_steps:
        batch = policy.stage2_step(
            obs_dict["policy"].to(args_cli.device),
            obs_dict["critic"].to(args_cli.device),
            obs_dict["proprio_hist"].to(args_cli.device),
        )
        loss = batch.latent_loss + args_cli.action_loss_weight * batch.action_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.adapt_tconv.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.inference_mode():
            obs_dict, _, _, _, _ = env.step(batch.actions.detach())

        global_step += int(unwrapped.num_envs)
        loss_value = float(loss.detach().item())
        latent_value = float(batch.latent_loss.detach().item())
        action_value = float(batch.action_loss.detach().item())
        running_loss += loss_value
        running_latent += latent_value
        running_action += action_value
        running_batches += 1

        if global_step >= next_log:
            mean_loss = running_loss / max(running_batches, 1)
            mean_latent = running_latent / max(running_batches, 1)
            mean_action = running_action / max(running_batches, 1)
            writer.add_scalar("loss/total", mean_loss, global_step)
            writer.add_scalar("loss/latent", mean_latent, global_step)
            writer.add_scalar("loss/action", mean_action, global_step)
            print(
                f"[stage2] step={global_step} loss={mean_loss:.6f} "
                f"latent={mean_latent:.6f} action={mean_action:.6f}",
                flush=True,
            )
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_checkpoint(policy, model_best, global_step, best_loss)
            running_loss = running_latent = running_action = 0.0
            running_batches = 0
            next_log += args_cli.log_every

        if global_step >= next_save:
            save_checkpoint(policy, model_last, global_step, loss_value)
            next_save += args_cli.save_every

    save_checkpoint(policy, model_last, global_step, loss_value if "loss_value" in locals() else float("nan"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
