#!/usr/bin/env python3

"""Run a trained Stage2 cylinder-rotation policy in simulation."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--stage1-cfg", type=str, required=True)
parser.add_argument("--stage2-checkpoint", type=str, required=True)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
parser.add_argument("--fixed-eval", action="store_true", default=False)
parser.add_argument("--adapt-encoder-type", choices=("auto", "tconv", "flatten_mlp", "gru"), default="auto")
parser.add_argument("--adapt-hist-len", type=int, default=None)
parser.add_argument("--latent-dim", type=int, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import LEAP_Isaaclab.tasks  # noqa: F401

from LEAP_Isaaclab.utils.hora_adaptation import HoraAdaptPolicy


def load_stage2_metadata() -> dict:
    weights = torch.load(args_cli.stage2_checkpoint, map_location="cpu")
    metadata = weights.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def main() -> None:
    metadata = load_stage2_metadata()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.fixed_eval:
        if hasattr(env_cfg, "enable_adr"):
            env_cfg.enable_adr = False
        for attr_name in ("randomize_priv_mass", "randomize_priv_friction", "randomize_priv_com"):
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, False)
        if hasattr(env_cfg, "object_scale_lower"):
            env_cfg.object_scale_lower = 1.0
        if hasattr(env_cfg, "object_scale_upper"):
            env_cfg.object_scale_upper = 1.0
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "object_scale_size"):
            env_cfg.events.object_scale_size.params["scale_range"] = (1.0, 1.0)

    adapt_hist_len = args_cli.adapt_hist_len or metadata.get("adapt_hist_len")
    if adapt_hist_len is not None:
        env_cfg.prop_hist_len = int(adapt_hist_len)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = os.path.abspath(os.path.join("logs", "videos", "stage2_play"))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    unwrapped = env.unwrapped
    encoder_type = args_cli.adapt_encoder_type
    if encoder_type == "auto":
        encoder_type = metadata.get("adapt_encoder_type", "tconv")
    latent_dim = args_cli.latent_dim if args_cli.latent_dim is not None else metadata.get("latent_dim")

    policy = HoraAdaptPolicy(
        stage1_cfg_path=args_cli.stage1_cfg,
        num_actions=unwrapped.single_action_space.shape[0],
        policy_obs_dim=unwrapped.single_observation_space["policy"].shape[0],
        priv_obs_dim=unwrapped.single_observation_space["critic"].shape[0],
        proprio_hist_shape=(unwrapped.cfg.prop_hist_len, 32),
        adapt_encoder_type=encoder_type,
        latent_dim=latent_dim,
        device=args_cli.device,
    )
    policy.load_stage2_checkpoint(args_cli.stage2_checkpoint)
    policy.assert_feedforward_stage1()
    policy.freeze_stage1()
    policy.eval()

    obs_dict, _ = env.reset()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            batch = policy.stage2_act_inference(
                obs_dict["policy"].to(args_cli.device),
                obs_dict["proprio_hist"].to(args_cli.device),
            )
            obs_dict, _, _, _, _ = env.step(batch.actions)
        step += 1
        if args_cli.video and step >= args_cli.video_length:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
