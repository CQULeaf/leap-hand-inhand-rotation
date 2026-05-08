from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rl_games.algos_torch import model_builder, torch_ext

from LEAP_Isaaclab.utils.rl_games_hora import register_hora_rl_games_components


class RunningMeanStd(nn.Module):
    """Small standalone running mean/std module for stage2 proprio-history normalization."""

    def __init__(self, insize, epsilon: float = 1e-5, per_channel: bool = False, norm_only: bool = False):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.per_channel = per_channel
        self.norm_only = norm_only

        if per_channel:
            if len(insize) == 3:
                self.axis = [0, 2, 3]
            elif len(insize) == 2:
                self.axis = [0, 2]
            elif len(insize) == 1:
                self.axis = [0]
            else:
                raise ValueError(f"Unsupported per-channel input shape: {insize}")
            stat_shape = insize[0]
        else:
            self.axis = [0]
            stat_shape = insize

        self.register_buffer("running_mean", torch.zeros(stat_shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(stat_shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @staticmethod
    def _update_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        total_count = count + batch_count
        new_mean = mean + delta * batch_count / total_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * count * batch_count / total_count
        new_var = m2 / total_count
        return new_mean, new_var, total_count

    def forward(self, x: torch.Tensor, unnorm: bool = False) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(self.axis)
            batch_var = x.var(self.axis, unbiased=False)
            self.running_mean, self.running_var, self.count = self._update_from_moments(
                self.running_mean, self.running_var, self.count, batch_mean, batch_var, x.size(0)
            )

        if self.per_channel:
            if len(self.insize) == 3:
                mean = self.running_mean.view(1, self.insize[0], 1, 1).expand_as(x)
                var = self.running_var.view(1, self.insize[0], 1, 1).expand_as(x)
            elif len(self.insize) == 2:
                mean = self.running_mean.view(1, self.insize[0], 1).expand_as(x)
                var = self.running_var.view(1, self.insize[0], 1).expand_as(x)
            else:
                mean = self.running_mean.view(1, self.insize[0]).expand_as(x)
                var = self.running_var.view(1, self.insize[0]).expand_as(x)
        else:
            mean = self.running_mean
            var = self.running_var

        if unnorm:
            y = torch.clamp(x, min=-5.0, max=5.0)
            return torch.sqrt(var.float() + self.epsilon) * y + mean.float()

        if self.norm_only:
            return x / torch.sqrt(var.float() + self.epsilon)

        y = (x - mean.float()) / torch.sqrt(var.float() + self.epsilon)
        return torch.clamp(y, min=-5.0, max=5.0)


def _conv1d_out_len(length: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class ProprioAdaptTConv(nn.Module):
    """Temporal adaptation module with configurable history length and output dim."""

    def __init__(self, hist_len: int = 30, latent_dim: int = 8):
        super().__init__()
        length = hist_len
        for kernel_size, stride in ((9, 2), (5, 1), (5, 1)):
            length = _conv1d_out_len(length, kernel_size=kernel_size, stride=stride)
        if length <= 0:
            raise ValueError(f"hist_len={hist_len} is too short for the temporal convolution stack.")

        self.channel_transform = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * length, latent_dim)

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        x = self.channel_transform(proprio_hist)
        x = x.permute(0, 2, 1)
        x = self.temporal_aggregation(x)
        return self.low_dim_proj(x.flatten(1))


class ProprioAdaptFlattenMLP(nn.Module):
    """Flattened-history MLP baseline for adaptation encoder ablations."""

    def __init__(self, hist_len: int = 30, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hist_len * 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        return self.net(proprio_hist)


class ProprioAdaptGRU(nn.Module):
    """GRU baseline for adaptation encoder ablations."""

    def __init__(self, hist_len: int = 30, latent_dim: int = 8):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(proprio_hist)
        return self.proj(hidden[-1])


def build_adapt_encoder(encoder_type: str, hist_len: int, latent_dim: int) -> nn.Module:
    if encoder_type == "tconv":
        return ProprioAdaptTConv(hist_len=hist_len, latent_dim=latent_dim)
    if encoder_type == "flatten_mlp":
        return ProprioAdaptFlattenMLP(hist_len=hist_len, latent_dim=latent_dim)
    if encoder_type == "gru":
        return ProprioAdaptGRU(hist_len=hist_len, latent_dim=latent_dim)
    raise ValueError(f"Unsupported adaptation encoder type: {encoder_type}")


@dataclass
class Stage2Batch:
    actions: torch.Tensor
    student_latent: torch.Tensor
    teacher_latent: torch.Tensor
    latent_loss: torch.Tensor
    action_loss: torch.Tensor


@dataclass
class Stage2InferenceBatch:
    actions: torch.Tensor
    student_latent: torch.Tensor


def load_yaml_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HoraAdaptPolicy(nn.Module):
    """Frozen stage1 actor + trainable stage2 adaptation module."""

    def __init__(
        self,
        stage1_cfg_path: str,
        num_actions: int,
        policy_obs_dim: int = 96,
        priv_obs_dim: int = 9,
        proprio_hist_shape: tuple[int, int] = (30, 32),
        adapt_encoder_type: str = "tconv",
        latent_dim: int | None = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        register_hora_rl_games_components()

        cfg = load_yaml_cfg(stage1_cfg_path)["params"]
        self.stage1_cfg_path = os.path.abspath(stage1_cfg_path)
        self.device_name = device
        self.policy_obs_dim = policy_obs_dim
        self.priv_obs_dim = priv_obs_dim
        self.adapt_encoder_type = adapt_encoder_type
        if latent_dim is None:
            network_name = cfg["network"].get("name", "")
            if network_name == "hora_stage1_direct_priv_actor_critic":
                latent_dim = priv_obs_dim
            else:
                latent_dim = int(cfg["network"].get("extrinsic_dim", 8))
        self.latent_dim = int(latent_dim)
        self.normalize_input = cfg["config"].get("normalize_input", True)
        self.normalize_value = cfg["config"].get("normalize_value", True)

        model_config = {
            "actions_num": num_actions,
            "input_shape": {"policy": (policy_obs_dim,), "critic": (priv_obs_dim,)},
            "num_seqs": 1,
            "value_size": 1,
            "normalize_input": self.normalize_input,
            "normalize_value": self.normalize_value,
        }
        builder = model_builder.ModelBuilder()
        network = builder.load(cfg)
        self.model = network.build(model_config).to(device)
        self.adapt_tconv = build_adapt_encoder(
            adapt_encoder_type,
            hist_len=proprio_hist_shape[0],
            latent_dim=self.latent_dim,
        ).to(device)
        self.sa_mean_std = RunningMeanStd(proprio_hist_shape).to(device)

    @property
    def actor(self):
        if hasattr(self.model, "a2c_network"):
            return self.model.a2c_network
        if hasattr(self.model, "network"):
            return self.model.network
        raise AttributeError("Could not find the underlying actor-critic network on the rl-games model.")

    def assert_feedforward_stage1(self) -> None:
        if hasattr(self.model, "is_rnn") and self.model.is_rnn():
            raise RuntimeError(
                "The loaded stage1 policy is recurrent. HORA stage2 expects a feedforward stage1 actor. "
                "Please retrain stage1 after removing the GRU block from rl_games_ppo_cfg.yaml."
            )

    def freeze_stage1(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def load_stage1_checkpoint(self, checkpoint_path: str) -> None:
        weights = torch_ext.load_checkpoint(checkpoint_path)
        self.model.load_state_dict(weights["model"], strict=True)
        if self.normalize_input and "running_mean_std" in weights and hasattr(self.model, "running_mean_std"):
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def load_stage2_checkpoint(self, checkpoint_path: str) -> dict:
        weights = torch.load(checkpoint_path, map_location=self.device_name)
        self.model.load_state_dict(weights["model"], strict=False)
        self.adapt_tconv.load_state_dict(weights["adapt_tconv"], strict=True)
        if self.normalize_input and "running_mean_std" in weights and hasattr(self.model, "running_mean_std"):
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        if "sa_mean_std" in weights:
            self.sa_mean_std.load_state_dict(weights["sa_mean_std"])
        return weights

    def save_stage2_checkpoint(self, checkpoint_path: str, metadata: dict | None = None) -> None:
        payload_metadata = {
            "adapt_encoder_type": self.adapt_encoder_type,
            "adapt_hist_len": self.sa_mean_std.insize[0],
            "latent_dim": self.latent_dim,
        }
        if metadata:
            payload_metadata.update(metadata)
        payload = {
            "model": self.model.state_dict(),
            "adapt_tconv": self.adapt_tconv.state_dict(),
            "stage1_cfg_path": self.stage1_cfg_path,
            "sa_mean_std": self.sa_mean_std.state_dict(),
            "metadata": payload_metadata,
        }
        if self.normalize_input and hasattr(self.model, "running_mean_std"):
            payload["running_mean_std"] = self.model.running_mean_std.state_dict()
        torch.save(payload, checkpoint_path)

    def _normalize_obs_groups(self, policy_obs: torch.Tensor, priv_obs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if priv_obs is None:
            priv_obs = torch.zeros(
                (policy_obs.shape[0], self.priv_obs_dim),
                dtype=policy_obs.dtype,
                device=policy_obs.device,
            )

        if self.normalize_input and hasattr(self.model, "running_mean_std"):
            normalized = self.model.running_mean_std({"policy": policy_obs, "critic": priv_obs})
            return normalized["policy"], normalized["critic"]
        return policy_obs, priv_obs

    def teacher_latent(self, priv_obs: torch.Tensor) -> torch.Tensor:
        dummy_policy = torch.zeros(
            (priv_obs.shape[0], self.policy_obs_dim),
            dtype=priv_obs.dtype,
            device=priv_obs.device,
        )
        _, normalized_priv = self._normalize_obs_groups(dummy_policy, priv_obs)
        if getattr(self.actor, "priv_direct_passthrough", False):
            return normalized_priv
        return torch.tanh(self.actor.priv_encoder(normalized_priv))

    def student_latent(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        proprio_hist = self.sa_mean_std(proprio_hist)
        return torch.tanh(self.adapt_tconv(proprio_hist))

    def actor_mu_from_latent(self, policy_obs: torch.Tensor, latent: torch.Tensor, priv_obs: torch.Tensor | None = None) -> torch.Tensor:
        normalized_policy, _ = self._normalize_obs_groups(policy_obs, priv_obs)
        merged_obs = torch.cat([normalized_policy, latent], dim=-1)
        actor_features = self.actor.actor_mlp(merged_obs)
        return self.actor.mu(actor_features)

    def stage2_step(self, policy_obs: torch.Tensor, priv_obs: torch.Tensor, proprio_hist: torch.Tensor) -> Stage2Batch:
        student_latent = self.student_latent(proprio_hist)
        teacher_latent = self.teacher_latent(priv_obs)
        teacher_mu = self.actor_mu_from_latent(policy_obs, teacher_latent, priv_obs)
        student_mu = self.actor_mu_from_latent(policy_obs, student_latent, priv_obs)
        latent_loss = F.mse_loss(student_latent, teacher_latent.detach())
        action_loss = F.mse_loss(torch.clamp(student_mu, -1.0, 1.0), torch.clamp(teacher_mu, -1.0, 1.0).detach())
        return Stage2Batch(
            actions=torch.clamp(student_mu, -1.0, 1.0),
            student_latent=student_latent,
            teacher_latent=teacher_latent,
            latent_loss=latent_loss,
            action_loss=action_loss,
        )

    @torch.inference_mode()
    def stage2_act_inference(self, policy_obs: torch.Tensor, proprio_hist: torch.Tensor) -> Stage2InferenceBatch:
        student_latent = self.student_latent(proprio_hist)
        mu = self.actor_mu_from_latent(policy_obs, student_latent)
        return Stage2InferenceBatch(actions=torch.clamp(mu, -1.0, 1.0), student_latent=student_latent)
