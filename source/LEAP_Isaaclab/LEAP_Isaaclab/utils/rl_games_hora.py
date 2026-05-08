from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch import model_builder, network_builder


_REGISTERED = False


class HoraStage1A2CBuilder(network_builder.A2CBuilder):
    """RL-Games network that encodes 9D privileged info into an 8D extrinsic embedding."""

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs.get("input_shape")
            if not isinstance(input_shape, dict):
                raise TypeError("HoraStage1A2CBuilder expects dict observations with 'policy' and 'critic' groups.")
            if "policy" not in input_shape or "critic" not in input_shape:
                raise KeyError("HoraStage1A2CBuilder requires 'policy' and 'critic' observation groups.")

            self.policy_input_shape = input_shape["policy"]
            self.priv_input_shape = input_shape["critic"]
            self.extrinsic_dim = int(params.get("extrinsic_dim", 8))
            self.priv_mlp_units = params.get("priv_mlp", {}).get("units", [32, self.extrinsic_dim])
            if self.priv_mlp_units[-1] != self.extrinsic_dim:
                raise ValueError("The last priv_mlp unit must match extrinsic_dim.")

            base_kwargs = dict(kwargs)
            base_kwargs["input_shape"] = (self.policy_input_shape[0] + self.extrinsic_dim,)
            super().__init__(params, **base_kwargs)

            self.priv_encoder = self._build_mlp(
                input_size=self.priv_input_shape[0],
                units=self.priv_mlp_units,
                activation=self.activation,
                dense_func=nn.Linear,
                norm_only_first_layer=False,
                norm_func_name=None,
                d2rl=False,
            )
            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.priv_encoder.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            policy_obs = obs["policy"]
            priv_obs = obs["critic"]

            extrinsic = torch.tanh(self.priv_encoder(priv_obs))
            merged_obs = torch.cat([policy_obs, extrinsic], dim=-1)

            input_dict = dict(obs_dict)
            input_dict["obs"] = merged_obs
            return super().forward(input_dict)

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)


class HoraStage1DirectPrivA2CBuilder(network_builder.A2CBuilder):
    """RL-Games network that feeds privileged info directly to the actor."""

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs.get("input_shape")
            if not isinstance(input_shape, dict):
                raise TypeError("HoraStage1DirectPrivA2CBuilder expects dict observations with 'policy' and 'critic' groups.")
            if "policy" not in input_shape or "critic" not in input_shape:
                raise KeyError("HoraStage1DirectPrivA2CBuilder requires 'policy' and 'critic' observation groups.")

            self.policy_input_shape = input_shape["policy"]
            self.priv_input_shape = input_shape["critic"]
            self.extrinsic_dim = self.priv_input_shape[0]
            self.priv_direct_passthrough = True

            base_kwargs = dict(kwargs)
            base_kwargs["input_shape"] = (self.policy_input_shape[0] + self.extrinsic_dim,)
            super().__init__(params, **base_kwargs)
            self.priv_encoder = nn.Identity()

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            policy_obs = obs["policy"]
            priv_obs = obs["critic"]

            merged_obs = torch.cat([policy_obs, priv_obs], dim=-1)

            input_dict = dict(obs_dict)
            input_dict["obs"] = merged_obs
            return super().forward(input_dict)

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)


def register_hora_rl_games_components() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    model_builder.register_network("hora_stage1_actor_critic", HoraStage1A2CBuilder)
    model_builder.register_network("hora_stage1_direct_priv_actor_critic", HoraStage1DirectPrivA2CBuilder)
    _REGISTERED = True
