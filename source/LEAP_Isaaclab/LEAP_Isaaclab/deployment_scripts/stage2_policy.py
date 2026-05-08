import torch


class HoraStage2Policy:
    def __init__(
        self,
        stage2_ckpt_path,
        stage1_cfg_path=None,
        policy_obs_dim=96,
        priv_obs_dim=9,
        proprio_hist_shape=(30, 32),
        action_space=16,
        device="cuda:0",
    ):
        from LEAP_Isaaclab.utils.hora_adaptation import HoraAdaptPolicy

        self.device = device
        self.stage2_ckpt_path = stage2_ckpt_path
        self.stage1_cfg_path = stage1_cfg_path or self._resolve_stage1_cfg_path(stage2_ckpt_path)
        metadata = self._read_metadata(stage2_ckpt_path)
        adapt_encoder_type = metadata.get("adapt_encoder_type", "tconv")
        latent_dim = metadata.get("latent_dim")
        adapt_hist_len = metadata.get("adapt_hist_len")
        if adapt_hist_len is not None:
            proprio_hist_shape = (int(adapt_hist_len), proprio_hist_shape[1])
        self.policy = HoraAdaptPolicy(
            stage1_cfg_path=self.stage1_cfg_path,
            num_actions=action_space,
            policy_obs_dim=policy_obs_dim,
            priv_obs_dim=priv_obs_dim,
            proprio_hist_shape=proprio_hist_shape,
            adapt_encoder_type=adapt_encoder_type,
            latent_dim=latent_dim,
            device=device,
        )
        self.policy.load_stage2_checkpoint(stage2_ckpt_path)
        self.policy.assert_feedforward_stage1()
        self.policy.freeze_stage1()
        self.policy.sa_mean_std.eval()
        self.policy.adapt_tconv.eval()

    @staticmethod
    def _resolve_stage1_cfg_path(stage2_ckpt_path):
        weights = torch.load(stage2_ckpt_path, map_location="cpu")
        stage1_cfg_path = weights.get("stage1_cfg_path")
        if not stage1_cfg_path:
            raise ValueError(
                "Stage2 checkpoint does not store stage1_cfg_path. "
                "Please pass stage1_cfg_path explicitly."
            )
        return stage1_cfg_path

    @staticmethod
    def _read_metadata(stage2_ckpt_path):
        weights = torch.load(stage2_ckpt_path, map_location="cpu")
        metadata = weights.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def step(self, policy_obs, proprio_hist):
        batch = self.policy.stage2_act_inference(policy_obs, proprio_hist)
        return {
            "selected_action": batch.actions,
            "student_latent": batch.student_latent,
        }

    def reset_hidden_state(self):
        return None
