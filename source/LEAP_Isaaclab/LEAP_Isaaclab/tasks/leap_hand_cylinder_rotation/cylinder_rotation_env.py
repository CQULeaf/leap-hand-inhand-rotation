# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Modified for HORA-style Cylinder Rotation Task
# --------------------------------------------------------

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

if TYPE_CHECKING:
    from LEAP_Isaaclab.tasks.leap_hand_cylinder_rotation.leap_hand_env_cfg import LeapHandCylinderRotationEnvCfg

from LEAP_Isaaclab.tasks.leap_hand_cylinder_rotation.grasp_init import sample_initial_poses_from_cfg
from LEAP_Isaaclab.utils import adr_utils, obs_utils
from LEAP_Isaaclab.utils.adr import LeapHandADR


class CylinderRotationEnv(DirectRLEnv):
    cfg: LeapHandCylinderRotationEnvCfg

    def __init__(self, cfg: LeapHandCylinderRotationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()

        self.finger_bodies = []
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] += 0.01

        self.override_default_joint_pos = torch.tensor(
            [[0.000, 0.500, 0.000, 0.000,
              -0.750, 1.300, 0.000, 0.750,
              1.750, 1.500, 1.750, 1.750,
              0.000, 1.000, 0.000, 0.000]],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.init_pose_buf = self.override_default_joint_pos.clone()
        self.scale_conditioned_pose_delta = torch.tensor(
            [[0.00, -0.40, -0.25, -0.15,
              0.00, -0.35, -0.20, -0.10,
              0.00, -0.35, -0.20, -0.10,
              0.08, -0.25, -0.15, -0.10]],
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_rot[:, 0] = 1.0
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.rot_axis_buf = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.priv_info_buf = torch.zeros((self.num_envs, self.cfg.state_space), dtype=torch.float, device=self.device)
        self.object_scale_xyz_buf = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_scale_buf = torch.ones((self.num_envs, 1), dtype=torch.float, device=self.device) * self.cfg.privileged_scale_value
        self.object_mass_buf = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.object_friction_buf = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.object_com_buf = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # HORA-style policy observation: 3 timesteps * (16 joint_pos + 16 target_pos) = 96 dims.
        self.obs_hist_buf = torch.zeros((self.num_envs, self.cfg.hist_len, 32), device=self.device, dtype=torch.float)
        # HORA stage2 adaptation history: 30 timesteps * (16 joint_pos + 16 target_pos).
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.cfg.prop_hist_len, 32), device=self.device, dtype=torch.float)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.randomized_episode_lengths = torch.randint(
            int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)),
            self.max_episode_length + 1,
            (self.num_envs,),
            dtype=torch.int32,
            device=self.device,
        )

        if self.cfg.enable_adr:
            self.leap_adr = LeapHandADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
            self.step_since_last_dr_change = 0
            self.leap_adr.set_num_increments(self.cfg.starting_adr_increments)
            adr_utils.init_adr_obs_act_noise(self)

            self.obs_hist_buf = torch.zeros(
                self.num_envs,
                self.cfg.observation_space // self.cfg.hist_len,
                self.cfg.hist_len + self.cfg.obs_max_latency,
                device=cfg.sim.device,
                dtype=torch.float,
            )
            self.obs_latency = torch.empty((self.num_envs, self.cfg.obs_per_timestep), device=self.cfg.sim.device)
            self.act_latency = torch.empty((self.num_envs, self.cfg.action_space), device=self.cfg.sim.device)
            self.act_hist_buf = torch.zeros(
                self.num_envs,
                self.cfg.action_space,
                self.cfg.act_max_latency + 1,
                device=self.cfg.sim.device,
                dtype=torch.float,
            )

            print("starting ranges: ")
            print(self.leap_adr.print_params())

        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        self.default_object_masses = self.object.root_physx_view.get_masses().clone()
        self.default_object_materials = self.object.root_physx_view.get_material_properties().clone()
        self.default_object_coms = self.object.root_physx_view.get_coms().clone()
        self._cache_object_scales_from_usd()
        self.object_mass_buf[:] = self.default_object_masses[:, :1].to(self.device)
        self.object_friction_buf[:] = self.default_object_materials[:, 0:1, 0].to(self.device)
        self.last_out_of_reach = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.last_time_out = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.sim_real_indices()

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.actions = actions.clone()

        if self.cfg.enable_adr:
            hand_noise = self.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise")
            if hand_noise > 0:
                noise = torch.randn_like(actions) * hand_noise
                self.actions = actions + noise
            self.actions = obs_utils.create_action_latency(self, self.actions)

        self.actions = torch.clamp(self.actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        if self.cfg.action_type == "relative":
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                targets,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        elif self.cfg.action_type == "absolute":
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                self.cur_targets[:, self.actuated_dof_indices],
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        else:
            raise ValueError(f"Unsupported action type: {self.cfg.action_type}. Must be relative or absolute.")

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        if self.cfg.enable_adr:
            adr_utils.apply_object_wrench(self, self.object, "object")

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        joint_pos_for_obs = self.hand_dof_pos
        if self.cfg.joint_pos_obs_noise > 0:
            joint_pos_noise = sample_uniform(
                -self.cfg.joint_pos_obs_noise,
                self.cfg.joint_pos_obs_noise,
                self.hand_dof_pos.shape,
                device=self.device,
            )
            joint_pos_for_obs = saturate(
                self.hand_dof_pos + joint_pos_noise,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
            )

        joint_pos_normalized = unscale(joint_pos_for_obs, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        current_frame = torch.cat([joint_pos_normalized, self.cur_targets], dim=-1)

        self.obs_hist_buf[:, :-1] = self.obs_hist_buf[:, 1:].clone()
        self.obs_hist_buf[:, -1] = current_frame
        self.proprio_hist_buf[:, :-1] = self.proprio_hist_buf[:, 1:].clone()
        self.proprio_hist_buf[:, -1] = current_frame
        self._update_privileged_info()
        return {
            "policy": self.obs_hist_buf.reshape(self.num_envs, -1),
            "critic": self.priv_info_buf,
            "proprio_hist": self.proprio_hist_buf,
        }

    def _get_rewards(self) -> torch.Tensor:
        pose_diff_penalty = ((self.hand_dof_pos[:, self.actuated_dof_indices] - self.init_pose_buf[:, self.actuated_dof_indices]) ** 2).sum(-1)
        torque_penalty = (self.hand.data.computed_torque ** 2).sum(-1)
        work_penalty = ((self.hand.data.computed_torque * self.hand_dof_vel).sum(-1)) ** 2
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        quat_delta = quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        object_angvel_est = quat_to_axis_angle(quat_delta) / (self.cfg.sim.dt * self.cfg.decimation)
        projected_spin = (object_angvel_est * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clamp(projected_spin, self.cfg.rot_reward_clip_min, self.cfg.rot_reward_clip_max)
        object_linvel_est = (self.object_pos - self.object_pos_prev) / (self.cfg.sim.dt * self.cfg.decimation)

        total_reward, rotation_term, pose_penalty, linvel_penalty, torque_penalty_scaled, work_penalty_scaled, action_penalty, goal_center_penalty = compute_cylinder_rotation_rewards(
            object_linvel_est,
            rotate_reward,
            self.cfg.rot_reward_scale,
            self.cfg.pose_diff_penalty_scale,
            self.cfg.linvel_penalty_scale,
            self.cfg.torque_penalty_scale,
            self.cfg.work_penalty_scale,
            pose_diff_penalty,
            torque_penalty,
            work_penalty,
            self.actions,
            self.cfg.action_penalty_scale,
            goal_dist,
            self.cfg.fall_dist,
            self.cfg.goal_center_penalty_scale,
        )
        fall_rate = (goal_dist >= self.cfg.fall_dist).float()

        self.extras["log"]["rotation_reward"] = rotate_reward.mean()
        self.extras["log"]["projected_spin"] = projected_spin.mean()
        self.extras["log"]["rotation_term"] = rotation_term.mean()
        self.extras["log"]["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["log"]["pose_penalty"] = pose_penalty.mean()
        self.extras["log"]["torque_info"] = torque_penalty.mean()
        self.extras["log"]["torque_penalty"] = torque_penalty_scaled.mean()
        self.extras["log"]["work_penalty"] = work_penalty_scaled.mean()
        self.extras["log"]["object_linvel"] = torch.norm(object_linvel_est, p=1, dim=-1).mean()
        self.extras["log"]["linvel_penalty"] = linvel_penalty.mean()
        self.extras["log"]["action_penalty"] = action_penalty.mean()
        self.extras["log"]["goal_center_penalty"] = goal_center_penalty.mean()
        self.extras["log"]["goal_dist"] = goal_dist.mean()
        self.extras["log"]["fall_rate"] = fall_rate.mean()
        # self.extras["log"]["roll"] = object_angvel_est[:, 0].mean()
        # self.extras["log"]["pitch"] = object_angvel_est[:, 1].mean()
        self.extras["log"]["yaw"] = object_angvel_est[:, 2].mean()
        self.extras["log"]["priv_scale_mean"] = self.object_scale_buf.mean()
        self.extras["log"]["priv_scale_std"] = self.object_scale_buf.std()
        self.extras["log"]["priv_mass_mean"] = self.object_mass_buf.mean()
        self.extras["log"]["priv_friction_mean"] = self.object_friction_buf.mean()
        self.extras["log"]["priv_com_norm"] = torch.norm(self.object_com_buf, p=2, dim=-1).mean()
        self.extras["log"]["joint_pos_obs_noise"] = torch.tensor(self.cfg.joint_pos_obs_noise, device=self.device)
        self.extras["log"]["avg_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).mean()
        self.extras["log"]["min_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).min()
        self.extras["log"]["max_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).max()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1
        self.last_out_of_reach[:] = out_of_reach
        self.last_time_out[:] = time_out
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        finished_steps = self.episode_length_buf[env_ids].float()
        if finished_steps.numel() > 0:
            finished_mask = finished_steps > 0
            if torch.any(finished_mask):
                finished_lengths_s = finished_steps[finished_mask] * self.cfg.sim.dt * self.cfg.decimation
                self.extras["log"]["realized_episode_length_s"] = finished_lengths_s.mean()
                self.extras["log"]["realized_episode_length_min_s"] = finished_lengths_s.min()
                self.extras["log"]["realized_episode_length_max_s"] = finished_lengths_s.max()
                self.extras["log"]["reset_out_of_reach_rate"] = self.last_out_of_reach[env_ids][finished_mask].float().mean()
                self.extras["log"]["reset_timeout_rate"] = self.last_time_out[env_ids][finished_mask].float().mean()

        super()._reset_idx(env_ids)
        self._randomize_hora_privileged_info(env_ids)

        self.randomized_episode_lengths[env_ids] = torch.randint(
            int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)),
            self.max_episode_length + 1,
            (len(env_ids),),
            dtype=torch.int32,
            device=self.device,
        )

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        dof_pos = self._get_scale_conditioned_grasp_pose(env_ids)
        dof_vel = self.hand.data.default_joint_vel[env_ids]
        scale_delta = self.object_scale_buf[env_ids, 0] - 1.0

        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        object_default_state[:, 1] += self.cfg.init_object_y_offset_gain * scale_delta
        object_default_state[:, 2] += self.cfg.init_object_z_offset_gain * scale_delta

        if self.cfg.init_object_xy_noise > 0:
            object_xy_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            object_default_state[:, 0:2] += object_xy_noise * self.cfg.init_object_xy_noise

        if self.cfg.init_object_yaw_noise > 0:
            object_yaw_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
            object_yaw_quat = quat_from_angle_axis(object_yaw_noise * self.cfg.init_object_yaw_noise, self.z_unit_tensor[env_ids])
            object_default_state[:, 3:7] = quat_mul(object_yaw_quat, object_default_state[:, 3:7])

        if self.cfg.init_joint_noise > 0:
            joint_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
            dof_pos += joint_pos_noise * self.cfg.init_joint_noise

        if self.cfg.enable_adr:
            x_width = self.leap_adr.get_custom_param_value("object_spawn", "x_width_spawn")
            y_width = self.leap_adr.get_custom_param_value("object_spawn", "y_width_spawn")
            x_rot = self.leap_adr.get_custom_param_value("object_spawn", "x_rotation")
            y_rot = self.leap_adr.get_custom_param_value("object_spawn", "y_rotation")
            z_rot = self.leap_adr.get_custom_param_value("object_spawn", "z_rotation")

            if x_width > 0 or y_width > 0:
                pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
                object_default_state[:, 0] += pos_noise[:, 0] * x_width
                object_default_state[:, 1] += pos_noise[:, 1] * y_width

            if x_rot > 0:
                x_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                x_rot_quat = quat_from_angle_axis(x_rot_noise * x_rot, self.x_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(x_rot_quat, object_default_state[:, 3:7])

            if y_rot > 0:
                y_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                y_rot_quat = quat_from_angle_axis(y_rot_noise * y_rot, self.y_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(y_rot_quat, object_default_state[:, 3:7])

            if z_rot > 0:
                z_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                z_rot_quat = quat_from_angle_axis(z_rot_noise * z_rot, self.z_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(z_rot_quat, object_default_state[:, 3:7])

            joint_pos_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_pos_noise")
            joint_vel_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_vel_noise")

            if joint_pos_noise_width > 0:
                joint_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
                dof_pos += joint_pos_noise * joint_pos_noise_width

            if joint_vel_noise_width > 0:
                joint_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
                dof_vel += joint_vel_noise * joint_vel_noise_width

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.init_pose_buf[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        if self.cfg.enable_adr and len(env_ids) > 0:
            adr_utils.update_adr_obs_act_noise(self, env_ids)

            obs_latency_resets = self.leap_adr.get_custom_param_value("obs_latency", "latency") - torch.randint(
                0, self.cfg.obs_latency_rand + 1, (len(env_ids), 1), device=self.cfg.sim.device
            )
            obs_latency_resets = torch.maximum(obs_latency_resets, torch.tensor(0))
            self.obs_latency[env_ids, :] = obs_latency_resets.expand(-1, self.cfg.obs_per_timestep)

            act_latency_resets = self.leap_adr.get_custom_param_value("action_latency", "hand_latency") - torch.randint(
                0, self.cfg.act_latency_rand + 1, (len(env_ids), 1), device=self.cfg.sim.device
            )
            act_latency_resets = torch.maximum(act_latency_resets, torch.tensor(0))
            self.act_latency[env_ids, :] = act_latency_resets.expand(-1, self.cfg.action_space)

            self.extras["log"]["num_adr_increases"] = self.leap_adr.num_increments()
            self.step_since_last_dr_change += 1

            self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device)
            self.apply_wrench = torch.where(
                torch.rand(self.num_envs, device=self.device) <= self.cfg.wrench_prob_per_rollout,
                True,
                False,
            )

        self._compute_intermediate_values()
        self._update_privileged_info(env_ids)
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]

        joint_pos_norm = unscale(
            self.hand_dof_pos[env_ids],
            self.hand_dof_lower_limits[env_ids],
            self.hand_dof_upper_limits[env_ids],
        )
        init_frame = torch.cat([joint_pos_norm, self.cur_targets[env_ids]], dim=-1)

        for t in range(self.cfg.hist_len):
            self.obs_hist_buf[env_ids, t] = init_frame
        for t in range(self.cfg.prop_hist_len):
            self.proprio_hist_buf[env_ids, t] = init_frame

    def _compute_intermediate_values(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def _randomize_hora_privileged_info(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.int64)

        if self.cfg.randomize_priv_mass:
            masses = self.default_object_masses.clone()
            sampled_mass = sample_uniform(
                self.cfg.priv_mass_lower,
                self.cfg.priv_mass_upper,
                (len(env_ids), 1),
                device=self.device,
            ).cpu()
            masses[env_ids_cpu, :1] = sampled_mass
            self.object.root_physx_view.set_masses(masses, env_ids_cpu)
            self.object_mass_buf[env_ids, 0] = sampled_mass[:, 0].to(self.device)
        else:
            self.object_mass_buf[env_ids, 0] = self.default_object_masses[env_ids_cpu, 0].to(self.device)

        if self.cfg.randomize_priv_friction:
            materials = self.default_object_materials.clone()
            sampled_friction = sample_uniform(
                self.cfg.priv_friction_lower,
                self.cfg.priv_friction_upper,
                (len(env_ids), 1),
                device=self.device,
            ).cpu()
            materials[env_ids_cpu, :, 0] = sampled_friction
            materials[env_ids_cpu, :, 1] = sampled_friction
            materials[env_ids_cpu, :, 2] = 0.0
            self.object.root_physx_view.set_material_properties(materials, env_ids_cpu)
            self.object_friction_buf[env_ids, 0] = sampled_friction[:, 0].to(self.device)
        else:
            self.object_friction_buf[env_ids, 0] = self.default_object_materials[env_ids_cpu, 0, 0].to(self.device)

        if self.cfg.randomize_priv_com:
            coms = self.default_object_coms.clone()
            sampled_com = sample_uniform(
                self.cfg.priv_com_lower,
                self.cfg.priv_com_upper,
                (len(env_ids), 3),
                device=self.device,
            ).cpu()
            if coms.ndim == 3:
                coms[env_ids_cpu, :, :3] = sampled_com.unsqueeze(1)
            elif coms.ndim == 2:
                coms[env_ids_cpu, :3] = sampled_com
            else:
                raise RuntimeError(f"Unsupported CoM tensor shape: {tuple(coms.shape)}")
            self.object.root_physx_view.set_coms(coms, env_ids_cpu)
            self.object_com_buf[env_ids] = sampled_com.to(self.device)
        else:
            if self.default_object_coms.ndim == 3:
                self.object_com_buf[env_ids] = self.default_object_coms[env_ids_cpu, 0, :3].to(self.device)
            elif self.default_object_coms.ndim == 2:
                self.object_com_buf[env_ids] = self.default_object_coms[env_ids_cpu, :3].to(self.device)
            else:
                raise RuntimeError(f"Unsupported default CoM tensor shape: {tuple(self.default_object_coms.shape)}")

        self.object_scale_buf[env_ids, 0] = self.object_scale_xyz_buf[env_ids, 0]

    def _update_privileged_info(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.priv_info_buf[env_ids] = 0.0
        offset = 0
        include_object_pos = (
            getattr(self.cfg, "include_object_pos", True)
            and getattr(self.cfg, "include_object_pos_in_privileged_info", True)
        )
        if include_object_pos:
            self.priv_info_buf[env_ids, offset:offset + 3] = self.object_pos[env_ids]
            offset += 3
        if getattr(self.cfg, "include_scale", True):
            self.priv_info_buf[env_ids, offset:offset + 1] = self.object_scale_buf[env_ids]
            offset += 1
        if getattr(self.cfg, "include_mass", True):
            self.priv_info_buf[env_ids, offset:offset + 1] = self.object_mass_buf[env_ids]
            offset += 1
        if getattr(self.cfg, "include_friction", True):
            self.priv_info_buf[env_ids, offset:offset + 1] = self.object_friction_buf[env_ids]
            offset += 1
        if getattr(self.cfg, "include_com", True):
            self.priv_info_buf[env_ids, offset:offset + 3] = self.object_com_buf[env_ids]
            offset += 3
        if offset != self.cfg.state_space:
            raise RuntimeError(f"Privileged information dimension mismatch: wrote {offset}, expected {self.cfg.state_space}.")

    def sim_real_indices(self):
        sim2real_idx_16, _ = self.hand.find_joints(self.cfg.actuated_joint_names, preserve_order=True)
        sim2real_idx_16 = torch.tensor(sim2real_idx_16) - min(sim2real_idx_16)
        real2sim_idx_16 = torch.empty_like(sim2real_idx_16)
        real2sim_idx_16[sim2real_idx_16] = torch.arange(len(sim2real_idx_16))

        print(f"sim2real_indices: {sim2real_idx_16}")
        print(f"real2sim_indices: {real2sim_idx_16}")

    def _cache_object_scales_from_usd(self) -> None:
        stage = get_current_stage()
        prim_paths = sim_utils.find_matching_prim_paths(self.object.cfg.prim_path)
        scale_values = []
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            scale_attr = prim.GetAttribute("xformOp:scale")
            if scale_attr.IsValid():
                scale = scale_attr.Get()
                scale_values.append((float(scale[0]), float(scale[1]), float(scale[2])))
            else:
                scale_values.append((1.0, 1.0, 1.0))
        scale_tensor = torch.tensor(scale_values, dtype=torch.float, device=self.device)
        self.object_scale_xyz_buf[:] = scale_tensor
        self.object_scale_buf[:, 0] = scale_tensor[:, 0]

    def _get_scale_conditioned_grasp_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        dof_pos = sample_initial_poses_from_cfg(
            cfg=self.cfg,
            object_scales=self.object_scale_buf[env_ids, 0],
            base_pose=self.override_default_joint_pos[env_ids],
            pose_delta=self.scale_conditioned_pose_delta[env_ids],
        )
        return saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids])


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def compute_cylinder_rotation_rewards(
    object_linvel: torch.Tensor,
    rotate_reward: torch.Tensor,
    rotate_reward_scale: float,
    pose_diff_penalty_scale: float,
    linvel_penalty_scale: float,
    torque_penalty_scale: float,
    work_penalty_scale: float,
    pose_diff_penalty: torch.Tensor,
    torque_penalty: torch.Tensor,
    work_penalty: torch.Tensor,
    actions: torch.Tensor,
    action_penalty_scale: float,
    goal_dist: torch.Tensor,
    fall_dist: float,
    goal_center_penalty_scale: float,
):
    rotation_term = rotate_reward_scale * rotate_reward
    pose_penalty = pose_diff_penalty * pose_diff_penalty_scale
    linvel_penalty = torch.norm(object_linvel, p=1, dim=-1) * linvel_penalty_scale
    torque_penalty_scaled = torque_penalty * torque_penalty_scale
    work_penalty_scaled = work_penalty * work_penalty_scale
    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale
    normalized_goal_dist = torch.clamp(goal_dist / fall_dist, min=0.0, max=1.0)
    goal_center_penalty = normalized_goal_dist * normalized_goal_dist * goal_center_penalty_scale
    total_reward = (
        rotation_term
        + pose_penalty
        + linvel_penalty
        + torque_penalty_scaled
        + work_penalty_scaled
        + action_penalty
        + goal_center_penalty
    )
    return total_reward, rotation_term, pose_penalty, linvel_penalty, torque_penalty_scaled, work_penalty_scaled, action_penalty, goal_center_penalty


@torch.jit.script
def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:4], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 0:1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:4] / sin_half_angles_over_angles
