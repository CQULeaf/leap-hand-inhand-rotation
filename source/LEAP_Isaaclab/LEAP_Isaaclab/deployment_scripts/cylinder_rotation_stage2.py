#!/usr/bin/env python3

import argparse
import importlib.util
import logging
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stage2_policy import HoraStage2Policy
from utils.leap_hand_utils.dynamixel_client import DynamixelClient


def _load_grasp_init_module():
    grasp_init_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "tasks"
        / "leap_hand_cylinder_rotation"
        / "grasp_init.py"
    )
    spec = importlib.util.spec_from_file_location("_leap_grasp_init", grasp_init_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load grasp_init module from {grasp_init_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GRASP_INIT = _load_grasp_init_module()
CANONICAL_BASE_POSE = _GRASP_INIT.CANONICAL_BASE_POSE
CANONICAL_SCALE_POSE_DELTA = _GRASP_INIT.CANONICAL_SCALE_POSE_DELTA
DEFAULT_GRASP_CACHE_PREFIX = _GRASP_INIT.DEFAULT_GRASP_CACHE_PREFIX
sample_initial_poses_from_cfg = _GRASP_INIT.sample_initial_poses_from_cfg


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class LEAPHandCylinderRotationStage2Controller:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.actions_num = 16
        self.hist_len = 3
        self.prop_hist_len = 30
        self.hz = args.hz
        self.control_dt = 1.0 / self.hz
        self.action_scale = 1.0 / 24.0
        self.motors = list(range(16))
        self._loop_hz_warning_emitted = False

        self.kP = args.kp
        self.kI = 0.0
        self.kD = args.kd
        self.curr_lim = args.curr_lim

        self.construct_sim_to_real_transformation()
        self.get_dof_limits()
        self.base_init_pose = CANONICAL_BASE_POSE.to(device=self.device, dtype=torch.float32)
        self.grasp_pose_delta = CANONICAL_SCALE_POSE_DELTA.to(device=self.device, dtype=torch.float32)
        self.warmup_close_delta = torch.tensor(
            [[0.00, 0.40, 0.25, 0.15,
              0.00, 0.35, 0.20, 0.10,
              0.00, 0.35, 0.20, 0.10,
              -0.08, 0.25, 0.15, 0.10]],
            device=self.device,
            dtype=torch.float32,
        )
        self.init_pose = self.fetch_grasp_state()

        self.policy = HoraStage2Policy(
            stage2_ckpt_path=args.stage2_checkpoint,
            stage1_cfg_path=args.stage1_cfg,
            policy_obs_dim=96,
            priv_obs_dim=9,
            proprio_hist_shape=(self.prop_hist_len, 32),
            action_space=self.actions_num,
            device=self.device,
        )
        self.policy.reset_hidden_state()

        self.dxl_client = self._connect_dxl_client(args.port, args.baudrate)
        self._configure_motors()

    def _connect_dxl_client(self, port, baudrate):
        candidate_ports = [port] if port != "auto" else [
            "/dev/ttyUSB0",
            "/dev/ttyUSB1",
            "/dev/ttyUSB2",
            "/dev/ttyUSB3",
            "COM5",
        ]
        last_error = None
        for candidate in candidate_ports:
            try:
                client = DynamixelClient(self.motors, candidate, baudrate)
                client.connect()
                print(f"[INFO] Connected to LEAP hand on {candidate}")
                return client
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[WARN] Failed to connect on {candidate}: {exc}")
        raise RuntimeError(f"Failed to connect to LEAP hand on any candidate port. Last error: {last_error}")

    def _configure_motors(self):
        if self.args.dry_run:
            print("[INFO] Dry run enabled; skipping motor configuration writes.")
            return
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = torch.tensor([4, 0, 8, 12, 6, 2, 10, 14, 7, 3, 11, 15, 1, 5, 9, 13], device=self.device)
        self.real_to_sim_indices = torch.tensor([1, 12, 5, 9, 0, 13, 4, 8, 2, 14, 6, 10, 3, 15, 7, 11], device=self.device)

    def real_to_sim(self, values):
        if values.dim() == 1:
            return values[self.real_to_sim_indices]
        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if values.dim() == 1:
            return values[self.sim_to_real_indices]
        return values[:, self.sim_to_real_indices]

    def get_leap_hand_joint_limits(self):
        upper_limits = [2.2300, 2.0940, 2.2300, 2.2300, 1.0470, 2.4430, 1.0470, 1.0470, 1.8850,
                        1.9000, 1.8850, 1.8850, 2.0420, 1.8800, 2.0420, 2.0420]
        lower_limits = [-0.3140, -0.3490, -0.3140, -0.3140, -1.0470, -0.4700, -1.0470, -1.0470,
                        -0.5060, -1.2000, -0.5060, -0.5060, -0.3660, -1.3400, -0.3660, -0.3660]
        return lower_limits, upper_limits

    def get_dof_limits(self):
        lower, upper = self.get_leap_hand_joint_limits()
        self.leap_dof_lower = torch.tensor(lower, device=self.device, dtype=torch.float32)
        self.leap_dof_upper = torch.tensor(upper, device=self.device, dtype=torch.float32)

    def _build_warmup_grasp_cfg(self):
        warmup_mode = getattr(self.args, "warmup_mode", "auto")
        grasp_cache_dir = getattr(self.args, "grasp_cache_dir", "")
        use_grasp_cache = warmup_mode == "cache" or (warmup_mode == "auto" and bool(grasp_cache_dir))
        return SimpleNamespace(
            use_grasp_cache=use_grasp_cache,
            grasp_cache_dir=grasp_cache_dir,
            grasp_cache_prefix=getattr(self.args, "grasp_cache_prefix", DEFAULT_GRASP_CACHE_PREFIX),
            grasp_scale_buckets=(float(getattr(self.args, "object_scale", 1.0)),) if use_grasp_cache else (),
        )

    def _select_warmup_base_pose(self):
        warmup_cfg = self._build_warmup_grasp_cfg()
        return sample_initial_poses_from_cfg(
            cfg=warmup_cfg,
            object_scales=float(getattr(self.args, "object_scale", 1.0)),
            base_pose=self.base_init_pose,
            pose_delta=self.grasp_pose_delta,
        ).to(device=self.device, dtype=torch.float32)

    def fetch_grasp_state(self):
        grasp_pose = self._select_warmup_base_pose()
        grasp_pose = grasp_pose + self.args.warmup_close_scale * self.warmup_close_delta
        return torch.clamp(grasp_pose, self.leap_dof_lower, self.leap_dof_upper)

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        return sim_min, sim_max

    @staticmethod
    def LEAPsim_to_LEAPhand(joints):
        return joints + 3.14159

    @staticmethod
    def LEAPhand_to_LEAPsim(joints):
        return joints - 3.14

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)
        return joints

    def command_joint_position(self, desired_pose):
        desired_pose = self.LEAPsim_to_LEAPhand(desired_pose)
        desired_pose = self.sim_to_real(desired_pose)
        desired_pose = desired_pose.detach().cpu().numpy().astype(float).flatten()
        if not self.args.dry_run:
            self.dxl_client.write_desired_pos(self.motors, desired_pose)

    def poll_joint_position(self):
        joint_position = self.dxl_client.read_pos(
            retries=self.args.read_retries,
            retry_interval=self.args.read_retry_interval,
        )
        joint_position = torch.from_numpy(joint_position).to(device=self.device, dtype=torch.float32)
        joint_position = self.LEAPhand_to_sim_ones(joint_position)
        joint_position = self.real_to_sim(joint_position)
        joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower
        return {"position": joint_position}

    def _compute_warmup_target(self, start_pose, step_idx, warmup_steps):
        safe_steps = max(int(warmup_steps), 1)
        alpha = min(1.0, float(step_idx + 1) / float(safe_steps))
        target_pose = torch.lerp(start_pose, self.init_pose, alpha)
        return torch.clamp(target_pose, self.leap_dof_lower, self.leap_dof_upper)

    def _warmup(self):
        print("[INFO] Commanding initial grasp pose...")
        print(f"[INFO] Warmup mode: {self.args.warmup_mode} | object scale: {self.args.object_scale:.3f}")
        print(f"[INFO] Warmup close scale: {self.args.warmup_close_scale:.3f}")
        warmup_steps = int(self.hz * self.args.warmup_seconds)
        start_pose = self.poll_joint_position()["position"].unsqueeze(0)
        for step_idx in range(max(warmup_steps, 1)):
            target_pose = self._compute_warmup_target(start_pose, step_idx, warmup_steps)
            self.command_joint_position(target_pose)
            _ = self.poll_joint_position()
            time.sleep(self.control_dt)
        print("[INFO] Initial pose reached.")

    def _build_initial_buffers(self, obses):
        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        prev_target = obses.clone()
        unscaled_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
        frame = torch.cat([unscaled_pos, prev_target], dim=-1).float()

        obs_hist_buf = torch.zeros((1, self.hist_len, 32), device=self.device, dtype=torch.float32)
        proprio_hist_buf = torch.zeros((1, self.prop_hist_len, 32), device=self.device, dtype=torch.float32)
        obs_hist_buf[:] = frame
        proprio_hist_buf[:] = frame
        return prev_target, obs_hist_buf, proprio_hist_buf

    def _maybe_warn_low_loop_hz(self, loop_hz):
        low_loop_hz_threshold = 0.9 * self.hz
        if loop_hz < low_loop_hz_threshold:
            if not self._loop_hz_warning_emitted:
                logging.warning(
                    "Deployment loop is running below target frequency: %.1f Hz observed vs %.1f Hz target. "
                    "If behavior remains stable, consider reducing read retries or retry interval.",
                    loop_hz,
                    self.hz,
                )
                self._loop_hz_warning_emitted = True
            return

        self._loop_hz_warning_emitted = False

    def deploy(self):
        self._warmup()
        robot_state = self.poll_joint_position()
        obses = robot_state["position"]
        prev_target, obs_hist_buf, proprio_hist_buf = self._build_initial_buffers(obses)

        print("[INFO] Starting stage2 cylinder rotation deployment loop.")
        step_idx = 0
        try:
            while True:
                step_idx += 1
                start_time = time.time()

                policy_obs = obs_hist_buf.reshape(1, -1)
                out = self.policy.step(policy_obs, proprio_hist_buf)
                action = out["selected_action"].squeeze(0)
                student_latent = out["student_latent"].squeeze(0)

                action = torch.clamp(action, -1.0, 1.0)
                target = prev_target + self.action_scale * action
                target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
                prev_target = target.clone()

                self.command_joint_position(target)
                robot_state = self.poll_joint_position()
                obses = robot_state["position"]

                unscaled_pos = (2.0 * obses - self.leap_dof_upper - self.leap_dof_lower) / (self.leap_dof_upper - self.leap_dof_lower)
                current_frame = torch.cat([unscaled_pos, target], dim=-1).float()[None]

                obs_hist_buf[:, :-1] = obs_hist_buf[:, 1:].clone()
                obs_hist_buf[:, -1] = current_frame
                proprio_hist_buf[:, :-1] = proprio_hist_buf[:, 1:].clone()
                proprio_hist_buf[:, -1] = current_frame

                elapsed = time.time() - start_time
                sleep_time = max(0.0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                loop_dt = time.time() - start_time

                if step_idx % self.args.print_every == 0:
                    loop_hz = 1.0 / max(loop_dt, 1e-6)
                    self._maybe_warn_low_loop_hz(loop_hz)
                    print(
                        f"[deploy] step={step_idx:06d} | loop_hz={loop_hz:5.1f} | "
                        f"action_l2={action.norm().item():.3f} | "
                        f"latent_norm={student_latent.norm().item():.3f} | "
                        f"target_min={target.min().item():.3f} | target_max={target.max().item():.3f}"
                    )

                if self.args.max_steps > 0 and step_idx >= self.args.max_steps:
                    print(f"[INFO] Reached max_steps={self.args.max_steps}, stopping deployment loop.")
                    break

        except KeyboardInterrupt:
            print("[INFO] Deployment interrupted by user.")
        finally:
            if self.args.disable_torque_on_exit:
                self.dxl_client.set_torque_enabled(self.motors, False)
                print("[INFO] Motors disabled.")


def build_argparser():
    parser = argparse.ArgumentParser(description="Deploy stage2 cylinder rotation policy to the real LEAP hand.")
    default_stage1_cfg = str(
        pathlib.Path(__file__).resolve().parents[1]
        / "tasks"
        / "leap_hand_cylinder_rotation"
        / "agents"
        / "rl_games_ppo_cfg.yaml"
    )
    parser.add_argument("--stage2-checkpoint", type=str, required=True, help="Path to the trained stage2 checkpoint.")
    parser.add_argument("--stage1-cfg", type=str, default=default_stage1_cfg, help="Stage1 rl_games config path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--port", type=str, default="auto", help="Serial port. Use 'auto' to try common candidates.")
    parser.add_argument("--baudrate", type=int, default=4000000, help="Dynamixel baudrate.")
    parser.add_argument("--hz", type=float, default=30.0, help="Control frequency.")
    parser.add_argument("--object-scale", type=float, default=1.0, help="Cylinder scale used to select the warmup grasp pose.")
    parser.add_argument(
        "--grasp-cache-dir",
        type=str,
        default="",
        help="Optional directory containing cached grasp pose tables for warmup selection.",
    )
    parser.add_argument(
        "--grasp-cache-prefix",
        type=str,
        default=DEFAULT_GRASP_CACHE_PREFIX,
        help="Filename prefix for cached grasp pose tables.",
    )
    parser.add_argument(
        "--warmup-mode",
        type=str,
        choices=("auto", "analytic", "cache"),
        default="auto",
        help="Warmup pose source: analytic only, cache only, or auto (use cache when configured, else analytic).",
    )
    parser.add_argument("--warmup-seconds", type=float, default=4.0, help="Seconds spent moving to the initial pose.")
    parser.add_argument("--warmup-close-scale", type=float, default=0.15, help="Extra grasp-closing factor applied to the warmup pose.")
    parser.add_argument("--read-retries", type=int, default=8, help="Number of joint-state read retries before surfacing an error.")
    parser.add_argument("--read-retry-interval", type=float, default=0.02, help="Seconds to wait between joint-state read retries.")
    parser.add_argument("--kp", type=float, default=800.0, help="Motor P gain.")
    parser.add_argument("--kd", type=float, default=200.0, help="Motor D gain.")
    parser.add_argument("--curr-lim", type=float, default=500.0, help="Motor current limit in mA.")
    parser.add_argument("--print-every", type=int, default=30, help="Print one deployment status line every N control steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max deployment steps. 0 means run until Ctrl+C.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Do not send motor commands; still run inference and polling.")
    parser.add_argument(
        "--disable-torque-on-exit",
        action="store_true",
        default=False,
        help="Disable motor torque when the script exits.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    controller = LEAPHandCylinderRotationStage2Controller(args)
    controller.deploy()


if __name__ == "__main__":
    main()
