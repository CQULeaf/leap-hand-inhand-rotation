# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Modified for HORA-style Cylinder Rotation Task
# --------------------------------------------------------

from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from LEAP_Isaaclab.tasks.leap_hand_cylinder_rotation.grasp_init import (
    DEFAULT_SCALE_BUCKETS,
    validate_grasp_scale_buckets,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from pathlib import Path

current_dir = Path(__file__).parent


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (3.0, 3.0),
            "damping_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_scale_size = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (1.0, 1.0),
        },
    )


@configclass
class LeapHandCylinderRotationEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    min_episode_length_s = 20.0
    episode_length_s = 20.0
    action_space = 16
    hist_len = 3
    prop_hist_len = 30
    store_cur_actions = True
    # HORA-style proprioceptive history:
    # hist_len * (16 joint_pos + 16 action_target) = 96
    observation_space = 96
    # privileged object info for HORA-style stage 1:
    # object position (3) + scale (1) + mass (1) + friction (1) + com (3) = 9
    state_space = 9
    rlgames_obs_groups = {"obs": ["policy", "critic"], "states": ["critic"]}
    rlgames_concate_obs_group = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**20
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuated_joint_names = [
        'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7',
        'a_8', 'a_9', 'a_10', 'a_11', 'a_12', 'a_13', 'a_14', 'a_15'
    ]
    fingertip_body_names = [
        'fingertip',
        'thumb_fingertip',
        'fingertip_2',
        'fingertip_3'
    ]

    # cylinder object - keep the default URDF axis aligned with world z
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{Path(__file__).parent.parent.parent}/assets/cylinder/cylinder.urdf",
            fix_base=False,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=None
                )
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.10, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)

    # reward scales
    # HORA-style continuous z-axis rotation task parameters
    z_rotation_steps: int = 16

    rot_reward_scale: float = 1.0
    rot_reward_clip_min: float = -0.5
    rot_reward_clip_max: float = 0.5

    pose_diff_penalty_scale: float = -0.1

    linvel_penalty_scale: float = -0.3

    torque_penalty_scale: float = -0.0005
    work_penalty_scale: float = -0.001

    action_penalty_scale: float = -0.00005

    fall_dist: float = 0.07
    goal_center_penalty_scale: float = -0.4
    alive_reward_scale: float = 0.0

    # HORA-style privileged information ranges for stage 1
    include_object_pos: bool = True
    include_object_pos_in_privileged_info: bool = True
    include_scale: bool = True
    include_mass: bool = True
    include_friction: bool = True
    include_com: bool = True
    privileged_scale_value: float = 1.0
    use_grasp_cache: bool = False
    grasp_cache_dir: str = ""
    grasp_cache_prefix: str = "leap_cylinder"
    grasp_scale_buckets: list[float] = list(DEFAULT_SCALE_BUCKETS)
    randomize_priv_mass: bool = True
    priv_mass_lower: float = 0.03
    priv_mass_upper: float = 0.20
    randomize_priv_friction: bool = True
    priv_friction_lower: float = 0.3
    priv_friction_upper: float = 3.0
    randomize_priv_com: bool = True
    priv_com_lower: float = -0.01
    priv_com_upper: float = 0.01
    object_scale_lower: float = 0.85
    object_scale_upper: float = 1.15

    # scale-conditioned stable grasp initialization
    init_joint_noise: float = 0.015
    init_object_xy_noise: float = 0.0025
    init_object_yaw_noise: float = 0.05
    init_object_y_offset_gain: float = -0.02
    init_object_z_offset_gain: float = 0.01
    joint_pos_obs_noise: float = 0.0

    # HORA-style robot randomization switches
    randomize_robot_pd_gains: bool = False
    robot_p_gain_lower: float = 3.0
    robot_p_gain_upper: float = 3.0
    robot_d_gain_lower: float = 0.1
    robot_d_gain_upper: float = 0.1

    # action parameters
    action_type: str = "relative"
    act_moving_average: float = 1./24

    # adr config
    enable_adr = False  # disable adr for cylinder task initially
    starting_adr_increments = 0
    min_rot_adr_coeff = 0.15
    min_steps_for_dr_change = 240 * 4
    obs_per_timestep = 32
    obs_timesteps = 3

    wrench_trigger_every = 90
    torsional_radius = 0.0
    wrench_prob_per_rollout = 0.5

    # domain randomization config
    events: EventCfg = EventCfg()

    adr_cfg_dict = {
        "num_increments": 25,
        "robot_physics_material": {
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.5)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (2.5, 3.1),
            "damping_distribution_params": (0.05, 0.15)
        },
        "object_physics_material": {
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.5)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.9, 1.3)
        }
    }

    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0.5, 5.)
        },
        "object_spawn": {
            "x_width_spawn": (0.0, 0.01),
            "y_width_spawn": (0.0, 0.01),
            "x_rotation": (0.0, 0.1),
            "y_rotation": (0.0, 0.1),
            "z_rotation": (0.0, 0.0),
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.00),
            "object_pos_bias": (0.0, 0.0),
            "object_rot_noise": (0.0, 0.0),
            "object_rot_bias": (0.0, 0.0),
        },
        "robot_spawn": {
            "joint_pos_noise": (0.0, 0.05),
            "joint_vel_noise": (0.0, 0.01)
        },
        "robot_state_noise": {
            "robot_noise": (0.0, 0.05),
            "robot_bias": (0.0, 0.03)
        },
        "robot_action_noise": {
            "hand_noise": (0.1, 0.2)
        },
        "action_latency": {
            "hand_latency": (0.0, 3.0),
        },
        "obs_latency": {
            "latency": (0.0, 0.0),
        },
    }

    act_max_latency = int(adr_custom_cfg_dict["action_latency"]["hand_latency"][1])
    act_latency_rand = 1
    obs_max_latency = int(adr_custom_cfg_dict["obs_latency"]["latency"][1])
    obs_latency_rand = 1

    def __post_init__(self):
        self.events.object_scale_size.params["scale_range"] = (self.object_scale_lower, self.object_scale_upper)
        if self.randomize_robot_pd_gains:
            self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (
                self.robot_p_gain_lower,
                self.robot_p_gain_upper,
            )
            self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (
                self.robot_d_gain_lower,
                self.robot_d_gain_upper,
            )
        else:
            self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (3.0, 3.0)
            self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (0.1, 0.1)
        self.grasp_scale_buckets = list(
            validate_grasp_scale_buckets(
                self.grasp_scale_buckets,
                self.object_scale_lower,
                self.object_scale_upper,
            )
        )


@configclass
class LeapHandCylinderRotationNoObjectPosPrivEnvCfg(LeapHandCylinderRotationEnvCfg):
    """Variant used for the minimal privileged-information ablation.

    This keeps the HORA-style two-stage pipeline unchanged except that stage1/stage2
    no longer use `object_pos` as privileged information. The critic input becomes:

    - scale (1)
    - mass (1)
    - friction (1)
    - com (3)
    """

    include_object_pos: bool = False
    include_object_pos_in_privileged_info: bool = False
    state_space = 6


@configclass
class LeapHandCylinderRotationNoComPrivEnvCfg(LeapHandCylinderRotationEnvCfg):
    """Privileged-information ablation that removes object CoM from stage1/stage2."""

    include_com: bool = False
    state_space = 6


@configclass
class LeapHandCylinderRotationNoDynPrivEnvCfg(LeapHandCylinderRotationEnvCfg):
    """Privileged-information ablation that removes mass and friction."""

    include_mass: bool = False
    include_friction: bool = False
    state_space = 7


@configclass
class LeapHandCylinderRotationHoraPDRandEnvCfg(LeapHandCylinderRotationEnvCfg):
    """Closer-to-HORA variant that only adds PD gain randomization."""

    randomize_robot_pd_gains: bool = True
    robot_p_gain_lower: float = 2.9
    robot_p_gain_upper: float = 3.1
    robot_d_gain_lower: float = 0.09
    robot_d_gain_upper: float = 0.11


@configclass
class LeapHandCylinderRotationHoraRandEnvCfg(LeapHandCylinderRotationHoraPDRandEnvCfg):
    """Closer-to-HORA variant that adds PD randomization and observation noise."""

    joint_pos_obs_noise: float = 0.02


@configclass
class LeapHandCylinderRotationHoraPRandLiteEnvCfg(LeapHandCylinderRotationEnvCfg):
    """LEAP-specific conservative variant: randomize only P gain in a narrow band."""

    randomize_robot_pd_gains: bool = True
    robot_p_gain_lower: float = 2.96
    robot_p_gain_upper: float = 3.04
    robot_d_gain_lower: float = 0.10
    robot_d_gain_upper: float = 0.10


@configclass
class LeapHandCylinderRotationHoraPDRandLiteEnvCfg(LeapHandCylinderRotationEnvCfg):
    """LEAP-specific conservative variant: randomize P/D gains in a narrow band."""

    randomize_robot_pd_gains: bool = True
    robot_p_gain_lower: float = 2.98
    robot_p_gain_upper: float = 3.02
    robot_d_gain_lower: float = 0.098
    robot_d_gain_upper: float = 0.102


@configclass
class LeapHandCylinderRotationHoraObsNoiseEnvCfg(LeapHandCylinderRotationEnvCfg):
    """LEAP-specific conservative variant: add proprioceptive observation noise only."""

    joint_pos_obs_noise: float = 0.01


@configclass
class LeapHandCylinderRotationHoraComWideEnvCfg(LeapHandCylinderRotationEnvCfg):
    """Stage2-targeted variant: widen only the CoM randomization to the OOD interval."""

    priv_com_lower: float = -0.015
    priv_com_upper: float = 0.015


@configclass
class LeapHandCylinderRotationHoraHighComSliceEnvCfg(LeapHandCylinderRotationEnvCfg):
    """Stage2-targeted variant: match the diagnostic high-CoM slice exactly."""

    object_scale_lower: float = 1.0
    object_scale_upper: float = 1.0
    grasp_scale_buckets: list[float] = [1.0]
    randomize_priv_mass: bool = True
    priv_mass_lower: float = 0.12
    priv_mass_upper: float = 0.12
    randomize_priv_friction: bool = True
    priv_friction_lower: float = 1.0
    priv_friction_upper: float = 1.0
    randomize_priv_com: bool = True
    priv_com_lower: float = 0.015
    priv_com_upper: float = 0.015
