"""
Leap Hand cylinder z-axis rotation environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

cylinder_rotation_task_entry = "LEAP_Isaaclab.tasks.leap_hand_cylinder_rotation"

gym.register(
    id="Isaac-CylinderRotation-Leap",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-NoObjPosPriv",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationNoObjectPosPrivEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-NoComPriv",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationNoComPrivEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-NoDynPriv",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationNoDynPrivEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-DirectPriv",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_direct_priv_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraPDRand",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraPDRandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraRand",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraRandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraPRandLite",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraPRandLiteEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraPDRandLite",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraPDRandLiteEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraObsNoise",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraObsNoiseEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraComWide",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraComWideEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CylinderRotation-Leap-HoraHighComSlice",
    entry_point=f"{cylinder_rotation_task_entry}.cylinder_rotation_env:CylinderRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCylinderRotationHoraHighComSliceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
