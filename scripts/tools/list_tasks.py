"""List registered LEAP Isaac Lab tasks."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from prettytable import PrettyTable

import LEAP_Isaaclab.tasks  # noqa: F401


def main():
    """Print all environments registered by this repository."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available LEAP Isaac Lab Tasks"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        entry_point = str(task_spec.entry_point)
        env_cfg_entry = str(task_spec.kwargs.get("env_cfg_entry_point", ""))
        if "LEAP_Isaaclab.tasks" in entry_point or "LEAP_Isaaclab.tasks" in env_cfg_entry:
            table.add_row([index + 1, task_spec.id, entry_point, env_cfg_entry])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
