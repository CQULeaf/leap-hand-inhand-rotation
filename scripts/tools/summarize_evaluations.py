#!/usr/bin/env python3

"""Collect evaluation summary.json files into a compact comparison table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


parser = argparse.ArgumentParser(description="Collect cylinder-rotation ablation summary.json files.")
parser.add_argument("summaries", nargs="+", type=Path, help="summary.json files or directories containing summary.json.")
parser.add_argument("--format", choices=("tsv", "markdown"), default="markdown", help="Output table format.")
args = parser.parse_args()


def load_summary(path: Path) -> tuple[str, dict]:
    summary_path = path / "summary.json" if path.is_dir() else path
    with summary_path.open("r", encoding="utf-8") as f:
        return summary_path.parent.name, json.load(f)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        return f"{value:.{digits}f}"
    return str(value)


def row(name: str, summary: dict) -> list[str]:
    ttf_norm = None
    if isinstance(summary.get("ttf_mean"), (int, float)):
        ttf_norm = float(summary["ttf_mean"]) / 20.0
    obj_vel = None
    if isinstance(summary.get("mean_object_linear_velocity_mean"), (int, float)):
        obj_vel = float(summary["mean_object_linear_velocity_mean"]) * 100.0
    return [
        name,
        str(summary.get("policy_type", "")),
        str(summary.get("eval_preset", "")),
        str(summary.get("task", "")),
        fmt(summary.get("success_rate")),
        fmt(ttf_norm),
        fmt(summary.get("net_rotation_turns_mean")),
        fmt(summary.get("mean_projected_angular_velocity_mean")),
        fmt(obj_vel),
        fmt(summary.get("mean_command_torque_l1_mean")),
        fmt(summary.get("mean_latent_mse_mean"), digits=6),
        fmt(summary.get("mean_teacher_student_action_l2_mean"), digits=6),
    ]


headers = [
    "Run",
    "Policy",
    "Preset",
    "Task",
    "Success",
    "TTF",
    "Turns",
    "Spin",
    "ObjVel(cm/s)",
    "Torque",
    "LatentMSE",
    "ActionGap",
]
rows = [row(name, summary) for name, summary in (load_summary(path) for path in args.summaries)]

if args.format == "tsv":
    print("\t".join(headers))
    for values in rows:
        print("\t".join(values))
else:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for values in rows:
        print("| " + " | ".join(values) + " |")
