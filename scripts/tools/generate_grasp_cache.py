#!/usr/bin/env python3
"""Generate deterministic LEAP cylinder grasp caches for each scale bucket."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "source" / "LEAP_Isaaclab" / "LEAP_Isaaclab" / "tasks"
if str(TASKS_ROOT) not in sys.path:
    sys.path.insert(0, str(TASKS_ROOT))

from leap_hand_cylinder_rotation.grasp_init import (  # noqa: E402
    CANONICAL_BASE_POSE,
    CANONICAL_SCALE_POSE_DELTA,
    DEFAULT_GRASP_CACHE_PREFIX,
    DEFAULT_SCALE_BUCKETS,
    build_scale_conditioned_pose,
    save_grasp_cache,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("samples-per-scale must be positive")
    return parsed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where one grasp cache .npy file per scale will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_GRASP_CACHE_PREFIX,
        help="Cache filename prefix, for example 'leap_cylinder'.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALE_BUCKETS),
        help="Scale buckets to generate.",
    )
    parser.add_argument(
        "--samples-per-scale",
        type=_positive_int,
        default=32,
        help="How many deterministic seed poses to write for each scale bucket.",
    )
    return parser


def generate_grasp_cache_files(
    output_dir: Path,
    scales: list[float] | tuple[float, ...],
    *,
    prefix: str = DEFAULT_GRASP_CACHE_PREFIX,
    samples_per_scale: int = 32,
) -> list[Path]:
    if samples_per_scale <= 0:
        raise ValueError("samples_per_scale must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for scale in scales:
        pose = build_scale_conditioned_pose(CANONICAL_BASE_POSE, CANONICAL_SCALE_POSE_DELTA, scale)
        poses = pose.repeat(samples_per_scale, 1)
        written_files.append(save_grasp_cache(output_dir, prefix, scale, poses))

    return written_files


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    written_files = generate_grasp_cache_files(
        args.output_dir,
        args.scales,
        prefix=args.prefix,
        samples_per_scale=args.samples_per_scale,
    )
    for path in written_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
