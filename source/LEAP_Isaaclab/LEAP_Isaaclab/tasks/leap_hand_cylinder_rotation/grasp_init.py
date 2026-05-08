"""Reusable grasp initialization helpers for LEAP cylinder rotation."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import warnings

import numpy as np

import torch

CANONICAL_BASE_POSE = torch.tensor(
    [[0.000, 0.500, 0.000, 0.000,
      -0.750, 1.300, 0.000, 0.750,
      1.750, 1.500, 1.750, 1.750,
      0.000, 1.000, 0.000, 0.000]],
    dtype=torch.float32,
)

CANONICAL_SCALE_POSE_DELTA = torch.tensor(
    [[0.00, -0.40, -0.25, -0.15,
      0.00, -0.35, -0.20, -0.10,
      0.00, -0.35, -0.20, -0.10,
      0.08, -0.25, -0.15, -0.10]],
    dtype=torch.float32,
)

DEFAULT_SCALE_BUCKETS = (0.85, 0.90, 1.00, 1.10, 1.15)
DEFAULT_GRASP_CACHE_PREFIX = "leap_cylinder"
_GRASP_CACHE_TABLES: dict[Path, torch.Tensor] = {}

__all__ = [
    "CANONICAL_BASE_POSE",
    "CANONICAL_SCALE_POSE_DELTA",
    "DEFAULT_SCALE_BUCKETS",
    "DEFAULT_GRASP_CACHE_PREFIX",
    "build_scale_conditioned_pose",
    "grasp_cache_file_for_scale",
    "load_grasp_cache",
    "sample_initial_poses_from_cfg",
    "save_grasp_cache",
    "validate_grasp_scale_buckets",
    "select_scale_bucket",
]


def select_scale_bucket(object_scale: float, buckets: Sequence[float]) -> float:
    if len(buckets) == 0:
        raise ValueError("buckets must not be empty")
    # On an exact midpoint, prefer the lower bucket so the policy is deterministic.
    return min(buckets, key=lambda bucket: (abs(bucket - object_scale), bucket))


def build_scale_conditioned_pose(
    base_pose: torch.Tensor,
    pose_delta: torch.Tensor,
    object_scale: float,
) -> torch.Tensor:
    if base_pose.shape != pose_delta.shape:
        raise ValueError(
            "base_pose and pose_delta must have the same shape; "
            f"got {tuple(base_pose.shape)} and {tuple(pose_delta.shape)}"
        )

    pose_delta = pose_delta.to(device=base_pose.device, dtype=base_pose.dtype)
    return base_pose + (object_scale - 1.0) * pose_delta


def _normalize_object_scales(
    object_scales: torch.Tensor | Sequence[float] | float,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    scales = torch.as_tensor(object_scales, dtype=dtype, device=device)
    if scales.ndim == 0:
        return scales.reshape(1)
    if scales.ndim == 2 and scales.shape[1] == 1:
        return scales[:, 0]
    if scales.ndim != 1:
        raise ValueError(f"object_scales must be a scalar, (N,) tensor, or (N, 1) tensor; got {tuple(scales.shape)}")
    return scales


def _expand_pose_batch(
    pose: torch.Tensor,
    batch_size: int,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pose_batch = pose.to(device=device, dtype=dtype)
    if pose_batch.ndim != 2 or pose_batch.shape[1] != 16:
        raise ValueError(f"{name} must have shape (N, 16); got {tuple(pose_batch.shape)}")
    if pose_batch.shape[0] == batch_size:
        return pose_batch.clone()
    if pose_batch.shape[0] == 1:
        return pose_batch.expand(batch_size, -1).clone()
    raise ValueError(f"{name} must have batch size 1 or {batch_size}; got {pose_batch.shape[0]}")


def _build_scale_conditioned_pose_batch(
    base_pose: torch.Tensor,
    pose_delta: torch.Tensor,
    object_scales: torch.Tensor,
) -> torch.Tensor:
    scale_delta = object_scales.unsqueeze(-1) - 1.0
    return base_pose + scale_delta * pose_delta


def _grasp_cache_scale_suffix(scale: float) -> str:
    scale_decimal = Decimal(str(scale)).normalize()
    scale_text = format(scale_decimal, "f").rstrip("0").rstrip(".")
    decimal_places = scale_text.partition(".")[2] if "." in scale_text else ""
    precision = max(2, len(decimal_places))
    scaled = scale_decimal * (Decimal(10) ** precision)
    scale_int = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    return f"s{scale_int:0{precision + 1}d}"


def grasp_cache_file_for_scale(cache_dir: str | Path, prefix: str, scale: float) -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{prefix}_grasp_{_grasp_cache_scale_suffix(scale)}.npy"


def _validate_grasp_pose_matrix(poses: np.ndarray, *, source: str) -> None:
    if poses.ndim != 2 or poses.shape[1] != 16:
        raise ValueError(
            "grasp cache must contain an N x 16 pose matrix; "
            f"got shape {tuple(poses.shape)} from {source}"
        )
    if poses.shape[0] <= 0:
        raise ValueError(
            "grasp cache must contain at least one pose row; "
            f"got shape {tuple(poses.shape)} from {source}"
        )


def load_grasp_cache(
    cache_dir: str | Path,
    prefix: str,
    scale: float,
    *,
    scale_buckets: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if scale_buckets is not None:
        scale = select_scale_bucket(scale, scale_buckets)

    cache_file = grasp_cache_file_for_scale(cache_dir, prefix, scale)
    if not cache_file.is_file():
        raise FileNotFoundError(f"Grasp cache file not found: {cache_file}")

    cache_key = cache_file.resolve()
    cached_poses = _GRASP_CACHE_TABLES.get(cache_key)
    if cached_poses is None:
        poses = np.load(cache_file, allow_pickle=False)
        _validate_grasp_pose_matrix(poses, source=str(cache_file))
        cached_poses = torch.as_tensor(poses, dtype=torch.float32, device="cpu")
        _GRASP_CACHE_TABLES[cache_key] = cached_poses

    return cached_poses.to(dtype=torch.float32, device=device)


def validate_grasp_scale_buckets(
    grasp_scale_buckets: Sequence[float],
    object_scale_lower: float,
    object_scale_upper: float,
) -> tuple[float, ...]:
    buckets = tuple(float(bucket) for bucket in grasp_scale_buckets)
    if len(buckets) == 0:
        raise ValueError("grasp_scale_buckets must not be empty")
    if len(set(buckets)) != len(buckets):
        raise ValueError("grasp_scale_buckets must be unique")
    if object_scale_lower > object_scale_upper:
        raise ValueError("object_scale_lower must be less than or equal to object_scale_upper")

    outside_range = [bucket for bucket in buckets if bucket < object_scale_lower or bucket > object_scale_upper]
    if outside_range:
        raise ValueError(
            "grasp_scale_buckets must lie within object_scale_lower/object_scale_upper; "
            f"got {outside_range} outside [{object_scale_lower}, {object_scale_upper}]"
        )

    return buckets


def sample_initial_poses_from_cfg(
    *,
    cfg,
    object_scales: torch.Tensor | Sequence[float] | float,
    base_pose: torch.Tensor,
    pose_delta: torch.Tensor,
) -> torch.Tensor:
    device = base_pose.device
    dtype = base_pose.dtype
    normalized_scales = _normalize_object_scales(object_scales, device=device, dtype=dtype)
    batch_size = int(normalized_scales.shape[0])
    base_pose_batch = _expand_pose_batch(base_pose, batch_size, name="base_pose", device=device, dtype=dtype)
    pose_delta_batch = _expand_pose_batch(pose_delta, batch_size, name="pose_delta", device=device, dtype=dtype)
    analytic_poses = _build_scale_conditioned_pose_batch(base_pose_batch, pose_delta_batch, normalized_scales)

    if not getattr(cfg, "use_grasp_cache", False):
        return analytic_poses

    scale_buckets = tuple(float(bucket) for bucket in getattr(cfg, "grasp_scale_buckets", ()))
    cache_dir = getattr(cfg, "grasp_cache_dir", "")
    cache_prefix = getattr(cfg, "grasp_cache_prefix", DEFAULT_GRASP_CACHE_PREFIX)

    sampled_poses = analytic_poses.clone()
    env_ids_by_bucket: dict[float, list[int]] = {}
    for env_idx, object_scale in enumerate(normalized_scales.detach().cpu().tolist()):
        scale_bucket = select_scale_bucket(object_scale, scale_buckets) if scale_buckets else float(object_scale)
        env_ids_by_bucket.setdefault(scale_bucket, []).append(env_idx)

    for scale_bucket, env_indices in env_ids_by_bucket.items():
        try:
            cached_poses = load_grasp_cache(
                cache_dir,
                cache_prefix,
                scale_bucket,
                device=device,
            ).to(dtype=dtype, device=device)
        except FileNotFoundError:
            warnings.warn(
                (
                    "Grasp cache missing for scale bucket "
                    f"{scale_bucket:.2f}; falling back to analytic pose for {len(env_indices)} env(s)."
                ),
                stacklevel=2,
            )
            continue

        sample_indices = torch.randint(0, cached_poses.shape[0], (len(env_indices),), device=device)
        sampled_poses[torch.as_tensor(env_indices, device=device, dtype=torch.long)] = cached_poses[sample_indices]

    return sampled_poses


def save_grasp_cache(
    cache_dir: str | Path,
    prefix: str,
    scale: float,
    poses: torch.Tensor | np.ndarray,
) -> Path:
    cache_file = grasp_cache_file_for_scale(cache_dir, prefix, scale)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    pose_array = poses.detach().cpu().numpy() if isinstance(poses, torch.Tensor) else np.asarray(poses)
    _validate_grasp_pose_matrix(pose_array, source="input poses")

    pose_array = pose_array.astype(np.float32, copy=False)
    np.save(cache_file, pose_array)
    _GRASP_CACHE_TABLES[cache_file.resolve()] = torch.as_tensor(pose_array, dtype=torch.float32, device="cpu")
    return cache_file
