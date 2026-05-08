#!/usr/bin/env python3
"""Validate required binary assets before Isaac Lab tries to load them."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_hand_v1_right"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class ExpectedAsset:
    path: Path
    size: int
    sha256: str


EXPECTED_ASSETS = (
    ExpectedAsset(
        ASSET_ROOT / "leap_hand_right.usd",
        1605,
        "13398e1850c7737e864b1bcd7205c5dd1e7c6831aa008c43d312c39cb2c12675",
    ),
    ExpectedAsset(
        ASSET_ROOT / "configuration/leap_hand_right_base.usd",
        2724719,
        "ead8bc1f84792b49fee017cbaf05edead1c2c4fc82d85fbe07c64c5a4244f8a6",
    ),
    ExpectedAsset(
        ASSET_ROOT / "configuration/leap_hand_right_physics.usd",
        6444,
        "c905b89cb698bfc5385439ab68329a371c722defc92512e32e14edc2afe81217",
    ),
    ExpectedAsset(
        ASSET_ROOT / "configuration/leap_hand_right_sensor.usd",
        650,
        "c560a256351c9469a305f98017df099dbe1f24d19c9d275df7cd080a8d5ba61c",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def main() -> int:
    failures: list[str] = []

    for asset in EXPECTED_ASSETS:
        if not asset.path.exists():
            failures.append(f"missing: {rel(asset.path)}")
            continue

        if is_lfs_pointer(asset.path):
            failures.append(f"git-lfs pointer, not a real USD: {rel(asset.path)}")
            continue

        size = asset.path.stat().st_size
        if size != asset.size:
            failures.append(f"size mismatch: {rel(asset.path)} expected {asset.size}, got {size}")
            continue

        digest = sha256_file(asset.path)
        if digest != asset.sha256:
            failures.append(f"sha256 mismatch: {rel(asset.path)} expected {asset.sha256}, got {digest}")

    if failures:
        print("[FAIL] LEAP Hand USD asset check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("\nRun `git lfs pull` or restore the real USD files before launching Isaac Lab.", file=sys.stderr)
        return 1

    print("[OK] LEAP Hand USD assets are present and match expected hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
