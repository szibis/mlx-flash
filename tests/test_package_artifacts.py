"""Artifact checks run after building distributions, without requiring MLX."""

import os
from pathlib import Path
from zipfile import ZipFile

import pytest


def test_wheel_contains_metal_sources_required_by_kernel_loader():
    directory = os.environ.get("MLX_FLASH_DIST_DIR")
    if not directory:
        pytest.skip("Set MLX_FLASH_DIST_DIR to inspect built distributions")
    wheels = list(Path(directory).glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as wheel:
        source = wheel.read("mlx_flash_compress/kernels/flash_dequant.metal")
    assert b"kernel" in source
