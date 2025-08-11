# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for loading and caching test data."""

from __future__ import annotations

import os

import boto3
from PIL import Image

_DEFAULT_CACHE_DIR = "~/.cache/modular/testdata"
_S3_BUCKET = "modular-bazel-artifacts-public"
_S3_PREFIX = f"s3://{_S3_BUCKET}/"


def load_from_s3(file_path: str, cache_dir: str | None = None) -> str:
    if not file_path.startswith(_S3_PREFIX):
        raise ValueError(f"Invalid S3 path: {file_path}")
    file_path = file_path[len(_S3_PREFIX) :]
    cache_dir = cache_dir or os.path.expanduser(_DEFAULT_CACHE_DIR)
    local_path = os.path.join(cache_dir, file_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        s3 = boto3.client("s3")
        s3.download_file(
            _S3_BUCKET,
            file_path,
            local_path,
        )
    return local_path


def load_bytes(file_path: str, cache_dir: str | None = None) -> bytes:
    with open(load_from_s3(file_path, cache_dir), "rb") as f:
        return f.read()


def load_image(image_path: str, cache_dir: str | None = None) -> Image.Image:
    return Image.open(load_from_s3(image_path, cache_dir)).convert("RGB")
