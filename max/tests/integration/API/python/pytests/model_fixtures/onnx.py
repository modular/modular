# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""All fixtures for end-to-end testing"""
from pathlib import Path

import pytest
from typing import Generator

from model_fixtures.utils import ModelFormat, fixture_generator


@pytest.fixture(scope="session")
def onnx_maskrcnn() -> Generator:
    """This is a path to a model about 180 MB in size."""

    key = "mask_rcnn_resnet50_fpn-pytorch.yaml"
    yield fixture_generator(key, ModelFormat.onnx)


@pytest.fixture(scope="session")
def onnx_camembert() -> Generator:
    """This is a path to a model about 400 MB in size."""

    key = "camembert-base"
    yield fixture_generator(key, ModelFormat.onnx)
