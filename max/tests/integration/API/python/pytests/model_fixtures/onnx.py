# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""All fixtures for end-to-end testing"""
from pathlib import Path
from typing import Generator

import pytest
from model_fixtures.utils import ModelFormat, fixture_generator


@pytest.fixture(scope="session")
def onnx_dlrm() -> Generator:
    """This is a path to a model about 180 MB in size."""

    key = "dlrm-rm1-multihot-pytorch-nobag"
    yield fixture_generator(key, ModelFormat.onnx)


@pytest.fixture(scope="session")
def onnx_bert() -> Generator:
    """This is a path to a model about 400 MB in size."""

    key = "bert-base-uncased-pytorch-seqlen-128"
    yield fixture_generator(key, ModelFormat.onnx)
