# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json
import os
from pathlib import Path

import pytest
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)


@pytest.fixture
def config() -> DeepseekV2Config:
    config = DeepseekV2Config()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path, "r") as file:
        data = json.load(file)
    config.update(data)
    return config
