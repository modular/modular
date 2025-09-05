# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path
from typing import Union

import pytest
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


@pytest.fixture(scope="session")
def fixture_testdatadirectory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("MAX_SERVE_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture(scope="session")
def fixture_tokenizer(
    fixture_testdatadirectory: Path,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(fixture_testdatadirectory)
    return tokenizer


class MockModelConfig(MAXModelConfig):
    def __init__(self):
        self.served_model_name = "echo"


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self.max_batch_size = 1
        self._model_config = MockModelConfig()


@pytest.fixture
def mock_pipeline_config() -> PipelineConfig:
    return MockPipelineConfig()
