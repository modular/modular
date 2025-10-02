# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path
from typing import Union

import pytest
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
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


class MockKVCacheConfig(KVCacheConfig):
    def __init__(self, enable_prefix_caching: bool):
        self.enable_prefix_caching = enable_prefix_caching


class MockModelConfig(MAXModelConfig):
    def __init__(self, enable_prefix_caching: bool):
        self.served_model_name = "echo"
        self._kv_cache_config = MockKVCacheConfig(
            enable_prefix_caching=enable_prefix_caching
        )


DEFAULT_ZMQ_ENDPOINT_BASE = "ipc:///tmp/my-secret-uuid-abc123"


class MockPipelineConfig(PipelineConfig):
    def __init__(self, enable_prefix_caching: bool):
        self.max_batch_size = 1
        self._model_config = MockModelConfig(
            enable_prefix_caching=enable_prefix_caching
        )
        self.zmq_endpoint_base: str = DEFAULT_ZMQ_ENDPOINT_BASE


@pytest.fixture
def enable_prefix_caching(request: pytest.FixtureRequest) -> bool:
    """Fixture for a whether prefix caching is enabled
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    # defaults to False if not specified
    return request.param if hasattr(request, "param") else False


@pytest.fixture
def mock_pipeline_config(enable_prefix_caching: bool) -> PipelineConfig:
    return MockPipelineConfig(enable_prefix_caching=enable_prefix_caching)
