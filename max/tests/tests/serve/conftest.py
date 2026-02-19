# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

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
) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(fixture_testdatadirectory)
    return tokenizer


DEFAULT_ZMQ_ENDPOINT_BASE = "ipc:///tmp/my-secret-uuid-abc123"


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
    pipeline_config = PipelineConfig.model_construct(
        # scheduler-required surface
        max_batch_size=1,
        zmq_endpoint_base=DEFAULT_ZMQ_ENDPOINT_BASE,
    )

    kv_cache_config = KVCacheConfig.model_construct(
        enable_prefix_caching=enable_prefix_caching,
    )

    model_config = MAXModelConfig.model_construct(
        served_model_name="echo",
    )

    model_config.kv_cache = kv_cache_config
    pipeline_config.model = model_config

    return pipeline_config
