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
"""Utilities for working with mocks for unit testing"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional, Union

import hf_repo_lock
from max.driver import DeviceSpec, scan_available_devices
from max.engine import GPUProfilingMode
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    SupportedEncoding,
    TextGenerationPipeline,
    generate_local_model_path,
)

from .pipeline_config import (
    DummyMAXModelConfig,
    DummyPipelineConfig,
    mock_estimate_memory_footprint,
    mock_huggingface_config,
    mock_huggingface_hub_repo_exists_with_retry,
    mock_pipeline_config_hf_dependencies,
    mock_pipeline_config_resolve,
)
from .pipeline_model import MockPipelineModel
from .tokenizer import MockTextTokenizer

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)


@contextmanager
def retrieve_mock_text_generation_pipeline(
    vocab_size: int,
    eos_token: int,
    seed: int = 42,
    eos_prob: float = 0.1,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    device_specs: list[DeviceSpec] | None = None,
) -> Generator[tuple[MockTextTokenizer, TextGenerationPipeline], None, None]:
    if eos_token > vocab_size:
        raise ValueError(
            f"eos_token provided '{eos_token}' must be less than vocab_size provided '{vocab_size}'"
        )

    if not device_specs:
        device_specs = scan_available_devices()

    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    mock_config = DummyPipelineConfig(
        model_path=generate_local_model_path(REPO_ID, REVISION),
        max_length=max_length,
        max_batch_size=None,
        device_specs=device_specs,
        quantization_encoding=SupportedEncoding.float32,
        kv_cache_strategy="paged",
        eos_prob=eos_prob,
        vocab_size=vocab_size,
        eos_token=eos_token,
    )

    tokenizer = MockTextTokenizer(
        max_new_tokens=max_new_tokens,
        seed=seed,
        vocab_size=vocab_size,
        max_length=max_length,
    )

    try:
        pipeline: TextGenerationPipeline[TextContext] = TextGenerationPipeline(
            pipeline_config=mock_config,
            pipeline_model=MockPipelineModel,
            eos_token_id=eos_token,
            weight_adapters={},
            tokenizer=tokenizer,
        )

        yield tokenizer, pipeline
    finally:
        ...


__all__ = [
    "DummyMAXModelConfig",
    "DummyPipelineConfig",
    "MockTextTokenizer",
    "mock_estimate_memory_footprint",
    "mock_huggingface_config",
    "mock_huggingface_hub_repo_exists_with_retry",
    "mock_pipeline_config_hf_dependencies",
    "mock_pipeline_config_resolve",
    "retrieve_mock_text_generation_pipeline",
]
