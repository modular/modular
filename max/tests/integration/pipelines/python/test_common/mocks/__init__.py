# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for working with mocks for unit testing"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional, Union

from max.driver import DeviceSpec, scan_available_devices
from max.engine import GPUProfilingMode
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedEncoding, TextGenerationPipeline

from .pipeline_config import (
    DummyMAXModelConfig,
    DummyPipelineConfig,
    mock_estimate_memory_footprint,
    mock_huggingface_config,
    mock_huggingface_hub_repo_exists_with_retry,
    mock_pipeline_config_hf_dependencies,
)
from .pipeline_model import MockPipelineModel
from .tokenizer import MockTextTokenizer


@contextmanager
def retrieve_mock_text_generation_pipeline(
    vocab_size: int,
    eos_token: int,
    seed: int = 42,
    eos_prob: float = 0.1,
    max_length: Optional[int] = None,
    max_new_tokens: Union[int, None] = None,
    device_specs: Optional[list[DeviceSpec]] = None,
) -> Generator[tuple[MockTextTokenizer, TextGenerationPipeline], None, None]:
    if eos_token > vocab_size:
        msg = f"eos_token provided '{eos_token}' must be less than vocab_size provided '{vocab_size}'"
        raise ValueError(msg)

    if not device_specs:
        device_specs = scan_available_devices()

    mock_config = DummyPipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M-Instruct",
        max_length=max_length,
        device_specs=device_specs,
        quantization_encoding=SupportedEncoding.float32,
        kv_cache_strategy=KVCacheStrategy.MODEL_DEFAULT,
        enable_structured_output=False,
        gpu_profiling=GPUProfilingMode.OFF,
        eos_prob=eos_prob,
        vocab_size=vocab_size,
        eos_token=eos_token,
    )

    tokenizer = MockTextTokenizer(
        max_new_tokens=max_new_tokens,
        seed=seed,
        vocab_size=vocab_size,
    )

    try:
        pipeline: TextGenerationPipeline[TextContext] = TextGenerationPipeline(
            pipeline_config=mock_config,
            pipeline_model=MockPipelineModel,
            eos_token_id=eos_token,
            weight_adapters={},
        )

        yield tokenizer, pipeline
    finally:
        ...


__all__ = [
    "MockTextTokenizer",
    "DummyMAXModelConfig",
    "DummyPipelineConfig",
    "mock_estimate_memory_footprint",
    "mock_huggingface_config",
    "mock_huggingface_hub_repo_exists_with_retry",
    "mock_pipeline_config_hf_dependencies",
    "retrieve_mock_text_generation_pipeline",
]
