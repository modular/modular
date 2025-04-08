# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for working with mocks for unit testing"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional, Union
from unittest.mock import MagicMock, PropertyMock, patch

from max.driver import DeviceSpec, scan_available_devices
from max.engine import GPUProfilingMode
from max.pipelines import (
    KVCacheConfig,
    MAXModelConfig,
    ProfilingConfig,
    SamplingConfig,
    SupportedEncoding,
    TextGenerationPipeline,
)
from max.pipelines.core import TextContext
from max.pipelines.kv_cache import KVCacheStrategy

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

    # Create a mock Pipeline Config
    mock_config = MagicMock()
    mock_config.profiling_config = ProfilingConfig(
        gpu_profiling=GPUProfilingMode.OFF,
    )
    mock_config.sampling_config = SamplingConfig(
        enable_structured_output=False,
    )
    mock_config.model_config = MAXModelConfig(
        model_path="HuggingFaceTB/SmolLM-135M-Instruct",
        device_specs=device_specs,
        quantization_encoding=SupportedEncoding.float32,
        _kv_cache_config=KVCacheConfig(
            cache_strategy=KVCacheStrategy.MODEL_DEFAULT,
        ),
    )

    mock_config.eos_prob = eos_prob
    mock_config.max_length = max_length
    mock_config.vocab_size = vocab_size
    mock_config.eos_token = [eos_token]

    tokenizer = MockTextTokenizer(
        max_new_tokens=max_new_tokens,
        seed=seed,
        vocab_size=vocab_size,
    )

    mock_hf_config = MagicMock()
    try:
        with patch.object(
            TextGenerationPipeline,
            "huggingface_config",
            new_callable=PropertyMock,
        ) as mock_property:
            mock_property.return_value = mock_hf_config
            pipeline: TextGenerationPipeline[TextContext] = (
                TextGenerationPipeline(
                    pipeline_config=mock_config,
                    pipeline_model=MockPipelineModel,
                    eos_token_id=eos_token,
                    weight_adapters={},
                )
            )

            yield tokenizer, pipeline
    finally:
        ...


__all__ = ["MockTextTokenizer", "retrieve_mock_text_generation_pipeline"]
