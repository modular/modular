# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from functools import wraps
from unittest.mock import MagicMock, patch

from max.driver import DeviceSpec
from max.engine import GPUProfilingMode
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import (
    KVCacheStrategy,
)
from max.pipelines import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    ProfilingConfig,
    SamplingConfig,
    SupportedEncoding,
)
from transformers import AutoConfig

from .memory_estimation import mock_estimate_memory_footprint


class DummyMAXModelConfig(MAXModelConfig):
    def weights_size(self) -> int:
        return 1000

    def validate_and_resolve_quantization_encoding_weight_path(
        self, default_encoding: SupportedEncoding
    ) -> None:
        pass

    def validate_and_resolve_with_set_quantization_encoding(
        self,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        default_weights_format: WeightsFormat,
    ) -> None:
        pass


class DummyPipelineConfig(PipelineConfig):
    def __init__(
        self,
        model_path: str,
        quantization_encoding: SupportedEncoding,
        max_batch_size: int | None = None,
        max_length: int | None = None,
        pdl_level: str = "1",
        device_specs: list[DeviceSpec] = [],
        kv_cache_strategy: KVCacheStrategy = KVCacheStrategy.MODEL_DEFAULT,
        gpu_profiling: GPUProfilingMode = GPUProfilingMode.OFF,
        enable_structured_output: bool = False,
        # TODO(AITLIB-328): These values do not belong in PipelineConfig,
        # but are somehow used by MockPipelineModel in pipeline_model.py.
        eos_prob: float | None = None,
        vocab_size: int | None = None,
        eos_token: int | None = None,
    ):
        self.model_path = model_path
        self.quantization_encoding = quantization_encoding
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.pdl_level = pdl_level
        self.device_specs = device_specs

        self._profiling_config = ProfilingConfig(
            gpu_profiling=gpu_profiling,
        )
        self._sampling_config = SamplingConfig(
            enable_structured_output=enable_structured_output,
        )

        self._model_config = DummyMAXModelConfig(
            model_path=model_path,
            device_specs=device_specs,
            quantization_encoding=quantization_encoding,
            _kv_cache_config=KVCacheConfig(
                cache_strategy=kv_cache_strategy,
            ),
            _huggingface_config=MagicMock(),
        )

        # TODO(AITLIB-328): These values do not belong in PipelineConfig,
        # but are somehow used by MockPipelineModel in pipeline_model.py.
        self.eos_prob = eos_prob
        self.vocab_size = vocab_size
        self.eos_token = eos_token


def mock_huggingface_config(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch.object(
            AutoConfig, "from_pretrained", return_value=MagicMock()
        ):
            return func(*args, **kwargs)

    return wrapper


def mock_huggingface_hub_repo_exists_with_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("huggingface_hub.revision_exists", return_value=True):
            return func(*args, **kwargs)

    return wrapper


def mock_huggingface_hub_file_exists(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("huggingface_hub.file_exists", return_value=True):
            return func(*args, **kwargs)

    return wrapper


def mock_pipeline_config_hf_dependencies(func):
    """Decorator that combines multiple mock decorators for pipeline testing.

    Combines:
    - mock_huggingface_hub_repo_exists_with_retry
    - mock_huggingface_hub_file_exists
    - mock_huggingface_config
    - mock_estimate_memory_footprint
    """
    return mock_huggingface_hub_repo_exists_with_retry(
        mock_huggingface_hub_file_exists(
            mock_huggingface_config(mock_estimate_memory_footprint(func))
        )
    )
