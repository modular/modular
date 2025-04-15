# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from functools import wraps
from unittest.mock import MagicMock, patch

from max.driver import DeviceSpec
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import (
    KVCacheStrategy,
)
from max.pipelines import (
    MEMORY_ESTIMATOR,
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    SupportedEncoding,
)
from transformers import AutoConfig


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
        max_batch_size: int | None,
        max_length: int | None,
        device_specs: list[DeviceSpec],
        quantization_encoding: SupportedEncoding,
    ):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.device_specs = device_specs
        self._model_config = DummyMAXModelConfig(
            model_path=model_path,
            device_specs=device_specs,
            quantization_encoding=quantization_encoding,
            _kv_cache_config=KVCacheConfig(
                cache_strategy=KVCacheStrategy.CONTINUOUS,
            ),
            _huggingface_config=MagicMock(),
        )


def mock_estimate_memory_footprint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch.object(
            MEMORY_ESTIMATOR, "estimate_memory_footprint", return_value=0
        ):
            return func(*args, **kwargs)

    return wrapper


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
        with patch("huggingface_hub.repo_exists", return_value=True):
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
