# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from unittest.mock import MagicMock

from max.driver import DeviceSpec
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import (
    KVCacheStrategy,
)
from max.pipelines import (
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.config_enums import (
    SupportedEncoding,
)
from max.pipelines.max_config import KVCacheConfig


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
        max_batch_size: int,
        max_length: int,
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
