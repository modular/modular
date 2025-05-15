# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from typing import Optional

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.nn.kv_cache import (
    KVCacheParams,
)
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    SupportedEncoding,
)
from transformers import AutoConfig

from ..mistral.model import MistralModel


class Mistral3Model(MistralModel):
    """Text-only Mistral3 pipeline model implementation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        huggingface_config = huggingface_config.text_config

        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return super().get_kv_params(
            huggingface_config.text_config,
            n_devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return super().get_num_layers(huggingface_config.text_config)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        huggingface_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return super().calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        return super().estimate_kv_cache_size(
            pipeline_config,
            available_cache_memory,
            devices,
            huggingface_config.text_config,
            kv_cache_config,
            cache_dtype,
        )
