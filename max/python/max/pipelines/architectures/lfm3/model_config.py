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
"""Config for the LFM3 hybrid conv and attention decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self

from ..gpt_oss.hybrid_kv_params_util import hybrid_swa_full_kv_params

CONV = "conv"
SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"
STATE_CACHE_KEY = "conv_state"
CONV_LEAF_ID = "lfm3/conv"


@dataclass(kw_only=True)
class Lfm3Config(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Configuration for an LFM3 decoder.

    Each layer is a gated short convolution, sliding-window attention or
    global attention, chosen by ``layer_types``.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: list[str]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    conv_kernel_size: int
    sliding_window: int
    norm_eps: float
    rope_theta: dict[str, float]
    partial_rotary_factor: dict[str, float]
    use_sandwich_norm: bool
    dtype: DType
    devices: list[DeviceRef]
    max_seq_len: int
    kv_params: MultiKVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Returns the text config's decoder layer count."""
        return int(huggingface_config.get_text_config().num_hidden_layers)

    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        """Returns the max sequence length the text config allows."""
        return super().calculate_max_seq_len(
            huggingface_config.get_text_config(), model_config
        )

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        """Returns the sliding and full KV caches beside the conv windows.

        Each kind of layer indexes its own cache 0, 1, 2, ... in layer order.
        A vision-language config contributes its text half.
        """
        huggingface_config = huggingface_config.get_text_config()
        layer_types = huggingface_config.layer_types
        attention = hybrid_swa_full_kv_params(
            layer_types=[t for t in layer_types if t != CONV],
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            allow_kv_head_replication=allow_kv_head_replication,
        )
        state = RecurrentStateParams(
            regions=(
                RecurrentStateRegion(
                    leaf_id=CONV_LEAF_ID,
                    num_layers=layer_types.count(CONV),
                    row_shape=(
                        huggingface_config.hidden_size,
                        huggingface_config.conv_L_cache - 1,
                    ),
                    dtype=DType.bfloat16,
                ),
            ),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )
        return MultiKVCacheParams.from_params(
            {**attention.children, STATE_CACHE_KEY: state}
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the checkpoint's text config."""
        model_config = model_config or pipeline_config.model
        if model_config.huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but it could not be loaded."
            )
        hf = model_config.huggingface_config.get_text_config()
        encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            hf,
            pipeline_config,
            devices,
            model_config.kv_cache,
            cache_dtype_for_encoding(
                encoding, model_config.kv_cache.kv_cache_format
            ),
        )
        rope = hf.rope_parameters
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            layer_types=list(hf.layer_types),
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=cls.get_head_dim(hf),
            conv_kernel_size=hf.conv_L_cache,
            sliding_window=hf.sliding_window,
            norm_eps=hf.norm_eps,
            rope_theta={k: float(v["rope_theta"]) for k, v in rope.items()},
            partial_rotary_factor={
                k: float(v.get("partial_rotary_factor", 1.0))
                for k, v in rope.items()
            },
            use_sandwich_norm=hf.use_sandwich_norm,
            dtype=supported_encoding_dtype(encoding),
            devices=devices,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Returns the config completed with the pipeline's output requirements.

        Args:
            huggingface_config: The checkpoint's model configuration.
            state_dict: The adapted checkpoint weights.
            outputs: The outputs the pipeline requires.

        Returns:
            A copy of this config that returns the logits ``outputs`` asks for.
        """
        del huggingface_config, state_dict
        return replace(self, return_logits=outputs.return_logits)
