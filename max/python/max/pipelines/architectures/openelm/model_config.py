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
"""Config for OpenELM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.kv_cache.config import (
    KVCacheConfig,
    cache_dtype_for_encoding,
)
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self


@dataclass
class OpenELMLayerConfig:
    """Per-layer dimensions for OpenELM's layer-wise scaling."""

    num_query_heads: int
    num_kv_heads: int
    ffn_hidden_dim: int
    head_dim: int


def compute_layer_configs(hf_config: AutoConfig) -> list[OpenELMLayerConfig]:
    """Computes per-layer dimensions from the HuggingFace config."""
    num_layers = hf_config.num_transformer_layers
    head_dim = hf_config.head_dim

    def to_list(
        val: float | list[int | float], length: int
    ) -> list[int | float]:
        return val if isinstance(val, list) else [val] * length

    query_heads = to_list(hf_config.num_query_heads, num_layers)
    kv_heads = to_list(hf_config.num_kv_heads, num_layers)
    ffn_mults = to_list(hf_config.ffn_multipliers, num_layers)

    model_dim = hf_config.model_dim
    divisor = getattr(hf_config, "ffn_dim_divisor", 256)

    def make_divisible(v: float, d: int) -> int:
        new_v = max(d, int(v + d / 2) // d * d)
        if new_v < 0.9 * v:
            new_v += d
        return new_v

    return [
        OpenELMLayerConfig(
            num_query_heads=int(query_heads[i]),
            num_kv_heads=int(kv_heads[i]),
            ffn_hidden_dim=make_divisible(ffn_mults[i] * model_dim, divisor),
            head_dim=head_dim,
        )
        for i in range(num_layers)
    ]


@dataclass(kw_only=True)
class OpenELMConfig(ArchConfigWithStoredKVParams):
    """Implementation of ArchConfig for OpenELM models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    # bfloat16 is declared for forward compatibility; only float32 has been
    # validated against the HuggingFace reference implementation.
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    dtype: DType
    max_seq_len: int
    model_dim: int
    vocab_size: int
    head_dim: int
    rms_norm_eps: float
    rope_base: float
    normalize_qk_projections: bool
    layer_configs: list[OpenELMLayerConfig]
    devices: list[DeviceRef]
    kv_params: KVCacheParamInterface
    quantization_encoding: SupportedEncoding | None = None

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """OpenELM stores layer count as ``num_transformer_layers``."""
        return int(huggingface_config.num_transformer_layers)

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
        """Builds one KV cache leaf per distinct per-layer KV head count."""
        layer_configs = compute_layer_configs(huggingface_config)
        head_dim = cls.get_head_dim(huggingface_config)

        layers_per_kv_heads: dict[int, int] = {}
        for layer_cfg in layer_configs:
            layers_per_kv_heads[layer_cfg.num_kv_heads] = (
                layers_per_kv_heads.get(layer_cfg.num_kv_heads, 0) + 1
            )

        children = {
            f"kv_heads_{n_kv_heads}": kv_cache_config.to_params(
                dtype=cache_dtype,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                num_layers=num_layers_in_group,
                devices=devices,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
                allow_kv_head_replication=allow_kv_head_replication,
            )
            for n_kv_heads, num_layers_in_group in sorted(
                layers_per_kv_heads.items()
            )
        }
        return MultiKVCacheParams.from_params(children)

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        model_max = getattr(huggingface_config, "max_context_length", 2048)
        user_max = model_config.max_length
        return model_max if user_max is None else min(user_max, model_max)

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config, huggingface_config, max_seq_len=max_seq_len
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        mc = model_config or pipeline_config.model
        quantization_encoding = _select_quantization_encoding(
            mc, cls.DEFAULT_ENCODING
        )
        devices = [
            DeviceRef(device_type=d.device_type, id=d.id)
            for d in mc.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=mc.kv_cache,
            cache_dtype=cache_dtype_for_encoding(
                quantization_encoding, mc.kv_cache.kv_cache_format
            ),
        )
        return cls(
            dtype=supported_encoding_dtype(quantization_encoding),
            max_seq_len=max_seq_len,
            model_dim=huggingface_config.model_dim,
            vocab_size=huggingface_config.vocab_size,
            head_dim=cls.get_head_dim(huggingface_config),
            rms_norm_eps=getattr(huggingface_config, "rms_norm_eps", 1e-6),
            rope_base=float(
                getattr(huggingface_config, "rope_freq_constant", 10000.0)
            ),
            normalize_qk_projections=getattr(
                huggingface_config, "normalize_qk_projections", False
            ),
            layer_configs=compute_layer_configs(huggingface_config),
            devices=devices,
            kv_params=kv_params,
            quantization_encoding=quantization_encoding,
        )
