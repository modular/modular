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
"""Configuration for Hy4 (``HYV4ForCausalLM``) models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheQuantizationConfig,
    MultiKVCacheParams,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ModelOutputConfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registration_runtime import (
    DeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override


def check_parallelism(pipeline_config: PipelineConfig, n_devices: int) -> None:
    """Raises unless several devices run tensor-parallel attention.

    The routed experts split over the devices either way: each device runs
    its own on every token and the outputs are summed, with no expert
    dispatch, so ``ep_size`` may be 1 or the device count.
    """
    dp_degree = pipeline_config.model.data_parallel_degree
    ep_size = pipeline_config.runtime.ep_size
    if n_devices > 1 and (dp_degree != 1 or ep_size not in (1, n_devices)):
        raise ValueError(
            f"Hy4 on {n_devices} devices runs tensor-parallel attention with"
            " its experts split over the devices: it needs"
            f" data_parallel_degree 1 and ep_size 1 or {n_devices}, got"
            f" {dp_degree} and {ep_size}"
        )


def full_indexer_layers(indexer_types: list[str]) -> list[int]:
    """Returns the layers that run their own DSA indexer."""
    return [i for i, t in enumerate(indexer_types) if t == "full"]


@dataclass(kw_only=True)
class HYV4Config(ArchConfigWithKVCache, DeploymentConfig):
    """Configuration for Hy4 models (ModuleV3)."""

    takes_batch_context_lengths: ClassVar[bool] = True
    expert_parallel_communication: ClassVar[bool] = False

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float8_e4m3fn"}

    dtype: DType
    kv_params: KVCacheParamInterface
    devices: list[DeviceRef]
    quantization_encoding: SupportedEncoding | None = None
    quant_config: QuantConfig | None = None
    max_batch_context_length: int = 131072
    max_seq_len: int

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    n_shared_experts: int
    n_routed_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    swiglu_limit: float
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: list[str]
    mlp_layer_types: list[str]
    hc_mult: int
    hc_magnitude: float
    hc_eps: float
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    num_nextn_predict_layers: int

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Completes checkpoint quantization before the decoder is built."""
        del outputs
        return replace(
            self,
            quant_config=parse_quant_config(
                huggingface_config, dict(state_dict), self.dtype
            ),
        )

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Validates local expert sharding and applies the planned token size."""
        n_devices = runtime.mesh.num_devices
        if runtime.data_parallel_degree != 1 or runtime.ep_size not in (
            1,
            n_devices,
        ):
            raise ValueError(
                "Hy4 requires tensor-parallel attention and local expert "
                "sharding: data_parallel_degree must be 1 and ep_size must "
                "be 1 or the device count"
            )
        return replace(
            self,
            max_batch_context_length=(
                runtime.max_batch_total_tokens or self.max_batch_context_length
            ),
        )

    def __post_init__(self) -> None:
        n = self.num_hidden_layers
        if len(self.indexer_types) != n or len(self.mlp_layer_types) != n:
            raise ValueError(
                f"indexer_types ({len(self.indexer_types)}) and"
                f" mlp_layer_types ({len(self.mlp_layer_types)}) must both"
                f" have num_hidden_layers ({n}) entries"
            )
        if self.indexer_types[0] != "full":
            raise ValueError(
                "indexer_types[0] must be 'full': a 'shared' layer reuses the"
                f" most recent full layer's top-k, got {self.indexer_types[0]!r}"
            )

    @property
    def num_key_value_heads(self) -> int:
        """MLA expands the latent to one KV head per query head."""
        return self.num_attention_heads

    @property
    def qk_head_dim(self) -> int:
        """The query and key width per head."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def full_indexer_layers(self) -> list[int]:
        """The layers whose indexer runs at full width."""
        return full_indexer_layers(self.indexer_types)

    def indexer_cache_layer(self, layer_idx: int) -> int:
        """Returns a full layer's slot in the indexer K cache.

        The cache holds only full layers, so the slot is the layer's ordinal
        among them.
        """
        return self.full_indexer_layers.index(layer_idx)

    def get_kv_params(self) -> KVCacheParamInterface:
        """The KV cache parameters the config declares."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """The maximum sequence length the config allows."""
        return self.max_seq_len

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """The checkpoint's context length, bounded by the config's."""
        return upper_bounded_default(
            upper_bound=huggingface_config.max_position_embeddings,
            default=model_config.max_length,
        )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Builds the MLA and indexer cache params for the devices."""
        dp_degree = pipeline_config.model.data_parallel_degree
        speculative = pipeline_config.speculative
        # Every leaf of a cache tree must agree on these.
        speculative_method = (
            speculative.speculative_method if speculative else None
        )
        num_draft_tokens = (
            (speculative.num_speculative_tokens or 0) if speculative else 0
        )
        mla = kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            num_layers=huggingface_config.num_hidden_layers,
            devices=devices,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            data_parallel_degree=dp_degree,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        # The DeepSeek V3.2 layout: FP8 keys with one float32 scale per token.
        indexer = kv_cache_config.to_params(
            dtype=DType.float8_e4m3fn,
            n_kv_heads=1,
            head_dim=huggingface_config.index_head_dim,
            num_layers=len(
                full_indexer_layers(huggingface_config.indexer_types)
            ),
            devices=devices,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            data_parallel_degree=dp_degree,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float32,
                quantization_granularity=huggingface_config.index_head_dim,
            ),
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        return MultiKVCacheParams.from_params({"mla": mla, "indexer": indexer})

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        check_parallelism(
            pipeline_config, len(pipeline_config.model.device_specs)
        )
        model_config = model_config or pipeline_config.model
        config = model_config.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be loaded."
            )
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype_for_encoding(
                quantization_encoding, model_config.kv_cache.kv_cache_format
            ),
        )
        return cls(
            dtype=supported_encoding_dtype(quantization_encoding),
            kv_params=kv_params,
            devices=devices,
            quantization_encoding=quantization_encoding,
            max_seq_len=max_seq_len,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            swiglu_limit=config.swiglu_limit,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            indexer_types=list(config.indexer_types),
            mlp_layer_types=list(config.mlp_layer_types),
            hc_mult=config.hc_mult,
            hc_magnitude=config.hc_magnitude,
            hc_eps=config.hc_eps,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=get_rope_theta(config),
            max_position_embeddings=config.max_position_embeddings,
            num_nextn_predict_layers=getattr(
                config, "num_nextn_predict_layers", 0
            ),
        )
