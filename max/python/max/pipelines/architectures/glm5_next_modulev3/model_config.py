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
"""Config for GLM-5.3-Flash (``glm5_next``) on the ModuleV3 API.

The ModuleV3 configs are their own hierarchy, so this derives from the
ModuleV3 DeepSeek-V3.2 config and carries GLM-5.3-Flash's KDA, mHC and k-pool
fields on top. Everything that does not depend on the module API -- the hybrid
schedule, the KDA dimensions, the three-child cache tree and the state and
batch sizing -- is read through the graph-API
:mod:`~max.pipelines.architectures.glm5_next.model_config` so the two
architectures cannot size their caches or batches differently.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface, spec_decode_cache_slack
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.glm5_next.model_config import (
    LINEAR_ATTENTION,
    MLA_KERNEL_LATENT_WIDTH,
    SPARSE_ATTENTION,
    layer_indices,
    recurrent_state_bytes,
    residual_bytes_per_token,
    state_bounded_batch_size,
)
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig as _GraphGlm5NextConfig,
)
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
    parse_quant_scheme,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights import resolve_hf_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override

from ..deepseekV3_2_modulev3.model_config import DeepseekV3_2Config

__all__ = ["Glm5NextConfig"]

# A ModuleV3 FP8 weight's scale, as the adapter names it.
_V3_SCALE = re.compile(
    r"^language_model\.(layers\.\d+\.(?:mlp|self_attn)\..+?)"
    r"\.weight_scale_inv$"
)


_DECLARED_DTYPES = {
    "bfloat16": DType.bfloat16,
    "float16": DType.float16,
    "float32": DType.float32,
}


def _declared_dtype(text_config: AutoConfig) -> DType | None:
    """The dtype the checkpoint declares for its unquantized tensors."""
    for attr in ("dtype", "torch_dtype"):
        declared = getattr(text_config, attr, None)
        if isinstance(declared, str) and declared in _DECLARED_DTYPES:
            return _DECLARED_DTYPES[declared]
    return None


@dataclass(kw_only=True)
class Glm5NextConfig(DeepseekV3_2Config):
    """Configuration for GLM-5.3-Flash on ModuleV3."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float8_e4m3fn"}

    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention family: KDA or sparse MLA."""

    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0
    state_pool_dtype: DType = DType.float32
    """Storage dtype of the KDA conv and recurrent pools."""

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True

    mla_latent_pad_to: int = 576
    """Latent width the MLA kernels compile against; the 512-wide NoPE latent
    is zero-padded to it on the way in."""

    swiglu_limit: float = 10.0
    """Clamp applied before the activation in every MLP and expert."""

    hf_quantization_config: dict[str, Any] | None = None
    quant_scheme: Glm5NextQuantScheme | None = None
    declared_dtype: DType | None = None
    """The dtype the checkpoint declares for its unquantized tensors."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.data_parallel_degree != 1:
            raise ValueError(
                "GLM-5.3-Flash is tensor-parallel only: data_parallel_degree "
                f"must be 1, got {self.data_parallel_degree}."
            )
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "GLM-5.3-Flash is NoPE: qk_rope_head_dim must be 0, got "
                f"{self.qk_rope_head_dim}."
            )
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}."
            )

    @property
    def compute_dtype(self) -> DType:
        """Dtype of activations and every unquantized weight."""
        if self.quant_scheme is not None:
            return self.quant_scheme.compute_dtype
        return self.declared_dtype or DType.bfloat16

    @property
    def kda_layers(self) -> tuple[int, ...]:
        """Indices of the KDA linear-attention layers."""
        return layer_indices(self.layer_types, LINEAR_ATTENTION)

    @property
    def sparse_attention_layers(self) -> tuple[int, ...]:
        """Indices of the sparse-MLA layers."""
        return layer_indices(self.layer_types, SPARSE_ATTENTION)

    @override
    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        """Bounds against the *text* config's ``max_position_embeddings``."""
        return super().calculate_max_seq_len(
            _GraphGlm5NextConfig._get_text_config(huggingface_config),
            model_config,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return _GraphGlm5NextConfig.get_num_layers(huggingface_config)

    @staticmethod
    def construct_kv_params(  # type: ignore[override]
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Builds the MLA, indexer and recurrent-state children of the cache."""
        return _GraphGlm5NextConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def resolve_quant_scheme(
        self, state_dict: Mapping[str, Any]
    ) -> QuantConfig | None:
        """Builds the mixed-precision map from the checkpoint's weight scales.

        The precision map is read off which weights carry a scale, the same
        derivation the graph-API model uses; only the names differ, so they are
        mapped to the ones that derivation reads.
        """
        renamed: dict[str, Any] = {}
        for name, weight in state_dict.items():
            match = _V3_SCALE.match(name)
            if match is not None:
                renamed[f"{match.group(1)}.weight_scale"] = weight
        self.quant_scheme = parse_quant_scheme(
            self.hf_quantization_config,
            renamed,
            self.declared_dtype or DType.bfloat16,
        )
        return self.quant_scheme.config if self.quant_scheme else None

    def per_request_state_bytes(self) -> int:
        """Returns GPU bytes one request's recurrent state occupies, summed
        over every device."""
        return recurrent_state_bytes(self.kv_params, len(self.devices))

    def activation_bytes_per_token(self) -> int:
        """Returns the bytes one live residual occupies per token."""
        return residual_bytes_per_token(
            self.hc_mult, self.hidden_size, self.compute_dtype
        )

    def infer_optimal_batch_size(
        self,
        devices: Sequence[Device],
        *,
        weights_size: int,
        device_memory_utilization: float,
    ) -> int:
        """Returns a memory-safe default ``max_batch_size``; see
        :func:`~max.pipelines.architectures.glm5_next.model_config.state_bounded_batch_size`."""
        return state_bounded_batch_size(
            self.per_request_state_bytes(),
            devices,
            weights_size=weights_size,
            device_memory_utilization=device_memory_utilization,
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the checkpoint's nested ``text_config``."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but config could not be loaded."
            )
        text_config = _GraphGlm5NextConfig._get_text_config(huggingface_config)

        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, kv_cache_config.kv_cache_format
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        kda_dims = _GraphGlm5NextConfig.declared_kda_dims(text_config)
        read_linear = _GraphGlm5NextConfig._linear_attn_field

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            data_parallel_degree=model_config.data_parallel_degree,
            quantization_encoding=quantization_encoding,
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            moe_intermediate_size=text_config.moe_intermediate_size,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            n_shared_experts=text_config.n_shared_experts,
            n_routed_experts=text_config.n_routed_experts,
            routed_scaling_factor=text_config.routed_scaling_factor,
            kv_lora_rank=text_config.kv_lora_rank,
            q_lora_rank=text_config.q_lora_rank,
            qk_rope_head_dim=text_config.qk_rope_head_dim,
            qk_nope_head_dim=text_config.qk_nope_head_dim,
            v_head_dim=text_config.v_head_dim,
            topk_method=text_config.topk_method,
            n_group=text_config.n_group,
            topk_group=text_config.topk_group,
            num_experts_per_tok=text_config.num_experts_per_tok,
            first_k_dense_replace=text_config.first_k_dense_replace,
            norm_topk_prob=text_config.norm_topk_prob,
            hidden_act=text_config.hidden_act,
            max_position_embeddings=text_config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            max_seq_len=max_seq_len,
            rms_norm_eps=text_config.rms_norm_eps,
            scoring_func=text_config.scoring_func,
            attention_bias=text_config.attention_bias,
            attention_dropout=text_config.attention_dropout,
            # Prefill runs the sparse decode kernel: the prefill route needs
            # absorbed FP8 scales, and this checkpoint ships ``kv_b_proj`` in
            # BF16.
            graph_mode="decode",
            index_head_dim=text_config.index_head_dim,
            index_n_heads=text_config.index_n_heads,
            index_topk=text_config.index_topk,
            indexer_types=list(
                getattr(text_config, "indexer_types", None)
                or ["full"] * text_config.num_hidden_layers
            ),
            layer_types=_GraphGlm5NextConfig.resolve_layer_types(text_config),
            linear_num_heads=kda_dims.num_heads,
            linear_head_dim=kda_dims.head_dim,
            linear_conv_kernel_dim=kda_dims.conv_kernel_dim,
            linear_lower_bound=read_linear(
                text_config, "gate_lower_bound", -5.0
            ),
            state_pool_dtype=_GraphGlm5NextConfig.declared_state_dtype(
                kv_cache_config
            ),
            hc_mult=getattr(text_config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(text_config, "hc_sinkhorn_iters", 20),
            hc_eps=getattr(text_config, "hc_eps", 1e-6),
            index_kpool=getattr(text_config, "index_kpool", 4),
            index_kpool_compress=getattr(
                text_config, "index_kpool_compress", True
            ),
            index_kpool_always_select_tail=getattr(
                text_config, "index_kpool_always_select_tail", True
            ),
            mla_latent_pad_to=max(
                text_config.kv_lora_rank + text_config.qk_rope_head_dim,
                MLA_KERNEL_LATENT_WIDTH,
            ),
            swiglu_limit=getattr(text_config, "swiglu_limit", 10.0),
            hf_quantization_config=resolve_hf_quant_config(
                huggingface_config, {}
            ),
            declared_dtype=_declared_dtype(text_config),
        )
