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
"""Inkling architecture config and its text and vision backbones."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import ClassVar, Final

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces import ArchConfigWithKVCache
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self

GLOBAL_ATTENTION = "full_attention"
LOCAL_ATTENTION = "sliding_attention"

# A layer's experts are packed FP4 iff its stacked weight ships block scales.
_EXPERT_BLOCK_SCALE = re.compile(
    r"layers\.(\d+)\.mlp\.experts\.w13_weight\.weight_scale"
)

# Options the graph hardcodes; a checkpoint that disagrees would run silently
# wrong, so parsing fails loudly instead.
_REQUIRED_TEXT_FLAGS: Final = (
    "use_gate_bias",
    "use_global_scale",
    "norm_after_topk",
    "shared_expert_sink",
    "use_sconv",
)


@dataclass(kw_only=True, frozen=True)
class InklingTextConfig:
    """Text-backbone config, parsed from the checkpoint's ``text_config``."""

    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float

    vocab_size: int

    unpadded_vocab_size: int
    """Trained rows; the tail up to ``vocab_size`` is padding."""

    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int

    swa_num_attention_heads: int
    swa_head_dim: int
    swa_num_key_value_heads: int

    sliding_window_size: int
    d_rel: int

    rel_extent: int
    """Relative-bias extent on global layers; local use the window size."""

    q_bias: bool
    o_bias: bool

    log_scaling_alpha: float
    log_scaling_n_floor: int

    use_sconv: bool
    sconv_kernel_size: int

    local_layer_ids: tuple[int, ...]

    dense_mlp_idx: int
    """First MoE layer index."""

    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    dense_intermediate_size: int
    route_scale: float
    use_gate_bias: bool
    use_global_scale: bool
    norm_after_topk: bool
    gate_activation: str

    shared_expert_sink: bool
    """Shared experts never enter the top-k but count in the renormalization."""

    use_embed_norm: bool

    logits_mup_width_multiplier: float
    """muP width multiplier, which is a *divisor* of the final logits."""

    final_logit_softcapping: float | None

    def __post_init__(self) -> None:
        if self.gate_activation != "sigmoid":
            raise ValueError(
                f"unsupported gate_activation {self.gate_activation!r}; "
                "Inkling routes with sigmoid"
            )
        if disabled := [
            name for name in _REQUIRED_TEXT_FLAGS if not getattr(self, name)
        ]:
            raise ValueError(
                f"unsupported Inkling text_config: {disabled} are false, and "
                "the graph is built for every one of them being true"
            )
        if self.final_logit_softcapping is not None:
            raise ValueError(
                "unsupported Inkling final_logit_softcapping "
                f"{self.final_logit_softcapping}; the logits are uncapped"
            )
        if self.q_bias:
            raise ValueError(
                "unsupported Inkling q_bias; q, k, v and r come out of one "
                "StackedLinear, which gives all four a bias or none"
            )

    @property
    def num_local_layers(self) -> int:
        return len(self.local_layer_ids)

    @property
    def num_global_layers(self) -> int:
        return self.num_hidden_layers - self.num_local_layers

    def is_local_attention(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.dense_mlp_idx

    def num_heads_for_layer(self, layer_idx: int) -> int:
        return (
            self.swa_num_attention_heads
            if self.is_local_attention(layer_idx)
            else self.num_attention_heads
        )

    def num_kv_heads_for_layer(self, layer_idx: int) -> int:
        return (
            self.swa_num_key_value_heads
            if self.is_local_attention(layer_idx)
            else self.num_key_value_heads
        )

    def head_dim_for_layer(self, layer_idx: int) -> int:
        return (
            self.swa_head_dim
            if self.is_local_attention(layer_idx)
            else self.head_dim
        )

    def rel_extent_for_layer(self, layer_idx: int) -> int:
        return (
            self.sliding_window_size
            if self.is_local_attention(layer_idx)
            else self.rel_extent
        )

    def attention_window_for_layer(self, layer_idx: int) -> int | None:
        """MAX counts the query in the window; vLLM's FA4 passes size - 1."""
        return (
            self.sliding_window_size
            if self.is_local_attention(layer_idx)
            else None
        )

    def applies_log_scaling(self, layer_idx: int) -> bool:
        """Global only; the factor must scale both ``q`` and the rel bias."""
        return not self.is_local_attention(layer_idx)

    def qkvr_out_dims_for_layer(
        self, layer_idx: int
    ) -> tuple[int, int, int, int]:
        """q, k, v, r output widths; ``r`` is ``num_heads * d_rel`` wide."""
        num_heads = self.num_heads_for_layer(layer_idx)
        kv_width = self.num_kv_heads_for_layer(
            layer_idx
        ) * self.head_dim_for_layer(layer_idx)
        return (
            num_heads * self.head_dim_for_layer(layer_idx),
            kv_width,
            kv_width,
            num_heads * self.d_rel,
        )

    def kv_conv_dim_for_layer(self, layer_idx: int) -> int:
        return self.num_kv_heads_for_layer(layer_idx) * self.head_dim_for_layer(
            layer_idx
        )

    @classmethod
    def from_hf(cls, text_config: PretrainedConfig) -> InklingTextConfig:
        """Every field reads the identically-named ``text_config`` attribute."""
        if missing := [
            field.name
            for field in fields(cls)
            if not hasattr(text_config, field.name)
        ]:
            raise ValueError(
                f"unsupported Inkling text_config: missing {missing}; every "
                "field of InklingTextConfig reads the checkpoint attribute of "
                "the same name"
            )
        values = {
            field.name: getattr(text_config, field.name)
            for field in fields(cls)
        }
        values["local_layer_ids"] = tuple(values["local_layer_ids"])
        return cls(**values)


@dataclass(kw_only=True, frozen=True)
class InklingVisionConfig:
    """Vision-tower config; ``hmlp`` is an attention-free hierarchical MLP."""

    vision_encoder_type: str
    decoder_dmodel: int
    patch_size: int
    temporal_patch_size: int
    n_channels: int
    n_layers: int
    use_vision_norm: bool

    def __post_init__(self) -> None:
        if self.vision_encoder_type != "hmlp":
            raise ValueError(
                "unsupported Inkling vision_encoder_type "
                f"{self.vision_encoder_type!r}; the tower is hmlp"
            )
        if self.patch_size <= 1:
            raise ValueError(
                f"unsupported Inkling patch_size {self.patch_size}; the tower "
                "folds a patch down in prime-factor steps, which needs > 1"
            )

    @classmethod
    def from_hf(cls, vision_config: PretrainedConfig) -> InklingVisionConfig:
        return cls(
            vision_encoder_type=vision_config.vision_encoder_type,
            decoder_dmodel=vision_config.decoder_dmodel,
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            n_channels=vision_config.n_channels,
            n_layers=vision_config.n_layers,
            use_vision_norm=vision_config.use_vision_norm,
        )


@dataclass(kw_only=True)
class InklingConfig(ArchConfigWithKVCache):
    """Top-level Inkling config wrapping the text and vision backbones."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    devices: list[DeviceRef]

    dtype: DType

    kv_params: MultiKVCacheParams

    max_seq_len: int

    text_config: InklingTextConfig

    vision_config: InklingVisionConfig

    quant_config: QuantConfig | None = None
    """Set by :meth:`finalize` when the routed experts are packed FP4."""

    use_subgraphs: bool = True

    def __post_init__(self) -> None:
        # Caught here so a mismatched checkpoint fails at config time with the
        # field names, not as a shape error inside Model.execute.
        if self.vision_config.decoder_dmodel != self.text_config.hidden_size:
            raise ValueError(
                "Inkling vision tower emits rows of width "
                f"{self.vision_config.decoder_dmodel} "
                "(vision_config.decoder_dmodel), but the decoder embeds "
                f"tokens at width {self.text_config.hidden_size} "
                "(text_config.hidden_size); the two must match"
            )

    def get_kv_params(self) -> MultiKVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
    ) -> int:
        model_config = model_config or pipeline_config.model
        # Relative-bias attention has no position-embedding table, so the
        # checkpoint bounds its context with `model_max_length` where most
        # architectures use `max_position_embeddings`.
        context_length = getattr(
            huggingface_config.text_config, "model_max_length", None
        )
        if context_length is None:
            raise ValueError(
                "Unable to infer a context length for Inkling: its "
                "text_config declares no model_max_length."
            )
        try:
            return upper_bounded_default(
                upper_bound=context_length, default=model_config.max_length
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Inkling, the provided "
                f"max_length ({model_config.max_length}) exceeds the model's "
                f"context length ({context_length})."
            ) from e

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """One cache per attention flavor; they differ in KV head count."""
        text_config = InklingTextConfig.from_hf(huggingface_config.text_config)

        def params_for(
            n_kv_heads: int, head_dim: int, num_layers: int
        ) -> KVCacheParams:
            return kv_cache_config.to_params(
                dtype=cache_dtype,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                num_layers=num_layers,
                devices=devices,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
            )

        return MultiKVCacheParams.from_params(
            {
                GLOBAL_ATTENTION: params_for(
                    text_config.num_key_value_heads,
                    text_config.head_dim,
                    text_config.num_global_layers,
                ),
                LOCAL_ATTENTION: params_for(
                    text_config.swa_num_key_value_heads,
                    text_config.swa_head_dim,
                    text_config.num_local_layers,
                ),
            }
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for Inkling but could not be "
                f"loaded for '{model_config.model_path}'; ensure the model "
                "repository contains a valid config.json."
            )
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype_for_encoding(
                quantization_encoding, model_config.kv_cache.kv_cache_format
            ),
        )
        return cls(
            devices=device_refs,
            # bfloat16 even for NVFP4: FP4 covers only the routed experts.
            dtype=DType.bfloat16,
            kv_params=kv_params,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            text_config=InklingTextConfig.from_hf(
                huggingface_config.text_config
            ),
            vision_config=InklingVisionConfig.from_hf(
                huggingface_config.vision_config
            ),
            use_subgraphs=model_config.use_subgraphs,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
    ) -> None:
        """Sets :attr:`quant_config` from which routed tensors ship block
        scales; the released NVFP4 checkpoint leaves one MoE layer bfloat16."""
        quantized_layers = {
            int(match.group(1))
            for name in state_dict
            if (match := _EXPERT_BLOCK_SCALE.fullmatch(name))
        }
        if not quantized_layers:
            self.quant_config = None
            return

        # The shared parser reads packed NVFP4 (two values per byte) as uint8.
        quant_config = parse_quant_config(
            huggingface_config, state_dict, DType.uint8
        )
        if quant_config is None:
            raise ValueError(
                "Inkling routed experts carry NVFP4 block scales, but no "
                "quantization the shared parser recognizes; loading them into "
                "a bfloat16 graph would fail on the packed weights"
            )
        self.quant_config = replace(
            quant_config,
            mlp_quantized_layers=quantized_layers,
            # Only the routed experts are quantized; the rest stays bfloat16.
            attn_quantized_layers=set(),
            shared_experts_weight_dtype=DType.bfloat16,
            embedding_output_dtype=DType.bfloat16,
        )
