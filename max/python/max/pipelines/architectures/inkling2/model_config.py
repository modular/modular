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
"""Inkling 2 architecture config: text backbone, KV cache and expert formats."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    GraphRole,
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces import ArchConfigWithKVCache
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.registration_runtime import DeploymentConfig
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self

from .state_cache import ConvStateLayout
from .vision import Inkling2VisionConfig

_MAX_IMAGE_PIXELS = 64 * 2**20
"""The engine's cap on one image's source pixels."""

GLOBAL_ATTENTION = "full_attention"
LOCAL_ATTENTION = "sliding_attention"
STATE_CACHE_KEY = "conv_state"
"""The cache child holding every layer's convolution state."""


class ExpertFormat(Enum):
    """How a layer's routed experts are stored in the checkpoint."""

    BF16 = "bf16"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"


@dataclass(kw_only=True, frozen=True)
class Inkling2TextConfig:
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
    """Relative-bias extent on global layers; local layers use the window."""

    log_scaling_alpha: float
    log_scaling_n_floor: int
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float

    sconv_kernel_size: int
    local_layer_ids: tuple[int, ...]

    n_layers_per_attn_res_block: int
    attn_res_normalize_values: bool

    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    route_scale: float
    mlp_gate_softcap: float
    mlp_up_softcap: float

    final_logit_softcapping: float
    model_max_length: int

    @property
    def num_local_layers(self) -> int:
        """How many layers use sliding-window attention."""
        return len(self.local_layer_ids)

    @property
    def num_global_layers(self) -> int:
        """How many layers attend over the whole sequence."""
        return self.num_hidden_layers - self.num_local_layers

    def is_local_attention(self, layer_idx: int) -> bool:
        """Whether this layer uses sliding-window attention."""
        return layer_idx in self.local_layer_ids

    def is_block_boundary(self, layer_idx: int) -> bool:
        """Whether this layer's input starts a new attn_res block."""
        return layer_idx % self.n_layers_per_attn_res_block == 0

    def num_heads(self, is_local: bool) -> int:
        """The attention head count, for the local or global layers."""
        return (
            self.swa_num_attention_heads
            if is_local
            else self.num_attention_heads
        )

    def num_kv_heads(self, is_local: bool) -> int:
        """The key/value head count, for the local or global layers."""
        return (
            self.swa_num_key_value_heads
            if is_local
            else self.num_key_value_heads
        )

    def head_dim_for(self, is_local: bool) -> int:
        """The head width, for the local or global layers."""
        return self.swa_head_dim if is_local else self.head_dim

    def rel_extent_for(self, is_local: bool) -> int:
        """The relative-bias extent, for the local or global layers."""
        return self.sliding_window_size if is_local else self.rel_extent

    def attention_window(self, is_local: bool) -> int | None:
        """The sliding window, counting the query itself, or ``None``."""
        return self.sliding_window_size if is_local else None

    def kv_conv_dim(self, is_local: bool) -> int:
        """The width the K and V short convolutions read."""
        return self.num_kv_heads(is_local) * self.head_dim_for(is_local)

    @classmethod
    def from_hf(cls, text_config: PretrainedConfig) -> Inkling2TextConfig:
        """Reads the checkpoint's ``text_config``."""
        return cls(
            hidden_size=text_config.hidden_size,
            num_hidden_layers=text_config.num_hidden_layers,
            rms_norm_eps=text_config.rms_norm_eps,
            vocab_size=text_config.vocab_size,
            unpadded_vocab_size=text_config.unpadded_vocab_size,
            num_attention_heads=text_config.num_attention_heads,
            head_dim=text_config.head_dim,
            num_key_value_heads=text_config.num_key_value_heads,
            swa_num_attention_heads=text_config.swa_num_attention_heads,
            swa_head_dim=text_config.swa_head_dim,
            swa_num_key_value_heads=text_config.swa_num_key_value_heads,
            sliding_window_size=text_config.sliding_window_size,
            d_rel=text_config.d_rel,
            rel_extent=text_config.rel_extent,
            log_scaling_alpha=text_config.log_scaling_alpha,
            log_scaling_n_floor=text_config.log_scaling_n_floor,
            qk_l2_clip_threshold=text_config.qk_l2_clip_threshold,
            rel_l2_clip_threshold=text_config.rel_l2_clip_threshold,
            sconv_kernel_size=text_config.sconv_kernel_size,
            local_layer_ids=tuple(text_config.local_layer_ids),
            n_layers_per_attn_res_block=text_config.n_layers_per_attn_res_block,
            attn_res_normalize_values=text_config.attn_res_normalize_values,
            n_routed_experts=text_config.n_routed_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            intermediate_size=text_config.intermediate_size,
            route_scale=text_config.route_scale,
            mlp_gate_softcap=text_config.mlp_gate_softcap,
            mlp_up_softcap=text_config.mlp_up_softcap,
            final_logit_softcapping=text_config.final_logit_softcapping,
            model_max_length=text_config.model_max_length,
        )


@dataclass(kw_only=True)
class Inkling2Config(ArchConfigWithKVCache, DeploymentConfig):
    """Top-level Inkling 2 config for the registered decoder."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }
    weight_prefixes: ClassVar[Mapping[str, str]] = {GraphRole.VISION: "vision."}

    devices: list[DeviceRef]
    dtype: DType
    kv_params: MultiKVCacheParams
    max_seq_len: int
    text_config: Inkling2TextConfig
    vision_config: Inkling2VisionConfig
    expert_formats: tuple[ExpertFormat, ...] = ()
    """One entry per layer, set by :meth:`finalize` from the weights."""

    outputs: ModelOutputConfig = field(default_factory=ModelOutputConfig)
    use_subgraphs: bool = True

    def get_kv_params(self) -> MultiKVCacheParams:
        """The KV cache parameters the text config declares."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """The maximum sequence length the config allows."""
        return self.max_seq_len

    @staticmethod
    def calculate_max_seq_len(
        huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        """The checkpoint's context length, bounded by the config's."""
        # Relative-bias attention has no position-embedding table, so the
        # checkpoint bounds its context with `model_max_length`.
        context_length = getattr(
            huggingface_config.text_config, "model_max_length", None
        )
        if context_length is None:
            raise ValueError(
                "Unable to infer a context length for Inkling 2: its "
                "text_config declares no model_max_length."
            )
        try:
            return upper_bounded_default(
                upper_bound=context_length, default=model_config.max_length
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Inkling 2, the provided "
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
        """One cache per attention flavor, plus the conv state."""
        text_config = Inkling2TextConfig.from_hf(huggingface_config.text_config)

        def params_for(
            n_kv_heads: int,
            head_dim: int,
            num_layers: int,
            window_size: int | None = None,
        ) -> KVCacheParams:
            return kv_cache_config.to_params(
                dtype=cache_dtype,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                num_layers=num_layers,
                devices=devices,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
                num_draft_tokens=0,
                window_size=window_size,
            )

        attention = MultiKVCacheParams.from_params(
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
                    window_size=text_config.sliding_window_size,
                ),
            }
        )
        state = RecurrentStateParams(
            regions=ConvStateLayout.from_config(
                text_config, tp_size=len(devices)
            ).regions(),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )
        return MultiKVCacheParams.from_params(
            {**attention.children, STATE_CACHE_KEY: state}
        )

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType]:
        """One decoder-width bfloat16 row per image token."""
        return huggingface_config.text_config.hidden_size, DType.bfloat16

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Bytes of the largest image's embeddings.

        Inkling 2 never resizes, so the largest image is the engine's source
        cap, one token per ``patch_size * spatial_merge_size`` pixel block.
        """
        hidden_size, dtype = cls.get_vision_cache_row_spec(huggingface_config)
        vision = Inkling2VisionConfig.from_hf(
            huggingface_config.vision_config, text_hidden_size=hidden_size
        )
        block = vision.patch_size * vision.spatial_merge_size
        max_tokens = -(-_MAX_IMAGE_PIXELS // (block * block))
        return max_tokens * hidden_size * dtype.size_in_bytes

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the pipeline configuration."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for Inkling 2 but could not "
                f"be loaded for '{model_config.model_path}'; ensure the model "
                "repository contains a valid config.json."
            )
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        text_config = Inkling2TextConfig.from_hf(huggingface_config.text_config)
        return cls(
            devices=device_refs,
            # bfloat16 even for NVFP4 and MXFP8: they cover only the experts.
            dtype=DType.bfloat16,
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=model_config.kv_cache,
                cache_dtype=cache_dtype_for_encoding(
                    quantization_encoding, model_config.kv_cache.kv_cache_format
                ),
            ),
            max_seq_len=max_seq_len,
            text_config=text_config,
            vision_config=Inkling2VisionConfig.from_hf(
                huggingface_config.vision_config,
                text_hidden_size=text_config.hidden_size,
            ),
            use_subgraphs=model_config.use_subgraphs,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        outputs: ModelOutputConfig | None = None,
    ) -> None:
        """Reads each layer's expert format off its stored weight dtype.

        ``config.json``'s ``quantization_config`` names NVFP4 for the whole
        model, which is wrong for the MXFP8 layers, so it is not read.
        """
        del huggingface_config
        self.outputs = outputs or ModelOutputConfig()
        by_dtype = {
            DType.uint8: ExpertFormat.NVFP4,
            DType.float8_e4m3fn: ExpertFormat.MXFP8,
            DType.bfloat16: ExpertFormat.BF16,
        }
        self.expert_formats = tuple(
            by_dtype[state_dict[f"layers.{i}.mlp.experts.w13_weight"].dtype]
            for i in range(self.text_config.num_hidden_layers)
        )
