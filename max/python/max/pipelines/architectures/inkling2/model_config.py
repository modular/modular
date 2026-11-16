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
"""The Inkling2 config, KV tree and weight prefixes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar, Final

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces import ArchConfigWithKVCache
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.registration_runtime import DeploymentConfig
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self

from .state_cache import conv_state_regions

GLOBAL_ATTENTION = "full_attention"
LOCAL_ATTENTION = "sliding_attention"
STATE_CACHE_KEY = "conv_state"
"""The child holding every layer's convolution state."""


class ExpertFormat(Enum):
    """How a layer's routed experts are stored in the checkpoint."""

    BF16 = "bf16"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"


FULL_ATTENTION_LAYER_TYPE: Final = "hybrid"
SLIDING_LAYER_TYPE: Final = "hybrid_sliding"
"""The two values of the reference's ``layer_types``."""

FULL_ATTENTION_PERIOD: Final = 6
"""One full-attention layer every sixth, per the reference's default."""

FULL_ATTENTION_ATTN_RES_QUERIES: Final = 3
"""Queries a full-attention layer's block residual bank asks for."""

DERIVED_TEXT_FIELDS: Final = frozenset({"local_layer_ids", "use_embed_norm"})
"""Fields :meth:`Inkling2TextConfig.from_hf` derives instead of reading."""


def _local_layer_ids(text_config: PretrainedConfig) -> tuple[int, ...]:
    num_layers = int(text_config.num_hidden_layers)
    declared = getattr(text_config, "local_layer_ids", None)
    if declared is not None and all(0 <= int(i) < num_layers for i in declared):
        # The checkpoint states its sliding-window layers by id. Trust the
        # list only when every entry indexes this config's layers: a slice
        # derived from a larger checkpoint can carry the parent's list
        # verbatim, and that list must fall through to the patterns below.
        return tuple(int(i) for i in declared)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        # configuration_inkling2.py:147-152: `"hybrid" if (i + 1) % 6 == 0`.
        return tuple(
            i for i in range(num_layers) if (i + 1) % FULL_ATTENTION_PERIOD
        )
    if len(layer_types) != num_layers:
        raise ValueError(
            f"unsupported Inkling2 layer_types: {len(layer_types)} entries "
            f"for num_hidden_layers {num_layers}"
        )
    known = {FULL_ATTENTION_LAYER_TYPE, SLIDING_LAYER_TYPE}
    if unknown := sorted(set(layer_types) - known):
        raise ValueError(
            f"unsupported Inkling2 layer_types {unknown}; the graph builds "
            f"only {sorted(known)}"
        )
    return tuple(
        i for i, kind in enumerate(layer_types) if kind == SLIDING_LAYER_TYPE
    )


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
    """Relative-bias extent on global layers; local use the window size."""

    log_scaling_alpha: float
    log_scaling_n_floor: int
    """Position from which queries and relative states scale up."""

    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float

    sconv_kernel_size: int

    use_embed_norm: bool = True
    """The checkpoint always ships ``embed_norm.weight``, so this is True for
    every Inkling2 checkpoint."""

    n_layers_per_attn_res_block: int
    """Decoder layers summed into one block-attention-residual source."""

    attn_res_normalize_values: bool
    """Carried from the checkpoint; the block residual normalizes either way
    (``modeling_inkling2.py:743-744``), so no port behaviour depends on it."""

    n_routed_experts: int
    num_experts_per_tok: int

    intermediate_size: int
    """Expert hidden width, the checkpoint's ``moe_intermediate_size``."""

    route_scale: float
    mlp_gate_softcap: float
    mlp_up_softcap: float
    final_logit_softcapping: float

    local_layer_ids: tuple[int, ...]
    """Sliding-window layers, from the config's layer declarations."""

    def __post_init__(self) -> None:
        if bad := [
            i
            for i in self.local_layer_ids
            if not 0 <= i < self.num_hidden_layers
        ]:
            raise ValueError(
                f"unsupported Inkling2 local_layer_ids {bad}; every entry "
                f"must index one of the {self.num_hidden_layers} layers"
            )
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids):
            raise ValueError(
                f"unsupported Inkling2 local_layer_ids {self.local_layer_ids}"
                "; a layer is local or full, never both"
            )
        if self.unpadded_vocab_size > self.vocab_size:
            raise ValueError(
                "unsupported Inkling2 unpadded_vocab_size "
                f"{self.unpadded_vocab_size} > vocab_size {self.vocab_size}; "
                "the logits are sliced to the unpadded width"
            )
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError(
                "unsupported Inkling2 num_experts_per_tok "
                f"{self.num_experts_per_tok} > n_routed_experts "
                f"{self.n_routed_experts}"
            )

    @property
    def num_local_layers(self) -> int:
        """Number of sliding-window layers."""
        return len(self.local_layer_ids)

    @property
    def num_global_layers(self) -> int:
        """Number of full-attention layers."""
        return self.num_hidden_layers - self.num_local_layers

    def is_local_attention(self, layer_idx: int) -> bool:
        """Whether this layer uses sliding-window attention."""
        return layer_idx in self.local_layer_ids

    def num_heads(self, is_local: bool) -> int:
        """Query heads on a layer of this flavor."""
        return (
            self.swa_num_attention_heads
            if is_local
            else self.num_attention_heads
        )

    def num_kv_heads(self, is_local: bool) -> int:
        """Key/value heads of a layer of this flavor."""
        return (
            self.swa_num_key_value_heads
            if is_local
            else self.num_key_value_heads
        )

    def head_dim_for(self, is_local: bool) -> int:
        """Head width of a layer of this flavor."""
        return self.swa_head_dim if is_local else self.head_dim

    def rel_extent_for(self, is_local: bool) -> int:
        """Relative-bias extent of a layer of this flavor."""
        return self.sliding_window_size if is_local else self.rel_extent

    def attention_window(self, is_local: bool) -> int | None:
        """MAX counts the query in the window; vLLM's FA4 passes size - 1."""
        return self.sliding_window_size if is_local else None

    def applies_log_scaling(self, is_local: bool) -> bool:
        """Global only; the factor scales both ``q`` and the rel states."""
        return not is_local

    def attn_scale(self, is_local: bool) -> float:
        """Softmax scale: ``head_dim**-0.5``, not Inkling v1's muP."""
        return self.head_dim_for(is_local) ** -0.5

    def qkvr_out_dims(self, is_local: bool) -> tuple[int, int, int, int]:
        """q, k, v, r output widths; ``r`` is ``num_heads * d_rel`` wide."""
        num_heads = self.num_heads(is_local)
        head_dim = self.head_dim_for(is_local)
        kv_width = self.num_kv_heads(is_local) * head_dim
        return (
            num_heads * head_dim,
            kv_width,
            kv_width,
            num_heads * self.d_rel,
        )

    def kv_conv_dim(self, is_local: bool) -> int:
        """Channels the k/v short convolution runs over."""
        return self.num_kv_heads(is_local) * self.head_dim_for(is_local)

    def attn_gate_dim(self, is_local: bool) -> int:
        """Width of the attention output gate and of the projection it gates."""
        return self.num_heads(is_local) * self.head_dim_for(is_local)

    def num_sinks(self, is_local: bool) -> int:
        """One sink logit per head: one extra zero-valued logit in the softmax."""
        return self.num_heads(is_local)

    def is_attn_res_block_boundary(self, layer_idx: int) -> bool:
        """Whether this layer closes a block-residual window."""
        return layer_idx % self.n_layers_per_attn_res_block == 0

    def attention_input_queries(self, layer_idx: int) -> int:
        """Queries this layer's attention input reads from its bank."""
        if layer_idx == 0:
            return 0
        return (
            1
            if self.is_local_attention(layer_idx)
            else FULL_ATTENTION_ATTN_RES_QUERIES
        )

    def attention_input_bank(self, layer_idx: int) -> int | None:
        """``attn_res.attn_queries`` bank this layer's attention input reads."""
        return None if layer_idx == 0 else layer_idx - 1

    @classmethod
    def from_hf(cls, text_config: PretrainedConfig) -> Inkling2TextConfig:
        """Every field reads the identically-named ``text_config`` attribute."""
        names = [
            field.name
            for field in fields(cls)
            if field.name not in DERIVED_TEXT_FIELDS
        ]
        if missing := [
            name for name in names if not hasattr(text_config, name)
        ]:
            raise ValueError(
                f"unsupported Inkling2 text_config: missing {missing}; every "
                "field of Inkling2TextConfig reads the checkpoint attribute "
                "of the same name"
            )
        values = {name: getattr(text_config, name) for name in names}
        values["local_layer_ids"] = _local_layer_ids(text_config)
        values["use_embed_norm"] = True
        # The router's activation is a config knob the graph hardcodes
        # (modeling_inkling2.py:703-716 scores with `ACT2FN[scoring_func]`).
        scoring = getattr(text_config, "scoring_func", "sigmoid")
        if scoring != "sigmoid":
            raise ValueError(
                f"unsupported Inkling2 scoring_func {scoring!r}; the port's "
                "router scores with sigmoid"
            )
        return cls(**values)


@dataclass(kw_only=True, frozen=True)
class Inkling2VisionConfig:
    """Vision tower: patch embedding, pre-norm blocks, one merger MLP."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int

    intermediate_size: int
    """MLP width: ``mlp_ratio * hidden_size`` when the checkpoint omits it."""

    mlp_ratio: float
    patch_size: int
    num_channels: int
    posemb_grid_size: int
    spatial_merge_size: int
    projection_hidden_dim: int
    projector_bias: bool
    text_hidden_size: int
    layer_norm_eps: float

    @property
    def patch_elements(self) -> int:
        """Input width of the patch embedding: one frame of one patch."""
        return self.num_channels * self.patch_size**2

    @property
    def merged_hidden_size(self) -> int:
        """Width of one merged row: ``spatial_merge_size**2`` patches."""
        return self.hidden_size * self.spatial_merge_size**2

    @classmethod
    def from_hf(cls, vision_config: PretrainedConfig) -> Inkling2VisionConfig:
        """Reads ``vision_config``, aliases included."""
        hidden_size = int(vision_config.hidden_size)
        mlp_ratio = float(getattr(vision_config, "mlp_ratio", 4.0))
        patch_size = getattr(vision_config, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(vision_config, "native_patch_size", None)
        if patch_size is None:
            raise ValueError(
                "unsupported Inkling2 vision_config: neither patch_size nor "
                "native_patch_size is declared"
            )
        merge_size = getattr(vision_config, "spatial_merge_size", None)
        if merge_size is None:
            merge_size = getattr(vision_config, "adapter_merge_size", None)
        if merge_size is None:
            raise ValueError(
                "unsupported Inkling2 vision_config: neither "
                "spatial_merge_size nor adapter_merge_size is declared"
            )
        # The reference derives it when absent (configuration_inkling2.py:176).
        intermediate_size = getattr(vision_config, "intermediate_size", None)
        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=int(vision_config.num_hidden_layers),
            num_attention_heads=int(vision_config.num_attention_heads),
            intermediate_size=int(
                intermediate_size or round(hidden_size * mlp_ratio)
            ),
            mlp_ratio=mlp_ratio,
            patch_size=int(patch_size),
            num_channels=int(getattr(vision_config, "num_channels", 3)),
            posemb_grid_size=int(vision_config.posemb_grid_size),
            spatial_merge_size=int(merge_size),
            projection_hidden_dim=int(vision_config.projection_hidden_dim),
            projector_bias=bool(
                getattr(vision_config, "projector_bias", False)
            ),
            text_hidden_size=int(vision_config.text_hidden_size),
            layer_norm_eps=float(vision_config.layer_norm_eps),
        )


@dataclass(kw_only=True)
class Inkling2Config(ArchConfigWithKVCache, DeploymentConfig):
    """Top-level Inkling2 config wrapping the text and vision backbones."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}
    """bfloat16 compute. The experts' storage format (bfloat16, NVFP4 or
    MXFP8) is per layer and detected from the checkpoint, not an encoding."""

    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "language_model.",
        GraphRole.VISION: "vision.",
    }
    """The prefix each role's weights carry out of the weight rules.

    ``language_model.`` is the decoder attribute the whole-network class holds
    the decoder under, so compiling that class needs no rewriting; serving
    strips the same prefix (``module_registration.py:1929-1937``) before
    compiling the role's graph, whose own module is name-transparent.
    """

    devices: list[DeviceRef]

    dtype: DType

    kv_params: MultiKVCacheParams

    max_seq_len: int

    text_config: Inkling2TextConfig

    vision_config: Inkling2VisionConfig

    image_token_id: int = 200054
    """Token the processor places where one image feature is spliced in."""

    max_batch_tokens: int = 512
    """Rows one forward step carries, from the scheduler's own budget.

    ``runtime.max_batch_input_tokens`` bounds a step's active rows (chunked
    prefill included), and the attention's cached-prefix read walks the query
    axis in bounded chunks whose count this sizes. A step wider than this
    would index past the rows it built, so the number is the scheduler's
    contract rather than a hint.
    """

    quant_config: QuantConfig | None = None
    """Always ``None``: expert quantization is a storage format the MoE layer
    handles itself, not an engine encoding."""

    expert_formats: tuple[ExpertFormat, ...] = ()
    """Each decoder layer's routed-expert storage format, detected from the
    checkpoint's weight dtypes in :meth:`finalize`."""

    use_subgraphs: bool = False
    """Stage the decoder layer as a reusable graph function.

    Off: the cached-prefix read in :mod:`layers.attention` types a tensor whose
    key axis is ``max_num_pages * page_size``, an algebraic dim, and the module
    compiler refuses to parameterize a graph function by one (``Expressions
    are not (currently) allowed in function signatures``, once per layer-group
    subgraph).
    """

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """What the graph emits; :meth:`finalize` sets it from the deployment."""

    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    """What the graph returns beside the logits. ``Inkling`` supports only
    ``NONE`` and refuses the alternatives."""

    def __post_init__(self) -> None:
        # Callers may pass transformers `PretrainedConfig` sub-configs rather
        # than the parsed dataclasses. Convert here so every caller gets one
        # shape.
        if isinstance(self.text_config, PretrainedConfig):
            object.__setattr__(
                self,
                "text_config",
                Inkling2TextConfig.from_hf(self.text_config),
            )
        if isinstance(self.vision_config, PretrainedConfig):
            object.__setattr__(
                self,
                "vision_config",
                Inkling2VisionConfig.from_hf(self.vision_config),
            )
        # The reference validates the same pair (configuration_inkling2.py:
        # 213-218); caught here so a mismatched checkpoint fails at config time
        # with the field names, not as a shape error inside Model.execute.
        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                "Inkling2 vision merger emits rows of width "
                f"{self.vision_config.text_hidden_size} "
                "(vision_config.text_hidden_size), but the decoder embeds "
                f"tokens at width {self.text_config.hidden_size} "
                "(text_config.hidden_size); the two must match"
            )

    def get_kv_params(self) -> MultiKVCacheParams:
        """The KV cache tree this model declares."""
        return self.kv_params

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        """Cache head width, so the graph build gets an MHA page tree."""
        return int(huggingface_config.text_config.head_dim)

    def get_max_seq_len(self) -> int:
        """The serving context length."""
        return self.max_seq_len

    @staticmethod
    def calculate_max_seq_len(
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        # Relative-bias attention has no position-embedding table, so the
        # checkpoint bounds its context with a config length where most
        # architectures use `max_position_embeddings`; the reference aliases
        # `model_max_length` to it (configuration_inkling2.py:96-103).
        """Resolve the context length from the checkpoint or the pipeline config."""
        text_config = huggingface_config.text_config
        context_length = getattr(
            text_config, "max_position_embeddings", None
        ) or getattr(text_config, "model_max_length", None)
        if context_length is None:
            raise ValueError(
                "Unable to infer a context length for Inkling2: its "
                "text_config declares neither max_position_embeddings nor "
                "model_max_length."
            )
        try:
            return upper_bounded_default(
                upper_bound=context_length, default=model_config.max_length
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Inkling2, the provided "
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
        """One cache per attention flavor, plus the conv-state child."""
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
            regions=conv_state_regions(text_config, tp_size=len(devices)),
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
        """Build the config from the pipeline config."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for Inkling2 but could not be "
                f"loaded for '{model_config.model_path}'; ensure the model "
                "repository contains a valid config.json."
            )
        if missing := [
            key
            for key in ("text_config", "vision_config")
            if not hasattr(huggingface_config, key)
        ]:
            raise ValueError(
                "unsupported Inkling2 config: the architecture needs both a "
                f"text_config and a vision_config; this checkpoint lacks "
                f"{missing}"
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
            # The compute dtype. A quantized checkpoint still computes in
            # bfloat16: NVFP4 and MXFP8 cover only the experts' storage.
            dtype=DType.bfloat16,
            kv_params=kv_params,
            max_seq_len=max_seq_len,
            text_config=Inkling2TextConfig.from_hf(
                huggingface_config.text_config
            ),
            vision_config=Inkling2VisionConfig.from_hf(
                huggingface_config.vision_config
            ),
            image_token_id=int(
                getattr(huggingface_config, "image_token_id", 200054)
            ),
            max_batch_tokens=int(
                getattr(
                    getattr(pipeline_config, "runtime", None),
                    "max_batch_input_tokens",
                    512,
                )
            ),
            use_subgraphs=model_config.use_subgraphs,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        return_logits: ReturnLogits | ModelOutputConfig | None = None,
        *,
        return_hidden_states: ReturnHiddenStates | None = None,
    ) -> None:
        """Records the deployment outputs and each layer's expert format.

        The format comes from the renamed state dict's weight dtypes, not
        from ``config.json``: its ``quantization_config`` names NVFP4 for
        the whole model, which the last eight MXFP8 layers contradict.
        """
        if isinstance(return_logits, ModelOutputConfig):
            requested = return_logits
        else:
            requested = ModelOutputConfig(
                return_logits=(
                    self.return_logits
                    if return_logits is None
                    else return_logits
                ),
                return_hidden_states=(
                    self.return_hidden_states
                    if return_hidden_states is None
                    else return_hidden_states
                ),
            )
        self.return_logits = requested.return_logits
        self.return_hidden_states = requested.return_hidden_states
        self.quant_config = None
        by_dtype = {
            DType.uint8: ExpertFormat.NVFP4,
            DType.float8_e4m3fn: ExpertFormat.MXFP8,
            DType.bfloat16: ExpertFormat.BF16,
        }
        prefix = type(self).weight_prefixes[GraphRole.DECODER]
        formats: list[ExpertFormat] = []
        for layer in range(self.text_config.num_hidden_layers):
            name = f"{prefix}layers.{layer}.mlp.experts.w13_weight"
            if name not in state_dict:
                raise ValueError(
                    f"unsupported Inkling2 checkpoint: layer {layer} ships "
                    f"no {name!r} tensor; every decoder layer routes to "
                    "experts"
                )
            dtype = state_dict[name].dtype
            if dtype not in by_dtype:
                raise ValueError(
                    f"unsupported Inkling2 expert dtype {dtype} on "
                    f"{name!r}; the port serves bfloat16, packed-NVFP4 "
                    "(uint8) and MXFP8 (float8_e4m3fn) experts"
                )
            formats.append(by_dtype[dtype])
        self.expert_formats = tuple(formats)


# The names the sibling modules import.
InklingTextConfig = Inkling2TextConfig
InklingVisionConfig = Inkling2VisionConfig
InklingConfig = Inkling2Config
