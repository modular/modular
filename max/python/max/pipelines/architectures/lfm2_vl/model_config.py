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
"""Config for Lfm2VlForConditionalGeneration (ModuleV3).

LiquidAI d1-3B / LFM2-VL: the LFM2 hybrid text decoder under the
SigLIP2 NaFlex vision tower and a 2x2-fold projector:

    input_ids -> embed_tokens -> [image rows scattered in at the
    image_token_id (124907) placeholders] -> 22 conv + 8 full-attention
    decoder layers -> embedding_norm (the FINAL norm, name historical)
    -> tied lm_head

    pixel rows [num_patches, 3 * 16 * 16]
      -> patch_embedding Linear(768 -> 1152) + interpolated learned
         position table (16x16 source grid, bilinear to each tile grid)
      -> 27 SigLIP2 encoder layers (LayerNorm-residual, bidirectional
         per-tile MHSA with biases, gelu_pytorch_tanh MLP)
      -> post_layernorm -> fold each 2x2 patch block -> linear_1 -> gelu
      -> linear_2 -> [tiles * tokens_per_tile, 2048]

The vision tower is NA-FLEX: the processor emits arbitrary per-tile
patch grids (spatial_shapes), not the fixed 16x16 the position table was
trained at, and the tower interpolates its 256-entry learned position
table bilinearly to each tile's grid. Tile grids stay even in both axes
because the processor resizes image dimensions to multiples of
encoder_patch_size * downsample_factor = 32 px, so the 2x2 fold is always
whole.

Encoder semantic layer_types (the scope's `qk_norm` verdict): the tower's
attention has no q/k norm and no rope; only the DECODER's full-attention
layers carry q/k layernorm weights ([64] each, the checkpoint says).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

ATTN_CACHE_KEY = "attn"
STATE_CACHE_KEY = "state"
"""The two children of the decoder's cache tree."""

CONV_LEAF_ID = "conv"
"""The decoder's single recurrent-state leaf: one conv window row per layer."""

ATTENTION_LAYER_TYPES = ("full_attention",)


def text_config_of(huggingface_config: AutoConfig) -> AutoConfig:
    """The decoder's sub-config of an LFM2-VL checkpoint."""
    text = getattr(huggingface_config, "text_config", None)
    if text is None:
        raise ValueError(
            "Lfm2VlForConditionalGeneration needs a nested `text_config`; "
            f"{type(huggingface_config).__name__} has none"
        )
    return text


def vision_config_of(huggingface_config: AutoConfig) -> AutoConfig:
    """The vision tower's sub-config of an LFM2-VL checkpoint."""
    vision = getattr(huggingface_config, "vision_config", None)
    if vision is None:
        raise ValueError(
            "Lfm2VlForConditionalGeneration needs a nested "
            f"`vision_config`; {type(huggingface_config).__name__} has none"
        )
    return vision


def _theme(value: object, key: str, default: object = None) -> object:
    """Reads ``key`` off a config object OR a dict (HF nests both forms)."""
    if hasattr(value, "get") and not isinstance(value, str):
        return value.get(key, default)
    return getattr(value, key, default)


@dataclass(kw_only=True)
class Lfm2VlConfig(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Model configuration for Lfm2VlForConditionalGeneration."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    # float32 is the check-gate dtype (the teacher-forced parity walk).
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
    }

    # MAX plumbing
    dtype: DType
    kv_params: MultiKVCacheParams
    devices: list[DeviceRef]
    quantization_encoding: SupportedEncoding | None = None

    # decoder (text_config)
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: tuple[str, ...]
    norm_eps: float
    tie_word_embeddings: bool
    max_seq_len: int

    rope_theta: float = 1_000_000.0
    head_dim: int | None = None
    conv_L_cache: int = 3
    conv_bias: bool = False
    max_position_embeddings: int = 32_768

    #: The tower's geometry belongs to the checkpoint's own sub-config:
    #: only `initialize()` may supply it, and the `vision_*` properties
    #: read it.
    vision: object | None = None

    # Projector / processor
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    projector_use_layernorm: bool = False
    projector_hidden_act: str = "gelu"
    downsample_factor: int = 2
    patch_size: int = 16
    num_channels: int = 3

    #: The ``image_token_id`` placeholder (124907 in the shipped
    #: checkpoint), read by the batch processor to find the token rows
    #: the tower's output replaces.
    image_token_id: int = -1

    # pipeline outputs
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.conv_bias:
            raise ValueError(
                "conv_bias=True is not supported (the shipped checkpoint "
                "ships conv_bias=false)"
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        for t in self.layer_types:
            if t not in ("conv", *ATTENTION_LAYER_TYPES):
                raise ValueError(f"unknown decoder layer type {t!r}")
        if not self.tie_word_embeddings:
            raise ValueError(
                "this port implements tied embeddings only (the shipped "
                "checkpoint ties); untied needs an lm_head branch"
            )
        if self.projector_use_layernorm:
            raise ValueError(
                "projector_use_layernorm is true in the config but this "
                "port implements the false branch the shipped checkpoint "
                "carries (projector_hidden_act=gelu, no LayerNorm)"
            )
        if self.projector_hidden_act != "gelu":
            raise ValueError(
                "only ACT2FN['gelu'] (the exact erf form) is implemented "
                f"for the projector, got {self.projector_hidden_act!r}"
            )

    # the tower's geometry, read off `vision`
    def _vision_field(self, key: str, default: object = None) -> Any:
        if self.vision is None:
            raise ValueError(
                "this config carries no `vision` sub-config, so it can only "
                "drive the DECODER graph: build it with "
                "Lfm2VlConfig.initialize(), which reads the checkpoint's "
                "`vision_config`"
            )
        return _theme(self.vision, key, default)

    @property
    def vision_hidden_size(self) -> int:
        """The SigLIP2 tower's hidden width."""
        return int(self._vision_field("hidden_size"))

    @property
    def vision_intermediate_size(self) -> int:
        """The SigLIP2 MLP's intermediate width."""
        return int(self._vision_field("intermediate_size"))

    @property
    def vision_num_hidden_layers(self) -> int:
        """The tower's encoder layer count."""
        return int(self._vision_field("num_hidden_layers"))

    @property
    def vision_num_attention_heads(self) -> int:
        """The tower's attention head count."""
        return int(self._vision_field("num_attention_heads"))

    @property
    def vision_layer_norm_eps(self) -> float:
        """The tower's LayerNorm epsilon."""
        return float(self._vision_field("layer_norm_eps", 1e-6))

    @property
    def vision_hidden_act(self) -> str:
        """The tower's MLP activation."""
        return str(self._vision_field("hidden_act", "gelu_pytorch_tanh"))

    @property
    def vision_use_head(self) -> bool:
        """Whether the reference carries a SigLIP2 head after the tower."""
        return bool(self._vision_field("vision_use_head", False))

    @property
    def vision_attention_dropout(self) -> float:
        """The tower's attention dropout rate."""
        return float(self._vision_field("attention_dropout", 0.0))

    @property
    def vision_num_patches(self) -> int:
        """Rows in the learned position table (256 = a 16x16 source grid)."""
        return int(self._vision_field("num_patches", 256))

    @property
    def vision_position_grid(self) -> int:
        """The learned position table's grid side (sqrt of its rows)."""
        return int(self.vision_num_patches**0.5)

    @property
    def vision_head_dim(self) -> int:
        """The tower's per-head attention width."""
        return self.vision_hidden_size // self.vision_num_attention_heads

    @property
    def vision_patch_dim(self) -> int:
        """The flat per-patch row width (patch x patch x channels)."""
        return self.num_channels * self.patch_size * self.patch_size

    # counts
    @property
    def num_attn_layers(self) -> int:
        """The KV cache's attention layer count."""
        return sum(1 for t in self.layer_types if t in ATTENTION_LAYER_TYPES)

    @property
    def num_conv_layers(self) -> int:
        """The state pool's conv layer count."""
        return sum(1 for t in self.layer_types if t == "conv")

    def get_kv_params(self) -> MultiKVCacheParams:
        """The arch's KV cache shape parameters."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """The configured sequence-length ceiling."""
        return self.max_seq_len

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """The sequence-length ceiling implied by the cache budget."""
        return upper_bounded_default(
            upper_bound=text_config_of(
                huggingface_config
            ).max_position_embeddings,
            default=model_config.max_length,
        )

    # cache tree
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
        """The decoder's paged attention cache beside its conv state pool.

        The tower takes NO cache (its attention runs over one request's
        patches through the cache-less ragged kernel), so the tree has
        "attn" for the 8 full-attention layers, "state" with one
        causal_conv1d_varlen_fwd window row ([hidden, conv_L_cache - 1],
        mutated in place) per conv layer.
        """
        text = text_config_of(huggingface_config)
        layer_types = tuple(text.layer_types)
        n_attn = sum(1 for t in layer_types if t in ATTENTION_LAYER_TYPES)
        n_conv = sum(1 for t in layer_types if t == "conv")
        head_dim = getattr(text, "head_dim", None) or (
            text.hidden_size // text.num_attention_heads
        )
        attn = kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=text.num_key_value_heads,
            head_dim=head_dim,
            num_layers=n_attn,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=None,
            num_draft_tokens=0,
        )
        state = RecurrentStateParams(
            regions=(
                RecurrentStateRegion(
                    leaf_id=CONV_LEAF_ID,
                    num_layers=n_conv,
                    # causal_conv1d_varlen_fwd's pool contract:
                    # [slots, dim, kernel - 1], mutated in place.
                    row_shape=(
                        text.hidden_size,
                        text.conv_L_cache - 1,
                    ),
                    dtype=cache_dtype,
                ),
            ),
            devices=attn.devices,
            data_parallel_degree=attn.data_parallel_degree,
        )
        return MultiKVCacheParams.from_params(
            {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Cache layers for the offline tools: decoder attention only."""
        text = text_config_of(huggingface_config)
        return sum(1 for t in text.layer_types if t in ATTENTION_LAYER_TYPES)

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        """The decoder's per-head attention width."""
        text = text_config_of(huggingface_config)
        return getattr(text, "head_dim", None) or (
            text.hidden_size // text.num_attention_heads
        )

    @staticmethod
    def _rope_theta(text: AutoConfig) -> float:
        # transformers 5.x normalizes rope fields into rope_parameters
        # (the shipped config: {"rope_theta": 1e6, "rope_type": "default"}).
        rp = getattr(text, "rope_parameters", None)
        if not isinstance(rp, dict):
            return float(getattr(text, "rope_theta", 10_000.0))
        rope_type = rp.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"only default rope_type supported, got {rope_type}"
            )
        theta = rp.get("rope_theta")
        if theta is None:
            # A rope_parameters dict with no theta is the apertus trap
            # in its 5.x shape: normalization ran, so a silent 10_000.0
            # would serve a wrong base to every layer. Refuse.
            raise ValueError(
                "rope_parameters carries no rope_theta: the config's "
                "normalized rope block omits the base, and serving a "
                "silent default here produced a wrong-rope parity "
                "failure before (the apertus lesson) — pass the real "
                "base through the config instead"
            )
        return float(theta)

    @staticmethod
    def _intermediate_size(text: AutoConfig, declared: int) -> int:
        # LFM2's SwiGLU sizing rule, kept exactly (monorepo lfm2
        # `_resolve_intermediate_size`): inactive for the shipped
        # checkpoint (block_auto_adjust_ff_dim=false -> 10752 as declared).
        if not bool(getattr(text, "block_auto_adjust_ff_dim", False)):
            return declared
        hidden = int(2 * declared / 3)
        multiplier = getattr(text, "block_ffn_dim_multiplier", None)
        if multiplier is not None:
            hidden = int(float(multiplier) * hidden)
        multiple_of = int(getattr(text, "block_multiple_of", 1))
        if multiple_of > 1:
            hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        return hidden

    # build
    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Build the arch config from the pipeline and checkpoint configs."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config required for "
                f"'{model_config.model_path}' but could not be loaded."
            )
        text = text_config_of(huggingface_config)
        vision = vision_config_of(huggingface_config)
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        return cls(
            dtype=dtype,
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            devices=device_refs,
            quantization_encoding=quantization_encoding,
            vocab_size=int(text.vocab_size),
            hidden_size=int(text.hidden_size),
            intermediate_size=cls._intermediate_size(
                text, int(text.intermediate_size)
            ),
            num_hidden_layers=int(text.num_hidden_layers),
            num_attention_heads=int(text.num_attention_heads),
            num_key_value_heads=int(text.num_key_value_heads),
            layer_types=tuple(text.layer_types),
            norm_eps=float(text.norm_eps),
            tie_word_embeddings=bool(
                getattr(text, "tie_word_embeddings", True)
            ),
            max_seq_len=max_seq_len,
            rope_theta=cls._rope_theta(text),
            head_dim=cls.get_head_dim(huggingface_config),
            conv_L_cache=int(text.conv_L_cache),
            conv_bias=bool(getattr(text, "conv_bias", False)),
            max_position_embeddings=int(
                getattr(text, "max_position_embeddings", 32_768)
            ),
            # The tower's geometry travels as the checkpoint's own
            # sub-config; the `vision_*` properties read it.
            vision=vision,
            projector_hidden_size=int(
                getattr(huggingface_config, "projector_hidden_size", 2048)
            ),
            projector_bias=bool(
                getattr(huggingface_config, "projector_bias", True)
            ),
            projector_use_layernorm=bool(
                getattr(huggingface_config, "projector_use_layernorm", False)
            ),
            projector_hidden_act=str(
                getattr(huggingface_config, "projector_hidden_act", "gelu")
            ),
            downsample_factor=int(
                getattr(huggingface_config, "downsample_factor", 2)
            ),
            patch_size=int(getattr(vision, "patch_size", 16)),
            num_channels=int(getattr(vision, "num_channels", 3)),
            image_token_id=int(
                getattr(huggingface_config, "image_token_id", -1)
            ),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, Any],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        """Checks the tower's fixed choices, then stores the output modes.

        `vision_use_head=True` or a nonzero attention_dropout would need
        paths this port does not implement; the shipped checkpoint ships
        `vision_use_head=false` and `attention_dropout 0.0`.
        """
        del huggingface_config, state_dict
        if self.vision is None:
            self.return_logits = return_logits
            self.return_hidden_states = return_hidden_states
            return
        if self.vision_use_head:
            raise ValueError(
                "vision_use_head=true: the shipped checkpoint ships "
                "false, and this port has no pooling head"
            )
        if self.vision_attention_dropout:
            raise ValueError(
                "attention_dropout != 0 in vision_config: this port "
                "serves eval dropout-free only"
            )
        if self.vision_hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                "only gelu_pytorch_tanh (the tanh approximation) is "
                "implemented for the tower's MLP, got "
                f"{self.vision_hidden_act!r}"
            )
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
