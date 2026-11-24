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
# Llama3Config and create_rope_embedding are copied from
# max/python/max/pipelines/architectures/llama3/model_config.py
# at main 4df8111afba.
"""Config for the DFlash2 draft of a Qwen3.5 target.

The draft HF config is the v1 ``dflash_config`` shape plus four keys v1 never
had (``conv_kernel_size``, ``conv_group_size``, ``selector_rank``,
``selector_top_k``), so the v1 parser is reused for the shared keys and
extended here rather than widened in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.legacy.rotary_embedding import (
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    RotaryEmbedding,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.quant_config import QuantConfig
from max.nn.rope_scaling import Llama3RopeScalingParams, LongRoPEScalingParams
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    parse_quant_config,
)
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
    supported_encoding_quantization,
)
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config
from max.pipelines.weights import gptq_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override

__all__ = [
    "DFlash2Qwen3_5Config",
    "Dflash2DraftHFConfig",
    "Llama3Config",
    "create_rope_embedding",
    "dflash2_draft_width",
    "parse_dflash2_draft_hf_config",
]

DRAFT_SLIDING_WINDOW = 2048
"""Positions the drafter can attend to, from the checkpoint's config.

The draft KV leaf is bounded at this rather than at the target's context: the
drafter's mask is windowed, so nothing older is ever read, and the leaf's cost
becomes a per-slot constant instead of scaling with ``--max-length``.
"""


def create_rope_embedding(
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float,
    max_seq_len: int,
    interleaved_rope_weights: bool,
    rope_scaling_params: Llama3RopeScalingParams | None,
    longrope_scaling_params: LongRoPEScalingParams | None,
    device: DeviceRef,
) -> RotaryEmbedding:
    """Create appropriate RoPE embedding based on scaling parameters.

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        rope_theta: RoPE theta parameter (typically 10000.0)
        max_seq_len: Maximum sequence length
        interleaved_rope_weights: Whether to use interleaved RoPE weights
        rope_scaling_params: Llama3 RoPE scaling parameters (if any)
        longrope_scaling_params: LongRoPE scaling parameters (if any)
        device: Device to place tensors on

    Returns:
        Configured RoPE embedding instance
    """
    if longrope_scaling_params is not None:
        return LongRoPERotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            interleaved=interleaved_rope_weights,
            scaling_params=longrope_scaling_params,
        )
    else:
        return Llama3RotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            interleaved=interleaved_rope_weights,
            scaling_params=rope_scaling_params,
        )


@dataclass(kw_only=True)
class Llama3Config(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Model configuration for Llama3 graph construction/execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "q4_k"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "gptq",
        "q4_k",
        "q4_0",
        "q6_k",
        "float32",
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    }

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    rope_scaling_params: Llama3RopeScalingParams | None
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    model_quantization_encoding: QuantizationEncoding | None
    quantization_config: QuantizationConfig | None
    kv_params: KVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    norm_dtype: DType | None = None
    attention_bias: bool = False
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool = False
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    devices: list[DeviceRef]
    clip_qkv: float | None
    quant_config: QuantConfig | None = None
    longrope_scaling_params: LongRoPEScalingParams | None = None
    logits_scaling: float = 1.0
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    target_layer_ids: list[int] | None = None
    use_subgraphs: bool = True
    data_parallel_degree: int = 1
    sliding_window: int | None = None
    quantization_encoding: SupportedEncoding | None = None

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
    ) -> float:
        """The attention multiplier is a scalar that scales the attention scores.

        It is used to control the variance of the attention scores.

        This function is used to get the attention multiplier from the
        huggingface config. If the attention multiplier is not set, it will be
        calculated as the square root of 1.0 divided by the head dimension.
        """
        # Base attention multiplier
        base_multiplier = getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(Llama3Config.get_head_dim(huggingface_config))
            ),
        )
        # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.

        return base_multiplier

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
    ) -> KVCacheParams:
        """Grouped-attention KV with EAGLE draft-token count when speculative is on.

        ``allow_kv_head_replication`` lets subclasses with fewer KV heads
        than the tensor-parallel width replicate each head across a device
        group.
        """
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            num_layers=cls.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=pipeline_config.speculative.speculative_method
            if pipeline_config.speculative
            else None,
            num_draft_tokens=(
                pipeline_config.speculative.num_speculative_tokens or 0
            )
            if pipeline_config.speculative
            else 0,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """The checkpoint's hidden layer count."""
        return huggingface_config.num_hidden_layers

    @override
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
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
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
        """Builds the config from the pipeline and HF configs."""
        model_config = model_config or pipeline_config.model
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        device_specs = model_config.device_specs
        n_devices = len(device_specs)

        interleaved_rope_weights = _interleaved_rope_weights(model_config)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in device_specs[:n_devices]
        ]

        embedding_multiplier = getattr(
            huggingface_config, "embedding_multiplier", 1.0
        )
        residual_multiplier = getattr(
            huggingface_config, "residual_multiplier", 1.0
        )
        rope_scaling_params: Llama3RopeScalingParams | None = None
        longrope_scaling_params: LongRoPEScalingParams | None = None
        rope_scaling = getattr(huggingface_config, "rope_scaling", None)

        if rope_scaling is not None:
            # Only the *flat* HuggingFace rope_scaling shape is supported
            # here, with either "type" or "rope_type" as the discriminator
            # (e.g. {"rope_type": "llama3", "factor": 8.0, ...}). The
            # nested per-layer-type shape used by some modern HF configs —
            # where `rope_parameters` is keyed by layer type such as
            # "full_attention" or "sliding_attention" — is not handled
            # here. Architectures using that shape need a config subclass
            # that pre-flattens the dominant layer-type's parameters
            # before delegating to this method.
            # Since "rope_type" huggingface config is not standardized, we need
            # to check for both "type" and "rope_type" keys.
            # TODO: A better solution would be for those family of models to
            # create their own subclass of MAXModelConfig or Llama3Config, then
            # parts of it like rope_scaling to account for such differences.
            rope_type = rope_scaling.get("type")
            rope_type_alt = rope_scaling.get("rope_type")
            if rope_type is None and rope_type_alt is None:
                raise ValueError(
                    "Neither 'type' nor 'rope_type' found in rope_scaling huggingface config"
                )
            if rope_type == "llama3" or rope_type_alt == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            elif rope_type == "longrope" or rope_type_alt == "longrope":
                longrope_scaling_params = LongRoPEScalingParams(
                    short_factor=rope_scaling["short_factor"],
                    long_factor=rope_scaling["long_factor"],
                    original_max_position=huggingface_config.original_max_position_embeddings,
                    max_position_embeddings=huggingface_config.max_position_embeddings,
                )
                rope_scaling_params = None

        # Calculate base attention multiplier
        base_attention_multiplier = Llama3Config.calculate_attention_multiplier(
            huggingface_config
        )

        # Apply LongRoPE attention scaling if needed
        attention_multiplier = base_attention_multiplier
        if longrope_scaling_params is not None:
            # Create temporary RoPE embedding to get proper attention scale
            rope_embedding = create_rope_embedding(
                hidden_size=huggingface_config.hidden_size,
                num_attention_heads=huggingface_config.num_attention_heads,
                rope_theta=get_rope_theta(huggingface_config),
                max_seq_len=max_seq_len,
                interleaved_rope_weights=interleaved_rope_weights,
                rope_scaling_params=rope_scaling_params,
                longrope_scaling_params=longrope_scaling_params,
                device=DeviceRef.CPU(),  # temporary device, not used for scale computation
            )
            attention_multiplier = rope_embedding.compute_scale()

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=get_rope_theta(huggingface_config),
            rope_scaling_params=rope_scaling_params,
            longrope_scaling_params=longrope_scaling_params,
            max_seq_len=max_seq_len,
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            model_quantization_encoding=supported_encoding_quantization(
                quantization_encoding
            ),
            quantization_config=gptq_quant_config(
                quantization_encoding,
                pipeline_config.model.huggingface_config,
            ),
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            attention_multiplier=attention_multiplier,
            embedding_multiplier=embedding_multiplier,
            residual_multiplier=residual_multiplier,
            devices=device_refs,
            clip_qkv=getattr(huggingface_config, "clip_qkv", None),
            use_subgraphs=pipeline_config.model.use_subgraphs,
            logits_scaling=getattr(huggingface_config, "logits_scaling", 1.0),
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config."""

        # Normalize the LLM state dict so downstream introspection sees canonical
        # Llama-style keys (no "language_model." or "model." prefix). This keeps
        # float8 parsing and feature detection resilient to pack variations.
        def _strip_prefix(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in state_dict)

        if has_lm_prefix:
            normalized_state_dict: dict[str, WeightData] = {
                _strip_prefix(k, "language_model."): v
                for k, v in state_dict.items()
                if k.startswith("language_model.")
            }
        elif has_model_prefix:
            normalized_state_dict = {
                _strip_prefix(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized_state_dict = dict(state_dict)

        quant_config = parse_quant_config(
            huggingface_config, normalized_state_dict, self.dtype
        )

        # Determine norm_dtype.
        # Note: due to automatic weight dtype casting, norm dtype is not always
        # correct. To avoid any issue, only set norm_dtype for float8 models
        # for now.
        norm_dtype = None
        if "layers.0.input_layernorm.weight" in normalized_state_dict:
            norm_dtype = normalized_state_dict[
                "layers.0.input_layernorm.weight"
            ].dtype

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        if "tie_word_embeddings" in huggingface_config:
            tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            tie_word_embeddings = (
                getattr(huggingface_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in normalized_state_dict
            )

        rms_norm_eps = None
        if norm_method == "rms_norm":
            rms_norm_eps = huggingface_config.rms_norm_eps

        self.norm_method = norm_method
        self.norm_dtype = norm_dtype
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.quant_config = quant_config
        self.stacked_mlp = (
            "layers.0.mlp.gate_up_proj.weight" in normalized_state_dict
        )
        self.stacked_qkv = (
            "layers.0.self_attn.qkv_proj.weight" in normalized_state_dict
        )

        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states


@dataclass(frozen=True)
class Dflash2DraftHFConfig:
    """The ``dflash_config`` block of a DFlash2 draft checkpoint."""

    mask_token_id: int
    target_layer_ids: list[int]
    block_size: int
    num_target_layers: int | None
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int

    def draft_width(self, speculative: SpeculativeConfig) -> int:
        """Returns ``block_size - 1``, refusing a flag that disagrees.

        The v1 helper warns and overrides. DFlash2 hard-errors instead
        (design D3): the width is not recoverable from the MEF, so the export
        records it alongside and the engine checks its own
        ``--num-speculative-tokens`` against that at boot. An overridden width
        would compile a block-8 graph and declare it was built for something
        else, and the engine would then feed the graph at the width it was
        told.

        Raises:
            ValueError: If ``num_speculative_tokens`` is set and is not
                ``block_size - 1``.
        """
        expected = self.block_size - 1
        requested = speculative.num_speculative_tokens
        if requested is not None and requested != expected:
            raise ValueError(
                f"DFlash2 was trained at block_size={self.block_size}, so it"
                f" drafts exactly {expected} tokens per step, but"
                f" --num-speculative-tokens is {requested}. The block width is"
                " structural (the drafter's convolution zeroes its t-1 tap at"
                " block-local position 0 and the selector walks a fixed"
                " tensor), so this is a hard error rather than an override."
            )
        return expected


def _dflash_config_of(huggingface_config: Any) -> Any:
    cfg = (
        huggingface_config.get("dflash_config")
        if isinstance(huggingface_config, dict)
        else getattr(huggingface_config, "dflash_config", None)
    )
    if cfg is None:
        raise ValueError(
            "DFlash2 draft HF config is missing ``dflash_config``."
        )
    return cfg


def _required_int(dflash_config: Any, name: str) -> int:
    value = (
        dflash_config.get(name)
        if isinstance(dflash_config, dict)
        else getattr(dflash_config, name, None)
    )
    if value is None:
        raise ValueError(
            f"DFlash2 draft HF config is missing ``dflash_config.{name}``."
        )
    return int(value)


def parse_dflash2_draft_hf_config(
    huggingface_config: Any,
) -> Dflash2DraftHFConfig:
    """Reads the DFlash2 drafter's structural constants off its HF config.

    Raises:
        ValueError: If a required key is missing, if the checkpoint declares no
            ``block_size``, or if it asks for an embedding/logit rescale this
            graph does not implement.
    """
    shared = parse_dflash_draft_hf_config(huggingface_config)
    if shared.block_size is None:
        raise ValueError(
            "DFlash2 draft HF config declares no ``dflash_config.block_size``."
            " The block width is structural — the drafter's convolution zeroes"
            " its t-1 tap at block-local position 0 — so it cannot be supplied"
            " by a flag."
        )
    dflash_config = _dflash_config_of(huggingface_config)

    # The reference multiplies the borrowed embedding and the borrowed head's
    # logits by these. Both are 1.0 on the shipped checkpoint and this graph
    # applies neither, so a checkpoint that sets them would draft from a
    # differently scaled space with no symptom but lost acceptance.
    for name in ("input_embedding_scale", "output_multiplier"):
        scale = (
            dflash_config.get(name, 1.0)
            if isinstance(dflash_config, dict)
            else getattr(dflash_config, name, 1.0)
        )
        if float(scale) != 1.0:
            raise ValueError(
                f"DFlash2 draft declares dflash_config.{name}={scale}; this"
                " graph borrows the target's embedding and head unscaled and"
                " implements neither rescale."
            )

    return Dflash2DraftHFConfig(
        mask_token_id=shared.mask_token_id,
        target_layer_ids=list(shared.target_layer_ids),
        block_size=shared.block_size,
        num_target_layers=shared.num_target_layers,
        conv_kernel_size=_required_int(dflash_config, "conv_kernel_size"),
        conv_group_size=_required_int(dflash_config, "conv_group_size"),
        selector_rank=_required_int(dflash_config, "selector_rank"),
        selector_top_k=_required_int(dflash_config, "selector_top_k"),
    )


def dflash2_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the width the DFlash2 draft checkpoint was trained for.

    Registered as the architecture's ``checkpoint_draft_width`` so that
    ``PipelineConfig.from_args`` resolves an omitted
    ``--num-speculative-tokens``. Without it the pipeline config keeps
    ``None`` while the graph is built at the checkpoint's width, and whatever
    reads the config instead of the graph -- the scheduler's draft-token
    handoff among them -- sees zero.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DFlash2 requires a draft model.")
    return parse_dflash2_draft_hf_config(draft_huggingface_config).draft_width(
        speculative
    )


@dataclass(frozen=True)
class DFlash2Qwen3_5Config:
    """The draft's Llama 3 config and the DFlash2 fields its checkpoint names."""

    llama: Llama3Config
    dflash: Dflash2DraftHFConfig

    @property
    def hidden_size(self) -> int:
        """Returns the draft's hidden size."""
        return self.llama.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the draft's vocabulary size."""
        return self.llama.vocab_size

    @property
    def mask_token_id(self) -> int:
        """Returns the token that fills the block's drafted slots."""
        return self.dflash.mask_token_id

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers the draft reads."""
        return self.dflash.target_layer_ids

    @property
    def block_size(self) -> int:
        """Returns the block the draft was trained at."""
        return self.dflash.block_size

    @property
    def num_target_layers(self) -> int | None:
        """Returns the target depth the draft was trained against.

        ``None`` when the checkpoint names no depth.
        """
        return self.dflash.num_target_layers
