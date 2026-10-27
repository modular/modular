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

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.legacy.quant_config import NVFP4_BLOCK_SIZE, QuantConfig
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
    ArchConfigWithKVCache,
)
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

logger = logging.getLogger("max.pipelines")

_EXCLUDED_LAYER = re.compile(r"layers\.(\d+)\.")
_LANGUAGE_MODULE = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")


def unpadded_vocab_size(pipeline_config: PipelineConfig) -> int:
    """Returns the tokenizer's size, the bound of the vocab tail mask.

    Raises:
        ValueError: If the tokenizer cannot be read. Without the bound,
            padding ids could be sampled.
    """
    # Loads the tokenizer while the config is built, on purpose: the mask
    # bound must be known before the graph is.
    try:
        tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
            pipeline_config.model.huggingface_model_repo
        )
    except Exception as e:
        raise ValueError(
            "Muse Spark: could not read the tokenizer size, which bounds the"
            " padded vocab tail mask"
        ) from e
    return len(tokenizer)


def _rope_theta(hf_config: AutoConfig) -> float:
    rope_params = hf_config.rope_parameters
    if rope_params.get("rope_type") != "default":
        raise ValueError(f"Rope parameters {rope_params} not supported")
    return rope_params["rope_theta"]


@dataclass(frozen=True)
class MuseSparkNVFP4:
    """The checkpoint's modelopt NVFP4 layout."""

    quant_config: QuantConfig
    """Quantizes the routed experts of ``mlp_quantized_layers`` only."""
    excluded_modules: frozenset[str]
    """Layer-relative (``mlp.pre_expert_proj``) or top-level (``lm_head``)
    module names that stay bfloat16 on every layer."""


def parse_nvfp4(
    quantization_config: Mapping[str, Any], num_hidden_layers: int
) -> MuseSparkNVFP4:
    """Parses the modelopt ``quantization_config`` of a Muse Spark checkpoint.

    Args:
        quantization_config: The top-level config's ``quantization_config``.
        num_hidden_layers: The text decoder's layer count.

    Returns:
        The NVFP4 layout.

    Raises:
        ValueError: If the config is not block-16 modelopt NVFP4, or its
            ``exclude_modules`` and ``modules_to_not_convert`` disagree on
            which layers keep bfloat16 experts.
    """
    method = quantization_config.get("quant_method")
    algo = quantization_config.get("quant_algo")
    group_size = quantization_config.get("group_size")
    if (method, algo, group_size) != ("modelopt", "NVFP4", NVFP4_BLOCK_SIZE):
        raise ValueError(
            "Muse Spark supports modelopt NVFP4 with group_size"
            f" {NVFP4_BLOCK_SIZE}, got quant_method={method!r},"
            f" quant_algo={algo!r}, group_size={group_size!r}"
        )

    excluded_layers: set[int] = set()
    excluded_modules: set[str] = set()
    for entry in quantization_config.get("exclude_modules", []):
        if match := _EXCLUDED_LAYER.fullmatch(entry):
            excluded_layers.add(int(match[1]))
        else:
            excluded_modules.add(entry)

    bf16_expert_layers: set[int] = set()
    for entry in quantization_config.get("modules_to_not_convert", []):
        if match := _LANGUAGE_MODULE.fullmatch(entry):
            layer, module = int(match[1]), match[2]
            if module == "mlp.experts":
                bf16_expert_layers.add(layer)
            elif layer not in excluded_layers:
                excluded_modules.add(module)
        elif entry == "lm_head":
            excluded_modules.add(entry)

    if bf16_expert_layers != excluded_layers:
        raise ValueError(
            f"exclude_modules leaves layers {sorted(excluded_layers)} bfloat16"
            " but modules_to_not_convert leaves the experts of layers"
            f" {sorted(bf16_expert_layers)} bfloat16"
        )
    return MuseSparkNVFP4(
        quant_config=QuantConfig.nvfp4(
            mlp_quantized_layers=set(range(num_hidden_layers))
            - excluded_layers,
            attn_quantized_layers=set(),
            embedding_output_dtype=DType.bfloat16,
        ),
        excluded_modules=frozenset(excluded_modules),
    )


def reconcile_expert_dtypes(
    quant_config: QuantConfig, expert_dtypes: Mapping[int, DType]
) -> QuantConfig:
    """Makes the quantized expert layers match the checkpoint's tensors.

    The config's exclude lists are a claim about the weights; the tensor
    dtypes are what the kernels read, so they win.

    Args:
        quant_config: The config parsed from ``quantization_config``.
        expert_dtypes: Each layer's routed-expert weight dtype, ``uint8`` for
            packed NVFP4.

    Returns:
        ``quant_config``, with ``mlp_quantized_layers`` replaced when the
        tensors disagree with it.
    """
    quantized = {
        i for i, dtype in expert_dtypes.items() if dtype == DType.uint8
    }
    if quantized == quant_config.mlp_quantized_layers:
        return quant_config
    logger.warning(
        "Muse Spark quantization_config marks expert layers %s NVFP4, but the"
        " checkpoint packs layers %s; following the checkpoint",
        sorted(quant_config.mlp_quantized_layers),
        sorted(quantized),
    )
    return replace(quant_config, mlp_quantized_layers=quantized)


@dataclass(kw_only=True)
class MuseSparkTextConfig(ArchConfigWithBoundedMaxSeqLen):
    """Text decoder configuration for Muse Spark."""

    dtype: DType
    devices: list[DeviceRef]
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    """Epsilon of the input, pre-FFN, pre-expert, embedding, qk and final
    norms."""
    post_norm_eps: float
    """Epsilon of the post-attention, post-FFN and post-expert norms."""
    layer_types: list[str]
    layer_rope_theta: list[float]
    """Per-layer RoPE theta; 0 marks a NoPE layer."""
    mlp_layer_types: list[str]
    sliding_window: int
    rope_theta: float
    max_position_embeddings: int
    max_seq_len: int
    qk_scale_factor: float
    output_multiplier: float
    """Scales the router logits, and the final hidden state before
    ``lm_head``."""
    final_logit_softcapping: float
    moe_hidden_size: int
    """Width ``pre_expert_proj`` pools the hidden state down to."""
    moe_intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    residual_gate_temperature: float
    tie_word_embeddings: bool = False
    unpadded_vocab_size: int
    """The tokenizer's size; ``lm_head`` rows past it are padding."""
    kv_params: MultiKVCacheParams
    nvfp4: MuseSparkNVFP4 | None = None
    """``None`` for a bfloat16 checkpoint."""
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    use_rope: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.use_rope = [theta != 0 for theta in self.layer_rope_theta]
        sliding = [t == "sliding_attention" for t in self.layer_types]
        # RoPE on sliding layers only and a routed MoE on every layer are
        # architecture invariants; a checkpoint that breaks one needs a model
        # change, not a config read.
        if len(self.layer_types) != self.num_hidden_layers or (
            self.use_rope != sliding
        ):
            raise ValueError(
                "layer_rope_theta must be nonzero exactly on the"
                f" sliding_attention layers: layer_types={self.layer_types},"
                f" layer_rope_theta={self.layer_rope_theta}"
            )
        if self.mlp_layer_types != ["sparse"] * self.num_hidden_layers:
            raise ValueError(
                "Muse Spark supports only sparse MLP layers, got"
                f" mlp_layer_types={self.mlp_layer_types}"
            )

    @property
    def quant_config(self) -> QuantConfig | None:
        """The NVFP4 config of the routed experts, if quantized."""
        return self.nvfp4.quant_config if self.nvfp4 else None

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Builds the ``sliding_attention`` + ``full_attention`` KV tree."""
        return hybrid_swa_full_kv_params(
            layer_types=huggingface_config.layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
        dtype: DType,
        cache_dtype: DType,
        nvfp4: MuseSparkNVFP4 | None,
    ) -> Self:
        """Initializes from the HuggingFace ``text_config``.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace text configuration.
            max_seq_len: The effective maximum sequence length.
            dtype: The weight dtype.
            cache_dtype: The KV cache dtype.
            nvfp4: The checkpoint's NVFP4 layout, or ``None`` for bfloat16.

        Returns:
            An initialized :obj:`MuseSparkTextConfig`.
        """
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]
        hf = huggingface_config
        return cls(
            dtype=dtype,
            devices=devices,
            hidden_size=hf.hidden_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hf.head_dim,
            num_hidden_layers=hf.num_hidden_layers,
            vocab_size=hf.vocab_size,
            rms_norm_eps=hf.rms_norm_eps,
            post_norm_eps=hf.post_norm_eps,
            layer_types=list(hf.layer_types),
            layer_rope_theta=list(hf.layer_rope_theta),
            mlp_layer_types=list(hf.mlp_layer_types),
            sliding_window=hf.sliding_window,
            rope_theta=_rope_theta(hf),
            max_position_embeddings=hf.max_position_embeddings,
            max_seq_len=max_seq_len,
            qk_scale_factor=hf.qk_scale_factor,
            output_multiplier=hf.output_multiplier,
            final_logit_softcapping=hf.final_logit_softcapping,
            moe_hidden_size=hf.moe_hidden_size,
            moe_intermediate_size=hf.moe_intermediate_size,
            num_local_experts=hf.num_local_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
            routed_scaling_factor=hf.routed_scaling_factor,
            residual_gate_temperature=hf.residual_gate_temperature,
            tie_word_embeddings=hf.tie_word_embeddings,
            unpadded_vocab_size=unpadded_vocab_size(pipeline_config),
            kv_params=cls.construct_kv_params(
                huggingface_config=hf,
                pipeline_config=pipeline_config,
                devices=devices,
                kv_cache_config=pipeline_config.model.kv_cache,
                cache_dtype=cache_dtype,
            ),
            nvfp4=nvfp4,
        )


@dataclass(kw_only=True)
class MuseSparkConfig(ArchConfigWithKVCache):
    """Top-level Muse Spark configuration."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    devices: list[DeviceRef]
    dtype: DType
    kv_params: MultiKVCacheParams
    image_token_id: int
    video_token_id: int
    audio_token_id: int
    text_config: MuseSparkTextConfig
    quantization_encoding: SupportedEncoding | None = None

    def get_kv_params(self) -> MultiKVCacheParams:
        """Returns the KV cache parameters."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length of the text decoder."""
        return self.text_config.get_max_seq_len()

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Constructs KV cache parameters from the top-level HuggingFace config."""
        return MuseSparkTextConfig.construct_kv_params(
            huggingface_config=huggingface_config.text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bounds ``max_length`` by the text decoder's context."""
        return MuseSparkTextConfig.calculate_max_seq_len(
            huggingface_config.text_config, model_config
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
        """Initializes from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: Optional model config override.
            max_seq_len: The effective maximum sequence length.

        Returns:
            An initialized config instance.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be loaded."
                " Please ensure the model repository contains a valid"
                " config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config, huggingface_config, max_seq_len=max_seq_len
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: Top-level HuggingFace model configuration.
            max_seq_len: The effective maximum sequence length.

        Returns:
            A config instance ready for finalization.
        """
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )

        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        quantization_config = getattr(
            huggingface_config, "quantization_config", None
        )
        if (quantization_config is not None) != (
            quantization_encoding == "float4_e2m1fnx2"
        ):
            raise ValueError(
                f"Encoding {quantization_encoding!r} does not match the"
                " checkpoint: an NVFP4 checkpoint needs float4_e2m1fnx2, a"
                " checkpoint without quantization_config needs bfloat16"
            )
        nvfp4 = (
            parse_nvfp4(quantization_config, hf_text_config.num_hidden_layers)
            if quantization_config is not None
            else None
        )
        text_config = MuseSparkTextConfig.initialize_from_config(
            pipeline_config,
            hf_text_config,
            max_seq_len=max_seq_len,
            dtype=dtype,
            cache_dtype=cache_dtype,
            nvfp4=nvfp4,
        )

        return cls(
            devices=text_config.devices,
            dtype=dtype,
            kv_params=text_config.kv_params,
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            audio_token_id=huggingface_config.audio_token_id,
            text_config=text_config,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Sets the fields that depend on the loaded weights and pipeline.

        Args:
            huggingface_config: HuggingFace model configuration.
            state_dict: Model weights dictionary.
            return_logits: Return logits configuration.
        """
        self.text_config.return_logits = return_logits
