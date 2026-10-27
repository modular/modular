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
"""Config for the Laguna checkpoint family.

This module exposes every Laguna-specific HF config field that the
graph in ``laguna.py`` consumes: per-layer attention shape and type,
per-layer MLP type, the RoPE tables, sigmoid MoE routing parameters,
shared-expert sizing, and the softplus attention-output gate.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    MultiKVCacheParams,
)
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class LagunaLayerSchedule:
    """Per-layer attention and MLP shape read off a Laguna HuggingFace config.

    Laguna varies attention type, query-head count, MLP type and RoPE table
    from one layer to the next. Resolving all of it once, here, keeps the graph
    code from re-deriving it -- differently each time -- out of raw HF fields.
    """

    layer_types: tuple[str, ...]
    """``"full_attention"`` or ``"sliding_attention"``, one entry per layer."""

    num_heads_per_layer: tuple[int, ...]
    """Query-head count per layer. KV heads stay uniform across layers."""

    mlp_layer_types: tuple[str, ...]
    """``"dense"`` or ``"sparse"``, one entry per layer."""

    gating_mode: str
    """``"per-element"`` (``g_proj`` emits ``heads * head_dim``) or
    ``"per-head"`` (``g_proj`` emits ``heads``). Uniform across layers; a
    ``gating_types`` list that disagrees is rejected rather than reconciled."""

    sliding_window: int | None
    """Window shared by every sliding layer, ``None`` when the model has none.
    Never inferred from the config's ``sliding_window`` field alone: the
    transformers config defaults it to 512 even for all-full-attention models,
    so ``layer_types`` is the only trustworthy discriminator."""

    full_rope: Mapping[str, Any]
    """RoPE table for full-attention layers."""

    sliding_rope: Mapping[str, Any] | None
    """RoPE table for sliding layers, ``None`` when the model has none."""


@dataclass(kw_only=True)
class LagunaConfig(Llama3Config):
    """Configuration for Laguna decoder-only MoE models.

    Extends ``Llama3Config`` with Laguna-specific fields:

    - **Per-layer attention schedule**: ``layer_schedule`` carries each
      layer's attention type, query-head count, MLP type, gating mode and
      RoPE table. A checkpoint whose config names no sliding layers and no
      per-layer head counts resolves to a uniform shape -- full causal
      attention everywhere against one RoPE table -- which is what the
      architecture did before per-layer support existed.
    - **Per-layer MLP type**: ``mlp_layer_types`` is ``dense`` (the dense
      prefix) or ``sparse`` (the rest). The decoder block dispatches between
      the dense MLP and the sparse MoE block.
    - **Sigmoid + correction-bias routing**: not softmax. See
      ``LagunaTopKRouter`` for the per-token routing math.
    - **Routed scaling factor + shared experts + softplus attention
      output gate**: the gate is one scalar per query head or one per
      ``(head, head_dim)`` channel, per ``layer_schedule.gating_mode``.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    # --- MoE (routed expert collection) ---
    num_local_experts: int = 256
    """Number of routed experts per sparse layer."""

    num_experts_per_tok: int = 8
    """Top-k experts selected per token."""

    moe_intermediate_size: int = 512
    """Per-expert intermediate dim (each routed expert is a SwiGLU MLP
    with this intermediate size)."""

    shared_expert_intermediate_size: int = 512
    """Intermediate dim of the always-on shared expert MLP added
    alongside the routed-expert output (sum, not gated)."""

    moe_routed_scaling_factor: float = 2.5
    """Scalar applied to routed-expert output before adding the shared
    expert output. Laguna-specific (default 1.0 in donors that have it
    at all)."""

    moe_router_logit_softcapping: float = 0.0
    """If > 0, router logits are passed through
    ``softcap * tanh(logits/softcap)`` before sigmoid. Disabled (0.0)
    in every released Laguna checkpoint."""

    norm_topk_prob: bool = True
    """Whether to L1-normalise the selected top-k routing weights so
    they sum to 1. True for Laguna (matches HF reference)."""

    # --- Sigmoid routing parameters ---
    correction_bias_dtype: DType | None = None
    """Data type of the e_score_correction_bias weight. Detected from
    state dict during finalize()."""

    gate_dtype: DType | None = None
    """Data type for the routed-expert gate Linear. Detected from state
    dict during finalize()."""

    attn_dtype: DType | None = None
    """Data type for attention weights. Detected from state dict during
    finalize()."""

    unquantized_dtype: DType = DType.bfloat16
    """Storage dtype of the layers the checkpoint leaves unquantized, read off
    ``embed_tokens.weight``. ``dtype`` is the packed FP4 dtype on a quantized
    checkpoint, so the bf16 MLPs of a mixed-precision model cannot use it."""

    ep_config: EPConfig | None = None
    """Expert parallelism configuration. None means no EP (single-GPU)."""

    # --- Per-layer MLP type ---
    mlp_layer_types: list[str] | None = None
    """Per-layer MLP type. Each entry is ``"dense"`` or ``"sparse"``.
    Length equals ``num_hidden_layers``. Typically layer 0 is dense
    and the rest are sparse MoE."""

    intermediate_size_dense: int = 8192
    """Intermediate dim of the dense-layer MLPs (the one or more layers
    where ``mlp_layer_types[i] == "dense"``). Distinct from
    ``intermediate_size`` (which Llama3Config uses for the active
    branch) and from ``moe_intermediate_size``."""

    # --- RoPE ---
    rope_scaling_params: Llama3RopeScalingParams | None = None
    """Unused: Laguna builds one RoPE table per attention type from
    ``layer_schedule``, not from Llama3's single flat scaling record."""

    # --- KV cache ---
    # TODO(SERVOPT-1607): Llama3Config narrows the KV contract to
    # KVCacheParams, which MultiKVCacheParams cannot satisfy. Both
    # suppressions here go when that contract widens.
    kv_params: KVCacheParams | MultiKVCacheParams  # type: ignore[assignment]
    """One flat cache, or the two-leaf sliding/full tree when the model has
    sliding-attention layers."""

    # --- Per-layer schedule ---
    layer_schedule: LagunaLayerSchedule | None = None
    """Per-layer attention type, query heads, MLP type, gating mode and RoPE
    tables. Populated by ``initialize_from_config``."""

    @staticmethod
    def read_layer_schedule(
        huggingface_config: AutoConfig,
    ) -> LagunaLayerSchedule:
        """Reads the per-layer schedule out of a Laguna HuggingFace config.

        Every list defaults to the uniform, all-full-attention shape that
        ``poolside/Laguna-M.1-NVFP4`` has, so a config that omits them parses
        to the behavior the architecture had before per-layer support existed.
        Anything the architecture cannot represent -- a list of the wrong
        length, a missing attention gate, per-layer mixed gating, sliding
        layers with no window, scaled RoPE on a sliding layer -- raises rather
        than silently resolving to a different model.

        Args:
            huggingface_config: The HuggingFace model configuration.

        Returns:
            The resolved per-layer schedule.

        Raises:
            ValueError: If the config describes a Laguna variant the
                architecture does not implement.
        """
        num_layers = LagunaConfig.get_num_layers(huggingface_config)

        layer_types = list(
            getattr(huggingface_config, "layer_types", None)
            or ["full_attention"] * num_layers
        )
        if len(layer_types) != num_layers:
            raise ValueError(
                f"Laguna layer_types has {len(layer_types)} entries but the "
                f"model has {num_layers} layers."
            )
        # These two strings key the KV tree's children, so an unrecognised one
        # would silently address the wrong cache leaf.
        unknown_types = set(layer_types) - {
            "full_attention",
            "sliding_attention",
        }
        if unknown_types:
            raise ValueError(
                f"Laguna layer_types contains unsupported attention types "
                f"{sorted(unknown_types)}; only 'full_attention' and "
                "'sliding_attention' are supported."
            )

        num_heads_per_layer = list(
            getattr(huggingface_config, "num_attention_heads_per_layer", None)
            or [huggingface_config.num_attention_heads] * num_layers
        )
        if len(num_heads_per_layer) != num_layers:
            raise ValueError(
                f"Laguna num_attention_heads_per_layer has "
                f"{len(num_heads_per_layer)} entries but the model has "
                f"{num_layers} layers."
            )
        num_key_value_heads = huggingface_config.num_key_value_heads
        for layer_idx, heads in enumerate(num_heads_per_layer):
            if heads <= 0 or heads % num_key_value_heads != 0:
                raise ValueError(
                    f"Laguna layer {layer_idx} has {heads} query heads, which "
                    f"is not a positive multiple of num_key_value_heads "
                    f"({num_key_value_heads})."
                )

        mlp_layer_types = list(
            getattr(huggingface_config, "mlp_layer_types", None)
            or ["dense"] + ["sparse"] * (num_layers - 1)
        )
        if len(mlp_layer_types) != num_layers:
            raise ValueError(
                f"Laguna mlp_layer_types has {len(mlp_layer_types)} entries "
                f"but the model has {num_layers} layers."
            )
        mlp_only_layers = getattr(huggingface_config, "mlp_only_layers", None)
        if mlp_only_layers is not None:
            dense_layers = {
                i for i, t in enumerate(mlp_layer_types) if t == "dense"
            }
            if set(mlp_only_layers) != dense_layers:
                raise ValueError(
                    f"Laguna mlp_only_layers {sorted(mlp_only_layers)} "
                    f"disagrees with the dense entries of mlp_layer_types "
                    f"{sorted(dense_layers)}."
                )

        gating = getattr(huggingface_config, "gating", True)
        if gating is True or gating == "per-element":
            gating_mode = "per-element"
        elif gating == "per-head":
            gating_mode = "per-head"
        else:
            raise ValueError(
                "Laguna without an attention gate is not supported "
                f"(gating={gating!r})."
            )
        gating_types = getattr(huggingface_config, "gating_types", None)
        if gating_types is not None:
            expected = gating_mode.replace("-", "_")
            disagreeing = {
                t for t in gating_types if str(t).replace("-", "_") != expected
            }
            if disagreeing:
                raise ValueError(
                    f"Laguna gating is {gating_mode!r} but gating_types also "
                    f"contains {sorted(disagreeing)}. Per-layer mixed gating "
                    "is not supported."
                )

        # The transformers config defaults ``sliding_window`` to 512 whether or
        # not the model has sliding layers, and M.1 ships an explicit 0, so the
        # window is only meaningful once ``layer_types`` says it is.
        has_sliding = "sliding_attention" in layer_types
        sliding_window: int | None = None
        if has_sliding:
            window = getattr(huggingface_config, "sliding_window", None)
            if not window:
                raise ValueError(
                    "Laguna config declares sliding_attention layers but "
                    f"sliding_window is {window!r}."
                )
            sliding_window = int(window)

        rope_parameters = (
            getattr(huggingface_config, "rope_parameters", {}) or {}
        )
        # Laguna keys the RoPE tables by attention type; fall back to the flat
        # dict so a config written in the unnested form still parses.
        full_rope = rope_parameters.get("full_attention", rope_parameters)
        # transformers folds a flat config's rope_scaling into rope_parameters
        # but leaves rope_theta and partial_rotary_factor as top-level
        # scalars. Reading them beats defaulting theta to 10000 and the rotary
        # factor to 1.0, which is silently wrong rather than an error.
        top_level = {}
        for key in ("rope_theta", "partial_rotary_factor"):
            value = getattr(huggingface_config, key, None)
            if key not in full_rope and value is not None:
                top_level[key] = value
        if top_level:
            full_rope = {**full_rope, **top_level}
        LagunaConfig._validate_yarn_attention_factor(full_rope)

        sliding_rope: Mapping[str, Any] | None = None
        if has_sliding:
            # The reference reuses the full-attention table when the sliding
            # one is absent.
            sliding_rope = rope_parameters.get("sliding_attention") or full_rope
            sliding_rope_type = sliding_rope.get("rope_type", "default")
            if sliding_rope_type != "default":
                raise ValueError(
                    "Laguna supports only default RoPE on sliding-attention "
                    f"layers, got {sliding_rope_type!r}."
                )

        return LagunaLayerSchedule(
            layer_types=tuple(layer_types),
            num_heads_per_layer=tuple(int(h) for h in num_heads_per_layer),
            mlp_layer_types=tuple(mlp_layer_types),
            gating_mode=gating_mode,
            sliding_window=sliding_window,
            full_rope=full_rope,
            sliding_rope=sliding_rope,
        )

    @staticmethod
    def _validate_yarn_attention_factor(full_rope: Mapping[str, Any]) -> None:
        """Rejects a YaRN table whose mscale MAX cannot reproduce.

        ``YarnRotaryEmbedding`` hardcodes the YaRN attention scale as
        ``0.1 * ln(factor) + 1`` and has no hook for the config's own
        ``attention_factor``, so a config that disagrees would be scaled
        wrongly with nothing in the logs to say so.

        Args:
            full_rope: The full-attention RoPE table.

        Raises:
            ValueError: If the table's ``attention_factor`` is not the value
                MAX computes.
        """
        if full_rope.get("rope_type") != "yarn":
            return
        if "attention_factor" not in full_rope:
            return
        attention_factor = float(full_rope["attention_factor"])
        expected = 0.1 * math.log(float(full_rope["factor"])) + 1.0
        if abs(attention_factor - expected) > 1e-6:
            raise ValueError(
                f"Laguna YaRN attention_factor is {attention_factor}, but MAX "
                f"computes the mscale as 0.1*ln(factor)+1 = {expected} and "
                "cannot be told otherwise."
            )

    @staticmethod
    def construct_kv_params(  # type: ignore[override]  # TODO(SERVOPT-1607)
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams | MultiKVCacheParams:
        """Constructs the KV cache parameters for a Laguna checkpoint.

        A model whose schedule has sliding layers gets a two-leaf tree --
        ``sliding_attention`` first, then ``full_attention`` -- so the sliding
        layers can eventually reserve only their window. A model with no
        sliding layer, ``poolside/Laguna-M.1-NVFP4`` among them, keeps the
        single flat cache it has always had.

        Args:
            huggingface_config: The HuggingFace configuration object.
            pipeline_config: The MAX Engine pipeline configuration.
            devices: Devices to use for the KV cache.
            kv_cache_config: Configuration for KV cache.
            cache_dtype: Data type for the cache.
            allow_kv_head_replication: Whether KV heads may be replicated
                across tensor-parallel ranks.

        Returns:
            A single ``KVCacheParams``, or the hybrid tree when the model has
            sliding-attention layers.

        Raises:
            ValueError: If an FP8 KV cache is requested for a model with
                sliding layers.
        """
        # The MiniMax-M2 donor defaulted to
        # ``data_parallel_degree = len(devices)``; this couples DP
        # degree to device count and requires N batched-per-request
        # tokens at inference. Laguna instead respects the
        # pipeline_config's explicit DP setting (default 1) so a single
        # prompt through ``/v1/completions`` works on any device count.
        # The pipeline owner can opt into DP=N via
        # ``pipeline_config.model.data_parallel_degree=N``.
        configured_dp = getattr(
            pipeline_config.model, "data_parallel_degree", 1
        )
        data_parallel_degree = max(1, int(configured_dp))
        # FP8 KV cache: EXPERIMENTAL and not yet accuracy-validated. Opt in via
        # ``--kv-cache-format float8_e4m3fn``; the default bf16 KV cache is the
        # validated configuration. When the resolved cache dtype is FP8, MAX
        # needs a quantization config to lay out the cache; mirror deepseekV3.
        # MAX scales KV dynamically at runtime — the checkpoint's static
        # ``self_attn.{k,v}_scale`` are dropped in the weight adapter (MAX never
        # consumes them). ``scale_dtype=int8`` selects the *unscaled*
        # ``is_fp8_kv`` cast path (full per-block scaling / SnapMLA is still WIP
        # upstream), so this can degrade quality on longer contexts and has NOT
        # been validated against the bf16 path. ``LagunaAttention`` uses the
        # fused ``rope_split_store_ragged`` store, which converts bf16 -> FP8 at
        # store time so the path at least runs.
        is_fp8_cache = cache_dtype in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        )

        schedule = LagunaConfig.read_layer_schedule(huggingface_config)
        if schedule.sliding_window is not None:
            if is_fp8_cache:
                raise ValueError(
                    "Laguna sliding-attention layers cannot use an FP8 KV "
                    f"cache: the fused rope_split_store_ragged path rejects a "
                    f"scaled quantized cache (requested {cache_dtype}). Serve "
                    "this model with the default bfloat16 KV cache."
                )
            return hybrid_swa_full_kv_params(
                layer_types=schedule.layer_types,
                sliding_window=schedule.sliding_window,
                pipeline_config=pipeline_config,
                devices=devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
                n_kv_heads=huggingface_config.num_key_value_heads,
                head_dim=huggingface_config.head_dim,
                allow_kv_head_replication=allow_kv_head_replication,
            )

        kvcache_quant_config = None
        if is_fp8_cache:
            kvcache_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.int8, quantization_granularity=32
            )
        return kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=LagunaConfig.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
        )

    @override
    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """Computes the attention multiplier from the config's head_dim.

        Args:
            huggingface_config: The HuggingFace configuration object.

        Returns:
            The attention multiplier value.
        """
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(huggingface_config.head_dim)),
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
        """Initializes a LagunaConfig from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized LagunaConfig instance.
        """
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

    @override
    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a LagunaConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration.
            model_config: The MAX Engine model configuration.

        Returns:
            An initialized LagunaConfig instance.
        """
        # Llama3Config expects flat HF fields (``rope_scaling`` with a
        # ``type`` / ``rope_type`` key, and ``rope_theta``). Laguna nests
        # those under ``rope_parameters[layer_type]``. Stub flat values
        # from the full-attention table so the parent can initialise; we
        # build the actual RoPE wiring later from the original dict.
        rope_parameters = (
            getattr(huggingface_config, "rope_parameters", {}) or {}
        )
        full_rope = rope_parameters.get("full_attention", {})

        _orig_rope_scaling = getattr(huggingface_config, "rope_scaling", None)
        _orig_rope_theta = getattr(huggingface_config, "rope_theta", None)
        try:
            huggingface_config.rope_scaling = None
            huggingface_config.rope_theta = float(
                full_rope.get("rope_theta", 10000.0)
            )
        except (AttributeError, TypeError):
            pass

        try:
            base_config = Llama3Config.initialize_from_config(
                pipeline_config,
                huggingface_config,
                model_config,
                max_seq_len=max_seq_len,
            )
        finally:
            # Restore the originals so downstream readers see the
            # unmodified HF config.
            try:
                huggingface_config.rope_scaling = _orig_rope_scaling
                if _orig_rope_theta is not None:
                    huggingface_config.rope_theta = _orig_rope_theta
            except (AttributeError, TypeError):
                pass

        kv_cache_config = pipeline_config.model.kv_cache
        cache_dtype = cache_dtype_for_encoding(
            base_config.quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )
        n_devices = len(pipeline_config.model.device_specs)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]

        kv_params = LagunaConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        attention_multiplier = LagunaConfig.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # --- MoE ---
        # Laguna uses ``num_experts`` (HF field). MiniMax called it
        # ``num_local_experts``. Accept both.
        num_local_experts = getattr(
            huggingface_config,
            "num_experts",
            getattr(huggingface_config, "num_local_experts", 256),
        )
        num_experts_per_tok = getattr(
            huggingface_config, "num_experts_per_tok", 8
        )
        moe_intermediate_size = getattr(
            huggingface_config, "moe_intermediate_size", 512
        )
        shared_expert_intermediate_size = getattr(
            huggingface_config, "shared_expert_intermediate_size", 512
        )
        moe_routed_scaling_factor = float(
            getattr(huggingface_config, "moe_routed_scaling_factor", 1.0)
        )
        moe_router_logit_softcapping = float(
            getattr(huggingface_config, "moe_router_logit_softcapping", 0.0)
        )

        norm_topk_prob = bool(
            getattr(huggingface_config, "norm_topk_prob", True)
        )

        # --- Per-layer schedule ---
        layer_schedule = LagunaConfig.read_layer_schedule(huggingface_config)
        intermediate_size_dense = getattr(
            huggingface_config, "intermediate_size", 8192
        )

        # NVFP4 convention (matches deepseekV3/gemma4): config.dtype IS the
        # packed-fp4 quant dtype (uint8), so the MoE/dense quant Linears and the
        # grouped-quantize strategy use the right out_type. Non-quant layers
        # (embed/lm_head) are handled via quant_config.embedding_output_dtype;
        # attention/norm dtypes are auto-detected from the checkpoint.
        resolved_dtype = base_config.dtype

        return cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            rope_theta=base_config.rope_theta,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            interleaved_rope_weights=base_config.interleaved_rope_weights,
            vocab_size=base_config.vocab_size,
            dtype=resolved_dtype,
            model_quantization_encoding=base_config.model_quantization_encoding,
            quantization_config=base_config.quantization_config,
            max_seq_len=base_config.max_seq_len,
            kv_params=kv_params,
            attention_multiplier=attention_multiplier,
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            data_parallel_degree=base_config.data_parallel_degree,
            quantization_encoding=base_config.quantization_encoding,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            moe_routed_scaling_factor=moe_routed_scaling_factor,
            moe_router_logit_softcapping=moe_router_logit_softcapping,
            norm_topk_prob=norm_topk_prob,
            mlp_layer_types=list(layer_schedule.mlp_layer_types),
            intermediate_size_dense=intermediate_size_dense,
            layer_schedule=layer_schedule,
        )
