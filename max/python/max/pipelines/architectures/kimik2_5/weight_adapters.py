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

"""Weight adapters for Kimi K2.5 components.

Maps HuggingFace checkpoint keys to the MAX module state_dict keys for:
- Language model (``language_model.*`` → DeepseekV3)
- Vision encoder (``vision_tower.*`` + projector → Transformer)

Two top-level adapters correspond to the two architectures in ``arch.py``:

- :func:`convert_kimik2_5_safetensor_state_dict` —
  ``KimiK25ForConditionalGeneration`` (e.g. nvidia/Kimi-K2.5-NVFP4).
  Uses ``mm_projector.*`` projector naming; drops ``.k_scale``/``.v_scale``.
- :func:`convert_kimivl_safetensor_state_dict` —
  ``KimiVLForConditionalGeneration`` (e.g. moonshotai/Kimi-VL-A3B).
  Uses ``multi_modal_projector.*`` projector naming.

Both adapters share :func:`_convert_merged_state_dict`, which processes
vision and language keys in a single loop over the raw checkpoint.

For MXFP4 checkpoints the model calls :func:`preshuffle_mxfp4_b_experts`
on the post-adapter state dict to lay expert ``B`` bytes out in
``Shuffler.b_5d_grouped_layout`` for the AMD preb grouped-matmul kernel.
The preshuffle is pure-numpy on CPU.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.weights._fp8 import dequantize_rowwise_fp8
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger("max.pipelines")

# ---------------------------------------------------------------------------
# Language model rename map
# ---------------------------------------------------------------------------

KIMI_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "language_model.",
    "gate.weight": "gate.gate_score.weight",
    "weight_scale_inv": "weight_scale",
}

# ---------------------------------------------------------------------------
# Vision tower + projector rename map
# ---------------------------------------------------------------------------

# String-replacement mapping for vision-related checkpoint keys.
# Replacements are applied in insertion order.  The first three entries
# swap HuggingFace prefixes for MAX module prefixes; the remaining
# entries rename sub-keys within those modules.
KIMIK2_5_VISION_MAPPING: dict[str, str] = {
    # Prefix swaps (must come first)
    "vision_tower.": "vision_encoder.",
    "mm_projector.": "vision_encoder.patch_merger.",
    "multi_modal_projector.": "vision_encoder.patch_merger.",
    # Vision tower sub-key renames
    "encoder.final_layernorm.": "encoder.norm.",
    ".mlp.fc0.": ".mlp.up_proj.",
    ".mlp.fc1.": ".mlp.down_proj.",
    # Projector sub-key renames — two conventions:
    #   mm_projector: Sequential indices (proj.0 / proj.2)
    #   multi_modal_projector: underscore names (linear_1 / linear_2)
    "proj.0.": "linear1.",
    "proj.2.": "linear2.",
    "linear_1.": "linear1.",
    "linear_2.": "linear2.",
}

# Regex patterns for attention renames that need the block index captured.
# HF: ``blocks.N.wqkv.*`` → MAX: ``blocks.N.attn.wqkv.*``
# HF: ``blocks.N.wo.*``   → MAX: ``blocks.N.attn.wo.*``
_ATTN_RENAME_PATTERNS = [
    (re.compile(r"blocks\.(\d+)\.wqkv\."), r"blocks.\1.attn.wqkv."),
    (re.compile(r"blocks\.(\d+)\.wo\."), r"blocks.\1.attn.wo."),
]

_VISION_PREFIXES = ("vision_tower.", "mm_projector.", "multi_modal_projector.")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rename_vision_key(checkpoint_name: str) -> str:
    """Apply KIMIK2_5_VISION_MAPPING and attention regex renames."""
    name = checkpoint_name
    for before, after in KIMIK2_5_VISION_MAPPING.items():
        name = name.replace(before, after)
    for pattern, replacement in _ATTN_RENAME_PATTERNS:
        name = pattern.sub(replacement, name)
    return name


def _cast_vision_weight(checkpoint_name: str, weight: Weights) -> WeightData:
    """Return WeightData, casting floats (non-FP8, non-scale) to bfloat16."""
    weight_data = weight.data()
    is_scale = checkpoint_name.endswith(
        ".weight_scale"
    ) or checkpoint_name.endswith(".input_scale")
    if (
        weight_data.dtype.is_float()
        and not weight_data.dtype.is_float8()
        and not is_scale
    ):
        weight_data = weight_data.astype(DType.bfloat16)
    return weight_data


_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.layers\.\d+\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_EXPERT_SCALE_RE = re.compile(
    r"^(?P<prefix>.+\.layers\.\d+\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight_scale$"
)


def _as_shuffleable_mxfp4_b(wd: WeightData) -> np.ndarray | None:
    """Return ``wd`` as a numpy view if it's a shuffleable MXFP4 B weight.

    A weight is shuffleable when its dtype is packed-MXFP4 (uint8) and its
    dims are MFMA-tile-aligned: ``N % 16 == 0`` (NLane=16) and
    ``K_BYTES % 64 == 0`` (4 KLane * 16 KPack). The shuffle reshape
    hardcodes those factors, so non-aligned dims would crash on reshape.
    Returns ``None`` when the weight isn't shuffleable.
    """
    if wd.dtype != DType.uint8:
        return None
    arr = np.from_dlpack(wd.data)
    if arr.ndim != 2 or arr.shape[0] % 16 != 0 or arr.shape[1] % 64 != 0:
        return None
    return arr


def _as_shuffleable_mxfp4_b_scale(wd: WeightData) -> np.ndarray | None:
    """Return ``wd`` as a uint8 view if it's a shuffleable MXFP4 B scale.

    Shuffleable when dtype is E8M0 and dims are cell-aligned for
    ``Shuffler.scale_4d_grouped_layout``: ``N % 32 == 0`` (S_MN_BLOCK) and
    ``K_SCALES % 8 == 0`` (S_K_BLOCK). The 2D src reshape used by
    :func:`_shuffle_scale_4d` hardcodes those factors. Returns ``None``
    when not shuffleable. E8M0 bytes are reinterpreted as uint8 for byte
    permutation; the dtype is restored to E8M0 by the caller.
    """
    if wd.dtype != DType.float8_e8m0fnu:
        return None
    try:
        arr = np.from_dlpack(wd.data)
    except (TypeError, BufferError):
        return None
    if arr.dtype != np.uint8:
        arr = arr.view(np.uint8)
    if arr.ndim != 2 or arr.shape[0] % 32 != 0 or arr.shape[1] % 8 != 0:
        return None
    return arr


def _shuffle_b_5d(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP4 expert B bytes into ``Shuffler.b_5d_grouped_layout``.

    Reshape ``[N, K_BYTES]`` row-major into the 5D tile structure
    ``(N0, NLane=16, K0, KLane=4, KPack=16)`` and transpose into
    ``(N0, K0, KLane, NLane, KPack)`` so C-order strides match
    ``b_5d_grouped_layout`` in ``mxfp4_preshuffle_layouts.mojo``. ``dst``
    is a contiguous ``(N, K_BYTES)`` slot the caller owns.
    """
    N, K_BYTES = src.shape
    src_v = src.reshape(N // 16, 16, K_BYTES // 64, 4, 16).transpose(
        0, 2, 3, 1, 4
    )
    dst_v = dst.reshape(N // 16, K_BYTES // 64, 4, 16, 16)
    np.copyto(dst_v, src_v)


def _shuffle_scale_4d(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP4 B-scale bytes into ``Shuffler.scale_4d_grouped_layout``.

    Reshape ``[MN, K_SCALES]`` row-major into the 6D decomposition
    ``(MN_block, MN_pack=2, MN_lane=16, K_block, K_pack=2, K_lane=4)``
    and transpose into the dst axis order
    ``(MN_block, K_block, K_lane, MN_lane, K_pack, MN_pack)`` so C-order
    strides match the 4D-cell byte layout addressed by
    ``Shuffler.scale_4d_byte_off``. Within each i32 cell the bytes land
    in ``(mn_pack, k_pack) = {(0,0), (1,0), (0,1), (1,1)}`` order at
    byte offsets ``{0, 1, 2, 3}`` — what the preb kernel's OPSEL byte
    selector reads.
    """
    MN, K_SCALES = src.shape
    src_v = src.reshape(MN // 32, 2, 16, K_SCALES // 8, 2, 4).transpose(
        0, 3, 5, 2, 4, 1
    )
    dst_v = dst.reshape(MN // 32, K_SCALES // 8, 4, 16, 2, 2)
    np.copyto(dst_v, src_v)


def preshuffle_mxfp4_b_experts(
    state_dict: dict[str, WeightData],
) -> None:
    """MXFP4 B preshuffle of all per-expert weights in-place on CPU.

    Walks ``state_dict``, groups expert weights by ``(prefix, proj)``,
    rewrites each group's WeightData entries with the bytes laid out in
    ``b_5d_grouped_layout`` so the AMD ``mxfp4_grouped_matmul_amd_preb``
    kernel reads them with coalesced DRAM->VGPR loads. Experts whose
    dtype/shape isn't MXFP4-packed uint8 with tile-aligned dims are
    silently skipped (they fall through to the row-major kernel).

    One numpy buffer per ``(prefix, proj)`` group keeps allocation count
    at ~180. Per-expert allocations would mean ~70k mmap chunks, blowing
    past glibc's M_MMAP_MAX (65536).
    """
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name in state_dict:
        if m := _EXPERT_WEIGHT_RE.match(name):
            groups[m["prefix"], m["proj"]].append(name)

    if not groups:
        return

    t0 = time.perf_counter()
    n_total = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for names in groups.values():
            shuffleable = [
                (name, arr)
                for name in names
                if (arr := _as_shuffleable_mxfp4_b(state_dict[name]))
                is not None
            ]
            if not shuffleable:
                continue

            kept_names, srcs = zip(*shuffleable, strict=True)
            N, K_BYTES = srcs[0].shape
            buf = np.empty((len(srcs), N, K_BYTES), dtype=np.uint8)
            list(pool.map(_shuffle_b_5d, srcs, buf))
            for name, slot in zip(kept_names, buf, strict=True):
                state_dict[name] = WeightData.from_numpy(
                    slot, name=state_dict[name].name
                )
            n_total += len(srcs)

    logger.info(
        "MXFP4 B preshuffle: %d experts across %d groups in %.1fs",
        n_total,
        len(groups),
        time.perf_counter() - t0,
    )


def preshuffle_mxfp4_b_scales(
    state_dict: dict[str, WeightData],
) -> None:
    """MXFP4 B-scale preshuffle of all per-expert scales in-place on CPU.

    Walks ``state_dict``, groups expert scales by ``(prefix, proj)``,
    rewrites each group's WeightData entries with bytes laid out in
    ``scale_4d_grouped_layout`` so the AMD preb grouped-matmul kernel
    can issue direct-VGPR i32 scale loads (one 2x2 cell per lane).
    Scales whose dtype isn't E8M0 or whose dims aren't cell-aligned
    (``N % 32 == 0`` and ``K_SCALES % 8 == 0``) are silently skipped.

    Companion to :func:`preshuffle_mxfp4_b_experts`; should be called
    immediately after it so weight and scale layouts stay in sync.
    """
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name in state_dict:
        if m := _EXPERT_SCALE_RE.match(name):
            groups[m["prefix"], m["proj"]].append(name)

    if not groups:
        return

    t0 = time.perf_counter()
    n_total = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for names in groups.values():
            shuffleable = [
                (name, arr)
                for name in names
                if (arr := _as_shuffleable_mxfp4_b_scale(state_dict[name]))
                is not None
            ]
            if not shuffleable:
                continue

            kept_names, srcs = zip(*shuffleable, strict=True)
            MN, K_SCALES = srcs[0].shape
            buf = np.empty((len(srcs), MN, K_SCALES), dtype=np.uint8)
            list(pool.map(_shuffle_scale_4d, srcs, buf))
            for name, slot in zip(kept_names, buf, strict=True):
                # from_numpy infers uint8 from the slab dtype; restore the
                # E8M0 metadata so downstream graph-compiler dtype checks
                # (e.g. grouped_dynamic_scaled_mxfp4_matmul) still pass.
                state_dict[name] = dataclasses.replace(
                    WeightData.from_numpy(slot, name=state_dict[name].name),
                    dtype=DType.float8_e8m0fnu,
                )
            n_total += len(srcs)

    logger.info(
        "MXFP4 B-scale preshuffle: %d experts across %d groups in %.1fs",
        n_total,
        len(groups),
        time.perf_counter() - t0,
    )


def _dequantize_fp8_attention(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Dequantize rowwise-FP8 attention projections to bfloat16.

    Some MXFP4 Kimi checkpoints keep the MoE in MXFP4 but store the attention
    projections as rowwise FP8. The MLA is built in bf16 when the MoE is FP4,
    so fold each FP8 attention weight into bf16 and drop its scale. Only the
    attention linears are FP8; the layernorms stay bf16. Runs only when the
    checkpoint carries packed-MXFP4 experts, leaving pure-FP8 checkpoints for
    the native FP8 path.
    """
    has_mxfp4_experts = any(
        _EXPERT_WEIGHT_RE.match(name) and wd.dtype == DType.uint8
        for name, wd in state_dict.items()
    )
    if not has_mxfp4_experts:
        return state_dict

    converted: dict[str, WeightData] = {}
    n_dequant = 0
    for name, wd in state_dict.items():
        if ".self_attn." not in name:
            converted[name] = wd
        elif name.endswith(".weight_scale"):
            # Folded into the dequantized weight, so drop it.
            pass
        elif wd.dtype == DType.float8_e4m3fn:
            scale = state_dict[name.removesuffix(".weight") + ".weight_scale"]
            converted[name] = dequantize_rowwise_fp8(
                wd, scale, wd.name, out_dtype=DType.bfloat16
            )
            n_dequant += 1
        else:
            converted[name] = wd

    if n_dequant:
        logger.info(
            "FP8 attention dequant: %d projections to bfloat16", n_dequant
        )
    return converted


def _convert_merged_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    drop_kv_scales: bool = False,
) -> dict[str, WeightData]:
    """Convert a full Kimi checkpoint to MAX module keys in a single pass.

    Vision keys (``vision_tower.*``, ``mm_projector.*``,
    ``multi_modal_projector.*``) are renamed and cast to bfloat16.
    Language-model keys (``language_model.*``) are renamed via
    :data:`KIMI_SAFETENSOR_MAP` with MTP-layer pruning.
    All other checkpoint keys are silently ignored.

    Args:
        state_dict: Raw HuggingFace checkpoint.
        huggingface_config: Model config.  If the config wraps a language-model
            sub-config under ``.text_config`` (as in KimiK25Config), that
            sub-config is used for MTP layer-index lookup; otherwise the
            top-level config is used directly.
        drop_kv_scales: When ``True``, language-model keys ending in
            ``.k_scale`` or ``.v_scale`` are dropped (required for
            FP8 nvidia/Kimi-K2.5-NVFP4 checkpoints).

    Returns:
        Single merged state dict whose keys match the ``KimiK2_5`` module:
        ``vision_encoder.*`` and ``language_model.*``.
    """
    llm_config = getattr(huggingface_config, "text_config", huggingface_config)
    mtp_layer_idx = llm_config.num_hidden_layers

    result: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # --- Vision tower + projector keys ---
        if any(checkpoint_name.startswith(p) for p in _VISION_PREFIXES):
            name = _rename_vision_key(checkpoint_name)
            result[name] = _cast_vision_weight(checkpoint_name, weight)
            continue

        # --- Language model keys ---
        if not checkpoint_name.startswith("language_model."):
            continue

        name = checkpoint_name
        for before, after in KIMI_SAFETENSOR_MAP.items():
            name = name.replace(before, after)

        # TODO(E2EOPT-673): Support MTP.  Prune the speculative MTP layer.
        # See https://github.com/deepseek-ai/DeepSeek-V3/blob/4592be48c07f036b32ef971474068aebc489e3e7/inference/convert.py#L53-L54
        if name.startswith(f"language_model.layers.{mtp_layer_idx}."):
            continue
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            continue
        if drop_kv_scales and (
            name.endswith(".k_scale") or name.endswith(".v_scale")
        ):
            continue

        data = weight.data()
        # Safetensors stores E8M0 scales as uint8; reinterpret.
        if name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)
        result[name] = data

    return _dequantize_fp8_attention(result)


# ---------------------------------------------------------------------------
# Public adapters — one per architecture in arch.py
# ---------------------------------------------------------------------------


def convert_kimik2_5_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``KimiK25ForConditionalGeneration`` safetensor checkpoint.

    Handles the ``mm_projector.*`` projector naming convention used by
    nvidia/Kimi-K2.5-NVFP4 checkpoints, and drops FP8 kv-scale weights.

    Args:
        state_dict: Raw checkpoint as dict from weight name to :class:`Weights`.
        huggingface_config: HuggingFace config (required for MTP pruning).

    Returns:
        State dict with ``vision_encoder.*`` and ``language_model.*`` keys
        matching the ``KimiK2_5`` MAX module.
    """
    if huggingface_config is None:
        raise ValueError(
            "convert_kimik2_5_safetensor_state_dict requires huggingface_config"
            " for language model weight conversion (MTP layer pruning)."
        )
    return _convert_merged_state_dict(
        state_dict,
        huggingface_config=huggingface_config,
        drop_kv_scales=True,
    )


def convert_kimivl_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``KimiVLForConditionalGeneration`` safetensor checkpoint.

    Handles the ``multi_modal_projector.*`` projector naming convention used
    by moonshotai/Kimi-VL-A3B checkpoints.

    Args:
        state_dict: Raw checkpoint as dict from weight name to :class:`Weights`.
        huggingface_config: HuggingFace config (required for MTP pruning).

    Returns:
        State dict with ``vision_encoder.*`` and ``language_model.*`` keys
        matching the ``KimiK2_5`` MAX module.
    """
    if huggingface_config is None:
        raise ValueError(
            "convert_kimivl_safetensor_state_dict requires huggingface_config"
            " for language model weight conversion (MTP layer pruning)."
        )
    return _convert_merged_state_dict(
        state_dict,
        huggingface_config=huggingface_config,
        drop_kv_scales=False,
    )


# ---------------------------------------------------------------------------
# Eagle3 draft checkpoint adapter
# ---------------------------------------------------------------------------

# Eagle3 checkpoint key prefix -> Eagle3KimiK25 module path.
_EAGLE3_KEY_MAP: dict[str, str] = {
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "decoder_layer.input_layernorm.",
    "layers.0.self_attn.": "decoder_layer.self_attn.",
    "layers.0.post_attention_layernorm.": "decoder_layer.post_attention_layernorm.",
    "layers.0.mlp.": "decoder_layer.mlp.",
}


def convert_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert an Eagle3 draft checkpoint to Eagle3KimiK25 module keys.

    The Eagle3 checkpoint (``nvidia/Kimi-K2.5-Thinking-Eagle3``) has keys:
    ``fc.*``, ``layers.0.*``, ``norm.*``, ``lm_head.*``.

    All weights are loaded; ``norm`` and ``lm_head`` are kept independent
    from the target model.  Only ``embed_tokens``
    is shared from the target.

    Args:
        state_dict: Raw Eagle3 checkpoint.

    Returns:
        State dict with keys matching ``Eagle3KimiK25`` module hierarchy.
    """
    result: dict[str, WeightData] = {}

    for name, weight in state_dict.items():
        # Apply key mapping
        mapped = False
        for before, after in _EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = _cast_vision_weight(name, weight)
                mapped = True
                break

        if not mapped:
            # fc.*, norm.*, lm_head.* pass through directly
            result[name] = _cast_vision_weight(name, weight)

    return result


# ---------------------------------------------------------------------------
# Llama-style Eagle3 draft checkpoint adapter
# ---------------------------------------------------------------------------

# Llama Eagle3 checkpoint key prefix -> Eagle3MHADraft module path.
# The MHA draft module's layer is flat (single block, no ``decoder_layer``
# namespace), so the mapping strips the single-layer prefix and inlines
# norms.
#
# Checkpoints in the wild use different single-layer prefix conventions
# (``midlayer.``, ``model.layers.0.``, ``layers.0.``). Handle all three.
_LLAMA_EAGLE3_KEY_MAP: dict[str, str] = {
    "midlayer.hidden_norm.": "hidden_norm.",
    "midlayer.input_layernorm.": "input_layernorm.",
    "midlayer.self_attn.": "self_attn.",
    "midlayer.post_attention_layernorm.": "post_attention_layernorm.",
    "midlayer.mlp.": "mlp.",
    "model.layers.0.hidden_norm.": "hidden_norm.",
    "model.layers.0.input_layernorm.": "input_layernorm.",
    "model.layers.0.self_attn.": "self_attn.",
    "model.layers.0.post_attention_layernorm.": "post_attention_layernorm.",
    "model.layers.0.mlp.": "mlp.",
    "model.norm.": "norm.",
    "model.embed_tokens.": "embed_tokens.",
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "input_layernorm.",
    "layers.0.self_attn.": "self_attn.",
    "layers.0.post_attention_layernorm.": "post_attention_layernorm.",
    "layers.0.mlp.": "mlp.",
}


def convert_llama_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``LlamaForCausalLMEagle3`` checkpoint to ``Eagle3MHADraft``.

    Handles both ``model.*``-prefixed (standard HF Llama) and
    ``layers.0.*``-prefixed (EAGLE-export) checkpoints. ``fc.*``,
    ``norm.*``, and ``lm_head.*`` pass through.
    """
    result: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        mapped = False
        for before, after in _LLAMA_EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = _cast_vision_weight(name, weight)
                mapped = True
                break
        if not mapped:
            # fc.*, norm.*, lm_head.* and any other top-level keys.
            result[name] = _cast_vision_weight(name, weight)
    return result
