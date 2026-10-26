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
"""Maps a Kimi K3 safetensors checkpoint onto the ModuleV3 registry's module
tree for the TEXT tower (:class:`~.model.KimiK3Model`).

Builds the ``dict[str, Weights] -> dict[str, WeightData]`` the registry
loads from, applying the same shape transforms (fused qkv conv, fused
gate/up, stacked experts, de-padded ``A_log``) that ``model.py``'s
``load_kimi_k3_weights`` applies to a live module -- re-derived locally
here rather than imported, to avoid a dependency on V2's package. Layer,
KDA, and expert counts instead come from
:class:`~.model_config.KimiK3Config`, the single source of truth for those.

Loads MXFP4-packed routed experts now that ``moe_latent.py`` has a compute
path for them (the AMD CDNA4 W4A8 arm). The packed bytes fuse and stack the
same way as the dense case; only the E8M0 scales need relabeling, since
safetensors has no such dtype. A checkpoint without packed experts still
loads through the dense path unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

from .model import KimiK3ModelConfig
from .model_config import KimiK3Config

_TEXT = "language_model.model"

_DIRECT_LAYER_WEIGHTS = (
    ("input_layernorm_weight", "input_layernorm.weight"),
    ("post_attention_layernorm_weight", "post_attention_layernorm.weight"),
    ("self_attention_res_proj_weight", "self_attention_res_proj.weight"),
    ("self_attention_res_norm_weight", "self_attention_res_norm.weight"),
    ("mlp_res_proj_weight", "mlp_res_proj.weight"),
    ("mlp_res_norm_weight", "mlp_res_norm.weight"),
)

# Per-sublayer tensors that keep the checkpoint's own leaf name, just
# appended to ``self_attn``'s MAX attribute path (``attention``) instead of
# the checkpoint's own ``self_attn``.
_DIRECT_ATTN_WEIGHTS = {
    "kda": (
        "q_proj",
        "k_proj",
        "v_proj",
        "f_a_proj",
        "f_b_proj",
        "b_proj",
        "o_norm",
        "g_proj",
        "o_proj",
    ),
    "mla": (
        "q_a_proj",
        "q_a_layernorm",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_a_layernorm",
        "kv_b_proj",
        "g_proj",
        "o_proj",
    ),
}

_PACKED = "weight_packed"

# Weights whose MAX parameter is explicitly declared fp32 regardless of the
# model's working dtype: KDA's `a_log`/`dt_bias` gate parameters
# (`kda_attention.py`), and the router's `gate_score`/`e_score_correction_bias`
# (built inside a `default_dtype(float32)` context in `moe_router.py`).
# `b_proj`/`conv_weight`/`o_norm.weight` only get cast to fp32 transiently at
# compute time -- their parameters themselves follow the model's working
# dtype like everything else, so they are deliberately not pinned here.
_FP32_SUFFIXES = (
    ".a_log",
    ".dt_bias",
    ".gate.gate_score.weight",
    ".gate.e_score_correction_bias",
)

# Plain checkpoint float dtypes it's safe to resolve to the model's working
# dtype. An MXFP4-packed expert's bytes (uint8) and its E8M0 scale are
# already in their final, non-float dtype and must pass through untouched.
_CASTABLE_DTYPES = (DType.float32, DType.float16, DType.bfloat16, DType.float64)


def _wd(state_dict: dict[str, Weights], key: str) -> WeightData:
    return state_dict[key].data()


def _elements(data: WeightData) -> np.ndarray:
    """Returns a numpy view of a weight's elements, bf16 viewed as ``uint16``.

    numpy has no bfloat16, so ``np.from_dlpack`` refuses a bf16 tensor
    outright. Only the SHAPE-LEVEL moves below may use this view --
    reshape, concatenate, stack -- because they relocate elements without
    reading them. Arithmetic on the result would interpret bf16 bit
    patterns as integers.
    """
    buffer = Buffer.from_dlpack(data.data)
    if data.dtype == DType.bfloat16:
        buffer = buffer.view(DType.uint16)
    return np.from_dlpack(buffer)


def _rewrap(array: np.ndarray, like: WeightData, name: str) -> WeightData:
    """Reinterprets a shape-moved result as ``like``'s dtype."""
    contiguous = np.ascontiguousarray(array)
    shape = tuple(int(d) for d in contiguous.shape)
    return WeightData(
        data=Buffer.from_dlpack(contiguous).view(dtype=like.dtype, shape=shape),
        name=name,
        dtype=like.dtype,
        shape=Shape(shape),
        quantization_encoding=like.quantization_encoding,
    )


def _fuse_gate_up(gate: WeightData, up: WeightData, name: str) -> WeightData:
    """Row-concatenates gate and up, gate first; mirrors ``mlp.py``'s
    ``KimiK3DenseMLP.gate_up_proj`` layout."""
    fused = np.concatenate([_elements(gate), _elements(up)], axis=0)
    return _rewrap(fused, gate, name)


def _fuse_qkv_conv_weight(
    q: WeightData, k: WeightData, v: WeightData, name: str
) -> WeightData:
    """Packs three ``[C, 1, K]`` depthwise convs into the one ``[3C, K]``
    weight ``KimiK3KDAAttention.conv_weight`` takes."""
    arrays = [_elements(w) for w in (q, k, v)]
    fused = np.concatenate([a.reshape(a.shape[0], -1) for a in arrays], axis=0)
    return _rewrap(fused, q, name)


def _canonical_a_log(
    a_log: WeightData, num_heads: int, head_dim: int, name: str
) -> WeightData:
    """Broadcasts per-head gate coefficients, removing exact-zero storage
    padding; mirrors ``model.py``'s ``_canonical_a_log``.

    The published checkpoint stores ``num_heads`` active coefficients in a
    longer vector with a zero tail (e.g. 96 of 128). The official
    initializer and independent runtime loaders index its active prefix by
    head, regardless of the stored length.
    """
    values = _elements(a_log)
    if values.ndim != 1 or values.size < num_heads:
        raise ValueError(
            f"Kimi K3: A_log has shape {values.shape}; expected a per-head"
            f" vector with at least {num_heads} coefficients."
        )
    if not np.all(np.isfinite(values[:num_heads])):
        raise ValueError(
            "Kimi K3: active A_log head coefficients must be finite."
        )
    if np.any(values[num_heads:] != 0):
        raise ValueError("Kimi K3: A_log storage padding must be exactly zero.")
    broadcast = np.broadcast_to(values[:num_heads, None], (num_heads, head_dim))
    return _rewrap(broadcast, a_log, name)


def _stack_expert_weights(
    w1: list[WeightData], w2: list[WeightData], w3: list[WeightData], name: str
) -> tuple[WeightData, WeightData]:
    """Stacks per-expert checkpoint tensors into the two grouped tensors
    ``KimiK3LatentMoE.experts_gate_up_proj``/``experts_down_proj`` take.

    ``w1`` is gate, ``w3`` is up, ``w2`` is down -- the checkpoint's own
    numbering does not follow the data flow. Also reused for the MXFP4
    case's ``weight_scale`` tensors, since the stacking itself is the same
    whether the values are packed or dense.
    """
    gate_up = np.stack(
        [
            np.concatenate([_elements(g), _elements(u)], axis=0)
            for g, u in zip(w1, w3, strict=True)
        ],
        axis=0,
    )
    down = np.stack([_elements(w) for w in w2], axis=0)
    return (
        _rewrap(gate_up, w1[0], f"{name}.experts_gate_up_proj"),
        _rewrap(down, w2[0], f"{name}.experts_down_proj"),
    )


def _experts_are_packed(state_dict: dict[str, Weights], moe: str) -> bool:
    """Whether this MoE block's routed experts arrived MXFP4-packed."""
    return f"{moe}.experts.0.w1.{_PACKED}" in state_dict


def _as_e8m0(data: WeightData, name: str) -> WeightData:
    """Relabels a checkpoint's raw-``uint8``-stored E8M0 scale bytes.

    safetensors has no E8M0 dtype, so this reinterprets the same bytes the
    way ``_elements``'s bf16-as-uint16 view does, just landing on a
    different dtype.
    """
    shape = tuple(int(d) for d in data.shape)
    return WeightData(
        data=Buffer.from_dlpack(data.data).view(
            dtype=DType.float8_e8m0fnu, shape=shape
        ),
        name=name,
        dtype=DType.float8_e8m0fnu,
        shape=Shape(shape),
        quantization_encoding=data.quantization_encoding,
    )


def convert_kimi_k3_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: Any,
) -> dict[str, WeightData]:
    """Converts a Kimi K3 checkpoint's text tower into
    :class:`~.model.KimiK3Model`'s bare module-tree names.

    Args:
        state_dict: Raw checkpoint tensors keyed by their original
            ``language_model.model``-prefixed checkpoint names.
        huggingface_config: Either the top-level multimodal ``config.json``
            or an already-extracted text sub-config -- the same either-shape
            acceptance :meth:`KimiK3Config._get_text_config` documents, via
            which this reads it.

    Returns:
        A dict keyed by :class:`~.model.KimiK3Model`'s own module-tree
        names -- ``KimiK3Model`` is served as the registry's top-level
        module directly, with no ``language_model`` wrapper attribute.
    """
    pipeline_config = unused_kwargs.get("pipeline_config")
    quantization_encoding = (
        _select_quantization_encoding(
            pipeline_config.model, KimiK3Config.DEFAULT_ENCODING
        )
        if pipeline_config is not None
        else KimiK3Config.DEFAULT_ENCODING
    )
    # MXFP4 only quantizes the routed experts; everything else still needs
    # a real working dtype, not the packed experts' own `uint8`.
    dtype = (
        DType.bfloat16
        if quantization_encoding == "float4_e2m1fnx2"
        else supported_encoding_dtype(quantization_encoding)
    )

    text_config = KimiK3Config._get_text_config(huggingface_config)
    model = KimiK3ModelConfig.from_huggingface_config(
        {"text_config": text_config.to_dict()}
    )
    num_layers = model.num_hidden_layers
    kda_layer_indices = set(model.kda_layer_indices)
    first_k_dense_replace = model.first_k_dense_replace
    num_experts = model.num_experts
    kda_num_heads = model.kda_num_heads
    kda_head_dim = model.kda_head_dim

    out: dict[str, WeightData] = {
        "embed_tokens.weight": _wd(state_dict, f"{_TEXT}.embed_tokens.weight"),
        "lm_head_weight": _wd(state_dict, "language_model.lm_head.weight"),
        "final_norm_weight": _wd(state_dict, f"{_TEXT}.norm.weight"),
        "output_attn_res_proj_weight": _wd(
            state_dict, f"{_TEXT}.output_attn_res_proj.weight"
        ),
        "output_attn_res_norm_weight": _wd(
            state_dict, f"{_TEXT}.output_attn_res_norm.weight"
        ),
    }

    for i in range(num_layers):
        src = f"{_TEXT}.layers.{i}"
        dst = f"layers.{i}"
        for dst_name, src_name in _DIRECT_LAYER_WEIGHTS:
            out[f"{dst}.{dst_name}"] = _wd(state_dict, f"{src}.{src_name}")

        is_kda = i in kda_layer_indices
        attn_src = f"{src}.self_attn"
        attn_dst = f"{dst}.attention"
        for name in _DIRECT_ATTN_WEIGHTS["kda" if is_kda else "mla"]:
            out[f"{attn_dst}.{name}.weight"] = _wd(
                state_dict, f"{attn_src}.{name}.weight"
            )
        if is_kda:
            q_conv, k_conv, v_conv = (
                _wd(state_dict, f"{attn_src}.{axis}_conv1d.weight")
                for axis in ("q", "k", "v")
            )
            out[f"{attn_dst}.conv_weight"] = _fuse_qkv_conv_weight(
                q_conv, k_conv, v_conv, f"{attn_dst}.conv_weight"
            )
            out[f"{attn_dst}.a_log"] = _canonical_a_log(
                _wd(state_dict, f"{attn_src}.A_log"),
                kda_num_heads,
                kda_head_dim,
                f"{attn_dst}.a_log",
            )
            dt_bias = _wd(state_dict, f"{attn_src}.dt_bias")
            out[f"{attn_dst}.dt_bias"] = _rewrap(
                _elements(dt_bias).reshape(kda_num_heads, kda_head_dim),
                dt_bias,
                f"{attn_dst}.dt_bias",
            )

        ffn_dst = f"{dst}.ffn"
        if i < first_k_dense_replace:
            out[f"{ffn_dst}.gate_up_proj.weight"] = _fuse_gate_up(
                _wd(state_dict, f"{src}.mlp.gate_proj.weight"),
                _wd(state_dict, f"{src}.mlp.up_proj.weight"),
                f"{ffn_dst}.gate_up_proj.weight",
            )
            out[f"{ffn_dst}.down_proj.weight"] = _wd(
                state_dict, f"{src}.mlp.down_proj.weight"
            )
            continue

        moe = f"{src}.block_sparse_moe"
        out[f"{ffn_dst}.gate.gate_score.weight"] = _wd(
            state_dict, f"{moe}.gate.weight"
        )
        out[f"{ffn_dst}.gate.e_score_correction_bias"] = _wd(
            state_dict, f"{moe}.gate.e_score_correction_bias"
        )
        out[f"{ffn_dst}.routed_expert_down_proj.weight"] = _wd(
            state_dict, f"{moe}.routed_expert_down_proj.weight"
        )
        out[f"{ffn_dst}.routed_expert_up_proj.weight"] = _wd(
            state_dict, f"{moe}.routed_expert_up_proj.weight"
        )
        out[f"{ffn_dst}.routed_expert_norm.weight"] = _wd(
            state_dict, f"{moe}.routed_expert_norm.weight"
        )
        if _experts_are_packed(state_dict, moe):
            packed1 = [
                _wd(state_dict, f"{moe}.experts.{e}.w1.{_PACKED}")
                for e in range(num_experts)
            ]
            packed2 = [
                _wd(state_dict, f"{moe}.experts.{e}.w2.{_PACKED}")
                for e in range(num_experts)
            ]
            packed3 = [
                _wd(state_dict, f"{moe}.experts.{e}.w3.{_PACKED}")
                for e in range(num_experts)
            ]
            gate_up, down = _stack_expert_weights(
                packed1, packed2, packed3, ffn_dst
            )
            out[f"{ffn_dst}.experts_gate_up_proj"] = gate_up
            out[f"{ffn_dst}.experts_down_proj"] = down

            scale1 = [
                _wd(state_dict, f"{moe}.experts.{e}.w1.weight_scale")
                for e in range(num_experts)
            ]
            scale2 = [
                _wd(state_dict, f"{moe}.experts.{e}.w2.weight_scale")
                for e in range(num_experts)
            ]
            scale3 = [
                _wd(state_dict, f"{moe}.experts.{e}.w3.weight_scale")
                for e in range(num_experts)
            ]
            gate_up_scale, down_scale = _stack_expert_weights(
                scale1, scale2, scale3, f"{ffn_dst}_scale"
            )
            out[f"{ffn_dst}.experts_gate_up_proj_scale"] = _as_e8m0(
                gate_up_scale, f"{ffn_dst}.experts_gate_up_proj_scale"
            )
            out[f"{ffn_dst}.experts_down_proj_scale"] = _as_e8m0(
                down_scale, f"{ffn_dst}.experts_down_proj_scale"
            )
        else:
            w1 = [
                _wd(state_dict, f"{moe}.experts.{e}.w1.weight")
                for e in range(num_experts)
            ]
            w2 = [
                _wd(state_dict, f"{moe}.experts.{e}.w2.weight")
                for e in range(num_experts)
            ]
            w3 = [
                _wd(state_dict, f"{moe}.experts.{e}.w3.weight")
                for e in range(num_experts)
            ]
            gate_up, down = _stack_expert_weights(w1, w2, w3, ffn_dst)
            out[f"{ffn_dst}.experts_gate_up_proj"] = gate_up
            out[f"{ffn_dst}.experts_down_proj"] = down
        out[f"{ffn_dst}.shared_experts.gate_up_proj.weight"] = _fuse_gate_up(
            _wd(state_dict, f"{moe}.shared_experts.gate_proj.weight"),
            _wd(state_dict, f"{moe}.shared_experts.up_proj.weight"),
            f"{ffn_dst}.shared_experts.gate_up_proj.weight",
        )
        out[f"{ffn_dst}.shared_experts.down_proj.weight"] = _wd(
            state_dict, f"{moe}.shared_experts.down_proj.weight"
        )

    return {
        name: data.astype(DType.float32)
        if name.endswith(_FP32_SUFFIXES)
        else data.astype(dtype)
        if data.dtype in _CASTABLE_DTYPES
        else data
        for name, data in out.items()
    }
