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
"""Maps a Kimi K3 safetensors checkpoint onto the module tree.

The tree is the ModuleV3 registry's layout for the text tower
(:class:`~.model.KimiK3Model`). Builds the
``dict[str, Weights] -> dict[str, WeightData]`` the registry loads from,
applying the same shape transforms (fused qkv projection, fused qkv conv,
fused gate/up, de-padded ``A_log``) that ``model.py``'s
``load_kimi_k3_weights`` applies to a live module -- re-derived locally
here rather than imported, to avoid a dependency on V2's package. Layer,
KDA, and expert counts instead come from :class:`~.model_config.KimiK3Config`,
the single source of truth for those.

The routed experts load per expert, as zero-copy views of the checkpoint's
own buffers: ``moe_latent.py`` stacks them in the graph, at model init.
MXFP4-packed experts load the same way; only their E8M0 scales need
relabeling, since safetensors has no such dtype.

Under tensor parallelism each fused tensor is written interleaved per TP
rank (:func:`interleave_for_tp`) rather than block-major. The loader splits
the checkpoint arrays contiguously along the sharded axis on the host and
never reshards them on device, so each rank's contiguous slice must already
hold its share of every fused block.
"""

from __future__ import annotations

from collections.abc import Sequence
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
# (`kda_attention.py`), and the router's `e_score_correction_bias` (built
# from the explicit `correction_bias_dtype` in `moe_router.py`). The
# router's `gate_score.weight` follows the working dtype like everything
# else: the checkpoint stores it bf16 and the gate GEMV computes
# bf16 x bf16 -> fp32, so pinning it fp32 would only double the replicated
# weight. `b_proj`/`conv_weight`/`o_norm.weight` only get cast to fp32
# transiently at compute time -- their parameters themselves follow the
# model's working dtype like everything else, so they are deliberately not
# pinned here either.
_FP32_SUFFIXES = (
    ".a_log",
    ".dt_bias",
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


def _interleaved_slices(
    widths: Sequence[int], tp_size: int
) -> list[tuple[int, slice, slice]]:
    """Returns ``(part, src, dst)`` row ranges for :func:`interleave_for_tp`.

    Raises:
        ValueError: If a part's width does not split evenly over ``tp_size``.
    """
    for width in widths:
        if width % tp_size:
            raise ValueError(
                f"Kimi K3: fused block width {width} is not divisible by the"
                f" TP degree {tp_size}."
            )
    moves: list[tuple[int, slice, slice]] = []
    dst = 0
    for r in range(tp_size):
        for p, width in enumerate(widths):
            per_rank = width // tp_size
            src = slice(r * per_rank, (r + 1) * per_rank)
            moves.append((p, src, slice(dst, dst + per_rank)))
            dst += per_rank
    return moves


def interleave_for_tp(
    parts: Sequence[np.ndarray], tp_size: int, axis: int = 0
) -> np.ndarray:
    """Concatenates ``parts`` along ``axis``, grouped per TP rank.

    The TP layers shard a fused weight contiguously along ``axis``, which
    over a plain ``[p0; p1; ...]`` concat would hand one rank whole parts
    and another none. This writes ``[p0_r0; p1_r0; ...; p0_r1; p1_r1; ...]``
    instead, so rank ``r``'s contiguous shard is its slice of every part.
    With ``tp_size == 1`` it is the plain concatenation.

    Args:
        parts: The blocks to fuse, equal in every dim but ``axis``.
        tp_size: The TP degree the fused weight is sharded over.
        axis: The fused axis.

    Returns:
        The fused array.

    Raises:
        ValueError: If a part's width along ``axis`` does not split evenly
            over ``tp_size``.
    """
    lead = (slice(None),) * axis
    moves = _interleaved_slices([p.shape[axis] for p in parts], tp_size)
    return np.concatenate(
        [parts[p][(*lead, src)] for p, src, _ in moves], axis=axis
    )


def _fuse_gate_up(
    gate: WeightData, up: WeightData, name: str, tp_size: int
) -> WeightData:
    """Row-fuses gate and up, interleaved per TP rank.

    Mirrors ``mlp.py``'s ``KimiK3DenseMLP.gate_up_proj`` layout, gate first
    within each rank's slice.
    """
    fused = interleave_for_tp([_elements(gate), _elements(up)], tp_size)
    return _rewrap(fused, gate, name)


def _fuse_qkv_proj(
    q: WeightData,
    k: WeightData,
    v: WeightData,
    name: str,
    tp_size: int,
    dtype: DType | None = None,
) -> WeightData:
    """Merges q, k and v into one ``[3 * inner, hidden]`` projection weight.

    ``KimiK3KDAAttention.qkv_proj`` takes the merged weight, interleaved per
    TP rank like the conv weight that consumes its output. Castable float
    parts are cast to ``dtype`` first (the dtype the final cast pass would
    give each), so parts stored in different float dtypes still merge.

    Raises:
        ValueError: If the parts differ in shape, or still differ in dtype
            after the cast.
    """
    parts = [q, k, v]
    if dtype is not None:
        parts = [
            part.astype(dtype) if part.dtype in _CASTABLE_DTYPES else part
            for part in parts
        ]
    for axis, part in zip(("q", "k", "v"), parts, strict=True):
        if part.dtype != parts[0].dtype or tuple(part.shape) != tuple(
            parts[0].shape
        ):
            raise ValueError(
                f"Kimi K3: {axis}_proj has dtype {part.dtype} and shape"
                f" {tuple(part.shape)}, but must match q_proj's"
                f" {parts[0].dtype} and {tuple(parts[0].shape)}."
            )
    fused = interleave_for_tp([_elements(part) for part in parts], tp_size)
    return _rewrap(fused, parts[0], name)


def _fuse_qkv_conv_weight(
    q: WeightData, k: WeightData, v: WeightData, name: str, tp_size: int
) -> WeightData:
    """Packs three ``[C, 1, K]`` depthwise convs into one ``[3C, K]`` weight.

    ``KimiK3KDAAttention.conv_weight`` takes the packed weight, interleaved
    per TP rank.
    """
    arrays = [_elements(w) for w in (q, k, v)]
    fused = interleave_for_tp(
        [a.reshape(a.shape[0], -1) for a in arrays], tp_size
    )
    return _rewrap(fused, q, name)


def _canonical_a_log(
    a_log: WeightData, num_heads: int, head_dim: int, name: str
) -> WeightData:
    """Broadcasts per-head gate coefficients.

    Exact-zero storage padding is dropped, mirroring ``model.py``'s
    ``_canonical_a_log``.
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


def tp_degree(pipeline_config: Any) -> int:
    """Returns the TP degree the module tree shards fused tensors over.

    The mesh ``KimiK3Config.apply_runtime`` builds is 1-D over every device: ``tp``,
    unless ``ep_size > 1`` makes it ``ep``, which shards nothing along TP.
    Under data parallelism (which ``KimiK3Config`` rejects) nothing is
    TP-sharded either.
    """
    if (
        pipeline_config is None
        or pipeline_config.runtime.ep_size > 1
        or pipeline_config.model.data_parallel_degree > 1
    ):
        return 1
    return len(pipeline_config.model.device_specs)


def convert_kimi_k3_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: Any,
) -> dict[str, WeightData]:
    """Converts a Kimi K3 checkpoint's text tower onto the module tree.

    The names it produces are :class:`~.model.KimiK3Model`'s own. Under TP,
    every fused tensor (the dense and shared gate/up, the KDA qkv projection,
    the KDA qkv conv) is interleaved per TP rank by :func:`interleave_for_tp`.
    Each routed expert keeps the checkpoint's own ``w1``/``w2``/``w3`` names
    and buffers.

    Args:
        **unused_kwargs: Accepted so every adapter shares one signature.
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
    tp_size = tp_degree(pipeline_config)

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
            out[f"{attn_dst}.qkv_proj.weight"] = _fuse_qkv_proj(
                _wd(state_dict, f"{attn_src}.q_proj.weight"),
                _wd(state_dict, f"{attn_src}.k_proj.weight"),
                _wd(state_dict, f"{attn_src}.v_proj.weight"),
                f"{attn_dst}.qkv_proj.weight",
                tp_size,
                dtype,
            )
            q_conv, k_conv, v_conv = (
                _wd(state_dict, f"{attn_src}.{axis}_conv1d.weight")
                for axis in ("q", "k", "v")
            )
            out[f"{attn_dst}.conv_weight"] = _fuse_qkv_conv_weight(
                q_conv, k_conv, v_conv, f"{attn_dst}.conv_weight", tp_size
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
                tp_size,
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
        packed = _experts_are_packed(state_dict, moe)
        for e in range(num_experts):
            for w in ("w1", "w2", "w3"):
                src_w = f"{moe}.experts.{e}.{w}"
                dst_w = f"{ffn_dst}.experts.{e}.{w}"
                if not packed:
                    out[f"{dst_w}.weight"] = _wd(state_dict, f"{src_w}.weight")
                    continue
                out[f"{dst_w}.weight"] = _wd(state_dict, f"{src_w}.{_PACKED}")
                out[f"{dst_w}.weight_scale"] = _as_e8m0(
                    _wd(state_dict, f"{src_w}.weight_scale"),
                    f"{dst_w}.weight_scale",
                )
        out[f"{ffn_dst}.shared_experts.gate_up_proj.weight"] = _fuse_gate_up(
            _wd(state_dict, f"{moe}.shared_experts.gate_proj.weight"),
            _wd(state_dict, f"{moe}.shared_experts.up_proj.weight"),
            f"{ffn_dst}.shared_experts.gate_up_proj.weight",
            tp_size,
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
