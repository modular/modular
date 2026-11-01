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
fused gate/up, stacked experts, de-padded ``A_log``) that ``model.py``'s
``load_kimi_k3_weights`` applies to a live module -- re-derived locally
here rather than imported, to avoid a dependency on V2's package. Layer,
KDA, and expert counts instead come from :class:`~.model_config.KimiK3Config`,
the single source of truth for those.

Loads MXFP4-packed routed experts now that ``moe_latent.py`` has a compute
path for them (the AMD CDNA4 W4A8 arm). The packed bytes fuse and stack the
same way as the dense case; only the E8M0 scales need relabeling, since
safetensors has no such dtype. A checkpoint without packed experts still
loads through the dense path unchanged.

Under tensor parallelism each fused tensor is written interleaved per TP
rank (:func:`interleave_for_tp`) rather than block-major. The loader splits
the checkpoint arrays contiguously along the sharded axis on the host and
never reshards them on device, so each rank's contiguous slice must already
hold its share of every fused block.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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
    rows = [_elements(part) for part in parts]
    shape = (sum(r.shape[0] for r in rows), *rows[0].shape[1:])
    out = _stack_output(shape, rows[0].dtype)
    pairs = []
    for p, src, dst in _interleaved_slices([r.shape[0] for r in rows], tp_size):
        pairs.append((out[dst], rows[p][src]))
    _copy_into(pairs)
    return _rewrap(_settle(out), parts[0], name)


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


# Each expert copy is a memcpy plus a page fault per source page, and numpy
# releases the GIL for both, so overlapping several copies in threads hides
# that latency; bounded low since the win is concurrent faults, not cores.
_STACK_WORKERS = min(16, os.cpu_count() or 8)

# Above this, an expert stack is backed by a file instead of anonymous memory,
# so its pages stay reclaimable under memory pressure.
_SPILL_BYTES = int(os.environ.get("MAX_KIMI_K3_STACK_SPILL_BYTES", 4 << 30))
_SPILL_DIR = os.environ.get("MAX_KIMI_K3_STACK_SPILL_DIR") or None


def _stack_output(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Allocates an expert stack, spilling to a file past ``_SPILL_BYTES``.

    The file is unlinked as soon as it is mapped: the mapping keeps the inode
    alive, so the space is returned when the array is freed and nothing is
    left behind if the process dies.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes < _SPILL_BYTES:
        return np.empty(shape, dtype=dtype)

    fd, path = tempfile.mkstemp(prefix="max-k3-expert-stack-", dir=_SPILL_DIR)
    os.close(fd)
    try:
        out = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    finally:
        os.unlink(path)
    return out


def _settle(out: np.ndarray) -> np.ndarray:
    """Writes a spilled stack back, so its pages become reclaimable.

    Spilling makes the bytes file-backed, but a freshly written shared mapping
    is DIRTY, and dirty page cache is no more reclaimable than anonymous
    memory -- so one ``msync`` per stack makes the pages clean and lets the
    kernel drop them under pressure.
    """
    if isinstance(out, np.memmap):
        out.flush()
    return out


def _copy_into(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Runs ``dst[...] = src`` for each pair, concurrently.

    Every ``dst`` is a disjoint slice of one preallocated output, so the
    writes do not race and the peak stays one output rather than two.
    """
    if len(pairs) < 2 or _STACK_WORKERS < 2:
        for dst, src in pairs:
            dst[...] = src
        return

    def one(pair: tuple[np.ndarray, np.ndarray]) -> None:
        dst, src = pair
        dst[...] = src

    with ThreadPoolExecutor(max_workers=_STACK_WORKERS) as pool:
        # list() rather than a bare map(): map is lazy, and an exception in a
        # worker only surfaces when its result is consumed.
        list(pool.map(one, pairs))


def _stack_experts_arrays(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Stacks per-expert arrays into ``[E, *parts[0].shape]``.

    ``np.stack`` allocates the same output and fills it serially; this fills
    it through :func:`_copy_into` instead.
    """
    if not parts:
        raise ValueError("_stack_experts_arrays needs at least one expert")
    out = _stack_output((len(parts), *parts[0].shape), parts[0].dtype)
    _copy_into([(out[e], p) for e, p in enumerate(parts)])
    return _settle(out)


def _stack_gate_up_arrays(
    gate: Sequence[np.ndarray], up: Sequence[np.ndarray], tp_size: int
) -> np.ndarray:
    """Stacks per-expert gate and up into ``[E, gate_rows + up_rows, cols]``.

    Each expert's rows are interleaved per TP rank, as
    :func:`interleave_for_tp` lays them out. Writes each rank's gate and up
    slices into a preallocated output instead of fusing every expert first
    and stacking the list, which costs the same result at twice the peak
    memory and serializes every expert's page faults.
    """
    if not gate:
        raise ValueError("_stack_gate_up_arrays needs at least one expert")
    moves = _interleaved_slices([gate[0].shape[0], up[0].shape[0]], tp_size)
    out = _stack_output(
        (len(gate), gate[0].shape[0] + up[0].shape[0], *gate[0].shape[1:]),
        gate[0].dtype,
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for e, parts in enumerate(zip(gate, up, strict=True)):
        for p, src, dst in moves:
            pairs.append((out[e, dst], parts[p][src]))
    _copy_into(pairs)
    return _settle(out)


def _stack_expert_weights(
    w1: list[WeightData],
    w2: list[WeightData],
    w3: list[WeightData],
    name: str,
    tp_size: int,
) -> tuple[WeightData, WeightData]:
    """Stacks per-expert checkpoint tensors into the two grouped tensors.

    ``KimiK3LatentMoE.experts_gate_up_proj`` and ``experts_down_proj`` take
    them. ``w1`` is gate, ``w3`` is up and ``w2`` is down: the checkpoint's own
    numbering does not follow the data flow. The MXFP4 case's ``weight_scale``
    tensors go through the same stacking, since whether the values are packed
    or dense makes no difference to it -- so the scales land in the same
    per-rank interleaved row order as the weights they scale.
    """
    gate_up = _stack_gate_up_arrays(
        [_elements(g) for g in w1], [_elements(u) for u in w3], tp_size
    )
    down = _stack_experts_arrays([_elements(w) for w in w2])
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


def tp_degree(pipeline_config: Any) -> int:
    """Returns the TP degree the module tree shards fused tensors over.

    The mesh ``pipeline_model.py`` builds is 1-D over every device: ``tp``,
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
    every fused tensor (gate/up, routed-expert gate/up and its MXFP4 scales,
    the KDA qkv projection, the KDA qkv conv) is interleaved per TP rank by
    :func:`interleave_for_tp`.

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
                packed1, packed2, packed3, ffn_dst, tp_size
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
                scale1, scale2, scale3, f"{ffn_dst}_scale", tp_size
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
            gate_up, down = _stack_expert_weights(w1, w2, w3, ffn_dst, tp_size)
            out[f"{ffn_dst}.experts_gate_up_proj"] = gate_up
            out[f"{ffn_dst}.experts_down_proj"] = down
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
