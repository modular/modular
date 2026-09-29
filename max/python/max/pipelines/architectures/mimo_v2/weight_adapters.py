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
"""Checkpoint -> MAX weights for the MiMo-V2.6-Flash NVFP4 export.

The export is a lossless transcode of Xiaomi's checkpoint, and MAX serves
Xiaomi's format. This adapter undoes the transcode and proves every step
exact, so a checkpoint it cannot read exactly fails to load:

* Routed experts keep their E2M1 codes (``uint8 [N, K/2]``, low nibble
  first) byte for byte. Each pair of E4M3 per-16 scales times the F32
  ``weight_scale_2`` must be one power of two, which becomes the E8M0 per-32
  scale. Layers 3-7 carry subnormal E4M3 bytes, so the decode uses the full
  table. Each layer's experts are stacked in the W4A8 MoE layout, with the
  E8M0 scales in the grouped matmul's own interleaved layout; see
  :func:`_stack_experts`.
* The F32 ``qkv_proj`` and dense MLPs become FP8 E4M3 with 128x128 block
  scales. A block's scale is ``amax / 448`` or one of its float32
  neighbours (``fl(448 * s) / 448 != s`` on about 10% of the export's
  blocks), and the block must dequantize to the stored F32 bit for bit.
* ``qkv_proj`` first goes from the export's global ``[Q; K; V]`` rows back to
  Xiaomi's TP=4 chunk order: in global order a full-attention K block mixes
  two chunks' scales and is not exact. Each chunk is zero-padded to whole
  128-row blocks (full attention: 3,392 -> 3,456 rows), so the scale grid is
  the plain ``[rows / 128, cols / 128]``; see :func:`qkv_chunk_layout`.
* The BF16 router weight is upcast to F32, since the reference scores in F32.
  Everything else passes through, ``lm_head`` included (it is not tied).

Every checkpoint tensor is either consumed or on the ignore list (vision,
audio, speech embeddings and the MTP layers). A missing, unexpected or
mistyped tensor raises, and so does any ``input_scale``: the experts run
with dynamic activation scales and would silently ignore one.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.weights._fp8 import e4m3fn_lut
from max.pipelines.weights.fp4_quantization import (
    MAX_E4M3,
    encode_f32_to_e4m3,
)
from transformers.configuration_utils import PretrainedConfig

from .quant import (
    FP8_BLOCK,
    MLP_PROJECTIONS,
    MXFP4_BLOCK,
    moe_layers,
    validate_checkpoint_config,
)

_IGNORED_PREFIXES = (
    "visual.",
    "audio_encoder.",
    "speech_embeddings.",
    "model.mtp.",
)
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
# E8M0 byte 255 is NaN.
_E8M0_MAX = 254
# The block-scaled matmul's scale granule: 128 rows (4 atoms of 32) by 4
# scale columns.
_SF_ATOM_ROWS = 32
_SF_ATOM_COLS = 4
_SF_GRANULE_ROWS = 4 * _SF_ATOM_ROWS


@dataclass(frozen=True)
class QkvChunkLayout:
    """Row layout of one layer's fused ``qkv_proj`` in Xiaomi's chunk order.

    The weight is ``chunks`` repeats of ``[q | k | v | pad]``: chunk ``r``
    holds the ``r``-th equal slice of the query heads and of the KV heads,
    then zero rows up to a whole 128-row block.
    """

    chunks: int
    q_rows: int
    k_rows: int
    v_rows: int

    @property
    def rows(self) -> int:
        """Rows of real weights per chunk."""
        return self.q_rows + self.k_rows + self.v_rows

    @property
    def padded_rows(self) -> int:
        """Rows per chunk, padding included."""
        return -(-self.rows // FP8_BLOCK) * FP8_BLOCK


def qkv_chunk_layout(config: PretrainedConfig, sliding: bool) -> QkvChunkLayout:
    """Returns the chunk layout of ``qkv_proj`` for one attention type.

    The chunk count is Xiaomi's TP degree, which the export's converter reads
    from ``num_key_value_heads`` (the full-attention KV head count, 4); the
    inverse has to use the same number.

    Args:
        config: The checkpoint's top-level Hugging Face config.
        sliding: Whether the layer is sliding-window.

    Returns:
        The layout.
    """

    def attn(key: str) -> int:
        # As in the reference: sliding layers read ``swa_<key>`` if present.
        value = getattr(config, f"swa_{key}", None) if sliding else None
        return int(getattr(config, key) if value is None else value)

    heads, kv_heads = attn("num_attention_heads"), attn("num_key_value_heads")
    head_dim, v_head_dim = attn("head_dim"), attn("v_head_dim")
    chunks = config.num_key_value_heads
    if heads % chunks or kv_heads % chunks:
        raise ValueError(
            f"MiMo-V2: {heads} query and {kv_heads} KV heads do not split "
            f"into {chunks} qkv_proj chunks."
        )
    return QkvChunkLayout(
        chunks=chunks,
        q_rows=heads * head_dim // chunks,
        k_rows=kv_heads * head_dim // chunks,
        v_rows=kv_heads * v_head_dim // chunks,
    )


def qkv_to_chunk_order(
    weight: npt.NDArray[np.float32], layout: QkvChunkLayout, name: str
) -> npt.NDArray[np.float32]:
    """Reorders global ``[Q; K; V]`` rows into padded chunk order.

    This is the inverse of the export's conversion from Xiaomi's layout.

    Args:
        weight: The ``[chunks * layout.rows, K]`` checkpoint tensor.
        layout: The layer's chunk layout.
        name: The tensor name, for errors.

    Returns:
        The ``[chunks * layout.padded_rows, K]`` tensor.
    """
    c, q, k, v = layout.chunks, layout.q_rows, layout.k_rows, layout.v_rows
    if weight.ndim != 2 or weight.shape[0] != c * layout.rows:
        raise ValueError(
            f"{name}: shape {weight.shape} does not match {c} chunks of "
            f"{q} + {k} + {v} rows."
        )
    cols = weight.shape[1]
    out = np.zeros((c, layout.padded_rows, cols), dtype=weight.dtype)
    out[:, :q] = weight[: c * q].reshape(c, q, cols)
    out[:, q : q + k] = weight[c * q : c * (q + k)].reshape(c, k, cols)
    out[:, q + k : layout.rows] = weight[c * (q + k) :].reshape(c, v, cols)
    return out.reshape(c * layout.padded_rows, cols)


def e8m0_scales_from_nvfp4(
    weight_scale: npt.NDArray[np.uint8],
    weight_scale_2: npt.NDArray[np.float32],
    name: str,
) -> npt.NDArray[np.uint8]:
    """Repacks NVFP4 scales as the MXFP4 scales they were transcoded from.

    Args:
        weight_scale: The E4M3 bytes, ``[N, K/16]``, one per 16 elements.
        weight_scale_2: The F32 per-tensor scale, one element.
        name: The projection name, for errors.

    Returns:
        The E8M0 bytes, ``[N, K/32]``.

    Raises:
        ValueError: On the first 32-element block whose two scales differ or
            whose scale is not a power of two in the E8M0 range.
    """
    if weight_scale.ndim != 2 or weight_scale.shape[1] % 2:
        raise ValueError(
            f"{name}: weight_scale shape {weight_scale.shape} is not [N, K/16] "
            "with an even K/16."
        )
    if weight_scale_2.size != 1:
        raise ValueError(
            f"{name}: weight_scale_2 has {weight_scale_2.size} elements, not 1."
        )
    global_scale = float(weight_scale_2.reshape(-1)[0])
    # The combined scale depends only on the byte, so decode all 256 once.
    # float64 holds the product of a 4-bit and a 24-bit significand exactly.
    with np.errstate(invalid="ignore", over="ignore"):
        combined = e4m3fn_lut().astype(np.float64) * global_scale
    mantissa, exponent = np.frexp(combined)
    biased = exponent.astype(np.int64) + 126
    representable = (
        np.isfinite(combined)
        & (combined > 0)
        & (mantissa == 0.5)
        & (biased >= 0)
        & (biased <= _E8M0_MAX)
    )
    first, second = weight_scale[:, 0::2], weight_scale[:, 1::2]
    conforming = (first == second) & representable[first]
    if not conforming.all():
        bad = np.argwhere(~conforming)
        row, block = (int(i) for i in bad[0])
        pair = (int(first[row, block]), int(second[row, block]))
        reason = (
            "its two scales differ"
            if pair[0] != pair[1]
            else "its scale is not a power of two in the E8M0 range"
        )
        raise ValueError(
            f"{name}: MXFP4 block (row {row}, block {block}) is not exactly "
            f"representable, {reason}: E4M3 bytes "
            f"{pair[0]:#04x}, {pair[1]:#04x} with weight_scale_2 "
            f"{global_scale!r} ({len(bad)} non-conforming blocks)."
        )
    return biased.astype(np.uint8)[first]


def interleaved_scale_shape(rows: int, cols: int) -> list[int]:
    """Returns the kernel-layout shape of one expert's ``[rows, cols]`` scales.

    Raises:
        ValueError: If the block is not whole 128-row by 4-column granules.
    """
    if rows % _SF_GRANULE_ROWS or cols % _SF_ATOM_COLS:
        raise ValueError(
            f"MiMo-V2: expert scales [{rows}, {cols}] are not whole "
            f"{_SF_GRANULE_ROWS}x{_SF_ATOM_COLS} interleave granules."
        )
    return [
        rows // _SF_GRANULE_ROWS,
        cols // _SF_ATOM_COLS,
        _SF_ATOM_ROWS,
        _SF_GRANULE_ROWS // _SF_ATOM_ROWS,
        _SF_ATOM_COLS,
    ]


def interleave_e8m0(block: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Permutes one expert's row-major E8M0 scales into the kernel layout.

    The SM100 block-scaled grouped matmul reads element ``(r, c)`` at
    ``[r // 128, c // 4, r % 32, (r % 128) // 32, c % 4]``, where
    ``set_scale_factor`` (``max/kernels/src/linalg/fp4_utils.mojo``) stores
    it. Rows are the outer axis, so a whole-granule row range of the result
    is the interleave of those rows, and tensor parallelism can slice the
    stored layout directly.

    Args:
        block: The ``[N, K/32]`` scales.

    Returns:
        The ``[N/128, K/128, 32, 4, 4]`` scales.
    """
    rows, cols = block.shape
    shape = interleaved_scale_shape(rows, cols)
    atoms = block.reshape(shape[0], shape[3], shape[2], shape[1], shape[4])
    return np.ascontiguousarray(atoms.transpose(0, 3, 2, 1, 4))


def _encode_tiles(
    tiles: npt.NDArray[np.float32], scale: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    """E4M3 codes of ``tiles / scale``, and which tiles decode bit-exactly."""
    scale = scale[..., None, None]
    codes = encode_f32_to_e4m3(tiles / scale)
    decoded = e4m3fn_lut()[codes] * scale
    exact = (decoded.view(np.uint32) == tiles.view(np.uint32)).all(
        axis=(-2, -1)
    )
    return codes, exact


def fp8_block_scaled_from_float32(
    weight: npt.NDArray[np.float32], name: str
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Encodes an F32 weight as FP8 E4M3 with exact 128x128 block scales.

    Args:
        weight: The ``[N, K]`` weight; both dims multiples of 128.
        name: The tensor name, for errors.

    Returns:
        The E4M3 bytes ``[N, K]`` and the F32 scales ``[N / 128, K / 128]``.

    Raises:
        ValueError: On a block with no scale that reproduces it bit for bit.
    """
    rows, cols = weight.shape
    if rows % FP8_BLOCK or cols % FP8_BLOCK:
        raise ValueError(
            f"{name}: shape {weight.shape} is not whole {FP8_BLOCK}x"
            f"{FP8_BLOCK} blocks."
        )
    tiles = weight.reshape(
        rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK
    ).swapaxes(1, 2)
    amax = np.abs(tiles).max(axis=(2, 3))
    scale = np.where(amax > 0, amax / np.float32(MAX_E4M3), np.float32(1.0))
    scale = scale.astype(np.float32)
    codes, exact = _encode_tiles(tiles, scale)
    for direction in (np.inf, 0.0):
        if exact.all():
            break
        retry = ~exact
        candidate = np.nextafter(scale[retry], np.float32(direction))
        retry_codes, retry_exact = _encode_tiles(tiles[retry], candidate)
        fixed = retry.copy()
        fixed[retry] = retry_exact
        scale[fixed] = candidate[retry_exact]
        codes[fixed] = retry_codes[retry_exact]
        exact |= fixed
    if not exact.all():
        bad = np.argwhere(~exact)
        row, col = (int(i) for i in bad[0])
        raise ValueError(
            f"{name}: FP8 block (row block {row}, column block {col}) does "
            "not dequantize to the stored F32 under any scale near "
            f"amax / {MAX_E4M3:g} ({len(bad)} inexact blocks)."
        )
    return np.ascontiguousarray(codes.swapaxes(1, 2)).reshape(rows, cols), scale


def _required_tensors(config: PretrainedConfig) -> dict[str, DType]:
    """Returns every checkpoint tensor the model reads, with its dtype."""
    bf16, f32 = DType.bfloat16, DType.float32
    required = {
        "model.embed_tokens.weight": bf16,
        "model.norm.weight": bf16,
        "lm_head.weight": bf16,
    }
    moe = set(moe_layers(config))
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        sliding = config.hybrid_layer_pattern[layer] == 1
        required |= {
            f"{prefix}input_layernorm.weight": bf16,
            f"{prefix}post_attention_layernorm.weight": bf16,
            f"{prefix}self_attn.qkv_proj.weight": f32,
            f"{prefix}self_attn.o_proj.weight": bf16,
        }
        if getattr(
            config,
            "add_swa_attention_sink_bias"
            if sliding
            else "add_full_attention_sink_bias",
            False,
        ):
            required[f"{prefix}self_attn.attention_sink_bias"] = bf16
        if layer not in moe:
            for proj in MLP_PROJECTIONS:
                required[f"{prefix}mlp.{proj}.weight"] = f32
            continue
        required[f"{prefix}mlp.gate.weight"] = bf16
        required[f"{prefix}mlp.gate.e_score_correction_bias"] = f32
        for expert in range(config.n_routed_experts):
            for proj in MLP_PROJECTIONS:
                base = f"{prefix}mlp.experts.{expert}.{proj}"
                required[f"{base}.weight"] = DType.uint8
                required[f"{base}.weight_scale"] = DType.float8_e4m3fn
                required[f"{base}.weight_scale_2"] = f32
    return required


def _check_tensor_set(names: set[str], required: Mapping[str, DType]) -> None:
    """Raises unless ``names`` is the required set plus ignored tensors."""
    if scales := sorted(n for n in names if n.endswith(".input_scale")):
        raise ValueError(
            f"MiMo-V2: the checkpoint has {len(scales)} input_scale "
            f"tensor(s), e.g. {scales[0]!r}. The experts quantize activations "
            "dynamically and would silently ignore a static scale."
        )
    present = {n for n in names if not n.startswith(_IGNORED_PREFIXES)}
    if unexpected := sorted(present - required.keys()):
        raise ValueError(
            f"MiMo-V2: {len(unexpected)} checkpoint tensor(s) are neither "
            f"read nor ignored, e.g. {unexpected[:3]}."
        )
    if missing := sorted(required.keys() - present):
        raise ValueError(
            f"MiMo-V2: {len(missing)} required tensor(s) are missing from the "
            f"checkpoint, e.g. {missing[:3]}."
        )


def _as_numpy(data: WeightData, view: DType | None = None) -> np.ndarray:
    """The elements of ``data``, optionally reinterpreted as ``view``."""
    buffer = (
        data.data
        if isinstance(data.data, Buffer)
        else Buffer.from_dlpack(data.data)
    )
    return np.from_dlpack(buffer if view is None else buffer.view(view))


def _wrap(array: np.ndarray, dtype: DType, name: str) -> WeightData:
    """``WeightData`` whose bytes are ``array`` reinterpreted as ``dtype``."""
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    if dtype != DType.from_numpy(array.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(buffer.shape))


def _stack_experts(
    read: Callable[[str], WeightData],
    config: PretrainedConfig,
    layer: int,
) -> dict[str, WeightData]:
    """Stacks one layer's routed experts in the W4A8 MoE layout.

    ``mlp.experts_gate_up_proj`` is ``uint8 [E, 2 * I, H / 2]``, each expert's
    gate rows then its up rows, and ``mlp.experts_down_proj`` is
    ``uint8 [E, H, I / 2]``. Each has an E8M0 ``_scale`` twin with ``/ 32`` in
    place of ``/ 2``. Expert ``e``'s slices hold its checkpoint codes
    verbatim and :func:`e8m0_scales_from_nvfp4` of its scales, so a loader can
    write each expert straight to its offset in a stacked buffer.

    Each expert's scales are then stored as :func:`interleave_e8m0` of them,
    declared ``[E, N/128, K/128, 32, 4, 4]``: the grouped matmul reads that
    layout, and interleaving in the graph would be hoisted to init as a copy
    of the stack. The offsets do not change.
    """
    experts = config.n_routed_experts
    inter, hidden = config.moe_intermediate_size, config.hidden_size
    stacks = {
        "gate_up_proj": (2 * inter, hidden // 2, DType.uint8),
        "gate_up_proj_scale": (
            2 * inter,
            hidden // MXFP4_BLOCK,
            DType.float8_e8m0fnu,
        ),
        "down_proj": (hidden, inter // 2, DType.uint8),
        "down_proj_scale": (hidden, inter // MXFP4_BLOCK, DType.float8_e8m0fnu),
    }
    shapes = {
        name: interleaved_scale_shape(rows, cols)
        if dtype == DType.float8_e8m0fnu
        else [rows, cols]
        for name, (rows, cols, dtype) in stacks.items()
    }
    # MAX host buffers, not np.empty: numpy asks for transparent huge pages on
    # large arrays, and on a host whose memory is mostly page cache every
    # fault then stalls in compaction (45x slower filling these, measured).
    buffers = {
        name: Buffer(DType.uint8, [experts, rows, cols], CPU())
        for name, (rows, cols, _) in stacks.items()
    }
    gate_up, gate_up_scale, down, down_scale = (
        np.from_dlpack(buffer) for buffer in buffers.values()
    )
    for expert in range(experts):
        for proj, codes_out, scale_out in (
            (
                "gate_proj",
                gate_up[expert, :inter],
                gate_up_scale[expert, :inter],
            ),
            ("up_proj", gate_up[expert, inter:], gate_up_scale[expert, inter:]),
            ("down_proj", down[expert], down_scale[expert]),
        ):
            source = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
            codes = _as_numpy(read(f"{source}.weight"))
            scale = _as_numpy(read(f"{source}.weight_scale"), DType.uint8)
            rows, packed = codes_out.shape
            if codes.shape != codes_out.shape or scale.shape != (
                rows,
                packed // 8,
            ):
                raise ValueError(
                    f"{source}: weight {codes.shape} and weight_scale "
                    f"{scale.shape} are not [{rows}, {packed}] and "
                    f"[{rows}, {packed // 8}]."
                )
            codes_out[...] = codes
            scale_out[...] = e8m0_scales_from_nvfp4(
                scale, _as_numpy(read(f"{source}.weight_scale_2")), source
            )
        for scales in (gate_up_scale[expert], down_scale[expert]):
            scales[...] = interleave_e8m0(scales).reshape(scales.shape)
    adapted = {}
    for name, (_, _, dtype) in stacks.items():
        max_name = f"layers.{layer}.mlp.experts_{name}"
        shape = [experts, *shapes[name]]
        adapted[max_name] = WeightData(
            buffers[name].view(dtype, shape), max_name, dtype, Shape(shape)
        )
    return adapted


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Adapts the NVFP4 export to MAX's MiMo-V2 weights.

    Args:
        state_dict: Every tensor in the checkpoint shards, by checkpoint name.
        huggingface_config: The checkpoint's top-level config.

    Returns:
        The weights keyed by MAX name: the checkpoint name without
        ``model.``, with the router's ``mlp.gate.weight`` as
        ``mlp.gate.gate_score.weight`` and each MoE layer's experts stacked
        as ``mlp.experts_{gate_up,down}_proj[_scale]``. FP8 projections carry
        a ``weight_scale`` beside their ``weight``.

    Raises:
        ValueError: If the config is not the NVFP4 export's, the tensor set
            or a dtype is not the expected one, or a transform is not exact.
    """
    config = huggingface_config
    validate_checkpoint_config(config)
    required = _required_tensors(config)
    _check_tensor_set(set(state_dict), required)

    consumed: set[str] = set()

    def read(name: str) -> WeightData:
        data = state_dict[name].data()
        if data.dtype != required[name]:
            raise ValueError(
                f"MiMo-V2: {name} is {data.dtype}; expected {required[name]}."
            )
        consumed.add(name)
        return data

    adapted: dict[str, WeightData] = {}
    for layer in moe_layers(config):
        adapted |= _stack_experts(read, config, layer)
    for name in required:
        if ".mlp.experts." in name:
            continue
        data = read(name)
        max_name = name.removeprefix("model.")
        base = max_name.removesuffix(".weight")
        if data.dtype == DType.float32 and name.endswith("_proj.weight"):
            weight = _as_numpy(data)
            if name.endswith("self_attn.qkv_proj.weight"):
                match = _LAYER.match(name)
                assert match is not None
                sliding = config.hybrid_layer_pattern[int(match[1])] == 1
                weight = qkv_to_chunk_order(
                    weight, qkv_chunk_layout(config, sliding), name
                )
            codes, scales = fp8_block_scaled_from_float32(weight, name)
            adapted[max_name] = _wrap(codes, DType.float8_e4m3fn, max_name)
            adapted[f"{base}.weight_scale"] = _wrap(
                scales, DType.float32, f"{base}.weight_scale"
            )
        elif name.endswith(".mlp.gate.weight"):
            router = f"{base}.gate_score.weight"
            bits = _as_numpy(data, DType.uint16).astype(np.uint32) << 16
            adapted[router] = _wrap(
                bits.view(np.float32), DType.float32, router
            )
        else:
            adapted[max_name] = dataclasses.replace(data, name=max_name)

    if unread := sorted(required.keys() - consumed):
        raise AssertionError(f"MiMo-V2: tensors never read: {unread[:3]}")
    return adapted
