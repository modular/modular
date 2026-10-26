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
"""Maps DeepSeek-V4.1-Flash checkpoint tensors onto MAX weight names.

Rules key on each tensor's on-disk dtype rather than its name, so a
dequantized checkpoint passes through the quantized-tensor rules unchanged.
Besides renaming, the trunk adapters drop vision tensors, stack per-expert
tensors, convert block scales to the layouts the kernels read, dequantize
projections the model runs in higher precision, and widen weights the model
keeps in float32. The vision adapter keeps only the vision tensors.
"""

from __future__ import annotations

import bisect
import ctypes
import functools
import mmap
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, SupportsInt

import numpy as np
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from max.pipelines.weights import (
    read_modelopt_quantized_layers,
    resolve_hf_quant_config,
)
from transformers.configuration_utils import PretrainedConfig

from .model_config import (
    NVFP4_SF_ATOM_COLS,
    NVFP4_SF_ATOM_ROWS,
    compressor_bf16_enabled,
    gate_up_rows_interleaved,
    head_bf16_enabled,
    num_dspark_stages,
    nvfp4_sf_preinterleaved,
    q_mxfp8_enabled,
    woa_mxfp8_enabled,
)

_SCALE_SUFFIX = ".scale"
_MAX_SCALE_SUFFIX = ".weight_scale"

# ``<prefix>.ffn.experts.<e>.<w1|w2|w3>.<kind>``: trunk NVFP4 and DSpark
# MXFP4 expert tensors.
_EXPERT_TENSOR = re.compile(
    r"^(?P<prefix>.*\.ffn\.experts)\.(?P<expert>\d+)\.(?P<proj>w[123])\."
    r"(?P<kind>weight_scale_2|weight_scale|input_scale|weight|scale)$"
)

# ``<block>.hc_<attn|ffn>_<fn|base|scale>``.
_HC_SITE_TENSOR = re.compile(
    r"^(?P<prefix>.*)\.hc_(?P<site>attn|ffn)_(?P<part>fn|base|scale)$"
)

GATE_UP_PROJ = "gate_up_proj"
DOWN_PROJ = "down_proj"

# FP8 projections the model may run in bf16: ``wo_a`` (trunk and DSpark
# stages) is dequantized only when ``woa_mxfp8_enabled()`` is false.
_HOST_DEQUANT_PROJECTIONS = (".attn.wo_a",)

# Checkpoint name prefix of the DSpark draft stages.
DSPARK_CHECKPOINT_PREFIX = "mtp."

# Rows and columns covered by one e8m0 scale of an MXFP8 dense weight.
MXFP8_BLOCK_SIZE = 32

# Attention projections host-dequantized to bf16 and concatenated, in this
# row order, into one ``query_proj.fused_wqa_wkv`` weight.
FUSED_QUERY_PROJECTIONS = ("wq_a", "wkv")


def is_host_dequantized(base: str, *, woa_mxfp8: bool | None = None) -> bool:
    """Returns whether projection ``base`` is host-dequantized to bf16.

    Args:
        base: The projection's dotted module name, e.g.
            ``layers.3.attn.wo_a``.
        woa_mxfp8: Overrides :func:`~.model_config.woa_mxfp8_enabled`.
    """
    if not base.endswith(_HOST_DEQUANT_PROJECTIONS):
        return False
    if woa_mxfp8 is None:
        woa_mxfp8 = woa_mxfp8_enabled()
    return not woa_mxfp8


# bf16 checkpoint weights the model computes with in float32; widened on
# load.
FLOAT32_WEIGHT_SUFFIXES = ("norm.weight",)
# Widened to float32 too unless ``head_bf16_enabled()``.
HEAD_WEIGHT = "head.weight"
# Widened to float32 too unless ``compressor_bf16_enabled()``.
COMPRESSOR_WEIGHT_SUFFIXES = (
    ".compressor.wkv.weight",
    ".compressor.wgate.weight",
)

# Scales kept as raw e8m0 (widening Engram's would quadruple their size).
_RAW_SCALE_PREFIXES = (".engram.embed",)

# VL checkpoint tensors, which the vision tower loads apart from the trunk.
_SKIPPED_PREFIXES = ("vision.", "aligner.")
_SKIPPED_EXACT = ("image_start", "image_end", "image_newline")

# Linux values, for Pythons built without these constants.
_MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21)
_MADV_WILLNEED = getattr(mmap, "MADV_WILLNEED", 3)
_MADV_DONTNEED = getattr(mmap, "MADV_DONTNEED", 4)
_LIBC = ctypes.CDLL(None)


def is_skipped_checkpoint_tensor(name: str) -> bool:
    """Whether ``name`` is a vision tensor, which the trunk adapters drop.

    Args:
        name: A checkpoint tensor's full dotted path.

    Returns:
        ``True`` for ``vision.*``, ``aligner.*``, or one of the three
        delimiter embeddings (``image_start``, ``image_end``,
        ``image_newline``); ``False`` otherwise.
    """
    return name.startswith(_SKIPPED_PREFIXES) or name in _SKIPPED_EXACT


def _view_same_width(buffer: Buffer, dtype: DType) -> Buffer:
    """Views ``buffer`` as same-width ``dtype``, including 0-d scalars."""
    shape = buffer.shape if buffer.rank == 0 else None
    return buffer.view(dtype, shape)


def as_uint8(data: DLPackArray) -> np.ndarray:
    """The raw bytes of a one-byte-per-element tensor (fp8, e8m0, int8)."""
    buffer = data if isinstance(data, Buffer) else Buffer.from_dlpack(data)
    return np.from_dlpack(_view_same_width(buffer, DType.uint8))


def as_float32(data: DLPackArray) -> np.ndarray:
    """The values of a float32 scalar tensor (e.g. ``input_scale``)."""
    buffer = data if isinstance(data, Buffer) else Buffer.from_dlpack(data)
    return np.from_dlpack(_view_same_width(buffer, DType.float32))


_READONLY_FILE_MAPPINGS: list[tuple[int, int]] = []
# Each read-only file mapping's start -> (file offset, path).
_READONLY_FILE_SOURCES: dict[int, tuple[int, str]] = {}


def _refresh_readonly_file_mappings() -> None:
    """Re-reads this process's read-only, file-backed mappings."""
    ranges = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                if (
                    len(fields) == 6
                    and fields[1].startswith("r-")
                    and fields[5].startswith("/")
                ):
                    low, _, high = fields[0].partition("-")
                    ranges.append((int(low, 16), int(high, 16)))
                    _READONLY_FILE_SOURCES[int(low, 16)] = (
                        int(fields[2], 16),
                        fields[5].rstrip("\n"),
                    )
    except OSError:
        pass
    ranges.sort()
    _READONLY_FILE_MAPPINGS[:] = ranges


def _in_readonly_file_mapping(start: int, end: int) -> bool:
    """Whether ``[start, end)`` lies in one read-only file mapping.

    Re-reads ``/proc/self/maps`` only on a miss.
    """
    for attempt in range(2):
        i = bisect.bisect_right(_READONLY_FILE_MAPPINGS, (start, sys.maxsize))
        if i and _READONLY_FILE_MAPPINGS[i - 1][1] >= end:
            return True
        if attempt == 0:
            _refresh_readonly_file_mappings()
    return False


def page_out(array: np.ndarray) -> None:
    """Drops an already-copied checkpoint tensor's pages from memory.

    Best effort, Linux only; a later read refaults from disk. Uses
    ``MADV_PAGEOUT``, then ``MADV_DONTNEED`` and ``POSIX_FADV_DONTNEED``
    inside read-only file mappings only, so ``array``'s contents never
    change.
    """
    if sys.platform != "linux":
        return
    page = mmap.PAGESIZE
    start = -(-array.ctypes.data // page) * page
    end = (array.ctypes.data + array.nbytes) // page * page
    if end > start:
        _LIBC.madvise(
            ctypes.c_void_p(start), ctypes.c_size_t(end - start), _MADV_PAGEOUT
        )
        if _in_readonly_file_mapping(start, end):
            _LIBC.madvise(
                ctypes.c_void_p(start),
                ctypes.c_size_t(end - start),
                _MADV_DONTNEED,
            )
            _drop_file_pages(start, end)


def _drop_file_pages(start: int, end: int) -> None:
    """Drops ``[start, end)``'s pages from the page cache of its file.

    Covers pages ``MADV_PAGEOUT`` skips (files this process cannot write).
    """
    i = bisect.bisect_right(_READONLY_FILE_MAPPINGS, (start, sys.maxsize))
    if not i:
        return
    low, high = _READONLY_FILE_MAPPINGS[i - 1]
    source = _READONLY_FILE_SOURCES.get(low)
    if high < end or source is None:
        return
    file_offset, path = source
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return
    try:
        os.posix_fadvise(
            fd, file_offset + (start - low), end - start, os.POSIX_FADV_DONTNEED
        )
    except OSError:
        pass
    finally:
        os.close(fd)


def drop_page_cache(paths: Iterable[os.PathLike[str] | str]) -> None:
    """Drops every unmapped page of each file in ``paths`` from page cache.

    Best effort (``POSIX_FADV_DONTNEED``): pages a process still maps stay.
    """
    if sys.platform != "linux":
        return
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(fd)


def _madvise(array: np.ndarray, advice: int) -> None:
    """``madvise`` over the whole pages ``array`` spans, best effort."""
    if sys.platform != "linux" or array.nbytes == 0:
        return
    page = mmap.PAGESIZE
    start = array.ctypes.data // page * page
    end = array.ctypes.data + array.nbytes
    _LIBC.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), advice)


def will_need(array: np.ndarray) -> None:
    """Starts reading a checkpoint tensor's pages ahead of its copy.

    Best effort (``MADV_WILLNEED``), like :func:`page_out`.
    """
    _madvise(array, _MADV_WILLNEED)


def e8m0_to_float32(raw: np.ndarray) -> np.ndarray:
    """``2 ** (e - 127)``; exact for every e8m0 code but NaN (255)."""
    return np.exp2(raw.astype(np.float32) - 127.0)


def _e4m3_table() -> np.ndarray:
    """The 256 ``float8_e4m3fn`` values, indexed by their byte."""
    code = np.arange(256, dtype=np.uint8)
    exponent = ((code >> 3) & 0x0F).astype(np.int32)
    mantissa = (code & 0x07).astype(np.float32)
    magnitude = np.where(
        exponent == 0,
        mantissa * 2.0**-9,
        (1.0 + mantissa / 8.0) * np.exp2((exponent - 7).astype(np.float32)),
    ).astype(np.float32)
    # e4m3fn has no infinities; the all-ones exponent with a full mantissa is
    # the only NaN.
    magnitude[0x7F] = np.nan
    magnitude[0xFF] = np.nan
    return np.where(code >> 7 == 1, -magnitude, magnitude).astype(np.float32)


_E4M3 = _e4m3_table()


def e4m3_to_float32(raw: np.ndarray) -> np.ndarray:
    """Decodes ``float8_e4m3fn`` byte codes to their float32 values."""
    return _E4M3[raw]


def with_dtype(array: np.ndarray, dtype: DType, name: str) -> WeightData:
    """``WeightData`` whose bytes are ``array`` reinterpreted as ``dtype``."""
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    if dtype != DType.from_numpy(array.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(buffer.shape))


def dequantize_fp8_blocks(
    weight: WeightData, scale: WeightData, name: str
) -> WeightData:
    """An fp8 weight times its block-scaled e8m0 scale, bfloat16.

    The block size is inferred per axis from ``weight.shape //
    scale.shape``. Exact: an e4m3 value times a power of two fits bf16.

    Raises:
        ValueError: If the weight's shape is not an integer multiple of the
            scale's on either axis.
    """
    values = _E4M3[as_uint8(weight.data)]
    raw_scale = as_uint8(scale.data)
    if (
        values.shape[0] % raw_scale.shape[0]
        or values.shape[1] % raw_scale.shape[1]
    ):
        raise ValueError(
            f"{name}: weight shape {values.shape} is not an integer "
            f"multiple of scale shape {raw_scale.shape}"
        )
    block_rows = values.shape[0] // raw_scale.shape[0]
    block_cols = values.shape[1] // raw_scale.shape[1]
    blocks = e8m0_to_float32(raw_scale)
    expanded = np.repeat(
        np.repeat(blocks, block_rows, axis=0), block_cols, axis=1
    )[: values.shape[0], : values.shape[1]]
    product = np.ascontiguousarray(values * expanded, dtype=np.float32)
    bf16_bits = (product.view(np.uint32) >> 16).astype(np.uint16)
    return with_dtype(bf16_bits, DType.bfloat16, name)


def expand_mxfp8_weight_scale(
    weight: WeightData, scale: WeightData, name: str
) -> WeightData:
    """The checkpoint's compact 2D block scale, row-expanded for the kernel.

    Repeats each row of the compact ``[N / 32, K / 32]`` E8M0 scale to the
    per-row ``[N, K / 32]`` layout the MXFP8 matmul expects. The
    row-block size is ``weight.shape[0] // scale.shape[0]``; the result
    stays ``float8_e8m0fnu``.

    Raises:
        ValueError: If the weight's row count is not an integer multiple of
            the scale's row count.
    """
    raw_scale = as_uint8(scale.data)
    weight_rows = int(weight.shape[0])
    if weight_rows % raw_scale.shape[0]:
        raise ValueError(
            f"{name}: weight row count {weight_rows} is not an integer "
            f"multiple of scale row count {raw_scale.shape[0]}"
        )
    block_rows = weight_rows // raw_scale.shape[0]
    expanded = np.repeat(raw_scale, block_rows, axis=0)
    return with_dtype(expanded, DType.float8_e8m0fnu, name)


def bfloat16_to_float32(weight: WeightData, name: str) -> WeightData:
    """A bf16 weight widened to float32 (exact: bf16 is f32's top 16 bits)."""
    buffer = (
        weight.data
        if isinstance(weight.data, Buffer)
        else Buffer.from_dlpack(weight.data)
    )
    bits = np.from_dlpack(buffer.view(DType.uint16)).astype(np.uint32) << 16
    return with_dtype(bits, DType.float32, name)


def _query_weight(
    state_dict: Mapping[str, WeightData], base: str, *, mxfp8: bool
) -> tuple[WeightData, WeightData | None]:
    """``base``'s weight, and its row-expanded scale if it stays MXFP8.

    An fp8 weight is host-dequantized to bf16 unless ``mxfp8``. An
    unquantized weight is returned unchanged.
    """
    weight = state_dict[f"{base}.weight"]
    scale = state_dict.get(f"{base}{_MAX_SCALE_SUFFIX}")
    if weight.dtype != DType.float8_e4m3fn or scale is None:
        return weight, None
    if mxfp8:
        return weight, scale
    return dequantize_fp8_blocks(weight, scale, f"{base}.weight"), None


def _as_same_width_bits(data: WeightData) -> np.ndarray:
    """``data``'s raw bytes as an unsigned integer array of the same width.

    Lets weights be concatenated by bit pattern, since numpy has no bf16.
    """
    buffer = (
        data.data
        if isinstance(data.data, Buffer)
        else Buffer.from_dlpack(data.data)
    )
    width_to_dtype = {1: DType.uint8, 2: DType.uint16, 4: DType.uint32}
    return np.from_dlpack(buffer.view(width_to_dtype[data.dtype.size_in_bytes]))


def _rewire_attention_projections(
    out: dict[str, WeightData], attn: str
) -> None:
    """Fuses and renames one attention's projections onto ``query_proj``.

    In place: ``wq_a`` + ``wkv`` -> ``query_proj.fused_wqa_wkv`` and
    ``wq_b`` -> ``query_proj.wq_b``, ``q_norm`` -> ``query_proj.q_norm``.
    These and the indexer's ``wq_b`` keep their fp8 weights and scales
    under :func:`~.model_config.q_mxfp8_enabled`, and are host-dequantized
    to bf16 otherwise. Used for trunk layers and DSpark stages.

    Raises:
        ValueError: If only one of ``wq_a`` and ``wkv`` stays MXFP8.
    """
    mxfp8 = q_mxfp8_enabled()
    fused = f"{attn}.query_proj.fused_wqa_wkv"
    parts = [
        _query_weight(out, f"{attn}.{projection}", mxfp8=mxfp8)
        for projection in FUSED_QUERY_PROJECTIONS
    ]
    scales = [scale for _, scale in parts if scale is not None]
    if scales and len(scales) != len(parts):
        raise ValueError(
            f"{fused}: {FUSED_QUERY_PROJECTIONS} must all be fp8 or none"
        )
    out[f"{fused}.weight"] = with_dtype(
        np.concatenate(
            [_as_same_width_bits(weight) for weight, _ in parts], axis=0
        ),
        DType.float8_e4m3fn if scales else DType.bfloat16,
        f"{fused}.weight",
    )
    projections = [(f"{attn}.wq_b", f"{attn}.query_proj.wq_b")]
    # Only index-source layers have an indexer.
    if f"{attn}.indexer.wq_b.weight" in out:
        projections.append((f"{attn}.indexer.wq_b", f"{attn}.indexer.wq_b"))
    for projection in FUSED_QUERY_PROJECTIONS:
        out.pop(f"{attn}.{projection}.weight")
        out.pop(f"{attn}.{projection}{_MAX_SCALE_SUFFIX}", None)
    if scales:
        out[f"{fused}{_MAX_SCALE_SUFFIX}"] = with_dtype(
            np.concatenate([_as_same_width_bits(s) for s in scales], axis=0),
            DType.float8_e8m0fnu,
            f"{fused}{_MAX_SCALE_SUFFIX}",
        )
    for source, target in projections:
        weight, scale = _query_weight(out, source, mxfp8=mxfp8)
        out.pop(f"{source}.weight")
        out.pop(f"{source}{_MAX_SCALE_SUFFIX}", None)
        out[f"{target}.weight"] = weight
        if scale is not None:
            out[f"{target}{_MAX_SCALE_SUFFIX}"] = scale
    out[f"{attn}.query_proj.q_norm.weight"] = out.pop(f"{attn}.q_norm.weight")


def _fill_gate_bias_vl(out: dict[str, WeightData], gate: str) -> None:
    """Fills a missing ``{gate}.bias_vl`` with zeros, in place.

    Only image tokens use ``bias_vl``, so text-only exports may omit it.
    """
    if f"{gate}.bias_vl" not in out and f"{gate}.bias" in out:
        out[f"{gate}.bias_vl"] = with_dtype(
            np.zeros(out[f"{gate}.bias"].shape.static_dims, np.float32),
            DType.float32,
            f"{gate}.bias_vl",
        )


def fuse_shared_expert_gate_up(out: dict[str, WeightData], shared: str) -> None:
    """``{shared}.w1`` + ``{shared}.w3`` -> ``{shared}.gate_up_proj``, in place.

    Concatenates rows ``[w1; w3]`` of the weights and, when present, their
    row-expanded weight scales. A state dict without ``w1`` is untouched.

    Raises:
        ValueError: If ``w1`` and ``w3`` do not pair up.
    """
    if f"{shared}.w1.weight" not in out:
        return
    for suffix in (".weight", _MAX_SCALE_SUFFIX):
        gate = out.pop(f"{shared}.w1{suffix}", None)
        up = out.pop(f"{shared}.w3{suffix}", None)
        if gate is None and up is None:
            continue
        if gate is None or up is None or gate.dtype != up.dtype:
            raise ValueError(
                f"{shared}: w1{suffix} and w3{suffix} do not pair up for the "
                "fused gate_up_proj"
            )
        name = f"{shared}.gate_up_proj{suffix}"
        out[name] = with_dtype(
            np.concatenate(
                [_as_same_width_bits(gate), _as_same_width_bits(up)], axis=0
            ),
            gate.dtype,
            name,
        )


def rewire_trunk_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Maps :func:`convert_weight_data`'s output onto the ``DeepseekV41`` tree.

    Prefixes ``embed`` and layers with ``trunk.``, renames ``engram`` to
    ``engram_fn``, and fuses the query projections
    (:func:`_rewire_attention_projections`) and shared-expert ``w1``/``w3``
    (:func:`fuse_shared_expert_gate_up`). Takes no ``mtp.*`` tensors.
    """
    layer_ids = sorted(
        {
            int(match.group(1))
            for name in state_dict
            if (match := re.match(r"^layers\.(\d+)\.", name))
        }
    )
    out = dict(state_dict)
    for layer in layer_ids:
        _rewire_attention_projections(out, f"layers.{layer}.attn")
        _fill_gate_bias_vl(out, f"layers.{layer}.ffn.gate")
        fuse_shared_expert_gate_up(out, f"layers.{layer}.ffn.shared_experts")

    final: dict[str, WeightData] = {}
    for name, data in out.items():
        renamed = name.replace(".engram.", ".engram_fn.", 1)
        if renamed == "embed.weight" or renamed.startswith("layers."):
            renamed = f"trunk.{renamed}"
        final[renamed] = data
    return final


_DSPARK_STAGE_TENSOR = re.compile(r"^mtp\.(?P<stage>\d+)\.(?P<rest>.+)$")
# Tensors owned by the drafter as a whole rather than by one stage's block:
# these prefixes under stage 0, and under the last stage.
_DSPARK_FIRST_STAGE_OWNED = ("main_proj.", "main_norm.")
_DSPARK_LAST_STAGE_OWNED = ("norm.", "markov_head.")
DSPARK_MODULE_PREFIX = "dspark"
"""Attribute that holds the drafter, and so the prefix of its weight names."""


def rewire_dspark_state_dict(
    state_dict: dict[str, WeightData], last_stage: int
) -> dict[str, WeightData]:
    """Maps ``mtp.{i}.*`` tensors onto the ``dspark`` drafter tree.

    Stages get the trunk's attention, gate and shared-expert rewiring; then
    ``mtp.0.main_{proj,norm}`` and ``mtp.{last_stage}.{norm,markov_head}``
    move to ``dspark.*``, and every other ``mtp.{i}.X`` to
    ``dspark.stages.{i}.X``.

    Args:
        state_dict: Only ``mtp.*`` tensors (already through
            :func:`convert_weight_data`).
        last_stage: The last stage's index (``num_dspark_layers - 1``).

    Raises:
        ValueError: On a tensor outside ``mtp.{i}.*``.
    """
    stages = set()
    for name in state_dict:
        match = _DSPARK_STAGE_TENSOR.match(name)
        if match is None:
            raise ValueError(f"{name}: not a DSpark stage (mtp.*) tensor")
        stages.add(int(match["stage"]))
    out = dict(state_dict)
    for stage in sorted(stages):
        if f"mtp.{stage}.attn.wq_a.weight" in out:
            _rewire_attention_projections(out, f"mtp.{stage}.attn")
        _fill_gate_bias_vl(out, f"mtp.{stage}.ffn.gate")
        fuse_shared_expert_gate_up(out, f"mtp.{stage}.ffn.shared_experts")

    final: dict[str, WeightData] = {}
    for name, data in out.items():
        match = _DSPARK_STAGE_TENSOR.match(name)
        assert match is not None
        stage, rest = int(match["stage"]), match["rest"]
        if (stage == 0 and rest.startswith(_DSPARK_FIRST_STAGE_OWNED)) or (
            stage == last_stage and rest.startswith(_DSPARK_LAST_STAGE_OWNED)
        ):
            final[f"{DSPARK_MODULE_PREFIX}.{rest}"] = data
        else:
            final[f"{DSPARK_MODULE_PREFIX}.stages.{stage}.{rest}"] = data
    return final


def stack_experts(
    experts: Mapping[int, Mapping[str, np.ndarray]], prefix: str
) -> dict[str, WeightData]:
    """Per-expert packed fp4 tensors -> the two stacked routed-expert weights.

    DSpark's MXFP4 W4A8 format: ``experts[e]`` holds ``w1``, ``w2``, ``w3``
    (uint8 ``[N, K/2]``) and ``w{1,2,3}.scale`` (uint8 e8m0 ``[N, K/32]``).

    Raises:
        ValueError: If expert ids are not dense ``0..E-1``.
    """
    ids = sorted(experts)
    if ids != list(range(len(ids))):
        raise ValueError(f"{prefix}: expert ids are not dense: {ids}")
    gate_up = np.stack(
        [
            np.concatenate([experts[e]["w1"], experts[e]["w3"]], axis=0)
            for e in ids
        ]
    )
    gate_up_scale = np.stack(
        [
            np.concatenate(
                [experts[e]["w1.scale"], experts[e]["w3.scale"]], axis=0
            )
            for e in ids
        ]
    )
    down = np.stack([experts[e]["w2"] for e in ids])
    down_scale = np.stack([experts[e]["w2.scale"] for e in ids])
    return {
        f"{prefix}.{GATE_UP_PROJ}.weight": with_dtype(
            gate_up, DType.uint8, f"{prefix}.{GATE_UP_PROJ}.weight"
        ),
        f"{prefix}.{GATE_UP_PROJ}.weight_scale": with_dtype(
            gate_up_scale,
            DType.float8_e8m0fnu,
            f"{prefix}.{GATE_UP_PROJ}.weight_scale",
        ),
        f"{prefix}.{DOWN_PROJ}.weight": with_dtype(
            down, DType.uint8, f"{prefix}.{DOWN_PROJ}.weight"
        ),
        f"{prefix}.{DOWN_PROJ}.weight_scale": with_dtype(
            down_scale,
            DType.float8_e8m0fnu,
            f"{prefix}.{DOWN_PROJ}.weight_scale",
        ),
    }


def _scalar(value: np.ndarray) -> float:
    """A rank-0 or rank-1 per-expert scale as a plain float."""
    return float(np.asarray(value).reshape(-1)[0])


def _shared_scalar(
    experts: Mapping[int, Mapping[str, np.ndarray]],
    ids: list[int],
    kind: str,
    prefix: str,
) -> np.ndarray:
    """``w1``'s per-expert scalar for the fused ``gate_up_proj``.

    Warns if ``w3``'s differs.
    """
    w1_values = np.array(
        [_scalar(experts[e][f"w1.{kind}"]) for e in ids], dtype=np.float32
    )
    w3_values = np.array(
        [_scalar(experts[e][f"w3.{kind}"]) for e in ids], dtype=np.float32
    )
    if not np.array_equal(w1_values, w3_values):
        warnings.warn(
            f"{prefix}: w1 and w3 {kind} disagree; using w1's value for the "
            "fused gate_up_proj",
            stacklevel=3,
        )
    return w1_values


def _layer_max_input_scale(
    experts: Mapping[int, Mapping[str, np.ndarray]],
    ids: list[int],
    projs: tuple[str, ...],
) -> np.ndarray:
    """The max ``input_scale`` of ``projs`` over all experts, once per expert.

    Every expert quantizes its activations against this one scale, as vLLM's
    FlashInfer NVFP4 MoE does. The checkpoint's per-expert scales spread 80x,
    and quantizing against them clips the small-scale experts. Reduced here
    rather than in the graph, where an expert-parallel shard sees only its own
    experts.
    """
    scale = max(
        _scalar(experts[e][f"{p}.input_scale"]) for e in ids for p in projs
    )
    return np.full(len(ids), scale, dtype=np.float32)


def interleave_nvfp4_scales(scales: np.ndarray) -> np.ndarray:
    """Reorders ``[rows, cols]`` block scales into the tiled layout.

    This is the scale layout the block-scaled grouped matmul reads, for
    NVFP4 and MXFP8 alike (only bytes move): entry ``[r, c]`` lands at
    ``[r // 128, c // 4, r % 32, (r % 128) // 32, c % 4]`` of ``[rows / 128, cols / 4, 32, 4, 4]``, flattened back to
    ``[rows, cols]``.

    Raises:
        ValueError: If ``scales`` does not tile the 128 x 4 scale tile
            exactly.
    """
    rows, cols = scales.shape
    if rows % NVFP4_SF_ATOM_ROWS or cols % NVFP4_SF_ATOM_COLS:
        raise ValueError(
            f"scales {scales.shape} do not tile the "
            f"{NVFP4_SF_ATOM_ROWS}x{NVFP4_SF_ATOM_COLS} SF atom"
        )
    atoms = scales.reshape(
        rows // NVFP4_SF_ATOM_ROWS,
        NVFP4_SF_ATOM_ROWS // 32,
        32,
        cols // NVFP4_SF_ATOM_COLS,
        NVFP4_SF_ATOM_COLS,
    )
    return np.ascontiguousarray(atoms.transpose(0, 3, 2, 1, 4)).reshape(
        rows, cols
    )


class DeferredExpertStack(WeightData):
    """A stacked routed-expert tensor, copied out of the checkpoint on demand.

    Holds each expert's mmap-backed checkpoint views so experts can be
    streamed to devices without a host copy of the whole stack. ``data`` is
    a zero-stride placeholder of the stacked shape for strict
    ``load_state_dict`` checks; it is non-contiguous, so the engine rejects
    it if it ever reaches a weights registry.
    """

    parts: list[tuple[np.ndarray, ...]]
    """Per expert, the checkpoint arrays concatenated into its stacked row.

    Concatenated along axis 0: ``w1``, ``w3`` for ``gate_up_proj``; ``w2``
    for ``down_proj``.
    """

    interleave_rows: bool
    """Whether the parts' rows alternate rather than follow one another.

    Row ``2i`` is the first part's row ``i`` and row ``2i + 1`` the second
    part's, the (gate, up) pairing the fused SwiGLU kernels read.
    """

    def __init__(
        self,
        parts: Sequence[Sequence[np.ndarray]],
        dtype: DType,
        name: str,
        *,
        sf_interleave: bool = False,
        interleave_rows: bool = False,
    ) -> None:
        first = parts[0]
        shape = (
            len(parts),
            sum(int(part.shape[0]) for part in first),
            *first[0].shape[1:],
        )
        placeholder = np.lib.stride_tricks.as_strided(
            np.zeros(1, np.uint8),
            shape=shape,
            strides=(0,) * len(shape),
            writeable=False,
        )
        super().__init__(placeholder, name, dtype, Shape(shape))
        self.parts = [tuple(expert) for expert in parts]
        self.sf_interleave = sf_interleave
        """Whether :meth:`materialize` reorders rows into the tiled layout."""
        if interleave_rows and (
            len(first) != 2 or first[0].shape != first[1].shape
        ):
            raise ValueError(f"{name}: only two equal parts interleave rows")
        self.interleave_rows = interleave_rows

    def materialize(
        self, start: int = 0, stop: int | None = None
    ) -> np.ndarray:
        """Experts ``[start, stop)`` stacked, as uint8 bytes (a fresh copy)."""
        stop = len(self.parts) if stop is None else stop
        out = np.empty(
            (stop - start, *self.shape.static_dims[1:]), dtype=np.uint8
        )
        for row, expert in zip(out, self.parts[start:stop], strict=True):
            if self.interleave_rows:
                pairs = row.reshape(-1, 2, *row.shape[1:])
                pairs[:, 0] = expert[0]
                pairs[:, 1] = expert[1]
            else:
                np.concatenate(expert, axis=0, out=row)
            if self.sf_interleave:
                row[...] = interleave_nvfp4_scales(row)
        return out

    def page_out(self, start: int = 0, stop: int | None = None) -> None:
        """Drops experts ``[start, stop)``'s checkpoint pages."""
        for expert in self.parts[start:stop]:
            for array in expert:
                page_out(array)

    def will_need(self, start: int = 0, stop: int | None = None) -> None:
        """Starts reading experts ``[start, stop)`` ahead."""
        for expert in self.parts[start:stop]:
            for array in expert:
                will_need(array)


def stack_nvfp4_experts(
    experts: Mapping[int, Mapping[str, np.ndarray]],
    prefix: str,
    *,
    defer: bool = False,
    interleave_gate_up: bool = False,
) -> dict[str, WeightData]:
    """Per-expert NVFP4 tensors -> the two stacked trunk routed-expert weights.

    ``experts[e]`` holds ``w{1,2,3}`` (uint8 ``[N, K/2]``, packed E2M1),
    ``w{1,2,3}.weight_scale`` (``float8_e4m3fn`` bytes ``[N, K/16]``) and
    float32 scalars ``w{1,2,3}.weight_scale_2`` and ``w{1,2,3}.input_scale``.
    Both projections' ``input_scale`` become the layer max
    (:func:`_layer_max_input_scale`). With ``defer``, each ``weight`` and
    ``weight_scale`` is returned as a :class:`DeferredExpertStack`. With
    ``interleave_gate_up``, the ``gate_up_proj`` rows (and their scale rows)
    alternate ``w1``, ``w3`` instead of all ``w1`` rows then all ``w3`` rows.

    Raises:
        ValueError: If expert ids are not dense ``0..E-1``.
    """
    ids = sorted(experts)
    if ids != list(range(len(ids))):
        raise ValueError(f"{prefix}: expert ids are not dense: {ids}")

    def stacked(keys: tuple[str, ...], dtype: DType, name: str) -> WeightData:
        parts = [tuple(experts[e][key] for key in keys) for e in ids]
        rows = sum(int(part.shape[0]) for part in parts[0])
        deferred = DeferredExpertStack(
            parts,
            dtype,
            name,
            # The expert module decides its scale layout with the same
            # predicate, so the stored and expected layouts agree.
            sf_interleave=dtype == DType.float8_e4m3fn
            and parts[0][0].ndim == 2
            and nvfp4_sf_preinterleaved(rows, int(parts[0][0].shape[1])),
            interleave_rows=interleave_gate_up and len(keys) == 2,
        )
        if defer:
            return deferred
        return with_dtype(deferred.materialize(), dtype, name)

    gate_up_scale_2 = _shared_scalar(experts, ids, "weight_scale_2", prefix)
    gate_up_input_scale = _layer_max_input_scale(experts, ids, ("w1", "w3"))
    down_scale_2 = np.array(
        [_scalar(experts[e]["w2.weight_scale_2"]) for e in ids],
        dtype=np.float32,
    )
    down_input_scale = _layer_max_input_scale(experts, ids, ("w2",))

    gate_up_name = f"{prefix}.{GATE_UP_PROJ}"
    down_name = f"{prefix}.{DOWN_PROJ}"
    return {
        f"{gate_up_name}.weight": stacked(
            ("w1", "w3"), DType.uint8, f"{gate_up_name}.weight"
        ),
        f"{gate_up_name}.weight_scale": stacked(
            ("w1.weight_scale", "w3.weight_scale"),
            DType.float8_e4m3fn,
            f"{gate_up_name}.weight_scale",
        ),
        f"{gate_up_name}.weight_scale_2": with_dtype(
            gate_up_scale_2, DType.float32, f"{gate_up_name}.weight_scale_2"
        ),
        f"{gate_up_name}.input_scale": with_dtype(
            gate_up_input_scale, DType.float32, f"{gate_up_name}.input_scale"
        ),
        f"{down_name}.weight": stacked(
            ("w2",), DType.uint8, f"{down_name}.weight"
        ),
        f"{down_name}.weight_scale": stacked(
            ("w2.weight_scale",),
            DType.float8_e4m3fn,
            f"{down_name}.weight_scale",
        ),
        f"{down_name}.weight_scale_2": with_dtype(
            down_scale_2, DType.float32, f"{down_name}.weight_scale_2"
        ),
        f"{down_name}.input_scale": with_dtype(
            down_input_scale, DType.float32, f"{down_name}.input_scale"
        ),
    }


# Keys per complete expert, to detect when a layer can be stacked.
# w{1,2,3} x {weight, scale}
_W4A8_KEYS_PER_EXPERT = 6
# w{1,2,3} x {weight, weight_scale, weight_scale_2, input_scale}
_NVFP4_KEYS_PER_EXPERT = 12

_StackFn = Callable[
    [Mapping[int, Mapping[str, np.ndarray]], str], dict[str, WeightData]
]


def _flush_complete_prefix(
    packed: dict[str, dict[int, dict[str, np.ndarray]]],
    prefix: str,
    num_experts: int,
    keys_per_expert: int,
    stack: _StackFn,
    new_state_dict: dict[str, WeightData],
    *,
    page_out_inputs: bool = True,
) -> None:
    """Stacks and evicts ``prefix`` once every one of its experts is complete.

    Bounds resident expert data to the layers still in flight. Pages out
    the inputs unless ``page_out_inputs`` is false (deferred stacks have not
    read them yet).
    """
    experts = packed.get(prefix)
    if experts is None or len(experts) != num_experts:
        return
    if not all(len(kinds) == keys_per_expert for kinds in experts.values()):
        return
    new_state_dict.update(stack(experts, prefix))
    if page_out_inputs:
        for kinds in experts.values():
            for array in kinds.values():
                page_out(array)
    del packed[prefix]


def convert_weight_data(
    state_dict: Mapping[str, WeightData],
    *,
    num_routed_experts: int = 384,
    num_dspark_experts: int = 128,
    defer_routed_experts: bool = False,
    woa_mxfp8: bool | None = None,
    head_bf16: bool | None = None,
    data_parallel_degree: int = 1,
) -> dict[str, WeightData]:
    """Converts checkpoint ``WeightData`` to MAX storage, keeping names.

    Args:
        state_dict: The checkpoint's weights, keyed by checkpoint tensor
            name. Vision tensors (see :func:`is_skipped_checkpoint_tensor`)
            must already be filtered out by the caller.
        num_routed_experts: Trunk experts per layer; a layer is stacked
            once all are seen.
        num_dspark_experts: DSpark experts per stage; likewise.
        defer_routed_experts: Return the trunk's NVFP4 expert
            ``weight``/``weight_scale`` as :class:`DeferredExpertStack`.
        woa_mxfp8: Overrides :func:`~.model_config.woa_mxfp8_enabled`
            (whether the trunk's ``wo_a`` stays MXFP8).
        head_bf16: Overrides :func:`~.model_config.head_bf16_enabled`
            (whether ``head.weight`` stays bf16 instead of float32).
        data_parallel_degree: The serving data-parallel degree. The trunk's
            routed ``gate_up_proj`` rows interleave when
            :func:`~.model_config.gate_up_rows_interleaved` holds.

    Returns:
        The converted state dict, still keyed by checkpoint names.
    """
    if woa_mxfp8 is None:
        woa_mxfp8 = woa_mxfp8_enabled()
    if head_bf16 is None:
        head_bf16 = head_bf16_enabled()
    compressor_bf16 = compressor_bf16_enabled()
    new_state_dict: dict[str, WeightData] = {}
    interleave_trunk = gate_up_rows_interleaved(
        num_routed_experts, data_parallel_degree
    )

    def stack_nvfp4(
        experts: Mapping[int, Mapping[str, np.ndarray]], prefix: str
    ) -> dict[str, WeightData]:
        is_trunk = not prefix.startswith(DSPARK_CHECKPOINT_PREFIX)
        return stack_nvfp4_experts(
            experts,
            prefix,
            defer=defer_routed_experts,
            interleave_gate_up=interleave_trunk and is_trunk,
        )

    host_dequantized = functools.partial(
        is_host_dequantized, woa_mxfp8=woa_mxfp8
    )
    # ``prefix -> expert id -> kind -> tensor``, filled only for packed-fp4
    # experts, one dict per format since the two stack differently.
    w4a8_packed_experts: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    nvfp4_packed_experts: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    for name, data in state_dict.items():
        hc_site = _HC_SITE_TENSOR.match(name)
        if hc_site is not None:
            new_state_dict[
                f"{hc_site['prefix']}.hc_{hc_site['site']}.hc_{hc_site['part']}"
            ] = data
            continue

        expert = _EXPERT_TENSOR.match(name)
        if expert is not None:
            base = f"{expert['prefix']}.{expert['expert']}.{expert['proj']}"
            weight = state_dict.get(f"{base}.weight")
            if weight is not None and weight.dtype == DType.int8:
                # DSpark's MXFP4 W4A8 experts: "weight" and "scale" only.
                if expert["kind"] in ("weight", "scale"):
                    key = expert["proj"] + (
                        ".scale" if expert["kind"] == "scale" else ""
                    )
                    experts = w4a8_packed_experts.setdefault(
                        expert["prefix"], {}
                    ).setdefault(int(expert["expert"]), {})
                    experts[key] = as_uint8(data.data)
                    _flush_complete_prefix(
                        w4a8_packed_experts,
                        expert["prefix"],
                        num_dspark_experts,
                        _W4A8_KEYS_PER_EXPERT,
                        stack_experts,
                        new_state_dict,
                    )
                    continue
            elif weight is not None and weight.dtype == DType.uint8:
                # The trunk's NVFP4 experts: four kinds per projection.
                key = (
                    expert["proj"]
                    if expert["kind"] == "weight"
                    else f"{expert['proj']}.{expert['kind']}"
                )
                is_bytes = expert["kind"] in ("weight", "weight_scale")
                experts = nvfp4_packed_experts.setdefault(
                    expert["prefix"], {}
                ).setdefault(int(expert["expert"]), {})
                experts[key] = (
                    as_uint8(data.data) if is_bytes else as_float32(data.data)
                )
                _flush_complete_prefix(
                    nvfp4_packed_experts,
                    expert["prefix"],
                    num_routed_experts,
                    _NVFP4_KEYS_PER_EXPERT,
                    stack_nvfp4,
                    new_state_dict,
                    page_out_inputs=not defer_routed_experts,
                )
                continue
            # Otherwise a dequantized expert copy: passed through below.

        if name.endswith(_SCALE_SUFFIX):
            base = name[: -len(_SCALE_SUFFIX)]
            if base.endswith(_RAW_SCALE_PREFIXES):
                # engram.embed.scale: kept raw.
                new_state_dict[name] = data
                continue
            if host_dequantized(base):
                # Folded into the weight below; the module has no scale.
                continue
            max_name = base + _MAX_SCALE_SUFFIX
            weight = state_dict.get(f"{base}.weight")
            if (
                weight is not None
                and weight.dtype == DType.float8_e4m3fn
                and data.dtype == DType.float8_e8m0fnu
            ):
                expanded = expand_mxfp8_weight_scale(weight, data, max_name)
                if base.endswith(_HOST_DEQUANT_PROJECTIONS):
                    # The MXFP8 ``wo_a`` runs as a grouped block-scaled
                    # matmul, which reads its scales tile-interleaved.
                    expanded = with_dtype(
                        interleave_nvfp4_scales(as_uint8(expanded.data)),
                        DType.float8_e8m0fnu,
                        max_name,
                    )
                new_state_dict[max_name] = expanded
                continue
            new_state_dict[max_name] = data
            continue

        if name.endswith(".weight") and data.dtype == DType.float8_e4m3fn:
            base = name[: -len(".weight")]
            scale = state_dict.get(base + _SCALE_SUFFIX)
            if host_dequantized(base) and scale is not None:
                new_state_dict[name] = dequantize_fp8_blocks(data, scale, name)
                continue

        if (
            name.endswith(FLOAT32_WEIGHT_SUFFIXES)
            or (name == HEAD_WEIGHT and not head_bf16)
            or (
                name.endswith(COMPRESSOR_WEIGHT_SUFFIXES)
                and not compressor_bf16
            )
        ) and data.dtype == DType.bfloat16:
            new_state_dict[name] = bfloat16_to_float32(data, name)
            continue

        new_state_dict[name] = data

    # A partial state dict can leave a prefix incomplete; stack the rest.
    for prefix, remaining_w4a8_experts in w4a8_packed_experts.items():
        new_state_dict.update(stack_experts(remaining_w4a8_experts, prefix))
    for prefix, remaining_nvfp4_experts in nvfp4_packed_experts.items():
        new_state_dict.update(stack_nvfp4(remaining_nvfp4_experts, prefix))
    return new_state_dict


@dataclass(frozen=True)
class LoudLoadAuditResult:
    """The result of :func:`audit_loaded_tensors`."""

    missing: frozenset[str]
    """Checkpoint tensors the converted output does not account for."""

    unexpected: frozenset[str]
    """Converted tensors that trace back to no checkpoint tensor."""

    @property
    def ok(self) -> bool:
        """``True`` if every checkpoint tensor was consumed and nothing else."""
        return not self.missing and not self.unexpected


def audit_loaded_tensors(
    checkpoint_tensor_names: Iterable[str],
    converted_tensor_names: Iterable[str],
    *,
    woa_mxfp8: bool | None = None,
) -> LoudLoadAuditResult:
    """Checks that every checkpoint tensor is consumed, and nothing else is.

    Derives the expected outputs from checkpoint names with the same rules
    as :func:`convert_weight_data`, honoring
    :func:`is_skipped_checkpoint_tensor` and :func:`is_unused_dspark_tensor`.

    Args:
        checkpoint_tensor_names: Every tensor name the checkpoint declares.
        converted_tensor_names: Every tensor name the conversion produced.
        woa_mxfp8: Overrides :func:`~.model_config.woa_mxfp8_enabled`, as
            in :func:`convert_weight_data`.

    Returns:
        The audit result; ``result.ok`` is ``True`` only when both sets are
        empty.
    """
    converted = set(converted_tensor_names)
    is_host_dequant = functools.partial(
        is_host_dequantized, woa_mxfp8=woa_mxfp8
    )
    missing: set[str] = set()
    expected_outputs: set[str] = set()
    # Stacked prefix -> an example input name, to report a missing group.
    stacked_prefixes: dict[str, str] = {}

    for name in checkpoint_tensor_names:
        if is_skipped_checkpoint_tensor(name) or is_unused_dspark_tensor(name):
            continue

        hc_site = _HC_SITE_TENSOR.match(name)
        if hc_site is not None:
            renamed = (
                f"{hc_site['prefix']}.hc_{hc_site['site']}.hc_{hc_site['part']}"
            )
            expected_outputs.add(renamed)
            if renamed not in converted:
                missing.add(name)
            continue

        expert = _EXPERT_TENSOR.match(name)
        if expert is not None:
            proj_group = (
                GATE_UP_PROJ if expert["proj"] in ("w1", "w3") else DOWN_PROJ
            )
            stacked_prefixes[f"{expert['prefix']}.{proj_group}"] = name
            continue

        if name.endswith(_SCALE_SUFFIX):
            base = name[: -len(_SCALE_SUFFIX)]
            if base.endswith(_RAW_SCALE_PREFIXES):
                # Kept raw, under the same name.
                expected_outputs.add(name)
                if name not in converted:
                    missing.add(name)
                continue
            if is_host_dequant(base):
                # Folded into the host-dequantized weight.
                expected_outputs.add(f"{base}.weight")
                continue
            renamed = base + _MAX_SCALE_SUFFIX
            expected_outputs.add(renamed)
            if renamed not in converted:
                missing.add(name)
            continue

        expected_outputs.add(name)
        if name not in converted:
            missing.add(name)

    for stacked, example in stacked_prefixes.items():
        kinds_present = [
            out for out in converted if out.startswith(f"{stacked}.")
        ]
        expected_outputs.update(kinds_present)
        if not kinds_present:
            missing.add(example)

    unexpected = frozenset(converted - expected_outputs)
    return LoudLoadAuditResult(
        missing=frozenset(missing), unexpected=unexpected
    )


_SCALE_LIKE_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".scale",
)


def find_unpaired_scales(
    checkpoint_tensor_names: Iterable[str],
) -> frozenset[str]:
    """Checkpoint scale tensors with no sibling ``<base>.weight`` tensor."""
    names = set(checkpoint_tensor_names)
    unpaired: set[str] = set()
    for name in names:
        for suffix in _SCALE_LIKE_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                if f"{base}.weight" not in names:
                    unpaired.add(name)
                break
    return frozenset(unpaired)


# Used only by adaptive verification, which DSpark drafting does not run.
_UNUSED_DSPARK = (".confidence_head.",)


def is_unused_dspark_tensor(name: str) -> bool:
    """Whether ``name`` is a DSpark stage tensor no converter ever reads.

    Args:
        name: A checkpoint tensor's full dotted path.

    Returns:
        ``True`` for an ``mtp.*`` tensor matching :data:`_UNUSED_DSPARK`;
        ``False`` otherwise.
    """
    return name.startswith(DSPARK_CHECKPOINT_PREFIX) and any(
        part in name for part in _UNUSED_DSPARK
    )


_NVFP4_ALGO = "NVFP4"
_NVFP4_GROUP_SIZE = 16


@dataclass(frozen=True)
class V41QuantizationInfo:
    """The validated facts this checkpoint's ``quantization_config`` carries.

    ``nvfp4_expert_layers`` is the allow-list from ``quantized_layers``,
    not the complement of ``ignore_patterns``.
    """

    nvfp4_expert_layers: frozenset[int]
    ignore_patterns: tuple[str, ...]


def resolve_quantization_info(
    huggingface_config: PretrainedConfig,
    state_dict: Mapping[str, object],
    num_hidden_layers: int,
) -> V41QuantizationInfo | None:
    """Validates and parses the checkpoint's MIXED_PRECISION quant config.

    Uses :func:`read_modelopt_quantized_layers` after normalizing the
    checkpoint's ``quant_method`` of ``"fp8"`` to ``"modelopt"``.

    Args:
        huggingface_config: The checkpoint's top-level Hugging Face config.
        state_dict: The checkpoint's weights; only the keys are read.
        num_hidden_layers: The trunk's layer count; every
            ``layers.{i}.ffn.experts`` must be declared NVFP4.

    Returns:
        The parsed info, or ``None`` for an unquantized checkpoint.

    Raises:
        ValueError: If ``quantization_config`` declares an algorithm,
            KV-cache format, per-expert group size, or MXFP8 weight block
            size other than this checkpoint's.
    """
    hf_quant_config = resolve_hf_quant_config(huggingface_config, state_dict)
    if not hf_quant_config:
        return None

    method = hf_quant_config.get("quant_method")
    if method not in (None, "modelopt", "fp8"):
        raise ValueError(
            "DeepSeek-V4.1-Flash NVFP4 expects quantization_config.quant_method "
            f"'fp8' or 'modelopt'; got {method!r}"
        )
    # Optional: hf_quant_config.json omits it; quantized_layers is checked
    # per module below.
    moe_quant_algo = hf_quant_config.get("moe_quant_algo")
    if moe_quant_algo is not None and moe_quant_algo != _NVFP4_ALGO:
        raise ValueError(
            "DeepSeek-V4.1-Flash NVFP4 expects quantization_config.moe_quant_algo "
            f"{_NVFP4_ALGO!r}; got {moe_quant_algo!r}"
        )
    block_size = hf_quant_config.get("weight_block_size")
    if block_size is not None and list(block_size) != [MXFP8_BLOCK_SIZE] * 2:
        raise ValueError(
            "DeepSeek-V4.1-Flash NVFP4 expects quantization_config."
            f"weight_block_size {[MXFP8_BLOCK_SIZE] * 2} for its MXFP8 dense "
            f"linears; got {block_size!r}"
        )

    # The shared reader requires quant_method "modelopt".
    normalized = dict(hf_quant_config, quant_method="modelopt")
    expert_modules = {
        f"layers.{i}.ffn.experts" for i in range(num_hidden_layers)
    }
    modules = read_modelopt_quantized_layers(
        normalized,
        allowed={(_NVFP4_ALGO, _NVFP4_GROUP_SIZE)},
        expected_modules=expert_modules,
        kv_cache_quant_algos=(None,),
        model_name="DeepSeek-V4.1-Flash",
    )

    ignore = hf_quant_config.get("ignore")
    if ignore is None:
        ignore = hf_quant_config.get("exclude_modules", [])

    return V41QuantizationInfo(
        nvfp4_expert_layers=frozenset(
            int(module.removeprefix("layers.").removesuffix(".ffn.experts"))
            for module in modules
        ),
        ignore_patterns=tuple(str(pattern) for pattern in ignore),
    )


def _data_parallel_degree(adapter_kwargs: Mapping[str, object]) -> int:
    """The data-parallel degree of the adapter's ``pipeline_config``, else 1."""
    pipeline_config = adapter_kwargs.get("pipeline_config")
    if pipeline_config is None:
        return 1
    assert isinstance(pipeline_config, PipelineConfig)
    return int(pipeline_config.model.data_parallel_degree)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    *,
    defer_routed_experts: bool = False,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """The trunk's weights; the DSpark stages (``mtp.*``) are dropped.

    For the non-speculative graph. Validates ``quantization_config``,
    drops vision tensors, then applies :func:`convert_weight_data` and
    :func:`rewire_trunk_state_dict`. ``defer_routed_experts`` is passed to
    :func:`convert_weight_data`.
    """
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    resolve_quantization_info(
        huggingface_config, state_dict, text_config.num_hidden_layers
    )
    return rewire_trunk_state_dict(
        convert_weight_data(
            {
                name: value.data()
                for name, value in state_dict.items()
                if not name.startswith(DSPARK_CHECKPOINT_PREFIX)
                and not is_skipped_checkpoint_tensor(name)
            },
            num_routed_experts=text_config.n_routed_experts,
            num_dspark_experts=text_config.dspark_n_routed_experts,
            defer_routed_experts=defer_routed_experts,
            data_parallel_degree=_data_parallel_degree(unused_kwargs),
        )
    )


def convert_dspark_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    *,
    defer_routed_experts: bool = False,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Returns the trunk's weights together with the DSpark stages' weights.

    Applies :func:`convert_weight_data` to both, then
    :func:`rewire_trunk_state_dict` to the trunk and
    :func:`rewire_dspark_state_dict` to the ``mtp.*`` stages. Unused DSpark
    tensors are dropped. ``defer_routed_experts`` defers only the trunk's
    NVFP4 experts.
    """
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    resolve_quantization_info(
        huggingface_config, state_dict, text_config.num_hidden_layers
    )
    converted = convert_weight_data(
        {
            name: value.data()
            for name, value in state_dict.items()
            if not is_unused_dspark_tensor(name)
            and not is_skipped_checkpoint_tensor(name)
        },
        num_routed_experts=text_config.n_routed_experts,
        num_dspark_experts=text_config.dspark_n_routed_experts,
        defer_routed_experts=defer_routed_experts,
        data_parallel_degree=_data_parallel_degree(unused_kwargs),
    )
    num_stages = num_dspark_stages(text_config)
    trunk = {
        name: data
        for name, data in converted.items()
        if not name.startswith(DSPARK_CHECKPOINT_PREFIX)
    }
    stages = {
        name: data
        for name, data in converted.items()
        if name.startswith(DSPARK_CHECKPOINT_PREFIX)
    }
    return {
        **rewire_trunk_state_dict(trunk),
        **rewire_dspark_state_dict(stages, last_stage=num_stages - 1),
    }


# The checkpoint stores the vision RMSNorm weights in bfloat16; the reference
# ``inference/vision.py`` applies them in float32.
VISION_NORM_WEIGHT = re.compile(
    r"vision\.(?:blocks\.\d+\.norm[12]|norm)\.weight"
)


class _TensorSource(Protocol):
    def data(self) -> WeightData: ...


def convert_safetensor_vision_state_dict(
    state_dict: Mapping[str, _TensorSource], **unused_kwargs: object
) -> dict[str, WeightData]:
    """The vision tower's, aligner's and span delimiters' weights.

    Keeps the tensors :func:`is_skipped_checkpoint_tensor` names, whose
    checkpoint names are already the attribute paths of
    :class:`~.vision.DeepseekV41VisionModel`, and widens the RMSNorm
    weights to float32.
    """
    out: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        if not is_skipped_checkpoint_tensor(name):
            continue
        data = value.data()
        out[name] = (
            bfloat16_to_float32(data, name)
            if data.dtype == DType.bfloat16
            and VISION_NORM_WEIGHT.fullmatch(name)
            else data
        )
    return out


def vision_load_mismatches(
    converted: Mapping[str, WeightData],
    expected: Mapping[str, tuple[Iterable[SupportsInt], DType]],
) -> list[str]:
    """Every way ``converted`` fails a strict load into the vision model.

    Args:
        converted: :func:`convert_safetensor_vision_state_dict`'s output.
        expected: Each vision model parameter's shape and dtype, by name.

    Returns:
        One line per missing, unexpected or mismatched tensor; empty when
        the load is exact.
    """
    problems = [f"missing {n}" for n in sorted(expected.keys() - converted)]
    problems += [f"unexpected {n}" for n in sorted(converted.keys() - expected)]
    for name in sorted(expected.keys() & converted.keys()):
        shape, dtype = [int(d) for d in expected[name][0]], expected[name][1]
        got = converted[name]
        got_shape = [int(d) for d in got.shape]
        if got_shape != shape or got.dtype != dtype:
            problems.append(
                f"{name}: {got.dtype} {got_shape}, expected {dtype} {shape}"
            )
    return problems
