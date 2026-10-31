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
"""Maps text-trunk weights and extracts the pinned host Engram tables.

Every dense FP8 pair retains its compact E8M0 scales, wo_a included: the
adapter never converts a byte, so the model loads the checkpoint's own
codes and Mach hands the compiled graph the checkpoint's own tensors.
Routed NVFP4 projections keep their separate per-expert input and weight
scales.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights

from ..deepseekV41.weight_adapters import (
    as_uint8,
    bfloat16_to_float32,
    dequantize_fp8_blocks,
    resolve_quantization_info,
    with_dtype,
)
from .model_config import DeepseekV41Config

# (first, second, stacked, sharded) module-path suffixes: two projections
# that read the same activation, stacked into one leaf that runs as one
# GEMM. ``stacked`` holds ``[first; second]`` along N over one device;
# ``sharded`` marks a pair whose leaf shards along N over the mesh, where
# the interleave places each rank's slice as ``[first_r; second_r]``.
STACKED_PROJECTIONS = (
    (".attn.wq_a", ".attn.wkv", ".attn.wqa_wkv", False),
    (
        ".ffn.shared_experts.w1",
        ".ffn.shared_experts.w3",
        ".ffn.shared_experts.w13",
        True,
    ),
)

# The one-byte dtypes a safetensors header may spell differently from what
# the tree declares; `_relabel` only ever relabels within this set.
_E8M0 = DType.float8_e8m0fnu

# Rename table A, the only checkpoint-name -> tree-name renames.
_EXPERT_TENSOR = re.compile(
    r"^(?P<head>.*\.ffn\.experts\.(?P<expert>\d+)\.w[123])"
    r"\.(?P<kind>weight_scale_2|weight_scale|input_scale|weight)$"
)

# The routed experts' static gate/up scalars a derived leaf reduces: the
# EP dispatch quantizes with one uniform scale per layer, the max of the
# layer's experts' ``w1`` and ``w3`` ``input_scale`` values (tree names,
# post-rename) -- both projections read the same dispatched activation,
# so either one's calibrated range can bind.
_EXPERT_GATE_INPUT_SCALE = re.compile(
    r"^(?P<layer>layers\.\d+)\.ffn\.experts\.\d+\.w[13]\.weight\.input_scale$"
)

_GLOBAL_INPUT_SCALE = "ffn.global_input_scale"
"""The derived leaf's suffix under the layer: ``layers.{L}.ffn.<this>``."""

# The Engram embed tables: graph inputs, never module parameters.
_ENGRAM_EMBED = re.compile(r"^layers\.\d+\.engram\.embed\.(weight|scale)$")

# Checkpoint tensors the tree does not hold, each for a stated reason; the
# skip lists account for all 2,671 of them (2,401 mtp + 259 vision + 4
# aligner + 3 image delimiters + 4 engram tables; 188,245 checkpoint
# tensors - 185,574 tree leaves). An unlisted new ``mtp.*``/``vision.*``/
# ``aligner.*`` name must fail loudly here, never silently pass.
_MTP_PREFIX = "mtp."
_VISION_PREFIXES = ("vision.", "aligner.")
_IMAGE_DELIMITERS = ("image_start", "image_end", "image_newline")


def is_skipped_checkpoint_tensor(name: str) -> bool:
    """Whether the frozen tree deliberately does not hold ``name``.

    One shared predicate for every load-path filter, so the skip lists
    cannot desync.

    * ``mtp.*`` -- the DSpark drafter (2,401 tensors). The prefix
      allows every stage tensor by design: the base trunk does not wire
      the drafter, so the whole subtree is intentionally filtered.
    * ``vision.*`` (259), ``aligner.*`` (4), and the three image delimiters
      -- the VL tower the text-only tree does not build. ``gate.bias_vl``
      is NOT here: the router holds it whenever the config has
      ``vision_config``, and the adapter drops it with the config.
    * ``layers.{EG}.engram.embed.{weight,scale}`` -- the 4 host tables,
      ABI graph inputs, returned by :func:`engram_embed_tables` instead.
    """
    if name.startswith(_MTP_PREFIX):
        return True
    if name.startswith(_VISION_PREFIXES) or name in _IMAGE_DELIMITERS:
        return True
    return _ENGRAM_EMBED.match(name) is not None


def to_tree_name(name: str) -> str:
    """Maps a checkpoint tensor name onto its frozen-tree path.

    The identity except rename table A (see the module docstring). Raises
    for nothing: names the tree does not hold are the caller's business
    (see :func:`is_skipped_checkpoint_tensor`).
    """
    match = _EXPERT_TENSOR.match(name)
    if match is None:
        return name
    return f"{match['head']}.weight.{match['kind']}"


def _as_buffer(data: DLPackArray) -> Buffer:
    """``data`` as a ``Buffer``, without copying when it already is one."""
    return data if isinstance(data, Buffer) else Buffer.from_dlpack(data)


def _view_same_width(buffer: Buffer, dtype: DType) -> Buffer:
    """Views ``buffer`` as ``dtype``, same byte width, including 0-d scalars.

    ``Buffer.view`` deduces an unspecified shape from the last axis, which a
    0-d (scalar) buffer like ``weight_scale_2``/``input_scale`` does not
    have; older drivers raise ``IndexError`` there, so the rank-0
    shape is passed through explicitly.
    """
    shape = buffer.shape if buffer.rank == 0 else None
    return buffer.view(dtype, shape)


def _relabel_e8m0(data: WeightData) -> WeightData:
    """Relabels a ``uint8`` read of an ``F8_E8M0`` tensor to ``float8_e8m0fnu``.

    A safetensors reader may yield ``uint8`` for the ``F8_E8M0`` header
    dtype (numpy has no e8m0). The relabel is a view: no byte changes, no
    value conversion (the scale bytes are consumed raw). Any other
    dtype is refused loudly -- a ``float32``/``bfloat16`` ``.scale``
    silently relabeled to e8m0 would keep its real dtype in the buffer
    while lying about it in ``WeightData.dtype``.
    """
    if data.dtype not in (DType.uint8, _E8M0):
        raise ValueError(
            f"{data.name}: F8_E8M0 scale read as {data.dtype}, expected "
            f"uint8 (or {_E8M0}); refusing to relabel"
        )
    buffer = _as_buffer(data.data)
    if data.dtype == DType.uint8:
        buffer = _view_same_width(buffer, _E8M0)
    return WeightData(buffer, data.name, _E8M0, data.shape)


def _relabel_dense_scale(name: str, data: WeightData) -> WeightData:
    """The compact E8M0 ``<proj>.scale`` leaf, dtype-relabeled only.

    No expansion here -- the per-row expansion happens in the graph after
    the TP slice -- and never a value conversion: the bytes are consumed
    raw as ``float8_e8m0fnu``.
    """
    assert name.endswith(".scale"), name
    return _relabel_e8m0(data)


# Rows covered by one E8M0 scale block of a dense MXFP8 weight. A stack's
# first part must end on a whole block so the second part's scale rows land
# on its own weight rows, and a sharded stack's per-rank boundary must not
# split one.
_SCALE_BLOCK = 32


def _row_bytes(weight: WeightData) -> np.ndarray:
    """``weight``'s bytes as ``[N, row_bytes]`` uint8, whatever its dtype."""
    buffer = _as_buffer(weight.data)
    rows = int(weight.shape[0])
    width = buffer.num_elements * buffer.element_size // rows
    return np.from_dlpack(buffer.view(DType.uint8, [rows, width]))


def concat_rows(first: WeightData, second: WeightData, name: str) -> WeightData:
    """``[first; second]`` along axis 0, bytes copied unchanged.

    Zero-byte stand-ins (shaped headers carrying no data) stack by
    name and shape alone, as they load.
    """
    dims = [[int(d) for d in w.shape] for w in (first, second)]
    if first.dtype != second.dtype or dims[0][1:] != dims[1][1:]:
        raise ValueError(
            f"{name}: cannot stack {first.name} {first.dtype} {dims[0]} "
            f"with {second.name} {second.dtype} {dims[1]}"
        )
    shape = [dims[0][0] + dims[1][0], *dims[0][1:]]
    if not dims[0][0] or not dims[1][0]:
        return WeightData(
            Buffer.from_dlpack(np.zeros((0,), np.uint8)),
            name,
            first.dtype,
            Shape(shape),
        )
    joined = np.concatenate([_row_bytes(first), _row_bytes(second)], axis=0)
    buffer = Buffer.from_numpy(joined).view(first.dtype, shape)
    return WeightData(buffer, name, first.dtype, Shape(shape))


def _slice_rows(weight: WeightData, start: int, count: int) -> WeightData:
    """Rows ``[start, start + count)`` of ``weight``, bytes copied."""
    shape = [count, *[int(d) for d in weight.shape[1:]]]
    rows = _row_bytes(weight)[start : start + count].copy()
    buffer = Buffer.from_numpy(rows).view(weight.dtype, shape)
    return WeightData(buffer, weight.name, weight.dtype, Shape(shape))


def _stack_sharded(
    first: WeightData,
    second: WeightData,
    name: str,
    n_devices: int,
) -> WeightData:
    """Stacks a sharded pair at per-rank row-block granularity.

    The stacked leaf shards along N over the mesh, and a rank's contiguous
    slice of a plain ``[first; second]`` concat would hold only one part's
    rows. Interleaving per rank instead -- ``[first_r; second_r]`` for each
    rank ``r`` -- makes every rank's slice the rows it needs, and the
    generic ``Sharded(0)`` host split loads it unchanged. Both parts carry
    equal rows per rank by construction, and the caller checks the split
    lands on whole scale blocks (which makes this leaf's row count divide
    evenly too, at 32 rows per scale row).
    """
    rows = int(first.shape[0])
    if n_devices == 1 or rows == 0:
        # One rank interleaves with itself, and a zero-byte stand-in (a
        # shaped header carrying no data) has nothing to interleave:
        # both are the plain concatenation.
        return concat_rows(first, second, name)
    if rows % n_devices:
        raise ValueError(
            f"{name}: {rows} rows do not split evenly over {n_devices}"
        )
    per_rank = rows // n_devices
    stacked: WeightData | None = None
    for r in range(n_devices):
        block = concat_rows(
            _slice_rows(first, r * per_rank, per_rank),
            _slice_rows(second, r * per_rank, per_rank),
            name,
        )
        stacked = (
            block if stacked is None else concat_rows(stacked, block, name)
        )
    assert stacked is not None
    return stacked


def stack_projections(
    state_dict: dict[str, WeightData], n_devices: int = 1
) -> None:
    """Replaces each :data:`STACKED_PROJECTIONS` pair with its stack, in place.

    ``weight`` always, ``scale`` with it: every dense FP8 projection's N is
    a multiple of the 32-row scale block, so scale block ``i`` of the second
    part lands at block ``N1 / 32 + i``, as its rows do. Nothing here
    changes a byte of either part -- the stack is the parts' own rows, so
    every output column keeps its separate-launch value. A ``sharded`` pair
    interleaves its parts per rank (:func:`_stack_sharded`) so each rank's
    contiguous slice holds both of its halves; ``n_devices`` is the TP
    degree the leaf shards over.

    A pair arrives as its parts or already stacked, never a mix: a prefix
    that carries any of the pair's four leaves must carry all of them (a
    half pair would otherwise boot a silently half-stacked projection --
    the parts are gone from the folded tree, so nothing downstream could
    catch it), parts beside their own stacked leaf are refused rather than
    resolved by overwriting one side with the other, and a stacked leaf
    with no parts passes through untouched (a pre-stacked checkpoint).
    """
    for first, second, stacked, sharded in STACKED_PROJECTIONS:
        leaves = tuple(
            f"{side}.{part}"
            for side in (first, second)
            for part in ("weight", "scale")
        )
        prefixes = sorted(
            {
                name[: -len(leaf)]
                for leaf in leaves
                for name in state_dict
                if name.endswith(leaf)
            }
        )
        for prefix in prefixes:
            present = {
                leaf for leaf in leaves if f"{prefix}{leaf}" in state_dict
            }
            missing = set(leaves) - present
            if missing:
                raise ValueError(
                    f"{prefix}{first} and {prefix}{second} must both be "
                    f"present (missing {', '.join(sorted(missing))})"
                )
            n_first = int(state_dict[f"{prefix}{first}.weight"].shape[0])
            if sharded and n_first != int(
                state_dict[f"{prefix}{second}.weight"].shape[0]
            ):
                # The interleave slices both parts at one per-rank row
                # count; unequal parts would clamp or drop rows and build a
                # wrong-shaped stack (numpy slicing does not raise).
                raise ValueError(
                    f"{prefix}{first}.weight and {prefix}{second}.weight "
                    f"must carry equal rows to interleave per rank over "
                    f"{n_devices}"
                )
            for side in (first, second):
                # Each half's scale must match its own weight's compact
                # shape before anything is popped or stacked: the sharded
                # interleave slices both scales at the first's row count,
                # so a mismatched half would silently truncate or
                # mis-shape the stack (numpy slicing does not raise).
                weight = state_dict[f"{prefix}{side}.weight"]
                scale = state_dict[f"{prefix}{side}.scale"]
                rows = int(weight.shape[0])
                if rows == 0:
                    # A zero-byte stand-in carries no geometry.
                    continue
                if rows % _SCALE_BLOCK:
                    raise ValueError(
                        f"{prefix}{side}.weight: {rows} rows are not whole "
                        f"{_SCALE_BLOCK}-row scale blocks"
                    )
                cols = int(weight.shape[1])
                if cols % _SCALE_BLOCK:
                    # The compact shape derives cols // 32; a nonmultiple
                    # would silently floor and mask a truncated weight.
                    raise ValueError(
                        f"{prefix}{side}.weight: {cols} columns are not "
                        f"whole {_SCALE_BLOCK}-column scale blocks"
                    )
                expected = (
                    rows // _SCALE_BLOCK,
                    cols // _SCALE_BLOCK,
                )
                actual = tuple(int(d) for d in scale.shape)
                if actual != expected:
                    raise ValueError(
                        f"{prefix}{side}.scale: scale shape {actual} does "
                        f"not match the compact shape {expected} of "
                        f"{prefix}{side}.weight"
                    )
            for part in ("weight", "scale"):
                a = f"{prefix}{first}.{part}"
                b = f"{prefix}{second}.{part}"
                out = f"{prefix}{stacked}.{part}"
                if out in state_dict:
                    raise ValueError(
                        f"{a} and {b} stack onto {out}, which the "
                        "checkpoint also supplies: a pair arrives as its "
                        "parts or already stacked, never both"
                    )
                if part == "scale" and n_first % _SCALE_BLOCK:
                    # The second part's scale rows would land on the wrong
                    # weight rows.
                    raise ValueError(
                        f"{a}: {n_first} rows are not whole "
                        f"{_SCALE_BLOCK}-row scale blocks"
                    )
                if sharded and n_first % (n_devices * _SCALE_BLOCK):
                    # A rank boundary would split a scale block of the
                    # first part, and the in-graph per-rank gate/up split
                    # would land mid-block.
                    raise ValueError(
                        f"{a}: {n_first} rows do not shard into whole "
                        f"{_SCALE_BLOCK}-row scale blocks per rank over "
                        f"{n_devices}"
                    )
                pair = (state_dict.pop(a), state_dict.pop(b))
                state_dict[out] = (
                    concat_rows(pair[0], pair[1], out)
                    if not sharded
                    else _stack_sharded(pair[0], pair[1], out, n_devices)
                )


def _max_input_scale(parts: list[WeightData], name: str) -> WeightData:
    """The uniform dispatch scale of one MoE layer, as a ``[1]`` f32 leaf.

    The value is the max of ``parts`` (the layer's experts' static
    ``w1``/``w3`` ``input_scale`` scalars), reusing the winning scalar's
    own bytes so the derived leaf is bit-identical to the in-graph max
    the EP forward used to compute. A zero-byte part marks a shape-only
    stand-in, which carries no values to reduce; the leaf then stands in
    the same way (``concat_rows``'s rule).

    Raises:
        ValueError: If a part is not a one-value float32 scalar (real
            parts) or the stand-ins disagree with real ones.
    """
    values: list[WeightData] = []
    stand_ins = sum(
        1 for part in parts if _as_buffer(part.data).num_elements == 0
    )
    if stand_ins:
        if stand_ins != len(parts):
            raise ValueError(
                f"{name}: zero-byte scale stand-ins beside real ones"
            )
        # A shapes-only dict carries no data; the leaf stands in the
        # same way (``concat_rows``'s rule for zero-byte parts).
        return WeightData(
            Buffer.from_dlpack(np.zeros((0,), np.uint8)),
            name,
            DType.float32,
            Shape([1]),
        )
    for part in parts:
        buffer = _as_buffer(part.data)
        if part.dtype != DType.float32 or buffer.num_elements != 1:
            raise ValueError(
                f"{part.name}: the NVFP4 global scale reduces one-value "
                f"float32 scalars, got {part.dtype} "
                f"{[int(d) for d in part.shape]}"
            )
        values.append(part)
    if not values:
        raise ValueError(f"{name}: no expert scales to reduce")
    best = max(
        values,
        key=lambda part: float(
            np.from_dlpack(_as_buffer(part.data).view(DType.float32, [1]))[0]
        ),
    )
    # Own bytes: the part's buffer is a view into the checkpoint mapping.
    array = np.array(
        np.from_dlpack(_as_buffer(best.data).view(DType.float32, [1]))
    )
    return WeightData(Buffer.from_numpy(array), name, DType.float32, Shape([1]))


def derive_global_input_scales(
    state_dict: dict[str, WeightData],
) -> None:
    """Materializes each MoE layer's uniform NVFP4 dispatch scale, in place.

    The EP dispatch quantizes every rank's activations with one global
    scale per layer -- the max of the layer's experts' static ``w1`` and
    ``w3`` ``input_scale`` values (both projections read the same
    dispatched activation, so either's calibrated range can bind; the
    legacy V4.1 adapter reduces the same pair), the exact reduction the
    EP forward ran in-graph as a cross-device scalar hub (whose
    per-step gather/scatter lowered to ``mgp.device_wait`` pairs). The
    graph now reads the derived leaf, replicated at load; the per-expert
    scales it reduces stay in the dict untouched, so every consumer of
    the originals is unchanged.

    A layer is derived only when the checkpoint supplies at least one of
    its expert gate scales (a full checkpoint supplies all of them); the
    reduction over what is present is exact for the full checkpoint and
    deterministic for partial fixtures.
    """
    per_layer: dict[str, list[WeightData]] = {}
    for name, data in sorted(state_dict.items()):
        match = _EXPERT_GATE_INPUT_SCALE.match(name)
        if match is not None:
            per_layer.setdefault(match["layer"], []).append(data)
    for layer, parts in per_layer.items():
        state_dict[f"{layer}.{_GLOBAL_INPUT_SCALE}"] = _max_input_scale(
            parts, f"{layer}.{_GLOBAL_INPUT_SCALE}"
        )


# The expert gate/up scalar whose value the fused kernel shares: an EP
# checkpoint must calibrate ``w1`` and ``w3`` alike, because the fused
# gate/up weight keeps one ``weight_scale_2`` for both halves.
_EXPERT_GATE_UP_SCALE_2 = re.compile(
    r"^(?P<path>.*\.ffn\.experts\.(?P<expert>\d+))\.w(?P<proj>[13])"
    r"\.weight_scale_2$"
)


def _f32_scalar(data: WeightData) -> float:
    """The value of a one-element float32 scalar leaf."""
    buffer = _as_buffer(data.data)
    if data.dtype != DType.float32 or buffer.num_elements != 1:
        raise ValueError(
            f"{data.name}: expected a one-value float32 scalar, got "
            f"{data.dtype} {[int(d) for d in data.shape]}"
        )
    return float(np.from_dlpack(buffer.view(DType.float32, [1]))[0])


def check_expert_gate_up_scale_pairs(
    state_dict: Mapping[str, WeightData],
) -> None:
    """Refuses a checkpoint whose expert gate/up scalar scales disagree.

    The EP forward fuses each expert's ``w1``/``w3`` into one grouped
    kernel, and the NVFP4 concat keeps the first tensor's
    ``weight_scale_2`` for both halves; a checkpoint calibrating the two
    projections apart would silently mis-scale the up half (the trunk's
    tensor-parallel path keeps them separate and stays exact, which is
    why the tensor-parallel legacy adapter only warns). The check runs
    only under expert parallelism, where the fusion's assumption is
    load-bearing; it reads checkpoint names, before rename table A.

    Raises:
        ValueError: If any expert's ``w1`` and ``w3`` carry a different
            ``weight_scale_2``, or one arrives without the other.
    """
    per_expert: dict[str, dict[str, WeightData]] = {}
    for name, data in sorted(state_dict.items()):
        match = _EXPERT_GATE_UP_SCALE_2.match(name)
        if match is not None:
            per_expert.setdefault(match["path"], {})[match["proj"]] = data
    for path, projections in sorted(per_expert.items()):
        missing = {"1", "3"} - projections.keys()
        if missing:
            raise ValueError(
                f"{path}: the expert's w1 and w3 must both carry "
                f"weight_scale_2 (missing w{min(missing)})"
            )
        w1, w3 = projections["1"], projections["3"]
        if _f32_scalar(w1) != _f32_scalar(w3):
            raise ValueError(
                f"{path}: w1 and w3 weight_scale_2 disagree "
                f"({_f32_scalar(w1)} vs {_f32_scalar(w3)}); the "
                "expert-parallel forward fuses the two projections with "
                "one shared scalar scale, so a checkpoint that calibrates "
                "them apart cannot serve under --ep-size. Serve with "
                "--ep-size 1 instead: the tensor-parallel path runs the "
                "projections separately and stays exact."
            )


class AdapterResult(dict[str, WeightData]):
    """Module weights with the host Engram tables kept alongside them."""

    def __init__(
        self,
        state_dict: dict[str, WeightData],
        engram_tables: list[EngramTable],
    ) -> None:
        super().__init__(state_dict)
        self.engram_tables = engram_tables

    @property
    def state_dict(self) -> dict[str, WeightData]:
        """The state dict the result carries."""
        return self


@dataclass(frozen=True)
class EngramTable:
    """One Engram embed table, as zero-copy views for the loader.

    ``weight_codes``/``scale_codes`` alias the checkpoint's own mapping
    (``float8_e4m3fn`` [rows, 256] / ``float8_e8m0fnu`` [rows, 8], viewed
    as uint8 bytes; widening either would materialize ~12.3 GB per
    table). The loader copies them into its pinned allocation and then
    drops the table: at production size a held copy is ~98 GB of host
    RAM. ``layer`` is the tree layer id that owns it, so the loader can
    name the ABI inputs by ordinal.
    """

    layer: int
    rows: int
    weight_codes: np.ndarray
    scale_codes: np.ndarray

    def table_names(self, ordinal: int) -> tuple[str, str]:
        """The ABI graph-input names this table feeds.

        :func:`.interface.extra_inputs` declares
        ``engram_table_{ordinal}_{weight,scale}``.
        """
        return (
            f"engram_table_{ordinal}_weight",
            f"engram_table_{ordinal}_scale",
        )


def _engram_tables(
    state_dict: Mapping[str, WeightData], engram_layer_ids: tuple[int, ...]
) -> list[EngramTable]:
    """Extracts the Engram embed tables from ``state_dict`` as views.

    Zero-copy uint8 views of the checkpoint mapping, never widened and
    never read into Python memory, and the tensors are dropped from the
    returned weight dict because the tables are graph inputs. Dummy
    loads must never random-fill these: a loader that has no table
    simply passes nothing, and this function rejects the missing or
    incomplete table; the model's bind validation checks the rows and
    columns the tables carry against the ABI.
    """
    tables: list[EngramTable] = []
    by_layer: dict[int, tuple[np.ndarray | None, np.ndarray | None, int]] = {}
    for name, data in state_dict.items():
        match = _ENGRAM_EMBED.match(name)
        if match is None:
            continue
        layer = int(name.split(".")[1])
        codes = as_uint8(data.data)
        weight, scale, rows = by_layer.get(layer, (None, None, 0))
        if match[1] == "weight":
            weight = codes
            rows = int(data.shape[0])
        else:
            scale = codes
        by_layer[layer] = (weight, scale, rows)
    for layer in engram_layer_ids:
        if layer not in by_layer:
            raise ValueError(
                f"layers.{layer}.engram.embed: missing from the checkpoint"
            )
        weight, scale, rows = by_layer[layer]
        if weight is None or scale is None:
            raise ValueError(f"layers.{layer}.engram.embed: incomplete table")
        tables.append(EngramTable(layer, rows, weight, scale))
    return tables


def engram_embed_tables(
    weights: Weights, engram_layer_ids: Sequence[int]
) -> list[EngramTable]:
    """Reads only the checkpoint's Engram embed tables, as zero-copy views.

    For a loader that binds the tables apart from the module's weights.
    """
    return _engram_tables(
        {
            name: value.data()
            for name, value in weights.items()
            if _ENGRAM_EMBED.match(name) is not None
        },
        tuple(engram_layer_ids),
    )


def convert_weight_data(
    state_dict: Mapping[str, WeightData],
    *,
    has_vision: bool = True,
    engram_layer_ids: tuple[int, ...] = (),
    n_devices: int = 1,
) -> AdapterResult:
    """Applies the frozen representation rules to loaded ``WeightData``.

    The transform list, as data:

    1. Rename table A on every routed-expert leaf (the only renames).
    2. Dtype relabel ``uint8`` -> ``float8_e8m0fnu`` on ``F8_E8M0`` scale
       leaves (dense ``.scale`` and the expert ``.weight_scale`` never
       needs it -- the expert scale is ``F8_E4M3``, one byte either way).
    3. Drop ``gate.bias_vl`` when the config has no ``vision_config``;
       every other gate tensor stays required.
    4. Extract the Engram embed tables into :class:`EngramTable` views.
    5. Derive each MoE layer's uniform NVFP4 dispatch scale
       (:func:`derive_global_input_scales`) -- the max of the layer's
       experts' static ``w1``/``w3`` ``input_scale`` values, a new tree
       leaf the EP dispatch reads instead of reducing it in-graph.
    6. Stack the :data:`STACKED_PROJECTIONS` pairs into their one-GEMM
       leaves, trunk tensors only.

    Nothing else changes a byte: no dense scale expansion, no norm
    upcast, no head upcast; ``wo_a`` keeps its checkpoint E4M3 codes
    and compact E8M0 scales like every other dense FP8 pair, so the
    graph runs the MXFP8 projection on the checkpoint's own bytes.

    Args:
        state_dict: The checkpoint's weights, keyed by checkpoint tensor
            name, allow-list entries already filtered by the caller (or
            not: they are skipped here either way).
        has_vision: Whether the config carries ``vision_config``; when it
            does not, ``gate.bias_vl`` is dropped from the output (the
            tree declares no such leaf).
        engram_layer_ids: The tree layers that own an Engram module; the
            embed tables of exactly these layers are extracted, in order.
        n_devices: The TP degree the sharded stacked pairs interleave at;
            the pipeline entry point reads it off the pipeline config's
            ``device_specs``.

    Raises:
        ValueError: If an Engram layer's table is missing or incomplete,
            if two checkpoint tensors rename onto one tree leaf (an
            adapter rule bug -- never a checkpoint property: the frozen
            map is injective), if a derived global scale reduces anything
            but one-value float32 scalars, or if a stacked pair arrives
            incomplete, beside its own stacked leaf, or off a whole
            32-row scale-block or per-rank boundary.
    """
    if engram_layer_ids:
        tables = _engram_tables(state_dict, engram_layer_ids)
    else:
        tables = []
    out = _rename_checkpoint(state_dict, has_vision=has_vision)
    derive_global_input_scales(out)
    stack_projections(out, n_devices)
    return AdapterResult(out, tables)


def _rename_checkpoint(
    state_dict: Mapping[str, WeightData], *, has_vision: bool
) -> dict[str, WeightData]:
    """Maps each checkpoint tensor onto its tree leaf.

    Each tensor is renamed, relabeled or dropped onto its leaf, before any
    derived or stacked leaf materializes. The stage Mach's Rust adapter mirrors
    this (``WeightLoader::adapt``) and the weight-contract record covers it.
    """
    out: dict[str, WeightData] = {}
    for name in sorted(state_dict):
        if is_skipped_checkpoint_tensor(name):
            continue
        data = state_dict[name]
        if not has_vision and name.endswith(".ffn.gate.bias_vl"):
            continue
        tree_name = to_tree_name(name)
        if tree_name in out:
            raise ValueError(
                f"{name} renames onto {tree_name}, already fed by an "
                "earlier tensor: the checkpoint-to-tree map must be "
                "injective"
            )
        if name.endswith(".scale") and not name.endswith("weight_scale"):
            # The only ``.scale`` leaves the tree holds are the dense
            # FP8 compact E8M0 scales -- ``wo_a``'s included -- relabel
            # only.
            data = _relabel_dense_scale(name, data)
        out[tree_name] = data
    return out


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: object,
    pipeline_config: Any | None = None,
    **unused_kwargs: object,
) -> AdapterResult:
    """The pipeline-facing entry point: the adapter's full result.

    Validates the checkpoint's quantization config up front (a loud
    failure before any tensor is converted rather than a silent
    misroute: ``quantized_layers``, never the ``ignore`` complement).
    Allow-listed tensors are dropped **before** ``.data()``: reading a
    skipped tensor would materialize it into host RAM (an unsupported
    header dtype on a tensor the tree never holds must not fail the
    load), so only the names the tree actually holds are read -- plus
    the Engram embeds, the one deliberate allow-listed fetch (they are
    this loader's own tables, not tree weights, and the adapter keeps
    only zero-copy views of them).

    Returns :class:`AdapterResult`: ``state_dict`` feeds the module's
    parameter tree, and ``engram_tables`` -- the Engram embed tables as
    zero-copy uint8 views, in ``engram_layer_ids`` order -- feed the ABI
    graph inputs ``engram_table_{ordinal}_{weight,scale}``: the pipeline
    model binds them as pinned static inputs
    (``_bind_engram_static_inputs``), named by
    :func:`.interface.extra_input_names`; the tables never enter the
    module weight dict. The interleave degree for the sharded stacked
    pairs reads off ``pipeline_config`` (the pipeline model builds one
    mesh over all its devices), so the interleave matches the mesh the
    module shards over -- except under data parallelism, where the
    stacked pairs stay whole per rank and the parts concat plainly.

    Raises:
        ValueError: Under expert parallelism (``ep_size > 1``), if any
            expert's ``w1`` and ``w3`` ``weight_scale_2`` disagree
            (:func:`check_expert_gate_up_scale_pairs`); and everything
            :func:`convert_weight_data` raises.
    """
    text = DeepseekV41Config.get_text_config(huggingface_config)
    resolve_quantization_info(
        huggingface_config,
        state_dict,
        int(text.num_hidden_layers),
    )
    has_vision = False
    engram_layer_ids = tuple(
        int(layer) for layer in getattr(text, "engram_layer_ids", ())
    )
    if pipeline_config is None:
        n_devices = 1
        ep_size = 1
    else:
        n_devices = len(pipeline_config.model.device_specs)
        if pipeline_config.model.data_parallel_degree > 1:
            # Under data parallelism the stacked pairs stay whole on
            # every rank (``StackedSharedExpert`` skips the shard), so
            # the parts concat plainly: the per-rank interleave is the
            # TP layout the module's half-split pairs with.
            n_devices = 1
        ep_size = pipeline_config.runtime.ep_size
    loaded = {
        name: value.data()
        for name, value in sorted(state_dict.items())
        if (
            not is_skipped_checkpoint_tensor(name)
            # The Engram embeds are allow-listed but still read: the
            # one deliberate allow-listed read (they are the loader's
            # tables -- dropped from the dict, not from the load).
            or _ENGRAM_EMBED.match(name) is not None
        )
        and (has_vision or not name.endswith(".ffn.gate.bias_vl"))
    }
    if ep_size > 1:
        check_expert_gate_up_scale_pairs(loaded)
    return convert_weight_data(
        loaded,
        has_vision=has_vision,
        engram_layer_ids=engram_layer_ids,
        n_devices=n_devices,
    )


# --------------------------------------------------------------------------- #
# The DSpark drafter's weights (mtp.* -> the drafter's parameter tree)
# --------------------------------------------------------------------------- #

# The trunk's allow-list skips ``mtp.*`` because the trunk tree never holds
# them; the drafter tree this section maps is their consumer. Names, dtypes
# and shapes follow the checkpoint (2,401 mtp tensors, 27 patterns).
DSPARK_CHECKPOINT_PREFIX = _MTP_PREFIX
DSPARK_MODULE_PREFIX = "dspark"
"""The model attribute holding the drafter, so the prefix of its names."""

_DSPARK_STAGE_TENSOR = re.compile(r"^mtp\.(?P<stage>\d+)\.(?P<rest>.+)$")
_DSPARK_EXPERT_TENSOR = re.compile(
    r"^(?P<stage>mtp\.\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"w(?P<proj>[123])\.(?P<kind>weight|scale)$"
)
# The attention projections the drafter runs on bf16 host-dequantized
# weights: the reference's MAX_QUANTIZED excludes them from activation
# quantization, so the checkpoint's e4m3-plus-compact-e8m0 pairs become
# plain bf16 GEMM operands at load time.
_DSPARK_HOST_DEQUANT = ("wq_a", "wq_b", "wkv", "wo_a")


def _stack_dspark_experts(
    state_dict: Mapping[str, WeightData], prefix: str
) -> dict[str, WeightData]:
    """Stacks one stage's per-expert MXFP4 tensors for the grouped kernels.

    ``_PackedExpertWeights`` reads ``[E, N, K/2]`` uint8 weights and
    ``[E, N, K/32]`` e8m0 scales stacked by expert id -- exactly this
    stacking, with ``gate_up_proj`` holding ``w1``'s rows ahead of
    ``w3``'s (the SwiGLU's gate is ``w1``'s output).

    Args:
        state_dict: The stage's whole ``mtp.*`` mapping.
        prefix: ``mtp.{stage}.ffn.experts``.

    Returns:
        ``gate_up_proj.weight/.weight_scale`` and ``down_proj.weight/
        .weight_scale`` keyed by their tree-relative names.

    Raises:
        ValueError: If expert ids are not dense ``0..E-1`` or an expert is
            missing one of its six tensors.
    """
    experts: dict[int, dict[str, str]] = {}
    for name in state_dict:
        match = _DSPARK_EXPERT_TENSOR.match(name)
        if match is None or f"{match['stage']}.ffn.experts" != prefix:
            continue
        experts.setdefault(int(match["expert"]), {})[
            f"w{match['proj']}.{match['kind']}"
        ] = name
    ids = sorted(experts)
    if not ids or ids != list(range(len(ids))):
        raise ValueError(f"{prefix}: expert ids are not dense: {ids}")

    parts: list[dict[str, np.ndarray]] = []
    for e in ids:
        held = experts[e]
        expected = {
            f"w{proj}.{kind}"
            for proj in (1, 2, 3)
            for kind in ("weight", "scale")
        }
        if set(held) != expected:
            raise ValueError(
                f"{prefix}.{e}: expected {sorted(expected)}, got {sorted(held)}"
            )
        parts.append(
            {key: as_uint8(state_dict[name].data) for key, name in held.items()}
        )
    return {
        "gate_up_proj.weight": with_dtype(
            np.stack(
                [
                    np.concatenate(
                        [part["w1.weight"], part["w3.weight"]], axis=0
                    )
                    for part in parts
                ]
            ),
            DType.uint8,
            f"{DSPARK_MODULE_PREFIX}.{prefix}.gate_up_proj.weight",
        ),
        "gate_up_proj.weight_scale": with_dtype(
            np.stack(
                [
                    np.concatenate([part["w1.scale"], part["w3.scale"]], axis=0)
                    for part in parts
                ]
            ),
            DType.float8_e8m0fnu,
            f"{DSPARK_MODULE_PREFIX}.{prefix}.gate_up_proj.weight_scale",
        ),
        "down_proj.weight": with_dtype(
            np.stack([part["w2.weight"] for part in parts]),
            DType.uint8,
            f"{DSPARK_MODULE_PREFIX}.{prefix}.down_proj.weight",
        ),
        "down_proj.weight_scale": with_dtype(
            np.stack([part["w2.scale"] for part in parts]),
            DType.float8_e8m0fnu,
            f"{DSPARK_MODULE_PREFIX}.{prefix}.down_proj.weight_scale",
        ),
    }


def rewire_dspark_state_dict(
    state_dict: Mapping[str, WeightData],
    *,
    last_stage: int,
    load_confidence_head: bool = False,
) -> dict[str, WeightData]:
    """Maps ``mtp.{i}.*`` tensors onto the DSpark drafter's parameter tree.

    The drafter's tree keeps the trunk's shapes (no query-projection
    fusing, no shared-expert ``gate_up`` fusion), so the mapping is renames
    plus three real transforms: the four unquantized attention projections
    are host-dequantized to bf16, the per-expert MXFP4 tensors are stacked
    into the grouped kernels' packed layout, and the compact e8m0 scales
    the tree keeps are relabeled to ``float8_e8m0fnu``. ``gate.bias_vl``
    drops (the drafter routes text only); the confidence head's
    projection drops by default -- the reference never runs adaptive
    verification -- and ``load_confidence_head`` loads and
    widens it instead, where a missing tensor fails loud rather than
    silently skipping the head.

    Args:
        state_dict: Only ``mtp.*`` tensors, as loaded from the checkpoint.
        last_stage: The last stage's index (``num_dspark_layers - 1``).
        load_confidence_head: Load ``confidence_head.proj.weight`` widened
            to float32 (the module holds it that way) instead of dropping.

    Returns:
        The drafter's weights keyed by tree path under ``dspark.``.

    Raises:
        ValueError: On a tensor outside ``mtp.{i}.*``, a missing partner
            (a dequant weight without its scale, a hoisted tensor under
            the wrong stage), non-dense expert ids, a requested-but-missing
            confidence head, or any input name the mapping does not consume.
    """
    stages: set[int] = set()
    for name in state_dict:
        match = _DSPARK_STAGE_TENSOR.match(name)
        if match is None:
            raise ValueError(f"{name}: not a DSpark stage (mtp.*) tensor")
        stages.add(int(match["stage"]))

    out: dict[str, WeightData] = {}
    consumed: set[str] = set()

    def take(name: str) -> WeightData:
        if name not in state_dict:
            raise ValueError(f"{name}: missing from the DSpark stages")
        consumed.add(name)
        return state_dict[name]

    def put(name: str, data: WeightData) -> None:
        if name in out:
            raise ValueError(f"{name}: produced twice")
        out[name] = data

    def keep_scale(prefix: str, target: str) -> None:
        put(target, _relabel_e8m0(take(prefix)))

    d = DSPARK_MODULE_PREFIX
    for stage in sorted(stages):
        s, p = f"{d}.stages.{stage}", f"mtp.{stage}"
        put(f"{s}.attn.attn_sink", take(f"{p}.attn.attn_sink"))
        put(f"{s}.attn.q_norm.weight", take(f"{p}.attn.q_norm.weight"))
        put(f"{s}.attn.kv_norm.weight", take(f"{p}.attn.kv_norm.weight"))
        for proj in _DSPARK_HOST_DEQUANT:
            put(
                f"{s}.attn.{proj}.weight",
                dequantize_fp8_blocks(
                    take(f"{p}.attn.{proj}.weight"),
                    take(f"{p}.attn.{proj}.scale"),
                    f"{s}.attn.{proj}.weight",
                ),
            )
        put(f"{s}.attn.wo_b.weight", take(f"{p}.attn.wo_b.weight"))
        keep_scale(f"{p}.attn.wo_b.scale", f"{s}.attn.wo_b.scale")
        put(f"{s}.attn_norm.weight", take(f"{p}.attn_norm.weight"))
        put(f"{s}.ffn_norm.weight", take(f"{p}.ffn_norm.weight"))
        for site in ("attn", "ffn"):
            for leaf in ("fn", "base", "scale"):
                put(f"{s}.hc_{site}_{leaf}", take(f"{p}.hc_{site}_{leaf}"))
        put(f"{s}.ffn.gate.weight", take(f"{p}.ffn.gate.weight"))
        put(f"{s}.ffn.gate.bias", take(f"{p}.ffn.gate.bias"))
        if f"{p}.ffn.gate.bias_vl" in state_dict:
            consumed.add(f"{p}.ffn.gate.bias_vl")
        for proj in ("w1", "w2", "w3"):
            put(
                f"{s}.ffn.shared_experts.{proj}.weight",
                take(f"{p}.ffn.shared_experts.{proj}.weight"),
            )
            keep_scale(
                f"{p}.ffn.shared_experts.{proj}.scale",
                f"{s}.ffn.shared_experts.{proj}.scale",
            )
        for name, data in _stack_dspark_experts(
            state_dict, f"{p}.ffn.experts"
        ).items():
            put(f"{s}.ffn.experts.{name}", data)
        for name in state_dict:
            match = _DSPARK_EXPERT_TENSOR.match(name)
            if match is not None and match["stage"] == p:
                consumed.add(name)

    put(f"{d}.main_proj.weight", take("mtp.0.main_proj.weight"))
    keep_scale("mtp.0.main_proj.scale", f"{d}.main_proj.scale")
    put(f"{d}.main_norm.weight", take("mtp.0.main_norm.weight"))
    put(f"{d}.norm.weight", take(f"mtp.{last_stage}.norm.weight"))
    put(
        f"{d}.markov_head.embed.weight",
        take(f"mtp.{last_stage}.markov_head.embed.weight"),
    )
    put(
        f"{d}.markov_head.head.weight",
        take(f"mtp.{last_stage}.markov_head.head.weight"),
    )
    confidence = f"mtp.{last_stage}.confidence_head.proj.weight"
    if load_confidence_head:
        put(
            f"{d}.confidence_head.proj.weight",
            bfloat16_to_float32(take(confidence), confidence),
        )
    elif confidence in state_dict:
        consumed.add(confidence)

    leftover = sorted(set(state_dict) - consumed)
    if leftover:
        raise ValueError(f"unconsumed DSpark tensors: {leftover}")
    return out


def convert_dspark_state_dict(
    state_dict: Mapping[str, Weights],
    *,
    num_stages: int,
    load_confidence_head: bool = False,
) -> dict[str, WeightData]:
    """The drafter-facing entry point: ``mtp.*`` in, drafter tree out.

    The DSpark stages live in the target checkpoint itself (the trunk
    loader skips the prefix), so the serving wrapper passes the same
    mapping here and reads out the drafter's half.

    Args:
        state_dict: The checkpoint's weights (mtp.* picked out here).
        num_stages: The stage count from the config
            (``num_dspark_layers``); the last stage owns the final norm
            and both heads.
        load_confidence_head: See :func:`rewire_dspark_state_dict`.

    Raises:
        ValueError: If the checkpoint holds no ``mtp.*`` tensors at all.
    """
    loaded = {
        name: value.data()
        for name, value in sorted(state_dict.items())
        if name.startswith(DSPARK_CHECKPOINT_PREFIX)
    }
    if not loaded:
        raise ValueError("the checkpoint holds no mtp.* DSpark tensors")
    return rewire_dspark_state_dict(
        loaded,
        last_stage=num_stages - 1,
        load_confidence_head=load_confidence_head,
    )
