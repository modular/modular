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

"""Graph-compiler shape-rearrange model cache for the MO interpreter.

Covers the shape-rearrange family: ``Concat``, ``Split``, ``Slice``, the pads
(``PadConstant``/``PadReflect``/``PadRepeat``), and ``Tile``. Each runs one
graph node that only rearranges/copies data (no arithmetic, no weights).

Structural parameters are runtime, not baked: pad widths, repeat counts, and
slice/split bounds arrive as host tensor operands, so one compiled graph serves
every value of them (the output shape is data-dependent). The handler always
knows the concrete output shape, so it re-views the model output to that shape
(zero-copy).

Two keying schemes:

- ``Concat``/``Split`` canonicalize to rank 3 ``[outer, axis, inner]`` (a
  zero-copy view; see :func:`canonical_rank3`) and key on ``(op, device,
  dtype)``. Concat folds its variadic operand count with a pairwise 2-input
  graph; split slices one chunk at a time.
- ``Pad``/``Tile``/``Slice`` keep their natural N-D form and key on
  ``(op, device, dtype, rank)``: their structural operand (paddings ``[2*rank]``,
  repeats ``[rank]``, starts/stops/steps ``[rank]``) needs a static length tied
  to rank, so rank rides in the key (bounded by ``MAX_RANK``).

Compile modes mirror :mod:`reduce_axis_gc` exactly (lazy-per-target by default,
batched sweep under ``MAX_EAGER_OP_PRECOMPILE=1``, warm-stamp adoption). Must
not import from ``handlers.py``.

These are pure data-movement ops (no arithmetic), so a tensor is copied by bit
width: the handler bit-casts every dtype to the same-width unsigned int (a
zero-copy view), runs the copy, and views the result back. One graph per width
(uint8/16/32/64) therefore serves every dtype uniformly — including float16
(no typed CPU kernel) and bool on GPU — and every width compiles on every
device. See :func:`uint_view_dtype`.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from math import prod

from max import _core, engine
from max._core.dialects import kgen, mo, rmo
from max._interpreter_ops import gc_compile
from max._mlir_context import in_default_mlir_context
from max.driver import Device, DeviceSpec, accelerator_count, load_devices
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, ops

logger = logging.getLogger(__name__)

MAX_RANK = 5  # Matches op_utils.MAX_RANK; sweep ranks 1..MAX_RANK.

_WIDTH_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]
_UINT_FOR_SIZE = {
    1: DType.uint8,
    2: DType.uint16,
    4: DType.uint32,
    8: DType.uint64,
}


def uint_view_dtype(dtype: DType) -> DType:
    """The same-bit-width unsigned int a dtype is bit-cast to for copying.

    Raises:
        NotImplementedError: For sub-byte dtypes (e.g. ``float4_e2m1fn``), which
            pack multiple elements per byte and so cannot be reinterpreted
            element-for-element as a whole-byte unsigned int.
    """
    bits = dtype.size_in_bits
    if bits % 8 != 0 or bits // 8 not in _UINT_FOR_SIZE:
        raise NotImplementedError(
            f"shape-rearrange GC path does not support sub-byte dtype {dtype}"
        )
    return _UINT_FOR_SIZE[bits // 8]


class DeviceClass(Enum):
    """Which devices an op supports."""

    ALL = "all"  # CPU + accelerators
    CPU_ONLY = "cpu"  # MO_HostOnly ops


@dataclass(frozen=True)
class RearrangeSpec:
    """How to build one op's graph, its devices, and rank-keying."""

    build_graph: Callable[["Module", str, Device, DType, "int | None"], None]
    devices: DeviceClass
    rank_keyed: bool
    max_rank: int | None = None  # Cap for rank sweep; None uses MAX_RANK.


_REARRANGE_OPS: dict[type[_core.Operation], RearrangeSpec] = {}


def _register_spec(op_type: type[_core.Operation], spec: RearrangeSpec) -> None:
    _REARRANGE_OPS[op_type] = spec


_REARRANGE_OPS_BY_NAME: dict[str, RearrangeSpec] = {}


def _refresh_by_name() -> None:
    _REARRANGE_OPS_BY_NAME.clear()
    _REARRANGE_OPS_BY_NAME.update(
        {op_type.__name__: spec for op_type, spec in _REARRANGE_OPS.items()}
    )


def _spec_for(op_type: type[_core.Operation]) -> RearrangeSpec | None:
    name = gc_compile.canonical_op_name(op_type, _REARRANGE_OPS_BY_NAME)
    return _REARRANGE_OPS_BY_NAME.get(name)


_MODEL_CACHE: dict[str, engine.Model] = {}

_DEVICES = load_devices([DeviceSpec.cpu()]) + load_devices(
    [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
)


def _devices_for(spec: RearrangeSpec) -> list[Device]:
    if spec.devices is DeviceClass.CPU_ONLY:
        return [d for d in _DEVICES if d.label == "cpu"]
    return list(_DEVICES)


def _supported_dtypes(spec: RearrangeSpec, device: Device) -> list[DType]:
    return list(_WIDTH_DTYPES)


def canonical_rank3(shape: Sequence[int], axis: int) -> tuple[int, int, int]:
    """Collapse *shape* to rank 3 ``[outer, axis, inner]`` (axis normalized)."""
    ndim = len(shape)
    if axis < 0:
        axis += ndim
    return (prod(shape[:axis]), shape[axis], prod(shape[axis + 1 :]))


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int | None,
) -> str:
    """Graph ``sym_name`` and cache key for one target."""
    name = gc_compile.canonical_op_name(op_type, _REARRANGE_OPS_BY_NAME)
    rank_tag = f"_r{rank}" if rank is not None else ""
    return f"rearrange_{name}_{device.label}_{device.id}_{dtype.name}{rank_tag}"


def _ranks_for(spec: RearrangeSpec) -> list[int | None]:
    if not spec.rank_keyed:
        return [None]
    top = spec.max_rank if spec.max_rank is not None else MAX_RANK
    return list(range(1, top + 1))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    spec = _spec_for(op_type)
    if spec is None:
        return False
    if device not in _devices_for(spec):
        return False
    return dtype in _supported_dtypes(spec, device)


_SWEPT = False


@in_default_mlir_context
def compile_shape_rearrange_sweep() -> None:
    """Compile every supported (op, device, dtype, rank) in one batched load_all."""
    global _SWEPT
    module = Module()
    for op_type, spec in _REARRANGE_OPS.items():
        for device in _devices_for(spec):
            for dtype in _supported_dtypes(spec, device):
                if not _is_supported(op_type, device, dtype):
                    continue
                for rank in _ranks_for(spec):
                    spec.build_graph(
                        module, op_type.__name__, device, dtype, rank
                    )
    session = engine.InferenceSession(devices=list(_DEVICES))
    _MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    _SWEPT = True


@in_default_mlir_context
def _compile_target(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int | None,
) -> engine.Model:
    module = Module()
    spec = _spec_for(op_type)
    assert spec is not None, f"unsupported op {op_type!r} reached compile"
    spec.build_graph(module, op_type.__name__, device, dtype, rank)
    session = gc_compile.session_for(device)
    _MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    return _MODEL_CACHE[_graph_name(op_type, device, dtype, rank)]


def model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int | None = None,
) -> engine.Model:
    """Return the compiled model for the given target (lazy by default)."""
    key = _graph_name(op_type, device, dtype, rank)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if not _is_supported(op_type, device, dtype):
        raise KeyError(
            f"Unsupported shape-rearrange op/device/dtype for key {key!r}."
        )
    spec = _spec_for(op_type)
    if spec is not None and spec.rank_keyed and rank not in _ranks_for(spec):
        # Same unsupported-target signal as the dtype check above: a clean error
        # rather than the GC kernel's comptime rank assert firing deep in
        # compilation (e.g. tile is capped at rank 4).
        raise KeyError(
            f"Unsupported shape-rearrange rank {rank} for {op_type.__name__}"
            f" (max {_ranks_for(spec)[-1]}); key {key!r}."
        )
    if gc_compile.should_precompile():
        raise KeyError(
            f"No pre-compiled shape-rearrange model for key {key!r}."
            f" Unset {gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR} to compile lazily."
        )
    with gc_compile.COMPILE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        global _SWEPT
        if not _SWEPT and gc_compile.warm_stamp_matches():
            _SWEPT = True
            try:
                compile_shape_rearrange_sweep()
            except Exception:
                logger.warning(
                    "Eager interpreter warm-cache adoption failed; compiling"
                    " shape-rearrange targets on demand.",
                    exc_info=True,
                )
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                return cached
        return _compile_target(op_type, device, dtype, rank)


def _build_concat_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """Pairwise 2-input concat at axis=1 of rank-3 [outer, a/b, inner]."""
    dev = DeviceRef.from_device(device)
    a = TensorType(dtype, ["d0", "a1", "d2"], device=dev)
    b = TensorType(dtype, ["d0", "b1", "d2"], device=dev)
    g = Graph(
        _graph_name(mo.ConcatOp, device, dtype, rank),
        input_types=[a, b],
        module=module,
    )
    with g:
        x, y = (v.tensor for v in g.inputs)
        g.output(ops.concat([x, y], axis=1))


_register_spec(
    mo.ConcatOp,
    RearrangeSpec(
        build_graph=_build_concat_graph,
        devices=DeviceClass.ALL,
        rank_keyed=False,
    ),
)


def _build_split_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """Slice one chunk along axis=1 of rank-3 [outer, D, inner]."""
    dev = DeviceRef.from_device(device)
    x = TensorType(dtype, ["d0", "d1", "d2"], device=dev)
    start = TensorType(DType.int64, [], device=DeviceRef.CPU())
    stop = TensorType(DType.int64, [], device=DeviceRef.CPU())
    g = Graph(
        _graph_name(mo.SplitOp, device, dtype, rank),
        input_types=[x, start, stop],
        module=module,
    )
    with g:
        data, lo, hi = (v.tensor for v in g.inputs)
        chunk = ops.slice_tensor(
            data,
            [slice(None), (slice(lo, hi), "s"), slice(None)],
        )
        g.output(chunk)


_register_spec(
    mo.SplitOp,
    RearrangeSpec(
        build_graph=_build_split_graph,
        devices=DeviceClass.ALL,
        rank_keyed=False,
    ),
)


def _symbolic_dims(rank: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(rank)]


def _build_tile_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """rmo.MoTileOp with repeats as a runtime [rank] host operand."""
    assert rank is not None
    dev = DeviceRef.from_device(device)
    x = TensorType(dtype, _symbolic_dims(rank, "d"), device=dev)
    repeats = TensorType(DType.int64, [rank], device=DeviceRef.CPU())
    g = Graph(
        _graph_name(mo.TileOp, device, dtype, rank),
        input_types=[x, repeats],
        module=module,
    )
    with g:
        data, reps = (v.tensor for v in g.inputs)
        out_type = TensorType(dtype, _symbolic_dims(rank, "o"), device=dev)
        tiled = Graph.current._add_op_generated(
            rmo.MoTileOp, out_type, data, reps, kgen.ParamDeclArrayAttr([])
        )[0].tensor
        g.output(tiled)


_register_spec(
    mo.TileOp,
    RearrangeSpec(
        build_graph=_build_tile_graph,
        devices=DeviceClass.CPU_ONLY,
        rank_keyed=True,
        max_rank=4,  # GC tile kernel supports up to rank 4.
    ),
)


def _build_pad_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
    *,
    mo_cls: type,
    rmo_cls: type,
    has_constant: bool,
) -> None:
    """Emit rmo.MoPad*Op with runtime paddings [2*rank] (+ rank-0 constant)."""
    assert rank is not None
    dev = DeviceRef.from_device(device)
    input_types = [
        TensorType(dtype, _symbolic_dims(rank, "d"), device=dev),
        TensorType(DType.int64, [2 * rank], device=DeviceRef.CPU()),
    ]
    if has_constant:
        input_types.append(TensorType(dtype, [], device=DeviceRef.CPU()))
    g = Graph(
        _graph_name(mo_cls, device, dtype, rank),
        input_types=input_types,
        module=module,
    )
    with g:
        ins = [v.tensor for v in g.inputs]
        out_type = TensorType(dtype, _symbolic_dims(rank, "o"), device=dev)
        kwargs = dict(
            result=out_type,
            input=ins[0],
            paddings=ins[1],
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )
        if has_constant:
            kwargs["constant"] = ins[2]
        g.output(Graph.current._add_op_generated(rmo_cls, **kwargs)[0].tensor)


def _pad_spec(
    mo_cls: type[_core.Operation],
    rmo_cls: type,
    has_constant: bool,
    devices: DeviceClass,
) -> RearrangeSpec:
    return RearrangeSpec(
        build_graph=partial(
            _build_pad_graph,
            mo_cls=mo_cls,
            rmo_cls=rmo_cls,
            has_constant=has_constant,
        ),
        devices=devices,
        rank_keyed=True,
    )


_register_spec(
    mo.PadConstantOp,
    _pad_spec(mo.PadConstantOp, rmo.MoPadConstantOp, True, DeviceClass.ALL),
)
_register_spec(
    mo.PadReflectOp,
    _pad_spec(mo.PadReflectOp, rmo.MoPadReflectOp, False, DeviceClass.CPU_ONLY),
)
_register_spec(
    mo.PadRepeatOp,
    _pad_spec(mo.PadRepeatOp, rmo.MoPadRepeatOp, False, DeviceClass.CPU_ONLY),
)


def _build_slice_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """rmo.MoSliceOp with runtime starts/stops/steps [rank]."""
    assert rank is not None
    dev = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    g = Graph(
        _graph_name(mo.SliceOp, device, dtype, rank),
        input_types=[
            TensorType(dtype, _symbolic_dims(rank, "d"), device=dev),
            TensorType(DType.int64, [rank], device=cpu),
            TensorType(DType.int64, [rank], device=cpu),
            TensorType(DType.int64, [rank], device=cpu),
        ],
        module=module,
    )
    with g:
        data, starts, stops, steps = (v.tensor for v in g.inputs)
        out_type = TensorType(dtype, _symbolic_dims(rank, "o"), device=dev)
        sliced = Graph.current._add_op_generated(
            rmo.MoSliceOp,
            result=out_type,
            input=data,
            start=starts,
            stop=stops,
            step=steps,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        g.output(sliced)


_register_spec(
    mo.SliceOp,
    RearrangeSpec(
        build_graph=_build_slice_graph,
        devices=DeviceClass.ALL,
        rank_keyed=True,
    ),
)


# Must stay last: every _register_spec call above feeds this snapshot.
_refresh_by_name()
SHAPE_REARRANGE_GC_OPS = tuple(_REARRANGE_OPS)
