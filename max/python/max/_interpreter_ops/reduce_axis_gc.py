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

"""Graph-compiler reduce-along-axis model cache for the MO interpreter.

Covers the reduce-along-axis family: the reductions (``ReduceMax``/``Min``/
``Add``/``Mul``/``Mean``), ``Softmax``/``Logsoftmax``, ``ArgMax``/``ArgMin``
(emit ``int64`` indices), and ``Cumsum``.

Every op applies one operation along a single ``axis``, which is a compile-time
attribute, not an operand. To keep the cache key ``(op, device, dtype)`` without
``axis`` in it, the handler canonicalizes any input to rank 3
``[outer, axis, inner]`` (a zero-copy view; see :func:`canonical_rank3`) and each
graph applies the op at ``axis=1``. The handler reads the op's MLIR result type
for the final shape and dtype, so reduced-axis ops (``[d0, 1, d2]``, ``int64``
for argmax) and same-shape ops (``[d0, d1, d2]``) share one handler.

``Cumsum`` additionally carries compile-time ``exclusive``/``reverse`` flags.
Decomposing them host-side would force device<->host copies on GPU buffers (and
baking them into the graph just produces variants anyway), so the four
``(exclusive, reverse)`` combinations are baked into the cache key as separate
graph variants. ``axis`` is still canonicalized out.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-3 graph.
- **Precompile sweep (``=1``).** :func:`compile_reduce_axis_sweep` compiles the
  full matrix at import; a :func:`reduce_model` miss is then a hard error.

Models serve the eager handler via :func:`reduce_model`. Must not import from
``handlers.py``.

The swept dtype set is deliberately conservative (the IR type category is only a
ceiling): softmax/logsoftmax sweep floats only; the reductions, argmax/argmin,
and cumsum sweep floats + ints; ``ReduceMax``/``ReduceMin`` additionally sweep
``bool`` (the logical-OR/AND reductions, issue #6067); cumsum excludes ``bool``.
CPU floats are f32/f64 (no 16-bit); GPU floats are f16/f32/bf16 (no f64).
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import TypeAlias, cast

from max import _core, engine
from max._core.dialects import mo
from max._interpreter_ops import gc_compile
from max._mlir_context import in_default_mlir_context
from max.driver import (
    Device,
    DeviceSpec,
    accelerator_count,
    load_devices,
)
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops

logger = logging.getLogger(__name__)

# Float dtypes diverge by device (only f32 is shared). CPU: f32 + f64 (the
# 16-bit float kernels don't compile on CPU). GPU: f16/f32/bf16 (no f64 —
# NVIDIA rejects it for some ops, Metal lacks it; f64-on-GPU tracked in
# MSTDL-2711).
_CPU_FLOAT_DTYPES = [DType.float32, DType.float64]
_GPU_FLOAT_DTYPES = [DType.float16, DType.float32, DType.bfloat16]
_SIGNED_INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
_UNSIGNED_INT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]

# CUDA's reduction kernels (the reduce/argmax family) only support 32- and
# 64-bit integer reduction: 8/16-bit int reduce fails to compile on the B200
# backend ("Failed to compile the model"). So on accelerators the reduction
# family is narrowed to the wide ints. Cumsum is exempt — on GPU it transfers to
# CPU (KERN-1095), so it keeps the full int set; CPU supports every width.
_WIDE_INT_DTYPES = [DType.int32, DType.int64, DType.uint32, DType.uint64]

# Cumsum's (exclusive, reverse) flags ride in the cache key as a variant; every
# other op uses the empty variant.
Variant: TypeAlias = tuple[bool, ...]
_NO_VARIANT: Variant = ()
_CUMSUM_VARIANTS: tuple[Variant, ...] = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)

# Builds the rank-3 graph body; the op is applied at axis=1 of [d0, d1, d2].
ReduceBuilder: TypeAlias = Callable[[TensorValue, Variant], TensorValue]


class DTypeClass(Enum):
    """The input-dtype set an op is swept over (see ``_supported_dtypes``)."""

    FLOAT = "float"
    NUMERIC = "numeric"
    NUMERIC_BOOL = "numeric_bool"
    CUMSUM = "cumsum"


@dataclass(frozen=True)
class ReduceSpec:
    """How to build one op's rank-3 graph, its dtype class, and its variants."""

    build: ReduceBuilder
    dtype_class: DTypeClass
    variants: tuple[Variant, ...] = (_NO_VARIANT,)


def _b_max(x: TensorValue, v: Variant) -> TensorValue:
    return ops.max(x, axis=1)


def _b_min(x: TensorValue, v: Variant) -> TensorValue:
    return ops.min(x, axis=1)


def _b_sum(x: TensorValue, v: Variant) -> TensorValue:
    return ops.sum(x, axis=1)


def _b_prod(x: TensorValue, v: Variant) -> TensorValue:
    return ops.prod(x, axis=1)


def _b_mean(x: TensorValue, v: Variant) -> TensorValue:
    return ops.mean(x, axis=1)


# The backend only reduces the *innermost* axis for these four ("axis other than
# innermost/-1 not supported"), but the canonical form reduces axis=1, so they
# transpose d1 to innermost, reduce at axis=-1, and transpose back (folded into
# the graph). argmax/argmin break ties to the lowest index on CPU but an
# arbitrary index on GPU (the graph compiler's documented contract).


def _b_softmax(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.softmax(xt, axis=-1), 1, 2)


def _b_logsoftmax(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.logsoftmax(xt, axis=-1), 1, 2)


def _b_argmax(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.argmax(xt, axis=-1), 1, 2)


def _b_argmin(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.argmin(xt, axis=-1), 1, 2)


def _b_cumsum(x: TensorValue, v: Variant) -> TensorValue:
    return ops.cumsum(x, axis=1, exclusive=v[0], reverse=v[1])


_REDUCE_OPS: dict[type[_core.Operation], ReduceSpec] = {
    mo.ReduceMaxOp: ReduceSpec(_b_max, DTypeClass.NUMERIC_BOOL),
    mo.ReduceMinOp: ReduceSpec(_b_min, DTypeClass.NUMERIC_BOOL),
    mo.ReduceAddOp: ReduceSpec(_b_sum, DTypeClass.NUMERIC),
    mo.ReduceMulOp: ReduceSpec(_b_prod, DTypeClass.NUMERIC),
    mo.ReduceMeanOp: ReduceSpec(_b_mean, DTypeClass.NUMERIC),
    mo.ReduceSoftmaxOp: ReduceSpec(_b_softmax, DTypeClass.FLOAT),
    mo.ReduceLogsoftmaxOp: ReduceSpec(_b_logsoftmax, DTypeClass.FLOAT),
    mo.ReduceArgMaxOp: ReduceSpec(_b_argmax, DTypeClass.NUMERIC),
    mo.ReduceArgMinOp: ReduceSpec(_b_argmin, DTypeClass.NUMERIC),
    mo.CumsumOp: ReduceSpec(
        _b_cumsum, DTypeClass.CUMSUM, variants=_CUMSUM_VARIANTS
    ),
}

REDUCE_AXIS_GC_OPS = tuple(_REDUCE_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_REDUCE_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _REDUCE_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> ReduceSpec | None:
    name = gc_compile.canonical_op_name(op_type, _REDUCE_OPS_BY_NAME)
    return _REDUCE_OPS_BY_NAME.get(name)


_REDUCE_MODEL_CACHE: dict[str, engine.Model] = {}


def _float_dtypes(device: Device) -> list[DType]:
    return _CPU_FLOAT_DTYPES if device.label == "cpu" else _GPU_FLOAT_DTYPES


def _reduce_int_dtypes(device: Device) -> list[DType]:
    """Integer dtypes the reduction/argmax family supports on *device*.

    CPU handles every width; accelerators are narrowed to 32/64-bit (CUDA's
    reduce kernels don't compile 8/16-bit int reduction — see ``_WIDE_INT_DTYPES``).
    """
    if device.label == "cpu":
        return _SIGNED_INT_DTYPES + _UNSIGNED_INT_DTYPES
    return _WIDE_INT_DTYPES


def _supported_dtypes(dtype_class: DTypeClass, device: Device) -> list[DType]:
    """Conservative swept dtype set for a (dtype_class, device)."""
    if dtype_class is DTypeClass.FLOAT:
        return _float_dtypes(device)
    if dtype_class is DTypeClass.CUMSUM:
        # Cumsum runs on CPU even for a GPU graph (KERN-1095), so it keeps every
        # int width on all devices.
        return _float_dtypes(device) + _SIGNED_INT_DTYPES + _UNSIGNED_INT_DTYPES
    numeric = _float_dtypes(device) + _reduce_int_dtypes(device)
    if dtype_class is DTypeClass.NUMERIC:
        return numeric
    if dtype_class is DTypeClass.NUMERIC_BOOL:
        # bool max/min are the logical any/all reductions (#6067); swept on GPU
        # too, matching the bool-on-GPU binary logical ops (And/Or/Xor).
        return numeric + [DType.bool]
    raise ValueError(f"Unknown dtype_class: {dtype_class!r}")


# Discovered at import so a missing driver fails here, not at first dispatch.
_DEVICES = load_devices([DeviceSpec.cpu()]) + load_devices(
    [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
)


def _variant_tag(variant: Variant) -> str:
    """Cache-key suffix for a variant; empty for the no-variant default."""
    if not variant:
        return ""
    exclusive, reverse = variant
    return f"_e{int(exclusive)}r{int(reverse)}"


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant = _NO_VARIANT,
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype, variant)."""
    name = gc_compile.canonical_op_name(op_type, _REDUCE_OPS_BY_NAME)
    return (
        f"reduce_{name}_{device.label}_{device.id}_{dtype.name}"
        f"{_variant_tag(variant)}"
    )


def canonical_rank3(shape: Sequence[int], axis: int) -> tuple[int, int, int]:
    """Collapses *shape* to rank 3 ``[outer, axis, inner]`` for reduction at 1.

    ``axis`` is normalized for negatives. ``prod(())`` is 1, so a leading or
    trailing axis yields an outer/inner dim of 1.
    """
    ndim = len(shape)
    if axis < 0:
        axis += ndim
    return (prod(shape[:axis]), shape[axis], prod(shape[axis + 1 :]))


def variant_for(op: _core.Operation) -> Variant:
    """The cache-key variant for *op*: (exclusive, reverse) for cumsum, else ().

    Matches by canonical name (not ``isinstance``) so an ``rmo.MoCumsumOp``
    dispatch carries its variant too, not just ``mo.CumsumOp``.
    """
    if (
        gc_compile.canonical_op_name(type(op), _REDUCE_OPS_BY_NAME)
        == mo.CumsumOp.__name__
    ):
        cumsum = cast(mo.CumsumOp, op)
        return (bool(cumsum.exclusive), bool(cumsum.reverse))
    return _NO_VARIANT


def _build_reduce_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: ReduceSpec,
    device: Device,
    dtype: DType,
    variant: Variant,
) -> None:
    """Adds one fully-symbolic rank-3 reduce graph into *module* in-place."""
    dev_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, ["d0", "d1", "d2"], device=dev_ref)
    g = Graph(
        _graph_name(op_type, device, dtype, variant),
        input_types=[in_type],
        module=module,
    )
    with g:
        (x,) = g.inputs
        g.output(spec.build(x.tensor, variant))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    """Whether (op, device, dtype) is in the conservatively-supported set.

    Single source of truth for the swept matrix: :func:`compile_reduce_axis_sweep`
    filters candidates through this predicate and lazy mode uses it as the
    support guard in :func:`reduce_model`, so the two can't diverge. Variant does
    not affect dtype support, so it is not an argument here.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    return dtype in _supported_dtypes(spec.dtype_class, device)


# True once a batched sweep has run, so dispatch attempts adoption at most once.
_SWEPT = False


@in_default_mlir_context
def compile_reduce_axis_sweep() -> None:
    """Compile every supported (op, device, dtype, variant) target in one batched
    ``load_all`` (parallel compile), warming the in-process cache.

    Used three ways, all the same call: the import-time precompile (``=1``); the
    ``warm-interpreter-cache`` CLI; and lazy dispatch *adopting* a warm stamp.
    Candidates are filtered through :func:`_is_supported`.
    """
    global _SWEPT
    module = Module()
    for op_type, spec in _REDUCE_OPS.items():
        for device in _DEVICES:
            for dtype in _supported_dtypes(spec.dtype_class, device):
                if not _is_supported(op_type, device, dtype):
                    continue
                for variant in spec.variants:
                    _build_reduce_graph(
                        module, op_type, spec, device, dtype, variant
                    )
    session = engine.InferenceSession(devices=list(_DEVICES))
    _REDUCE_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    _SWEPT = True


@in_default_mlir_context
def _compile_reduce_axis_target(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant,
) -> engine.Model:
    """Build and compile a single (op, device, dtype, variant) reduce graph."""
    module = Module()
    spec = _spec_for(op_type)
    assert spec is not None, f"unsupported op {op_type!r} reached compile"
    _build_reduce_graph(module, op_type, spec, device, dtype, variant)
    session = gc_compile.session_for(device)
    _REDUCE_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    return _REDUCE_MODEL_CACHE[_graph_name(op_type, device, dtype, variant)]


def reduce_model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant = _NO_VARIANT,
) -> engine.Model:
    """Returns the reduce :class:`~max.engine.Model` for the given target.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. If a ``warm-interpreter-cache`` stamp is present for this context,
    the first miss adopts the warm with one batched sweep instead of compiling
    each target singly.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled.
        device: The realized input's device.
        dtype: The realized input's dtype.
        variant: The cumsum ``(exclusive, reverse)`` pair, or ``()`` otherwise.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set; or,
            with ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target was not
            swept.
    """
    key = _graph_name(op_type, device, dtype, variant)
    model = _REDUCE_MODEL_CACHE.get(key)
    if model is not None:
        return model
    if not _is_supported(op_type, device, dtype):
        spec = _spec_for(op_type)
        supported = _supported_dtypes(spec.dtype_class, device) if spec else []
        raise KeyError(
            f"Unsupported reduce op/device/dtype for key {key!r}."
            f"  Supported dtypes for this op/device: {supported}"
        )
    if gc_compile.should_precompile():
        # TODO(MXF-510): raise UnsupportedGraphError so executors fall back.
        raise KeyError(
            f"No pre-compiled reduce model for key {key!r}."
            f"  Available: {sorted(_REDUCE_MODEL_CACHE)}."
            f"  Unset {gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR} (the default)"
            " to compile targets lazily on first use."
        )
    with gc_compile.COMPILE_LOCK:
        # Re-check under the lock (another thread may have compiled it).
        model = _REDUCE_MODEL_CACHE.get(key)
        if model is not None:
            return model
        global _SWEPT
        if not _SWEPT and gc_compile.warm_stamp_matches():
            # Mark _SWEPT before attempting so a stale stamp can't loop; guard so
            # an adoption failure falls through to per-target, not the op.
            _SWEPT = True
            try:
                compile_reduce_axis_sweep()
            except Exception:
                logger.warning(
                    "Eager interpreter warm-cache adoption failed; compiling"
                    " reduce targets on demand.",
                    exc_info=True,
                )
            model = _REDUCE_MODEL_CACHE.get(key)
            if model is not None:
                return model
        return _compile_reduce_axis_target(op_type, device, dtype, variant)
