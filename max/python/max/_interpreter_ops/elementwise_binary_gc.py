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

"""Graph-compiler binary-elementwise model cache for the MO interpreter.

Covers both arithmetic/bitwise binary ops (``Add``, ``Sub``, ``Mul``, ``Div``,
``Mod``, ``Max``, ``Min``, ``And``, ``Or``, ``Xor``, ``Pow``) and the comparison
predicates (``Equal``, ``Greater``, ``GreaterEqual``, ``NotEqual``). A comparison
graph emits ``bool``; the handler reads the realized output dtype, so the two
families share one builder and cache.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-1 graph.
- **Precompile sweep (``=1``).** :func:`compile_binary_sweep` compiles the full
  matrix at import; a :func:`binary_model` miss is then a hard error.

Lazy mode avoids a trivial program JIT-compiling the whole kernel library on a
cold cache (~3000+ kernels, minutes; MXF-508). Models serve the eager handler
via :func:`binary_model`. Must not import from ``handlers.py``.

Both operands reach the handler already cast to a common dtype and broadcast to
a common shape (the RMO->MO lowering inserts the ``BroadcastToOp`` chain), so a
graph with one dtype shared by both rank-1 inputs is exact.

The swept dtype set is deliberately conservative (the IR type category is only a
ceiling): general arithmetic (``Add``/``Sub``/``Mul``/``Max``/``Min``/``Mod``),
``Pow``, and the comparisons sweep floats + integers; ``Div`` sweeps floats only;
the logical ops (``And``/``Or``/``Xor``) sweep ``bool``. CPU floats are f32/f64
(no 16-bit); GPU floats are f16/f32/bf16 (no f64). ``dtype_class`` keys the
*input* dtype.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import TypeAlias

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
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue
from max.graph.ops import elementwise

logger = logging.getLogger(__name__)

# Float dtypes diverge by device (only f32 is shared), matching
# unary_elementwise_gc. CPU: f32 + f64 (the 16-bit float kernels don't compile
# on CPU). GPU: f16/f32/bf16 (no f64 — NVIDIA rejects it for some ops, Metal
# lacks it; f64-on-GPU tracked in MSTDL-2711).
_CPU_FLOAT_DTYPES = [DType.float32, DType.float64]
_GPU_FLOAT_DTYPES = [DType.float16, DType.float32, DType.bfloat16]
_SIGNED_INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
_UNSIGNED_INT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]


# Builds an op's graph body from its two input tensors (e.g. ``elementwise.add``).
MoBinaryOpBuilder: TypeAlias = Callable[[TensorValue, TensorValue], TensorValue]


class DTypeClass(Enum):
    """The input-dtype set an op is swept over (see ``_supported_dtypes``)."""

    NUMERIC = "numeric"
    FLOAT = "float"
    BOOL = "bool"


@dataclass(frozen=True)
class BinarySpec:
    """How to build one binary op's graph and which dtype class it sweeps."""

    builder: MoBinaryOpBuilder
    dtype_class: DTypeClass


# The builder is a two-arg elementwise op; comparison builders emit a bool
# tensor, which the handler picks up from the realized output dtype.
_BINARY_OPS: dict[type[_core.Operation], BinarySpec] = {
    mo.AddOp: BinarySpec(elementwise.add, DTypeClass.NUMERIC),
    mo.SubOp: BinarySpec(elementwise.sub, DTypeClass.NUMERIC),
    mo.MulOp: BinarySpec(elementwise.mul, DTypeClass.NUMERIC),
    mo.MaxOp: BinarySpec(elementwise.max, DTypeClass.NUMERIC),
    mo.MinOp: BinarySpec(elementwise.min, DTypeClass.NUMERIC),
    mo.ModOp: BinarySpec(elementwise.mod, DTypeClass.NUMERIC),
    # div promotes int operands to f64 in the lowering, so an int div never
    # reaches the handler; pow has no such promotion, so it must sweep ints.
    mo.DivOp: BinarySpec(elementwise.div, DTypeClass.FLOAT),
    mo.PowOp: BinarySpec(elementwise.pow, DTypeClass.NUMERIC),
    mo.AndOp: BinarySpec(elementwise.logical_and, DTypeClass.BOOL),
    mo.OrOp: BinarySpec(elementwise.logical_or, DTypeClass.BOOL),
    mo.XorOp: BinarySpec(elementwise.logical_xor, DTypeClass.BOOL),
    # Comparison predicates: numeric input, bool output.
    mo.EqualOp: BinarySpec(elementwise.equal, DTypeClass.NUMERIC),
    mo.GreaterOp: BinarySpec(elementwise.greater, DTypeClass.NUMERIC),
    mo.GreaterEqualOp: BinarySpec(
        elementwise.greater_equal, DTypeClass.NUMERIC
    ),
    mo.NotEqualOp: BinarySpec(elementwise.not_equal, DTypeClass.NUMERIC),
}

BINARY_GC_OPS = tuple(_BINARY_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_BINARY_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _BINARY_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> BinarySpec | None:
    name = gc_compile.canonical_op_name(op_type, _BINARY_OPS_BY_NAME)
    return _BINARY_OPS_BY_NAME.get(name)


_BINARY_MODEL_CACHE: dict[str, engine.Model] = {}


def _float_dtypes(device: Device) -> list[DType]:
    return _CPU_FLOAT_DTYPES if device.label == "cpu" else _GPU_FLOAT_DTYPES


def _supported_dtypes(dtype_class: DTypeClass, device: Device) -> list[DType]:
    """Conservative swept dtype set for a (dtype_class, device)."""
    if dtype_class is DTypeClass.FLOAT:
        return _float_dtypes(device)
    if dtype_class is DTypeClass.NUMERIC:
        return _float_dtypes(device) + _SIGNED_INT_DTYPES + _UNSIGNED_INT_DTYPES
    if dtype_class is DTypeClass.BOOL:
        return [DType.bool]
    raise ValueError(f"Unknown dtype_class: {dtype_class!r}")


# Discovered at import so a missing driver fails here, not at first dispatch.
_DEVICES = load_devices([DeviceSpec.cpu()]) + load_devices(
    [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
)


def _graph_name(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype)."""
    name = gc_compile.canonical_op_name(op_type, _BINARY_OPS_BY_NAME)
    return f"binary_{name}_{device.label}_{device.id}_{dtype.name}"


def canonical_shape(shape: Sequence[int]) -> tuple[int]:
    """Flattens to rank 1; bare ``prod`` keeps scalars at 1 and empty at 0."""
    return (prod(shape),)


def _build_binary_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: BinarySpec,
    device: Device,
    dtype: DType,
) -> None:
    """Adds one fully-symbolic rank-1 binary graph into *module* in-place."""
    dev_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, ["n"], device=dev_ref)
    g = Graph(
        _graph_name(op_type, device, dtype),
        input_types=[in_type, in_type],
        module=module,
    )
    with g:
        lhs, rhs = g.inputs
        g.output(spec.builder(lhs.tensor, rhs.tensor))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    """Whether (op, device, dtype) is in the conservatively-supported set.

    Single source of truth for the swept matrix: :func:`compile_binary_sweep`
    filters its candidates through this predicate, and lazy mode uses it as the
    support guard in :func:`binary_model`, so the two can't diverge. Each op
    supports only its ``dtype_class``'s dtypes.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    return dtype in _supported_dtypes(spec.dtype_class, device)


# True once a batched sweep has run, so dispatch attempts adoption at most once.
_SWEPT = False


@in_default_mlir_context
def compile_binary_sweep() -> None:
    """Compile every supported (op, device, dtype) binary target in one batched
    ``load_all`` (parallel compile), warming the in-process cache.

    Used three ways, all the same call: the import-time precompile (``=1``);
    the ``warm-interpreter-cache`` CLI; and lazy dispatch *adopting* a warm
    stamp (where the identical batched module hits the warm on-disk cache, so
    ``load_all`` loads rather than recompiles).

    Candidates are filtered through :func:`_is_supported` so an unsupported
    target never reaches the backend.
    """
    global _SWEPT
    module = Module()
    for op_type, spec in _BINARY_OPS.items():
        for device in _DEVICES:
            for dtype in _supported_dtypes(spec.dtype_class, device):
                if _is_supported(op_type, device, dtype):
                    _build_binary_graph(module, op_type, spec, device, dtype)
    session = engine.InferenceSession(devices=list(_DEVICES))
    _BINARY_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    _SWEPT = True


@in_default_mlir_context
def _compile_binary_target(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Build and compile a single (op, device, dtype) binary graph."""
    module = Module()
    spec = _spec_for(op_type)
    assert spec is not None, f"unsupported op {op_type!r} reached compile"
    _build_binary_graph(module, op_type, spec, device, dtype)
    session = gc_compile.session_for(device)
    _BINARY_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    return _BINARY_MODEL_CACHE[_graph_name(op_type, device, dtype)]


def binary_model(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Returns the binary :class:`~max.engine.Model` for *op_type* / *device* / *dtype*.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. If a ``warm-interpreter-cache`` stamp is present for this context,
    the first miss adopts the warm with one batched sweep (a cache load) instead
    of compiling each target singly.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled.
        device: The realized inputs' device.
        dtype: The realized inputs' (shared) dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set (e.g.
            ``Div`` on an int dtype); or, with ``MAX_EAGER_OP_PRECOMPILE=1``, if
            a supported target was not swept.
    """
    key = _graph_name(op_type, device, dtype)
    model = _BINARY_MODEL_CACHE.get(key)
    if model is not None:
        return model
    if not _is_supported(op_type, device, dtype):
        spec = _spec_for(op_type)
        supported = _supported_dtypes(spec.dtype_class, device) if spec else []
        raise KeyError(
            f"Unsupported binary op/device/dtype for key {key!r}."
            f"  Supported dtypes for this op/device: {supported}"
        )
    if gc_compile.should_precompile():
        # TODO(MXF-510): raise UnsupportedGraphError so executors fall back.
        raise KeyError(
            f"No pre-compiled binary model for key {key!r}."
            f"  Available: {sorted(_BINARY_MODEL_CACHE)}."
            f"  Unset {gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR} (the default)"
            " to compile targets lazily on first use."
        )
    with gc_compile.COMPILE_LOCK:
        # Re-check under the lock (another thread may have compiled it).
        model = _BINARY_MODEL_CACHE.get(key)
        if model is not None:
            return model
        global _SWEPT
        if not _SWEPT and gc_compile.warm_stamp_matches():
            # Mark _SWEPT before attempting so a stale stamp can't loop; guard
            # so an adoption failure falls through to per-target, not the op.
            _SWEPT = True
            try:
                compile_binary_sweep()
            except Exception:
                logger.warning(
                    "Eager interpreter warm-cache adoption failed; compiling"
                    " binary targets on demand.",
                    exc_info=True,
                )
            model = _BINARY_MODEL_CACHE.get(key)
            if model is not None:
                return model
        return _compile_binary_target(op_type, device, dtype)
