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

"""Graph-compiler unary-elementwise model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-1 graph.
- **Precompile sweep (``=1``).** :func:`compile_unary_sweep` compiles the full
  matrix at import; a :func:`unary_model` miss is then a hard error.

Lazy mode avoids a trivial program JIT-compiling the whole kernel library on a
cold cache (~3000+ kernels, minutes; MXF-508). Models serve the eager handler
via :func:`unary_model`. Must not import from ``handlers.py``.

The swept dtype set is deliberately conservative (floats-first): the IR type
category is only a ceiling, so transcendental/activation ops are swept on float
dtypes only, ``Abs``/``Negative`` additionally get integer dtypes, and ``Not``
gets ``bool``. CPU floats are f32/f64 (no 16-bit); GPU floats are f16/f32/bf16
(no f64). ``dtype_class`` keys the *input*; ``IsNan``/``IsInf`` take a float
input and emit a constant ``bool``.
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
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops

logger = logging.getLogger(__name__)

# Float dtypes diverge by device (only f32 is shared). CPU: f32 + f64 (the
# 16-bit float kernels don't compile on CPU). GPU: f16/f32/bf16 (no f64 —
# NVIDIA rejects it for approx ops, Metal lacks it; f64-on-GPU tracked in
# https://linear.app/modularml/issue/MSTDL-2711).
_CPU_FLOAT_DTYPES = [DType.float32, DType.float64]
_GPU_FLOAT_DTYPES = [DType.float16, DType.float32, DType.bfloat16]
_SIGNED_INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
_UNSIGNED_INT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]


# Builds an op's graph body from its input tensor (e.g. ``ops.sqrt``).
MoOpBuilder: TypeAlias = Callable[[TensorValue], TensorValue]


class DTypeClass(Enum):
    """The input-dtype set an op is swept over (see ``_supported_dtypes``)."""

    FLOAT = "float"
    ABS = "abs"
    NEGATIVE = "negative"
    BOOL = "bool"


@dataclass(frozen=True)
class UnarySpec:
    """How to build one unary op's graph and which dtype class it sweeps."""

    builder: MoOpBuilder
    dtype_class: DTypeClass


def _gelu_none(x: TensorValue) -> TensorValue:
    return ops.gelu(x, approximate="none")


def _gelu_tanh(x: TensorValue) -> TensorValue:
    return ops.gelu(x, approximate="tanh")


def _gelu_quick(x: TensorValue) -> TensorValue:
    return ops.gelu(x, approximate="quick")


# The builder is a callable (an op or a named helper), so the Gelu variants and
# the plain one-op wrappers share one registry shape.
_UNARY_OPS: dict[type[_core.Operation], UnarySpec] = {
    mo.NegativeOp: UnarySpec(ops.negate, DTypeClass.NEGATIVE),
    mo.AbsOp: UnarySpec(ops.abs, DTypeClass.ABS),
    mo.CeilOp: UnarySpec(ops.ceil, DTypeClass.FLOAT),
    mo.FloorOp: UnarySpec(ops.floor, DTypeClass.FLOAT),
    mo.RoundOp: UnarySpec(ops.round, DTypeClass.FLOAT),
    mo.ExpOp: UnarySpec(ops.exp, DTypeClass.FLOAT),
    mo.LogOp: UnarySpec(ops.log, DTypeClass.FLOAT),
    mo.Log1pOp: UnarySpec(ops.log1p, DTypeClass.FLOAT),
    mo.SqrtOp: UnarySpec(ops.sqrt, DTypeClass.FLOAT),
    mo.RsqrtOp: UnarySpec(ops.rsqrt, DTypeClass.FLOAT),
    mo.TanhOp: UnarySpec(ops.tanh, DTypeClass.FLOAT),
    mo.AtanhOp: UnarySpec(ops.atanh, DTypeClass.FLOAT),
    mo.TruncOp: UnarySpec(ops.trunc, DTypeClass.FLOAT),
    mo.SinOp: UnarySpec(ops.sin, DTypeClass.FLOAT),
    mo.CosOp: UnarySpec(ops.cos, DTypeClass.FLOAT),
    mo.ErfOp: UnarySpec(ops.erf, DTypeClass.FLOAT),
    mo.SigmoidOp: UnarySpec(ops.sigmoid, DTypeClass.FLOAT),
    mo.SiluOp: UnarySpec(ops.silu, DTypeClass.FLOAT),
    mo.GeluOp: UnarySpec(_gelu_none, DTypeClass.FLOAT),
    mo.GeluTanhOp: UnarySpec(_gelu_tanh, DTypeClass.FLOAT),
    mo.GeluQuickOp: UnarySpec(_gelu_quick, DTypeClass.FLOAT),
    mo.NotOp: UnarySpec(ops.logical_not, DTypeClass.BOOL),
    # Predicates: float input, constant bool output; dtype_class keys input.
    mo.IsNanOp: UnarySpec(ops.is_nan, DTypeClass.FLOAT),
    mo.IsInfOp: UnarySpec(ops.is_inf, DTypeClass.FLOAT),
}

UNARY_GC_OPS = tuple(_UNARY_OPS)

# These lower to libm calls the GC backend only supports on CPU ("libm
# operations are only available on CPU targets") — verified failing on both
# Metal and CUDA (B200). Swept on CPU only, matching the historical interpreter
# binding's GPU allowlist, which excluded exactly these four.
_CPU_ONLY_OPS = frozenset({mo.Log1pOp, mo.AtanhOp, mo.ErfOp, mo.GeluOp})

_UNARY_MODEL_CACHE: dict[str, engine.Model] = {}


def _float_dtypes(device: Device) -> list[DType]:
    return _CPU_FLOAT_DTYPES if device.label == "cpu" else _GPU_FLOAT_DTYPES


def _supported_dtypes(dtype_class: DTypeClass, device: Device) -> list[DType]:
    """Conservative swept dtype set for a (dtype_class, device)."""
    if dtype_class is DTypeClass.FLOAT:
        return _float_dtypes(device)
    if dtype_class is DTypeClass.ABS:
        return _float_dtypes(device) + _SIGNED_INT_DTYPES + _UNSIGNED_INT_DTYPES
    if dtype_class is DTypeClass.NEGATIVE:
        return _float_dtypes(device) + _SIGNED_INT_DTYPES
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
    return f"unary_{op_type.__name__}_{device.label}_{device.id}_{dtype.name}"


def canonical_shape(shape: Sequence[int]) -> tuple[int]:
    """Flattens to rank 1; bare ``prod`` keeps scalars at 1 and empty at 0."""
    return (prod(shape),)


def _build_unary_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: UnarySpec,
    device: Device,
    dtype: DType,
) -> None:
    """Adds one fully-symbolic rank-1 unary graph into *module* in-place."""
    dev_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, ["n"], device=dev_ref)
    g = Graph(
        _graph_name(op_type, device, dtype),
        input_types=[in_type],
        module=module,
    )
    with g:
        (x,) = g.inputs
        g.output(spec.builder(x.tensor))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    """Whether (op, device, dtype) is in the conservatively-supported set.

    Single source of truth for the swept matrix: :func:`compile_unary_sweep`
    filters its candidates through this predicate, and lazy mode uses it as the
    support guard in :func:`unary_model`, so the two can't diverge. CPU-only ops
    are unsupported on accelerators, and each op supports only its
    ``dtype_class``'s dtypes.
    """
    spec = _UNARY_OPS.get(op_type)
    if spec is None:
        return False
    if device.label != "cpu" and op_type in _CPU_ONLY_OPS:
        return False
    return dtype in _supported_dtypes(spec.dtype_class, device)


# True once a batched sweep has run, so dispatch attempts adoption at most once.
_swept = False


@in_default_mlir_context
def compile_unary_sweep() -> None:
    """Compile every supported (op, device, dtype) unary target in one batched
    ``load_all`` (parallel compile), warming the in-process cache.

    Used three ways, all the same call: the import-time precompile (``=1``);
    the ``warm-interpreter-cache`` CLI; and lazy dispatch *adopting* a warm
    stamp (where the identical batched module hits the warm on-disk cache, so
    ``load_all`` loads rather than recompiles).

    Candidates are filtered through :func:`_is_supported` so an unsupported
    target never reaches the backend; a derived supported set is the real fix
    (MXF-477).
    """
    global _swept
    module = Module()
    for op_type, spec in _UNARY_OPS.items():
        for device in _DEVICES:
            for dtype in _supported_dtypes(spec.dtype_class, device):
                if _is_supported(op_type, device, dtype):
                    _build_unary_graph(module, op_type, spec, device, dtype)
    session = engine.InferenceSession(devices=list(_DEVICES))
    _UNARY_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    _swept = True


@in_default_mlir_context
def _compile_unary_target(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Build and compile a single (op, device, dtype) unary graph."""
    module = Module()
    _build_unary_graph(module, op_type, _UNARY_OPS[op_type], device, dtype)
    session = gc_compile.session_for(device)
    _UNARY_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    return _UNARY_MODEL_CACHE[_graph_name(op_type, device, dtype)]


def unary_model(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Returns the unary :class:`~max.engine.Model` for *op_type* / *device* / *dtype*.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. If a ``warm-interpreter-cache`` stamp is present for this context,
    the first miss adopts the warm with one batched sweep (a cache load) instead
    of compiling each target singly.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled.
        device: The realized input's device.
        dtype: The realized input's dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set (e.g.
            a transcendental op on an int dtype); or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target was not swept.
    """
    key = _graph_name(op_type, device, dtype)
    model = _UNARY_MODEL_CACHE.get(key)
    if model is not None:
        return model
    if not _is_supported(op_type, device, dtype):
        raise KeyError(
            f"Unsupported unary op/device/dtype for key {key!r}."
            "  Supported dtypes for this op/device: "
            f"{_supported_dtypes(_UNARY_OPS[op_type].dtype_class, device) if op_type in _UNARY_OPS else '[]'}"
        )
    if gc_compile.should_precompile():
        # TODO(MXF-510): raise UnsupportedGraphError so executors fall back.
        raise KeyError(
            f"No pre-compiled unary model for key {key!r}."
            f"  Available: {sorted(_UNARY_MODEL_CACHE)}."
            f"  Unset {gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR} (the default)"
            " to compile targets lazily on first use."
        )
    with gc_compile.COMPILE_LOCK:
        # Re-check under the lock (another thread may have compiled it).
        model = _UNARY_MODEL_CACHE.get(key)
        if model is not None:
            return model
        global _swept
        if not _swept and gc_compile.warm_stamp_matches():
            # Mark _swept before attempting so a stale stamp can't loop; guard
            # so an adoption failure falls through to per-target, not the op.
            _swept = True
            try:
                compile_unary_sweep()
            except Exception:
                logger.warning(
                    "Eager interpreter warm-cache adoption failed; compiling"
                    " unary targets on demand.",
                    exc_info=True,
                )
            model = _UNARY_MODEL_CACHE.get(key)
            if model is not None:
                return model
        return _compile_unary_target(op_type, device, dtype)
