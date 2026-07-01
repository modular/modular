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

"""Graph-compiler matmul model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a (device, dtype) compiles
  just that target's fully-symbolic rank-3 batched-matmul graph.
- **Precompile sweep (``=1``).** :func:`compile_matmul_sweep` compiles the full
  matrix at import; a :func:`matmul_model` miss is then a hard error.

Lazy mode avoids a trivial matmul JIT-compiling the whole kernel library on a
cold cache (~3000+ kernels, minutes; MXF-508). Models serve the eager
``mo.matmul`` / ``mo.batch_matmul`` handler via :func:`matmul_model`. Must not
import from ``handlers.py``.
"""

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from max import engine
from max._interpreter_ops import gc_compile
from max._mlir_context import in_default_mlir_context
from max.driver import Device, DeviceSpec, accelerator_count, load_devices
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

logger = logging.getLogger(__name__)

_GRAPH_BASE_NAME = "matmul"

_ACCELERATOR_DEVICES = load_devices(
    [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
)

_ACCELERATOR_DTYPES = [DType.float32, DType.float16, DType.bfloat16]

_CPU_DEVICES = load_devices([DeviceSpec.cpu()])

# Conservative set proven to compile on every CI architecture. float16/bfloat16
# fail matmul kernel codegen on ARM, so widen only with per-arch CI confirmation.
_CPU_DTYPES = [
    DType.float32,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
]


@dataclass(frozen=True)
class CompilationTarget:
    graph_op_name: str
    device: Device
    # A single dtype shared by both operands. In principle lhs and rhs can
    # have different dtypes. In that case extend the dataclass
    dtype: DType

    @property
    def graph_name(self) -> str:
        """Returns the string used both as the graph ``sym_name`` and cache key."""
        return f"{self.graph_op_name}_{self.device.label}_{self.device.id}_{self.dtype}"


_COMPILATION_TARGETS = [
    CompilationTarget(_GRAPH_BASE_NAME, device, dtype)
    for device, dtype in itertools.chain(
        itertools.product(_CPU_DEVICES, _CPU_DTYPES),
        itertools.product(_ACCELERATOR_DEVICES, _ACCELERATOR_DTYPES),
    )
]

_MATMUL_MODEL_CACHE: dict[str, engine.Model] = {}


def canonical_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    """Flattens an arbitrary-rank matmul operand to canonical rank 3.

    ``[d0, ..., dn, i, j]`` becomes ``(d0*...*dn, i, j)``; a rank-2 ``[i, j]``
    becomes ``(1, i, j)`` because ``prod(())`` is the empty product ``1``,
    keeping the rank-2 case branchless.
    """
    *batch_dims, i, j = shape
    return (prod(batch_dims), i, j)


def _build_matmul_graph(
    module: Module, compilation_target: CompilationTarget
) -> None:
    """Adds one fully-symbolic rank-3 matmul graph into *module* in-place."""
    dev_ref = DeviceRef.from_device(compilation_target.device)
    lhs_type = TensorType(
        compilation_target.dtype, ["batch", "m", "k"], device=dev_ref
    )
    rhs_type = TensorType(
        compilation_target.dtype, ["batch", "k", "n"], device=dev_ref
    )
    graph_name = compilation_target.graph_name
    g = Graph(graph_name, input_types=[lhs_type, rhs_type], module=module)
    with g:
        lhs, rhs = g.inputs
        g.output(graph_ops.matmul(lhs.tensor, rhs.tensor))


# True once a batched sweep has run, so dispatch attempts adoption at most once.
_SWEPT = False


@in_default_mlir_context
def compile_matmul_sweep() -> None:
    """Compile every supported (device, dtype) matmul target in one batched
    ``load_all`` (parallel compile), warming the in-process cache.

    Used three ways, all the same call: the import-time precompile (``=1``);
    the ``warm-interpreter-cache`` CLI; and lazy dispatch *adopting* a warm
    stamp. In the adoption case the identical batched module hashes to the warm
    on-disk cache key, so ``load_all`` is a fast load rather than a recompile.
    """
    global _SWEPT
    module = Module()
    for compilation_target in _COMPILATION_TARGETS:
        _build_matmul_graph(module, compilation_target)
    devices = {ct.device for ct in _COMPILATION_TARGETS}
    session = engine.InferenceSession(devices=devices)
    _MATMUL_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    _SWEPT = True


@in_default_mlir_context
def _compile_matmul_target(target: CompilationTarget) -> engine.Model:
    """Build and compile a single (device, dtype) matmul graph."""
    module = Module()
    _build_matmul_graph(module, target)
    session = gc_compile.session_for(target.device)
    _MATMUL_MODEL_CACHE.update(session.load_all(module, weights_registry={}))
    return _MATMUL_MODEL_CACHE[target.graph_name]


def matmul_model(device: Device, dtype: DType) -> engine.Model:
    """Return the matmul :class:`~max.engine.Model` for *device* + *dtype*.

    Lazy by default: compiled on first use and cached in ``_MATMUL_MODEL_CACHE``
    for the process lifetime. With ``MAX_EAGER_OP_PRECOMPILE=1`` it was
    precompiled at import and this is a lookup. If a ``warm-interpreter-cache``
    stamp is present for this context, the first miss adopts the warm with one
    batched sweep (a cache load) instead of compiling each target singly.

    Args:
        device: The target device (CPU or GPU accelerator).
        dtype: The element dtype for both operands.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: With ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not in
            the import-time sweep.

    Note:
        No support guard (unlike :func:`unary_elementwise_gc.unary_model`):
        RMO->MO lowering casts both operands to a common dtype the backend can
        always compile a matmul for, so an unsupported target is unreachable.
    """
    target = CompilationTarget(_GRAPH_BASE_NAME, device, dtype)
    model = _MATMUL_MODEL_CACHE.get(target.graph_name)
    if model is not None:
        return model
    if gc_compile.should_precompile():
        # TODO(MXF-510): raise UnsupportedGraphError so executors fall back.
        raise KeyError(
            f"No pre-compiled matmul model for key {target.graph_name!r}."
            f"  Available: {sorted(_MATMUL_MODEL_CACHE)}."
            f"  Unset {gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR} (the default)"
            " to compile targets lazily on first use."
        )
    with gc_compile.COMPILE_LOCK:
        # Re-check under the lock (another thread may have compiled it).
        model = _MATMUL_MODEL_CACHE.get(target.graph_name)
        if model is not None:
            return model
        global _SWEPT
        if not _SWEPT and gc_compile.warm_stamp_matches():
            # Mark _SWEPT before attempting so a stale stamp can't loop; guard
            # so an adoption failure falls through to per-target, not the op.
            _SWEPT = True
            try:
                compile_matmul_sweep()
            except Exception:
                logger.warning(
                    "Eager interpreter warm-cache adoption failed; compiling"
                    " matmul targets on demand.",
                    exc_info=True,
                )
            model = _MATMUL_MODEL_CACHE.get(target.graph_name)
            if model is not None:
                return model
        return _compile_matmul_target(target)
