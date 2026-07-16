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
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

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
_swept = False


def _sweep_devices() -> list[Device]:
    """Every device the matmul sweep spans: CPU + all accelerator slots."""
    return list(_CPU_DEVICES) + list(_ACCELERATOR_DEVICES)


def build_matmul_module() -> Module:
    """Build the full batched matmul module: every ``_COMPILATION_TARGETS`` slot
    (CPU + all accelerators, all dtypes) in one module.

    Host-ELF and cubins both embed self-contained in the exported MEF, so one
    force-load populates every device class at once. Shared by the warm producer
    (export) and :func:`compile_matmul_sweep` (compile into cache).
    """
    module = Module()
    for compilation_target in _COMPILATION_TARGETS:
        _build_matmul_graph(module, compilation_target)
    return module


def build_matmul_module_for_device(device: Device) -> Module:
    """Build the matmul module for a single device slot: every dtype target on
    *device* (matched by label + id), and nothing else.

    Per-slot counterpart of :func:`build_matmul_module`. The warm producer
    exports one MEF per slot so the warm is device-count-independent: a k-GPU
    consumer force-loads only slots ``0..k-1``, letting a warm made for a higher
    count still adopt.
    """
    module = Module()
    for compilation_target in _COMPILATION_TARGETS:
        if (
            compilation_target.device.label == device.label
            and compilation_target.device.id == device.id
        ):
            _build_matmul_graph(module, compilation_target)
    return module


@in_default_mlir_context
def compile_matmul_sweep() -> None:
    """Warm the in-process matmul cache: force-load an adoptable manifest if one
    is present, else compile every supported (device, dtype) target in one
    batched ``load_all`` (parallel compile).

    The manifest-first check is what makes the import-time precompile
    (``_precompile_gc_models``) manifest-aware, precompile and lazy dispatch
    funnel through the one :func:`_adopt_matmul_manifest_if_adoptable`. Absent an
    adoptable manifest this is the batched compile used three ways, all the same
    call: the precompile (``=1``); the ``warm-interpreter-cache`` CLI; and lazy
    dispatch adopting a warm stamp (the identical batched module hashes to the
    warm on-disk cache key, so ``load_all`` loads rather than recompiles).
    """
    global _swept
    if _adopt_matmul_manifest_if_adoptable():
        return
    session = engine.InferenceSession(devices=_sweep_devices())
    _MATMUL_MODEL_CACHE.update(
        session.load_all(build_matmul_module(), weights_registry={})
    )
    _swept = True


def _adopt_matmul_from_manifest(manifest: dict[str, Any]) -> bool:
    """Force-load the per-slot matmul MEFs named in the manifest.

    Returns True iff every needed slot was force-loaded (models added to the
    cache). Force-load (session.load_all by path) bypasses the compile + cache
    key, so it adopts across a toolchain-ID mismatch the key-based path cannot.
    The manifest holds one single-device MEF per slot; a k-GPU box loads the CPU
    entry plus ``gpu:0..gpu:k-1`` into one multi-device session, which binds each
    slot's graphs to the matching real device by embedded id, so a warm made
    for more slots than this box has still adopts (extra slots go unloaded).
    """
    device_classes = ["cpu"] + [f"gpu:{i}" for i in range(accelerator_count())]
    try:
        session = engine.InferenceSession(devices=_sweep_devices())
        loaded: dict[str, engine.Model] = {}
        names: list[str] = []
        for device_class in device_classes:
            mef = gc_compile.manifest_entry_path(
                manifest, "matmul", device_class
            )
            if mef is None or not mef.exists():
                return False
            loaded.update(session.load_all(str(mef), weights_registry={}))
            names.append(mef.name)
    except Exception:
        logger.warning(
            "Eager warm manifest force-load failed; compiling on demand.",
            exc_info=True,
        )
        return False
    _MATMUL_MODEL_CACHE.update(loaded)
    gc_compile.note_manifest_adoption("matmul")
    # Force-load bypasses the keyed cache lookup, so [modular-cache] logging is
    # silent; the stderr marker + gc_compile.adopted_from_manifest signal it ran.
    print(
        f"[eager-warm] manifest force-load: matmul {names} ({len(loaded)} models)",
        file=sys.stderr,
        flush=True,
    )
    return True


def _adopt_matmul_manifest_if_adoptable() -> bool:
    """Force-load the matmul manifest if one is present and adoptable here.

    The single adoption check both the lazy dispatch and the import-time
    precompile (via :func:`compile_matmul_sweep`) funnel through, so the two
    paths can't diverge. Marks ``_swept`` once the manifest is adoptable, so a
    later dispatch won't re-attempt an adoptable-but-failing force-load; returns
    True only when the force-load fully succeeded. On failure the triggering
    dispatch degrades to a *batched* recompile when a warm stamp is also present
    (the composed warm dir carries both, and :func:`compile_matmul_sweep`
    re-checks this manifest once more before its batched ``load_all``), or to
    per-target compilation when only a manifest was present.
    """
    global _swept
    manifest = gc_compile.read_manifest()
    if manifest is None or not gc_compile.manifest_adoptable(manifest):
        return False
    _swept = True
    return _adopt_matmul_from_manifest(manifest)


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
    precompiled at import and this is a lookup. On the first miss an available
    warm cache is adopted whole instead of compiling each target singly --
    force-loaded from a manifest when one is present and adoptable, else via a
    batched sweep of a matching ``warm-interpreter-cache`` stamp.

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
        global _swept
        if not _swept:
            # Adopt a warm cache whole rather than compile per target: force-load
            # an adoptable manifest first (bypasses the toolchain key; the same
            # check the precompile path uses via compile_matmul_sweep), else a
            # matching warm stamp's batched sweep.
            if _adopt_matmul_manifest_if_adoptable():
                pass
            elif gc_compile.warm_stamp_matches():
                # Key-based batched adoption (unchanged fallback).
                _swept = True
                try:
                    compile_matmul_sweep()
                except Exception:
                    logger.warning(
                        "Eager interpreter warm-cache adoption failed;"
                        " compiling matmul targets on demand.",
                        exc_info=True,
                    )
            model = _MATMUL_MODEL_CACHE.get(target.graph_name)
            if model is not None:
                return model
        return _compile_matmul_target(target)
