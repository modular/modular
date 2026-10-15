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

"""Shared compile-mode state for the eager interpreter's GC model caches.

The per-op-type GC caches (``matmul_gc``, ``unary_elementwise_gc``, and any
future one) share process-wide singletons defined here (or re-exported from
``_eager_policy``):

- :func:`should_precompile` — whether to compile the full GC matrix at import
  or compile each target lazily on first dispatch (the default).
- :data:`COMPILE_LOCK` — serializes the lazy check-compile-cache so concurrent
  first-dispatches don't race.
- :func:`session_for` — a per-device :class:`~max.engine.InferenceSession`
  cache, shared across op types so lazy compiles reuse one session per device.
- :func:`write_warm_stamp` / :func:`warm_stamp_matches` — marker letting a
  separate lazy process adopt a batched warm instead of compiling target-by-target.
- :func:`warm_store` — the build-time warm a test names with
  ``MAX_EAGER_WARM_STORE``, which every family then loads instead of
  compiling.

This module must not import from ``handlers.py``.
"""

import functools
import json
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Protocol, TypeVar

from max import _core, _eager_policy, engine
from max._core.engine import get_config_value
from max._mlir_context import in_default_mlir_context
from max.driver import (
    Device,
    DeviceSpec,
    accelerator_architecture_name,
    accelerator_count,
    load_devices,
)
from max.dtype import DType
from max.graph import Module

logger = logging.getLogger(__name__)


def canonical_op_name(
    op_type: type[_core.Operation], known_names: Container[str]
) -> str:
    """Canonical ``mo`` op-name for an mo- or rmo-dialect *op_type*.

    The ops library emits ``rmo`` ops; most share the ``mo`` name (``AddOp``)
    but some prefix ``Mo`` (``rmo.MoExpOp`` vs ``mo.ExpOp``). Strip a leading
    ``Mo`` when that yields a *known_names* entry, reusing the ``mo`` sweep's
    cache entry; unknown names pass through.
    """
    name = op_type.__name__
    if name in known_names:
        return name
    if name.startswith("Mo") and name[2:] in known_names:
        return name[2:]
    return name


# Stateless building blocks the eager GC op-family caches share (binary, unary,
# reduce-axis, shape-rearrange); hoisted here so a new family imports them.
CPU_FLOAT_DTYPES = [DType.float32, DType.float64]
GPU_FLOAT_DTYPES = [DType.float16, DType.float32, DType.bfloat16]
SIGNED_INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
UNSIGNED_INT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]

_UINT_FOR_SIZE = {
    1: DType.uint8,
    2: DType.uint16,
    4: DType.uint32,
    8: DType.uint64,
}

WIDTH_DTYPES = list(_UINT_FOR_SIZE.values())
"""The four uint widths a pure-copy family sweeps instead of real dtypes."""

MAX_RANK = 5
"""Historical interpreter rank cap, shared by every rank-capped family."""


def uint_view_dtype(dtype: DType) -> DType:
    """Returns the same-bit-width unsigned int a dtype is bit-cast to for copying.

    Pure-copy ops (shape rearrange, gather, band-part masking) never interpret
    an element's value, so one graph per width serves every dtype of that
    width -- including float16, which has no typed CPU kernel, and bool on GPU.

    Args:
        dtype: The dtype to reinterpret.

    Returns:
        The unsigned integer dtype of the same bit width.

    Raises:
        NotImplementedError: For sub-byte dtypes (such as ``float4_e2m1fn``), which
            pack multiple elements per byte and so cannot be reinterpreted
            element-for-element as a whole-byte unsigned int.
    """
    bits = dtype.size_in_bits
    if bits % 8 != 0 or bits // 8 not in _UINT_FOR_SIZE:
        raise NotImplementedError(
            f"eager GC copy path does not support sub-byte dtype {dtype}"
        )
    return _UINT_FOR_SIZE[bits // 8]


_SpecT = TypeVar("_SpecT")


def float_dtypes(device: Device) -> list[DType]:
    """Returns the float dtypes swept on *device*.

    CPU is f32/f64 (16-bit float kernels don't compile on CPU); GPU is
    f16/f32/bf16 (no f64 — NVIDIA rejects it for some ops, Metal lacks it;
    MSTDL-2711).
    """
    return CPU_FLOAT_DTYPES if device.label == "cpu" else GPU_FLOAT_DTYPES


def canonical_shape_rank1(shape: Sequence[int]) -> tuple[int]:
    """Flattens to rank 1; bare ``prod`` keeps scalars at 1 and empty at 0."""
    return (prod(shape),)


def canonical_rank2(shape: Sequence[int]) -> tuple[int, int]:
    """Collapses *shape* to rank 2 ``[rows, features]``.

    The last dim is ``features``; every leading dim is flattened into
    ``rows``. ``prod(())`` is 1, so a rank-1 input (no leading dims) yields
    ``rows == 1``, keeping the rank-1 case branchless.
    """
    return (prod(shape[:-1]), shape[-1])


def canonical_rank3(shape: Sequence[int], axis: int) -> tuple[int, int, int]:
    """Collapses *shape* to rank 3 ``[outer, axis, inner]``.

    *axis* is normalized for negatives; ``prod(())`` is 1, so a leading or
    trailing axis yields an outer/inner dim of 1.
    """
    ndim = len(shape)
    if axis < 0:
        axis += ndim
    return (prod(shape[:axis]), shape[axis], prod(shape[axis + 1 :]))


def flat_index_strides(indexed_shape: Sequence[int]) -> list[int]:
    """Row-major strides of *indexed_shape* (product of trailing sizes).

    Used by gather_nd/scatter_nd to flatten a multi-dimensional index vector
    into a single scalar: ``flat_idx = sum(idx[j] * stride[j])``. E.g.
    ``(4, 3, 2) -> [6, 2, 1]``. Plain Python arithmetic on a list of at most
    a handful of small ints (bounded by the interpreter's rank cap) --
    never touches a Buffer.
    """
    n = len(indexed_shape)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * indexed_shape[i + 1]
    return strides


def discover_devices() -> list[Device]:
    """Returns CPU + every accelerator slot in sweep/warm key order."""
    return load_devices([DeviceSpec.cpu()]) + load_devices(
        [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
    )


# Discovered once at import (not per family) so a missing driver fails here,
# not at first dispatch; every family sweeps this same device set.
DISCOVERED_DEVICES = discover_devices()


def device_class_of(device: Device) -> str:
    """Human-readable slot name for *device*: ``"cpu"`` or ``"gpu:{id}"``."""
    return "cpu" if device.label == "cpu" else f"gpu:{device.id}"


def device_spec_of(device: Device) -> DeviceSpec:
    """The :class:`~max.driver.DeviceSpec` that reloads *device* elsewhere.

    Lets an out-of-process warm worker rebuild a session over the same
    devices this process discovered (specs pickle; live devices don't).
    """
    return (
        DeviceSpec.cpu()
        if device.label == "cpu"
        else DeviceSpec.accelerator(device.id)
    )


def spec_for(
    op_type: type[_core.Operation], ops_by_name: Mapping[str, _SpecT]
) -> _SpecT | None:
    """Looks up *op_type*'s spec by canonical (mo) name in *ops_by_name*."""
    name = canonical_op_name(op_type, ops_by_name)
    return ops_by_name.get(name)


# Lazy-per-dispatch by default; ``=1`` precompiles the whole matrix at import
# (MXF-508).
# Re-exported so the CLI can consult it before importing this module.
EAGER_OP_PRECOMPILE_ENV_VAR = _eager_policy.OP_PRECOMPILE_ENV_VAR
should_precompile = _eager_policy.should_precompile

# Stored in the MEF cache dir (see _cache_dir) so it can't outlive the artifacts
# it vouches for.
_WARM_STAMP_NAME = "eager_gc_warm_stamp.json"


def _derived_root() -> Path | None:
    """The ``MODULAR_DERIVED_PATH`` root, or None if unset or empty."""
    try:
        derived = get_config_value("derived_path")
    except KeyError:
        return None
    return Path(derived) if derived else None


def _cache_dir() -> Path | None:
    """MEF cache dir the warm stamp lives in, or None if unresolvable.

    Prefers ``MODULAR_DERIVED_PATH``, else asks the engine, which may create
    the directory as a side effect.
    """
    root = _derived_root()
    if root is not None:
        return root / "cache" / ".max_cache"
    return _core.engine.max_cache_dir()


def _context_signature() -> str:
    """Signature a warm must match before a lazy process adopts it.

    Pins accelerator count + host arch + SKU + MAX version — versioned
    because the engine prunes sibling version directories beneath the stamp.
    """
    n = accelerator_count()
    accel = accelerator_architecture_name() if n else ""
    return (
        f"accelerators={n};cpu={platform.machine()};accel={accel}"
        f";version={_core.__version__}"
    )


def _split_device_count(signature: str) -> tuple[int, str]:
    """Split a context signature into ``(device count, everything else)``.

    Everything after the leading ``accelerators=N`` field must match exactly
    between warm and consumer; only the count is compared as a ceiling.

    Example:
        ``"accelerators=8;cpu=x86_64;accel=sm_100a;version=26.5.0.dev0"`` ->
        ``(8, "cpu=x86_64;accel=sm_100a;version=26.5.0.dev0")``.
    """
    head, _, sku = signature.partition(";")
    return int(head.removeprefix("accelerators=")), sku


def write_warm_stamp() -> bool:
    """Records a batched warm for this context. Returns False if the cache dir
    can't be located (the warm is then unadoptable; caller should surface it).

    Written via a temp file and rename, because concurrent
    ``MAX_EAGER_OP_PRECOMPILE=1`` processes race here.
    """
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"context": _context_signature()})
    tmp = cache_dir / f"{_WARM_STAMP_NAME}.{os.getpid()}.tmp"
    tmp.write_text(payload)
    tmp.replace(cache_dir / _WARM_STAMP_NAME)
    return True


def warm_stamp_matches() -> bool:
    """Returns whether a warm stamp this process can adopt is present.

    Device count is a ceiling; every other field must match exactly.
    """
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    try:
        stamp = json.loads((cache_dir / _WARM_STAMP_NAME).read_text())
        warmed_count, warmed_sku = _split_device_count(stamp["context"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    current_count, current_sku = _split_device_count(_context_signature())
    return current_sku == warmed_sku and current_count <= warmed_count


WARM_STORE_ENV_VAR = "MAX_EAGER_WARM_STORE"


def warm_artifact_name(family: str, device: Device) -> str:
    """Names the artifact holding *family*'s sweep for *device*.

    One source for the name, shared by the warm producer (which exports one
    artifact per family and device slot) and adoption (which loads them).
    """
    slot = "cpu" if device.label == "cpu" else f"slot_{device.id}"
    return f"{family}_{slot}.mef"


@functools.cache
def warm_store() -> engine.MefStore | None:
    """The build-time warm this process was given, or None.

    ``MAX_EAGER_WARM_STORE`` names its directories, separated by
    ``os.pathsep``: one per warm build action, which the store unions. Once
    named, a warm is all-or-nothing -- each family loads its artifacts from it
    or raises -- because a test that compiles after asking for a warm is a test
    whose warm is broken.

    Raises:
        FileNotFoundError: If a named directory was not written by the warm.
    """
    value = os.environ.get(WARM_STORE_ENV_VAR)
    if not value:
        return None
    return engine.MefStore.for_import(value.split(os.pathsep))


def provisioned() -> bool:
    """Returns whether this process can get its eager op models without
    compiling.

    True when a build-time warm store is named, or a compatible warm stamp is
    present — the single definition of "warmed", shared by the import-time
    gate and ``max warm-interpreter-cache --check``.
    """
    return warm_store() is not None or warm_stamp_matches()


def adopted_from_store(family: str) -> bool:
    """Whether ``family`` loaded its models from the warm store this process,
    rather than a cold or lazy compile."""
    cache = _REGISTRY.get(family)
    return cache.store_adopted if cache is not None else False


# Serializes lazy first-dispatches so concurrent threads don't race on cache
# mutation or a shared session's ``load_all``.
COMPILE_LOCK = threading.Lock()

# Per-device InferenceSession cache. Keyed by (label, id) since a CPU and an
# accelerator can share id 0.
_SESSION_CACHE: dict[tuple[str, int], engine.InferenceSession] = {}


def session_for(device: Device) -> engine.InferenceSession:
    """Returns a cached single-device :class:`~max.engine.InferenceSession`.

    Caching keeps lazy single-target compiles from recreating a session on
    every cache miss; the session is reused for the process lifetime.
    """
    cache_key = (device.label, device.id)
    session = _SESSION_CACHE.get(cache_key)
    if session is None:
        session = engine.InferenceSession(devices=[device])
        _SESSION_CACHE[cache_key] = session
    return session


class GCFamilySpec(Protocol):
    """Immutable per-family strategy: how one ``*_gc.py`` builds/sweeps its
    models.

    Each ``*_gc.py`` provides a concrete class implementing this Protocol
    (such as ``matmul_gc._MatmulFamily``); registered once via
    :func:`register_family` by wrapping it in a :class:`GCOpFamily`.
    Implementations are typically stateless, with methods closing over
    their module's own globals rather than instance state.
    """

    name: str
    """Family id; namespaces cache keys and per-slot MEF names."""

    def build_module(self) -> Module:
        """Builds the full sweep matrix (every device it sweeps)."""
        ...

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Builds one device's slot module.

        Args:
            device: The device to build for.
            module: When given, appends this device's graphs into it instead
                of a fresh :class:`Module`, so :meth:`build_module` can share
                one module across every device in the sweep.
        """
        ...

    def sweep_devices(self) -> list[Device]:
        """Returns the devices this family sweeps.

        Every family sweeps the same discovered set by default, so this is a
        concrete, inherited implementation rather than a per-family override
        point -- except for a family whose kernel is restricted to a device
        subset, which must override this to exclude the devices it never
        builds for: the ``MO_HostOnly`` families (such as ``nms_gc``)
        which are CPU-only, since sweeping an
        accelerator for a host-only op would only produce an empty per-slot
        module. The warm producer fails on a device this returns but builds
        nothing for, and adoption loads an artifact for every device it
        returns, so the two agree by construction. An empty list (none of
        the family's devices are present) warms as a no-op.
        """
        return list(DISCOVERED_DEVICES)


@dataclass
class GCOpFamily:
    """One eager GC op family's runtime state plus warm/adopt/dispatch logic.

    Wraps a :class:`GCFamilySpec` (registered once per family via
    :func:`register_family`); replaces the per-module ``_swept`` /
    ``_X_MODEL_CACHE`` globals each ``*_gc.py`` copied.
    """

    spec: GCFamilySpec
    """This family's build/sweep strategy."""

    cache: dict[str, engine.Model] = field(default_factory=dict)
    """Compiled models, keyed by graph_name."""
    swept: bool = False
    """Whether a whole-matrix sweep or adoption has been attempted."""
    store_adopted: bool = False
    """Whether the cache was loaded from the warm store."""

    @property
    def name(self) -> str:
        """Family id; namespaces cache keys and per-slot MEF names."""
        return self.spec.name

    def sweep_devices(self) -> list[Device]:
        return self.spec.sweep_devices()

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        return self.spec.build_module_for_device(device, module)

    @in_default_mlir_context
    def build_sweep_module(self) -> Module:
        """Builds the full sweep matrix module without compiling it.

        Exists so the out-of-process warm can compile the same module
        :meth:`compile_sweep` builds — its MEF cache key must match a
        later in-process sweep's.
        """
        return self.spec.build_module()

    @in_default_mlir_context
    def compile_sweep(self) -> bool:
        """Load the warm store if named, else batch-compile every target.

        Returns:
            True if it batch-compiled; False if it loaded the warm store
            instead, in which case the caller must not stamp.
        """
        if self._adopt_from_store():
            return False
        devices = self.sweep_devices()
        if not devices:
            # InferenceSession(devices=[]) silently defaults to a CPU session.
            self.swept = True
            return True
        session = engine.InferenceSession(devices=devices)
        self.cache.update(
            session.load_all(self.spec.build_module(), weights_registry={})
        )
        self.swept = True
        return True

    def ensure_swept(self) -> None:
        """Attempt whole-cache adoption once before per-target compilation.

        Loads the warm store if named, else batch-sweeps a matching warm
        stamp — the one path that may batch-compile under
        ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
        """
        if self.swept:
            return
        if self._adopt_from_store():
            return
        if warm_stamp_matches():
            self.swept = True
            start = time.perf_counter()
            try:
                self.compile_sweep()
            except RuntimeError:
                # If the sweep fails to compile, fall back to lazy per-target
                # compilation rather than breaking import.
                logger.warning(
                    "Eager interpreter warm-cache adoption failed;"
                    " compiling %s targets on demand.",
                    self.name,
                    exc_info=True,
                )
            else:
                _eager_policy.note_batched_adopt_end(
                    self.name, time.perf_counter() - start
                )

    def model_for(
        self,
        key: str,
        device: Device,
        build: Callable[[Module], None],
        *,
        unsupported_reason: Callable[[], str | None] | None = None,
    ) -> engine.Model:
        """Shared lazy-dispatch skeleton every family's ``*_model`` function
        wraps: cache hit, optional support-guard, then a locked adopt, the
        cold-start policy check, and a per-target compile.

        The policy check runs after adoption, not before: refusing ahead of
        adoption would leave a warmed machine unable to serve any eager op.

        Args:
            key: The cache key for this target (family-specific format).
            device: The target device, passed through to *build* on a lazy
                miss.
            build: Builds the single-target graph into a fresh ``Module``.
            unsupported_reason: When given, called only on a cache miss (a
                cache hit never pays for the support check) to test whether
                the target is outside the family's supported set; a
                non-``None`` return is raised as ``KeyError`` verbatim before
                anything is adopted or compiled (each family formats its own
                wording/details).

        Returns:
            The compiled :class:`~max.engine.Model`.

        Raises:
            UnsupportedOpVariant: *unsupported_reason*'s message if it
                returns one.
            EagerLazyCompileDisallowed: if the target is still missing after
                adoption and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
        """
        model = self.cache.get(key)
        if model is not None:
            return model
        if unsupported_reason is not None:
            reason = unsupported_reason()
            if reason is not None:
                raise _eager_policy.UnsupportedOpVariant(reason)
        with COMPILE_LOCK:
            model = self.cache.get(key)
            if model is not None:
                return model
            self.ensure_swept()
            model = self.cache.get(key)
            if model is not None:
                return model
            if not _eager_policy.allow_lazy_compile():
                raise _eager_policy.EagerLazyCompileDisallowed(
                    _eager_policy.refusal_message(
                        key, provisioned=provisioned()
                    )
                )
            start = time.perf_counter()
            model = compile_single_target(self, key, device, build)
            _eager_policy.record_cold_compile(
                key, time.perf_counter() - start, provisioned=provisioned()
            )
            return model

    def _adopt_from_store(self) -> bool:
        """Load this family's per-slot artifacts from the warm store.

        Returns:
            Whether a warm store is named; when it is, every slot loaded.

        Raises:
            MissingArtifactError: If the warm has no artifact for one of the
                devices this process sweeps -- more accelerators than were
                warmed, say.
        """
        store = warm_store()
        if store is None:
            return False
        self.swept = True
        # Load the slots for exactly the devices the session spans: both come
        # from the one sweep_devices() call, so they can't diverge.
        devices = self.sweep_devices()
        if not devices:
            # Nothing to adopt; avoid the silent default-CPU session.
            return True
        session = engine.InferenceSession(devices=devices)
        names = [warm_artifact_name(self.name, device) for device in devices]
        for name in names:
            self.cache.update(
                session.load_all(str(store.artifact(name)), weights_registry={})
            )
        self.store_adopted = True
        print(
            f"[eager-warm] loaded {self.name} from the warm store {names}",
            file=sys.stderr,
            flush=True,
        )
        return True


_REGISTRY: dict[str, GCOpFamily] = {}


def register_family(family: GCOpFamily) -> None:
    """Register *family*; each ``*_gc.py`` calls this once at import."""
    _REGISTRY[family.name] = family


def registered_families() -> tuple[GCOpFamily, ...]:
    """Every registered family, in registration (import) order."""
    return tuple(_REGISTRY.values())


@in_default_mlir_context
def compile_single_target(
    family: GCOpFamily,
    key: str,
    device: Device,
    build_graph: Callable[[Module], None],
) -> engine.Model:
    """Compile one graph into *family*'s cache and return it (lazy path)."""
    module = Module()
    build_graph(module)
    session = session_for(device)
    family.cache.update(session.load_all(module, weights_registry={}))
    return family.cache[key]
