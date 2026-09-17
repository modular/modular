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

"""Lazy per-target interpreter bindings for user custom ops.

Compiles one rank-polymorphic binding per declared ``CustomOp`` call
signature (:func:`binding_for`). Keeps its own lazy cache and skips
:mod:`~max._interpreter_ops.gc_compile`'s sweep-based ``model_for``/family
registration, since a custom op's key set isn't known until the user
defines it.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from max import engine
from max._interpreter_ops import gc_compile
from max._mlir_context import in_default_mlir_context
from max.driver import Device
from max.dtype import DType
from max.experimental.custom import CustomOp, DTypeVar, TemplateType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.graph import _resolved_custom_extensions

#: One :class:`~max.graph.TensorType` canonicalized for a key; see `_type_key`.
_TypeKey = tuple[str, int, str, tuple[str, ...]]


@dataclass(frozen=True)
class BindingKey:
    """Cache key for one CustomOp-declared custom op's compiled binding.

    Keyed on the binding's own signature (:func:`_binding_types`), whose
    operands carry symbolic dims, so every shape of a given rank shares one
    entry while a def that pins an input dim keeps its own. Excludes the
    ``CustomOp``'s own identity, so an identical re-declaration shares an
    entry. ``lib_hashes`` covers the process-global overlay as well as the
    def's own extensions, since the compiled graph links against both.
    """

    symbol: str
    operands: tuple[_TypeKey, ...]
    results: tuple[_TypeKey, ...]
    lib_hashes: tuple[str, ...]


def _lib_hashes(extensions: Sequence[Path]) -> tuple[str, ...]:
    """Content-hashes *extensions*, mirroring ``_eager_model_cache_key``.

    Re-reads bytes on every call: no stat-keyed memo is sound, since no
    combination of size, mtime, and inode rules out a size-preserving
    rebuild that leaves the recorded mtime where it was, and a stale digest
    here serves a binding compiled from kernel bytes that no longer exist.

    Walks a directory extension's files (sorted, for determinism) rather
    than invoking the Mojo compiler, which is the expensive step the binding
    cache exists to skip.

    TODO(pprovins): runs per dispatch, including on a cache hit, so a
    source-directory extension costs a full recursive read of that tree per
    custom-op call; a packaged extension costs one read. Revisit if a
    dispatch-heavy eager workload hits it -- the fix needs an invalidation
    protocol, not a stat memo (see above).
    """
    hashes: list[str] = []
    for ext in sorted(str(Path(p)) for p in extensions):
        path = Path(ext)
        files = (
            sorted(f for f in path.rglob("*") if f.is_file())
            if path.is_dir()
            else [path]
        )
        hashes.extend(hashlib.sha256(f.read_bytes()).hexdigest() for f in files)
    return tuple(hashes)


def _type_key(t: TensorType) -> _TypeKey:
    """Canonicalizes one binding signature type into a hashable key field.

    Each dim contributes its own ``repr``, so two bindings differing only in
    how a dim is spelled -- an out dim computed as ``w // 2`` vs. ``w // 3``,
    or an input dim pinned to a parameter value vs. left symbolic -- get
    distinct keys.
    """
    return (
        t.dtype.name,
        t.rank,
        str(t.device),
        tuple(repr(d) for d in t.shape),
    )


def _key(
    defn: CustomOp,
    binding_types: tuple[list[TensorType], list[TensorType]],
    extensions: Sequence[Path],
) -> BindingKey:
    """Builds the cache key for a binding of *binding_types*.

    Keys both halves of the signature: nothing else distinguishes two defs
    sharing a kernel symbol and extensions but different declared ``inputs``
    or ``outputs``. The device isn't a separate field, since each type
    already carries its own.

    Hashes *extensions* plus the process-global overlay, the libraries
    :func:`_compile`'s ``Graph`` actually links: on *extensions* alone, a
    binding compiled inside an overlay scope would keep serving after the
    scope exits. Normalizes to ``Path``, since the overlay union dedupes by
    equality and a ``Path`` never equals the ``str`` spelling the same file.
    """
    sym_types, out_types = binding_types
    return BindingKey(
        symbol=defn.name,
        operands=tuple(_type_key(t) for t in sym_types),
        results=tuple(_type_key(t) for t in out_types),
        lib_hashes=_lib_hashes(
            _resolved_custom_extensions(Path(p) for p in extensions)
        ),
    )


@in_default_mlir_context
def make_key(
    defn: CustomOp,
    in_types: Sequence[TensorType],
    extensions: Sequence[Path],
) -> BindingKey:
    """Returns the key a call of *in_types* binds under."""
    return _key(defn, _binding_types(defn, in_types), extensions)


# Bounds memory: each entry pins an `engine.Model`, its MEF buffer, and the
# device allocations behind it.
_CACHE_MAX_SIZE = int(os.environ.get("MAX_CUSTOM_OP_BINDING_CACHE_SIZE", "128"))
_CACHE_LOCK = threading.Lock()
_CACHE: OrderedDict[BindingKey, engine.Model] = OrderedDict()


def _cached(key: BindingKey) -> engine.Model | None:
    """Returns *key*'s binding and refreshes its recency, else ``None``."""
    with _CACHE_LOCK:
        model = _CACHE.get(key)
        if model is not None:
            _CACHE.move_to_end(key)
        return model


@in_default_mlir_context
def binding_for(
    defn: CustomOp,
    device: Device,
    in_types: Sequence[TensorType],
    extensions: Sequence[Path],
) -> engine.Model:
    """Returns *defn*'s binding for *in_types*, compiling on a cache miss.

    Derives the binding signature once and both keys and compiles from that
    one tuple, so the cached key cannot describe a different signature than
    the graph stored under it.

    :data:`~max._interpreter_ops.gc_compile.COMPILE_LOCK` serializes
    concurrent first dispatches so two don't compile the same binding; cache
    reads and writes take :data:`_CACHE_LOCK` instead, so a hit never waits
    behind an unrelated compile. The cache is a bounded LRU (see
    :data:`_CACHE_MAX_SIZE`): a process that keeps declaring fresh
    signatures would otherwise pin every model it ever compiled.
    """
    binding_types = _binding_types(defn, in_types)
    key = _key(defn, binding_types, extensions)
    model = _cached(key)
    if model is not None:
        return model
    with gc_compile.COMPILE_LOCK:
        model = _cached(key)
        if model is None:
            model = _compile(defn, key, device, binding_types, extensions)
            with _CACHE_LOCK:
                _CACHE[key] = model
                if len(_CACHE) > _CACHE_MAX_SIZE:
                    _CACHE.popitem(last=False)
    return model


def _graph_name(key: BindingKey) -> str:
    """Deterministic, unique ``mo.graph`` symbol name for *key*."""
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return f"custom_op_binding_{key.symbol}_{digest}"


def _binding_types(
    defn: CustomOp, in_types: Sequence[TensorType]
) -> tuple[list[TensorType], list[TensorType]]:
    """Returns one binding's (symbolic input types, symbolic output types).

    The declared signature *is* the binding signature, so this reads
    ``defn.inputs``/``defn.outputs`` as-is, never minting a fresh name.
    Depends only on dtype and rank, never a concrete dim, which is what lets
    one binding serve every shape of a given rank. :func:`binding_for` evaluates this once per
    dispatch and hands the result to both :func:`_key` and :func:`_compile`.
    """
    device = in_types[0].device
    dtype_bindings: dict[str, DType] = {}
    for spec, actual in zip(defn.inputs.values(), in_types, strict=False):
        if isinstance(spec.dtype, DTypeVar):
            dtype_bindings.setdefault(spec.dtype.name, actual.dtype)

    def concrete_dtype(spec: TemplateType) -> DType:
        return (
            dtype_bindings[spec.dtype.name]
            if isinstance(spec.dtype, DTypeVar)
            else spec.dtype
        )

    sym_types = [
        TensorType(concrete_dtype(spec), list(spec.shape), device)
        for spec in defn.inputs.values()
    ]
    out_types = [
        TensorType(concrete_dtype(spec), list(spec.shape), device)
        for spec in defn.outputs
    ]
    return sym_types, out_types


def _compile(
    defn: CustomOp,
    key: BindingKey,
    device: Device,
    binding_types: tuple[list[TensorType], list[TensorType]],
    extensions: Sequence[Path],
) -> engine.Model:
    """Builds and compiles one rank-polymorphic binding for *defn* at *key*.

    Compiles via the ``Graph`` branch of ``InferenceSession.load_all``, not
    the bare-``Module`` branch: only ``Graph`` threads
    ``kernel_libraries_paths`` into ``custom_extensions`` and seeds the
    kernel decls, so the kernel symbol actually resolves. A bare ``Module``
    would compile without error but leave it unbound.
    """
    sym_types, out_types = binding_types
    graph = Graph(
        _graph_name(key),
        input_types=sym_types,
        custom_extensions=[Path(p) for p in extensions],
    )
    with graph:
        results = ops.custom(
            defn.name,
            DeviceRef.from_device(device),
            list(graph.inputs),
            out_types=out_types,
        )
        graph.output(*results)
    session = gc_compile.session_for(device)
    models = session.load_all(graph, weights_registry={})
    return next(iter(models.values()))
