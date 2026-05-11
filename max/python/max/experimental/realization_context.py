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

"""Implementations of various realization strategies.

**"Eager" execution**: tensors are realized as soon as the realization context
exits. This is the default behavior.

This has a huge concrete advantage over eagerly executing one operation
at a time: by controlling the boundary of where the eager context starts
and ends, we can give advanced users a tool to _enable fine-grained
bounds for automatic fusion_!

In practice the easiest way to do this is to mark a function as
`F.functional`. This function is then assumed to be "atomic" for the
purposes of eager execution. All ops within the function execute as
part of the same graph, meaning the compiler is free to fuse operations
and generate fused kernels within this region.

**"Lazy" execution**: tensors are realized only when code later tries to use
them.

This enables a class of interface design common in the ML world, in
which layers are constructed with randomized weights which are never
used. Lazy execution neatly allows constructing entire models,
only performing the weight initialization and allocating memory for
them if and when those weights are actually used.

**Graph compilation**: tensors must never be realized.

This allows tensor operations to be composed with direct usage of
the Graph API, for instance `Module.compile`, or using `F.*` operations
in another Graph API usage.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
import weakref
from collections import OrderedDict
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar, cast

from max import _core, driver, engine
from max._core.dialects import builtin, rmo
from max._mlir_context import in_default_mlir_context
from max.dtype import DType
from max.experimental import _passes
from max.experimental import functional as F
from max.experimental.support import driver_tensor_type
from max.experimental.tensor import (
    GraphValue,
    RealizationContext,
    RealizationState,
    Tensor,
    current_realization_context,
    realization_context,
)
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    Shape,
    Value,
    ops,
)

if TYPE_CHECKING:
    from max.experimental.sharding import DeviceMapping, DeviceMesh

Ex = TypeVar("Ex", bound=BaseException)

_SESSION_LOCK = threading.Lock()
_SESSION: engine.api.InferenceSession | None = None
_SEED: Tensor | None = None

# Each distinct (op name, input dtypes/shapes) combination produces a unique
# graph and thus a unique cache entry.  128 is generous for typical workloads
# (a handful of custom ops x a few shape variants) while bounding memory.
_EAGER_MODEL_CACHE_MAX_SIZE = 128
_EAGER_MODEL_CACHE_LOCK = threading.Lock()
_EAGER_MODEL_CACHE: OrderedDict[
    tuple[str, tuple[tuple[str, str], ...]],
    engine.Model,
] = OrderedDict()
_EAGER_MODEL_CACHE_SESSION: engine.api.InferenceSession | None = None

# Environment variable to control interpreter usage.
# Set to "0" or "false" to disable the interpreter (always compile).
_USE_INTERPRETER_ENV_VAR = "MAX_USE_EAGER_INTERPRETER"

# Environment variable to control the maximum number of dispatchable ops
# for which the interpreter is preferred over the graph compiler.
# Graphs with more ops than this threshold are compiled so the graph
# compiler can apply fusion.
# Benchmarks (CPU & A10G GPU, [64,64] f32 tensors) show the interpreter
# is 7-10x faster than the compiler for up to 10 user-visible ops
# (~30 dispatchable IR ops). Distributed dispatch and shape-heavy ops
# routinely produce well beyond 30 IR nodes per single user-visible op,
# so the threshold is set high enough to keep eager paths on the
# interpreter rather than falling back to a full compile.
_INTERPRETER_MAX_OPS_ENV_VAR = "MAX_INTERPRETER_MAX_OPS"
_DEFAULT_INTERPRETER_MAX_OPS = 1024


def _default_use_interpreter() -> bool:
    """Get the default value for use_interpreter from environment.

    The interpreter is **enabled by default** for small graphs.  Set
    ``MAX_USE_EAGER_INTERPRETER=0`` or ``false`` to force compilation.

    Returns:
        True if interpreter should be used by default, False otherwise.
    """
    env_value = os.environ.get(_USE_INTERPRETER_ENV_VAR, "").lower()
    return env_value not in ("0", "false")


def _interpreter_max_ops() -> int:
    """Get the maximum dispatchable-op count for interpreter execution.

    Reads ``MAX_INTERPRETER_MAX_OPS`` from the environment.  Graphs with
    more dispatchable ops than this value fall through to the graph
    compiler so fusion optimizations can kick in.

    Returns:
        The op-count threshold (default 30).
    """
    raw = os.environ.get(_INTERPRETER_MAX_OPS_ENV_VAR, "")
    if raw.strip().isdigit():
        return int(raw.strip())
    return _DEFAULT_INTERPRETER_MAX_OPS


def seed() -> Tensor:
    """Gets the global random seed tensor used in eager execution mode."""
    global _SEED
    if _SEED is None:
        seed_type = ops.random.SeedType(DeviceRef.CPU())
        shape = [int(d) for d in seed_type.shape]
        seed_data = driver.Buffer(
            seed_type.dtype, shape, seed_type.device.to_device()
        )
        _SEED = Tensor(storage=seed_data)
    return _SEED


def set_seed(value: int) -> None:
    """Sets the global random seed value.

    Updates the global random seed to the specified value. This affects all
    subsequent random number generation in eager execution mode.

    Args:
        value: The integer seed value to set.
    """
    seed().driver_tensor[0] = value


def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels on tensors."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            device_specs = driver.scan_available_devices()
            if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
                device_specs.append(cpu)
            devices = driver.load_devices(device_specs)
            _SESSION = engine.api.InferenceSession(devices=devices)
        return _SESSION


# ─── Shared signal-buffer cache (allocated once per device set) ──────────


# maxsize=4: one entry per unique device-set configuration.  Most users
# have a single configuration; testing may use a few more.  On eviction
# the next call re-allocates (expensive but correct).
@functools.lru_cache(maxsize=4)
def _cached_signal_buffers(
    device_ids: tuple[int, ...],
) -> tuple[list[driver.Buffer], list[BufferType]]:
    """Returns (runtime_buffers, buffer_types) for the given GPU device IDs.

    Signal buffers are 513 MB each — far too expensive to re-allocate per
    eager graph.  ``lru_cache`` ensures they are allocated once for each
    unique device set and reused for all subsequent graphs.

    Using ``lru_cache`` on an immutable key (tuple of ints) is thread-safe
    and avoids mutable module-level state.  In pytest-xdist each worker is
    a separate process, so there are no cross-worker conflicts.
    """
    # Signal buffers: 1 MB signal + 512 MB communication scratch per GPU.
    # Must stay in sync with the Mojo ``Signal`` struct size.
    _NUM_BYTES = (1 + 512) * 1024 * 1024

    try:
        driver.enable_all_peer_access()
    except RuntimeError:
        logging.getLogger(__name__).warning(
            "Failed to enable peer-to-peer GPU access. "
            "Collective operations will fall back to slower paths."
        )

    accelerators = [driver.Accelerator(id=i) for i in device_ids]
    runtime_bufs = [
        driver.Buffer.zeros(
            shape=(_NUM_BYTES,), dtype=DType.uint8, device=accel
        )
        for accel in accelerators
    ]
    for accel in accelerators:
        accel.synchronize()

    buf_types = [
        BufferType(
            dtype=DType.uint8, shape=(_NUM_BYTES,), device=DeviceRef.GPU(id=i)
        )
        for i in device_ids
    ]
    return runtime_bufs, buf_types


def _make_unrealized(
    ctx: RealizationContext,
    values: tuple[GraphValue, ...],
    mapping: DeviceMapping | None,
    global_shape: Shape | None,
) -> Tensor:
    """Wraps graph values into a Tensor, dispatching to sharded constructor if needed."""
    state = RealizationState(values, ctx)
    if mapping is not None and mapping.mesh.num_devices > 1:
        placements = mapping.to_placements()
        return Tensor._from_unrealized_shards(
            state, mapping.mesh, placements, global_shape
        )
    return Tensor(state=state)


# ─── In-memory cache for compiled custom-op models ───────────────────────


def _eager_model_cache_key(
    graph: Graph,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Builds a compact, stable cache key for a finalized eager graph.

    Uses a SHA-256 hash of the MLIR module ASM (with debug info stripped)
    combined with the resolved kernel library paths and SHA-256 hashes of
    their contents.  Hashing file contents (rather than ``st_mtime``)
    avoids a time-of-check/time-of-use race and produces a deterministic
    key regardless of filesystem timestamp granularity.

    Args:
        graph: A finalized graph ready for compilation.

    Returns:
        A tuple of ``(asm_hex_digest, ((resolved_path, content_hash), ...))``.
    """
    module_asm = graph._module.asm(
        assume_verified=True,
        enable_debug_info=False,
        pretty_debug_info=False,
        use_local_scope=True,
    )
    asm_hash = hashlib.sha256(module_asm.encode()).hexdigest()
    kernel_paths = tuple(
        (
            str(Path(p).resolve()),
            hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        )
        for p in graph.kernel_libraries_paths
    )
    return (asm_hash, kernel_paths)


def _load_eager_model(graph: Graph) -> engine.Model:
    """Loads or retrieves a cached compiled model for an eager graph.

    Only caches graphs that use custom kernel libraries (custom ops),
    since those bypass the interpreter and incur expensive per-call
    compilation.  Regular graphs use the interpreter fast path and are
    not cached.

    The compiled ``Model`` is keyed by a hash of the graph IR plus the
    resolved kernel library paths and content hashes so that recompiling
    a ``.mojopkg`` automatically invalidates the cache.

    Returns:
        A compiled ``engine.Model`` ready for execution.
    """
    global _EAGER_MODEL_CACHE_SESSION

    session = _session()
    if not graph.kernel_libraries_paths:
        return session.load(graph)

    key = _eager_model_cache_key(graph)

    with _EAGER_MODEL_CACHE_LOCK:
        if _EAGER_MODEL_CACHE_SESSION is not session:
            _EAGER_MODEL_CACHE.clear()
            _EAGER_MODEL_CACHE_SESSION = session

        cached = _EAGER_MODEL_CACHE.get(key)
        if cached:
            _EAGER_MODEL_CACHE.move_to_end(key)
            return cached

    model = session.load(graph)

    with _EAGER_MODEL_CACHE_LOCK:
        if _EAGER_MODEL_CACHE_SESSION is session:
            _EAGER_MODEL_CACHE[key] = model
            if len(_EAGER_MODEL_CACHE) > _EAGER_MODEL_CACHE_MAX_SIZE:
                _EAGER_MODEL_CACHE.popitem(last=False)

    return model


class EagerRealizationContext(RealizationContext):
    """Computation graph for managing tensor operations.

    This class manages the directed acyclic graph (DAG) of tensor operations
    for lazy evaluation and optimization. It tracks both realized tensors
    (with concrete data in memory) and unrealized tensors (pending computations)
    to enable efficient batch compilation and execution.
    """

    graph: Graph
    #: Keeps a strong reference to tensor data that we need to compute graph values
    sources: dict[_core.Value[Any], Tensor]
    #: Reverse map of sources (TensorValue for read-only, BufferValue for mutable)
    source_values: dict[int, Value[Any]]
    #: Unrealized values
    unrealized: list[weakref.ref[Tensor]]
    #: Signal buffer graph values for multi-device collectives (lazily created).
    signal_buffers: list[BufferValue] | None

    def __init__(self, use_interpreter: bool | None = None):
        # When use_interpreter is None (the default), the op-count threshold
        # gates whether the interpreter is used.  When the caller explicitly
        # passes True, the threshold is bypassed so the interpreter is always
        # attempted (falling back only on truly unsupported ops).
        self._auto_interpreter = use_interpreter is None
        if use_interpreter is None:
            use_interpreter = _default_use_interpreter()
        self._use_interpreter = use_interpreter
        self.sources = {}
        self.source_values = {}
        self.unrealized = []
        self.signal_buffers = None

        self.graph = Graph("main", input_types=[])

        with realization_context(self), self.graph:
            ops.random.set_seed(seed())

    def finalize_graph(self) -> tuple[list[Tensor], Graph]:
        """Finalizes the computation graph for execution.

        Prepares the graph for compilation by setting outputs, removing dead
        code and unused arguments, and replacing static shapes with symbolic
        parameters. This method is called internally before graph execution.

        Returns:
            tuple[list[Tensor], Graph]: A tuple containing the list of output
                tensors (including the seed) and the finalized graph.
        """
        with realization_context(self), self.graph:
            # peek rather than next! If compilation or execute fails
            # the seed should remain the same.
            outputs = [
                Tensor.from_graph_value(ops.random._peek_seed()),
                *(
                    tensor
                    for ref in self.unrealized
                    if (tensor := ref()) is not None
                ),
            ]
            flat_values = [
                s._graph_value for t in outputs for s in t.local_shards
            ]
            self.graph.output(*flat_values)
        # Remove sources that no longer exist from the graph
        _core.lower(
            self.graph._module,
            [
                builtin.passes.RemoveDeadValuesPass(),
                rmo.passes.LegalizeRMOOps(),
            ],
        )
        # The graph symbol is public, so RemoveDeadValues won't remove
        # unused arguments. Do that explicitly.
        _passes.remove_unused_arguments(self.graph)
        return outputs, self.graph

    # Lazy realize fires after the surrounding `with` exits — re-enter on bg threads.
    @in_default_mlir_context
    async def realize_all(self) -> list[Tensor]:
        """Compiles and executes the computation graph, realizing all tensors.

        Finalizes the computation graph, compiles it using the inference
        session, and executes it to produce concrete values for all pending
        (unrealized) tensors. After execution, all tensors tracked by this
        context will have their data in memory.

        Returns:
            list[Tensor]: The list of realized output tensors (excluding the
                internal seed tensor).

        Raises:
            TypeError: If called while still inside this realization context.
        """
        if current_realization_context(None) is self:
            raise TypeError(
                "Can't realize tensor before realization context is completed."
            )

        outputs, graph = self.finalize_graph()

        # Execute graph via interpreter or compilation.
        # The interpreter is faster for small graphs where fusion has no
        # benefit; larger graphs are compiled so the graph compiler can
        # fuse and optimize across ops.  The op-count threshold only
        # applies when the interpreter was auto-selected (not explicitly
        # requested by the caller).
        use_interpreter = self._use_interpreter
        if use_interpreter:
            from max._interpreter import MOInterpreter

            interp = MOInterpreter()
            max_ops = _interpreter_max_ops() if self._auto_interpreter else None
            if not interp.can_execute(graph, max_ops=max_ops):
                use_interpreter = False

        # All graph inputs (tensor data + signal buffers) go through
        # self.sources — signal buffers are registered there by
        # ensure_signal_buffers().
        input_buffers = [
            self.sources[inp._mlir_value].driver_tensor for inp in graph.inputs
        ]

        if use_interpreter:
            if self._auto_interpreter:
                try:
                    results = interp.execute(graph, input_buffers)
                except Exception:
                    logging.getLogger("max.experimental").debug(
                        "Interpreter failed, falling back to graph compiler",
                        exc_info=True,
                    )
                    use_interpreter = False
            else:
                results = interp.execute(graph, input_buffers)
        if not use_interpreter:
            model = _load_eager_model(graph)
            results = model(*input_buffers)

        # Update tensors to realized.
        # Each tensor consumes num_shards consecutive results (1 for
        # unsharded, N for sharded).
        result_idx = 0
        for tensor in outputs:
            n = tensor.num_shards
            extracted = results[result_idx : result_idx + n]
            if not all(isinstance(buf, driver.Buffer) for buf in extracted):
                raise TypeError(
                    "Expected all results to be driver.Buffer, got: "
                    + str([type(b).__name__ for b in extracted])
                )
            tensor._storages = tuple(cast(list[driver.Buffer], extracted))
            tensor._state = None
            result_idx += n

        # Update mutated buffer inputs to realized
        for source in self.sources.values():
            # This was set by calling `__buffervalue__` on the source.
            # Mark the tensor as realized again.
            if source._state and source._state.ctx is self:
                source._state = None

        new_seed, *outputs = outputs
        set_seed(new_seed.item())

        return outputs

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a realized tensor as an input source to the computation graph.

        Registers a realized tensor as a graph input, allowing it to be used
        in subsequent graph operations. The tensor's data will be passed to
        the compiled graph during execution. This operation is idempotent;
        adding the same tensor multiple times returns the same state.

        Args:
            tensor: A realized tensor to add as a graph input source.

        Returns:
            RealizationState: The state associating the tensor with its graph
                value and this context.

        Raises:
            TypeError: If the tensor is not realized (has no concrete data).
        """
        if not tensor.real:
            raise TypeError("Only realized tensors may be graph sources.")

        return self._add_source(tensor, mutable=False)

    def add_mutable_source(self, tensor: Tensor) -> RealizationState:
        """Adds a realized tensor as a mutable graph input.

        Like :meth:`add_source` but creates a ``BufferType`` (mutable) input
        so the tensor can be mutated in-place by ``buffer_store``.
        """
        return self._add_source(tensor, mutable=True)

    def _add_source(self, tensor: Tensor, *, mutable: bool) -> RealizationState:
        if not tensor.real:
            raise TypeError("Only realized tensors may be graph sources.")

        # Safe to use IDs because self.sources keeps references alive.
        # If already added, return the cached value — but upgrade to
        # mutable if requested and not already mutable.
        if (cached := self.source_values.get(id(tensor))) is not None:
            if mutable and not isinstance(cached, BufferValue):
                # Need to upgrade: remove old read-only input, add mutable.
                pass  # Fall through to create a new mutable input.
            else:
                return RealizationState((cast(GraphValue, cached),), self)

        assert tensor.storage
        src_type = driver_tensor_type(tensor.storage)
        input_type = src_type.as_buffer() if mutable else src_type
        value = _passes.add_input(self.graph, input_type)
        if mutable:
            assert isinstance(value, BufferValue)
        self.sources[value._mlir_value] = tensor
        self.source_values[id(tensor)] = value
        return RealizationState((cast(GraphValue, value),), self)

    def create_unrealized(
        self,
        values: tuple[GraphValue, ...],
        *,
        mapping: DeviceMapping | None = None,
        global_shape: Shape | None = None,
    ) -> Tensor:
        """Creates an unrealized tensor backed by graph value(s)."""
        tensor = _make_unrealized(self, values, mapping, global_shape)
        self.unrealized.append(weakref.ref(tensor))
        return tensor

    def ensure_signal_buffers(
        self, mesh: DeviceMesh
    ) -> list[BufferValue] | None:
        """Lazily creates signal buffers for multi-device collectives on *mesh*.

        Called by collective ops when they detect a multi-GPU mesh.  On the
        first call, this adds ``BufferType`` graph inputs and caches the
        resulting ``BufferValue`` list so subsequent collectives in the
        same graph reuse the same buffers.

        The runtime ``driver.Buffer`` objects (513 MB each) are allocated
        once per device set via :func:`_cached_signal_buffers` and shared
        across all eager contexts to avoid repeated allocation.

        Returns ``None`` for single-device or CPU-only meshes.
        """
        if self.signal_buffers is not None:
            return self.signal_buffers

        from max.driver import Accelerator as _Acc

        gpu_ids: list[int] = []
        seen: set[int] = set()
        for dev in mesh.devices:
            if isinstance(dev, _Acc) and dev.id not in seen:
                gpu_ids.append(dev.id)
                seen.add(dev.id)

        if len(gpu_ids) < 2:
            return None

        # Get or allocate shared runtime buffers (expensive — 513 MB each).
        runtime_bufs, buf_types = _cached_signal_buffers(tuple(gpu_ids))

        # Add signal buffer types as new graph inputs (per-graph, cheap).
        # Register them in self.sources so the execution loop picks them
        # up naturally alongside tensor data — no special-casing needed.
        buf_values: list[BufferValue] = []
        for i, bt in enumerate(buf_types):
            value = _passes.add_input(self.graph, bt)
            assert isinstance(value, BufferValue)
            buf_values.append(value)
            self.sources[value._mlir_value] = Tensor(storage=runtime_bufs[i])

        self.signal_buffers = buf_values
        return self.signal_buffers

    def __enter__(self):
        self.graph.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)
        if not exception:
            F._run(self.realize_all())


class LazyRealizationContext(EagerRealizationContext):
    """A realization context that defers execution until explicitly requested.

    Unlike :class:`~max.experimental.realization_context.EagerRealizationContext`, this context does not automatically
    execute the computation graph when the context exits. Tensors remain
    unrealized until explicitly awaited via ``await tensor.realize``.

    This is useful for batching many operations together before execution,
    improving performance by reducing compilation overhead.

    Example::

        with F.lazy():
            a = Tensor.zeros([5, 5])
            b = a + 1
            c = b * 2
        # No execution yet - all tensors are unrealized
        assert not c.real

        await c.realize  # Now compile and execute
        assert c.real
    """

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)


class GraphRealizationContext(RealizationContext):
    """A realization context for ahead-of-time graph compilation.

    This context is used when building computation graphs that will be compiled
    and executed later (e.g., during :meth:`~max.experimental.nn.Module.compile`). Tensors in this
    context remain as symbolic graph values and cannot be realized.

    Unlike eager contexts, this context does not support executing operations
    immediately. Attempting to realize tensors will raise a TypeError.

    Attributes:
        graph: The graph being constructed in this context.

    Example::

        graph = Graph("my_model", input_types=[TensorType(...)])
        with GraphRealizationContext(graph) as ctx:
            x = Tensor.from_graph_value(graph.inputs[0])
            y = x + 1  # Creates graph operation, not computation
            graph.output(y)
        # Graph can now be compiled and executed separately
    """

    graph: Graph
    signal_buffers: list[BufferValue] | None

    def __init__(
        self,
        graph: Graph,
        signal_buffers: list[BufferValue] | None = None,
    ):
        """Initializes the graph realization context.

        Args:
            graph: The graph to construct operations in.
            signal_buffers: GPU signal buffer graph values for
                multi-device collective ops.
        """
        self.graph = graph
        self.signal_buffers = signal_buffers

    async def realize_all(self) -> list[Tensor]:
        """Raises TypeError - graph contexts cannot realize tensors.

        Raises:
            TypeError: Always raised, as graph contexts are for symbolic
                graph construction only.
        """
        raise TypeError("Can't realize from a graph context.")

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a tensor as a constant in the graph.

        In graph context, source tensors become constant values embedded
        in the graph rather than graph inputs.

        Args:
            tensor: The tensor to embed as a constant.

        Returns:
            RealizationState: The state with the constant graph value.
        """
        return RealizationState((ops.constant(tensor),), self)

    def add_mutable_source(self, tensor: Tensor) -> RealizationState:
        """In graph context, same as add_source (constants are immutable)."""
        return self.add_source(tensor)

    def create_unrealized(
        self,
        values: tuple[GraphValue, ...],
        *,
        mapping: DeviceMapping | None = None,
        global_shape: Shape | None = None,
    ) -> Tensor:
        """Creates a tensor backed by graph value(s)."""
        return _make_unrealized(self, values, mapping, global_shape)

    def __enter__(self):
        self.graph.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)
