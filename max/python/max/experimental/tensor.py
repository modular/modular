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

"""Provides tensor operations with eager execution capabilities.

This module provides the :class:`~max.experimental.tensor.Tensor` class which supports
eager execution of tensor operations, complementing the graph-based execution
model provided by :obj:`~max.graph`. The tensor operations automatically compile
and execute using the MAX runtime.

**Key Features:**

- **Eager semantics**: Operations give immediate results for quick iteration and feedback.
- **High performance**: All operations use high-performance Mojo implementations
    compiled specifically for the available hardware.
- **Automatic compilation**: Tensors are compiled and optimized automatically.
    Operations may be easily fused into larger graphs to take advantage of
    the graph compiler's automatic fusions.
- **Lazy evaluation**: Tensors may be computed lazily until their values are needed.
- **Familiar API**: Supports common array operations and indexing.

.. note::

  Tensors use lazy evaluation and JIT compilation, which incurs compilation
  overhead on first execution. This can result in higher latency for initial
  operations compared to eager frameworks like NumPy or PyTorch. Subsequent
  executions reuse compiled kernels for better performance.

Create and manipulate tensors with automatic compilation and optimization:

.. code-block:: python

    from max.experimental.tensor import Tensor

    # Create and operate on tensors
    x = Tensor.ones((2, 3))
    y = Tensor.zeros_like(x)
    result = x + y  # Eager execution with automatic compilation

Operations may be combined into a single execution graph to take advantage
of automatic kernel fusion:

.. code-block:: python

    from max.experimental import functional as F

    @F.functional
    def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        return x @ weight.T + bias

    # Create and operate on tensors
    x = Tensor.ones([2, 3])
    weight = Tensor.ones([6, 3])
    bias = Tensor.ones([6])

    # Eager execution with a single fused graph
    result = linear(x, weight, bias)

Users may opt in to lazy execution. This is primarily useful for
1. Operations which may never execute, for instance creating modules
with randomly initialized weights before loading weights
2. Combining many operations into a single execution

.. code-block:: python

    from max.experimental.nn import Linear
    from max.driver import CPU
    from max.dtype import DType
    from max.graph import TensorType

    with F.lazy():
        model = Linear(2, 3)

    print(model)  # Lazy weights not initialized

    # Load pretrained weights
    weights = {
        "weight": Tensor.zeros([3, 2]),
        "bias": Tensor.zeros([3]),
    }
    model.load_state_dict(weights)

    # Or compile directly without ever initializing weights
    input_type = TensorType(DType.float32, ["batch", 2], CPU())
    model = model.compile(input_type, weights=weights)
"""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import Generator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

from max import driver, graph
from max.driver import CPU, Accelerator, Device, DLPackArray, accelerator_count
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    DistributedTensorType,
    NamedMapping,
    Placement,
    PlacementMapping,
    Replicated,
    Sharded,
    shard_shape,
)
from max.experimental.support import contextvar_context, driver_tensor_type
from max.graph import (
    DimLike,
    ShapeLike,
    TensorType,
    TensorValueLike,
    ops,
)
from max.graph.ops.constant import NestedArray, Number
from max.graph.value import HasTensorValue
from rich.pretty import pretty_repr

GraphValue: TypeAlias = graph.BufferValue | graph.TensorValue

_CONTEXT: ContextVar[RealizationContext] = ContextVar("_CONTEXT")
_DEFAULT_DEVICE: ContextVar[Device] = ContextVar("_DEFAULT_DEVICE")
_DEFAULT_DTYPE: ContextVar[DType] = ContextVar("_DEFAULT_DTYPE")

current_realization_context = _CONTEXT.get


def realization_context(
    ctx: RealizationContext,
) -> contextlib.AbstractContextManager[RealizationContext]:
    """Sets the current realization context, within a context manager.

    New tensors created within this block will use the given realization
    context to execute.

    See :class:`~max.experimental.tensor.RealizationContext`.

    Args:
        ctx: The realization context to set as the current context.

    Returns:
        A context manager. When the context manager is entered, it will
        set `ctx` as the current realization context. When exited the
        current realization context will be reset to its previous value.
    """
    return contextvar_context(_CONTEXT, ctx)


@dataclass
class RealizationState:
    """State for an unrealized tensor.

    ``values`` is always a tuple of ``GraphValue`` — one entry for an
    unsharded tensor, N entries (one per shard) for a sharded tensor.
    All values live in the same graph and realization context, which
    guarantees atomic realization: all shards compile and execute together.

    See :class:`~max.experimental.tensor.RealizationContext`.
    """

    #: The symbolic value(s) representing the computation backing this tensor.
    #: Always a tuple: length-1 for unsharded, length-N for sharded.
    values: tuple[GraphValue, ...]
    #: The realization context used to create this tensor. This context
    #: is responsible for realizing the tensor to a real value.
    ctx: RealizationContext

    @property
    def num_values(self) -> int:
        """Returns the number of graph values (1 for unsharded, N for sharded)."""
        return len(self.values)

    @property
    def value(self) -> GraphValue:
        """Returns the single graph value. Raises if this is a sharded state."""
        if len(self.values) != 1:
            raise TypeError(
                "Cannot access single value on a sharded RealizationState. "
                f"This state has {len(self.values)} shard values."
            )
        return self.values[0]

    @value.setter
    def value(self, v: GraphValue) -> None:
        if len(self.values) != 1:
            raise TypeError("Cannot set single value on a sharded state.")
        self.values = (v,)


class RealizationContext(
    Protocol, contextlib.AbstractContextManager["RealizationContext"]
):
    """Implements a way to realize unrealized tensors.

    Most users should never have to think about the existence of this type.
    It exists to facilitate optimizations around where and when tensor
    operations are executed.

    - Each tensor is either `real` or associated with a RealizationContext.
    - If a tensor is not `real`, ie. "unrealized", then it is backed by some
      symbolic computation.
    - The RealizationContext is responsible for tracking this symbolic
      computation and "realizing" the tensor (executing the computation and
      backing the tensor with real data) if and when it is asked to do so.
    - A RealizationContext can only realize tensors associated with it.

    RealizationContext abstracts over various semantics of tensor construction.

    **"Eager" execution**: tensors are realized as soon as the realization context
    exits. This is the default behavior.

    This has a huge concrete advantage over eagerly executing one operation
    at a time: by controlling the boundary of where the eager context starts
    and ends, we can give advanced users a tool to _enable fine-grained
    bounds for automatic fusion!

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

    **Graph compilation**: tensors may never be realized.

    This allows tensor operations to be composed with direct usage of
    the Graph API, for instance `Module.compile`, or using `F.*` operations
    in another Graph API usage.

    **Async execution**: Tensors are realized as `async` functions,
    allowing clean integration in async systems like web services.
    """

    # NB: Ideally `graph` should not be required. There are 3 types of context
    #   managers used to manage the active realization context, and they're
    #   all subtly different. This complexity in the implementation is
    #   annoying and can probably be simplified, but works well when
    #   held correctly.
    #   - The "current" realization context ContextVar -- Operations on
    #     tensors are executed within this context.
    #     - Invariant: the "current" realization context should always also
    #       be "active", ie. entered but not exited.
    #   - The realization context as a context manager -- This communicates
    #       to the realization context when it may think about itself
    #       as activated and complete.
    #   - The compute graph associated with a realization context being
    #     `Graph.current`.
    #     - Invariant: the "current" realization context, if any, should
    #       always match Graph.current.
    #     - Complexity: there isn't always a "current" realization context,
    #       in particular when using the Graph API directly. As such
    #       tensors look at `Graph.current` to understand when they may
    #       be passed between realization contexts.
    #: The graph used by the realization context.
    graph: graph.Graph

    async def realize_all(self) -> list[Tensor]:
        """Realizes all unrealized tensors associated with this context."""

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a realized tensor as a source of the realization state.

        The source is one on whose values unrealized tensors depend.

        Args:
            tensor: The realized tensor to add as a source to the computation.

        Returns:
            A realization state for the tensor. This may be used to compute
            downstream unrealized values. _If it is used in any mutating
            operations, it should be assigned to `tensor.state` to mark
            the tensor as having been mutated.
        """

    def add_mutable_source(self, tensor: Tensor) -> RealizationState:
        """Like ``add_source``, but creates a mutable (BufferType) graph input.

        Used by ``__buffervalue__`` when a tensor needs in-place mutation.
        """

    def create_unrealized(
        self,
        values: tuple[GraphValue, ...],
        *,
        mapping: DeviceMapping | None = None,
        global_shape: Any = None,
    ) -> Tensor:
        """Registers unrealized graph value(s) with the realization context.

        Args:
            values: Per-shard graph values (length-1 for unsharded).
            mapping: Device mapping for distributed tensors.
            global_shape: The global tensor shape for distributed tensors.

        Returns:
            A new tensor associated with the unrealized value(s).
        """


def _default_dtype(device: Device) -> DType:
    if dtype := _DEFAULT_DTYPE.get(None):
        return dtype
    return DType.float32 if isinstance(device, CPU) else DType.bfloat16


def _default_device() -> Device:
    if device := _DEFAULT_DEVICE.get(None):
        return device
    return Accelerator() if accelerator_count() else CPU()


def defaults(
    dtype: DType | None = None, device: Device | None = None
) -> tuple[DType, Device]:
    """Gets the default dtype and device for tensor creation.

    Returns a tuple containing the dtype and device to use for tensor creation,
    applying defaults when values are not specified. If no dtype is provided,
    defaults to :obj:`DType.float32` for CPU and :obj:`DType.bfloat16` for
    accelerators. If no device is provided, defaults to an accelerator if
    available, otherwise CPU.

    Args:
        dtype: The data type to use. If not specified, a default dtype based
            on the device is returned.
        device: The device to use. If not specified, defaults to an available
            accelerator or CPU.

    Returns:
        tuple[DType, Device]: A tuple containing the resolved dtype and device.
    """
    device = device or _default_device()
    return (dtype or _default_dtype(device)), device


def default_device(
    device: Device | graph.DeviceRef,
) -> contextlib.AbstractContextManager[Device]:
    """Context manager for setting the default device for tensor creation.

    Sets the default device used for tensor creation within the context. All
    tensors created inside the context block without an explicit device
    parameter will use this device.

    .. code-block:: python

        from max.experimental import tensor
        from max.driver import CPU

        # Use CPU as default device in this context
        with tensor.default_device(CPU()):
            x = tensor.Tensor.ones((2, 3))  # Created on CPU
            y = tensor.Tensor.zeros((2, 3))  # Also on CPU

    Args:
        device: The device to use as the default for tensor creation within
            the context.

    Returns:
        A context manager that sets the default device.
    """
    if isinstance(device, graph.DeviceRef):
        device = device.to_device()
    return contextvar_context(_DEFAULT_DEVICE, device)


def default_dtype(dtype: DType) -> contextlib.AbstractContextManager[DType]:
    """Context manager for setting the default dtype for tensor creation.

    Sets the default data type used for tensor creation within the context. All
    tensors created inside the context block without an explicit dtype parameter
    will use this data type.

    .. code-block:: python

        from max.experimental import tensor
        from max.dtype import DType

        # Use int32 as default dtype in this context
        with tensor.default_dtype(DType.int32):
            x = tensor.Tensor.ones((2, 3))  # Created with int32
            y = tensor.Tensor.zeros((2, 3))  # Also int32

    Args:
        dtype: The data type to use as the default for tensor creation within
            the context.

    Returns:
        A context manager that sets the default dtype.
    """
    return contextvar_context(_DEFAULT_DTYPE, dtype)


@contextlib.contextmanager
def defaults_like(like: Tensor | TensorType) -> Generator[None]:
    """Context manager setting the default dtype and device for tensor creation.

    Sets the default data type and device used for tensor creation within the
    context. All tensors created inside the context block without explicit
    dtypes or devices will use these parameters.

    .. code-block:: python

        from max.experimental import tensor
        from max.driver import CPU
        from max.dtype import DType

        x = Tensor.zeros([1], dtype=DType.int32, device=CPU())
        # Use int32 as default dtype in this context
        with tensor.defaults_like(x):
            y = tensor.Tensor.zeros((2, 3))  # int32, cpu
            z = tensor.Tensor.zeros((2, 3), dtype=DType.float32)  # float32, cpu

    Args:
        like: Tensor or tensor type whose dtype and device to use as defaults.

    Returns:
        A context manager that sets the default dtype and device.
    """
    with default_dtype(like.dtype), default_device(like.device):
        yield


class Tensor(DLPackArray, HasTensorValue):
    """A multi-dimensional array of numeric values on a CPU or accelerator device.

    You can create tensors using:

    - The :class:`Tensor` constructor.
    - Factory methods like :meth:`ones`, :meth:`zeros`, or :meth:`arange`.
    - Other array libraries via :meth:`from_dlpack`.

    Tensors support the DLPack protocol for zero-copy data exchange with
    NumPy, PyTorch, JAX, and other array libraries.

    .. code-block:: python

        import numpy as np
        from max.experimental.tensor import Tensor
        from max.dtype import DType

        # Create from a Python scalar or nested list
        x = Tensor(42, dtype=DType.int32)
        y = Tensor([[1.0, 2.0], [3.0, 4.0]])

        # Create from any DLPack-compatible array; dtype is inherited
        z = Tensor(np.array([1, 2, 3], dtype=np.int16))

        # Use factory methods like ones, zeros, arange
        zeros = Tensor.zeros((2, 3))

        # Compute with Python operators or the functional API
        result = y + zeros
        print(result)

    A tensor is either *realized* (backed by a concrete
    :class:`~max.driver.Buffer` in memory) or *unrealized* (backed by a
    symbolic graph value). MAX realizes tensors by running pre-compiled Mojo
    kernels or JIT-compiled graphs.

    Args:
        data: The value for the tensor. Can be a scalar number, a nested
            Python list, or any DLPack-compatible array (NumPy, PyTorch,
            etc.). If not provided, exactly one of ``storage`` or ``state``
            must be supplied.
        dtype: The data type for the tensor elements. For DLPack arrays this
            defaults to the array's own dtype; passing a conflicting value
            raises :exc:`ValueError`. For Python scalars and lists, defaults
            to :obj:`DType.float32` on CPU and :obj:`DType.bfloat16` on
            accelerators.
        device: The device where the tensor will be allocated. Defaults to
            an accelerator if available, otherwise CPU. Only valid when
            ``data`` is provided.
        storage: Internal backing buffer for a realized tensor. Mutually
            exclusive with ``data``.
        state: Internal realization state for an unrealized tensor. Mutually
            exclusive with ``data``.
    """

    # ─── Internal storage ──────────────────────────────────────────────
    # For an unsharded tensor exactly one of _storages[0] / _state is
    # set.  For a sharded tensor _storages has one entry per shard but
    # _state is still singular (one RealizationContext for all shards).
    # The public ``storage`` / ``state`` properties provide
    # backward-compatible access for the unsharded case and raise for
    # sharded tensors.

    _storages: tuple[driver.Buffer, ...] | None
    _state: RealizationState | None

    # ─── Device mapping (always set) ────────────────────────────────────
    _mapping: DeviceMapping
    _global_shape: graph.Shape | None  # always set for distributed tensors

    # ─── Placement helpers ────────────────────────────────────────────────

    @property
    def mapping(self) -> DeviceMapping:
        """Returns the device mapping describing where this tensor lives."""
        return self._mapping

    @property
    def is_distributed(self) -> bool:
        """Returns ``True`` if this tensor spans multiple devices."""
        return self._mapping.mesh.num_devices > 1

    @property
    def mesh(self) -> DeviceMesh:
        """Returns the device mesh."""
        return self._mapping.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """Returns per-axis placement descriptors.

        For :class:`~max.experimental.sharding.NamedMapping`,
        this converts to placements on the fly.  Raises
        :class:`~max.experimental.sharding.ConversionError`
        if the spec contains compiler-only annotations.
        """
        return self._mapping.to_placements()

    @property
    def num_shards(self) -> int:
        """Returns the number of shards (1 for an unsharded tensor)."""
        if self._storages is not None:
            return len(self._storages)
        if self._state is not None:
            return self._state.num_values
        raise TypeError("Tensor has no storage and no state.")

    @property
    def local_shards(self) -> tuple[Tensor, ...]:
        """Returns per-device shard views as independent unsharded Tensors.

        Each returned Tensor is a lightweight, standalone, unsharded Tensor
        backed by a single shard's storage or graph value.  They can be
        passed directly to ``F.*`` ops or used as ``Module`` parameters.

        For realized sharded tensors, each shard wraps one ``driver.Buffer``.
        For unrealized sharded tensors, each shard wraps one ``GraphValue``
        from the shared ``RealizationState``.
        For unsharded tensors, returns a 1-tuple containing ``self``.
        """
        if not self.is_distributed:
            return (self,)

        if self._storages is not None:
            # Realized: wrap each buffer as an unsharded Tensor.
            return tuple(Tensor(storage=buf) for buf in self._storages)

        # Unrealized: wrap each graph value as an unsharded Tensor,
        # all sharing the same realization context.
        assert self._state is not None
        ctx = self._state.ctx
        return tuple(ctx.create_unrealized((v,)) for v in self._state.values)

    @property
    def graph_values(self) -> tuple[GraphValue, ...]:
        """Returns per-shard graph values directly from the realization state.

        For unrealized tensors (both distributed and single-device), returns
        the underlying ``GraphValue``s (``TensorValue | BufferValue``) without
        wrapping in intermediate Tensor objects.

        For realized tensors, creates graph values via ``__tensorvalue__()``
        on each shard.

        This is the primary way to access graph-level shard values for
        custom dispatch rules and SPMD loops.
        """
        if self._state is not None:
            return self._state.values

        return tuple(s.__tensorvalue__() for s in self.local_shards)

    def _check_not_distributed(self, op: str) -> None:
        """Raises if this tensor is sharded."""
        if self.is_distributed:
            raise ValueError(
                f"Cannot call {op!r} on a sharded tensor distributed over "
                f"{self._mapping}. Use per-shard access or transfer_to first."
            )

    # ─── Backward-compatible singular storage/state properties ───────

    @property
    def storage(self) -> driver.Buffer | None:
        """Returns the single backing buffer (unsharded tensors only)."""
        self._check_not_distributed("storage")
        if self._storages is None:
            return None
        return self._storages[0]

    @storage.setter
    def storage(self, value: driver.Buffer | None) -> None:
        self._check_not_distributed("storage")
        self._storages = (value,) if value is not None else None

    @property
    def state(self) -> RealizationState | None:
        """Returns the realization state (unsharded tensors only)."""
        self._check_not_distributed("state")
        return self._state

    @state.setter
    def state(self, value: RealizationState | None) -> None:
        self._check_not_distributed("state")
        self._state = value

    # ─── Construction ────────────────────────────────────────────────────

    def __new__(
        cls,
        data: DLPackArray | NestedArray | Number | None = None,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        storage: driver.Buffer | None = None,
        state: RealizationState | None = None,
    ) -> Tensor:
        """Allocates the tensor, delegating to ``F.constant`` when data is given.

        When ``data`` is provided, returns the tensor produced by
        ``F.constant`` directly so that lazy/eager realization contexts track
        the correct object.  For internal construction (``storage`` or
        ``state``), falls through to the normal allocation path.
        """
        if data is not None:
            if storage is not None or state is not None:
                raise TypeError(
                    "Cannot supply both 'data' and internal 'storage'/'state'."
                )
            if isinstance(data, DLPackArray):
                # Preserve the array's own dtype/device by default so that
                # round-tripping (e.g. torch bfloat16 → Tensor) never silently
                # casts.  The user can still supply explicit dtype/device to
                # override.  ops.constant will raise a clear error if the
                # explicit dtype conflicts with the array's dtype.
                resolved_device = device or _default_device()
                return F.constant(data, dtype, resolved_device)
            else:
                # Python scalars and nested lists carry no dtype information,
                # so we always resolve from defaults.
                resolved_dtype, resolved_device = defaults(dtype, device)
                return F.constant(data, resolved_dtype, resolved_device)
        return super().__new__(cls)

    def __init__(
        self,
        data: DLPackArray | NestedArray | Number | None = None,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        storage: driver.Buffer | None = None,
        state: RealizationState | None = None,
    ):
        if data is not None:
            # __new__ already returned the tensor produced by F.constant;
            # __init__ is invoked on that object but nothing remains to do.
            return
        if dtype is not None or device is not None:
            raise TypeError(
                "'dtype' and 'device' are only valid when 'data' is provided."
            )
        if (storage is None) == (state is None):
            raise TypeError("Must supply exactly one of 'storage' and 'state'.")
        # Single-device tensor: single-element storage tuple, trivial mapping.
        self._storages = (storage,) if storage is not None else None
        self._state = state
        if storage is not None:
            device = storage.device
        else:
            assert state is not None
            dev = state.value.device
            device = dev if isinstance(dev, Device) else dev.to_device()
        self._mapping = PlacementMapping(
            DeviceMesh.single(device), (Replicated(),)
        )
        self._global_shape = None

    @classmethod
    def from_graph_value(cls, value: graph.Value[Any]) -> Tensor:
        """Creates a tensor from a graph value.

        Constructs a tensor from an existing graph value, which can be either
        a :obj:`~max.graph.TensorValue` or :obj:`~max.graph.BufferValue`. This
        is used for converting graph level values into tensor objects.
        The new tensor is registered as unrealized, backed by the current
        realization context.

        Args:
            value: The graph value to wrap. Can be either a TensorValue or
                BufferValue from the MAX graph API.

        Returns:
            Tensor: A new tensor backed by the provided graph value.
        """
        if not isinstance(value, GraphValue):
            raise TypeError(f"{value=} must be a tensor or buffer value")
        return current_realization_context().create_unrealized((value,))

    @classmethod
    def from_shard_values(
        cls,
        shard_values: Sequence[GraphValue],
        mapping: DeviceMapping | None = None,
    ) -> Tensor:
        """Creates a tensor from one or more per-shard graph values.

        For a single shard value with no mapping, behaves like
        :meth:`from_graph_value`. For multiple shard values, a
        :class:`~max.experimental.sharding.DeviceMapping` is required
        and the result is a distributed tensor.

        Args:
            shard_values: Per-device graph values (TensorValue or
                BufferValue). One per device in the mesh.
            mapping: Device mapping describing how shards map to mesh
                devices and their placements. Required when
                ``len(shard_values) > 1``.

        Returns:
            A tensor backed by the provided shard values.

        Raises:
            ValueError: If multiple shard values are given without a mapping.
            TypeError: If any shard value is not a graph value.
        """
        if len(shard_values) > 1 and mapping is None:
            raise ValueError(
                "DeviceMapping is required when providing multiple "
                "shard values. Pass a PlacementMapping describing how "
                "shards map to mesh devices."
            )
        for v in shard_values:
            if not isinstance(v, GraphValue):
                raise TypeError(f"{v=} must be a tensor or buffer value")
        if mapping is None:
            return current_realization_context().create_unrealized(
                (shard_values[0],)
            )
        return current_realization_context().create_unrealized(
            tuple(shard_values), mapping=mapping
        )

    @classmethod
    def from_dlpack(cls, array: DLPackArray) -> Tensor:
        """Creates a tensor from a DLPack array.

        Constructs a tensor by importing data from any object that supports
        the DLPack protocol (such as NumPy arrays and PyTorch tensors).
        This enables zero-copy interoperability with other array libraries.

        .. code-block:: python

            import numpy as np
            from max.experimental import tensor

            # Create a NumPy array
            np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

            # Convert to MAX tensor via DLPack
            x = tensor.Tensor.from_dlpack(np_array)

        Args:
            array: Any object supporting the DLPack protocol, such as NumPy
                arrays, PyTorch tensors, or JAX arrays.

        Returns:
            Tensor: A new tensor containing the data from the DLPack array.
        """
        if isinstance(array, Tensor):
            return array
        return Tensor(storage=driver.Buffer.from_dlpack(array))

    @classmethod
    def _from_shards(
        cls,
        storages: tuple[driver.Buffer, ...],
        mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        global_shape: graph.ShapeLike,
    ) -> Tensor:
        """Creates a realized sharded tensor from per-device buffers.

        This is an internal constructor. ``storages`` must have one entry per
        device in the mesh, in row-major order.  All shards are realized
        (concrete storage, no pending graph values).
        """
        if len(storages) != mesh.num_devices:
            raise ValueError(
                f"Expected {mesh.num_devices} storages for mesh {mesh}, "
                f"got {len(storages)}."
            )
        if len(placements) != mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis ({mesh.ndim}), "
                f"got {len(placements)}."
            )
        instance = object.__new__(cls)
        instance._storages = storages
        instance._state = None
        instance._mapping = PlacementMapping(mesh, placements)
        instance._global_shape = graph.Shape(global_shape)
        return instance

    @classmethod
    def _from_unrealized_shards(
        cls,
        state: RealizationState,
        mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        global_shape: graph.ShapeLike | None = None,
    ) -> Tensor:
        """Creates an unrealized sharded tensor from a single state.

        ``state.values`` must have one entry per shard — all in the same
        graph.  Realization is atomic: all shards compile and execute
        together.  If ``global_shape`` is omitted, it is derived from the
        first shard's shape, placements, and mesh.
        """
        if len(state.values) != mesh.num_devices:
            raise ValueError(
                f"Expected {mesh.num_devices} shard values for mesh {mesh}, "
                f"got {len(state.values)}."
            )
        if len(placements) != mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis ({mesh.ndim}), "
                f"got {len(placements)}."
            )
        instance = object.__new__(cls)
        instance._storages = None
        instance._state = state
        instance._mapping = PlacementMapping(mesh, placements)
        instance._global_shape = (
            graph.Shape(global_shape) if global_shape is not None else None
        )
        return instance

    def _as_constant_external(self, name: str) -> Tensor:
        """Creates graph external constant(s) matching ``self``'s layout.

        For unsharded tensors, creates a single ``constant_external`` and
        transfers it to ``self.device``.  For sharded tensors, creates one
        ``constant_external`` per shard and assembles them into a sharded
        Tensor preserving ``self``'s mesh, placements, and global shape.

        Shard constants are named ``name._shard.0``, ``name._shard.1``, etc.
        """
        if not self.is_distributed:
            stype = TensorType(self.dtype, self.shape, CPU())
            return F.constant_external(name, stype).to(self.device)
        assert self._mapping is not None
        _mesh = self._mapping.mesh
        _placements = self._mapping.to_placements()
        local = shard_shape(self.shape, _placements, _mesh.mesh_shape)
        values = []
        for i in range(_mesh.num_devices):
            stype = TensorType(self.dtype, local, CPU())
            t = F.constant_external(f"{name}._shard.{i}", stype)
            t = t.to(_mesh.devices[i])
            values.append(t._graph_value)
        return current_realization_context().create_unrealized(
            tuple(values),
            mapping=self._mapping,
            global_shape=self.shape,
        )

    def _from_buffers_like(self, buffers: Sequence[driver.Buffer]) -> Tensor:
        """Reconstructs a Tensor from flat result buffers.

        Uses ``self`` as a sharding template.
        For unsharded tensors, wraps ``buffers[0]`` as a plain Tensor.
        For sharded tensors, wraps all buffers into a sharded Tensor
        preserving ``self``'s mesh, placements, and global shape.
        """
        if not self.is_distributed:
            return Tensor(storage=buffers[0])
        assert self._mapping is not None
        return Tensor._from_shards(
            tuple(buffers),
            self._mapping.mesh,
            self._mapping.to_placements(),
            self.shape,
        )

    @classmethod
    def constant(
        cls,
        value: DLPackArray | NestedArray | Number,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Creates a tensor from a scalar, array, or nested list.

        .. deprecated::
            Use ``Tensor(value, dtype=dtype, device=device)`` instead.
            ``Tensor.constant`` will be removed in a future release.

        Args:
            value: The constant value for the tensor. Can be a scalar number,
                a nested Python list, or any DLPack-compatible array.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device where the tensor will be allocated. If not
                specified, defaults to an accelerator if available, otherwise CPU.

        Returns:
            Tensor: A new tensor containing the constant value(s).
        """
        warnings.warn(
            "Tensor.constant() is deprecated. Use Tensor(value, dtype=dtype,"
            " device=device) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(value, dtype=dtype, device=device)

    @classmethod
    def full(
        cls,
        shape: ShapeLike,
        value: Number,
        *,
        dtype: DType | None = None,
        device: Device | DeviceMapping | None = None,
    ) -> Tensor:
        """Creates a tensor filled with a specified value.

        Returns a new tensor with the given shape where all elements are
        initialized to the specified value. This is useful for creating
        tensors with uniform values other than zero or one.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a 3x3 tensor filled with 7
            x = tensor.Tensor.full((3, 3), value=7, dtype=DType.int32)

            # Create a 2x4 tensor filled with pi
            y = tensor.Tensor.full((2, 4), value=3.14159)

        Args:
            shape: The shape of the output tensor. Can be a tuple of integers,
                a list of integers, or any value that can be converted to a shape.
            value: The scalar value to fill the tensor with.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device or device mapping where the tensor will be
                allocated. If not specified, defaults to an accelerator if
                available, otherwise CPU. Pass a
                :class:`~max.experimental.sharding.DeviceMapping` to create
                a distributed tensor.

        Returns:
            Tensor: A new tensor with the specified shape filled with the given value.
        """
        return F.full(shape, value, dtype=dtype, device=device)

    @classmethod
    def full_like(cls, input: Tensor | TensorType, value: Number) -> Tensor:
        """Creates a tensor filled with a value, matching a given tensor's properties.

        Returns a new tensor filled with the specified value that matches the
        shape, data type, and device of the input tensor. This behaves like
        NumPy's ``full_like`` and PyTorch's ``full_like``.

        .. code-block:: python

            from max.experimental import tensor

            # Create a reference tensor
            ref = tensor.Tensor.ones([2, 3])

            # Create tensor filled with 5.0 matching the reference tensor
            x = tensor.Tensor.full_like(ref, value=5.0)

        Args:
            input: The tensor or tensor type to match. The returned tensor will
                have the same shape, dtype, and device as this input.
            value: The scalar value to fill the tensor with.

        Returns:
            Tensor: A new tensor filled with the specified value, matching the
                properties of the input.
        """
        tensor_type = input.type if isinstance(input, Tensor) else input
        return cls.full(
            tensor_type.shape,
            value=value,
            dtype=tensor_type.dtype,
            device=tensor_type.device.to_device(),
        )

    @classmethod
    def zeros(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | DeviceMapping | None = None,
    ) -> Tensor:
        """Creates a tensor filled with zeros.

        Returns a new tensor with the specified shape where all elements are
        initialized to zero. The tensor is created with eager execution and
        automatic compilation.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x3 tensor of zeros
            x = tensor.Tensor.zeros((2, 3))
            # Result: [[0.0, 0.0, 0.0],
            #          [0.0, 0.0, 0.0]]

            # Create a 1D tensor using default dtype and device
            y = tensor.Tensor.zeros((5,))

        Args:
            shape: The shape of the output tensor. Can be a tuple of integers,
                a list of integers, or any value that can be converted to a shape.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device or device mapping where the tensor will be
                allocated. If not specified, defaults to an accelerator if
                available, otherwise CPU.

        Returns:
            Tensor: A new tensor with the specified shape filled with zeros.
        """
        return cls.full(shape, value=0, dtype=dtype, device=device)

    @classmethod
    def zeros_like(cls, input: Tensor | TensorType) -> Tensor:
        """Creates a tensor of zeros matching a given tensor's properties.

        Returns a new tensor filled with zeros that matches the shape, data type,
        and device of the input tensor. This behaves like NumPy's ``zeros_like``
        and PyTorch's ``zeros_like``.

        .. code-block:: python

            from max.experimental import tensor

            # Create a reference tensor
            ref = tensor.Tensor.ones([3, 4])

            # Create zeros tensor matching the reference tensor
            x = tensor.Tensor.zeros_like(ref)
            # Result: 3x4 tensor of zeros with dtype float32

        Args:
            input: The tensor or tensor type to match. The returned tensor will
                have the same shape, dtype, and device as this input.

        Returns:
            Tensor: A new tensor filled with zeros matching the properties of the
                input.
        """
        tensor_type = input.type if isinstance(input, Tensor) else input
        return cls.zeros(
            tensor_type.shape,
            dtype=tensor_type.dtype,
            device=tensor_type.device.to_device(),
        )

    @classmethod
    def ones(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | DeviceMapping | None = None,
    ) -> Tensor:
        """Creates a tensor filled with ones.

        Returns a new tensor with the specified shape where all elements are
        initialized to one.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x3 tensor of ones
            x = tensor.Tensor.ones((2, 3))

        Args:
            shape: The shape of the output tensor.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device or device mapping where the tensor will be
                allocated. If not specified, defaults to an accelerator if
                available, otherwise CPU.

        Returns:
            Tensor: A new tensor with the specified shape filled with ones.
        """
        return cls.full(shape, value=1, dtype=dtype, device=device)

    @classmethod
    def ones_like(cls, input: Tensor | TensorType) -> Tensor:
        """Creates a tensor of ones matching a given tensor's properties.

        Returns a new tensor filled with ones that matches the shape, data type,
        and device of the input tensor. This behaves like NumPy's ``ones_like``
        and PyTorch's ``ones_like``.

        .. code-block:: python

            from max.experimental import tensor

            # Create a reference tensor
            ref = tensor.Tensor.zeros([3, 4])

            # Create ones tensor matching the reference tensor
            x = tensor.Tensor.ones_like(ref)
            # Result: 3x4 tensor of ones with dtype float32

        Args:
            input: The tensor or tensor type to match. The returned tensor will
                have the same shape, dtype, and device as this input.

        Returns:
            Tensor: A new tensor filled with ones matching the properties of the
                input.
        """
        tensor_type = input.type if isinstance(input, Tensor) else input
        return cls.ones(
            tensor_type.shape,
            dtype=tensor_type.dtype,
            device=tensor_type.device.to_device(),
        )

    @classmethod
    def arange(
        cls,
        start: TensorValueLike = 0,
        stop: TensorValueLike | None = None,
        step: TensorValueLike = 1,
        out_dim: DimLike | None = None,
        *,
        dtype: DType | None = None,
        device: Device | DeviceMapping | None = None,
    ) -> Tensor:
        """Creates a tensor with evenly spaced values within a given interval.

        Returns a new 1D tensor containing a sequence of values starting from
        ``start`` (inclusive) and ending before ``stop`` (exclusive), with values
        spaced by ``step``. This is similar to Python's built-in ``range()``
        function and NumPy's ``arange()``.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a range from 0 to 10 (exclusive)
            x = tensor.Tensor.arange(10)
            # Result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            # Create a range from 5 to 15 with step 2
            y = tensor.Tensor.arange(5, 15, 2)
            # Result: [5, 7, 9, 11, 13]

            # Use a specific dtype
            z = tensor.Tensor.arange(0, 5, dtype=DType.float32)
            # Result: [0.0, 1.0, 2.0, 3.0, 4.0]

            # Create a range with float step (like numpy/pytorch)
            w = tensor.Tensor.arange(0.0, 1.0, 0.2)
            # Result: [0.0, 0.2, 0.4, 0.6, 0.8]

            # Create a descending range with negative step
            v = tensor.Tensor.arange(5, 0, -1, dtype=DType.float32)
            # Result: [5.0, 4.0, 3.0, 2.0, 1.0]

        Args:
            start: The starting value of the sequence. If ``stop`` is not provided,
                this becomes the ``stop`` value and ``start`` defaults to 0.
            stop: The end value of the sequence (exclusive). If not specified,
                the sequence ends at ``start`` and begins at 0.
            step: The spacing between values in the sequence. Must be non-zero.
            out_dim: The expected output dimension. Required when ``start``,
                ``stop``, or ``step`` are tensors rather than scalar literals.
                If not specified, the output dimension is computed from the
                scalar values of the inputs.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device where the tensor will be allocated. If not
                specified, defaults to an accelerator if available, otherwise CPU.

        Returns:
            Tensor: A 1D tensor containing the evenly spaced values.
        """
        if stop is None:
            start, stop = 0, start
        return F.arange(
            start,
            stop,
            step,
            out_dim,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def range_like(cls, type: TensorType) -> Tensor:
        """Creates a range tensor matching a given type's properties.

        Returns a new tensor containing sequential indices along the last
        dimension, broadcasted to match the shape of the specified tensor type.
        Each row (along the last dimension) contains values from 0 to the
        dimension size minus one. This is useful for creating position indices
        or coordinate tensors.

        .. code-block:: python

            from max.experimental import tensor
            from max.graph import TensorType
            from max.dtype import DType

            # Create a reference tensor type with shape (2, 4)
            ref_type = TensorType(DType.int32, (2, 4))

            # Create range tensor matching the reference type
            x = tensor.Tensor.range_like(ref_type)
            # Result: [[0, 1, 2, 3],
            #          [0, 1, 2, 3]]

        Args:
            type: The tensor type to match. The returned tensor will have the
                same shape, dtype, and device as this type, with values
                representing indices along the last dimension.

        Returns:
            Tensor: A new tensor with sequential indices broadcasted to match
                the input type's shape.
        """
        dim = type.shape[-1]
        range = F.arange(
            start=0,
            stop=dim,
            out_dim=dim,
            dtype=type.dtype,
            device=type.device.to_device(),
        )
        return F.broadcast_to(range, type.shape)

    @property
    def real(self) -> bool:
        """Returns ``True`` if this tensor is realized (has concrete storage).

        For sharded tensors this is all-or-nothing: either every shard is
        realized (``_state is None``) or none are.
        """
        return self._state is None

    @property
    def _backing_value(self) -> driver.Buffer | GraphValue:
        self._check_not_distributed("_backing_value")
        return self.driver_tensor if self.real else self._graph_value

    @property
    def _graph_value(self) -> GraphValue:
        self._check_not_distributed("_graph_value")
        if self.real:
            raise TypeError("Can't get symbolic value for real tensor.")
        assert self._state
        return self._state.value

    @property
    def driver_tensor(self) -> driver.Buffer:
        """A pointer to the underlying memory.

        Raises if the tensor is unrealized or sharded.
        """
        self._check_not_distributed("driver_tensor")
        if self._storages is None:
            raise TypeError("Can't get driver tensor for symbolic tensor")
        return self._storages[0]

    @property
    def buffers(self) -> tuple[driver.Buffer, ...]:
        """The underlying per-shard driver buffers.

        Returns one buffer for non-distributed tensors, N buffers for
        a distributed tensor with N shards.

        Raises:
            TypeError: If the tensor is unrealized (lazy/symbolic).
        """
        if self._storages is None:
            raise TypeError(
                "Can't get buffers for unrealized tensor — "
                "realize the tensor first."
            )
        return self._storages

    @property
    def type(self) -> graph.TensorType:
        """Gets the tensor type information.

        Returns:
            TensorType: The type information for the tensor.

        Raises:
            TypeError: If the tensor is distributed.
        """
        if self.is_distributed:
            dist_type = DistributedTensorType(
                self.dtype, self.shape, self.mesh, self.placements
            )
            raise TypeError(
                f"Cannot get a single TensorType for a distributed tensor. "
                f"The distributed type is: {dist_type!r}. "
                f"This API may change in the future to return "
                f"DistributedTensorType directly."
            )
        type = (
            driver_tensor_type(self.driver_tensor)
            if self.real
            else self._graph_value.type
        )
        return type.as_tensor() if isinstance(type, graph.BufferType) else type

    @property
    def rank(self) -> int:
        """Gets the number of dimensions in the tensor.

        Returns the rank (number of dimensions) of the tensor. For example,
        a scalar has rank 0, a vector has rank 1, and a matrix has rank 2.

        Returns:
            int: The number of dimensions in the tensor.
        """
        return len(self.shape)

    @property
    def shape(self) -> graph.Shape:
        """Gets the global shape of the tensor.

        For sharded tensors this returns the logical global shape (not the
        per-shard shape).  If no explicit global shape was set, it is
        derived from the first shard's shape, placements, and mesh.

        Returns:
            Shape: The shape of the tensor.
        """
        if self._global_shape is not None:
            return self._global_shape
        if self.is_distributed:
            # Get the shard shape from the first shard.
            if self._storages is not None:
                shard_shape = list(self._storages[0].shape)
            else:
                assert self._state is not None
                sv = self._state.values[0]
                shard_shape = list(sv.shape)
            # Scale sharded dims back up by mesh size.
            assert self._mapping is not None
            _placements = self._mapping.to_placements()
            _mesh = self._mapping.mesh
            for ax, p in enumerate(_placements):
                if isinstance(p, Sharded):
                    d = p.axis % len(shard_shape)
                    shard_shape[d] = shard_shape[d] * _mesh.mesh_shape[ax]
            return graph.Shape(shard_shape)
        shape = self._backing_value.shape
        return shape if isinstance(shape, graph.Shape) else graph.Shape(shape)

    @property
    def dtype(self) -> DType:
        """Gets the data type of the tensor elements.

        Returns:
            DType: The data type of the tensor elements.
        """
        if self._storages is not None:
            return self._storages[0].dtype
        assert self._state is not None
        return self._state.values[0].dtype

    @property
    def device(self) -> Device:
        """Gets the device where the tensor is stored.

        Returns the device (CPU or accelerator) where the tensor's data is
        located.  Raises for distributed tensors that span multiple devices.

        Returns:
            Device: The device where the tensor is stored.
        """
        if self.is_distributed:
            raise ValueError(
                f"Cannot access single device on a distributed tensor "
                f"spanning {self._mapping.mesh.num_devices} devices. "
                f"Use tensor.mesh.devices instead."
            )
        return self._mapping.mesh.devices[0]

    def __await__(self):
        """Force the tensor to realize if it is not already."""
        self._check_not_distributed("__await__")
        if not self.real:
            assert self._state is not None
            yield from asyncio.create_task(self._state.ctx.realize_all())
            assert self.real
        return self

    @property
    async def realize(self) -> Tensor:
        """Force the tensor to realize if it is not already."""
        return await self

    def _sync_realize(self) -> None:
        if not self.real:
            F._run(self.realize)

    def __tensorvalue__(self) -> graph.TensorValue:
        """Gets a TensorValue for the underlying data.

        If the tensor is backed by a BufferValue, calls `ops.buffer_load`.
        The load is for ordering mutable operations and will be optimized away.
        """
        self._check_not_distributed("__tensorvalue__")
        if not self.real:
            assert self._state
            if graph.Graph.current != self._state.ctx.graph:
                # Can't pass unrealized tensors between graphs
                self._sync_realize()

        if self.real:
            state = current_realization_context().add_source(self)
            value = state.value
        else:
            assert self._state
            value = self._state.value

        if isinstance(value, graph.BufferValue):
            return value[...]
        assert isinstance(value, graph.TensorValue)
        return value

    def __buffervalue__(self) -> graph.BufferValue:
        """Gets a BufferValue for the underlying data.

        Afterwards this tensor will always be unrealized. Assume that
        the resulting BufferValue is passed into a staged mutating op,
        and the backing data is not accurate until the graph has executed.

        If self is backed by a TensorValue
            - create a new BufferValue via `ops.buffer_create` and
            `ops.buffer_store` containing the same data
            - `self` is updated to be backed by the new BufferValue
            - further ops on the same tensor will then load from the
            buffer to ensure proper sequencing with mutation
        """
        self._check_not_distributed("__buffervalue__")
        if not self.real:
            assert self._state
            if graph.Graph.current != self._state.ctx.graph:
                # Can't pass unrealized tensors between graphs
                self._sync_realize()

        if self.real:
            # This is a realized tensor that needs a mutable graph input so
            # the BufferValue can be stored into in-place. add_mutable_source
            # creates a BufferType input (vs add_source's TensorType).
            self._state = current_realization_context().add_mutable_source(self)

        if isinstance(value := self._backing_value, graph.BufferValue):
            return value

        # This tensor is currently backed by an unrealized TensorValue.
        # Create a BufferValue and assign the current value to it
        tensor = self.__tensorvalue__()
        assert self._state is not None
        self._state.value = buffer = ops.buffer_create(tensor.type.as_buffer())
        buffer[...] = tensor
        return buffer

    def __bool__(self) -> bool:
        self._check_not_distributed("__bool__")
        return bool(self.item())

    def _values(self) -> Generator[Any]:
        self._check_not_distributed("_values")
        self._sync_realize()
        dt = self.driver_tensor.to(CPU())
        for idx in dt._iterate_indices():
            yield dt[idx].item()

    def __hash__(self):
        return id(self)

    def __dlpack__(self, stream: int | None = None):
        self._check_not_distributed("__dlpack__")
        self._sync_realize()
        assert self._storages is not None
        return self._storages[0].__dlpack__(stream=stream)

    def __dlpack_device__(self):
        self._check_not_distributed("__dlpack_device__")
        self._sync_realize()
        assert self._storages is not None
        return self._storages[0].__dlpack_device__()

    def __rich_repr__(self):
        yield "<unrealized>"
        yield "shape", self.shape
        yield "dtype", self.dtype
        yield "device", self.device

    def __repr__(self) -> str:
        """Returns a formatted string representation of the tensor.

        For realized tensors, displays the data using a matrix-of-matrices
        algorithm that preserves the multi-dimensional structure.
        For unrealized tensors, shows shape, dtype, and device information.
        For sharded tensors, shows global shape, dtype, mesh, and placements.

        Returns:
            A string representation of the tensor.
        """
        if self.is_distributed:
            shape_str = ", ".join(str(d) for d in self.shape)
            return (
                f"Tensor(shape=[{shape_str}], dtype={self.dtype}, "
                f"mapping={self._mapping!r})"
            )
        if self.real:
            from max.experimental import _tensor_repr

            return _tensor_repr.render(self)
        return pretty_repr(self)

    def __deepcopy__(self, memo: object) -> Tensor:
        # Tensors are value-semantic
        return self

    def item(self) -> Any:
        """Gets the scalar value from a single-element tensor.

        Extracts and returns the scalar value from a tensor containing exactly
        one element. The tensor is realized if needed and transferred to CPU
        before extracting the value.

        For replicated distributed tensors, the value is read from the first
        shard (all shards hold identical data).

        Returns:
            The scalar value from the tensor. The return type matches the tensor's
            dtype (e.g., float for float32, int for int32).

        Raises:
            TypeError: If the tensor contains more than one element.
            ValueError: If the tensor is distributed and not fully replicated.
        """
        if self.is_distributed:
            if not self._mapping.is_fully_replicated:
                # Reuse the standard error for non-replicated distributed
                # tensors (Sharded, Partial, etc.).
                self._check_not_distributed("item")
            # All shards are identical — read from the first one.
            self._sync_realize()
            assert self._storages is not None
            return self._storages[0].to(CPU()).item()
        if self.num_elements() != 1:
            raise TypeError()
        self._sync_realize()
        return self.driver_tensor.to(CPU()).item()

    def num_elements(self) -> int:
        """Gets the total number of elements in the tensor.

        Computes the product of all dimensions in the tensor's shape to
        determine the total number of elements.

        Returns:
            int: The total number of elements in the tensor.
        """
        elts = 1
        for dim in self.shape:
            elts *= int(dim)
        return elts

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Tensor:
        """Transfers the tensor to a different device, mesh, or mapping.

        This method supports three target types:

        1. **Device**: Transfers a single-device tensor to the target device.
           For realized tensors, performs a direct driver-level transfer via
           :meth:`~max.driver.Buffer.to`. For unrealized tensors, inserts a
           :func:`~max.graph.ops.transfer_to` op into the computation graph.

        2. **DeviceMapping**: Reassigns the tensor's device mesh and placements.
           For single-device mappings, equivalent to ``.to(device)``.
           For multi-device mappings on an unsharded tensor, distributes the
           tensor across the mesh using the shard collective.

        3. **DeviceMesh**: Replaces the device mesh while keeping existing
           placements. For unsharded tensors targeting a multi-device mesh,
           creates a fully replicated mapping. For distributed tensors,
           transfers shards to the new mesh devices.

        .. code-block:: python

            from max.experimental import tensor
            from max.driver import CPU, Accelerator

            # Create a tensor on CPU
            x = tensor.Tensor.ones((2, 3), device=CPU())
            print(x.device)  # CPU

            # Transfer to accelerator
            y = x.to(Accelerator())
            print(y.device)  # Accelerator(0)

            # Same-device transfer is a no-op
            z = y.to(y.device)
            assert z is y

        Args:
            target: The target for the tensor. Can be:

                - :class:`~max.driver.Device`: Target device for transfer.
                - :class:`~max.experimental.sharding.DeviceMesh`: New mesh,
                  keeping existing placements (or fully replicated for
                  unsharded tensors).
                - :class:`~max.experimental.sharding.DeviceMapping`: New mesh
                  and placements; triggers shard collective for multi-device.

        Returns:
            Tensor: A tensor on the specified target. Returns ``self`` if no
            transfer is needed.
        """
        mapping: DeviceMapping
        if isinstance(target, Device):
            mapping = PlacementMapping(
                DeviceMesh.single(target), self.placements
            )
        elif isinstance(target, DeviceMesh):
            if isinstance(self._mapping, NamedMapping):
                mapping = NamedMapping(target, self._mapping.original_spec)
            else:
                mapping = PlacementMapping(target, self.placements)
        elif isinstance(target, DeviceMapping):
            mapping = target
        else:
            raise TypeError(
                f"to() expects Device, DeviceMesh, or DeviceMapping, "
                f"got {type(target).__name__}"
            )

        return F.transfer_to(self, mapping)

    def materialize(self) -> Tensor:
        """Gather a distributed tensor into a single local tensor.

        Allreduces Partial axes, allgathers Sharded axes, and transfers
        the result to CPU.  Returns ``self`` unchanged for non-distributed
        tensors.
        """
        if not self.is_distributed:
            return self
        return _transfer_to(self, CPU())

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """Convert this tensor to a NumPy array.

        Materializes distributed tensors and transfers to CPU if needed.
        """
        t = _transfer_to(self, CPU()) if self.is_distributed else self
        if t.device != CPU():
            t = t.to(CPU())
        return np.from_dlpack(t)

    def argmax(self, axis: int | None = -1) -> Tensor:
        """Finds the indices of the maximum values along an axis.

        Returns a tensor containing the indices of the maximum values along
        the specified axis. This is useful for finding the position of the
        largest element, such as determining predicted classes in classification.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x4 tensor
            x = tensor.Tensor(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
            )

            # Find argmax along last axis (within each row)
            indices = x.argmax(axis=-1)
            # Result: [1, 2] (index 1 in first row, index 2 in second row)

            # Find argmax over all elements
            index = x.argmax(axis=None)
            # Result: 6 (flattened index of maximum value 4.2)

        Args:
            axis: The axis along which to find the maximum indices. Defaults
                to -1 (the last axis). If None, finds the index of the maximum
                value across all elements.

        Returns:
            Tensor: A tensor containing the indices of the maximum values.
        """
        return F.argmax(self, axis=axis)

    def max(self, axis: int | None = -1) -> Tensor:
        """Computes the maximum values along an axis.

        Returns a tensor containing the maximum values along the specified axis.
        This is useful for reduction operations and finding peak values in data.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x4 tensor
            x = tensor.Tensor(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
            )

            # Find max along last axis (within each row)
            row_max = x.max(axis=-1)
            # Result: [3.5, 4.2]

            # Find max along first axis (within each column)
            col_max = x.max(axis=0)
            # Result: [2.3, 3.5, 4.2, 3.1]

            # Find max over all elements
            overall_max = x.max(axis=None)
            # Result: 4.2 (maximum value across all elements)

        Args:
            axis: The axis along which to compute the maximum. Defaults to -1
                (the last axis). If None, computes the maximum across all elements.

        Returns:
            Tensor: A tensor containing the maximum values along the specified axis.
        """
        return F.max(self, axis=axis)

    def min(self, axis: int | None = -1) -> Tensor:
        """Computes the minimum values along an axis.

        Returns a tensor containing the minimum values along the specified axis.
        This is useful for reduction operations and finding the smallest values
        in data.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x4 tensor
            x = tensor.Tensor(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
            )

            # Find min along last axis (within each row)
            row_min = x.min(axis=-1)
            # Result: [0.8, 1.9]

            # Find min along first axis (within each column)
            col_min = x.min(axis=0)
            # Result: [1.2, 1.9, 2.1, 0.8]

            # Find min over all elements
            overall_min = x.min(axis=None)
            # Result: 0.8 (minimum value across all elements)

        Args:
            axis: The axis along which to compute the minimum. Defaults to -1
                (the last axis). If None, computes the minimum across all elements.

        Returns:
            Tensor: A tensor containing the minimum values along the specified axis.
        """
        return F.min(self, axis=axis)

    def mean(self, axis: int | None = -1) -> Tensor:
        """Computes the mean values along an axis.

        Returns a tensor containing the arithmetic mean of values along the
        specified axis. This is useful for computing averages, normalizing data,
        or aggregating statistics.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x4 tensor
            x = tensor.Tensor(
                [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]],
            )

            # Compute mean along last axis (within each row)
            row_mean = x.mean(axis=-1)
            # Result: [5.0, 4.0] (mean of each row)

            # Compute mean along first axis (within each column)
            col_mean = x.mean(axis=0)
            # Result: [1.5, 3.5, 5.5, 7.5] (mean of each column)

            # Compute mean over all elements
            overall_mean = x.mean(axis=None)
            # Result: 4.5 (mean of all elements)

        Args:
            axis: The axis along which to compute the mean. Defaults to -1
                (the last axis). If None, computes the mean across all elements.

        Returns:
            Tensor: A tensor containing the mean values along the specified axis.
        """
        return F.mean(self, axis=axis)

    def sum(self, axis: int | None = -1) -> Tensor:
        """Computes the sum of values along an axis.

        Returns a tensor containing the sum of values along the specified axis.
        This is a fundamental reduction operation used for aggregating data,
        computing totals, and implementing other operations like mean.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x3 tensor
            x = tensor.Tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )

            # Sum along last axis (within each row)
            row_sum = x.sum(axis=-1)
            # Result: [6.0, 15.0] (sum of each row)

            # Sum along first axis (within each column)
            col_sum = x.sum(axis=0)
            # Result: [5.0, 7.0, 9.0] (sum of each column)

            # Sum over all elements
            total = x.sum(axis=None)
            # Result: 21.0 (sum of all elements)

        Args:
            axis: The axis along which to compute the sum. Defaults to -1
                (the last axis). If None, computes the sum across all elements.

        Returns:
            Tensor: A tensor containing the sum along the specified axis.
        """
        return F.sum(self, axis=axis)

    def prod(self, axis: int | None = -1) -> Tensor:
        """Computes the product of values along an axis.

        Args:
            axis: The axis along which to compute the product. Defaults to -1
                (the last axis). If None, computes the product across all
                elements.

        Returns:
            Tensor: A tensor containing the product along the specified axis.
        """
        return F.prod(self, axis=axis)

    def clip(
        self,
        *,
        min: TensorValueLike | None = None,
        max: TensorValueLike | None = None,
    ) -> Tensor:
        """Clips values outside a range to the boundaries of the range.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 2x4 tensor
            x = tensor.Tensor(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
            )

            # Find max along last axis (within each row)
            clipped_above = x.clip(max=3.)
            # Result: [[1.2, 3., 2.1, 0.8], [2.3, 1.9, 3, 3.]]

            clipped_below = x.clip(min=3.)
            # Result: [[3., 3.5, 3., 3.], [3., 3., 4.2, 3.]]

        Args:
            min: The minimum value of the range. If not specified, do not
                clip values for being too small.
            max: The maximum value of the range. If not specified, do not
                clip values for being too large.

        Returns:
            Tensor: A tensor containing the values clipped to the specified range.
        """
        x: Tensor = self
        if min is not None:
            x = F.max(x, min)
        if max is not None:
            x = F.min(x, max)
        return x

    def squeeze(self, axis: int) -> Tensor:
        """Removes a size-1 dimension from the tensor.

        Returns a tensor with the specified size-1 dimension removed. This is
        useful for removing singleton dimensions from tensors after operations
        that may have added them.

        .. code-block:: python

            from max.experimental import tensor

            # Create a tensor with a size-1 dimension
            x = tensor.Tensor.ones([4, 1, 6])
            print(x.shape)  # (4, 1, 6)

            # Squeeze out the size-1 dimension
            y = x.squeeze(axis=1)
            print(y.shape)  # (4, 6)

        Args:
            axis: The dimension to remove from the tensor's shape. If negative,
                this indexes from the end of the tensor. The dimension at this
                axis must have size 1.

        Returns:
            Tensor: A tensor with the specified dimension removed.

        Raises:
            ValueError: If the dimension at the specified axis is not size 1.
        """
        return F.squeeze(self, axis)

    def unsqueeze(self, axis: int) -> Tensor:
        """Inserts a size-1 dimension into the tensor.

        Returns a tensor with a new size-1 dimension inserted at the specified
        position. This is the inverse of :meth:`squeeze` and is useful for
        adding dimensions needed for broadcasting or matrix operations.

        .. code-block:: python

            from max.experimental import tensor

            # Create a 1D tensor
            x = tensor.Tensor([1.0, 2.0, 3.0])
            print(x.shape)  # (3,)

            # Add dimension at the end
            y = x.unsqueeze(axis=-1)
            print(y.shape)  # (3, 1)

            # Add dimension at the beginning
            z = x.unsqueeze(axis=0)
            print(z.shape)  # (1, 3)

        Args:
            axis: The index at which to insert the new dimension. If negative,
                indexes relative to 1 plus the rank of the tensor. For example,
                ``axis=-1`` adds a dimension at the end.

        Returns:
            Tensor: A tensor with an additional size-1 dimension.
        """
        return F.unsqueeze(self, axis)

    def split(
        self, split_size_or_sections: int | list[int], axis: int = 0
    ) -> list[Tensor]:
        """Splits the tensor into multiple tensors along a given dimension.

        This method supports two modes, matching PyTorch's behavior:

        - If ``split_size_or_sections`` is an **int**, splits into chunks of
          that size (the last chunk may be smaller if not evenly divisible).
        - If ``split_size_or_sections`` is a **list of ints**, splits into
          chunks with exactly those sizes (must sum to the dimension size).

        .. code-block:: python

            from max.experimental import tensor

            # Create a 10x4 tensor
            x = tensor.Tensor.ones([10, 4])

            # Split into chunks of size 3 (last chunk is size 1)
            chunks = x.split(3, axis=0)
            # Result: 4 tensors with shapes [3,4], [3,4], [3,4], [1,4]

            # Split into exact sizes
            chunks = x.split([2, 3, 5], axis=0)
            # Result: 3 tensors with shapes [2,4], [3,4], [5,4]

        Args:
            split_size_or_sections: Either an int (chunk size) or a list of
                ints (exact sizes for each output tensor).
            axis: The dimension along which to split. Defaults to 0.

        Returns:
            list[Tensor]: A list of tensors resulting from the split.
        """
        return cast(list[Tensor], F.split(self, split_size_or_sections, axis))

    def reshape(self, shape: ShapeLike) -> Tensor:
        """Reshapes the tensor to a new shape.

        Returns a tensor with the same data but a different shape. The total
        number of elements must remain the same. This is useful for changing
        tensor dimensions for different operations, such as flattening a
        multi-dimensional tensor or converting a 1D tensor into a matrix.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a 2x3 tensor
            x = tensor.Tensor([[1, 2, 3], [4, 5, 6]], dtype=DType.int32)
            print(x.shape)  # (2, 3)

            # Flatten to 1D
            y = x.reshape((6,))
            print(y.shape)  # (6,)
            # Values: [1, 2, 3, 4, 5, 6]

        Args:
            shape: The desired output shape. Can be a tuple or list of integers.
                The total number of elements must equal the original tensor's
                element count.

        Returns:
            Tensor: A reshaped tensor with the specified shape.
        """
        return F.reshape(self, shape)

    def broadcast_to(self, shape: ShapeLike) -> Tensor:
        """Broadcasts the tensor to the specified shape.

        Returns a tensor broadcast to the target shape, following NumPy
        broadcasting semantics. Dimensions of size 1 in the input can be
        expanded to match larger dimensions in the target shape.

        This is equivalent to PyTorch's :func:`torch.broadcast_to` and
        :meth:`torch.Tensor.expand`.

        .. code-block:: python

            from max.experimental import tensor

            # Create a tensor with shape (3, 1)
            x = tensor.Tensor.ones([3, 1])

            # Broadcast to (3, 4) - expands the second dimension
            y = x.broadcast_to([3, 4])
            print(y.shape)  # (3, 4)

            # Add a new leading dimension
            w = x.broadcast_to([2, 3, 1])
            print(w.shape)  # (2, 3, 1)

        Args:
            shape: The target shape. Each dimension must either match the input
                dimension or be broadcastable from size 1.

        Returns:
            Tensor: A tensor broadcast to the specified shape.
        """
        return F.broadcast_to(self, shape)

    def cast(self, dtype: DType) -> Tensor:
        """Casts the tensor to a different data type.

        Returns a new tensor with the same values but a different data type.
        This is useful for type conversions between different numeric types,
        such as converting ``float32`` to ``int32`` for indexing operations or
        ``float32`` to ``bfloat16`` for memory-efficient computations.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a float32 tensor
            x = tensor.Tensor([1.7, 2.3, 3.9], dtype=DType.float32)
            print(x.dtype)  # DType.float32

            # Cast to int32 (truncates decimal values)
            y = x.cast(DType.int32)
            print(y.dtype)  # DType.int32
            # Values: [1, 2, 3]

        Args:
            dtype: The target data type for the tensor.

        Returns:
            Tensor: A new tensor with the specified data type, or ``self``
            if the tensor already has the target dtype.
        """
        if self.real and self.dtype == dtype:
            return self
        return F.cast(self, dtype)

    def permute(self, dims: list[int]) -> Tensor:
        """Permutes the dimensions of the tensor.

        Returns a tensor with its dimensions reordered according to the
        specified permutation. This is useful for changing the layout of
        multi-dimensional data, such as converting between different tensor
        layout conventions (e.g., from ``[batch, channels, height, width]``
        to ``[batch, height, width, channels]``).

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.dtype import DType

            # Create a 3D tensor (batch_size=2, channels=3, length=4)
            x = Tensor(
                [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                 [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]],
                dtype=DType.int32,
            )
            print(f"Original shape: {x.shape}")
            # Output: Original shape: [Dim(2), Dim(3), Dim(4)]

            # Rearrange to (batch, length, channels)
            y = x.permute([0, 2, 1])
            print(f"Permuted shape: {y.shape}")
            # Output: Permuted shape: [Dim(2), Dim(4), Dim(3)]

        Args:
            dims: A list specifying the new order of dimensions. For example,
                ``[2, 0, 1]`` moves dimension 2 to position 0, dimension 0 to
                position 1, and dimension 1 to position 2.

        Returns:
            A tensor with permuted dimensions.
        """
        return F.permute(self, dims)

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        """Returns a tensor that is a transposed version of input.

        The given dimensions ``dim1`` and ``dim2`` are swapped.

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.dtype import DType

            # Create a 2x3 matrix
            x = Tensor([[1, 2, 3], [4, 5, 6]], dtype=DType.int32)
            print(f"Original shape: {x.shape}")
            # Output: Original shape: [Dim(2), Dim(3)]
            print(x)

            # Transpose dimensions 0 and 1 to get a 3x2 matrix
            y = x.transpose(0, 1)
            print(f"Transposed shape: {y.shape}")
            # Output: Transposed shape: [Dim(3), Dim(2)]
            print(y)

        Args:
            dim1: The first dimension to be transposed.
            dim2: The second dimension to be transposed.

        Returns:
            A tensor with dimensions ``dim1`` and ``dim2`` swapped.
        """
        return F.transpose(self, dim1, dim2)

    @property
    def T(self) -> Tensor:
        """Returns a tensor with the last two dimensions transposed.

        This is equivalent to calling ``transpose(-1, -2)``, which swaps
        the last two dimensions of the tensor. For a 2D matrix, this produces
        the standard matrix transpose.

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.dtype import DType

            # Create a 2x3 matrix
            x = Tensor([[1, 2, 3], [4, 5, 6]], dtype=DType.int32)
            print(f"Original shape: {x.shape}")
            # Output: Original shape: [Dim(2), Dim(3)]

            # Use .T property (equivalent to transpose(-1, -2))
            y = x.T
            print(f"Transposed shape: {y.shape}")
            # Output: Transposed shape: [Dim(3), Dim(2)]
            print(y)

        Returns:
            A tensor with the last two dimensions transposed.
        """
        return self.transpose(-1, -2)

    def __getitem__(self, idx):  # noqa: ANN001
        if self.is_distributed:
            if not isinstance(idx, tuple):
                idx = (idx,)
            return F.slice_tensor(self, idx)
        return F.functional(graph.TensorValue.__getitem__)(self, idx)

    def __setitem__(self, idx, val) -> None:  # noqa: ANN001
        """Write into a slice of this tensor in-place.

        Delegates to :func:`functional.buffer_store_slice`
        which handles both single-device and distributed tensors.

        Args:
            idx: Index or slice specification (same syntax as
                ``__getitem__``).
            val: A ``Tensor`` whose data is copied into the selected
                region.
        """
        if not isinstance(val, Tensor):
            raise TypeError(
                "__setitem__ requires a Tensor value; "
                "use Buffer indexing for scalar writes."
            )
        indices = idx if isinstance(idx, tuple) else (idx,)
        _buffer_store_slice(self, val, indices)

    def __abs__(self) -> Tensor:
        return F.abs(self)

    def __neg__(self) -> Tensor:
        return F.negate(self)

    def __eq__(self, rhs: Any) -> Tensor:  # type: ignore[override]
        return F.equal(self, rhs)

    def __ne__(self, rhs: Any) -> Tensor:  # type: ignore[override]
        return F.not_equal(self, rhs)

    def __ge__(self, rhs: Any) -> Tensor:
        return F.greater_equal(self, rhs)

    def __gt__(self, rhs: Any) -> Tensor:
        return F.greater(self, rhs)

    def __lt__(self, rhs: Any) -> Tensor:
        return ~(self >= rhs)

    def __le__(self, rhs: Any) -> Tensor:
        return ~(self > rhs)

    def __add__(self, rhs: TensorValueLike) -> Tensor:
        return F.add(self, rhs)

    def __radd__(self, lhs: TensorValueLike) -> Tensor:
        return F.add(lhs, self)

    def __sub__(self, rhs: TensorValueLike) -> Tensor:
        return F.sub(self, rhs)

    def __rsub__(self, lhs: TensorValueLike) -> Tensor:
        return F.sub(lhs, self)

    def __mul__(self, rhs: TensorValueLike) -> Tensor:
        return F.mul(self, rhs)

    def __rmul__(self, lhs: TensorValueLike) -> Tensor:
        return F.mul(lhs, self)

    def __truediv__(self, rhs: TensorValueLike) -> Tensor:
        return F.div(self, rhs)

    def __rtruediv__(self, lhs: TensorValueLike) -> Tensor:
        return F.div(lhs, self)

    def __floordiv__(self, rhs: TensorValueLike) -> Tensor:
        return F.floor(F.div(self, rhs))

    def __rfloordiv__(self, lhs: TensorValueLike) -> Tensor:
        return F.floor(F.div(lhs, self))

    def __mod__(self, rhs: TensorValueLike) -> Tensor:
        return F.mod(self, rhs)

    def __rmod__(self, lhs: TensorValueLike) -> Tensor:
        return F.mod(lhs, self)

    def __divmod__(self, rhs: TensorValueLike) -> tuple[Tensor, Tensor]:
        return (self // rhs, self % rhs)

    def __rdivmod__(self, lhs: TensorValueLike) -> tuple[Tensor, Tensor]:
        return (self.__rfloordiv__(lhs), self.__rmod__(lhs))

    def __matmul__(self, rhs: TensorValueLike) -> Tensor:
        return F.matmul(self, rhs)

    def __rmatmul__(self, lhs: TensorValueLike) -> Tensor:
        return F.matmul(lhs, self)

    def __pow__(self, rhs: TensorValueLike) -> Tensor:
        return F.pow(self, rhs)

    def __rpow__(self, lhs: TensorValueLike) -> Tensor:
        return F.pow(lhs, self)

    def __and__(self, rhs: TensorValueLike) -> Tensor:
        return F.logical_and(self, rhs)

    def __rand__(self, lhs: TensorValueLike) -> Tensor:
        return F.logical_and(lhs, self)

    def __or__(self, rhs: TensorValueLike) -> Tensor:
        return F.logical_or(self, rhs)

    def __ror__(self, lhs: TensorValueLike) -> Tensor:
        return F.logical_or(lhs, self)

    def __xor__(self, rhs: TensorValueLike) -> Tensor:
        return F.logical_xor(self, rhs)

    def __rxor__(self, lhs: TensorValueLike) -> Tensor:
        return F.logical_xor(lhs, self)

    def __invert__(self) -> Tensor:
        return F.logical_not(self)


# ─── Sharding helpers (pure functions) ────────────────────────────────────


# Import functional at module end to avoid circular import.
# This works because method bodies are evaluated at call time, not definition time.
import numpy as np  # isort: skip

from max.experimental import functional as F  # isort: skip

# Access via module attribute to avoid importing names from a
# partially-initialized package (circular import guard).
_buffer_store_slice = lambda *a, **kw: F.buffer_store_slice(*a, **kw)
_transfer_to = lambda *a, **kw: F.transfer_to(*a, **kw)
