# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Provides experimental tensor operations with eager execution capabilities.

**Warning:** This module contains experimental APIs that are subject to change
or removal in future versions. Use with caution in production environments.

This module provides the :class:`~max.experimental.tensor` class which supports
eager execution of tensor operations, complementing the graph-based execution
model provided by :obj:`~max.graph`. The tensor operations automatically compile
and execute using the MAX runtime.

**Key Features:**

- **Eager execution**: Operations execute immediately rather than building a graph.
- **Automatic compilation**: Tensors are compiled and optimized automatically.
- **Lazy evaluation**: Tensors may be computed lazily until their values are needed.
- **NumPy compatibility**: Supports common NumPy-like operations and indexing.

Create and manipulate tensors with automatic compilation and optimization:

.. code-block:: python

    from max.experimental import tensor
    from max.driver import CPU
    from max.dtype import DType

    # Create and operate on tensors
    x = tensor.Tensor.ones((2, 3), dtype=DType.float32, device=CPU())
    y = tensor.Tensor.zeros((2, 3), dtype=DType.float32, device=CPU())
    result = x + y  # Eager execution with automatic compilation
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import sys
import warnings
import weakref
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from itertools import chain
from typing import Any, TypeVar

from max.graph.value import HasTensorValue
from rich.pretty import pretty_repr

from .. import _core, driver, engine, graph, mlir
from .._core.dialects import builtin, kgen, mo
from ..driver import (
    CPU,
    Accelerator,
    Device,
    DLPackArray,
    accelerator_count,
)
from ..dtype import DType
from ..graph import (
    ShapeLike,
    TensorType,
    TensorValueLike,
    Value,
    ops,
)
from ..graph.graph import _location
from ..graph.ops.constant import NestedArray, Number
from . import functional as F

_SESSION: ContextVar[engine.api.InferenceSession] = ContextVar("_SESSION")
_DEFAULT_DEVICE: ContextVar[Device] = ContextVar("_DEFAULT_DEVICE")
_DEFAULT_DTYPE: ContextVar[DType] = ContextVar("_DEFAULT_DTYPE")


T = TypeVar("T")


@contextlib.contextmanager
def contextvar_context(var: ContextVar[T], value: T):  # noqa: ANN201
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


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


def default_device(device: Device):  # noqa: ANN201
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
    return contextvar_context(_DEFAULT_DEVICE, device)


def default_dtype(dtype: DType):  # noqa: ANN201
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


def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels on tensors."""
    device_specs = driver.scan_available_devices()
    if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = driver.load_devices(device_specs)
    if not (session := _SESSION.get(None)):
        _SESSION.set(session := engine.api.InferenceSession(devices=devices))
    return session


def _in_running_loop() -> bool:
    """Check whether the caller is inside a running event loop."""
    # - asyncio.get_event_loop().is_running() works in most scenarios
    # - asyncio.get_event_loop() raises in some environments
    # - use asyncio.get_running_loop() and check if it fails instead
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class Tensor(DLPackArray, HasTensorValue):
    """A multi-dimensional array with eager execution and automatic compilation.

    The Tensor class provides a high-level interface for numerical computations
    with automatic compilation and optimization via the MAX runtime. Operations
    on tensors execute eagerly while benefiting from lazy evaluation and
    graph-based optimizations behind the scenes.

    **Key Features:**

    - **Eager execution**: Operations execute immediately with automatic compilation.
    - **Lazy evaluation**: Computation may be deferred until results are needed.
    - **High performance**: Uses the Mojo compiler and optimized kernels.
    - **NumPy-like API**: Supports familiar array operations and indexing.
    - **Device flexibility**: Works seamlessly across CPU and accelerators.

    **Creating Tensors:**

    Create tensors using factory methods like :meth:`ones`, :meth:`zeros`,
    :meth:`constant`, :meth:`arange`, or from other array libraries via
    :meth:`from_dlpack`.

    .. code-block:: python

        from max.experimental import tensor
        from max.dtype import DType

        # Create tensors with factory methods
        x = tensor.Tensor.ones((2, 3), dtype=DType.float32)
        y = tensor.Tensor.zeros((2, 3), dtype=DType.float32)

        # Perform operations
        result = x + y  # Eager execution with automatic compilation

        # Access values
        print(result.shape)  # (2, 3)
        print(result.dtype)  # DType.float32

    **Implementation Notes:**

    Tensors use lazy evaluation internally - they don't always hold concrete
    data in memory. A tensor may be "unrealized" (not yet computed) until its
    value is actually needed (e.g., when printing, converting to other formats,
    or calling :meth:`item`). This allows the runtime to optimize sequences of
    operations efficiently.

    Operations on tensors build a computation graph behind the scenes, which is
    compiled and executed when needed. All illegal operations fail immediately
    with clear error messages, ensuring a smooth development experience.

    **Interoperability:**

    Tensors support the DLPack protocol for zero-copy data exchange with NumPy,
    PyTorch, JAX, and other array libraries. Use :meth:`from_dlpack` to import
    arrays and standard DLPack conversion for export.
    """

    #: Underlying memory for a realized tensor.
    storage: driver.Tensor | None
    #: - For a realized tensor this is a graph input BufferValue
    #: - For an unrealized tensor this is a value in the graph
    _value: graph.BufferValue | graph.TensorValue
    _real: bool = False

    def __init__(
        self,
        *,
        storage: driver.Tensor | None = None,
        value: graph.BufferValue | graph.TensorValue | None = None,
    ):
        self.storage = storage
        if value is not None:
            self._value = value
        else:
            GRAPH.add_source(self)
        self.real = storage is not None

    @classmethod
    def from_graph_value(
        cls, value: graph.TensorValue | graph.BufferValue
    ) -> Tensor:
        """Creates a tensor from a graph value.

        Constructs a tensor from an existing graph value, which can be either
        a :obj:`~max.graph.TensorValue` or :obj:`~max.graph.BufferValue`. This
        is useful for converting graph level values into tensor objects for
        eager execution.

        Args:
            value: The graph value to wrap. Can be either a TensorValue or
                BufferValue from the MAX graph API.

        Returns:
            Tensor: A new tensor backed by the provided graph value.
        """
        return cls(value=value)

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
        return Tensor(storage=driver.Tensor.from_dlpack(array))

    @classmethod
    def constant(
        cls,
        value: DLPackArray | NestedArray | Number,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Creates a constant tensor from a scalar, array, or nested list.

        Constructs a tensor with constant values that can be a scalar, a nested
        Python list, or a DLPack-compatible array. The shape is automatically
        inferred from the input data structure.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create from scalar
            x = tensor.Tensor.constant(42, dtype=DType.int32)

            # Create from nested list
            y = tensor.Tensor.constant([[1.0, 2.0], [3.0, 4.0]])

            # Create from NumPy array
            import numpy as np

            z = tensor.Tensor.constant(np.array([1, 2, 3]))

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
        dtype, device = defaults(dtype, device)
        return F.constant(value, dtype, device)

    @classmethod
    def full(
        cls,
        shape: ShapeLike,
        value: Number,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
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
            device: The device where the tensor will be allocated. If not
                specified, defaults to an accelerator if available, otherwise CPU.

        Returns:
            Tensor: A new tensor with the specified shape filled with the given value.
        """
        return F.broadcast_to(
            cls.constant(value, dtype=dtype, device=device), shape
        )

    @classmethod
    def full_like(cls, type: TensorType, value: Number) -> Tensor:
        """Creates a tensor filled with a value, matching a given type's properties.

        Returns a new tensor filled with the specified value that matches the
        shape, data type, and device of the given tensor type. This is useful
        when you need to create a tensor with uniform values that's compatible
        with an existing tensor's properties.

        .. code-block:: python

            from max.experimental import tensor
            from max.graph import TensorType
            from max.driver import CPU
            from max.dtype import DType

            # Create a reference tensor type
            ref_type = TensorType(DType.float32, (2, 3), device=CPU())

            # Create tensor filled with 5.0 matching the reference type
            x = tensor.Tensor.full_like(ref_type, value=5.0)

        Args:
            type: The tensor type to match. The returned tensor will have the
                same shape, dtype, and device as this type.
            value: The scalar value to fill the tensor with.

        Returns:
            Tensor: A new tensor filled with the specified value, matching the
                properties of the input type.
        """
        return cls.full(
            type.shape,
            value=value,
            dtype=type.dtype,
            device=type.device.to_device(),
        )

    @classmethod
    def zeros(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Creates a tensor filled with zeros.

        Returns a new tensor with the specified shape where all elements are
        initialized to zero. The tensor is created with eager execution and
        automatic compilation.

        .. code-block:: python

            from max.experimental import tensor
            from max.driver import CPU
            from max.dtype import DType

            # Create a 2x3 tensor of zeros
            x = tensor.Tensor.zeros((2, 3), dtype=DType.float32, device=CPU())
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
            device: The device where the tensor will be allocated. If not
                specified, defaults to an accelerator if available, otherwise CPU.

        Returns:
            Tensor: A new tensor with the specified shape filled with zeros.
        """
        return cls.full(shape, value=0, dtype=dtype, device=device)

    @classmethod
    def zeros_like(cls, type: TensorType) -> Tensor:
        """Creates a tensor of zeros matching a given type's properties.

        Returns a new tensor filled with zeros that matches the shape, data type,
        and device of the specified tensor type. This is useful when you need to
        create a zero tensor that's compatible with an existing tensor's properties.

        .. code-block:: python

            from max.experimental import tensor
            from max.graph import TensorType
            from max.driver import CPU
            from max.dtype import DType

            # Create a reference tensor type
            ref_type = TensorType(DType.float32, (3, 4), device=CPU())

            # Create zeros tensor matching the reference type
            x = tensor.Tensor.zeros_like(ref_type)
            # Result: 3x4 tensor of zeros with dtype float32 on CPU

        Args:
            type: The tensor type to match. The returned tensor will have the
                same shape, dtype, and device as this type.

        Returns:
            Tensor: A new tensor filled with zeros matching the properties of the
                input type.
        """
        return cls.zeros(
            type.shape, dtype=type.dtype, device=type.device.to_device()
        )

    @classmethod
    def ones(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Creates a tensor filled with ones.

        Returns a new tensor with the specified shape where all elements are
        initialized to one. The tensor is created with eager execution and
        automatic compilation.

        .. code-block:: python

            from max.experimental import tensor
            from max.driver import CPU
            from max.dtype import DType

            # Create a 2x3 tensor of ones
            x = tensor.Tensor.ones((2, 3), dtype=DType.float32, device=CPU())
            # Result: [[1.0, 1.0, 1.0],
            #          [1.0, 1.0, 1.0]]

            # Create a 1D tensor using default dtype and device
            y = tensor.Tensor.ones((5,))

        Args:
            shape: The shape of the output tensor. Can be a tuple of integers,
                a list of integers, or any value that can be converted to a shape.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device where the tensor will be allocated. If not
                specified, defaults to an accelerator if available, otherwise CPU.

        Returns:
            Tensor: A new tensor with the specified shape filled with ones.
        """
        return cls.full(shape, value=1, dtype=dtype, device=device)

    @classmethod
    def ones_like(cls, type: TensorType) -> Tensor:
        """Creates a tensor of ones matching a given type's properties.

        Returns a new tensor filled with ones that matches the shape, data type,
        and device of the specified tensor type. This is useful when you need to
        create a ones tensor that's compatible with an existing tensor's properties.

        .. code-block:: python

            from max.experimental import tensor
            from max.graph import TensorType
            from max.driver import CPU
            from max.dtype import DType

            # Create a reference tensor type
            ref_type = TensorType(DType.float32, (3, 4), device=CPU())

            # Create ones tensor matching the reference type
            x = tensor.Tensor.ones_like(ref_type)
            # Result: 3x4 tensor of ones with dtype float32 on CPU

        Args:
            type: The tensor type to match. The returned tensor will have the
                same shape, dtype, and device as this type.

        Returns:
            Tensor: A new tensor filled with ones matching the properties of the
                input type.
        """
        return cls.ones(
            type.shape, dtype=type.dtype, device=type.device.to_device()
        )

    @classmethod
    def arange(
        cls,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Creates a tensor with evenly spaced values within a given interval.

        Returns a new 1D tensor containing a sequence of values starting from
        ``start`` (inclusive) and ending before ``stop`` (exclusive), with values
        spaced by `step`. This is similar to Python's built-in ``range()`` function
        and NumPy's ``arange()``.

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

        Args:
            start: The starting value of the sequence. If ``stop`` is not provided,
                this becomes the ``stop`` value and ``start`` defaults to 0.
            stop: The end value of the sequence (exclusive). If not specified,
                the sequence ends at ``start`` and begins at 0.
            step: The spacing between values in the sequence.
            dtype: The data type for the tensor elements. If not specified,
                defaults to :obj:`DType.float32` for CPU devices and
                :obj:`DType.bfloat16` for accelerator devices.
            device: The device where the tensor will be allocated. If not
                specified, defaults to an accelerator if available, otherwise CPU.

        Returns:
            Tensor: A 1D tensor containing the evenly spaced values.
        """
        dtype, device = defaults(dtype, device)
        if stop is None:
            start, stop = 0, start
        return F.range(
            start,
            stop,
            step,
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
            from max.driver import CPU
            from max.dtype import DType

            # Create a reference tensor type with shape (2, 4)
            ref_type = TensorType(DType.int32, (2, 4), device=CPU())

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
        range = F.range(
            start=0,
            stop=dim,
            out_dim=dim,
            dtype=type.dtype,
            device=type.device.to_device(),
        )
        return F.broadcast_to(range, type.shape)

    @property
    def type(self) -> graph.TensorType:
        """Gets the tensor type information.

        Returns the type information for the tensor, including shape, dtype,
        and device. If the underlying value is a buffer type, it's converted
        to a tensor type.

        Returns:
            TensorType: The type information for the tensor.
        """
        type = self._value.type
        return type.as_tensor() if isinstance(type, graph.BufferType) else type

    @property
    def rank(self) -> int:
        """Gets the number of dimensions in the tensor.

        Returns the rank (number of dimensions) of the tensor. For example,
        a scalar has rank 0, a vector has rank 1, and a matrix has rank 2.

        Returns:
            int: The number of dimensions in the tensor.
        """
        return self._value.rank

    @property
    def shape(self) -> graph.Shape:
        """Gets the shape of the tensor.

        Returns the dimensions of the tensor as a shape object.

        Returns:
            Shape: The shape of the tensor.
        """
        return self._value.shape

    @property
    def dtype(self) -> DType:
        """Gets the data type of the tensor elements.

        Returns the data type (dtype) of the elements stored in the tensor,
        such as ``float32``, ``int32``, or ``bfloat16``.

        Returns:
            DType: The data type of the tensor elements.
        """
        return self._value.dtype

    @property
    def device(self) -> Device:
        """Gets the device where the tensor is stored.

        Returns the device (CPU or accelerator) where the tensor's data is
        located.

        Returns:
            Device: The device where the tensor is stored.
        """
        return self._value.device.to_device()

    @property
    def driver_tensor(self) -> driver.Tensor:
        """A pointer to the underlying memory.

        Raises if the tensor is unrealized.
        """
        if (storage := self.storage) is None:
            raise TypeError("Can't get driver tensor for symbolic tensor")
        return storage

    @property
    def real(self) -> bool:
        return self._real

    @real.setter
    def real(self, real: bool) -> None:
        if not real and self._in_global_compute_graph:
            GRAPH.add_unrealized(self)
        self._real = real

    def __tensorvalue__(self) -> graph.TensorValue:
        """Gets a TensorValue for the underlying data.

        If self is backed by a BufferValue, this will do a `ops.buffer_load`.
        The load is for ordering mutable operations and will be optimized away.
        """
        if isinstance(self._value, graph.BufferValue):
            return self._value[...]
        assert isinstance(self._value, graph.TensorValue)
        return self._value

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
        self.real = False

        if isinstance(self._value, graph.BufferValue):
            return self._value
        assert isinstance(self._value, graph.TensorValue)
        tensor = self._value
        self._value = buffer = ops.buffer_create(tensor.type.as_buffer())
        buffer[...] = tensor
        return buffer

    @property
    def _in_global_compute_graph(self) -> bool:
        mlir_value = self._value.to_mlir()
        graph_op = mlir_value.owner.parent_op
        return graph_op == _core.Operation._from_cmlir(GRAPH.graph._mlir_op)

    def __await__(self):
        """Force the tensor to realize if it is not already."""
        yield from asyncio.create_task(GRAPH.evaluate(self))
        assert self.real
        return self

    @property
    async def realize(self):  # noqa: ANN201
        """Force the tensor to realize if it is not already."""
        return await self

    def _sync_realize(self) -> Tensor:
        if self.real:
            return self

        if not self._in_global_compute_graph:
            raise TypeError(
                "Can't realize symbolic tensors in graph compilation."
            )

        # If there's no running loop, just use asyncio.run
        if not _in_running_loop():
            return asyncio.run(self.realize)

        # If there is a running loop, execute using a ThreadPoolExecutor
        # - This is a common case inside a Jupyter notebook, eg.
        #   printing a tensor.
        # - Otherwise, this is probably accidental. The code is running
        #   inside an async environment, but for some reason is trying
        #   to synchronously await. Check for this case explicitly and warn.
        def is_interactive() -> bool:
            import __main__ as main

            return not hasattr(main, "__file__")

        if not is_interactive():
            warnings.warn(
                "Use of synchronous tensor method inside another event loop. "
                "Use `await tensor`."
            )

        # Run self.realize in another thread
        loop = asyncio.new_event_loop()
        with ThreadPoolExecutor() as pool:
            fut = pool.submit(loop.run_until_complete, self.realize)
        return fut.result()

    def __bool__(self) -> bool:
        return bool(self.item())

    def _values(self):  # noqa: ANN202
        self._sync_realize()
        dt = self.driver_tensor.to(CPU())
        for idx in dt._iterate_indices():
            yield dt[idx].item()

    def __hash__(self):
        return id(self)

    def __dlpack__(self, stream: int | None = None):
        self._sync_realize()
        assert self.storage is not None
        return self.storage.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        self._sync_realize()
        assert self.storage is not None
        return self.storage.__dlpack_device__()

    def __rich_repr__(self):
        yield "shape", self.shape
        yield "dtype", self.dtype
        yield "device", self.device

    def __repr__(self):
        if not self._in_global_compute_graph:
            return pretty_repr(self)
        # Janky repr for bootstrapping, we can do much better.
        return f"{self.type}: [{', '.join(str(v) for v in self._values())}]"

    def __deepcopy__(self, memo: object) -> Tensor:
        # Tensors are value-semantic
        return self

    def item(self):  # noqa: ANN201
        """Gets the scalar value from a single-element tensor.

        Extracts and returns the scalar value from a tensor containing exactly
        one element. The tensor is realized if needed and transferred to CPU
        before extracting the value.

        Returns:
            The scalar value from the tensor. The return type matches the tensor's
            dtype (e.g., float for float32, int for int32).

        Raises:
            TypeError: If the tensor contains more than one element.
        """
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

    def to(self, device: Device) -> Tensor:
        """Transfers the tensor to a different device.

        Creates a new tensor with the same data on the specified device. This
        allows moving tensors between CPU and accelerators or between different
        accelerator devices.

        .. code-block:: python

            from max.experimental import tensor
            from max.driver import CPU, Accelerator

            # Create a tensor on CPU
            x = tensor.Tensor.ones((2, 3), device=CPU())
            print(x.device)  # CPU

            # Transfer to accelerator
            y = x.to(Accelerator())
            print(y.device)  # Accelerator(0)

        Args:
            device: The target device for the tensor.

        Returns:
            Tensor: A new tensor with the same data on the specified device.
        """
        return F.transfer_to(self, device)

    def argmax(self, axis: int = -1) -> Tensor:
        """Finds the indices of the maximum values along an axis.

        Returns a tensor containing the indices of the maximum values along
        the specified axis. This is useful for finding the position of the
        largest element, such as determining predicted classes in classification.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a 2x4 tensor
            x = tensor.Tensor.constant(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]], dtype=DType.float32
            )

            # Find argmax along last axis (within each row)
            indices = x.argmax(axis=-1)
            # Result: [1, 2] (index 1 in first row, index 2 in second row)

        Args:
            axis: The axis along which to find the maximum indices. Defaults
                to -1 (the last axis).

        Returns:
            Tensor: A tensor containing the indices of the maximum values.
        """
        return F.argmax(self, axis=axis)

    def max(self, axis: int = -1) -> Tensor:
        """Computes the maximum values along an axis.

        Returns a tensor containing the maximum values along the specified axis.
        This is useful for reduction operations and finding peak values in data.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a 2x4 tensor
            x = tensor.Tensor.constant(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]], dtype=DType.float32
            )

            # Find max along last axis (within each row)
            row_max = x.max(axis=-1)
            # Result: [3.5, 4.2]

            # Find max along first axis (within each column)
            col_max = x.max(axis=0)
            # Result: [2.3, 3.5, 4.2, 3.1]

        Args:
            axis: The axis along which to compute the maximum. Defaults to -1
                (the last axis).

        Returns:
            Tensor: A tensor containing the maximum values along the specified axis.
        """
        return F.max(self, axis=axis)

    def mean(self, axis: int = -1) -> Tensor:
        """Computes the mean values along an axis.

        Returns a tensor containing the arithmetic mean of values along the
        specified axis. This is useful for computing averages, normalizing data,
        or aggregating statistics.

        .. code-block:: python

            from max.experimental import tensor
            from max.dtype import DType

            # Create a 2x4 tensor
            x = tensor.Tensor.constant(
                [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]], dtype=DType.float32
            )

            # Compute mean along last axis (within each row)
            row_mean = x.mean(axis=-1)
            # Result: [5.0, 4.0] (mean of each row)

            # Compute mean along first axis (within each column)
            col_mean = x.mean(axis=0)
            # Result: [1.5, 3.5, 5.5, 7.5] (mean of each column)

        Args:
            axis: The axis along which to compute the mean. Defaults to -1
                (the last axis).

        Returns:
            Tensor: A tensor containing the mean values along the specified axis.
        """
        return F.mean(self, axis=axis)

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
            x = tensor.Tensor.constant([[1, 2, 3], [4, 5, 6]], dtype=DType.int32)
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
            x = tensor.Tensor.constant([1.7, 2.3, 3.9], dtype=DType.float32)
            print(x.dtype)  # DType.float32

            # Cast to int32 (truncates decimal values)
            y = x.cast(DType.int32)
            print(y.dtype)  # DType.int32
            # Values: [1, 2, 3]

        Args:
            dtype: The target data type for the tensor.

        Returns:
            Tensor: A new tensor with the specified data type.
        """
        return F.cast(self, dtype)

    def permute(self, dims: list[int]) -> Tensor:
        """Permutes the dimensions of the tensor.

        Returns a tensor with its dimensions reordered according to the
        specified permutation. This is useful for changing the layout of
        multi-dimensional data.

        Args:
            dims: A list specifying the new order of dimensions. For example,
                ``[2, 0, 1]`` moves dimension 2 to position 0, dimension 0 to
                position 1, and dimension 1 to position 2.

        Returns:
            Tensor: A tensor with permuted dimensions.
        """
        return F.permute(self, dims)

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        """Transposes two dimensions of the tensor.

        Returns a tensor with the specified dimensions swapped. This is a
        special case of permutation that swaps exactly two dimensions.

        Args:
            dim1: The first dimension to swap.
            dim2: The second dimension to swap.

        Returns:
            Tensor: A tensor with the specified dimensions transposed.
        """
        return F.transpose(self, dim1, dim2)

    @property
    def T(self) -> Tensor:
        """Gets the transposed tensor.

        Returns a tensor with the last two dimensions transposed. This is
        equivalent to calling ``transpose(-1, -2)`` and is commonly used for
        matrix operations.

        Returns:
            Tensor: A tensor with the last two dimensions swapped.
        """
        return self.transpose(-1, -2)

    def __getitem__(self, idx):  # noqa: ANN001
        return F.functional(graph.TensorValue.__getitem__)(self, idx)

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


class ComputeGraph:
    """Computation graph for managing tensor operations.

    This class manages the directed acyclic graph (DAG) of tensor operations
    for lazy evaluation and optimization. It tracks both realized tensors
    (with concrete data in memory) and unrealized tensors (pending computations)
    to enable efficient batch compilation and execution.
    """

    graph: graph.Graph
    #: Keeps a weak reference to tensor data that we need to compute graph values
    sources: weakref.WeakValueDictionary[_core.Value[Any], Tensor]
    #: Keeps a strong reference to tensors that are part of the current graph
    _current_graph_refs: list[Tensor]
    #: Keeps weak references to intermediate unrealized tensor values, which may
    #: never need to be realized.
    unrealized: weakref.WeakValueDictionary[int, Tensor]

    def __init__(
        self,
        context: mlir.Context | None = None,
        sources: Iterable[Tensor] = (),
        seed: int = 0,
    ):
        self.context = context or mlir.Context()
        self.sources = weakref.WeakValueDictionary()
        self._current_graph_refs = []

        self.unrealized = weakref.WeakValueDictionary()
        self.graph = graph.Graph("main", input_types=[], context=self.context)

        with self.graph:
            ops.random.set_seed(seed)

        for source in sources:
            self.add_source(source)

    async def evaluate(self, tensor: Tensor) -> None:
        """Evaluates and realizes the specified tensor.

        Compiles and executes the computation graph to produce concrete values
        for the input tensor and any other pending computations. This triggers
        lazy evaluation, converting unrealized tensors into realized ones with
        data in memory.

        Args:
            tensor: The tensor to realize. This triggers evaluation of its
                computation and any dependencies.
        """
        # Single-global-graph is (unsurprisingly) causing the spooky action at a distance :(
        # - Specifically, bad things give a nice compiler error!
        #   ... but then the exception hangs around, which holds a traceback
        #   which holds the call frame which holds the tensors that caused the error.
        # - Since these tensors don't get garbage collected they survive as unrealized
        # - Then the next evaluate call tries to realize again and necessarily
        #   will have the same error.
        # - HACK/workaround: running tensor evaluation clears the last exception.
        # This also means a user needs to explicitly drop any references to the
        # offending tensors. Ultimately this feels like a strong case for only
        # running partial graphs.
        sys.last_value = None
        sys.last_traceback = None
        gc.collect()

        # create strong references during execution
        unrealized = list(self.unrealized.values())
        with self.graph:
            # peek rather than next! If the seed is rotated but compilation
            # or execute fails, we need the seed to not rotate.
            self.graph.output(
                ops.random._peek_seed(), *map(graph.TensorValue, unrealized)
            )
        # Remove dead values and inputs
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            self.graph._module.operation
        )  # type: ignore
        # Remove sources that no longer exist from the graph
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        # The graph symbol is public, so RemoveDeadValues won't remove
        # unused arguments. Do that explicitly.
        _remove_unused_arguments(self.graph)
        inputs = [
            self.sources[input._mlir_value] for input in self.graph.inputs
        ]

        try:
            model = _session().load(self.graph)
            # This will become an await when `model` supports it
            seed, *results = model(*(input.driver_tensor for input in inputs))
            assert isinstance(seed, driver.Tensor)
        except BaseException as e:
            # If we've tried and failed to compile the graph, remove its
            # terminator so future ops can modify the graph.
            #  - Can failed lowerings leave the module in a partially lowered state?
            self.graph._erase_output_if_present()
            raise RuntimeError(
                "Failed to compile and execute graph! Please file an issue. "
                "This error should have been caught at op creation time."
            ) from e

        for tensor, storage in zip(unrealized, results, strict=True):
            # This will eventually support Mojo values also.
            assert isinstance(storage, driver.Tensor)
            tensor.storage = storage
            tensor.real = True

        # Remove strong references to inputs to allow garbage collection
        del inputs
        # Reset the graph to a new empty graph with only inputs
        ComputeGraph.__init__(
            self,
            context=self.graph._context,
            # - Re-add any sources that still have live references
            # - All evaluated tensors become realized sources of a new graph
            sources=chain(self.sources.values(), unrealized),
            seed=seed.item(),
        )

    def add_source(self, tensor: Tensor) -> None:
        if tensor.storage is None:
            raise TypeError("Only realized tensors may be graph sources.")

        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front
        # Update the GraphOP to reflect the new argument
        with self.graph:
            type = driver_tensor_type(tensor.storage).as_buffer().to_mlir()
            inputs = op.function_type.inputs
            op.function_type = builtin.FunctionType([*inputs, type])  # type: ignore
            tensor._value = graph.BufferValue.from_mlir(
                block.add_argument(type, _location())
            )
        self.sources[tensor._value._mlir_value] = tensor
        self._current_graph_refs.append(tensor)

    def add_unrealized(self, tensor: Tensor) -> None:
        self.unrealized[id(tensor)] = tensor


def _remove_unused_arguments(graph: graph.Graph) -> None:
    # Obviously this is deeply tied to the implementation of Graph.
    #  - GraphOp should be simplified to have a single API for managing arguments
    #  - Graph should expose this behavior
    op = _core.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)

    block = op.regions[0].front
    # reverse so indices don't during iteration+mutation
    for i, input in reversed(list(enumerate(graph.inputs))):
        if not input._mlir_value.num_uses:
            block.erase_argument(i)

    # graph.inputs is a cached_property, so reset it
    graph.inputs = [Value.from_mlir(arg) for arg in block.arguments]

    # update the graph op to correctly reflect the input changes
    with graph:
        op.function_type = builtin.FunctionType(  # type: ignore
            [input.type.to_mlir() for input in graph.inputs],
            op.function_type.results,
        )
        op.signature = kgen.FuncTypeGeneratorType([], op.function_type)  # type: ignore
        op.discardable_attributes["argument_names"] = builtin.ArrayAttr(
            [builtin.StringAttr(f"input{i}") for i in range(len(graph.inputs))]
        )


def driver_tensor_type(t: driver.Tensor) -> TensorType:
    """Converts a driver tensor to a :obj:TensorType.

    Creates a TensorType instance from a driver-level tensor by extracting
    its dtype, shape, and device information.

    Args:
        t: The driver tensor to convert.

    Returns:
        TensorType: A tensor type representing the driver tensor's properties.
    """
    return TensorType(t.dtype, t.shape, graph.DeviceRef.from_device(t.device))


GRAPH = ComputeGraph()
