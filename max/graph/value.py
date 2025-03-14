# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import numpy as np
from max import mlir
from max._core import graph as _graph
from max.dtype import DType

from . import ops
from .type import (
    BufferType,
    DeviceRef,
    Dim,
    DimLike,
    Shape,
    ShapeLike,
    TensorType,
    Type,
    _ChainType,
)


class Value:
    """Represents a symbolic value within a `Graph`.

    A `Value` can represent the output of a node, the arguments of a
    `Graph` (as seen from within its body), and more generally any symbolic
    value available within the `Graph`. Other nodes receive `Value`
    values as inputs to form a computation graph.

    A `Value` may also refer to an existing input or output of a node,
    and you can change them, such as by swapping a new `Value`.

    Conceptually, think of a `Value` as an edge in the dataflow graph,
    with the other end being the user of that value.

    Similar to a regular variable, a `Value` has a data type.
    """

    _mlir_value: mlir.Value

    def __new__(cls, value: Union[Value, mlir.Value, TensorValueLike]):
        # If a subclass is being requested, let the subclass `__init__` deal with everything.
        # Note, we can't use `issubclass` here cause `Value` is a subclass of `Value` according to python.
        if cls is not Value:
            return super().__new__(cls)

        # Otherwise, the user is requesting a Value(x) directly.
        # we need to dispatch to the correct subclass.
        if isinstance(value, mlir.Value):
            # Value(x) where x is an MLIR value
            # Return the correct subclass based on the mlir value's type.
            if _graph.type_is_opaque(value.type):
                return super().__new__(_OpaqueValue)
            elif _graph.type_is_buffer(value.type):
                return super().__new__(BufferValue)
            elif _graph.type_is_chain(value.type):
                return super().__new__(_ChainValue)
            elif _graph.type_is_tensor(value.type):
                return super().__new__(TensorValue)
            else:
                raise TypeError(
                    "Value() argument is an mlir.Value of unknown type"
                    f" '{value.type}'"
                )
        elif isinstance(value, Value):
            # Value(x) where x is an instance of a subclass of Value.
            # Return the correct subclass based on the type of value.
            return super().__new__(type(value))
        elif isinstance(value, _tensor_value_like):
            # Value(x) where x is a TensorValueLike.
            # Explicitly tell the user to call `TensorValue` if that is what they wanted.
            raise TypeError(
                "Value() can not be created directly from a"
                f" '{type(value).__name__}'. If you are trying to create a"
                " tensor, call TensorValue() directly."
            )
        else:
            raise TypeError(
                "Value() argument must be an mlir.Value, a graph.Value, or"
                f" an np.ndarray, not '{type(value).__name__}'"
            )

    def __repr__(self):
        return str(self._mlir_value.type)

    @property
    def buffer(self) -> BufferValue:
        """Returns the Value as a :obj:`BufferValue`.

        Raises an exception if the Value is not a BufferValue.
        """
        if isinstance(self, BufferValue):
            return self

        msg = f"Value is not a BufferValue, was '{type(self).__name__}'"
        raise TypeError(msg)

    @property
    def tensor(self) -> TensorValue:
        """Returns the Value as a :obj:`TensorValue`.

        Raises an exception if the Value is not a TensorValue.
        """
        if isinstance(self, TensorValue):
            return self

        msg = f"Value is not a TensorValue, was '{type(self).__name__}'"
        raise TypeError(msg)

    @property
    def opaque(self) -> _OpaqueValue:
        """Returns the Value as an :obj:`_OpaqueValue`.

        Raises an exception if the Value is not a _OpaqueValue.
        """
        if isinstance(self, _OpaqueValue):
            return self

        msg = f"Value is not a TensorValue, was '{type(self).__name__}'"
        raise TypeError(msg)

    @property
    def type(self) -> Type:
        """Returns the type of the :obj:`Value` as a :obj:`Type`."""
        raise NotImplementedError


class _ChainValue(Value):
    def __init__(self, value: Union[Value, mlir.Value]):
        if isinstance(value, mlir.Value) and _graph.type_is_chain(value.type):
            self._mlir_value = value
        elif isinstance(value, _ChainValue):
            self._mlir_value = value._mlir_value
        else:
            raise TypeError(
                "_ChainValue() argument must be an mlir.Value of chain type "
                f"or a graph._ChainValue, not {type(value).__name__!r}"
            )

    @property
    def type(self) -> _ChainType:
        """Returns the type of the :obj:`_ChainValue` as a :obj:`_ChainType`."""
        return _ChainType.from_mlir(self._mlir_value.type)


class _OpaqueValue(Value):
    """Represents an opaque value within a `Graph`."""

    def __init__(self, value: Union[Value, mlir.Value]) -> None:
        if isinstance(value, mlir.Value) and _graph.type_is_opaque(value.type):
            self._mlir_value = value
        elif isinstance(value, _OpaqueValue):
            self._mlir_value = value._mlir_value
        else:
            raise TypeError(
                "_OpaqueValue() argument must be an mlir.Value of opaque type "
                f"or a graph._OpaqueValue, not {type(value).__name__!r}"
            )


class BufferValue(Value):
    """Represents a mutable semantic tensor within a `Graph`."""

    def __init__(self, value: Union[Value, mlir.Value]) -> None:
        if isinstance(value, mlir.Value) and _graph.type_is_buffer(value.type):
            self._mlir_value = value
        elif isinstance(value, BufferValue):
            self._mlir_value = value._mlir_value
        else:
            raise TypeError(
                "BufferValue() argument must be an mlir.Value of buffer type "
                f"or a graph.BufferValue, not '{type(value).__name__}'"
            )

    @property
    def type(self) -> BufferType:
        """Returns the type of the :obj:`BufferValue` as a :obj:`BufferType`."""
        return BufferType.from_mlir(self._mlir_value.type)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the BufferValue."""
        return self.type.shape

    @property
    def device(self) -> Optional[DeviceRef]:
        """Returns the device of the BufferValue."""
        return self.type.device

    # dtype and rank are implemented like TensorValue implementation.
    # They use _graph directly to avoid loading the shape dimension if they
    # aren't needed.
    # This also avoids accidentally loading algebraic expression dimensions
    # (which will throw an exception).
    @property
    def dtype(self) -> DType:
        """Returns the tensor data type."""
        t = self._mlir_value.type
        if not _graph.type_is_buffer(t):
            raise TypeError(f"Expected BufferType, got: {t}")

        return DType(_graph.buffer_type_get_dtype(t))

    @property
    def rank(self) -> int:
        """Returns the rank (number of dims) of the buffer."""
        t = self._mlir_value.type
        if not _graph.type_is_buffer(t):
            msg = f"Expected BufferType, got: {t}"
            raise TypeError(msg)

        return _graph.buffer_type_get_rank(t)

    def __repr__(self):
        dtype = self.dtype
        shape = self.shape
        device = self.device
        return f"{type(self).__name__}({dtype=}, {shape=}, {device=})"

    def __getitem__(self, index) -> TensorValue:
        x = ops.buffer_load(self)
        if index is Ellipsis:
            return x
        return ops.slice_tensor(
            x,
            index if isinstance(index, Iterable) else (index,),  # type: ignore
        )

    def __setitem__(
        self,
        index,
        val: TensorValue,
    ) -> None:
        if index is Ellipsis:
            return ops.buffer_store(self, val)
        return ops.buffer_store_slice(
            self,
            val,
            index if isinstance(index, Iterable) else (index,),  # type: ignore
        )

    def print(self, label: str = "debug_buffer"):
        ops.print(self[...], label=label)


class TensorValue(Value):
    """
    Represents a value semantic tensor within a :obj:`Graph`. It provides
    various methods and properties to manipulate and query tensor attributes
    such as :obj:`shape`, data type (:obj:`dtype`), device placement (:obj:`device`), and more.

    For example:

    .. code-block:: python

        import numpy as np
        from max.dtype import DType
        from max.graph import Graph, ops

        # Create a sample matrix
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Create a Graph context to work with tensors
        with Graph("tensor_demo") as graph:
            # Create a constant tensor from the matrix
            tensor = ops.constant(matrix, dtype=DType.float32)

            # Perform a simple operation: transpose the tensor
            transposed_tensor = tensor.T  # Output: Tensor representing [[1, 3], [2, 4]]
    """

    # Disallow special methods that would fall back to __getitem__ and hang.
    __contains__ = None
    __iter__ = None

    def __init__(self, value: TensorValueLike) -> None:
        if isinstance(value, mlir.Value) and _graph.type_is_tensor(value.type):
            self._mlir_value = value
        elif isinstance(value, TensorValue):
            self._mlir_value = value._mlir_value
        elif isinstance(value, Dim):
            self._mlir_value = TensorValue.from_dim(value)._mlir_value
        elif isinstance(value, Shape):
            self._mlir_value = TensorValue.from_shape(value)._mlir_value
        elif isinstance(value, _numeric):
            raise TypeError(
                "TensorValue() can not be created directly from a"
                f" '{type(value).__name__}'. Use ops.constant to"
                " convert to a TensorValue with a specific dtype."
            )
        else:
            raise TypeError(
                "TensorValue() argument must be an mlir.Value of tensor type,"
                " a graph.TensorValue, or a graph.Weight, not"
                f" '{type(value).__name__}'"
            )

    @staticmethod
    def from_dim(dim: DimLike) -> TensorValue:
        """Creates a new tensor based on provided MLIR dimension type.

        Args:
            dim: The dimension value.

        Returns:
            A new :obj:`TensorValue`.
        """
        return ops.shape_to_tensor([dim]).reshape(())

    @staticmethod
    def from_shape(shape: ShapeLike) -> TensorValue:
        """Creates a new tensor with the specified shape.

        Args:
            shape: An iterable of integers or symbolic dimensions.

        Returns:
            A new :obj:`TensorValue`.
        """
        return ops.shape_to_tensor(shape)

    def __repr__(self):
        dtype = self.dtype
        shape = self.shape
        device = self.device
        return f"{type(self).__name__}({dtype=}, {shape=}, {device=})"

    @property
    def type(self) -> TensorType:
        """Returns the type of the :obj:`TensorValue` as a :obj:`TensorType`."""
        return TensorType.from_mlir(self._mlir_value.type)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the :obj:`TensorValue`.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("shape_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)

                # Access tensor properties
                print(f"Shape: {tensor.shape}")  # Output: Shape: (2, 2)
        """
        return self.type.shape

    @property
    def device(self) -> Optional[DeviceRef]:
        """Returns the device of the TensorValue."""
        return self.type.device

    # dtype and rank are special.
    # They use _graph directly to avoid loading the shape dimension if they aren't needed.
    # This also avoids accidentally loading algebraic expression dimensions (which will throw an exception).
    @property
    def dtype(self) -> DType:
        """Returns the tensor data type.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("dtype_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)
                print(f"Data type: {tensor.dtype}")  # Output: Data type: DType.float32
        """
        t = self._mlir_value.type
        if not _graph.type_is_tensor(t):
            raise TypeError(f"Expected TensorType, got: {t}")

        return DType(_graph.tensor_type_get_dtype(t))

    @property
    def rank(self) -> int:
        """Returns the rank (number of dims) of the buffer.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("rank_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)
                print(f"Rank (number of dimensions): {tensor.rank}")  # Output: Rank: 2
        """
        t = self._mlir_value.type
        if not _graph.type_is_tensor(t):
            raise TypeError(f"Expected TensorType, got: {t}")

        return _graph.tensor_type_get_rank(t)

    def print(self, label: str = "debug_tensor"):
        """Prints detailed information about the tensor.

        Args:
            label: A string label for the printed output. Defaults ``debug_tensor``.
        """
        ops.print(self, label=label)

    def reshape(self, shape: ShapeLike) -> TensorValue:
        """Creates a new tensor with the same data but reshaped.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("reshape_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)

                reshaped_tensor = tensor.reshape((1, 4)) # Output: Tensor representing [[1, 2, 3, 4]]

        Args:
            shape: The new shape as an iterable of integers or symbolic dimensions.

        Returns:
            A new :obj:`TensorValue` with the reshaped dimensions.
        """
        return ops.reshape(self, shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> TensorValue:
        """Flattens the specified dims of a symbolic tensor.

        The number and order of the elements in the tensor is unchanged.
        All dimensions from ``start_dim`` to ``end_dim`` (inclusive) are merged into a single output dim.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("flatten_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)
                # Flatten the tensor
                flattened_tensor = tensor.flatten() # Output: Tensor representing [1, 2, 3, 4]

        Args:
            start_dim: The starting dimension to flatten. Defaults to ``1``.
            end_dim: The ending dimension to flatten. Defaults to ``-1``.

        Returns:
            A new :obj:`TensorValue` with the broadcasted shape.
        """
        return ops.flatten(self, start_dim, end_dim)

    def broadcast_to(self, shape: ShapeLike) -> TensorValue:
        """Broadcasts the tensor to a new shape.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("broadcast_to_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)

                broadcasted_tensor = tensor.broadcast_to((3, 2, 2)) # Output: Tensor with shape (3, 2, 2)

        Args:
            shape: An iterable of integers or symbolic dimensions.

        Returns:
            A new :obj:`TensorValue` with the broadcasted shape.
        """
        return ops.broadcast_to(self, shape)

    def cast(self, dtype: DType) -> TensorValue:
        """Casts a symbolic tensor to a different data type.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("cast_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32)

                casted_tensor = tensor.cast(DType.int32) # Output: Tensor representing [[1, 2], [3, 4]] with dtype=int32

        Args:
            dtype: The target data type (e.g., ``DType.int32``, ``DType.float64``).

        Returns:
            A new :obj:`TensorValue` with the casted data type.
        """
        return ops.cast(self, dtype)

    def rebind(self, shape: ShapeLike, message: str = "") -> TensorValue:
        """Rebinds the tensor to a new shape with error handling.

        Args:
            shape: The new shape as an iterable of integers or symbolic dimensions.
            message: (optional) A message for logging or debugging.

        Returns:
            A new :obj:`TensorValue` with the updated shape.
        """
        return ops.rebind(self, shape, message)

    def permute(self, dims: list[int]) -> TensorValue:
        """Permutes the tensor's dimensions based on provided indices.

        Args:
            dims: A list of integers specifying the new order of dimensions.

        Returns:
            A new :obj:`TensorValue` with permuted dimensions.
        """
        return ops.permute(self, dims)

    def transpose(self, dim_1: int, dim_2: int) -> TensorValue:
        """Swaps two dimensions of the tensor.

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import Graph, ops

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
            with Graph("transpose_demo") as graph:
                tensor = ops.constant(matrix, dtype=DType.float32)
                transposed_tensor = tensor.transpose(dim_1=0, dim_2=1)
                print(transposed_tensor)

        Args:
            dim_1: The first dimension to swap.
            dim_2: The second dimension to swap.

        Returns:
            A new :obj:`TensorValue` with swapped dimensions.
        """
        return ops.transpose(self, dim_1, dim_2)

    def to(self, device: DeviceRef) -> TensorValue:
        """Transfers the tensor to a specified device without mutation.

        .. code-block:: python

            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            with Graph("to_example") as graph:
                tensor = ops.constant(matrix, dtype=DType.float32)
                print(tensor.device)

        Args:
            device: A :obj:`DeviceRef` object specifying the target device.

        Returns:
            A new :obj:`TensorValue` on the specified device.
        """
        return ops.transfer_to(self, device)

    @property
    def T(self) -> TensorValue:
        """Returns the transposed tensor.
        :obj:`T` is the shorthand notation for transposing.
        For more information, see :obj:`transpose()`.

        Returns:
            A new :obj:`TensorValue` with swapped dimensions.
        """
        return self.transpose(-1, -2)

    def __getitem__(self, index: Any) -> TensorValue:
        return ops.slice_tensor(
            self,
            index if isinstance(index, Iterable) else (index,),  # type: ignore
        )

    def __eq__(self, rhs: Any) -> TensorValue:  # type: ignore[override]
        if _is_tensor_value_like(rhs):
            return ops.equal(self, rhs)
        else:
            raise TypeError(
                "'==' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __neg__(self) -> TensorValue:
        return ops.negate(self)

    def __ne__(self, rhs: Any) -> TensorValue:  # type: ignore[override]
        if _is_tensor_value_like(rhs):
            return ops.not_equal(self, rhs)
        else:
            raise TypeError(
                "'!=' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __ge__(self, rhs: Any) -> TensorValue:
        if _is_tensor_value_like(rhs):
            return ops.greater_equal(self, rhs)
        else:
            raise TypeError(
                "'>=' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __gt__(self, rhs: Any) -> TensorValue:
        if _is_tensor_value_like(rhs):
            return ops.greater(self, rhs)
        else:
            raise TypeError(
                f"'>' not supported between instance of '{type(self).__name__}'"
                f" and '{type(rhs).__name__}'"
            )

    def __lt__(self, rhs: Any) -> TensorValue:
        return ops.logical_not(self >= rhs)

    def __le__(self, rhs: Any) -> TensorValue:
        return ops.logical_not(self > rhs)

    def __add__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.add(self, rhs)

    def __radd__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.add(lhs, self)

    def __sub__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.sub(self, rhs)

    def __rsub__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.sub(lhs, self)

    def __mul__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.mul(self, rhs)

    def __rmul__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.mul(lhs, self)

    def __truediv__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.div(self, rhs)

    def __rtruediv__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.div(lhs, self)

    def __floordiv__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.floor(ops.div(self, rhs))

    def __rfloordiv__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.floor(ops.div(lhs, self))

    def __mod__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.mod(self, rhs)

    def __rmod__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.mod(lhs, self)

    def __divmod__(
        self, rhs: TensorValueLike
    ) -> tuple[TensorValue, TensorValue]:
        return (self // rhs, self % rhs)

    def __rdivmod__(
        self, lhs: TensorValueLike
    ) -> tuple[TensorValue, TensorValue]:
        return (lhs // self, lhs % self)

    def __matmul__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.matmul(self, rhs)

    def __rmatmul__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.matmul(lhs, self)

    def __pow__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.pow(self, rhs)

    def __rpow__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.pow(lhs, self)

    def __and__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.logical_and(self, rhs)

    def __rand__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.logical_and(lhs, self)

    def __or__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.logical_or(self, rhs)

    def __ror__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.logical_or(lhs, self)

    def __xor__(self, rhs: TensorValueLike) -> TensorValue:
        return ops.logical_xor(self, rhs)

    def __rxor__(self, lhs: TensorValueLike) -> TensorValue:
        return ops.logical_xor(lhs, self)


Numeric = Union[int, float, np.integer, np.floating, np.ndarray]
StrongTensorValueLike = Union[mlir.Value, BufferValue, TensorValue, Shape, Dim]
TensorValueLike = Union[StrongTensorValueLike, Numeric]

# This is needed for python 3.9 compatibility.
# `isinstance` only works with tuples and not unions in 3.9.
_numeric = (int, float, np.integer, np.floating, np.ndarray)
_strong_tensor_value_like = (mlir.Value, BufferValue, TensorValue, Shape, Dim)
_tensor_value_like = _strong_tensor_value_like + _numeric


def _is_tensor_value_like(obj: Any) -> TypeGuard[TensorValueLike]:
    return isinstance(obj, _tensor_value_like)
