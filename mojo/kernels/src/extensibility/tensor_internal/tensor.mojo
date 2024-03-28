# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the `Tensor` type.

Example:

```mojo
from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from random import rand

var height = 256
var width = 256
var channels = 3

# Create the tensor of dimensions height, width, channels
# and fill with random values.
var image = rand[DType.float32](height, width, channels)

# Declare the grayscale image.
var spec = TensorSpec(DType.float32, height, width)
var gray_scale_image = Tensor[DType.float32](spec)

# Perform the RGB to grayscale transform.
for y in range(height):
  for x in range(width):
    var r = image[y,x,0]
    var g = image[y,x,1]
    var b = image[y,x,2]
    gray_scale_image[Index(y,x)] = 0.299 * r + 0.587 * g + 0.114 * b

print(gray_scale_image.shape().__str__())
```
"""

import math
from collections import List
from pathlib import Path

from algorithm.functional import elementwise, vectorize
from algorithm.reduction import argmax, argmin
from buffer import Buffer, NDBuffer
from buffer.list import Dim
from memory import memset_zero
from memory.unsafe import bitcast

from utils._serialize import _serialize
from utils.index import Index
from utils.loop import unroll
from utils.static_tuple import StaticTuple

from .tensor_shape import TensorShape
from .tensor_spec import TensorSpec

# ===----------------------------------------------------------------------===#
# Tensor
# ===----------------------------------------------------------------------===#


@always_inline
fn _elementwise[
    op: fn[dtype: DType, simd_width: Int] (x: SIMD[dtype, simd_width]) -> SIMD[
        dtype, simd_width
    ],
    dtype: DType,
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[tensor.dtype](tensor._spec)
    var buffer = tensor._to_buffer()
    var result_buffer = result._to_buffer()

    @__copy_capture(result_buffer, buffer)
    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        var idx = indices[0]
        result_buffer.store(
            idx, op[dtype, width](buffer.load[width=width](idx))
        )

    elementwise[func=func, simd_width = simdwidthof[dtype](), rank=1](
        Index(len(buffer))
    )

    return result


@always_inline
fn _elementwise[
    op: fn[dtype: DType, simd_width: Int] (
        x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width],
    dtype: DType,
](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[a.dtype](a._spec)
    var a_buffer = a._to_buffer()
    var b_buffer = b._to_buffer()
    var result_buffer = result._to_buffer()

    @__copy_capture(result_buffer, a_buffer, b_buffer)
    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        var idx = indices[0]
        result_buffer.store(
            idx,
            op[dtype, width](
                a_buffer.load[width=width](idx), b_buffer.load[width=width](idx)
            ),
        )

    elementwise[func=func, simd_width = simdwidthof[dtype](), rank=1](
        Index(len(a_buffer))
    )

    return result


@always_inline
fn _elementwise[
    op: fn[dtype: DType, simd_width: Int] (
        x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width],
    dtype: DType,
](a: Tensor[dtype], b: Scalar[dtype]) -> Tensor[dtype]:
    var result = Tensor[a.dtype](a._spec)
    var a_buffer = a._to_buffer()
    var result_buffer = result._to_buffer()

    @__copy_capture(result_buffer, a_buffer)
    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        var idx = indices[0]
        result_buffer.store(
            idx,
            op[dtype, width](
                a_buffer.load[width=width](idx), SIMD[dtype, width](b)
            ),
        )

    elementwise[func=func, simd_width = simdwidthof[dtype](), rank=1](
        Index(len(a_buffer))
    )

    return result


@always_inline
fn _elementwise[
    op: fn[dtype: DType, simd_width: Int] (
        x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width],
    dtype: DType,
](a: Scalar[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[b.dtype](b._spec)
    var b_buffer = b._to_buffer()
    var result_buffer = result._to_buffer()

    @__copy_capture(result_buffer, b_buffer)
    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        var idx = indices[0]
        result_buffer.store(
            idx,
            op[dtype, width](
                SIMD[dtype, width](a), b_buffer.load[width=width](idx)
            ),
        )

    elementwise[func=func, simd_width = simdwidthof[dtype](), rank=1](
        Index(len(b_buffer))
    )

    return result


struct Tensor[dtype: DType](Stringable, CollectionElement, EqualityComparable):
    """A tensor type which owns its underlying data and is parameterized on
    DType.


    Parameters:
      dtype: The underlying element type of the tensor.
    """

    var _spec: TensorSpec
    """The underlying specification of the tensor."""
    var _ptr: DTypePointer[dtype]
    """The underlying data of the tensor."""

    @always_inline
    fn __init__(inout self):
        """Default initializer for TensorShape."""
        self._spec = TensorSpec()
        self._ptr = DTypePointer[dtype]()

    @always_inline
    fn __init__(inout self, other: Self):
        """Creates a deep copy of an existing tensor.

        Args:
            other: The tensor to copy from.
        """
        var num_elements = other.num_elements()
        self._spec = other._spec
        self._ptr = DTypePointer[dtype].alloc(num_elements)
        memcpy(self._ptr, other._ptr, num_elements)

    @always_inline
    fn __init__(inout self, *dims: Int):
        """Allocates a tensor using the shape provided.

        Args:
          dims: The tensor dimensions.
        """
        self = Self(TensorSpec(dtype, dims))

    @always_inline
    fn __init__(inout self, owned shape: TensorShape):
        """Allocates a tensor using the shape provided.

        Args:
          shape: The tensor shape.
        """
        self = Self(TensorSpec(dtype, shape^))

    @always_inline
    fn __init__(inout self, owned spec: TensorSpec):
        """Allocates a tensor using the spec provided.

        Args:
          spec: The tensor spec.
        """
        var num_elements = spec.num_elements()
        self._spec = spec
        self._ptr = DTypePointer[dtype].alloc(num_elements)
        memset_zero(self._ptr, num_elements)

    @always_inline
    fn __init__(
        inout self, owned shape: TensorShape, owned ptr: DTypePointer[dtype]
    ):
        """Initializes a Tensor from the pointer and shape provided. The caller
        relinquishes the ownership of the pointer being passed in.

        Args:
          shape: The tensor shapes.
          ptr: The data pointer.
        """
        self = Self(TensorSpec(dtype, shape^), ptr)

    @always_inline
    fn __init__(
        inout self, owned spec: TensorSpec, owned ptr: DTypePointer[dtype]
    ):
        """Initializes a Tensor from the pointer and shape provided. The caller
        relinquishes the ownership of the pointer being passed in.

        Args:
          spec: The tensor spec.
          ptr: The data pointer.
        """
        self._spec = spec^
        self._ptr = ptr

    @always_inline
    fn __init__(inout self, shape: TensorShape, *data: Scalar[dtype]):
        """Initializes a Tensor from the shape and data provided. If a single
        scalar is passed in, then the scalar is splatted to all elements in the
        tensor.

        Args:
          shape: The tensor shape.
          data: Elements to place into the created tensor.
        """
        var num_elements = shape.num_elements()
        var ptr = DTypePointer[dtype].alloc(num_elements)
        if len(data) == 1:
            var data0 = data[0]

            if data0:

                @parameter
                fn splat_val[simd_width: Int](idx: Int):
                    ptr.store[width=simd_width](idx, data0)

                vectorize[splat_val, simdwidthof[dtype](), unroll_factor=8](
                    num_elements
                )

            else:
                memset_zero(ptr, num_elements)
        else:
            for i in range(len(data)):
                ptr[i] = data[i]
        self = Self(shape, ptr)

    @always_inline
    fn __init__(
        inout self, shape: TensorShape, owned list: List[Scalar[dtype]]
    ):
        """Initializes a 1-dimensional Tensor from the provided list.

        Args:
            shape: The tensor shape.
            list: The list to construct this Tensor from.
        """
        # Store the list length before we do a wiping take from it
        var list_len = len(list)

        var data_anyptr = list.steal_data()
        var data_ptr = Pointer[Scalar[dtype]].__from_index(int(data_anyptr))
        var data_dptr = DTypePointer[dtype](data_ptr)

        self = Self(shape, data_dptr)

    @always_inline
    fn __init__(inout self, owned list: List[Scalar[dtype]]):
        """Initializes a 1-dimensional Tensor from the provided list.

        Args:
            list: The list to construct this Tensor from.
        """
        # Store the list length before we do a wiping take from it
        var list_len = len(list)

        var data_anyptr = list.steal_data()
        var data_ptr = Pointer[Scalar[dtype]].__from_index(int(data_anyptr))
        var data_dptr = DTypePointer[dtype](data_ptr)

        self = Self(TensorShape(list_len), data_dptr)

    @always_inline
    fn __del__(owned self):
        """Delete the spec and release any owned memory."""
        self._ptr.free()

    @always_inline
    fn __copyinit__(inout self, other: Self):
        """Creates a deep copy of an existing tensor.

        Args:
            other: The tensor to copy from.
        """
        var num_elements = other.num_elements()
        self._spec = other._spec
        self._ptr = DTypePointer[dtype].alloc(num_elements)
        memcpy(self._ptr, other._ptr, num_elements)

    fn __moveinit__(inout self, owned existing: Self):
        """Move initializer for the tensor.

        Args:
            existing: The tensor to move.
        """
        self._spec = existing._spec^
        self._ptr = existing._ptr
        existing._spec = TensorSpec()
        existing._ptr = DTypePointer[dtype]()

    @always_inline
    fn ireshape(inout self, new_shape: TensorShape) raises -> None:
        """(Inplace) Reshapes the tensor by assigning it a new shape.

        Args:
            new_shape: The new shape.
        """
        if new_shape.num_elements() != self.num_elements():
            raise "Number of elements must match in reshape"

        self._spec = TensorSpec(dtype, new_shape)

    @always_inline
    fn reshape(inout self, new_shape: TensorShape) raises -> Tensor[dtype]:
        """Returns a reshaped tensor.

        Args:
            new_shape: The new shape.

        Returns:
            A Tensor that is a reshaped version of the original tensor.
        """
        var result = self
        result.ireshape(new_shape)

        return result

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two tensors are the same and False otherwise.

        Args:
          other: The other Tensor to compare against.

        Returns:
          True if the two tensors are the same and False otherwise.
        """
        if self._spec != other._spec:
            return False

        return memcmp(self.data(), other.data(), self.num_elements()) == 0

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two tensors are not the same and False otherwise.

        Args:
          other: The other Tensor to compare against.

        Returns:
          True if the two tensors are the not the same and False otherwise.
        """

        return not (self == other)

    @always_inline
    fn __add__(self, other: Self) raises -> Self:
        """Adds this tensor with another tensor.

        Constraints:
             The two tensors must have the same rank, type, and dimensions.

        Args:
            other: The RHS of the add operation.

        Returns:
            The addition of both tensors.
        """
        if self._spec != other._spec:
            raise "shape mismatch during tensor addition"

        return _elementwise[math.add](self, other)

    @always_inline
    fn __add__(self, other: Scalar[dtype]) -> Self:
        """Adds this tensor with a scalar.

        Args:
            other: The RHS of the add operation.

        Returns:
            The addition result.
        """
        return _elementwise[math.add](self, other)

    @always_inline
    fn __radd__(self, other: Scalar[dtype]) -> Self:
        """Adds this tensor with a scalar.

        Args:
            other: The LHS of the add operation.

        Returns:
            The addition result.
        """
        return _elementwise[math.add](other, self)

    @always_inline
    fn __sub__(self, other: Self) raises -> Self:
        """Subtracts a tensor from this tensor.

        Constraints:
             The two tensors must have the same rank, type, and dimensions.

        Args:
            other: The RHS of the sub operation.

        Returns:
            The addition of both tensors.
        """
        if self._spec != other._spec:
            raise "shape mismatch during tensor subtraction"

        return _elementwise[math.sub](self, other)

    @always_inline
    fn __sub__(self, other: Scalar[dtype]) -> Self:
        """Subtracts a scalar from this tensor.

        Args:
            other: The RHS of the sub operation.

        Returns:
            The subtraction result.
        """
        return _elementwise[math.sub](self, other)

    @always_inline
    fn __rsub__(self, other: Scalar[dtype]) -> Self:
        """Subtracts this tensor from a scalar.

        Args:
            other: The LHS of the sub operation.

        Returns:
            The addition result.
        """
        return _elementwise[math.sub](other, self)

    @always_inline
    fn __mul__(self, other: Self) raises -> Self:
        """Multiplies this tensor with another tensor.

        Constraints:
             The two tensors must have the same rank, type, and dimensions.

        Args:
            other: The RHS of the mul operation.

        Returns:
            The multiplication of both tensors.
        """
        if self._spec != other._spec:
            raise "shape mismatch during tensor multiplication"

        return _elementwise[math.mul](self, other)

    @always_inline
    fn __mul__(self, other: Scalar[dtype]) -> Self:
        """Multiplies this tensor with a scalar.

        Args:
            other: The RHS of the mul operation.

        Returns:
            The multiplication result.
        """
        return _elementwise[math.mul](self, other)

    @always_inline
    fn __rmul__(self, other: Scalar[dtype]) -> Self:
        """Multiplies this tensor with a scalar.

        Args:
            other: The LHS of the mul operation.

        Returns:
            The multiplication result.
        """
        return _elementwise[math.mul](other, self)

    @always_inline
    fn __truediv__(self, other: Self) raises -> Self:
        """Divides this tensor by another tensor.

        TODO: Change the return type if inputs are int

        Constraints:
             The two tensors must have the same rank, type, and dimensions.

        Args:
            other: The RHS of the div operation.

        Returns:
            The division of both tensors.
        """
        if self._spec != other._spec:
            raise "shape mismatch during tensor multiplication"

        return _elementwise[math.div](self, other)

    @always_inline
    fn __truediv__(self, other: Scalar[dtype]) -> Self:
        """Divides this tensor by a scalar.

        Args:
            other: The RHS of the div operation.

        Returns:
            The division result.
        """
        return _elementwise[math.div](self, other)

    @always_inline
    fn __rtruediv__(self, other: Scalar[dtype]) -> Self:
        """Divides a scalar by this tensor, broadcasting elementwise.

        Args:
            other: The LHS of the div operation.

        Returns:
            The division result.
        """
        return _elementwise[math.div](other, self)

    @always_inline
    fn __ipow__(inout self, exponent: Int) -> None:
        """In-place pow operator.

        Raises each element of the tensor to the power of `exponent` in place.

        Constraints:
             For integral values the exponent cannot be negative.

        Args:
            exponent: Integer power to raise tensor to.
        """
        self = self**exponent

    @always_inline
    fn __pow__(self, exponent: Int) -> Self:
        """Returns a copy of the tensor with each element raised to the power
        of `exponent`.

        Constraints:
             For integral values the exponent cannot be negative.

        Args:
            exponent: Integer power to raise tensor to.

        Returns:
            An exponentiated copy of tensor.
        """
        var result = self
        var buffer = result._to_buffer()

        # Define an elementwise pow that captures and modifies `buffer`.
        @__copy_capture(buffer)
        @parameter
        fn _pow[width: Int, rank: Int](indices: StaticIntTuple[rank]) -> None:
            var idx = indices[0]
            var val = buffer.load[width=width](idx)
            var res = math.pow(val, exponent)
            buffer.store(idx, res)

        # Use the `elementwise` generator to run `pow` in parallel.
        alias dtype_simd_width = simdwidthof[dtype]()

        elementwise[func=_pow, simd_width=dtype_simd_width, rank=1](
            Index(len(buffer))
        )

        return result

    @always_inline
    fn astype[new_dtype: DType](self) -> Tensor[new_dtype]:
        """Copy the Tensor with elements cast to the new type.

        Parameters:
            new_dtype: The type to cast the values to.

        Returns:
            A new tensor with the same values but the new type.
        """
        var result = Tensor[new_dtype](self._spec)
        var buffer = self._to_buffer()
        var result_buffer = result._to_buffer()

        @__copy_capture(result_buffer, buffer)
        @parameter
        fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
            var idx = indices[0]
            result_buffer.store(
                idx, buffer.load[width=width](idx).cast[new_dtype]()
            )

        elementwise[func=func, simd_width = simdwidthof[dtype](), rank=1](
            Index(len(buffer))
        )

        return result

    @always_inline
    fn clip(
        self,
        lower_bound: Scalar[dtype],
        upper_bound: Scalar[dtype],
    ) -> Self:
        """Clips the values of the tensor.

        Args:
            lower_bound: The lower bound.
            upper_bound: The upper bound.

        Returns:
            A clipped version of the tensor.
        """
        var result = Self(self._spec)
        var buffer = self._to_buffer()
        var result_buffer = result._to_buffer()

        @__copy_capture(result_buffer, buffer)
        @parameter
        fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
            var idx = indices[0]
            result_buffer.store(
                idx,
                math.clamp[dtype, width](
                    buffer.load[width=width](idx), lower_bound, upper_bound
                ),
            )

        elementwise[func=func, simd_width = simdwidthof[dtype](), rank=1](
            Index(len(buffer))
        )

        return result

    @always_inline
    fn data(self) -> DTypePointer[dtype]:
        """Gets the underlying Data pointer to the Tensor.

        Returns:
          The underlying data pointer of the tensor.
        """
        return self._ptr

    @always_inline
    fn type(self) -> DType:
        """Gets the underlying DType of the tensor.

        Returns:
          The underlying DType of the tensor.
        """
        return dtype

    @always_inline
    fn rank(self) -> Int:
        """Gets the rank of the tensor.

        Returns:
          The rank of the tensor.
        """
        return self._spec.rank()

    @always_inline
    fn num_elements(self) -> Int:
        """Gets the total number of elements in the tensor.

        Returns:
          The total number of elements in the tensor.
        """
        return self._spec.num_elements()

    @always_inline
    fn bytecount(self) -> Int:
        """Gets the total bytecount of the tensor.

        Returns:
          The total bytecount of the tensor.
        """
        return self._spec.bytecount()

    @always_inline
    fn spec(self) -> TensorSpec:
        """Gets the specification of the tensor.

        Returns:
          The underlying tensor spec of the tensor.
        """
        return self._spec

    @always_inline
    fn shape(self) -> TensorShape:
        """Gets the shape of the tensor.

        Returns:
          The underlying tensor shape of the tensor.
        """
        return self._spec.shape

    @always_inline
    fn dim(self, idx: Int) -> Int:
        """Gets the dimension at the specified index.

        Args:
          idx: The dimension index.

        Returns:
          The dimension at the specified index.
        """
        return self.spec()[idx]

    @no_inline
    fn __str__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string of the tensor.
        """
        var res = String("Tensor(")

        @parameter
        fn serialize[T: Stringable](val: T):
            res += str(val)

        var shape = List[Int]()
        for i in range(self.rank()):
            shape.append(self.shape()[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.data(), shape
        )

        return res + ")"

    @no_inline
    fn __repr__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string representation of the tensor.
        """
        return self.__str__()

    @always_inline
    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        """Gets the value at the specified index.

        Args:
          index: The index of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        return self._ptr.load(index)

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn __getitem__(self, indices: VariadicList[Int]) -> Scalar[dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn __getitem__[
        len: Int
    ](self, indices: StaticIntTuple[len]) -> Scalar[dtype]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          len: The length of the indecies.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn load[*, width: Int = 1](self, index: Int) -> SIMD[dtype, width]:
        """Gets the SIMD value at the specified index.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          index: The index of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        return self._ptr.load[width=width](index)

    @always_inline
    fn load[*, width: Int = 1](self, *indices: Int) -> SIMD[dtype, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        return self.load[width=width](indices)

    @always_inline
    fn load[
        *, width: Int = 1
    ](self, indices: VariadicList[Int]) -> SIMD[dtype, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(len(indices) == self.rank(), "invalid rank value")
        return self._ptr.load[width=width](self._compute_linear_offset(indices))

    @always_inline
    fn load[
        len: Int, /, *, width: Int = 1
    ](self, indices: StaticIntTuple[len]) -> SIMD[dtype, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          len: The length of the indecies.
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(len == self.rank(), "invalid length value")
        return self._ptr.load[width=width](self._compute_linear_offset(indices))

    @always_inline
    fn __setitem__(inout self, index: Int, val: Scalar[dtype]):
        """Sets the value at the specified index.

        Args:
          index: The index of the value to set.
          val: The value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        self.store[width=1](index, val)

    @always_inline
    fn __setitem__(inout self, indices: VariadicList[Int], val: Scalar[dtype]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.store[width=1](indices, val)

    @always_inline
    fn __setitem__[
        len: Int
    ](inout self, indices: StaticIntTuple[len], val: Scalar[dtype]):
        """Sets the value at the specified indices.

        Parameters:
          len: The length of the indecies.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.store[len, width=1](indices, val)

    @always_inline
    fn store[
        *, width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]):
        """Sets the SIMD value at the specified index.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          index: The index of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        self._ptr.store[width=width](index, val)

    @always_inline
    fn store[
        *, width: Int = 1
    ](inout self, indices: VariadicList[Int], val: SIMD[dtype, width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(len(indices) == self.rank(), "invalid rank value")
        self._ptr.store[width=width](self._compute_linear_offset(indices), val)

    @always_inline
    fn store[
        len: Int, /, *, width: Int = 1
    ](inout self, indices: StaticIntTuple[len], val: SIMD[dtype, width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          len: The length of the indecies.
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(len == self.rank(), "invalid length value")
        self._ptr.store[width=width](self._compute_linear_offset(indices), val)

    @always_inline
    fn _compute_linear_offset[
        rank: Int
    ](self, indices: StaticIntTuple[rank]) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Parameters:
          rank: The rank of the indices.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        var result = indices[0]

        @unroll
        for i in range(rank - 1):
            result = self.dim(i + 1) * result + indices[i + 1]
        return result

    @always_inline
    fn _compute_linear_offset(self, *indices: Int) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        return self._compute_linear_offset(indices)

    @always_inline
    fn _compute_linear_offset(self, indices: VariadicList[Int]) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        var rank = len(indices)
        var result = indices[0]
        for i in range(rank - 1):
            result = self.dim(i + 1) * result + indices[i + 1]
        return result

    @always_inline
    fn _to_ndbuffer[rank: Int](self) -> NDBuffer[dtype, rank]:
        debug_assert(
            rank == self.rank(), "to_ndbuffer rank must match Tensor rank"
        )
        var shape = StaticIntTuple[rank](0)

        @unroll
        for i in range(rank):
            shape[i] = self.dim(i)

        return NDBuffer[dtype, rank](self._ptr, shape)

    @always_inline
    fn _to_buffer(self) -> Buffer[dtype]:
        return Buffer[dtype](self._ptr, self.num_elements())

    fn _truncate_axis_dim(self, axis: Int, keep_dims: Bool = True) -> List[Int]:
        var output_shape = List[Int](capacity=self.rank())
        for i in range(self.rank()):
            if i == axis or i == axis + self.rank():
                if keep_dims:
                    output_shape.append(1)
                else:
                    pass
            else:
                output_shape.append(self.dim(i))
        return output_shape^

    fn argmax(self, *, axis: Int = -1) raises -> Tensor[DType.index]:
        """
        Finds the indices of the maximum element along the specified axis.

        Args:
            axis: The axis.

        Returns:
            A new tensor containing the indices of the maximum elements along axis.
        """

        alias ARGMAX_MAX_TENSOR_RANK = 8

        if self.rank() > ARGMAX_MAX_TENSOR_RANK:
            raise "unsupported tensor rank. The tensor rank must be at most " + str(
                ARGMAX_MAX_TENSOR_RANK
            )

        var output_shape = self._truncate_axis_dim(axis)
        var output = Tensor[DType.index](output_shape)

        @parameter
        @always_inline
        fn rank_dispatch[idx: Int]() raises:
            alias rank = idx + 1
            if rank == self.rank():
                argmax(
                    self._to_ndbuffer[rank](),
                    axis,
                    output._to_ndbuffer[rank](),
                )

        unroll[rank_dispatch, ARGMAX_MAX_TENSOR_RANK]()

        output.ireshape(
            TensorShape(self._truncate_axis_dim(axis, keep_dims=False))
        )

        return output

    fn argmin(self, *, axis: Int = -1) raises -> Tensor[DType.index]:
        """
        Finds the indices of the minimum element along the specified axis.

        Args:
            axis: The axis.

        Returns:
            A new tensor containing the indices of the minimum elements along axis.
        """
        alias ARGMIN_MAX_TENSOR_RANK = 8

        if self.rank() > ARGMIN_MAX_TENSOR_RANK:
            raise "unsupported tensor rank. The tensor rank must be at most " + str(
                ARGMIN_MAX_TENSOR_RANK
            )

        var output_shape = self._truncate_axis_dim(axis)
        var output = Tensor[DType.index](output_shape)

        @parameter
        @always_inline
        fn rank_dispatch[idx: Int]() raises:
            alias rank = idx + 1
            if rank == self.rank():
                argmin(
                    self._to_ndbuffer[rank](),
                    axis,
                    output._to_ndbuffer[rank](),
                )

        unroll[rank_dispatch, ARGMIN_MAX_TENSOR_RANK]()

        output.ireshape(
            TensorShape(self._truncate_axis_dim(axis, keep_dims=False))
        )

        return output

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        self._to_buffer().tofile(path)

    @always_inline
    fn _steal_ptr(inout self) -> DTypePointer[dtype]:
        """Transfer ownership of pointer to the underlying memory.
        The caller is responsible for freeing up the memory.

        Returns:
            The pointer to the underlying memory.
        """
        var ptr = self._ptr
        self._ptr = DTypePointer[dtype]()
        self._spec = TensorSpec()
        return ptr

    @staticmethod
    fn fromfile(path: Path) raises -> Self:
        """Read tensor from a file.

        Args:
          path: Path to the output file.

        Returns:
          The tensor read from file.
        """
        var byte_tensor = Tensor[DType.int8](path.read_bytes())
        var num_elements = byte_tensor.num_elements()
        return Self(
            num_elements // dtype.sizeof(),
            bitcast[dtype](byte_tensor._steal_ptr()),
        )

    fn save(self, path: Path) raises:
        """Save given tensor to file. This method preserves
           shape and datatype information.

        Args:
          path: Path of file.
        """
        _serialize_to_file(self, path)

    @staticmethod
    fn load(path: Path) raises -> Tensor[dtype]:
        """Read tensor from a file.
           The path should be a file saved with Tensor.save method.

        Args:
          path: Path to the output file.

        Returns:
          The tensor read from file.
        """
        var bytes = Tensor[DType.int8](path.read_bytes())
        var minimum_size = len(_SERIALIZATION_HEADER) + (3 * sizeof[UInt32]())

        if bytes.num_elements() < minimum_size:
            raise "given file is not a serialized mojo tensor."

        for i in range(len(_SERIALIZATION_HEADER)):
            if bytes[i] != _SERIALIZATION_HEADER[i]:
                raise "given file is not a serialized mojo tensor."

        fn _uint32_from_bytes(data: DTypePointer[DType.int8]) -> UInt32:
            var ptr = data._as_scalar_pointer()
            var spec_ptr = bitcast[UInt32](ptr)
            return __get_address_as_owned_value(spec_ptr.address)

        var major_format_ptr = bytes.data() + len(_SERIALIZATION_HEADER)
        var major_format = _uint32_from_bytes(major_format_ptr)
        var minor_format_ptr = major_format_ptr + sizeof[UInt32]()
        var minor_format = _uint32_from_bytes(minor_format_ptr)
        if (
            major_format != _SERIALIZATION_MAJOR_FORMAT
            or minor_format != _SERIALIZATION_MINOR_FORMAT
        ):
            raise "cannot load tensor of format: " + String(
                major_format
            ) + "." + String(minor_format)

        var spec_size_ptr = minor_format_ptr + sizeof[UInt32]()
        var spec_size = _uint32_from_bytes(spec_size_ptr)
        if spec_size != sizeof[TensorSpec]():
            raise "invalid tensor spec."
        var spec_ptr = spec_size_ptr + sizeof[UInt32]()
        var spec = TensorSpec.from_bytes(spec_ptr)
        if dtype != spec.dtype():
            raise "requested type doesn't match the dtype in serialized tensor."
        var data = spec_ptr + sizeof[TensorSpec]()
        var tensor = Self(spec)
        if spec.num_elements() == 0:
            return tensor
        memcpy(tensor.data(), bitcast[dtype](data), spec.num_elements())
        _ = bytes^
        return tensor


# ===----------------------------------------------------------------------===#
# serialize
# ===----------------------------------------------------------------------===#

# Serialization constants
alias _SERIALIZATION_MAJOR_FORMAT: UInt32 = 0
alias _SERIALIZATION_MINOR_FORMAT: UInt32 = 1
# 0x93 🔥 0x93
alias _SERIALIZATION_HEADER = StaticTuple[Int8, 6](
    0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x93
)


fn _serialize_as_tensor[
    type: AnyRegType
](inout object: type) -> Tensor[DType.int8]:
    """Serialize the given object into a Tensor of bytes.

    Args:
      object: Object to serialize.

    Returns:
      Tensor containing the bytes of object.
    """
    var self_ptr = bitcast[Int8](Pointer.address_of(object))
    alias size = sizeof[type]()
    var bytes = Tensor[DType.int8](size)
    memcpy(bytes.data(), DTypePointer[DType.int8](self_ptr.address), size)
    return bytes^


fn _serialize_to_file[type: DType](tensor: Tensor[type], path: Path) raises:
    """Serialize given tensor to file. This method preserves
       shape and datatype information.

    Args:
      tensor: Tensor to serialize.
      path: Path of file.
    """
    var header_size = len(_SERIALIZATION_HEADER)
    var header_bytes = Tensor[DType.int8](header_size)

    for i in range(header_size):
        header_bytes.store(i, _SERIALIZATION_HEADER[i])

    var major_format: UInt32 = _SERIALIZATION_MAJOR_FORMAT
    var major_format_bytes = _serialize_as_tensor(major_format)
    var minor_format: UInt32 = _SERIALIZATION_MINOR_FORMAT
    var minor_format_bytes = _serialize_as_tensor(minor_format)
    var spec_size: UInt32 = sizeof[TensorSpec]()
    var spec_size_bytes = _serialize_as_tensor(spec_size)
    var spec = tensor.spec()
    var spec_bytes = _serialize_as_tensor[TensorSpec](spec)

    var bytes = Tensor[DType.int8](
        header_bytes.num_elements()
        + major_format_bytes.num_elements()
        + minor_format_bytes.num_elements()
        + spec_size_bytes.num_elements()
        + spec_bytes.num_elements()
        + tensor.num_elements() * type.sizeof()
    )
    var copied: Int = 0

    @always_inline("nodebug")
    fn _copy_bytes(
        inout dest: Tensor[DType.int8], offset: Int, src: Tensor[DType.int8]
    ) -> Int:
        var size = src.num_elements()
        memcpy(
            dest.data() + offset,
            src.data(),
            size,
        )
        return offset + size

    copied = _copy_bytes(bytes, copied, header_bytes)
    copied = _copy_bytes(bytes, copied, major_format_bytes)
    copied = _copy_bytes(bytes, copied, minor_format_bytes)
    copied = _copy_bytes(bytes, copied, spec_size_bytes)
    # TODO: Numpy aligns this to 64 byte boundary.
    copied = _copy_bytes(bytes, copied, spec_bytes)

    # TODO: Avoid this copy.
    memcpy(
        bytes.data() + copied,
        bitcast[DType.int8](tensor.data()),
        tensor.num_elements() * type.sizeof(),
    )
    copied += tensor.num_elements() * type.sizeof()

    debug_assert(bytes.num_elements() == copied, "expected these to be same.")

    bytes.tofile(path)
