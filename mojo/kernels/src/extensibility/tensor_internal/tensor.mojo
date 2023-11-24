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

let height = 256
let width = 256
let channels = 3

# Create the tensor of dimensions height, width, channels
# and fill with random values.
let image = rand[DType.float32](height, width, channels)

# Declare the grayscale image.
let spec = TensorSpec(DType.float32, height, width)
var gray_scale_image = Tensor[DType.float32](spec)

# Perform the RGB to grayscale transform.
for y in range(height):
  for x in range(width):
    let r = image[y,x,0]
    let g = image[y,x,1]
    let b = image[y,x,2]
    gray_scale_image[Index(y,x)] = 0.299 * r + 0.587 * g + 0.114 * b

print(gray_scale_image.shape().__str__())
```
"""

import math
from algorithm.functional import elementwise
from builtin.io import _Printable
from memory import memset_zero
from memory.buffer import NDBuffer
from memory.unsafe import bitcast

from utils._serialize import (
    _serialize,
    _serialize_to_file,
    _SERIALIZATION_HEADER,
    _SERIALIZATION_MAJOR_FORMAT,
    _SERIALIZATION_MINOR_FORMAT,
)

from .tensor_shape import TensorShape
from .tensor_spec import TensorSpec
from utils.list import Dim
from utils.static_tuple import StaticTuple
from utils.index import Index

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
    let result = Tensor[tensor.dtype](tensor._spec)
    let buffer = tensor._to_buffer()
    let result_buffer = result._to_buffer()

    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        let idx = indices[0]
        result_buffer.simd_store(
            idx, op[dtype, width](buffer.simd_load[width](idx))
        )

    elementwise[rank=1, simd_width = simdwidthof[dtype](), func=func](
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
    let result = Tensor[a.dtype](a._spec)
    let a_buffer = a._to_buffer()
    let b_buffer = b._to_buffer()
    let result_buffer = result._to_buffer()

    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        let idx = indices[0]
        result_buffer.simd_store(
            idx,
            op[dtype, width](
                a_buffer.simd_load[width](idx), b_buffer.simd_load[width](idx)
            ),
        )

    elementwise[rank=1, simd_width = simdwidthof[dtype](), func=func](
        Index(len(a_buffer))
    )

    return result


@always_inline
fn _elementwise[
    op: fn[dtype: DType, simd_width: Int] (
        x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width],
    dtype: DType,
](a: Tensor[dtype], b: SIMD[dtype, 1]) -> Tensor[dtype]:
    let result = Tensor[a.dtype](a._spec)
    let a_buffer = a._to_buffer()
    let result_buffer = result._to_buffer()

    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        let idx = indices[0]
        result_buffer.simd_store(
            idx,
            op[dtype, width](
                a_buffer.simd_load[width](idx), SIMD[dtype, width](b)
            ),
        )

    elementwise[rank=1, simd_width = simdwidthof[dtype](), func=func](
        Index(len(a_buffer))
    )

    return result


@always_inline
fn _elementwise[
    op: fn[dtype: DType, simd_width: Int] (
        x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width],
    dtype: DType,
](a: SIMD[dtype, 1], b: Tensor[dtype]) -> Tensor[dtype]:
    let result = Tensor[b.dtype](b._spec)
    let b_buffer = b._to_buffer()
    let result_buffer = result._to_buffer()

    @parameter
    fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
        let idx = indices[0]
        result_buffer.simd_store(
            idx,
            op[dtype, width](
                SIMD[dtype, width](a), b_buffer.simd_load[width](idx)
            ),
        )

    elementwise[rank=1, simd_width = simdwidthof[dtype](), func=func](
        Index(len(b_buffer))
    )

    return result


struct Tensor[dtype: DType](StringTrait):
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
    fn __init__(inout self, *dims: Int):
        """Allocates a tensor using the shape provided.

        Args:
          dims: The tensor dimensions.
        """
        self = Tensor[dtype](TensorSpec(dtype, dims))

    @always_inline
    fn __init__(inout self, owned shape: TensorShape):
        """Allocates a tensor using the shape provided.

        Args:
          shape: The tensor shape.
        """
        self = Tensor[dtype](TensorSpec(dtype, shape ^))

    @always_inline
    fn __init__(inout self, owned spec: TensorSpec):
        """Allocates a tensor using the spec provided.

        Args:
          spec: The tensor spec.
        """
        let num_elements = spec.num_elements()
        self._spec = spec
        self._ptr = DTypePointer[dtype].alloc(num_elements)
        memset_zero(self._ptr, num_elements)

    @always_inline
    fn __init__(
        inout self, owned ptr: DTypePointer[dtype], owned shape: TensorShape
    ):
        """Initializes a Tensor from the pointer and shape provided. The caller
        relinquishes the ownership of the pointer being passed in.

        Args:
          ptr: The data pointer.
          shape: The tensor shapes.
        """
        self = Tensor[dtype](ptr, TensorSpec(dtype, shape ^))

    @always_inline
    fn __init__(
        inout self, owned ptr: DTypePointer[dtype], owned spec: TensorSpec
    ):
        """Initializes a Tensor from the pointer and shape provided. The caller
        relinquishes the ownership of the pointer being passed in.

        Args:
          ptr: The data pointer.
          spec: The tensor spec.
        """
        self._spec = spec ^
        self._ptr = ptr

    @always_inline
    fn __init__(inout self, shape: TensorShape, *data: SIMD[dtype, 1]):
        """Initializes a Tensor from the shape and data provided.
        The caller assumes ownership of the new tensor data.

        Args:
          shape: The tensor shape.
          data: Elements to place into the created tensor.
        """
        let ptr = DTypePointer[dtype].alloc(len(data))
        for i in range(len(data)):
            ptr.store(i, data[i])
        self.__init__(ptr, shape)

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
        let num_elements = other.num_elements()
        self._spec = other._spec
        self._ptr = DTypePointer[dtype].alloc(num_elements)
        memcpy(self._ptr, other._ptr, num_elements)

    fn __moveinit__(inout self, owned existing: Self):
        """Move initializer for the tensor.

        Args:
            existing: The tensor to move.
        """
        self._spec = existing._spec ^
        self._ptr = existing._ptr
        existing._spec = TensorSpec()
        existing._ptr = DTypePointer[dtype]()

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
    fn __add__(self, other: SIMD[dtype, 1]) -> Self:
        """Adds this tensor with a scalar.

        Args:
            other: The RHS of the add operation.

        Returns:
            The addition result.
        """
        return _elementwise[math.add](self, other)

    @always_inline
    fn __radd__(self, other: SIMD[dtype, 1]) -> Self:
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
    fn __sub__(self, other: SIMD[dtype, 1]) -> Self:
        """Subtracts a scalar from this tensor.

        Args:
            other: The RHS of the sub operation.

        Returns:
            The subtraction result.
        """
        return _elementwise[math.sub](self, other)

    @always_inline
    fn __rsub__(self, other: SIMD[dtype, 1]) -> Self:
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
    fn __mul__(self, other: SIMD[dtype, 1]) -> Self:
        """Multiplies this tensor with a scalar.

        Args:
            other: The RHS of the mul operation.

        Returns:
            The multiplication result.
        """
        return _elementwise[math.mul](self, other)

    @always_inline
    fn __rmul__(self, other: SIMD[dtype, 1]) -> Self:
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
    fn __truediv__(self, other: SIMD[dtype, 1]) -> Self:
        """Divides this tensor by a scalar.

        Args:
            other: The RHS of the div operation.

        Returns:
            The division result.
        """
        return _elementwise[math.div](self, other)

    @always_inline
    fn __rtruediv__(self, other: SIMD[dtype, 1]) -> Self:
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
        let result = self
        let buffer = result._to_buffer()

        # Define an elementwise pow that captures and modifies `buffer`.
        @parameter
        fn _pow[width: Int, rank: Int](indices: StaticIntTuple[rank]) -> None:
            let idx = indices[0]
            let val = buffer.simd_load[width](idx)
            let res = math.pow(val, exponent)
            buffer.simd_store(idx, res)

        # Use the `elementwise` generator to run `pow` in parallel.
        alias dtype_simd_width = simdwidthof[dtype]()

        elementwise[rank=1, simd_width=dtype_simd_width, func=_pow](
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
        let result = Tensor[new_dtype](self._spec)
        let buffer = self._to_buffer()
        let result_buffer = result._to_buffer()

        @parameter
        fn func[width: Int, rank: Int](indices: StaticIntTuple[rank]):
            let idx = indices[0]
            result_buffer.simd_store(
                idx, buffer.simd_load[width](idx).cast[new_dtype]()
            )

        elementwise[rank=1, simd_width = simdwidthof[dtype](), func=func](
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
        fn serialize(val: _Printable):
            res += val.__str__()

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.data(), self.shape()
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
    fn __getitem__(self, index: Int) -> SIMD[dtype, 1]:
        """Gets the value at the specified index.

        Args:
          index: The index of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        return self._ptr.load(index)

    @always_inline
    fn __getitem__(self, *indices: Int) -> SIMD[dtype, 1]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.simd_load[1](indices)

    @always_inline
    fn __getitem__(self, indices: VariadicList[Int]) -> SIMD[dtype, 1]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.simd_load[1](indices)

    @always_inline
    fn __getitem__[
        len: Int
    ](self, indices: StaticIntTuple[len]) -> SIMD[dtype, 1]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          len: The length of the indecies.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.simd_load[1](indices)

    @always_inline
    fn simd_load[simd_width: Int](self, index: Int) -> SIMD[dtype, simd_width]:
        """Gets the SIMD value at the specified index.

        Parameters:
          simd_width: The SIMD width of the vector.

        Args:
          index: The index of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        return self._ptr.simd_load[simd_width](index)

    @always_inline
    fn simd_load[
        simd_width: Int
    ](self, *indices: Int) -> SIMD[dtype, simd_width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          simd_width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        return self.simd_load[simd_width](indices)

    @always_inline
    fn simd_load[
        simd_width: Int
    ](self, indices: VariadicList[Int]) -> SIMD[dtype, simd_width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          simd_width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(len(indices) == self.rank(), "invalid rank value")
        return self._ptr.simd_load[simd_width](
            self._compute_linear_offset(indices)
        )

    @always_inline
    fn simd_load[
        simd_width: Int, len: Int
    ](self, indices: StaticIntTuple[len]) -> SIMD[dtype, simd_width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          simd_width: The SIMD width of the vector.
          len: The length of the indecies.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(len == self.rank(), "invalid length value")
        return self._ptr.simd_load[simd_width](
            self._compute_linear_offset(indices)
        )

    @always_inline
    fn __setitem__(inout self, index: Int, val: SIMD[dtype, 1]):
        """Sets the value at the specified index.

        Args:
          index: The index of the value to set.
          val: The value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        self.simd_store[1](index, val)

    @always_inline
    fn __setitem__(inout self, indices: VariadicList[Int], val: SIMD[dtype, 1]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.simd_store[1](indices, val)

    @always_inline
    fn __setitem__[
        len: Int
    ](inout self, indices: StaticIntTuple[len], val: SIMD[dtype, 1]):
        """Sets the value at the specified indices.

        Parameters:
          len: The length of the indecies.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.simd_store[1, len](indices, val)

    @always_inline
    fn simd_store[
        simd_width: Int
    ](inout self, index: Int, val: SIMD[dtype, simd_width]):
        """Sets the SIMD value at the specified index.

        Parameters:
          simd_width: The SIMD width of the vector.

        Args:
          index: The index of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        self._ptr.simd_store[simd_width](index, val)

    @always_inline
    fn simd_store[
        simd_width: Int
    ](inout self, indices: VariadicList[Int], val: SIMD[dtype, simd_width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          simd_width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(len(indices) == self.rank(), "invalid rank value")
        self._ptr.simd_store[simd_width](
            self._compute_linear_offset(indices), val
        )

    @always_inline
    fn simd_store[
        simd_width: Int, len: Int
    ](inout self, indices: StaticIntTuple[len], val: SIMD[dtype, simd_width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          simd_width: The SIMD width of the vector.
          len: The length of the indecies.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(len == self.rank(), "invalid length value")
        self._ptr.simd_store[simd_width](
            self._compute_linear_offset(indices), val
        )

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
        let rank = len(indices)
        var result = indices[0]
        for i in range(rank - 1):
            result = self.dim(i + 1) * result + indices[i + 1]
        return result

    @always_inline
    fn _to_ndbuffer[
        rank: Int
    ](self) -> NDBuffer[rank, DimList.create_unknown[rank](), dtype]:
        debug_assert(
            rank == self.rank(), "to_ndbuffer rank must match Tensor rank"
        )
        var shape = StaticIntTuple[rank](0)

        @unroll
        for i in range(rank):
            shape[i] = self.dim(i)

        return NDBuffer[rank, DimList.create_unknown[rank](), dtype](
            self._ptr, shape
        )

    @always_inline
    fn _to_buffer(self) -> Buffer[Dim(), dtype]:
        return Buffer[Dim(), dtype](self._ptr, self.num_elements())

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
        let ptr = self._ptr
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
        var byte_tensor = path.read_bytes()
        let num_elements = byte_tensor.num_elements()
        return Tensor(
            bitcast[dtype](byte_tensor._steal_ptr()),
            num_elements // dtype.sizeof(),
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
        let bytes = path.read_bytes()
        let minimum_size = len(_SERIALIZATION_HEADER) + (3 * sizeof[UInt32]())

        if bytes.num_elements() < minimum_size:
            raise "given file is not a serialized mojo tensor."

        for i in range(len(_SERIALIZATION_HEADER)):
            if bytes[i] != _SERIALIZATION_HEADER[i]:
                raise "given file is not a serialized mojo tensor."

        fn _uint32_from_bytes(data: DTypePointer[DType.int8]) -> UInt32:
            let ptr = data._as_scalar_pointer()
            let spec_ptr = bitcast[UInt32](ptr)
            return __get_address_as_owned_value(spec_ptr.address)

        let major_format_ptr = bytes.data() + len(_SERIALIZATION_HEADER)
        let major_format = _uint32_from_bytes(major_format_ptr)
        let minor_format_ptr = major_format_ptr + sizeof[UInt32]()
        let minor_format = _uint32_from_bytes(minor_format_ptr)
        if (
            major_format != _SERIALIZATION_MAJOR_FORMAT
            or minor_format != _SERIALIZATION_MINOR_FORMAT
        ):
            raise "cannot load tensor of format: " + String(
                major_format
            ) + "." + String(minor_format)

        let spec_size_ptr = minor_format_ptr + sizeof[UInt32]()
        let spec_size = _uint32_from_bytes(spec_size_ptr)
        if spec_size != sizeof[TensorSpec]():
            raise "invalid tensor spec."
        let spec_ptr = spec_size_ptr + sizeof[UInt32]()
        let spec = TensorSpec.from_bytes(spec_ptr)
        if dtype != spec.dtype():
            raise "requested type doesn't match the dtype in serialized tensor."
        let data = spec_ptr + sizeof[TensorSpec]()
        let tensor = Tensor[dtype](spec)
        if spec.num_elements() == 0:
            return tensor
        memcpy(tensor.data(), bitcast[dtype](data), spec.num_elements())
        _ = bytes ^
        return tensor
