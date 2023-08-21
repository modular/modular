# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the TensorSpec, TensorShape, and Tensor type.

Example:

```mojo
from Tensor import Tensor, TensorSpec, TensorShape
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

from .tensor_shape import TensorShape
from .tensor_spec import TensorSpec

# ===----------------------------------------------------------------------===#
# Tensor
# ===----------------------------------------------------------------------===#


struct Tensor[dtype: DType]:
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
        return self[VariadicList[Int](indices)]

    @always_inline
    fn __getitem__(self, indices: VariadicList[Int]) -> SIMD[dtype, 1]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(indices.__len__() == self.rank(), "invalid rank value")
        return self._ptr.load(self._compute_linear_offset(indices))

    @always_inline
    fn __getitem__[
        rank: Int
    ](self, indices: StaticIntTuple[rank]) -> SIMD[dtype, 1]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(rank == self.rank(), "invalid rank value")
        return self._ptr.load(self._compute_linear_offset(indices))

    @always_inline
    fn __setitem__(inout self, index: Int, val: SIMD[dtype, 1]):
        """Sets the value at the specified index.

        Args:
          index: The index of the value to set.
          val: The value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        return self._ptr.store(index, val)

    @always_inline
    fn __setitem__(inout self, indices: VariadicList[Int], val: SIMD[dtype, 1]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        debug_assert(indices.__len__() == self.rank(), "invalid rank value")
        return self._ptr.store(self._compute_linear_offset(indices), val)

    @always_inline
    fn __setitem__[
        rank: Int
    ](inout self, indices: StaticIntTuple[rank], val: SIMD[dtype, 1]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        debug_assert(rank == self.rank(), "invalid rank value")
        return self._ptr.store(self._compute_linear_offset(indices), val)

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
        return self._compute_linear_offset(VariadicList[Int](indices))

    @always_inline
    fn _compute_linear_offset(self, indices: VariadicList[Int]) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        let rank = indices.__len__()
        var result = indices[0]
        for i in range(rank - 1):
            result = self.dim(i + 1) * result + indices[i + 1]
        return result
