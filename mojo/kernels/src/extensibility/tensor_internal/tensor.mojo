# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the `Tensor` type - a tensor type which owns its underlying data.
"""

from collections import List
from pathlib import Path
from random import rand, randn
from sys import simdwidthof, sizeof

from algorithm.functional import elementwise, vectorize
from buffer import NDBuffer
from buffer.dimlist import Dim
from memory import UnsafePointer, bitcast, memcmp, memcpy, memset_zero

from utils import IndexList
from utils._serialize import _serialize
from utils.index import Index
from utils.loop import unroll
from utils.static_tuple import StaticTuple

from .tensor_shape import TensorShape
from .tensor_spec import TensorSpec

# ===-----------------------------------------------------------------------===#
# Tensor
# ===-----------------------------------------------------------------------===#


@value
struct Tensor[type: DType](
    Stringable,
    Writable,
    CollectionElement,
    EqualityComparable,
):
    """A tensor type which owns its underlying data and is parameterized on
    DType.

    Example:

    ```mojo
    from max.tensor import Tensor, TensorSpec, TensorShape
    from utils.index import Index

    def main():
        height = 256
        width = 256
        channels = 3

        # Create the tensor of dimensions height, width, channels
        # and fill with random values.
        image = Tensor[DType.float32].rand(TensorShape(height, width, channels))

        # Declare the grayscale image.
        spec = TensorSpec(DType.float32, height, width)
        gray_scale_image = Tensor[DType.float32](spec)

        # Perform the RGB to grayscale transform.
        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
                gray_scale_image[Index(y, x)] = 0.299 * r + 0.587 * g + 0.114 * b

        print(gray_scale_image.shape())
    ```


    Parameters:
      type: The underlying element type of the tensor.
    """

    var _spec: TensorSpec
    """The underlying specification of the tensor."""
    var _ptr: UnsafePointer[Scalar[type]]
    """The underlying data of the tensor."""

    @always_inline
    fn __init__(out self):
        """Default initializer for TensorShape."""
        self._spec = TensorSpec()
        self._ptr = UnsafePointer[Scalar[type]]()

    @always_inline
    @implicit
    fn __init__(out self, other: Self):
        """Creates a deep copy of an existing tensor.

        Args:
            other: The tensor to copy from.
        """
        var num_elements = other.num_elements()
        self._spec = other._spec
        self._ptr = UnsafePointer[Scalar[type]].alloc(num_elements)
        memcpy(self._ptr, other._ptr, num_elements)

    @always_inline
    @implicit
    fn __init__(out self, *dims: Int):
        """Allocates a tensor using the shape provided.

        Args:
          dims: The tensor dimensions.
        """
        self = Self(TensorSpec(type, dims))

    @always_inline
    @implicit
    fn __init__(out self, owned shape: TensorShape):
        """Allocates a tensor using the shape provided.

        Args:
          shape: The tensor shape.
        """
        self = Self(TensorSpec(type, shape^))

    @always_inline
    @implicit
    fn __init__(out self, owned spec: TensorSpec):
        """Allocates a tensor using the spec provided.

        Args:
          spec: The tensor spec.
        """
        var num_elements = spec.num_elements()
        self._spec = spec
        self._ptr = UnsafePointer[Scalar[type]].alloc(num_elements)
        memset_zero(self._ptr, num_elements)

    @always_inline
    @implicit
    fn __init__(out self, shape: Tuple):
        """Allocates a tensor using the shape provided.

        Args:
          shape: The tensor shape.
        """
        self._spec = TensorSpec(type, shape)
        var num_elements = self._spec.num_elements()
        self._ptr = UnsafePointer[Scalar[type]].alloc(num_elements)
        memset_zero(self._ptr, num_elements)

    @always_inline
    fn __init__(
        mut self,
        owned shape: TensorShape,
        owned ptr: UnsafePointer[Scalar[type]],
    ):
        """Initializes a Tensor from the pointer and shape provided. The caller
        relinquishes the ownership of the pointer being passed in.

        Args:
          shape: The tensor shapes.
          ptr: The data pointer.
        """
        self = Self(TensorSpec(type, shape^), ptr)

    @always_inline
    fn __init__(
        mut self,
        owned spec: TensorSpec,
        owned ptr: UnsafePointer[Scalar[type]],
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
    fn __init__(out self, shape: TensorShape, *data: Scalar[type]):
        """Initializes a Tensor from the shape and data provided. If a single
        scalar is passed in, then the scalar is splatted to all elements in the
        tensor.

        Args:
          shape: The tensor shape.
          data: Elements to place into the created tensor.
        """
        var num_elements = shape.num_elements()
        var ptr = UnsafePointer[Scalar[type]].alloc(num_elements)
        if len(data) == 1:
            var data0 = data[0]

            if data0:

                @parameter
                fn splat_val[simd_width: Int](idx: Int):
                    ptr.store(idx, SIMD[type, simd_width](data0))

                vectorize[splat_val, simdwidthof[type](), unroll_factor=8](
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
        mut self, shape: TensorShape, owned list: List[Scalar[type], *_]
    ):
        """Initializes a 1-dimensional Tensor from the provided list.

        Args:
            shape: The tensor shape.
            list: The list to construct this Tensor from.
        """
        # Store the list length before we do a wiping take from it
        var data_anyptr = list.steal_data()

        self = Self(shape, data_anyptr)

    @always_inline
    @implicit
    fn __init__(out self, owned list: List[Scalar[type], *_]):
        """Initializes a 1-dimensional Tensor from the provided list.

        Args:
            list: The list to construct this Tensor from.
        """
        # Store the list length before we do a wiping take from it
        var list_len = len(list)

        var data_anyptr = list.steal_data()

        self = Self(TensorShape(list_len), data_anyptr)

    @always_inline
    fn __del__(owned self):
        """Delete the spec and release any owned memory."""
        if self._ptr:
            self._ptr.free()

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Creates a deep copy of an existing tensor.

        Args:
            other: The tensor to copy from.
        """
        var num_elements = other.num_elements()
        self._spec = other._spec
        self._ptr = UnsafePointer[Scalar[type]].alloc(num_elements)
        memcpy(self._ptr, other._ptr, num_elements)

    @staticmethod
    fn rand(owned shape: TensorShape) -> Tensor[type]:
        """Constructs a new tensor with the specified shape and fills it with
        random elements.

        Args:
            shape: The tensor shape.

        Returns:
            A new tensor of specified shape and filled with random elements.
        """
        var tensor = Tensor[type](shape^)
        rand(tensor.unsafe_ptr(), tensor.num_elements())
        return tensor

    @staticmethod
    fn randn(
        owned shape: TensorShape, mean: Float64 = 0.0, variance: Float64 = 1.0
    ) -> Tensor[type]:
        """Constructs a new Tensor from the shape and fills it with random
        values from a Normal(mean, variance) distribution.

        Constraints:
            The type should be floating point.

        Args:
            shape: The shape of the Tensor to fill with random values.
            mean: Normal distribution mean.
            variance: Normal distribution variance.

        Returns:
            A Tensor filled with random dtype samples from Normal(mean,
            variance).
        """

        var tensor = Tensor[type](shape^)
        randn(tensor.unsafe_ptr(), tensor.num_elements(), mean, variance)
        return tensor

    fn _take_data_ptr(mut self) -> UnsafePointer[Scalar[type]]:
        """Return ownership of the data pointer from within the Tensor.
        Returns:
            A pointer that owns the underlying buffer.
        """

        var result = self._ptr
        self._ptr = UnsafePointer[Scalar[type]]()
        return result

    @always_inline
    fn ireshape(mut self, new_shape: TensorShape) raises -> None:
        """(Inplace) Reshapes the tensor by assigning it a new shape.

        Args:
            new_shape: The new shape.
        """
        if new_shape.num_elements() != self.num_elements():
            raise "Number of elements must match in reshape"

        self._spec = TensorSpec(type, new_shape)

    @always_inline
    fn reshape(mut self, new_shape: TensorShape) raises -> Tensor[type]:
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

        return (
            memcmp(
                self.unsafe_ptr(),
                other.unsafe_ptr(),
                self.num_elements(),
            )
            == 0
        )

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
    fn astype[new_type: DType](self) raises -> Tensor[new_type]:
        """Copy the Tensor with elements cast to the new type.

        Parameters:
            new_type: The type to cast the values to.

        Returns:
            A new tensor with the same values but the new type.
        """
        var result = Tensor[new_type](self._spec.shape)
        var buffer = self._to_buffer()
        var result_buffer = result._to_buffer()

        @__copy_capture(result_buffer, buffer)
        @parameter
        fn func[width: Int, rank: Int](indices: IndexList[rank]):
            var idx = indices[0]
            result_buffer.store(
                idx, buffer.load[width=width](idx).cast[new_type]()
            )

        elementwise[func=func, simd_width = simdwidthof[type]()](
            Index(len(buffer))
        )

        return result

    @always_inline
    fn clip(
        self,
        lower_bound: Scalar[type],
        upper_bound: Scalar[type],
    ) raises -> Self:
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

        @parameter
        fn func[width: Int, rank: Int](indices: IndexList[rank]):
            var idx = indices[0]
            result_buffer.store(
                idx,
                buffer.load[width=width](idx).clamp(lower_bound, upper_bound),
            )

        elementwise[func=func, simd_width = simdwidthof[type]()](
            Index(len(buffer))
        )

        return result

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Scalar[type]]:
        """Gets the underlying Data pointer to the Tensor.

        Returns:
          The underlying data pointer of the tensor.
        """
        return self._ptr

    @always_inline
    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        """Gets the underlying Data pointer to the Tensor.

        Returns:
          The underlying data pointer of the tensor.
        """
        return rebind[UnsafePointer[UInt8]](self._ptr)

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

        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this Tensor to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("Tensor(")

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int, hint_trivial_type=True]()
        for i in range(self.rank()):
            shape.append(self.shape()[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.unsafe_ptr(), shape
        )

        writer.write(")")

    @no_inline
    fn __repr__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string representation of the tensor.
        """
        return self.__str__()

    @always_inline
    fn __getitem__(self, index: Int) -> Scalar[type]:
        """Gets the value at the specified index.

        Args:
          index: The index of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        return self._ptr.load(index)

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn __getitem__(self, indices: VariadicList[Int]) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn __getitem__[len: Int](self, indices: IndexList[len]) -> Scalar[type]:
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
    fn load[*, width: Int = 1](self, index: Int) -> SIMD[type, width]:
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
    fn load[*, width: Int = 1](self, *indices: Int) -> SIMD[type, width]:
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
    ](self, indices: VariadicList[Int]) -> SIMD[type, width]:
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
    ](self, indices: IndexList[len]) -> SIMD[type, width]:
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
    fn __setitem__(mut self, index: Int, val: Scalar[type]):
        """Sets the value at the specified index.

        Args:
          index: The index of the value to set.
          val: The value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        self.store[width=1](index, val)

    @always_inline
    fn __setitem__(mut self, indices: VariadicList[Int], val: Scalar[type]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.store[width=1](indices, val)

    @always_inline
    fn __setitem__[
        len: Int
    ](mut self, indices: IndexList[len], val: Scalar[type]):
        """Sets the value at the specified indices.

        Parameters:
          len: The length of the indecies.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.store[len, width=1](indices, val)

    @always_inline
    fn store[*, width: Int = 1](mut self, index: Int, val: SIMD[type, width]):
        """Sets the SIMD value at the specified index.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          index: The index of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(self.rank() == 1, "rank must be 1")
        self._ptr.store(index, val)

    @always_inline
    fn store[
        *, width: Int = 1
    ](mut self, indices: VariadicList[Int], val: SIMD[type, width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(len(indices) == self.rank(), "invalid rank value")
        self._ptr.store(self._compute_linear_offset(indices), val)

    @always_inline
    fn store[
        len: Int, /, *, width: Int = 1
    ](mut self, indices: IndexList[len], val: SIMD[type, width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          len: The length of the indecies.
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        debug_assert(len == self.rank(), "invalid length value")
        self._ptr.store(self._compute_linear_offset(indices), val)

    @always_inline
    fn _compute_linear_offset[rank: Int](self, indices: IndexList[rank]) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Parameters:
          rank: The rank of the indices.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        var result = indices[0]

        @parameter
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
    fn _to_ndbuffer[rank: Int](self) -> NDBuffer[type, rank, __origin_of(self)]:
        debug_assert(
            rank == self.rank(), "to_ndbuffer rank must match Tensor rank"
        )
        var shape = IndexList[rank](0)

        @parameter
        for i in range(rank):
            shape[i] = self.dim(i)

        return NDBuffer[type, rank](self._ptr, shape)

    @always_inline
    fn _to_buffer(ref self) -> NDBuffer[type, 1, __origin_of(self)]:
        return NDBuffer[type, 1](self._ptr, self.num_elements())

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        self._to_buffer().tofile(path)

    @always_inline
    fn _steal_ptr(mut self) -> UnsafePointer[Scalar[type]]:
        """Transfer ownership of pointer to the underlying memory.
        The caller is responsible for freeing up the memory.

        Returns:
            The pointer to the underlying memory.
        """
        var ptr = self._ptr
        self._ptr = UnsafePointer[Scalar[type]]()
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
        var byte_tensor = Tensor[DType.uint8](path.read_bytes())
        var num_elements = byte_tensor.num_elements()
        return Self(
            num_elements // sizeof[type](),
            byte_tensor._steal_ptr().bitcast[Scalar[type]](),
        )

    fn save(self, path: Path) raises:
        """Save given tensor to file. This method preserves
           shape and datatype information.

        Args:
          path: Path of file.
        """
        _serialize_to_file(self, path)

    @staticmethod
    fn load(path: Path) raises -> Tensor[type]:
        """Read tensor from a file.
           The path should be a file saved with Tensor.save method.

        Args:
          path: Path to the output file.

        Returns:
          The tensor read from file.
        """
        var bytes = Tensor[DType.uint8](path.read_bytes())
        var minimum_size = len(_SERIALIZATION_HEADER) + (3 * sizeof[UInt32]())

        if bytes.num_elements() < minimum_size:
            raise "given file is not a serialized mojo tensor."

        for i in range(len(_SERIALIZATION_HEADER)):
            if bytes[i] != _SERIALIZATION_HEADER[i]:
                raise "given file is not a serialized mojo tensor."

        fn _uint32_from_bytes(data: UnsafePointer[UInt8]) -> UInt32:
            var ptr = data.bitcast[UInt32]()
            return UnsafePointer(ptr).take_pointee()

        var major_format_ptr = bytes.unsafe_ptr() + len(_SERIALIZATION_HEADER)
        var major_format = _uint32_from_bytes(major_format_ptr)
        var minor_format_ptr = major_format_ptr + sizeof[UInt32]()
        var minor_format = _uint32_from_bytes(minor_format_ptr)
        if (
            major_format != _SERIALIZATION_MAJOR_FORMAT
            or minor_format != _SERIALIZATION_MINOR_FORMAT
        ):
            raise String(
                "cannot load tensor of format: ",
                major_format,
                ".",
                minor_format,
            )

        var spec_size_ptr = minor_format_ptr + sizeof[UInt32]()
        var spec_size = _uint32_from_bytes(spec_size_ptr)
        if spec_size != sizeof[TensorSpec]():
            raise "invalid tensor spec."
        var spec_ptr = spec_size_ptr + sizeof[UInt32]()
        var spec = TensorSpec.from_bytes(spec_ptr)
        if type != spec.dtype():
            raise "requested type doesn't match the type in serialized tensor."
        var data = spec_ptr + sizeof[TensorSpec]()
        var tensor = Self(spec)
        if spec.num_elements() == 0:
            return tensor
        memcpy(
            tensor.unsafe_ptr(),
            data.bitcast[Scalar[type]](),
            spec.num_elements(),
        )
        return tensor


# ===-----------------------------------------------------------------------===#
# serialize
# ===-----------------------------------------------------------------------===#

# Serialization constants
alias _SERIALIZATION_MAJOR_FORMAT: UInt32 = 0
alias _SERIALIZATION_MINOR_FORMAT: UInt32 = 1
# 0x93 🔥 0x93
alias _SERIALIZATION_HEADER = StaticTuple[UInt8, 6](
    0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x93
)


fn _serialize_as_tensor[type: AnyType](ref object: type) -> Tensor[DType.uint8]:
    """Serialize the given object into a Tensor of bytes.

    Args:
      object: Object to serialize.

    Returns:
      Tensor containing the bytes of object.
    """
    var self_ptr = UnsafePointer.address_of(object).bitcast[UInt8]()
    alias size = sizeof[type]()
    var bytes = Tensor[DType.uint8](size)
    memcpy(bytes.unsafe_ptr(), self_ptr, size)
    return bytes^


fn _serialize_to_file[type: DType](tensor: Tensor[type], path: Path) raises:
    """Serialize given tensor to file. This method preserves
       shape and datatype information.

    Args:
      tensor: Tensor to serialize.
      path: Path of file.
    """
    var header_size = len(_SERIALIZATION_HEADER)
    var header_bytes = Tensor[DType.uint8](header_size)

    for i in range(header_size):
        header_bytes.store(i, _SERIALIZATION_HEADER[i])

    var major_format: UInt32 = _SERIALIZATION_MAJOR_FORMAT
    var major_format_bytes = _serialize_as_tensor(major_format)
    var minor_format: UInt32 = _SERIALIZATION_MINOR_FORMAT
    var minor_format_bytes = _serialize_as_tensor(minor_format)
    var spec_size: UInt32 = sizeof[TensorSpec]()
    var spec_size_bytes = _serialize_as_tensor(spec_size)
    var spec = tensor.spec()
    var spec_bytes = _serialize_as_tensor(spec)

    var bytes = Tensor[DType.uint8](
        header_bytes.num_elements()
        + major_format_bytes.num_elements()
        + minor_format_bytes.num_elements()
        + spec_size_bytes.num_elements()
        + spec_bytes.num_elements()
        + tensor.num_elements() * sizeof[type]()
    )
    var copied: Int = 0

    @always_inline("nodebug")
    fn _copy_bytes(
        mut dest: Tensor[DType.uint8], offset: Int, src: Tensor[DType.uint8]
    ) -> Int:
        var size = src.num_elements()
        memcpy(
            dest.unsafe_ptr() + offset,
            src.unsafe_ptr(),
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
        bytes.unsafe_ptr() + copied,
        tensor.unsafe_ptr().bitcast[UInt8](),
        tensor.num_elements() * sizeof[type](),
    )
    copied += tensor.num_elements() * sizeof[type]()

    debug_assert(bytes.num_elements() == copied, "expected these to be same.")

    bytes.tofile(path)
