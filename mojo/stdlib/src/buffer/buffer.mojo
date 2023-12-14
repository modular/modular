# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the Buffer class.

You can import these APIs from the `memory` package. For example:

```mojo
from memory.buffer import Buffer
```
"""

from math import fma, iota, max, min
from pathlib import Path
from sys.info import alignof, simdwidthof, sizeof, triple_is_nvidia_cuda
from sys.intrinsics import PrefetchOptions, masked_load, masked_store

from algorithm import unroll, vectorize
from runtime.llcl import OutputChainPtr

from utils._serialize import _serialize
from utils.index import StaticIntTuple
from utils.index import product as tuple_product
from utils.list import Dim, DimList
from utils.static_tuple import StaticTuple
from gpu.memory import AddressSpace as GPUAddressSpace

from .memory import stack_allocation
from .unsafe import DTypePointer, Pointer, AddressSpace

alias _MAX_RANK = 8
"""The maximum tensor rank for any tensor shape.
This value must match kMaxRank in Support/include/Support/ML/TensorShape.h
"""

# ===----------------------------------------------------------------------===#
# Buffer
# ===----------------------------------------------------------------------===#


# This type is "async safe" (see async_parallelize).
@value
@register_passable
struct Buffer[
    size: Dim,
    type: DType,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
](Sized):
    """Defines a Buffer which can be parametrized on a static size and Dtype.

    The Buffer does not own its underlying pointer.

    Parameters:
      size: The static size (if known) of the Buffer.
      type: The element type of the Buffer.
      address_space: The address space of the Buffer.
    """

    var data: DTypePointer[type, address_space]
    """The underlying data pointer of the data."""
    var dynamic_size: Int
    """The dynamic size of the buffer."""
    var dtype: DType
    """The dynamic data type of the buffer."""

    @always_inline
    fn __init__() -> Self:
        """Default initializer for Buffer. By default the fields are all
        initialized to 0.

        Returns:
            The NDBuffer object.
        """

        return Self {
            data: DTypePointer[type, address_space](),
            dynamic_size: 0,
            dtype: type,
        }

    @always_inline
    fn __init__(ptr: Pointer[SIMD[type, 1], address_space]) -> Self:
        """Constructs a Buffer with statically known size and type.

        Constraints:
            The size is known.

        Args:
            ptr: Pointer to the data.

        Returns:
            The buffer object.
        """
        # Construct a Buffer type with statically known size
        constrained[size.has_value(), "must have known size"]()
        return Self {data: ptr, dynamic_size: size.get(), dtype: type}

    @always_inline
    fn __init__(ptr: DTypePointer[type, address_space]) -> Self:
        """Constructs a Buffer with statically known size and type.

        Constraints:
            The size is known.

        Args:
            ptr: Pointer to the data.

        Returns:
            The buffer object.
        """
        # Construct a Buffer type with statically known size
        constrained[size.has_value(), "must have known size"]()
        return Self {data: ptr, dynamic_size: size.get(), dtype: type}

    @always_inline
    fn __init__(
        ptr: Pointer[SIMD[type, 1], address_space],
        in_size: Int,
    ) -> Self:
        """Constructs a Buffer with statically known type.

        Constraints:
            The size is unknown.

        Args:
            ptr: Pointer to the data.
            in_size: Dynamic size of the buffer.

        Returns:
            The buffer object.
        """

        @parameter
        if size:
            debug_assert(
                in_size == size.get(),
                "if static size is known, static size must equal dynamic size",
            )
        return Self {data: ptr, dynamic_size: in_size, dtype: type}

    @always_inline
    fn __init__(ptr: DTypePointer[type, address_space], in_size: Int) -> Self:
        """Constructs a Buffer with statically known type.

        Constraints:
            The size is unknown.

        Args:
            ptr: Pointer to the data.
            in_size: Dynamic size of the buffer.

        Returns:
            The buffer object.
        """

        @parameter
        if size:
            debug_assert(
                in_size == size.get(),
                "if static size is known, static size must equal dynamic size",
            )
        return Self {data: ptr, dynamic_size: in_size, dtype: type}

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size if it is a known constant, otherwise it gets the
        dynamic_size.

        This method is used by `Buffer.__len__` to get the size of the buffer.
        If the Buffer size is a known constant, then the size is returned.
        Otherwise, the dynamic_size is returned.

        Returns:
            The size if static otherwise dynamic_size.
        """

        @parameter
        if not size:
            return self.dynamic_size

        return size.get()

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[type, 1]:
        """Loads a single element (SIMD of size 1) from the buffer at the
        specified index.

        Args:
            idx: The index into the Buffer.

        Returns:
            The value at the `idx` position.
        """
        return self.simd_load[1](idx)

    @always_inline
    fn simd_load[width: Int](self, idx: Int) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Parameters:
            width: The simd_width of the load.

        Args:
            idx: The index into the Buffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.data.simd_load[width](idx)

    @always_inline
    fn aligned_simd_load[
        width: Int, alignment: Int
    ](self, idx: Int) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.data.aligned_simd_load[width, alignment](idx)

    @always_inline
    fn __setitem__(
        self,
        idx: Int,
        val: __mlir_type[`!pop.scalar<`, type.value, `>`],
    ):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.simd_store[1](idx, SIMD[type, 1](val))

    @always_inline
    fn __setitem__(self, idx: Int, val: SIMD[type, 1]):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.simd_store[1](idx, val)

    @always_inline
    fn simd_store[
        width: Int,
    ](self, idx: Int, val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Parameters:
            width: The width of the simd vector.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.data.simd_store[width](idx, val)

    @always_inline
    fn aligned_simd_store[
        width: Int, alignment: Int
    ](self, idx: Int, val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Parameters:
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.data.aligned_simd_store[width, alignment](idx, val)

    @always_inline
    fn simd_nt_store[width: Int](self, idx: Int, val: SIMD[type, width]):
        """Stores a simd value using non-temporal store.

        Constraints:
            The address must be properly aligned, 64B for avx512, 32B for avx2,
              and 16B for avx.

        Parameters:
            width: The width of the simd vector.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.data.simd_nt_store[width](idx, val)

    @always_inline
    fn prefetch[params: PrefetchOptions](self, idx: Int):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            idx: The index of the prefetched location.
        """
        self.data.offset(idx).prefetch[params]()

    @always_inline
    fn bytecount(self) -> Int:
        """Returns the size of the Buffer in bytes.

        Returns:
            The size of the Buffer in bytes.
        """
        return len(self) * sizeof[type]()

    @always_inline
    fn zero(self):
        """Sets all bytes of the Buffer to 0."""
        llvm_intrinsic["llvm.memset", NoneType](
            self.data.address,
            UInt8(0).value,
            self.bytecount().value,
            (False).__mlir_i1__(),
        )

    fn simd_fill[simd_width: Int](self, val: SIMD[type, 1]):
        """Assigns val to all elements in chunks of size simd_width.

        Parameters:
            simd_width: The simd_width of the fill.

        Args:
            val: The value to store.
        """
        if val == 0:
            self.zero()
            return

        @always_inline
        @parameter
        fn _fill[simd_width: Int](idx: Int):
            self.simd_store[simd_width](idx, SIMD[type, simd_width].splat(val))

        vectorize[simd_width, _fill](len(self))

    @always_inline
    fn fill(self, val: SIMD[type, 1]):
        """Assigns val to all elements in the Buffer.

        The fill is performed in chunks of size N, where N is the native SIMD
        width of type on the system.

        Args:
            val: The value to store.
        """
        self.simd_fill[simdwidthof[type]()](val)

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        with open(path.__str__(), "w") as f:
            f._write(self.data.bitcast[DType.int8](), self.bytecount())

    @staticmethod
    @always_inline("nodebug")
    fn aligned_stack_allocation[alignment: Int]() -> Self:
        """Constructs a buffer instance backed by stack allocated memory space.

        Parameters:
            alignment: Address alignment requirement for the allocation.

        Returns:
            Constructed buffer with the allocated space.
        """
        constrained[size.has_value(), "must have known size"]()
        let data_pointer = stack_allocation[
            size.get(), type, alignment=alignment, address_space=address_space
        ]()
        return Self(data_pointer)

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation() -> Self:
        """Constructs a buffer instance backed by stack allocated memory space.

        Returns:
            Constructed buffer with the allocated space.
        """
        return Self.aligned_stack_allocation[alignof[type]()]()


# ===----------------------------------------------------------------------===#
# NDBuffer Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _compute_nd_index[
    rank: Int,
    shape: DimList,
    type: DType,
    address_space: AddressSpace,
](
    buf: NDBuffer[rank, shape, type, address_space], index: Int
) -> StaticIntTuple[rank]:
    """Computes the NDBuffer's offset using the index positions provided.

    Parameters:
        rank: The rank of the NDBuffer.
        shape: The shape of the NDBuffer.
        type: The element-type of the NDBuffer.
        address_space: The address space of the NDBuffer.

    Args:
        buf: The NDBuffer.
        index: The flat index position.

    Returns:
        The index positions.
    """

    @parameter
    if rank == 0:
        return StaticIntTuple[rank](0)

    var result = StaticIntTuple[rank]()

    result[rank - 1] = index

    @unroll
    for idx in range(rank - 1):
        result[rank - idx - 2] = result[rank - idx - 1]._positive_div(
            buf.dim(rank - idx - 1)
        )
        result[rank - idx - 1] = result[rank - idx - 1]._positive_rem(
            buf.dim(rank - idx - 1)
        )
    return result


@always_inline
fn _compute_ndbuffer_offset[
    rank: Int, shape: DimList, type: DType, address_space: AddressSpace
](
    buf: NDBuffer[rank, shape, type, address_space=address_space],
    index: VariadicList[Int],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Parameters:
        rank: The rank of the NDBuffer.
        shape: The shape of the NDBuffer.
        type: The element-type of the NDBuffer.
        address_space: The address space of the NDBuffer.

    Args:
        buf: The NDBuffer.
        index: The index positions.

    Returns:
        The offset into the NDBuffer given the indices.
    """

    @parameter
    if rank == 0:
        return 0

    @parameter
    if triple_is_nvidia_cuda() and address_space == GPUAddressSpace.SHARED:
        var result: Int32 = 0

        @unroll
        for i in range(rank):
            result = fma(Int32(buf.stride(i)), Int32(index[i]), result)

        return int(result)

    else:
        var result: Int = 0

        @unroll
        for i in range(rank):
            result = fma(buf.stride(i), index[i], result)

        return result


@always_inline
fn _compute_ndbuffer_offset[
    rank: Int, shape: DimList, type: DType, address_space: AddressSpace
](
    buf: NDBuffer[rank, shape, type, address_space=address_space],
    idx: StaticIntTuple[rank],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Parameters:
        rank: The rank of the NDBuffer.
        shape: The shape of the NDBuffer.
        type: The element-type of the NDBuffer.
        address_space: The address space of the NDBuffer.

    Args:
        buf: The NDBuffer.
        idx: The index positions.

    Returns:
        The offset into the NDBuffer given the indices.
    """
    return _compute_ndbuffer_offset(buf, idx.as_tuple())


@always_inline
fn _compute_ndbuffer_offset[
    rank: Int, shape: DimList, type: DType, address_space: AddressSpace
](
    buf: NDBuffer[rank, shape, type, address_space=address_space],
    index: StaticTuple[rank, Int],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Parameters:
        rank: The rank of the NDBuffer.
        shape: The shape of the NDBuffer.
        type: The element-type of the NDBuffer.
        address_space: The address space of the NDBuffer.

    Args:
        buf: The NDBuffer.
        index: The index positions.

    Returns:
        The offset into the NDBuffer given the indices.
    """

    @parameter
    if rank == 0:
        return 0

    @parameter
    if triple_is_nvidia_cuda() and address_space == GPUAddressSpace.SHARED:
        var result: Int32 = 0

        @unroll
        for i in range(rank):
            result = fma(Int32(buf.stride(i)), Int32(index[i]), result)

        return int(result)

    else:
        var result: Int = 0

        @unroll
        for i in range(rank):
            result = fma(buf.stride(i), index[i], result)

        return result


@always_inline
fn _compute_ndbuffer_stride[
    rank: Int
](shape: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    """Computes the NDBuffer's default dynamic strides using the input shape.
    The default strides correspond to contiguous memory layout.

    Parameters:
        rank: The rank of the NDBuffer.

    Args:
        shape: The shape of the NDBuffer.

    Returns:
        The default strides of the NDBuffer.
    """
    constrained[rank > 0]()

    @parameter
    if rank == 1:
        return StaticIntTuple[rank](1)

    var stride = shape
    stride[rank - 1] = 1

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias i = rank - idx - 1
        stride[i - 1] = shape[i] * stride[i]

    unroll[rank - 1, body]()
    return stride


# ===----------------------------------------------------------------------===#
# NDBuffer
# ===----------------------------------------------------------------------===#


# This type is "async safe" (see async_parallelize).
@value
@register_passable("trivial")
struct NDBuffer[
    rank: Int,
    shape: DimList,
    type: DType,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
](Sized, Stringable):
    """An N-dimensional Buffer.

    NDBuffer can be parametrized on rank, static dimensions and Dtype. It does
    not own its underlying pointer.

    Parameters:
        rank: The rank of the buffer.
        shape: The static size (if known) of the buffer.
        type: The element type of the buffer.
        address_space: The address space of the buffer.
    """

    var data: DTypePointer[type, address_space]
    """The underlying data for the buffer. The pointer is not owned by the
    NDBuffer."""
    var dynamic_shape: StaticIntTuple[rank]
    """The dynamic value of the shape."""
    var dynamic_stride: StaticIntTuple[rank]
    """The dynamic stride of the buffer."""
    var is_contiguous: Bool
    """True if the contents of the buffer are contiguous in memory."""

    @always_inline
    fn __init__() -> Self:
        """Default initializer for NDBuffer. By default the fields are all
        initialized to 0.

        Returns:
            The NDBuffer object.
        """

        return Self {
            data: DTypePointer[type, address_space](),
            dynamic_shape: StaticIntTuple[rank](),
            dynamic_stride: StaticIntTuple[rank](),
            is_contiguous: False,
        }

    @always_inline
    fn __init__(
        ptr: Pointer[SIMD[type, 1], address_space],
    ) -> Self:
        """Constructs an NDBuffer with statically known rank, shapes and
        type.

        Constraints:
            The rank, shapes, and type are known.

        Args:
            ptr: Pointer to the data.

        Returns:
            The NDBuffer object.
        """
        constrained[
            shape.all_known[rank](),
            "dimensions must all be known",
        ]()

        return Self {
            data: ptr,
            dynamic_shape: shape,
            dynamic_stride: _compute_ndbuffer_stride[rank](shape),
            is_contiguous: True,
        }

    @always_inline
    fn __init__(
        ptr: DTypePointer[type, address_space],
    ) -> Self:
        """Constructs an NDBuffer with statically known rank, shapes and
        type.

        Constraints:
            The rank, shapes, and type are known.

        Args:
            ptr: Pointer to the data.

        Returns:
            The NDBuffer object.
        """
        constrained[
            shape.all_known[rank](),
            "dimensions must all be known",
        ]()

        return Self {
            data: ptr,
            dynamic_shape: shape,
            dynamic_stride: _compute_ndbuffer_stride[rank](shape),
            is_contiguous: True,
        }

    # @always_inline
    # fn __init__(
    #     ptr: __mlir_type[
    #         `!kgen.pointer<scalar<`,
    #         type.value,
    #         `>`,
    #         address_space.value().value,
    #         `>`,
    #     ],
    #     dynamic_shape: StaticIntTuple[rank],
    # ) -> Self:
    #     """Constructs an NDBuffer with statically known rank, but dynamic
    #     shapes and type.

    #     Constraints:
    #         The rank is known.

    #     Args:
    #         ptr: Pointer to the data.
    #         dynamic_shape: A static tuple of size 'rank' representing shapes.

    #     Returns:
    #         The NDBuffer object.
    #     """
    #     return Self {
    #         data: ptr,
    #         dynamic_shape: dynamic_shape,
    #         dynamic_stride: _compute_ndbuffer_stride[rank](dynamic_shape),
    #         is_contiguous: True,
    #     }

    @always_inline
    fn __init__(
        ptr: Pointer[
            __mlir_type[`!pop.scalar<`, type.value, `>`], address_space
        ],
        dynamic_shape: StaticIntTuple[rank],
    ) -> Self:
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.

        Returns:
            The NDBuffer object.
        """
        return Self {
            data: ptr.bitcast[SIMD[type, 1]](),
            dynamic_shape: dynamic_shape,
            dynamic_stride: _compute_ndbuffer_stride[rank](dynamic_shape),
            is_contiguous: True,
        }

    @always_inline
    fn __init__(
        ptr: DTypePointer[type, address_space],
        dynamic_shape: StaticIntTuple[rank],
    ) -> Self:
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.

        Returns:
            The NDBuffer object.
        """
        return Self {
            data: ptr,
            dynamic_shape: dynamic_shape,
            dynamic_stride: _compute_ndbuffer_stride[rank](dynamic_shape),
            is_contiguous: True,
        }

    @always_inline
    fn __init__(
        ptr: Pointer[SIMD[type, 1], address_space],
        dynamic_shape: StaticIntTuple[rank],
        dynamic_stride: StaticIntTuple[rank],
    ) -> Self:
        """Constructs a strided NDBuffer with statically known rank, but
        dynamic shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.

        Returns:
            The NDBuffer object.
        """
        return Self {
            data: ptr,
            dynamic_shape: dynamic_shape,
            dynamic_stride: dynamic_stride,
            is_contiguous: _compute_ndbuffer_stride[rank](dynamic_shape)
            == dynamic_stride,
        }

    @always_inline
    fn __init__(
        ptr: DTypePointer[type, address_space],
        dynamic_shape: StaticIntTuple[rank],
        dynamic_stride: StaticIntTuple[rank],
    ) -> Self:
        """Constructs a strided NDBuffer with statically known rank, but
        dynamic shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.

        Returns:
            The NDBuffer object.
        """
        return Self {
            data: ptr,
            dynamic_shape: dynamic_shape,
            dynamic_stride: dynamic_stride,
            is_contiguous: _compute_ndbuffer_stride[rank](dynamic_shape)
            == dynamic_stride,
        }

    @always_inline
    fn get_rank(self) -> Int:
        """Returns the rank of the buffer.

        Returns:
            The rank of NDBuffer.
        """
        return rank

    @always_inline
    fn get_shape(self) -> StaticIntTuple[rank]:
        """Returns the shapes of the buffer.

        Returns:
            A static tuple of size 'rank' representing shapes of the NDBuffer.
        """
        var res = StaticIntTuple[rank]()

        @unroll
        for i in range(rank):
            res[i] = self.dim(i)
        return res

    @always_inline
    fn get_nd_index(self, idx: Int) -> StaticIntTuple[rank]:
        """Computes the NDBuffer's ND-index based on the flat index.

        Args:
            idx: The flat index.

        Returns:
            The index positions.
        """
        return _compute_nd_index(self, idx)

    @always_inline
    fn __len__(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            The total number of elements in the NDBuffer.
        """
        return self.size()

    @always_inline
    fn num_elements(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            The total number of elements in the NDBuffer.
        """
        return self.size()

    @always_inline
    fn size(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            The total number of elements in the NDBuffer.
        """
        var product: Int = 1

        @unroll
        for i in range(rank):
            product *= self.dim(i)

        return product

    @no_inline
    fn __str__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string of the buffer.
        """
        var res = String("NDBuffer(")

        @parameter
        fn serialize[T: Stringable](val: T):
            res += val.__str__()

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.data, self.dynamic_shape
        )

        return res + ")"

    @no_inline
    fn __repr__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string representation of the buffer.
        """
        return self.__str__()

    @always_inline
    fn _offset(
        self, idx: VariadicList[Int]
    ) -> DTypePointer[type, address_space]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx: The index positions.

        Returns:
            The offset into the NDBuffer given the indices.
        """
        constrained[rank <= _MAX_RANK]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    @always_inline
    fn _offset(
        self, idx: StaticIntTuple[rank]
    ) -> DTypePointer[type, address_space]:
        return self._offset(idx.as_tuple())

    @always_inline
    fn _offset(
        self, idx: StaticTuple[rank, Int]
    ) -> DTypePointer[type, address_space]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx: The index positions.

        Returns:
            The offset into the NDBuffer given the indices.
        """
        constrained[rank <= _MAX_RANK]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    @always_inline
    fn __getitem__(self, *idx: Int) -> SIMD[type, 1]:
        """Gets an element from the buffer from the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The value of the element.
        """
        return self.simd_load[1](idx)

    @always_inline
    fn __getitem__(self, idx: StaticIntTuple[rank]) -> SIMD[type, 1]:
        """Gets an element from the buffer from the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The value of the element.
        """
        return self.simd_load[1](idx)

    @always_inline
    fn simd_load[
        width: Int,
    ](self, *idx: Int) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.simd_load[width](idx)

    @always_inline
    fn simd_load[
        width: Int,
    ](self, idx: VariadicList[Int]) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).simd_load[width]()

    @always_inline
    fn simd_load[
        width: Int,
    ](self, idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.simd_load[width](idx.as_tuple())

    @always_inline
    fn simd_load[
        width: Int,
    ](self, idx: StaticTuple[rank, Int]) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).simd_load[width]()

    @always_inline
    fn aligned_simd_load[
        width: Int, alignment: Int
    ](self, *idx: Int) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.aligned_simd_load[width, alignment](idx)

    @always_inline
    fn aligned_simd_load[
        width: Int, alignment: Int
    ](self, idx: VariadicList[Int]) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).aligned_simd_load[width, alignment]()

    @always_inline
    fn aligned_simd_load[
        width: Int, alignment: Int
    ](self, idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.aligned_simd_load[width, alignment](idx.as_tuple())

    @always_inline
    fn aligned_simd_load[
        width: Int, alignment: Int
    ](self, idx: StaticTuple[rank, Int]) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).aligned_simd_load[width, alignment]()

    @always_inline
    fn __setitem__(self, idx: StaticIntTuple[rank], val: SIMD[type, 1]):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.simd_store[1](idx, val)

    @always_inline
    fn simd_store[
        width: Int
    ](self, idx: StaticIntTuple[rank], val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The width of the simd vector.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.simd_store[width](idx.as_tuple(), val)

    @always_inline
    fn simd_store[
        width: Int
    ](self, idx: StaticTuple[rank, Int], val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The width of the simd vector.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        self._offset(idx).simd_store[width](val)

    @always_inline
    fn aligned_simd_store[
        width: Int, alignment: Int
    ](self, idx: StaticIntTuple[rank], val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.aligned_simd_store[width, alignment](idx.as_tuple(), val)

    @always_inline
    fn aligned_simd_store[
        width: Int, alignment: Int
    ](self, idx: StaticTuple[rank, Int], val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        self._offset(idx).aligned_simd_store[width, alignment](val)

    @always_inline
    fn simd_nt_store[
        width: Int
    ](self, idx: StaticIntTuple[rank], val: SIMD[type, width]):
        """Stores a simd value using non-temporal store.

        Constraints:
            The buffer must be contiguous.
            The address must be properly aligned, 64B for avx512, 32B for avx2,
              and 16B for avx.

        Parameters:
            width: The width of the simd vector.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.simd_nt_store[width](idx.as_tuple(), val)

    @always_inline
    fn simd_nt_store[
        width: Int
    ](self, idx: StaticTuple[rank, Int], val: SIMD[type, width]):
        """Stores a simd value using non-temporal store.

        Constraints:
            The buffer must be contiguous.
            The address must be properly aligned, 64B for avx512, 32B for avx2,
              and 16B for avx.

        Parameters:
            width: The width of the simd vector.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        self._offset(idx).simd_nt_store[width](val)

    @always_inline
    fn dim[index: Int](self) -> Int:
        """Gets the buffer dimension at the given index.

        Parameters:
            index: The number of dimension to get.

        Returns:
            The buffer size at the given dimension.
        """
        # First try to extract the static info on this dimension, could be either a
        # meta constant or an unknown.
        alias static_dim_value = shape.at[index]()

        @parameter
        if static_dim_value.has_value():
            return static_dim_value.get()
        return self.dynamic_shape[index]

    @always_inline
    fn dim(self, index: Int) -> Int:
        """Gets the buffer dimension at the given index.

        Args:
            index: The number of dimension to get.

        Returns:
            The buffer size at the given dimension.
        """
        return self.dynamic_shape[index]

    @always_inline
    fn stride(self, index: Int) -> Int:
        """Gets the buffer stride at the given index.

        Args:
            index: The number of dimension to get the stride for.

        Returns:
            The stride at the given dimension.
        """
        return self.dynamic_stride[index]

    @always_inline
    fn flatten(self) -> Buffer[Dim(), type, address_space=address_space]:
        """Constructs a flattened Buffer counterpart for this NDBuffer.

        Constraints:
            The buffer must be contiguous.

        Returns:
            Constructed Buffer object.
        """
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        return Buffer[Dim(), type, address_space=address_space](
            self.data, self.size()
        )

    @always_inline
    fn make_dims_unknown(
        self,
    ) -> NDBuffer[
        rank, DimList.create_unknown[rank](), type, address_space=address_space
    ]:
        """Rebinds the NDBuffer to one with unknown shape.

        Returns:
            The rebound NDBuffer with unknown shape.
        """
        return rebind[
            NDBuffer[
                rank,
                DimList.create_unknown[rank](),
                type,
                address_space=address_space,
            ]
        ](self)

    @always_inline
    fn bytecount(self) -> Int:
        """Returns the size of the NDBuffer in bytes.

        Returns:
            The size of the NDBuffer in bytes.
        """
        return self.size() * sizeof[type]()

    @always_inline
    fn zero(self):
        """Sets all bytes of the NDBuffer to 0.

        Constraints:
            The buffer must be contiguous.
        """
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        memset_zero(self.data, len(self))

    fn simd_fill[simd_width: Int](self, val: SIMD[type, 1]):
        """Assigns val to all elements in chunks of size simd_width.

        Parameters:
            simd_width: The simd_width of the fill.

        Args:
            val: The value to store.
        """
        if val == 0:
            self.zero()
            return
        self.flatten().simd_fill[simd_width](val)

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        with open(path.__str__(), "w") as f:
            f._write(self.data.bitcast[DType.int8](), self.bytecount())

    @always_inline
    fn fill(self, val: SIMD[type, 1]):
        """Assigns val to all elements in the Buffer.

        The fill is performed in chunks of size N, where N is the native SIMD
        width of type on the system.

        Args:
            val: The value to store.
        """
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        self.simd_fill[simdwidthof[type]()](val)

    @staticmethod
    @always_inline("nodebug")
    fn aligned_stack_allocation[alignment: Int]() -> Self:
        """Constructs an NDBuffer instance backed by stack allocated memory space.

        Parameters:
            alignment: Address alignment requirement for the allocation.

        Returns:
            Constructed NDBuffer with the allocated space.
        """
        let data_pointer = stack_allocation[
            shape.product[rank]().get(),
            type,
            alignment=alignment,
            address_space=address_space,
        ]()
        return Self(data_pointer)

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation() -> Self:
        """Constructs an NDBuffer instance backed by stack allocated memory space.

        Returns:
            Constructed NDBuffer with the allocated space.
        """
        return Self.aligned_stack_allocation[alignof[type]()]()

    @always_inline
    fn prefetch[params: PrefetchOptions](self, *idx: Int):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            idx: The N-D index of the prefetched location.
        """
        self._offset(idx).prefetch[params]()

    @always_inline
    fn prefetch[params: PrefetchOptions](self, indices: StaticIntTuple[rank]):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            indices: The N-D index of the prefetched location.
        """
        self._offset(indices).prefetch[params]()

    @always_inline("nodebug")
    fn __imul__(inout self, rhs: Float32):
        """In-place multiplies a scalar.

        Args:
            rhs: The RHS of the mul operation.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type == DType.float32]()

        alias TM = shape.at[0]().get()
        alias TN = shape.at[1]().get()

        alias simd_size = simdwidthof[DType.float32]()

        @unroll
        for i in range(TM):

            @unroll
            for j in range(0, TN, simd_size):
                let idx = i * TN + j
                let vec = self.data.simd_load[simd_size](i * TN + j)
                self.data.simd_store[simd_size](idx, vec * rhs.cast[type]())

    @always_inline("nodebug")
    fn __imul__(inout self, rhs: NDBuffer):
        """In-place multiplies a NDBuffer.

        Args:
            rhs: The RHS of the mul operation.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type == DType.float32]()

        alias TM = shape.at[0]().get()
        alias TN = shape.at[1]().get()

        alias simd_size = simdwidthof[type]()

        @unroll
        for i in range(TM):

            @unroll
            for j in range(0, TN, simd_size):
                let idx = i * TN + j
                let vec = self.data.simd_load[simd_size](idx)
                let rhs_vec = SIMD[type, simd_size](
                    rhs.data.load(i).cast[type]()
                )
                self.data.simd_store[simd_size](idx, vec * rhs_vec)

    @always_inline("nodebug")
    fn __itruediv__(inout self, rhs: NDBuffer):
        """In-place divides a NDBuffer.

        Args:
            rhs: The RHS of the div operation.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type == DType.float32]()

        alias TM = shape.at[0]().get()
        alias TN = shape.at[1]().get()

        alias simd_size = simdwidthof[type]()

        @unroll
        for i in range(TM):

            @unroll
            for j in range(0, TN, simd_size):
                let idx = i * TN + j
                let vec = self.data.simd_load[simd_size](idx)
                let rhs_vec = SIMD[type, simd_size](
                    rhs.data.load(i).cast[type]()
                )
                self.data.simd_store[simd_size](idx, vec / rhs_vec)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        """Multiplies a NDBuffer.

        Args:
            rhs: The RHS of the mul operation.

        Returns:
            The division result.
        """
        constrained[rank == 1]()
        constrained[shape.all_known[1]()]()
        constrained[type == DType.float32]()

        alias m = shape.at[0]().get()

        alias simd_size = simdwidthof[type]()

        let res = Self.stack_allocation()

        @unroll
        for i in range(m):
            res.data.store(i, self.data.load(i) * rhs.data.load(i))

        return res

    @always_inline("nodebug")
    fn __add__(self, rhs: NDBuffer) -> Self:
        """Adds a NDBuffer.

        Args:
            rhs: The RHS of the add operation.

        Returns:
            The addition result.
        """
        constrained[rank == 1]()
        constrained[shape.all_known[1]()]()
        constrained[type == DType.float32]()

        alias m = shape.at[0]().get()

        alias simd_size = simdwidthof[type]()

        let res = Self.stack_allocation()

        @unroll
        for i in range(m):
            res.data.store(i, self.data.load(i) + rhs.data.load(i).cast[type]())

        return res

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        """Subtracts a scalar.

        Args:
            rhs: The RHS of the sub operation.

        Returns:
            The subtraction result.
        """
        constrained[rank == 1]()
        constrained[shape.all_known[1]()]()
        constrained[type == DType.float32]()

        alias simd_size = simdwidthof[type]()

        alias m = shape.at[0]().get()

        let res = Self.stack_allocation()

        @unroll
        for i in range(m):
            res.data.simd_store[simd_size](
                i, self.data.load(i) - rhs.data.load(i)
            )

        return res

    @always_inline("nodebug")
    fn __sub__[
        rhs_shape: DimList
    ](self, rhs: NDBuffer[1, rhs_shape, type]) -> Self:
        """Subtracts a NDBuffer.

        Args:
            rhs: The RHS of the sub operation.

        Returns:
            The subtraction result.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type == DType.float32]()

        alias m = shape.at[0]().get()
        alias n = shape.at[1]().get()
        alias simd_size = simdwidthof[type]()

        let res = Self.stack_allocation()

        @unroll
        for i in range(m):

            @unroll
            for j in range(0, n, simd_size):
                let idx = i * n + j
                res.data.simd_store[simd_size](
                    idx, self.data.simd_load[simd_size](idx) - rhs.data.load(i)
                )

        return res


fn partial_simd_load[
    type: DType, width: Int
](
    storage: DTypePointer[type],
    lbound: Int,
    rbound: Int,
    pad_value: SIMD[type, 1],
) -> SIMD[type, width]:
    """Loads a vector with dynamic bound.

    Out of bound data will be filled with pad value. Data is valid if
    lbound <= idx < rbound for idx from 0 to (simd_width-1). For example:

        addr 0  1  2  3
        data x 42 43  x

        partial_simd_load[4](addr0,1,3) #gives [0 42 43 0]

    Parameters:
        type: The underlying dtype of computation.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        pad_value: Value to fill for out of bound indices.

    Returns:
        The SIMD vector loaded and zero-filled.
    """
    # Create a mask based on input bounds.
    let effective_lbound = max(0, lbound)
    let effective_rbound = min(width, rbound)
    let incr = iota[DType.int32, width]()
    let mask = (incr >= effective_lbound) & (incr < effective_rbound)

    return masked_load[type, width](storage, mask, pad_value)


fn partial_simd_store[
    type: DType, width: Int
](
    storage: DTypePointer[type],
    lbound: Int,
    rbound: Int,
    data: SIMD[type, width],
):
    """Stores a vector with dynamic bound.

    Out of bound data will ignored. Data is valid if lbound <= idx < rbound for
    idx from 0 to (simd_width-1).

    e.g.
        addr 0 1 2  3
        data 0 0 0  0

        partial_simd_load[4](addr0,1,3, [-1, 42,43, -1]) #gives [0 42 43 0]

    Parameters:
        type: The underlying dtype of computation.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        data: The vector value to store.
    """
    # Create a mask based on input bounds.
    let effective_lbound = max(0, lbound)
    let effective_rbound = min(width, rbound)
    let incr = iota[DType.int32, width]()
    let mask = (incr >= effective_lbound) & (incr < effective_rbound)

    return masked_store[type, width](data, storage, mask)


# ===----------------------------------------------------------------------===#
# DynamicRankBuffer
# ===----------------------------------------------------------------------===#


# This type is "async safe" (see async_parallelize).
# This struct must match DynamicRankBuffer in Kernels/lib/MojoKernels/Kernels.cpp
@register_passable("trivial")
struct DynamicRankBuffer:
    """DynamicRankBuffer represents a buffer with unknown rank, shapes and dtype.

    It is not as efficient as the statically ranked buffer, but is useful when
    interacting with external functions. In particular the shape is represented
    as a fixed (ie _MAX_RANK) array of dimensions to simplify the ABI."""

    var data: DTypePointer[DType.invalid.value]
    """The pointer to the buffer."""
    var rank: Int
    """The buffer rank. Has a max value of `_MAX_RANK`."""
    var shape: StaticIntTuple[_MAX_RANK]
    """The dynamic shape of the buffer."""
    var type: DType
    """The dynamic dtype of the buffer."""

    @always_inline
    fn __init__(
        data: DTypePointer[DType.invalid.value],
        rank: Int,
        shape: StaticIntTuple[_MAX_RANK],
        type: DType,
    ) -> DynamicRankBuffer:
        """Construct DynamicRankBuffer.

        Args:
            data: Pointer to the underlying data.
            rank: Rank of the buffer.
            shape: Shapes of the buffer.
            type: `dtype` of the buffer.

        Returns:
            Constructed DynamicRankBuffer.
        """
        return DynamicRankBuffer {
            data: data,
            rank: rank,
            shape: shape,
            type: type.value,
        }

    @always_inline
    fn to_buffer[type: DType](self) -> Buffer[Dim(), type]:
        """Casts DynamicRankBuffer to Buffer.

        Parameters:
            type: `dtype` of the buffer.

        Returns:
            Constructed Buffer.
        """
        return Buffer[Dim(), type](
            self.data.bitcast[type](), tuple_product(self.shape, self.rank)
        )

    @always_inline
    fn to_ndbuffer[
        rank: Int, type: DType
    ](self) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
        """Casts the buffer to NDBuffer.

        Constraints:
            Rank of DynamicRankBuffer must equal rank of NDBuffer.

        Parameters:
            rank: Rank of the buffer.
            type: `dtype` of the buffer.

        Returns:
            Constructed NDBuffer.
        """
        debug_assert(
            self.rank == rank,
            "rank of DynamicRankBuffer must equal rank of NDBuffer",
        )
        return NDBuffer[rank, DimList.create_unknown[rank](), type](
            self.data.bitcast[type](), self._shape_to_static_tuple[rank]()
        )

    @always_inline
    fn to_ndbuffer[
        rank: Int, type: DType
    ](self, stride: StaticIntTuple[rank]) -> NDBuffer[
        rank, DimList.create_unknown[rank](), type
    ]:
        """Casts the buffer to NDBuffer.

        Constraints:
            Rank of DynamicRankBuffer must equal rank of NDBuffer.

        Parameters:
            rank: Rank of the buffer.
            type: `dtype` of the buffer.

        Args:
            stride: Strides of the buffer.

        Returns:
            Constructed NDBuffer.
        """
        debug_assert(
            self.rank == rank,
            "rank of DynamicRankBuffer must equal rank of NDBuffer",
        )
        return NDBuffer[rank, DimList.create_unknown[rank](), type](
            self.data.bitcast[type](),
            self._shape_to_static_tuple[rank](),
            stride,
        )

    @always_inline
    fn rank_dispatch[func: fn[rank: Int] () capturing -> None](self):
        """Dispatches the function call based on buffer rank.

        Constraints:
            Rank must be positive and less or equal to 8.

        Parameters:
            func: Function to dispatch. The function should be parametrized on
              an index parameter, which will be used for rank when the function
              will be called.
        """
        debug_assert(
            self.rank > 0 and self.rank <= _MAX_RANK,
            "rank must be positive and less or equal to 8",
        )

        if self.rank == 1:
            func[1]()
            return

        if self.rank == 2:
            func[2]()
            return

        if self.rank == 3:
            func[3]()
            return

        if self.rank == 4:
            func[4]()
            return

        if self.rank == 5:
            func[5]()
            return

        if self.rank == 6:
            func[6]()
            return

        if self.rank == 7:
            func[7]()
            return

        if self.rank == 8:
            func[8]()
            return

    @always_inline
    fn rank_dispatch[
        func: fn[rank: Int] () capturing -> None
    ](self, out_chain: OutputChainPtr):
        """Dispatches the function call based on buffer rank.

        Constraints:
            Rank must be positive and less or equal to 8.

        Parameters:
            func: Function to dispatch. The function should be parametrized on
              an index parameter, which will be used for rank when the function
              will be called.

        Args:
            out_chain: The output chain.
        """
        if self.rank <= 0 or self.rank > _MAX_RANK:
            out_chain.mark_error(
                "invalid rank, the rank bust be positive and less than or"
                " equal to 8"
            )
            return

        if self.rank == 1:
            func[1]()
            return

        if self.rank == 2:
            func[2]()
            return

        if self.rank == 3:
            func[3]()
            return

        if self.rank == 4:
            func[4]()
            return

        if self.rank == 5:
            func[5]()
            return

        if self.rank == 6:
            func[6]()
            return

        if self.rank == 7:
            func[7]()
            return

        if self.rank == 8:
            func[8]()
            return

    @always_inline
    fn num_elements(self) -> Int:
        """Gets number of elements in the buffer.

        Returns:
            The number of elements in the buffer.
        """
        return tuple_product(self.shape, self.rank)

    @always_inline
    fn get_shape[rank: Int](self) -> StaticIntTuple[rank]:
        """Gets a static tuple representing the buffer shape.

        Parameters:
            rank: Rank of the buffer.

        Returns:
            A static tuple of size 'Rank' filled with buffer shapes.
        """
        return self._shape_to_static_tuple[rank]()

    @always_inline
    fn dim(self, idx: Int) -> Int:
        """Gets given dimension.

        Args:
            idx: The dimension index.

        Returns:
            The buffer size on the given dimension.
        """
        debug_assert(idx < self.rank, "dimension index is out of bounds")
        return self.shape[idx]

    @always_inline
    fn _shape_to_static_tuple[rank: Int](self) -> StaticIntTuple[rank]:
        var result = StaticIntTuple[rank]()

        @unroll
        for idx in range(rank):
            result[idx] = self.dim(idx)
        return result


fn _collapse_batch_dim(input: DynamicRankBuffer) -> DynamicRankBuffer:
    """Collapse the batch so that the tensor has rank of at most 3. The output
    shape will therefore be
    [input.shape[0] * ... * input.shape[input.rank-2],
     input.shape[rank-2],
     input.shape[rank-1],
     0, ..., 0
    ]

    Args:
      input: The input DynamicRankBuffer.

    Returns:
      The DynamicRankBuffer where the batch dim has been flattened.
    """
    var batch_size = 1
    for i in range(0, input.rank - 2):
        batch_size *= input.shape[i]
    var collapsed_shape = StaticIntTuple[_MAX_RANK]()
    collapsed_shape[0] = batch_size
    collapsed_shape[1] = input.shape[input.rank - 2]
    collapsed_shape[2] = input.shape[input.rank - 1]
    return DynamicRankBuffer(input.data, 3, collapsed_shape, input.type)


@always_inline
fn prod_dims[
    start_dim: Int,
    end_dim: Int,
    rank: Int,
    shape: DimList,
    type: DType,
](x: NDBuffer[rank, shape, type]) -> Int:
    """Computes the product of a slice of the given buffer's dimensions.

    Parameters:
        start_dim: The index at which to begin computing the product.
        end_dim: The index at which to stop computing the product.
        rank: The rank of the NDBuffer.
        shape: The shape of the NDBuffer.
        type: The element-type of the NDBuffer.

    Args:
        x: The NDBuffer whose dimensions will be multiplied.

    Returns:
        The product of the specified slice of the buffer's dimensions.
    """

    var product: Int = 1

    @unroll
    for idx in range(start_dim, end_dim):
        product *= x.dim(idx)

    return product
