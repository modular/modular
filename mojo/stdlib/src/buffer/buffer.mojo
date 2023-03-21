# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import (
    assert_param,
    debug_assert,
    assert_param_bool_msg,
    assert_param_bool,
)
from DType import DType
from Functional import unroll, vectorize
from Int import Int
from Index import StaticIntTuple
from List import Dim, DimList, VariadicList
from Math import fma, min, max
from Memory import stack_allocation, memset_zero
from Pointer import DTypePointer, product as pointer_product
from Intrinsics import PrefetchOptions
from SIMD import SIMD
from Tuple import StaticTuple
from TargetInfo import dtype_sizeof, dtype_simd_width, dtype_alignof
from TypeUtilities import rebind
from Range import range


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _raw_stack_allocation[
    count: Int,
    type: DType,
    alignment: Int,
]() -> DTypePointer[type]:
    """Allocates data buffer space on the stack given a data type and number of elements.

    Args:
        count: number of elements to allocate memory for.
        type: the data type of each element.
        alignment: address alignment of the allocated data.

    Returns:
        A data pointer of the given dtype pointing to the allocated space.
    """
    let ptr = stack_allocation[
        count, __mlir_type[`!pop.scalar<`, type.value, `>`], alignment
    ]()
    return ptr.address


# ===----------------------------------------------------------------------===#
# Buffer
# ===----------------------------------------------------------------------===#

# TODO: This should not be implicitly copyable when we have ownership set up!
@register_passable("trivial")
struct Buffer[size: Dim, type: DType]:
    """Defines a Buffer which can be parametrized on a static size and Dtype.
    The Buffer does not own its underlying pointer.
    """

    var data: DTypePointer[type]
    var dynamic_size: Int
    var dtype: DType

    fn __init__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`]
    ) -> Buffer[size, type]:
        """Constructor for a Buffer with statically known size and type.

        Constraints:
            The size is known.

        Args:
            ptr (!pop.pointer<scalar<type>>): Pointer to the data.

        Returns:
            Buffer[size, type]: The buffer object.
        """
        # Construct a Buffer type with statically known size
        assert_param_bool_msg[size.has_value(), "must have known size"]()
        return Buffer[size, type] {
            data: DTypePointer[type](ptr), dynamic_size: size.get(), dtype: type
        }

    fn __init__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`],
        in_size: Int,
    ) -> Buffer[size, type]:
        """Constructor for a Buffer with statically known type.

        Constraints:
            The size is unknown.

        Args:
            ptr (!pop.pointer<scalar<type>>): Pointer to the data.
            size (Int): Dynamic size of the buffer.

        Returns:
            Buffer[size, type]: The buffer object.
        """

        @parameter
        if size:
            debug_assert(
                in_size == size.get(),
                "if static size is known, static size must equal dynamic size",
            )
        return Buffer[size, type] {
            data: DTypePointer[type](ptr), dynamic_size: in_size, dtype: type
        }

    fn __init__(
        ptr: DTypePointer[type],
        in_size: Int,
    ) -> Buffer[size, type]:
        """Constructor for a Buffer with statically known type.

        Constraints:
            The size is unknown.

        Args:
            ptr (DTypePointer[type]): Pointer to the data.
            size (Int): Dynamic size of the buffer.

        Returns:
            Buffer[size, type]: The buffer object.
        """

        @parameter
        if size:
            debug_assert(
                in_size == size.get(),
                "if static size is known, static size must equal dynamic size",
            )
        return Buffer[size, type] {
            data: ptr, dynamic_size: in_size, dtype: type
        }

    fn __len__(self) -> Int:
        """Gets the size if it is a known constant, otherwise it gets the
        dynamic_size.

        This method is used by `Buffer.__len__` to get the size of the buffer.
        If the Buffer size is a known constant, then the size is returned.
        Otherwise, the dynamic_size is returned.

        Returns:
            Int: The size if static otherwise dynamic_size.
        """

        @parameter
        if not size:
            return self.dynamic_size

        return size.get()

    fn __getitem__(self, idx: Int) -> SIMD[1, type]:
        """Loads a single element (SIMD of size 1) from the buffer at the
        specified index.

        Args:
            idx (Idx): The index into the Buffer.

        Returns:
            SIMD[1, type]: The value at the `idx` position.
        """
        return self.simd_load[1](idx)

    fn simd_load[width: Int](self, idx: Int) -> SIMD[width, type]:
        """Loads a simd value from the buffer at the specified index.

        Args:
            width (__mlir_type.index): The simd_width of the load.
            idx (Idx): The index into the Buffer.

        Returns:
            SIMD[width, type]: The simd value starting at the `idx` position
            and ending at `idx+width`.
        """
        return self.data.simd_load[width](idx)

    fn __setitem__(
        self,
        idx: Int,
        val: __mlir_type[`!pop.scalar<`, type.value, `>`],
    ):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx (Idx): The index into the Buffer.
            val (!pop.scalar<type>): The value to store.
        """
        var simd_val: SIMD[1, type]
        simd_val.value = val
        self.simd_store[1](idx, simd_val)

    fn __setitem__(self, idx: Int, val: SIMD[1, type]):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx (Idx): The index into the Buffer.
            val (SIMD[1, type]): The value to store.
        """
        self.simd_store[1](idx, val)

    fn simd_store[
        width: Int,
    ](self, idx: Int, val: SIMD[width, type]):
        """Stores a simd value into the buffer at the specified index.

        Args:
            width (__mlir_type.index): The width of the simd vector.
            idx (Idx): The index into the Buffer.
            val (SIMD[width, type]): The value to store.
        """
        self.data.simd_store[width](idx, val)

    fn simd_nt_store[width: Int](self, idx: Int, val: SIMD[width, type]):
        """Stores a simd value into the buffer at the specified index
           using non-temporal store. The address must be properly
           aligned, 64B for avx512, 32B for avx2, and 16B for avx.

        Args:
            width (__mlir_type.index): The width of the simd vector.
            idx (Idx): The index into the Buffer.
            val (SIMD[width, type]): The value to store.
        """
        self.data.simd_nt_store[width](idx, val)

    @always_inline
    fn prefetch[params: PrefetchOptions](self, idx: Int):
        """Prefetch the data at the given index.

        Args:
            params (PrefetchOptions): The prefetch configuration.
            idx (Int): The index of the prefetched location.
        """
        self.data.offset(idx).prefetch[params]()

    @always_inline
    fn bytecount(self) -> Int:
        """Return the size of the Buffer in bytes."""
        return self.__len__() * dtype_sizeof[type]()

    @always_inline
    fn zero(self):
        """Set all bytes of the Buffer to 0"""
        memset_zero(self.data, self.bytecount())

    fn simd_fill[
        simd_width: Int
    ](self, val: SIMD[1, type]) -> Buffer[size, type]:
        """Assigns val to all elements in the Buffer in chunks of size
        simd_width.

            Args:
                val (SIMD[1, type]):  value to store
        """
        if val == 0:
            self.zero()
            return self

        @always_inline
        fn _fill[simd_width: Int](idx: Int):
            self.simd_store[simd_width](idx, val)

        vectorize[simd_width, _fill](self.__len__())
        return self

    @always_inline
    fn fill(self, val: SIMD[1, type]) -> Buffer[size, type]:
        """Assigns val to all elements in the Buffer in chunks of size
        N, where N is the native SIMD width of type on the system.

            Args:
                val (SIMD[1, type]):  value to store
        """
        return self.simd_fill[dtype_simd_width[type]()](val)

    @staticmethod
    @always_inline
    fn aligned_stack_allocation[alignment: Int]() -> Buffer[size, type]:
        """Constructs a buffer instance backed by stack allocated memory space.

        Args:
            alignment: address alignment requirement for the allocation.
        Returns:
            Constructed buffer with the allocated space.
        """
        assert_param_bool_msg[size.has_value(), "must have known size"]()
        var data_pointer = _raw_stack_allocation[
            size.get().__as_mlir_index(), type, alignment.__as_mlir_index()
        ]()
        return Buffer[size, type](data_pointer.address)

    @staticmethod
    @always_inline
    fn stack_allocation[]() -> Buffer[size, type]:
        """Constructs a buffer instance backed by stack allocated memory space.

        Returns:
            Constructed buffer with the allocated space.
        """
        return Buffer[size, type].aligned_stack_allocation[
            dtype_alignof[type]()
        ]()


# ===----------------------------------------------------------------------===#
# NDBuffer Utilities
# ===----------------------------------------------------------------------===#


fn _compute_ndbuffer_offset[
    rank: Int,
    shape: DimList[rank],
    type: DType,
](buf: NDBuffer[rank, shape, type], index: VariadicList[Int]) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        rank (Int): The rank of the NDBuffer.
        shape (DimList[rank]): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        index (VariadicList[index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """

    @parameter
    if rank == 0:
        return 0

    var result: Int = 0

    @always_inline
    fn body[idx: Int]():
        result = fma(buf.stride(idx), index[idx], result)

    unroll[rank, body]()
    return result


fn _compute_ndbuffer_offset[
    rank: Int,
    shape: DimList[rank],
    type: DType,
](
    buf: NDBuffer[rank, shape, type],
    idx: StaticIntTuple[rank.__as_mlir_index()],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        rank (Int): The rank of the NDBuffer.
        shape (DimList[rank]): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (StaticIntTuple[rank]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """
    return _compute_ndbuffer_offset(buf, idx.as_tuple())


fn _compute_ndbuffer_offset[
    rank: Int,
    shape: DimList[rank],
    type: DType,
](
    buf: NDBuffer[rank, shape, type],
    index: StaticTuple[rank.__as_mlir_index(), __mlir_type.index],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        rank (Int): The rank of the NDBuffer.
        shape (DimList[rank]): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        index (StaticTuple[rank, index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """

    @parameter
    if rank == 0:
        return 0

    var result: Int = 0

    @always_inline
    fn body[idx: Int]():
        result = fma(buf.stride(idx), index[idx], result)

    unroll[rank, body]()
    return result


fn _compute_ndbuffer_stride[
    rank: Int
](shape: StaticIntTuple[rank.__as_mlir_index()]) -> StaticIntTuple[
    rank.__as_mlir_index()
]:
    """Computes the NDBuffer's default dynamic strides using the input shape.
    The default strides correspond to contiguous memory layout.

    Args:
        rank (index): The rank of the NDBuffer.
        shape (StaticTuple[rank, index]): The shape of the NDBuffer.

    Returns:
        StaticIntTuple[rank]: The default strides of the NDBuffer.
    """
    assert_param_bool[rank > 0]()

    @parameter
    if rank == 1:
        return StaticIntTuple[rank.__as_mlir_index()](1)

    var stride: StaticIntTuple[rank.__as_mlir_index()] = shape
    stride.__setitem__[(rank - 1).__as_mlir_index()](1)

    @always_inline
    fn body[idx: Int]():
        alias i = rank - idx - 1
        stride.__setitem__[(i - 1).__as_mlir_index()](shape[i] * stride[i])

    unroll[rank - 1, body]()
    return stride


# ===----------------------------------------------------------------------===#
# NDBuffer
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct NDBuffer[
    rank: Int,
    shape: DimList[rank],
    type: DType,
]:

    var data: DTypePointer[type]
    # This is added just to make it aligned with the zap.ndbuffer
    var _rank: Int
    var dynamic_shape: StaticIntTuple[rank.__as_mlir_index()]
    var dynamic_dtype: DType
    var dynamic_stride: StaticIntTuple[rank.__as_mlir_index()]
    var is_contiguous: Bool

    fn __init__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`],
    ) -> NDBuffer[rank, shape, type]:
        assert_param_bool_msg[
            shape.all_known(),
            "dimensions must all be known",
        ]()

        return Self {
            data: ptr,
            _rank: rank,
            dynamic_shape: shape,
            dynamic_dtype: type.value,
            dynamic_stride: _compute_ndbuffer_stride[rank](shape),
            is_contiguous: True,
        }

    fn __init__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`],
        dynamic_shape: StaticIntTuple[rank.__as_mlir_index()],
        dynamic_dtype: DType,
    ) -> NDBuffer[rank, shape, type]:
        return Self {
            data: ptr,
            _rank: rank,
            dynamic_shape: dynamic_shape,
            dynamic_dtype: dynamic_dtype.value,
            dynamic_stride: _compute_ndbuffer_stride[rank](dynamic_shape),
            is_contiguous: True,
        }

    fn __init__(
        ptr: DTypePointer[type],
        dynamic_shape: StaticIntTuple[rank.__as_mlir_index()],
        dynamic_dtype: DType,
    ) -> NDBuffer[rank, shape, type]:
        return NDBuffer[rank, shape, type] {
            data: ptr,
            _rank: rank,
            dynamic_shape: dynamic_shape,
            dynamic_dtype: dynamic_dtype.value,
            dynamic_stride: _compute_ndbuffer_stride[rank](dynamic_shape),
            is_contiguous: True,
        }

    fn __init__(
        ptr: DTypePointer[type],
        dynamic_shape: StaticIntTuple[rank.__as_mlir_index()],
        dynamic_dtype: DType,
        dynamic_stride: StaticIntTuple[rank.__as_mlir_index()],
    ) -> NDBuffer[rank, shape, type]:
        return NDBuffer[rank, shape, type] {
            data: ptr,
            _rank: rank,
            dynamic_shape: dynamic_shape,
            dynamic_dtype: dynamic_dtype.value,
            dynamic_stride: dynamic_stride,
            is_contiguous: _compute_ndbuffer_stride[rank](dynamic_shape)
            == dynamic_stride,
        }

    @always_inline
    fn get_rank(self) -> Int:
        return rank

    @always_inline
    fn size(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            Int: The total number of elements in the NDBuffer.
        """
        var product: Int = 1

        @always_inline
        fn _compute_product[idx: Int]():
            product *= self.dim[idx.__as_mlir_index()]()

        unroll[rank, _compute_product]()
        return product

    @always_inline
    fn _offset(self, idx: VariadicList[Int]) -> DTypePointer[type]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx (VariadicList[index]): The index positions.

        Returns:
            Int: The offset into the NDBuffer given the indices.
        """
        assert_param_bool[rank <= 5]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    @always_inline
    fn _offset(
        self, idx: StaticIntTuple[rank.__as_mlir_index()]
    ) -> DTypePointer[type]:
        return self._offset(idx.as_tuple())

    @always_inline
    fn _offset(
        self, idx: StaticTuple[rank.__as_mlir_index(), __mlir_type.index]
    ) -> DTypePointer[type]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx (StaticTuple[rank, __mlir_type.index]): The index positions.

        Returns:
            Int: The offset into the NDBuffer given the indices.
        """
        assert_param_bool[rank <= 5]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    @always_inline
    fn __getitem__(self, *idx: Int) -> SIMD[1, type]:
        return self.simd_load[1](VariadicList[Int](idx))

    @always_inline
    fn __getitem__(
        self, idx: StaticIntTuple[rank.__as_mlir_index()]
    ) -> SIMD[1, type]:
        return self.simd_load[1](idx)

    @always_inline
    fn simd_load[
        width: Int,
    ](self, idx: VariadicList[Int]) -> SIMD[width, type]:
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).simd_load[width]()

    fn simd_load[
        width: Int,
    ](self, idx: StaticIntTuple[rank.__as_mlir_index()]) -> SIMD[width, type]:
        return self.simd_load[width](idx.as_tuple())

    @always_inline
    fn simd_load[
        width: Int,
    ](
        self, idx: StaticTuple[rank.__as_mlir_index(), __mlir_type.index]
    ) -> SIMD[width, type]:
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).simd_load[width]()

    @always_inline
    fn __setitem__(
        self, idx: StaticIntTuple[rank.__as_mlir_index()], val: SIMD[1, type]
    ):
        # Stores a single value into the ndbuffer at the specified index
        self.simd_store[1](idx, val)

    @always_inline
    fn __setitem__(
        self,
        idx: StaticTuple[rank.__as_mlir_index(), __mlir_type.index],
        val: SIMD[1, type],
    ):
        # Stores a single value into the ndbuffer at the specified index
        self.simd_store[1](idx, val)

    @always_inline
    fn simd_store[
        width: Int
    ](
        self,
        idx: StaticIntTuple[rank.__as_mlir_index()],
        val: SIMD[width, type],
    ):
        # Stores a simd value into the ndbuffer at the specified index
        self.simd_store[width](idx.as_tuple(), val)

    @always_inline
    fn simd_store[
        width: Int
    ](
        self,
        idx: StaticTuple[rank.__as_mlir_index(), __mlir_type.index],
        val: SIMD[width, type],
    ):
        debug_assert(
            self.is_contiguous or width == 1,
            "Function requires contiguous buffer.",
        )
        # Stores a simd value into the ndbuffer at the specified index
        self._offset(idx).simd_store[width](val)

    @always_inline
    fn simd_nt_store[
        width: Int
    ](
        self,
        idx: StaticIntTuple[rank.__as_mlir_index()],
        val: SIMD[width, type],
    ):
        # Stores a simd value into the ndbuffer at the specified index.
        # The address must properly aligned, see Buffer::simd_nt_store.
        self.simd_nt_store[width](idx.as_tuple(), val)

    @always_inline
    fn simd_nt_store[
        width: Int
    ](
        self,
        idx: StaticTuple[rank.__as_mlir_index(), __mlir_type.index],
        val: SIMD[width, type],
    ):
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        # Stores a simd value into the ndbuffer at the specified index
        # The address must properly aligned, see Buffer::simd_nt_store.
        self._offset(idx).simd_nt_store[width](val)

    @always_inline
    fn dim[index: Int](self) -> Int:
        # First try to extract the static info on this dimension, could be either a
        # meta constant or an unknown.
        alias static_dim_value = shape.at[index]()

        @parameter
        if static_dim_value.has_value():
            return static_dim_value.get()
        return self.dynamic_shape[index]

    @always_inline
    fn dim(self, index: Int) -> Int:
        return self.dynamic_shape[index]

    @always_inline
    fn stride(self, index: Int) -> Int:
        return self.dynamic_stride[index]

    @always_inline
    fn flatten(self) -> Buffer[Dim(), type]:
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        return Buffer[Dim(), type](self.data.address, self.size())

    @always_inline
    fn bytecount(self) -> Int:
        """Return the size of the NDBuffer in bytes."""
        return self.size() * dtype_sizeof[type]()

    @always_inline
    fn zero(self):
        """Set all bytes of the NDBuffer to 0"""
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        memset_zero(self.data, self.bytecount())

    fn simd_fill[
        simd_width: Int
    ](self, val: SIMD[1, type]) -> NDBuffer[rank, shape, type]:
        """Assigns val to all elements in the NDBuffer in chunks of size
        simd_width.

            Args:
                val (SIMD[1, type]):  value to store
        """
        if val == 0:
            self.zero()
            return self
        let _ = self.flatten().simd_fill[simd_width](val)
        return self

    @always_inline
    fn fill(self, val: SIMD[1, type]) -> NDBuffer[rank, shape, type]:
        """Assigns val to all elements in the NDBuffer in chunks of size
        N, where N is the native SIMD width of type on the system.

            Args:
                val (SIMD[1, type]):  value to store
        """
        debug_assert(self.is_contiguous, "Function requires contiguous buffer.")
        return self.simd_fill[dtype_simd_width[type]()](val)

    @staticmethod
    @always_inline
    fn aligned_stack_allocation[
        alignment: Int
    ]() -> NDBuffer[rank, shape, type]:
        """Constructs an ndbuffer instance backed by stack allocated memory space.

        Args:
            alignment: address alignment requirement for the allocation.
        Returns:
            Constructed ndbuffer with the allocated space.
        """
        var data_pointer = _raw_stack_allocation[
            shape.product().get(), type, alignment
        ]()
        return NDBuffer[rank, shape, type](data_pointer.address)

    @staticmethod
    @always_inline
    fn stack_allocation() -> NDBuffer[rank, shape, type]:
        """Constructs an ndbuffer instance backed by stack allocated memory space.

        Returns:
            Constructed ndbuffer with the allocated space.
        """
        return NDBuffer[rank, shape, type].aligned_stack_allocation[
            dtype_alignof[type]()
        ]()

    @always_inline
    fn prefetch[params: PrefetchOptions](self, *idx: Int):
        """Prefetch the data at the given index.

        Args:
            params (PrefetchOptions): The prefetch configuration.
            idx (*Int): The N-D index of the prefetched location.
        """
        self._offset(idx).prefetch[params]()


fn _neg[val: __mlir_type.i1]() -> __mlir_type.i1:
    """Negates an i1 value"""
    if val:
        return __mlir_attr.`0:i1`
    return __mlir_attr.`1:i1`


fn partial_simd_load[
    width: Int, type: DType
](
    storage: DTypePointer[type],
    lbound: Int,
    rbound: Int,
    pad_value: SIMD[1, type],
) -> SIMD[width, type]:
    """Loads a vector with dynamic bound, out of bound data will be filled
    with pad value. Data is valid if lbound <= idx < rbound for idx from 0
    to (simd_width-1).

        e.g.
            addr 0 1   2  3
            data x 42 43  x

            partial_simd_load[4](addr0,1,3) #gives [0 42 43 0]

        Args:
            width (mlir_index): The system simd vector size.
            type (dtype): The underlying dtype of computation.
            storage (DtypePointer): Pointer to the address to perform load.
            lbound: lower bound of valid index within simd (inclusive).
            rbound: upper bound of valid index within simd (non-inclusive).
            pad_value: value to fill for out of bound indices.

        Returns:
            The SIMD vector loaded and zero-filled.
    """
    # Create a buffer view of the allocated space.
    let vector = Buffer[width.__as_mlir_index(), type].stack_allocation()

    # Initialize vector with pad values.
    vector.simd_store[width](0, SIMD[width, type].splat(pad_value))

    # Compute intersection of given bound and the vector range.
    let effective_lbound = max(0, lbound)
    let effective_rbound = min(width, rbound)

    # Fill values in valid range.
    for idx in range(effective_lbound, effective_rbound):
        vector[idx] = storage.load(idx)

    # Return the resulting vector.
    return vector.simd_load[width](0)


fn partial_simd_store[
    width: Int, type: DType
](
    storage: DTypePointer[type],
    lbound: Int,
    rbound: Int,
    data: SIMD[width, type],
):
    """Stores a vector with dynamic bound, out of bound data will ignored.
    Data is valid if lbound <= idx < rbound for idx from 0 to (simd_width-1).
        e.g.
            addr 0 1 2  3
            data 0 0 0  0

            partial_simd_load[4](addr0,1,3, [-1, 42,43, -1]) #gives [0 42 43 0]

        Args:
            width (mlir_index): The system simd vector size.
            type (dtype): The underlying dtype of computation.
            storage (DtypePointer): Pointer to the address to perform load.
            lbound: lower bound of valid index within simd (inclusive).
            rbound: upper bound of valid index within simd (non-inclusive).
            data: The vector value to store.
    """
    # Create a buffer view of the storage space.
    let vector = Buffer[width.__as_mlir_index(), type].stack_allocation()

    # Put the given vector data in the allocated buffer.
    vector.simd_store[width](0, data)

    # Compute intersection of valid bound and the simd vector range.
    let effective_lbound = max(0, lbound)
    let effective_rbound = min(width, rbound)

    # Store the valid on the valid range.
    for idx in range(effective_lbound, effective_rbound):
        let storageVal = vector[idx]
        storage.store(idx, storageVal)


# ===----------------------------------------------------------------------===#
# DynamicRankBuffer
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct DynamicRankBuffer:
    """This buffer struct does not assume the rank to be static. It is not as
    efficient as the statically ranked buffer, but is useful when interacting
    with external functions"""

    var data: DTypePointer[DType.invalid.value]
    var rank: Int
    var shape: DTypePointer[DType.index]
    var type: DType

    @always_inline
    fn __init__(
        data: DTypePointer[DType.invalid.value],
        rank: Int,
        shape: DTypePointer[DType.index],
        type: DType,
    ) -> DynamicRankBuffer:
        return DynamicRankBuffer {
            data: data,
            rank: rank,
            shape: shape,
            type: type.value,
        }

    @always_inline
    fn to_buffer[type: DType](self) -> Buffer[Dim(), type]:
        return Buffer[Dim(), type](
            self.data.bitcast[type](), pointer_product(self.shape, self.rank)
        )

    @always_inline
    fn to_ndbuffer[
        rank: Int, type: DType
    ](self) -> NDBuffer[rank, DimList[rank].create_unknown(), type]:
        debug_assert(
            self.rank == rank,
            "rank of DynamicRankBuffer must equal rank of NDBuffer",
        )
        return NDBuffer[rank, DimList[rank].create_unknown(), type](
            self.data.bitcast[type](),
            self._shape_to_static_tuple[rank](),
            self.type.value,
        )

    @always_inline
    fn to_ndbuffer[
        rank: Int, type: DType
    ](self, stride: StaticIntTuple[rank.__as_mlir_index()]) -> NDBuffer[
        rank, DimList[rank].create_unknown(), type
    ]:
        debug_assert(
            self.rank == rank,
            "rank of DynamicRankBuffer must equal rank of NDBuffer",
        )
        return NDBuffer[rank, DimList[rank].create_unknown(), type](
            self.data.bitcast[type](),
            self._shape_to_static_tuple[rank](),
            self.type.value,
            stride,
        )

    @always_inline
    fn rank_dispatch[
        func: __mlir_type.`!kgen.signature<<rank:index>() -> !lit.none>`
    ](self):
        debug_assert(
            self.rank > 0 and self.rank <= 5,
            "rank be be positive and less or equal to 5",
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

    @always_inline
    fn num_elements(self) -> Int:
        return pointer_product(self.shape, self.rank)

    @always_inline
    fn dim(self, idx: Int) -> Int:
        debug_assert(idx < self.rank, "dimension index is out of bounds")
        return self.shape.load(idx)[0].value

    @always_inline
    fn _shape_to_static_tuple[
        rank: Int
    ](self) -> StaticIntTuple[rank.__as_mlir_index()]:
        var result: StaticIntTuple[rank.__as_mlir_index()]

        @always_inline
        fn _fill[idx: Int]():
            result.__setitem__[idx.__as_mlir_index()](
                self.dim(idx).__as_mlir_index()
            )

        unroll[rank, _fill]()

        return result
