# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, debug_assert, assert_param_bool_msg
from DType import DType
from Functional import unroll, vectorize
from Int import Int
from Index import StaticIntTuple
from List import (
    product,
    contains,
    _get_kgen_list_item,
    is_all_known,
    create_kgen_list_unknown,
    VariadicList,
)
from Math import fma
from Memory import stack_allocation, memset_zero
from Pointer import DTypePointer, product as pointer_product
from Intrinsics import PrefetchOptions
from SIMD import SIMD
from Tuple import StaticTuple
from TargetInfo import dtype_sizeof, dtype_simd_width
from Range import range


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _raw_stack_allocation[
    count: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    alignment: __mlir_type.index,
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
        count, __mlir_type[`!pop.scalar<`, type, `>`], alignment
    ]()
    return ptr.address


# ===----------------------------------------------------------------------===#
# Buffer
# ===----------------------------------------------------------------------===#


struct Buffer[size: __mlir_type.index, type: __mlir_type.`!kgen.dtype`]:
    """Defines a Buffer which can be parametrized on a static size and Dtype.
    The Buffer does not own its underlying pointer.
    """

    var data: DTypePointer[type]
    var dynamic_size: Int
    var dtype: __mlir_type.`!kgen.dtype`

    # TODO: This should not be implicitly copyable when we have ownership all
    # set up!
    fn __clone__(self&) -> Self:
        return Self {
            data: self.data, dynamic_size: self.dynamic_size, dtype: self.dtype
        }

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`]
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
        assert_param[size != __mlir_attr.`#kgen.unknown : index`]()
        return Buffer[size, type] {
            data: DTypePointer[type](ptr), dynamic_size: size, dtype: type
        }

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
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
        if size != __mlir_attr.`#kgen.unknown : index`:
            debug_assert(
                in_size == size,
                "if static size is known, static size must equal dynamic size",
            )
        return Buffer[size, type] {
            data: DTypePointer[type](ptr), dynamic_size: in_size, dtype: type
        }

    fn __new__(
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
        if size != __mlir_attr.`#kgen.unknown : index`:
            debug_assert(
                in_size == size,
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
        if size == __mlir_attr.`#kgen.unknown : index`:
            return self.dynamic_size

        return size

    fn __getitem__(self, idx: Int) -> SIMD[1, type]:
        """Loads a single element (SIMD of size 1) from the buffer at the
        specified index.

        Args:
            idx (Idx): The index into the Buffer.

        Returns:
            SIMD[1, type]: The value at the `idx` position.
        """
        return self.simd_load[1](idx)

    fn simd_load[width: __mlir_type.index](self, idx: Int) -> SIMD[width, type]:
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
        val: __mlir_type[`!pop.scalar<`, type, `>`],
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
        width: __mlir_type.index
    ](self, idx: Int, val: SIMD[width, type]):
        """Stores a simd value into the buffer at the specified index.

        Args:
            width (__mlir_type.index): The width of the simd vector.
            idx (Idx): The index into the Buffer.
            val (SIMD[width, type]): The value to store.
        """
        self.data.simd_store[width](idx, val)

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
        simd_width: __mlir_type.index
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
        fn _fill[simd_width: __mlir_type.index](idx: Int):
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
        return self.simd_fill[dtype_simd_width[type]().__as_mlir_index()](val)

    @staticmethod
    @always_inline
    fn aligned_stack_allocation[
        alignment: __mlir_type.index
    ]() -> Buffer[size, type]:
        """Constructs a buffer instance backed by stack allocated memory space.

        Args:
            alignment: address alignment requirement for the allocation.
        Returns:
            Constructed buffer with the allocated space.
        """
        var data_pointer = _raw_stack_allocation[size, type, alignment]()
        return Buffer[size, type](data_pointer.address)

    @staticmethod
    @always_inline
    fn stack_allocation[]() -> Buffer[size, type]:
        """Constructs a buffer instance backed by stack allocated memory space.

        Returns:
            Constructed buffer with the allocated space.
        """
        return Buffer[size, type].aligned_stack_allocation[1]()


# ===----------------------------------------------------------------------===#
# NDBuffer Utilities
# ===----------------------------------------------------------------------===#


fn _compute_ndbuffer_offset[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type], idx: VariadicList[Int]) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (VariadicList[index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """

    @parameter
    if rank == 0:
        return 0

    @parameter
    if rank == 1:
        return idx[0]

    var result: Int = idx[0]
    for i in range(1, rank):
        result = fma(buf.dim(i), result, idx[i])
    return result


fn _compute_ndbuffer_offset[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type], idx: StaticIntTuple[rank]) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (StaticIntTuple[rank]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """
    return _compute_ndbuffer_offset(buf, idx.as_tuple())


fn _compute_ndbuffer_offset[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (StaticTuple[rank, index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """

    @parameter
    if rank == 0:
        return 0

    @parameter
    if rank == 1:
        return idx[0]

    var result: Int = idx[0]
    for i in range(1, rank):
        result = fma(buf.dim(i), result, idx[i])
    return result


# ===----------------------------------------------------------------------===#
# NDBuffer
# ===----------------------------------------------------------------------===#


struct NDBuffer[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
]:

    var data: DTypePointer[type]
    # This is added just to make it aligned with the zap.ndbuffer
    var _rank: Int
    var dynamic_shape: StaticIntTuple[rank]
    var dynamic_dtype: DType

    @always_inline("nodebug")
    fn __clone__(self&) -> Self:
        return Self {
            data: self.data,
            _rank: self._rank,
            dynamic_shape: self.dynamic_shape,
            dynamic_dtype: self.dynamic_dtype,
        }

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
    ) -> NDBuffer[rank, shape, type]:
        assert_param_bool_msg[
            is_all_known[rank, shape](),
            "dimensions must all be known",
        ]()

        return Self {
            data: ptr,
            _rank: rank,
            dynamic_shape: shape,
            dynamic_dtype: type,
        }

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
        dynamic_shape: StaticIntTuple[rank],
        dynamic_dtype: DType,
    ) -> NDBuffer[rank, shape, type]:
        return Self {
            data: ptr,
            _rank: rank,
            dynamic_shape: dynamic_shape,
            dynamic_dtype: dynamic_dtype,
        }

    fn __new__(
        ptr: DTypePointer[type],
        dynamic_shape: StaticIntTuple[rank],
        dynamic_dtype: DType,
    ) -> NDBuffer[rank, shape, type]:
        return NDBuffer[rank, shape, type] {
            data: ptr,
            _rank: rank,
            dynamic_shape: dynamic_shape,
            dynamic_dtype: dynamic_dtype,
        }

    fn get_rank(self) -> Int:
        return rank

    fn size(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            Int: The total number of elements in the NDBuffer.
        """
        var product: Int = 1

        @always_inline
        fn _compute_product[idx: __mlir_type.index]():
            product *= self.dim[idx]()

        unroll[rank, _compute_product]()
        return product

    fn _offset(self, idx: VariadicList[Int]) -> DTypePointer[type]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx (VariadicList[index]): The index positions.

        Returns:
            Int: The offset into the NDBuffer given the indices.
        """
        assert_param[rank <= 5]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    fn _offset(self, idx: StaticIntTuple[rank]) -> DTypePointer[type]:
        return self._offset(idx.as_tuple())

    fn _offset(
        self, idx: StaticTuple[rank, __mlir_type.index]
    ) -> DTypePointer[type]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx (StaticTuple[rank, __mlir_type.index]): The index positions.

        Returns:
            Int: The offset into the NDBuffer given the indices.
        """
        assert_param[rank <= 5]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    fn __getitem__(self, *idx: Int) -> SIMD[1, type]:
        return self.simd_load[1](VariadicList[Int](idx))

    fn __getitem__(self, idx: StaticIntTuple[rank]) -> SIMD[1, type]:
        return self.simd_load[1](idx)

    fn simd_load[
        width: __mlir_type.index
    ](self, idx: VariadicList[Int]) -> SIMD[width, type]:
        return self._offset(idx).simd_load[width]()

    fn simd_load[
        width: __mlir_type.index
    ](self, idx: StaticIntTuple[rank]) -> SIMD[width, type]:
        return self.simd_load[width](idx.as_tuple())

    fn simd_load[
        width: __mlir_type.index
    ](self, idx: StaticTuple[rank, __mlir_type.index]) -> SIMD[width, type]:
        return self._offset(idx).simd_load[width]()

    fn __setitem__(self, idx: StaticIntTuple[rank], val: SIMD[1, type]):
        # Stores a single value into the ndbuffer at the specified index
        self.simd_store[1](idx, val)

    fn __setitem__(
        self, idx: StaticTuple[rank, __mlir_type.index], val: SIMD[1, type]
    ):
        # Stores a single value into the ndbuffer at the specified index
        self.simd_store[1](idx, val)

    fn simd_store[
        width: __mlir_type.index
    ](self, idx: StaticIntTuple[rank], val: SIMD[width, type]):
        # Stores a simd value into thendbuffer at the specified index
        self.simd_store[width](idx.as_tuple(), val)

    fn simd_store[
        width: __mlir_type.index
    ](self, idx: StaticTuple[rank, __mlir_type.index], val: SIMD[width, type]):
        # Stores a simd value into thendbuffer at the specified index
        self._offset(idx).simd_store[width](val)

    fn dim[index: __mlir_type.index](self) -> Int:
        # First try to extract the static info on this dimension, could be either a
        # meta constant or an unknown.
        alias static_dim_value = _get_kgen_list_item[
            index, rank, __mlir_type.index
        ](shape)

        @parameter
        if static_dim_value != __mlir_attr.`#kgen.unknown : index`:
            return static_dim_value
        return self.dynamic_shape[index]

    fn dim(self, index: Int) -> Int:
        return self.dynamic_shape[index]

    fn flatten(self) -> Buffer[__mlir_attr.`#kgen.unknown : index`, type]:
        return Buffer[__mlir_attr.`#kgen.unknown : index`, type](
            self.data.address, self.size()
        )

    @always_inline
    fn bytecount(self) -> Int:
        """Return the size of the NDBuffer in bytes."""
        return self.size() * dtype_sizeof[type]()

    @always_inline
    fn zero(self):
        """Set all bytes of the NDBuffer to 0"""
        memset_zero(self.data, self.bytecount())

    fn simd_fill[
        simd_width: __mlir_type.index
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
        return self.simd_fill[dtype_simd_width[type]().__as_mlir_index()](val)

    @staticmethod
    @always_inline
    fn aligned_stack_allocation[
        alignment: __mlir_type.index
    ]() -> NDBuffer[rank, shape, type]:
        """Constructs an ndbuffer instance backed by stack allocated memory space.

        Args:
            alignment: address alignment requirement for the allocation.
        Returns:
            Constructed ndbuffer with the allocated space.
        """
        var data_pointer = _raw_stack_allocation[
            product[rank](shape).__as_mlir_index(), type, alignment
        ]()
        return NDBuffer[rank, shape, type](data_pointer.address)

    @staticmethod
    @always_inline
    fn stack_allocation() -> NDBuffer[rank, shape, type]:
        """Constructs an ndbuffer instance backed by stack allocated memory space.

        Returns:
            Constructed ndbuffer with the allocated space.
        """
        return NDBuffer[rank, shape, type].aligned_stack_allocation[1]()

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
    width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
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
    let vector = Buffer[width, type].stack_allocation()

    # Initialize vector with pad values.
    vector.simd_store[width](0, SIMD[width, type].splat(pad_value))

    # Compute intersection of given bound and the vector range.
    let effective_lbound = Int.max(0, lbound)
    let effective_rbound = Int.min(width, rbound)

    # Fill values in valid range.
    for idx in range(effective_lbound, effective_rbound):
        let storageVal = storage.load(idx)
        vector.__setitem__(idx, storageVal)

    # Return the resulting vector.
    return vector.simd_load[width](0)


fn partial_simd_store[
    width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
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
    let vector = Buffer[width, type].stack_allocation()

    # Put the given vector data in the allocated buffer.
    vector.simd_store[width](0, data)

    # Compute intersection of valid bound and the simd vector range.
    let effective_lbound = Int.max(0, lbound)
    let effective_rbound = Int.min(width, rbound)

    # Store the valid on the valid range.
    for idx in range(effective_lbound, effective_rbound):
        let storageVal = vector.__getitem__(idx)
        storage.store(idx, storageVal)


# ===----------------------------------------------------------------------===#
# DynamicRankBuffer
# ===----------------------------------------------------------------------===#


struct DynamicRankBuffer:
    """This buffer struct does not assume the rank to be static. It is not as
    efficient as the statically ranked buffer, but is useful when interacting
    with external functions"""

    var data: DTypePointer[DType.invalid.value]
    var rank: Int
    var shape: DTypePointer[DType.index.value]
    var type: DType

    fn __clone__(self&) -> Self:
        return Self {
            data: self.data, rank: self.rank, shape: self.shape, type: self.type
        }

    fn __new__(
        data: DTypePointer[DType.invalid.value],
        rank: Int,
        shape: DTypePointer[DType.index.value],
        type: DType,
    ) -> DynamicRankBuffer:
        return DynamicRankBuffer {
            data: data,
            rank: rank,
            shape: shape,
            type: type,
        }

    fn to_buffer[
        type: __mlir_type.`!kgen.dtype`
    ](self) -> Buffer[__mlir_attr.`#kgen.unknown : index`, type]:
        return Buffer[__mlir_attr.`#kgen.unknown : index`, type](
            self.data.bitcast[type](), pointer_product(self.shape, self.rank)
        )

    fn to_ndbuffer[
        rank: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
    ](self) -> NDBuffer[rank, create_kgen_list_unknown[rank](), type]:
        return NDBuffer[rank, create_kgen_list_unknown[rank](), type](
            self.data.bitcast[type](),
            self._shape_to_static_tuple[rank](),
            self.type,
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

    fn num_elements(self) -> Int:
        return pointer_product(self.shape, self.rank)

    fn dim(self, idx: Int) -> Int:
        debug_assert(idx < self.rank, "dimension index is out of bounds")
        return self.shape.load(idx).__getitem__(0)

    fn _shape_to_static_tuple[
        rank: __mlir_type.index
    ](self) -> StaticIntTuple[rank]:
        var result: StaticIntTuple[rank]

        @always_inline
        fn _fill[idx: __mlir_type.index]():
            result.__setitem__[idx](self.dim(idx).__as_mlir_index())

        unroll[rank, _fill]()

        return result
