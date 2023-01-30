# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Bool import Bool
from DType import DType
from Int import Int
from List import product, contains, _get_kgen_list_item
from MemoryUtilities import stack_allocation
from Pointer import DTypePointer
from SIMD import SIMD
from Tuple import StaticTuple


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


@interface
fn _get_buffer_len[size: __mlir_type.index](dynamic_size: Int) -> Int:
    """Gets the size if it is a known constant, otherwise it gets the
    dynamic_size.

    This method is used by `Buffer.__len__` to get the size of the buffer.
    If the Buffer size is a known constant, then the size is returned.
    Otherwise, the dynamic_size is returned.


    Args:
        size (__mlir_type.index): The static size.
        dynamic_size (Int): The dynamic size.

    Returns:
        Int: The size if static otherwise dynamic_size.
    """
    ...


@implements(_get_buffer_len)
fn _get_buffer_len_dynamic[size: __mlir_type.index](dynamic_size: Int) -> Int:
    """Gets the dynamic size.

    Constraints:
        The size is unknown.

    Args:
        size (__mlir_type.index): The static size.
        dynamic_size (Int): The dynamic size.

    Returns:
        Int: The dynamic size.
    """
    assert_param[size == __mlir_attr.`#kgen.unknown : index`]()
    return dynamic_size


@implements(_get_buffer_len)
fn _get_buffer_len_static[size: __mlir_type.index](dynamic_size: Int) -> Int:
    """Gets the static size.

    Constraints:
        The size is known.

    Args:
        size (__mlir_type.index): The static size.
        dynamic_size (Int): The dynamic size.

    Returns:
        Int: The static size.
    """
    assert_param[size != __mlir_attr.`#kgen.unknown : index`]()
    return size


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
        assert_param[size == __mlir_attr.`#kgen.unknown : index`]()
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
        assert_param[size == __mlir_attr.`#kgen.unknown : index`]()
        return Buffer[size, type] {
            data: ptr, dynamic_size: in_size, dtype: type
        }

    fn __len__(self) -> Int:
        """Returns the dynamic size if the buffer is not statically known,
        otherwise returns the statically known size parameter.
        """
        return _get_buffer_len[size](self.dynamic_size)

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
    assert_param[rank > 0]()
    return _compute_ndbuffer_offset_impl[rank - 1, rank, shape, type](buf, idx)


@interface
fn _compute_ndbuffer_offset_impl[
    iter: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    """Helper function to recursively compute the NDBuffer's offset using the
    index positions provided.

    Args:
        iter (index): The induction variable.
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (StaticTuple[rank, index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """
    ...


@implements(_compute_ndbuffer_offset_impl)
fn _compute_ndbuffer_offset_impl_base[
    iter: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    """Base case for computing the NDBuffer's offset using the index positions
    provided. This case is triggered when the induction variable (iter) is 0.

    Args:
        iter (index): The induction variable.
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (StaticTuple[rank, index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """
    assert_param[iter == 0]()
    return idx.__getitem__[0]()


@implements(_compute_ndbuffer_offset_impl)
fn _compute_ndbuffer_offset_impl_iter[
    iter: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    """The recursive case for computing the NDBuffer's offset using the index
    positions provided. This case is triggered when the induction variable
    (iter) is greater than 0.

    Args:
        iter (index): The induction variable.
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.
        idx (StaticTuple[rank, index]): The index positions.

    Returns:
        Int: The offset into the NDBuffer given the indices.
    """
    assert_param[iter > 0]()
    return idx.__getitem__[iter]() + buf.dim[
        iter
    ]() * _compute_ndbuffer_offset_impl[iter - 1, rank, shape, type](buf, idx)


fn _compute_ndbuffer_size[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    """Computes the NDBuffer's number of elements.

    Args:
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.

    Returns:
        Int: The total number of elements in the NDBuffer.
    """
    return _compute_ndbuffer_size_impl[rank - 1, rank, shape, type](buf)


@interface
fn _compute_ndbuffer_size_impl[
    iter: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    """Helper function to recursively compute the number of elements in the
    NDBuffer.

    Args:
        iter (index): The induction variable.
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.

    Returns:
        Int: The total number of elements in the NDBuffer.
    """
    ...


@implements(_compute_ndbuffer_size_impl)
fn _compute_ndbuffer_size_impl_base[
    iter: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    """Base case for computing the total number of elements in the NDBuffer.
    This case is triggered when the induction variable (iter) is 0.

    Args:
        iter (index): The induction variable.
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.

    Returns:
        Int: The number of elements at dim 0.
    """
    assert_param[iter == 0]()
    return buf.dim[iter]()


@implements(_compute_ndbuffer_size_impl)
fn _compute_ndbuffer_size_impl_iter[
    iter: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    """The recursive case for computing the total number of elements in the
    NDBuffer. This case is triggered when the induction variable (iter) greater
    than 0.

    Args:
        iter (index): The induction variable.
        rank (index): The rank of the NDBuffer.
        shape (kgen.list<index[rank]>): The shape of the NDBuffer.
        type (dtype): The element-type of the NDBuffer.
        buf (NDBuffer[rank, shape, type]): The NDBuffer.

    Returns:
        Int: The number of elements in the dimensions starting at iter.
    """
    assert_param[iter > 0]()
    return buf.dim[iter]() * _compute_ndbuffer_size_impl[
        iter - 1, rank, shape, type
    ](buf)


@interface
fn _get_dim_helper[
    rank: __mlir_type.index, elem: __mlir_type.index, index: __mlir_type.index
](dynamic_shape: StaticTuple[rank, __mlir_type.index]) -> Int:
    """Helper function to support both static and dynamic size parameters.
    Returns `elem` directly if `elem` is a known static value.
    Returns dynamic_shape[index] if `elem` is unknown.
    """
    ...


# Implementation if elem is statically known.
@implements(_get_dim_helper)
fn _get_dim_static[
    rank: __mlir_type.index, elem: __mlir_type.index, index: __mlir_type.index
](dynamic_shape: StaticTuple[rank, __mlir_type.index]) -> Int:
    assert_param[elem != __mlir_attr.`#kgen.unknown : index`]()
    return elem


# Implementation if elem is not statically known.
@implements(_get_dim_helper)
fn _get_dim_dynamic[
    rank: __mlir_type.index, elem: __mlir_type.index, index: __mlir_type.index
](dynamic_shape: StaticTuple[rank, __mlir_type.index]) -> Int:
    assert_param[elem == __mlir_attr.`#kgen.unknown : index`]()
    return dynamic_shape.__getitem__[index]()


# Implementation of NDBuffer::get_dim.
fn _get_dim_impl[
    # Rank of the ndbuffer.
    rank: __mlir_type.index,
    # Static shape info on this ndbuffer, could be unknown.
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    # Index of the dimension to get the dimension value.
    index: __mlir_type.index,
](
    #  Tuple containing the runtime shape of the ndbuffer.
    dynamic_shape: StaticTuple[rank, __mlir_type.index]
) -> Int:
    # First try to extract the static info on this dimension,
    #  could be either a meta constant or an unknown.
    alias static_dim_value = _get_kgen_list_item[
        index, rank, __mlir_type.index
    ](shape)
    # Call the helper to resolve unknown by dynamic shape lookup if any.
    return _get_dim_helper[rank, static_dim_value, index](dynamic_shape)


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
    var dynamic_shape: StaticTuple[rank, __mlir_type.index]
    var dynamic_dtype: DType

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
    ) -> NDBuffer[rank, shape, type]:
        # Construct an NDBuffer type with statically known shape.
        # TODO: Verify that the shape is valid (i.e. does not have #kgen.unknown)
        var buf: NDBuffer[rank, shape, type]
        buf.data = DTypePointer[type](ptr)
        buf._rank = rank
        return buf

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
        dynamic_shape: StaticTuple[rank, __mlir_type.index],
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
        return _compute_ndbuffer_size[rank, shape, type](self)

    fn _offset(
        self, idx: StaticTuple[rank, __mlir_type.index]
    ) -> DTypePointer[type]:
        assert_param[rank <= 5]()
        return self.data.offset(
            _compute_ndbuffer_offset[rank, shape, type](self, idx)
        )

    fn __getitem__(
        self, idx: StaticTuple[rank, __mlir_type.index]
    ) -> SIMD[1, type]:
        return self.simd_load[1](idx)

    fn simd_load[
        width: __mlir_type.index
    ](self, idx: StaticTuple[rank, __mlir_type.index]) -> SIMD[width, type]:
        return self._offset(idx).simd_load[width]()

    fn __setitem__(
        self,
        idx: StaticTuple[rank, __mlir_type.index],
        val: __mlir_type[`!pop.scalar<`, type, `>`],
    ):
        # Stores a single value into the ndbuffer at the specified index
        var simd_val: SIMD[1, type]
        simd_val.value = val
        self.simd_store[1](idx, simd_val)

    fn __setitem__(
        self, idx: StaticTuple[rank, __mlir_type.index], val: SIMD[1, type]
    ):
        # Stores a single value into the ndbuffer at the specified index
        self.simd_store[1](idx, val)

    fn simd_store[
        width: __mlir_type.index
    ](self, idx: StaticTuple[rank, __mlir_type.index], val: SIMD[width, type]):
        # Stores a simd value into thendbuffer at the specified index
        self._offset(idx).simd_store[width](val)

    fn dim[index: __mlir_type.index](self) -> Int:
        return _get_dim_impl[rank, shape, index](self.dynamic_shape)

    fn flatten(self) -> Buffer[__mlir_attr.`#kgen.unknown : index`, type]:
        assert_param[
            _neg[contains[rank](__mlir_attr.`#kgen.unknown : index`, shape)]()
        ]()
        return Buffer[__mlir_attr.`#kgen.unknown : index`, type](
            self.data.address, product[rank](shape)
        )

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
    var idx: Int = effective_lbound
    while idx < effective_rbound:
        let storageVal = storage.load(idx)
        vector.__setitem__(idx, storageVal)
        idx += 1

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
    var idx: Int = effective_lbound
    while idx < effective_rbound:
        let storageVal = vector.__getitem__(idx)
        storage.store(idx, storageVal)
        idx += 1
