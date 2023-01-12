# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Bool import Bool
from Int import Int
from SIMD import SIMD
from Assert import assert_param
from Pointer import DTypePointer
from Tuple import StaticTuple


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
        var result: Buffer[size, type]
        result.data = DTypePointer[type](ptr)
        result.dynamic_size = size
        result.dtype = type
        return result

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
        var result: Buffer[size, type]
        result.data = DTypePointer[type](ptr)
        result.dynamic_size = in_size
        return result

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
        let offset = self.data.offset(idx)
        return offset.simd_load[width]()

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
        let offset = self.data.offset(idx)
        offset.simd_store[width](val)


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
    alias static_dim_value = __mlir_attr[
        `#kgen.param.expr<get_list_element, `,
        shape,
        `, `,
        index,
        `> : index`,
    ]
    # Call the helper to resolve unknown by dynamic shape lookup if any.
    return _get_dim_helper[rank, static_dim_value, index](dynamic_shape)


struct NDBuffer[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
]:
    var data: DTypePointer[type]
    var dynamic_shape: StaticTuple[rank, __mlir_type.index]
    var dynamic_rank: Int

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
    ) -> NDBuffer[rank, shape, type]:
        # Construct an NDBuffer type with statically known shape.
        # TODO: Verify that the shape is valid (i.e. does not have #kgen.unknown)
        var buf: NDBuffer[rank, shape, type]
        buf.data = DTypePointer[type](ptr)
        return buf

    fn get_rank(self) -> Int:
        if rank == __mlir_attr.`#kgen.unknown : index`:
            return self.dynamic_rank
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
