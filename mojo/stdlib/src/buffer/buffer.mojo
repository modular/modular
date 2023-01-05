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


struct Buffer[size: __mlir_type.index, type: __mlir_type.`!kgen.dtype`]:
    var data: DTypePointer[type]
    var dynamic_size: Int

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`]
    ) -> Buffer[size, type]:
        # Construct a Buffer type with statically known size
        assert_param[size != __mlir_attr.`#kgen.unknown : index`]()
        return __mlir_op.`kgen.struct.create`[_type : Buffer[size, type]](
            DTypePointer[type](ptr), Int(size)
        )

    fn __new__(
        ptr: __mlir_type[`!pop.pointer<scalar<`, type, `>>`],
        in_size: Int,
    ) -> Buffer[size, type]:
        # Construct a Buffer type with dynamic size
        assert_param[size == __mlir_attr.`#kgen.unknown : index`]()
        return __mlir_op.`kgen.struct.create`[_type : Buffer[size, type]](
            DTypePointer[type](ptr), in_size
        )

    fn __len__(self) -> Int:
        # Returns the dynamic size if the buffer is not statically known,
        # otherwise returns the statically known size parameter.
        if size == __mlir_attr.`#kgen.unknown : index`:
            return self.dynamic_size
        return size

    fn __getitem__(self, idx: Int) -> SIMD[1, type]:
        # Loads a single element (SIMD of size 1) from the buffer at the
        # specified index
        return self.simd_load[1](idx)

    fn simd_load[width: __mlir_type.index](self, idx: Int) -> SIMD[width, type]:
        # Loads a simd value from the buffer at the specified index
        let offset = self.data.offset(idx)
        return offset.simd_load[width]()

    fn __setitem__(self, idx: Int, val: SIMD[1, type]):
        # Stores a single value into the buffer at the specified index
        self.simd_store[1](idx, val)

    fn simd_store[
        width: __mlir_type.index
    ](self, idx: Int, val: SIMD[width, type]):
        # Stores a simd value into the buffer at the specified index
        let offset = self.data.offset(idx)
        offset.simd_store[width](val)


@interface
fn _compute_ndbuffer_offset[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    ...


@implements(_compute_ndbuffer_offset)
fn _compute_ndbuffer_offset_rank_0[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    assert_param[rank == 0]()
    return 0


@implements(_compute_ndbuffer_offset)
fn _compute_ndbuffer_offset_rank_1[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    assert_param[rank == 1]()
    return idx.__getitem__[0]()


@implements(_compute_ndbuffer_offset)
fn _compute_ndbuffer_offset_rank_2[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    assert_param[rank == 2]()
    return idx.__getitem__[1]() + buf.dim[0]() * idx.__getitem__[0]()


@implements(_compute_ndbuffer_offset)
fn _compute_ndbuffer_offset_rank_3[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    assert_param[rank == 3]()
    return idx.__getitem__[2]() + buf.dim[1]() * (
        idx.__getitem__[1]() + buf.dim[0]() * idx.__getitem__[0]()
    )


@implements(_compute_ndbuffer_offset)
fn _compute_ndbuffer_offset_rank_4[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    assert_param[rank == 4]()
    return idx.__getitem__[3]() + buf.dim[2]() * (
        idx.__getitem__[2]()
        + buf.dim[1]()
        * (idx.__getitem__[1]() + buf.dim[0]() * idx.__getitem__[0]())
    )


@implements(_compute_ndbuffer_offset)
fn _compute_ndbuffer_offset_rank_5[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, shape, type], idx: StaticTuple[rank, __mlir_type.index]
) -> Int:
    assert_param[rank == 5]()
    return idx.__getitem__[4]() + buf.dim[4]() * (
        idx.__getitem__[3]()
        + buf.dim[2]()
        * (
            idx.__getitem__[2]()
            + buf.dim[1]()
            * (idx.__getitem__[1]() + buf.dim[0]() * idx.__getitem__[0]())
        )
    )


@interface
fn _compute_ndbuffer_size[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    ...


@implements(_compute_ndbuffer_size)
fn _compute_ndbuffer_size_rank_0[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    assert_param[rank == 0]()
    return 0


@implements(_compute_ndbuffer_size)
fn _compute_ndbuffer_size_rank_1[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    assert_param[rank == 1]()
    return buf.dim[0]()


@implements(_compute_ndbuffer_size)
fn _compute_ndbuffer_size_rank_2[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    assert_param[rank == 2]()
    return buf.dim[0]() * buf.dim[1]()


@implements(_compute_ndbuffer_size)
fn _compute_ndbuffer_size_rank_3[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    assert_param[rank == 3]()
    return buf.dim[0]() * buf.dim[1]() * buf.dim[2]()


@implements(_compute_ndbuffer_size)
fn _compute_ndbuffer_size_rank_4[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    assert_param[rank == 4]()
    return buf.dim[0]() * buf.dim[1]() * buf.dim[2]() * buf.dim[3]()


@implements(_compute_ndbuffer_size)
fn _compute_ndbuffer_size_rank_5[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[rank, shape, type]) -> Int:
    assert_param[rank == 5]()
    return (
        buf.dim[0]() * buf.dim[1]() * buf.dim[2]() * buf.dim[3]() * buf.dim[4]()
    )


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
        alias elem = __mlir_attr[
            `#kgen.param.expr<get_list_element, `,
            shape,
            `, `,
            index,
            `> : index`,
        ]
        if elem == __mlir_attr.`#kgen.unknown : index`:
            return self.dynamic_shape.__getitem__[index]()
        return elem
