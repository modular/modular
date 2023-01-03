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
        var buf : NDBuffer[rank, shape, type]
        buf.data = DTypePointer[type](ptr)
        return buf

    fn get_rank(self) -> Int:
        if rank == __mlir_attr.`#kgen.unknown : index`:
            return self.dynamic_rank
        return rank

    fn size(self, idx: StaticTuple[rank, __mlir_type.index]) -> Int:
        assert_param[rank <= 5]()
        var size = self.dim[0]()
        if rank == 1:
            return size
        size *= self.dim[1]()
        if rank == 2:
            return size
        size *= self.dim[2]()
        if rank == 3:
            return size
        size *= self.dim[3]()
        if rank == 4:
            return size
        size *= self.dim[4]()
        return size

    fn _offset(
        self, idx: StaticTuple[rank, __mlir_type.index]
    ) -> DTypePointer[type]:
        assert_param[rank <= 5]()
        if rank == 1:
            return self.data.offset(idx.__getitem__[0]())
        if rank == 2:
            return self.data.offset(
                idx.__getitem__[0]() + self.dim[0]() * idx.__getitem__[1]()
            )
        if rank == 3:
            return self.data.offset(
                idx.__getitem__[0]()
                + self.dim[0]()
                * (idx.__getitem__[1]() + self.dim[1]() * idx.__getitem__[2]())
            )
        if rank == 4:
            return self.data.offset(
                idx.__getitem__[0]()
                + self.dim[0]()
                * (
                    idx.__getitem__[1]()
                    + self.dim[1]()
                    * (
                        idx.__getitem__[2]()
                        + self.dim[2]() * idx.__getitem__[3]()
                    )
                )
            )
        if rank == 5:
            return self.data.offset(
                idx.__getitem__[0]()
                + self.dim[0]()
                * (
                    idx.__getitem__[1]()
                    + self.dim[1]()
                    * (
                        idx.__getitem__[2]()
                        + self.dim[2]()
                        * (
                            idx.__getitem__[3]()
                            + self.dim[3]() * idx.__getitem__[4]()
                        )
                    )
                )
            )
        # This should not be reachable.
        return self.data

    fn __getitem__(
        self, idx: StaticTuple[rank, __mlir_type.index]
    ) -> SIMD[1, type]:
        return self.simd_load[1](idx)

    fn simd_load[
        width: __mlir_type.index
    ](self, idx: StaticTuple[rank, __mlir_type.index]) -> SIMD[width, type]:
        return self._offset(idx).simd_load[width]()

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
