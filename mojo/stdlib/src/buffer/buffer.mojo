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

    fn _is_dynamic(self) -> Bool:
        # Returns true if the size is not statically known.
        return __mlir_attr[`#kgen.param.expr<eq, #kgen.unknown : index,`,
                           size, `> : i1`]

    fn __len__(self) -> Int:
        # Returns the dynamic size if the buffer is not statically known,
        # otherwise returns the statically known size parameter.
        if self._is_dynamic():
            return self.dynamic_size
        return size

    fn __getitem__(self, idx: Int) -> SIMD[1, type]:
        # Loads a single element (SIMD of size 1) from the buffer at the
        # specified index
        return self.simd_load[1](idx)

    fn simd_load[width: __mlir_type.index](self, idx: Int) -> SIMD[width, type]:
        # Loads a simd value from the buffer at the specified index
        var offset = self.data.offset(idx)
        return offset.simd_load[width]()

    fn __setitem__(self, idx: Int, val: SIMD[1, type]):
        # Stores a single value into the buffer at the specified index
        self.simd_store[1](idx, val)

    fn simd_store[
        width: __mlir_type.index
    ](self, idx: Int, val: SIMD[width, type]):
        # Stores a simd value into the buffer at the specified index
        var offset = self.data.offset(idx)
        offset.simd_store[width](val)
