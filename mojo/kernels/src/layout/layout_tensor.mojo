# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .layout import *
from memory.unsafe import DTypePointer


struct LayoutTensor[dtype: DType]:
    var ptr: DTypePointer[dtype]
    var layout: Layout

    @always_inline
    fn __init__(inout self, layout: Layout, ptr: DTypePointer[dtype]):
        self.ptr = ptr
        self.layout = layout

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.layout = existing.layout

    @always_inline
    fn __getitem__(self, idx: IntTuple) -> SIMD[dtype, 1]:
        return self.ptr.simd_load[1](self.layout(idx))

    @always_inline
    fn __setitem__(self, idx: IntTuple, val: SIMD[dtype, 1]):
        self.ptr.simd_store[1](self.layout(idx), val)


fn tile[
    dtype: DType
](src: LayoutTensor[dtype], tiler: Tiler, coords: IntTuple) -> LayoutTensor[
    dtype
]:
    var tiled_layout = zipped_divide(src.layout, tiler)
    if len(coords) > 0:
        let offset = inner_product(coords, tiled_layout[1].stride)
        let res_tensor = LayoutTensor[dtype](
            tiled_layout[0], src.ptr.offset(offset)
        )
        return res_tensor
    return LayoutTensor[dtype](tiled_layout, src.ptr)
