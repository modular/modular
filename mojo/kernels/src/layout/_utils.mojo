# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled

from gpu.host.memory import _free, _malloc_managed
from layout import *

alias alloc_fn_type = fn[layout: Layout, dtype: DType] () -> DTypePointer[dtype]

alias free_fn_type = fn[dtype: DType] (ptr: DTypePointer[dtype]) -> None


fn cpu_alloc[layout: Layout, dtype: DType]() -> DTypePointer[dtype]:
    return DTypePointer[dtype].alloc(layout.size())


@always_inline
fn cpu_free[dtype: DType](ptr: DTypePointer[dtype]):
    ptr.free()


fn gpu_managed_alloc[layout: Layout, dtype: DType]() -> DTypePointer[dtype]:
    try:
        return _malloc_managed[dtype](layout.size())
    except e:
        return abort[DTypePointer[dtype]]("Can't alloc gpu memory")


fn gpu_free[dtype: DType](ptr: DTypePointer[dtype]):
    try:
        return _free(ptr)
    except e:
        abort("Can't free gpu memory")


struct ManagedLayoutTensor[
    dtype: DType,
    layout: Layout,
    alloc_fn: alloc_fn_type = cpu_alloc,
    free_fn: free_fn_type = cpu_free,
]:
    var tensor: LayoutTensor[dtype, layout]

    @always_inline
    fn __init__(inout self):
        self.tensor = LayoutTensor[dtype, layout](alloc_fn[layout, dtype]())

    @always_inline
    fn __del__(owned self):
        free_fn[dtype](self.tensor.ptr)
