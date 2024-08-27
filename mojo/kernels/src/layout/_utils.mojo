# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled

from os import abort

from gpu.host.memory import _free, _malloc_managed
from layout import *

alias alloc_fn_type = fn[layout: Layout, dtype: DType] () -> UnsafePointer[
    Scalar[dtype]
]

alias alloc_runtime_fn = fn[layout: Layout, dtype: DType] (
    runtime_layout: RuntimeLayout[layout]
) -> UnsafePointer[Scalar[dtype]]

alias free_fn_type = fn[dtype: DType] (
    ptr: UnsafePointer[Scalar[dtype]]
) -> None


fn cpu_alloc[layout: Layout, dtype: DType]() -> UnsafePointer[Scalar[dtype]]:
    return UnsafePointer[Scalar[dtype]].alloc(layout.size())


fn cpu_alloc_runtime[
    layout: Layout, dtype: DType
](runtime_layout: RuntimeLayout[layout]) -> UnsafePointer[Scalar[dtype]]:
    return UnsafePointer[Scalar[dtype]].alloc(runtime_layout.size())


@always_inline
fn cpu_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    ptr.free()


fn gpu_managed_alloc[
    layout: Layout, dtype: DType
]() -> UnsafePointer[Scalar[dtype]]:
    try:
        return _malloc_managed[Scalar[dtype]](layout.size())
    except e:
        return abort[UnsafePointer[Scalar[dtype]]]("Can't alloc gpu memory")


fn gpu_managed_alloc_runtime[
    layout: Layout, dtype: DType
](runtime_layout: RuntimeLayout[layout]) -> UnsafePointer[Scalar[dtype]]:
    try:
        return _malloc_managed[Scalar[dtype]](runtime_layout.size())
    except e:
        return abort[UnsafePointer[Scalar[dtype]]]("Can't alloc gpu memory")


fn gpu_free[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
    try:
        return _free(ptr)
    except e:
        abort("Can't free gpu memory")


struct ManagedLayoutTensor[
    dtype: DType,
    layout: Layout,
    alloc_fn: alloc_fn_type = cpu_alloc,
    free_fn: free_fn_type = cpu_free,
    alloc_runtime_fn: alloc_runtime_fn = cpu_alloc_runtime,
]:
    var tensor: LayoutTensor[dtype, layout]

    @always_inline
    fn __init__(inout self):
        self.tensor = LayoutTensor[dtype, layout](alloc_fn[layout, dtype]())

    @always_inline
    fn __init__(inout self, runtime_layout: RuntimeLayout[layout]):
        self.tensor = LayoutTensor[dtype, layout](
            alloc_runtime_fn[layout, dtype](runtime_layout),
            runtime_layout,
        )

    @always_inline
    fn __del__(owned self):
        free_fn[dtype](self.tensor.ptr)


alias ManagedLayoutGPUTensor = ManagedLayoutTensor[
    alloc_fn=gpu_managed_alloc,
    free_fn=gpu_free,
    alloc_runtime_fn=gpu_managed_alloc_runtime,
]
