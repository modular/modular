# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import Context, Function
from gpu.host.memory import _malloc_managed


fn kernel_with_list(res: Pointer[Float32]):
    var list = List[Float32](10)
    for i in range(4):
        list.append(i + 1)
    res.store(list[0] * list[1] + list[2] * list[3])


fn test_kernel_with_list() raises:
    print("== test_kernel_with_list")
    var res = _malloc_managed[Float32](1)
    res[] = 0
    # CHECK: call.uni
    # CHECK: malloc,
    # CHECK: (
    # CHECK: param0
    # CHECK: );
    # CHECK: call.uni
    # CHECK: free,
    # CHECK: (
    # CHECK: param0
    # CHECK: );
    var kernel = Function[kernel_with_list](dump_ptx=True)
    kernel(res, block_dim=(1), grid_dim=(1))
    # CHECK: 16.0
    print("Res=", res[0])


fn main():
    try:
        with Context() as ctx:
            test_kernel_with_list()

    except e:
        print("CUDA error", e)
