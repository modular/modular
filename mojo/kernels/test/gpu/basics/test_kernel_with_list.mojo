# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from memory import UnsafePointer


fn kernel_with_list(res: UnsafePointer[Float32]):
    var list = List[Float32](10)
    for i in range(4):
        list.append(i + 1)
    res[] = list[0] * list[1] + list[2] * list[3]


fn test_kernel_with_list(ctx: DeviceContext) raises:
    print("== test_kernel_with_list")
    var res_host = UnsafePointer[Float32].alloc(1)
    var res_device = ctx.enqueue_create_buffer[DType.float32](1)
    res_host[0] = 0
    ctx.enqueue_copy(res_device, res_host)
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
    ctx.enqueue_function[kernel_with_list, dump_asm=True](
        res_device, block_dim=(1), grid_dim=(1)
    )
    ctx.enqueue_copy(res_host, res_device)
    ctx.synchronize()
    # CHECK: 16.0
    print("Res=", res_host[0])

    _ = res_device
    res_host.free()


def main():
    with DeviceContext() as ctx:
        test_kernel_with_list(ctx)
