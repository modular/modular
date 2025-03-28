# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: H100-GPU
# RUN: not --crash %mojo-no-debug %s

from gpu.host import DeviceContext
from memory import UnsafePointer
from testing import *


fn kernel(res: UnsafePointer[UInt32]):
    res[] = 0


# Here the argument is a host pointer and not a device pointer, so we expect
# an error about an illegal memory address.
def test_function_error(ctx: DeviceContext):
    print("== test_function_error")
    try:
        var res_host = UnsafePointer[UInt32].alloc(1)
        ctx.enqueue_function[kernel](res_host, block_dim=(1), grid_dim=(1))
        ctx.synchronize()
        res_host.free()
    except e:
        assert_true(
            "Kernels/test/gpu/device_context/test_function_error.mojo:27:24"
            in String(e)
        )


def main():
    with DeviceContext() as ctx:
        test_function_error(ctx)
