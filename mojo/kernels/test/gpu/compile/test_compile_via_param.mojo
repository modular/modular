# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-no-debug-no-assert -D DUMP_MOJO_ASM=True %s | FileCheck %s


from gpu import thread_idx
from gpu.host import DeviceContext
from memory import UnsafePointer


# CHECK-LABEL: test_compile_function
def test_compile_function():
    print("== test_compile_function")

    fn kernel(x: UnsafePointer[Int]):
        x[0] = thread_idx.x

    # CHECK: tid.x

    with DeviceContext() as ctx:
        _ = ctx.compile_function[kernel]()


def main():
    test_compile_function()
