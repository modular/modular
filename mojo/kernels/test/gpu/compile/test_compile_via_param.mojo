# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-no-debug -D DUMP_GPU_ASM=True %s | FileCheck %s
# RUN: rm -fr %tmp-dir/test_compile_via_param/
# RUN: mkdir -p %tmp-dir/test_compile_via_param/
# RUN: %mojo-no-debug -D DUMP_GPU_ASM=%tmp-dir/test_compile_via_param/test_compile_via_param.ptx %s
# RUN: cat %tmp-dir/test_compile_via_param/test_compile_via_param.ptx | FileCheck %s
# RUN: rm -fr %tmp-dir/test_compile_via_param/

from gpu import thread_idx
from gpu.host import DeviceContext
from memory import UnsafePointer


def test_compile_function():
    print("== test_compile_function")

    fn kernel(x: UnsafePointer[Int]):
        x[0] = thread_idx.x

    # CHECK: tid.x

    with DeviceContext() as ctx:
        _ = ctx.compile_function[kernel]()


def main():
    test_compile_function()
