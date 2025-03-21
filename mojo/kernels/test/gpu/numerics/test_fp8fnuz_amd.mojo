# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import DeviceContext
from memory import UnsafePointer


def test_fp8_constructor(ctx: DeviceContext):
    fn kernel(ptr: UnsafePointer[Float8_e5m2fnuz]):
        ptr[] = Float8_e5m2fnuz(42.0)

    # CHECK: v_mov_b32_e32 {{.*}}, 0x55
    # CHECK: store i8 85, ptr %{{.*}}, align 1
    _ = ctx.compile_function[
        kernel,
        dump_llvm=True,
        dump_asm=True,
    ]()


def main():
    with DeviceContext() as ctx:
        test_fp8_constructor(ctx)
