# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from builtin._location import __source_location
from builtin.io import _printf
from gpu.host import DeviceContext
from layout import Layout

from collections.string.inline_string import _FixedString


# CHECK-LABEL: == test_gpu_printf
fn test_gpu_printf() raises:
    print("== test_gpu_printf")

    #
    # Test that stdlib _printf works on GPU
    #

    fn do_print(x: Int, y: Float64):
        # CHECK: printf printed 98 123.456!
        _printf["printf printed %ld %g!\n"](x, y)

    with DeviceContext() as ctx:
        var func = ctx.compile_function[do_print]()
        ctx.enqueue_function(
            func, Int(98), Float64(123.456), grid_dim=1, block_dim=1
        )
        # Ensure queued function finished before proceeding.
        ctx.synchronize()


fn main() raises:
    test_gpu_printf()
