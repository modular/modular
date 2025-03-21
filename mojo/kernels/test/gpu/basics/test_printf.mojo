# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# RUN: %mojo-no-debug %s | FileCheck %s

from collections.string.inline_string import _FixedString

from builtin._location import __source_location
from builtin.io import _printf
from gpu.host import DeviceContext
from layout import Layout


# CHECK-LABEL: == test_gpu_printf
fn test_gpu_printf() raises:
    print("== test_gpu_printf")

    #
    # Test that stdlib _printf works on GPU
    #

    fn do_print(x: Int, y: Float64):
        # CHECK: printf printed 98 123.456!
        _printf["printf printed %ld %g!\n"](x, y)
        # CHECK: printf printed more 0 1 2 3 4 5 6 7 8 9
        _printf[
            "printf printed more %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n"
        ](0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](
            Int(98), Float64(123.456), grid_dim=1, block_dim=1
        )
        # Ensure queued function finished before proceeding.
        ctx.synchronize()


fn main() raises:
    test_gpu_printf()
