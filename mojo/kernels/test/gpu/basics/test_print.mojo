# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import Context, Function

from layout import Layout

from builtin.io import _print_fmt, _printf
from utils.inlined_string import _FixedString


# CHECK-LABEL: == test_gpu_printf
fn test_gpu_printf() raises:
    print("== test_gpu_printf")

    #
    # Test that stdlib _printf works on GPU
    #

    fn do_print(x: Int, y: Float64):
        # CHECK: printf printed 98 123.456!
        _printf("printf printed %ld %g!\n", x, y)

    with Context() as ctx:
        var func = Function[do_print]()
        func(Int(98), Float64(123.456), grid_dim=1, block_dim=1)


# CHECK-LABEL: == test_gpu_print_formattable
fn test_gpu_print_formattable() raises:
    print("== test_gpu_print_formattable")

    fn do_print(x: Int, y: Float64):
        #
        # Test printing primitive types
        #

        # CHECK: Hello I got 42 7.2
        _print_fmt("Hello I got", x, y)

        #
        # Test printing SIMD values
        #

        var simd = SIMD[DType.float64, 4](
            0.0, -1.0, Float64.MIN, Float64.MAX_FINITE
        )
        # CHECK: [0, -1, -inf, 1.79769e+308]
        _print_fmt("SIMD values are:", simd)

        #
        # Test printing some non-primitive types
        #

        alias layout_str = _FixedString[50].format_sequence(
            Layout.row_major(2, 3)
        )

        # CHECK: layout from GPU: ((2, 3):(3, 1))
        _print_fmt("layout from GPU: ", layout_str)

    with Context() as ctx:
        var func = Function[do_print]()
        func(Int(42), Float64(7.2), grid_dim=1, block_dim=1)


fn main() raises:
    test_gpu_printf()
    test_gpu_print_formattable()
