# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.device_print import _printf
from gpu.host import Context, Function

from layout import Layout

from builtin.io import _print_fmt
from utils.inlined_string import _FixedString


fn main() raises:
    fn do_print(x: Int, y: Float64):
        #
        # Test printing primitive types
        #

        # CHECK: Hello I got 42 7.2
        _print_fmt("Hello I got ", x, end="")
        # TODO: Make Float64 support Formattable
        _printf(" %g\n", [y])

        #
        # Test printing some non-primitive types
        #

        alias layout_str = _FixedString[50].format_sequence(
            Layout.row_major(2, 3)
        )

        # CHECK: layout from GPU: ((2, 3):(3, 1))
        _print_fmt("layout from GPU: ", layout_str)

    with Context() as ctx:
        var func = Function[__type_of(do_print), do_print]()
        func(Int(42), Float64(7.2), grid_dim=1, block_dim=1)
