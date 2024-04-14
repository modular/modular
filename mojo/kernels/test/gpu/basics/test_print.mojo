# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.device_print import _printf
from gpu.host import Context, Function


# CHECK: Hello I got 42 7.2
fn main() raises:
    fn do_print(x: Int, y: Float64):
        _printf("Hello I got %lld %g\n", x, y)

    with Context() as ctx:
        var func = Function[__type_of(do_print), do_print]()
        func(Int(42), Float64(7.2), grid_dim=1, block_dim=1)
