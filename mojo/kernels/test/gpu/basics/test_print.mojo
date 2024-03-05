# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu.host import Function, Context
from gpu.device_print import _printf


# CHECK: Hello I got 42
fn main() raises:
    fn do_print(x: Int):
        _printf("Hello I got %lld\n", x)

    with Context() as ctx:
        var func = Function[__type_of(do_print), do_print]()
        func(Int(42), grid_dim=1, block_dim=1)
