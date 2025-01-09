# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from gpu import amd

from buffer import DimList, NDBuffer


fn main():
    try:
        with DeviceContext() as ctx:

            @parameter
            fn do_print():
                # CHECK-LABEL: 32
                amd.print()

            var func = ctx.compile_function[do_print]()
            ctx.enqueue_function(func, grid_dim=1, block_dim=1)
            ctx.synchronize()
    except e:
        print("HIP_ERROR:", e)
