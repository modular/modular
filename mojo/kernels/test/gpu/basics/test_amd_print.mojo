# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from sys.intrinsics import lane_id


fn main():
    try:
        # CHECK-LABEL: lane_ids
        # CHECK 0
        # CHECK 1
        # CHECK 2
        # CHECK 3
        print("=== lane_ids ===")
        with DeviceContext() as ctx:

            @parameter
            fn do_print():
                print(lane_id())

            var func = ctx.compile_function[do_print]()

            ctx.enqueue_function(func, grid_dim=1, block_dim=4)
            ctx.synchronize()
    except e:
        print("HIP_ERROR:", e)
