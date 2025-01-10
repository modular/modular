# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from sys._amdgpu import printf_begin, printf_append_string_n


fn main():
    try:
        with DeviceContext() as ctx:

            @parameter
            fn do_print():
                # CHECK-LABEL: 32 hello
                print(32, "hello")

                # Note 511 chars plus the implicit \n that the writer will add
                # therefore checking max buffer size
                # CHECK-LABEL: HihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihih
                print(
                    "HihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihihiHihihih"
                )

                var func = ctx.compile_function[do_print]()

            ctx.enqueue_function(func, grid_dim=1, block_dim=1)
            ctx.synchronize()
    except e:
        print("HIP_ERROR:", e)
