# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceContext


fn main() raises:
    var ctx = create_test_device_context()

    alias length = 20

    var in_buf = ctx.enqueue_create_buffer[DType.int64](length)
    var out_buf = ctx.enqueue_create_buffer[DType.int64](length)

    with ctx.map_to_host(in_buf) as in_map:
        for i in range(length):
            in_map[i] = i

    in_buf.enqueue_copy_to(out_buf)

    with ctx.map_to_host(out_buf) as out_map:
        for i in range(length):
            expect_eq(out_map[i], i)

    print("Done")
