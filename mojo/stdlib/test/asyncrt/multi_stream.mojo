# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Note: CPU function compilation not supported
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid]  # breakpoint1


fn test_concurrent_copy(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    print("-------")
    print(
        "Running test_concurrent_copy("
        + ctx1.name()
        + ", "
        + ctx2.name()
        + "):"
    )

    alias length = 1 * 1024 * 1024

    var in0_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in0_dev2 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in0_dev3 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in1_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in1_dev2 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in1_dev3 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev2 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev3 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_host1 = ctx2.enqueue_create_host_buffer[DType.float32](length)
    var out_host2 = ctx2.enqueue_create_host_buffer[DType.float32](length)
    var out_host3 = ctx2.enqueue_create_host_buffer[DType.float32](length)

    # Initialize the variable inputs with known values.
    with ctx1.map_to_host(in0_dev1) as in_host1, ctx1.map_to_host(
        in0_dev2
    ) as in_host2, ctx1.map_to_host(in0_dev3) as in_host3:
        for i in range(length):
            var index = i % 2048
            in_host1[i] = index
            in_host2[i] = 2 * index
            in_host3[i] = 3 * index

    # Initialize the fixed (right) inputs.
    ctx1.enqueue_memset(in1_dev1, 1.0)
    ctx1.enqueue_memset(in1_dev2, 2.0)
    ctx1.enqueue_memset(in1_dev3, 3.0)
    # Initialize the outputs with known bad values
    ctx1.enqueue_memset(out_dev1, 101.0)
    ctx1.enqueue_memset(out_dev2, 102.0)
    ctx1.enqueue_memset(out_dev3, 103.0)
    for i in range(length):
        out_host1[i] = 0.5
        out_host2[i] = 0.25
        out_host3[i] = 0.125

    # Pre-compile and pre-register the device function
    var dev_func = ctx1.compile_function[vec_func]()

    ctx1.synchronize()

    var block_dim = 1

    ctx1.enqueue_function(
        dev_func,
        in0_dev1,
        in1_dev1,
        out_dev1,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev1.reassign_ownership_to(ctx2)
    ctx2.enqueue_copy_from_device(out_host1.unsafe_ptr(), out_dev1)
    ctx1.enqueue_function(
        dev_func,
        in0_dev2,
        in1_dev2,
        out_dev2,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev2.reassign_ownership_to(ctx2)
    ctx2.enqueue_copy_from_device(out_host2.unsafe_ptr(), out_dev2)
    ctx1.enqueue_function(
        dev_func,
        in0_dev3,
        in1_dev3,
        out_dev3,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev3.reassign_ownership_to(ctx2)
    ctx2.enqueue_copy_from_device(out_host3.unsafe_ptr(), out_dev3)

    # Wait for the copies to be completed.
    ctx2.synchronize()

    for i in range(length):
        var index = i % 2048
        if i < 10:
            print("at index", i, "the value is", out_host1[i])
            print("at index", i, "the value is", out_host2[i])
            print("at index", i, "the value is", out_host3[i])
        expect_eq(
            out_host1[i],
            Float32(index + 1.0),
            "out_host1[",
            i,
            "] is ",
            out_host1[i],
        )
        expect_eq(
            out_host2[i],
            Float32(2.0 * index + 2.0),
            "out_host2[",
            i,
            "] is ",
            out_host2[i],
        )
        expect_eq(
            out_host3[i],
            Float32(3.0 * index + 3.0),
            "out_host3[",
            i,
            "] is ",
            out_host3[i],
        )

    print("Done.")


fn test_concurrent_func(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    print("-------")
    print(
        "Running test_concurrent_func("
        + ctx1.name()
        + ", "
        + ctx2.name()
        + "):"
    )

    alias length = 20 * 1024 * 1024

    var in_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in_dev2 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in_dev3 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in_dev4 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in_dev5 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev2 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev3 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev4 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_host = ctx2.enqueue_create_host_buffer[DType.float32](length)

    # Initialize the variable inputs with known values.
    with ctx1.map_to_host(in_dev1) as in_host1, ctx1.map_to_host(
        in_dev2
    ) as in_host2, ctx1.map_to_host(in_dev3) as in_host3:
        for i in range(length):
            var index = i % 2048
            in_host1[i] = index
            in_host2[i] = 2 * index
            in_host3[i] = 3 * index

    # Initialize the fixed (right) inputs.
    ctx1.enqueue_memset(in_dev4, 1.0)
    ctx1.enqueue_memset(in_dev5, 2.0)
    # Initialize the outputs with known bad values
    ctx1.enqueue_memset(out_dev1, 101.0)
    ctx1.enqueue_memset(out_dev2, 102.0)
    ctx1.enqueue_memset(out_dev3, 103.0)
    ctx1.enqueue_memset(out_dev4, 104.0)
    for i in range(length):
        out_host[i] = 0.5

    # Pre-compile and pre-register the device function
    var dev_func1 = ctx1.compile_function[vec_func]()
    var dev_func2 = ctx2.compile_function[vec_func]()

    ctx1.synchronize()
    ctx2.synchronize()

    var block_dim = 1

    ctx1.enqueue_function(
        dev_func1,
        in_dev1,
        in_dev4,
        out_dev1,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx2.enqueue_wait_for(ctx1)
    ctx2.enqueue_function(
        dev_func2,
        in_dev2,
        out_dev1,
        out_dev2,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx1.enqueue_function(
        dev_func1,
        in_dev3,
        in_dev5,
        out_dev3,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx2.enqueue_wait_for(ctx1)
    ctx2.enqueue_function(
        dev_func2,
        out_dev2,
        out_dev3,
        out_dev4,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx1.enqueue_wait_for(ctx2)
    ctx1.enqueue_copy_from_device(out_host.unsafe_ptr(), out_dev4)

    # Wait for the copies to be completed.
    ctx1.synchronize()

    for i in range(length):
        var index = i % 2048
        var o1 = index + 1
        var o2 = 2 * index + o1
        var o3 = 3 * index + 2
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            Float32(o2 + o3),
            "out_host[",
            i,
            "] is ",
            out_host[i],
        )

    print("Done.")


fn main() raises:
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    test_concurrent_copy(ctx1, ctx2)
    test_concurrent_func(ctx1, ctx2)
