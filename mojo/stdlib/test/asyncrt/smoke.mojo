# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V1 %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cuda %s

from asyncrt_test_utils import (
    create_test_device_context,
    expect_eq,
    is_v2_context,
)

from gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceAttribute,
    DeviceStream,
    device_count,
)


fn _ownership_helper(
    owned ctx: DeviceContext,
) raises -> DeviceContext:
    var ctx_copy = ctx
    print("local ctx_copy: " + ctx_copy.name())
    return ctx_copy


fn _ownership_helper_buf[
    type: DType
](owned buf: DeviceBuffer[type]) raises -> DeviceBuffer[type]:
    var buf_copy = buf
    print("local buf_copy: " + str(len(buf)))
    return buf_copy


fn _run_ownership_transfer(ctx: DeviceContext) raises:
    print("-")
    print("_run_ownership_transfer()")

    var ctx_copy = _ownership_helper(ctx)
    print("ctx_copy: " + ctx_copy.name())

    var buf = ctx.create_buffer_sync[DType.float32](32)
    print("buf: " + str(len(buf)))
    var buf_copy = _ownership_helper_buf(buf)
    print("buf_copy: " + str(len(buf_copy)))

    # Make sure buf survives to the end of the test function.
    _ = buf


fn _run_device_info(ctx: DeviceContext) raises:
    print("-")
    print("_run_device_info()")

    (free_before, total_before) = ctx.get_memory_info()

    var buf = ctx.create_buffer_sync[DType.float32](20 * 1024 * 1024)

    (free_after, total_after) = ctx.get_memory_info()
    print(
        "Memory info (before -> after) - total: "
        + str(total_before)
        + " -> "
        + str(total_after)
        + " , free: "
        + str(free_before)
        + " -> "
        + str(free_after)
    )

    # Make sure buf survives to the end of the test function.
    _ = buf


fn _run_compute_capability(ctx: DeviceContext) raises:
    print("-")
    print("_run_compute_capability()")

    print("Compute capability: " + str(ctx.compute_capability()))


fn _run_get_attribute(ctx: DeviceContext) raises:
    print("-")
    print("_run_get_attribute()")

    print("clock_rate: " + str(ctx.get_attribute(DeviceAttribute.CLOCK_RATE)))
    print("warp_size: " + str(ctx.get_attribute(DeviceAttribute.WARP_SIZE)))


fn _run_get_stream(ctx: DeviceContext) raises:
    print("-")
    print("_run_get_stream()")

    if not is_v2_context():
        print("Skipping test.")
        return

    print("Getting the stream.")
    var stream = ctx.stream()
    print("Synchronizing on `stream`.")
    stream.synchronize()


fn _run_access_peer(ctx: DeviceContext, peer: DeviceContext) raises:
    print("-")
    print("_run_access_peer()")

    print(
        "can access "
        + ctx.name()
        + "->"
        + peer.name()
        + ": "
        + str(ctx.can_access(peer))
    )


fn _run_enable_disable_peer_access(ctx: DeviceContext) raises:
    print("-")
    print("_run_enable_disable_peer_access()")
    var num_gpus = device_count()

    if num_gpus < 2:
        print("Only one GPU found, skipping test.")
        return

    var peer = create_test_device_context(gpu_id=1)

    if not ctx.can_access(peer):
        print("Devices can't access each other, skipping test.")
        return

    ctx.enable_peer_access(peer)
    ctx.disable_peer_access(peer)


fn main() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    _run_ownership_transfer(ctx)
    _run_device_info(ctx)
    _run_compute_capability(ctx)
    _run_get_attribute(ctx)

    _run_get_stream(ctx)

    _run_access_peer(ctx, create_test_device_context())
    _run_enable_disable_peer_access(ctx)

    print("Done.")
