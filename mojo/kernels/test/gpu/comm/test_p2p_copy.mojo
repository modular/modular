# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv
from sys import env_get_int

from gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
from testing import assert_almost_equal


fn p2p_copy_kernel(
    dst: UnsafePointer[Float32],
    src: UnsafePointer[Float32],
    num_elements: Int,
):
    var tid = global_idx.x
    if tid < num_elements:
        dst[tid] = src[tid]


fn launch_p2p_copy_kernel(
    ctx1: DeviceContext,
    dst_buf: DeviceBuffer[DType.float32],
    src_buf: DeviceBuffer[DType.float32],
    num_elements: Int,
) raises:
    alias BLOCK_SIZE = 256
    var grid_size = ceildiv(num_elements, BLOCK_SIZE)

    # Launch the kernel on both devices
    ctx1.enqueue_function[p2p_copy_kernel](
        dst_buf.unsafe_ptr(),
        src_buf.unsafe_ptr(),
        num_elements,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )

    # Synchronize both contexts to ensure completion
    ctx1.synchronize()


def main():
    alias log2_length = env_get_int["log2_length", 20]()
    constrained[log2_length > 0]()
    var length = 1 << log2_length

    var num_devices = DeviceContext.number_of_devices()
    if num_devices == 1:
        print("Only one device found, skipping peer-to-peer copy")
        return

    # Create contexts for both devices
    var ctx1 = DeviceContext(device_id=0)
    var ctx2 = DeviceContext(device_id=1)
    var can_access_p2p = ctx1.can_access(ctx2)
    print("ctx1 can access ctx2: ", can_access_p2p)
    if not can_access_p2p:
        print("Skipping test as ctx1 cannot access ctx2")
        return
    ctx1.enable_peer_access(ctx2)
    print("Checkpoint - successfully enabled peer access")

    # Create and initialize device buffers
    var dst_buf = ctx1.create_buffer_sync[DType.float32](length).enqueue_fill(
        1.0
    )
    var src_buf = ctx2.create_buffer_sync[DType.float32](length)

    # Initialize source data
    with src_buf.map_to_host() as host_data:
        for i in range(length):
            host_data[i] = Float32(i * 0.5)

    # Launch the P2P copy kernel
    launch_p2p_copy_kernel(ctx1, dst_buf, src_buf, length)

    # Wait for the copy to complete
    ctx1.synchronize()

    # Verify the data was copied correctly
    with dst_buf.map_to_host() as host_data:
        for i in range(length):
            assert_almost_equal(host_data[i], Float32(i * 0.5))

    print("P2P Direct Addressing Copy Test Passed")
