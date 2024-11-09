# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv
from gpu.host import DeviceContext, device_count, DeviceBufferVariant
from gpu import BlockIdx, GridDim, ThreadIdx, BlockDim
from memory import UnsafePointer
from sys import env_get_int
from testing import assert_almost_equal


fn p2p_copy_kernel(
    dst: UnsafePointer[Scalar[DType.float32]],
    src: UnsafePointer[Scalar[DType.float32]],
    num_elements: Int,
):
    var tid = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    if tid < num_elements:
        dst[tid] = src[tid]


fn launch_p2p_copy_kernel(
    ctx1: DeviceContext,
    dst_buf: DeviceBufferVariant[DType.float32],
    src_buf: DeviceBufferVariant[DType.float32],
    num_elements: Int,
) raises:
    alias BLOCK_SIZE = 256
    var grid_size = ceildiv(num_elements, BLOCK_SIZE)

    # Compile the kernel for both devices
    var kernel1 = ctx1.compile_function[p2p_copy_kernel]()

    # Launch the kernel on both devices
    ctx1.enqueue_function(
        kernel1,
        dst_buf.ptr,
        src_buf.ptr,
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

    var num_devices = device_count()
    if num_devices == 1:
        print("Only one device found, skipping peer-to-peer copy")
        return

    # Create contexts for both devices
    var ctx1 = DeviceContext(gpu_id=0)
    var ctx2 = DeviceContext(gpu_id=1)
    var can_access_p2p = ctx1.can_access(ctx2)
    print("ctx1 can access ctx2: ", can_access_p2p)
    if not can_access_p2p:
        print("Skipping test as ctx1 cannot access ctx2")
        return
    ctx1.enable_peer_access(ctx2)
    print("Checkpoint - successfully enabled peer access")

    # Create and initialize device buffers
    var dst_buf = ctx1.create_buffer_sync[DType.float32](length)
    var src_buf = ctx2.create_buffer_sync[DType.float32](length)

    # Initialize source data
    var host_data = UnsafePointer[Scalar[DType.float32]].alloc(length)
    for i in range(length):
        host_data[i] = 1.0

    # Copy initial data to source buffer
    ctx2.enqueue_copy_to_device(src_buf, host_data)

    # Launch the P2P copy kernel
    launch_p2p_copy_kernel(ctx1, dst_buf, src_buf, length)

    # Verify the data was copied correctly
    var host_verify = UnsafePointer[Scalar[DType.float32]].alloc(length)
    ctx1.enqueue_copy_from_device(host_verify, dst_buf)
    ctx1.synchronize()

    for i in range(length):
        assert_almost_equal(host_verify[i], 1.0)

    print("P2P Direct Addressing Copy Test Passed")

    # Cleanup
    host_data.free()
    host_verify.free()
