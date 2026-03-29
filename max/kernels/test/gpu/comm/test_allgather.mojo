# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from std.sys import size_of, has_amd_gpu_accelerator

from comm.allgather import allgather
from comm import MAX_GPUS, Signal
from comm.sync import enable_p2p
import comm.vendor.ccl as vendor_ccl
from std.gpu.host import DeviceBuffer, DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from std.testing import assert_equal, assert_true


def all_gather_test[
    dtype: DType, ngpus: Int
](list_of_ctx: List[DeviceContext], lengths: List[Int]) raises -> None:
    """Test allgather with new variadic output semantics.

    Each device should receive individual copies of all inputs,
    not a single concatenated buffer.
    """

    # Create device buffers for all GPUs.
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[List[DeviceBuffer[dtype]]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[dtype], MutExternalOrigin]](
        capacity=ngpus
    )

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    # Calculate temp buffer size for signals.
    var max_length = 0
    for i in range(ngpus):
        max_length = max(max_length, lengths[i])
    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * max_length

    # Initialize input buffers and signal buffers.
    for i in range(ngpus):
        var length = lengths[i]

        # Create device buffer.
        in_bufs_list.append(list_of_ctx[i].create_buffer_sync[dtype](length))

        # Create host buffer with test data.
        var host_buffer = alloc[Scalar[dtype]](length)
        host_buffers.append(host_buffer)

        # Initialize with unique values per device.
        for j in range(length):
            host_buffer[j] = Scalar[dtype](
                i * 1000 + j
            )  # Device i has values i*1000 + index

        # Create and initialize signal buffers.
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

        # Copy to device.
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create output buffers - each device needs ngpus output buffers.
    for device_idx in range(ngpus):
        var device_outputs = List[DeviceBuffer[dtype]](capacity=ngpus)
        for input_idx in range(ngpus):
            var length = lengths[input_idx]
            device_outputs.append(
                list_of_ctx[device_idx].create_buffer_sync[dtype](length)
            )
        out_bufs_list.append(device_outputs^)

    # Build TileTensor arrays directly.
    comptime InTileType = type_of(
        TileTensor(
            in_bufs_list[0].unsafe_ptr(), row_major(Idx(lengths[0]))
        ).as_immut()
    )
    var tt_in_bufs = InlineArray[InTileType, ngpus](uninitialized=True)
    comptime for i in range(ngpus):
        tt_in_bufs[i] = TileTensor(
            in_bufs_list[i].unsafe_ptr(), row_major(Idx(lengths[i]))
        ).as_immut()

    comptime OutTileType = type_of(
        TileTensor(out_bufs_list[0][0].unsafe_ptr(), row_major(Idx(lengths[0])))
    )
    var tt_out_bufs = InlineArray[OutTileType, ngpus * ngpus](
        uninitialized=True
    )
    comptime for i in range(ngpus * ngpus):
        comptime device_idx = i // ngpus
        comptime input_idx = i % ngpus
        tt_out_bufs[i] = TileTensor(
            out_bufs_list[device_idx][input_idx].unsafe_ptr(),
            row_major(Idx(lengths[input_idx])),
        )

    # Optional: vendor CCL (only if all lengths are equal; NCCL/RCCL requires uniform count).
    var uniform = True
    for i in range(1, ngpus):
        if lengths[i] != lengths[0]:
            uniform = False
            break

    if uniform and has_amd_gpu_accelerator():
        # Reset outputs for vendor test
        for device_idx in range(ngpus):
            for input_idx in range(ngpus):
                list_of_ctx[device_idx].enqueue_memset[dtype](
                    out_bufs_list[device_idx][input_idx], val=0
                )

        try:
            print("  Testing vendor CCL allgather (uniform counts)")
            vendor_ccl.allgather[dtype=dtype, ngpus=ngpus](
                tt_in_bufs, tt_out_bufs, list_of_ctx
            )

            for i in range(ngpus):
                list_of_ctx[i].synchronize()
            _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)
        except:
            pass

    # Test the implementation with rank_sigs (P2P-capable).
    print("  Testing implementation with rank_sigs (P2P-capable)")

    for gpu_idx in range(ngpus):
        var device_out = InlineArray[OutTileType, ngpus](uninitialized=True)
        comptime for src_idx in range(ngpus):
            device_out[src_idx] = tt_out_bufs[gpu_idx * ngpus + src_idx]
        allgather(
            tt_in_bufs, device_out, rank_sigs, list_of_ctx[gpu_idx], gpu_idx
        )

    # Synchronize all devices.
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results for new implementation.
    _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)

    # Clean up.
    for i in range(ngpus):
        host_buffers[i].free()


def _verify_results[
    dtype: DType
](
    out_bufs_list: List[List[DeviceBuffer[dtype]]],
    list_of_ctx: List[DeviceContext],
    lengths: List[Int],
    ngpus: Int,
) raises:
    """Helper function to verify allgather results."""

    # Verify results - each device should have copies of all inputs.
    for device_idx in range(ngpus):
        for input_idx in range(ngpus):
            var length = lengths[input_idx]
            var host_output = alloc[Scalar[dtype]](length)

            # Copy output back to host.
            list_of_ctx[device_idx].enqueue_copy(
                host_output, out_bufs_list[device_idx][input_idx]
            )
            list_of_ctx[device_idx].synchronize()

            # Verify this matches the original input from device input_idx.
            for j in range(length):
                var expected = Scalar[dtype](input_idx * 1000 + j)
                try:
                    assert_equal(host_output[j], expected)
                except e:
                    print(
                        "Verification failed: device",
                        device_idx,
                        "should have copy of input",
                        input_idx,
                    )
                    print(
                        "Index",
                        j,
                        "value:",
                        host_output[j],
                        "expected:",
                        expected,
                    )
                    raise e^

            host_output.free()


def main() raises -> None:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    # Test configurations.
    comptime test_lengths: List[List[Int]] = [
        [8 * 1024, 8 * 1024],
        [128 * 1024, 8 * 1024],
        [8 * 1024, 256 * 1024],
        [8 * 1024, 8 * 1024, 8 * 1024, 8 * 1024],
        [128 * 1024, 256 * 1024, 8 * 1024, 64 * 1024],
        # Test uneven shapes.
        [37919, 37919, 37918, 37918],
        # Simple uneven case.
        [4, 3, 3],
        # Another uneven case with 2 GPUs.
        [1025, 1024],
        # Zero length cases
        [0, 0],
        [8 * 1024, 0],
        [0, 8 * 1024],
    ]

    comptime for test_idx in range(len(test_lengths)):
        comptime lengths = test_lengths[test_idx]
        comptime num_gpus = len(lengths)

        if DeviceContext.number_of_devices() < num_gpus:
            continue

        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        print("  Testing configuration:", test_idx, "with", num_gpus, "GPUs")
        all_gather_test[DType.bfloat16, ngpus=num_gpus](
            ctx, materialize[lengths]()
        )
