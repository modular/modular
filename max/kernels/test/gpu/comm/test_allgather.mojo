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


from sys import size_of, has_amd_gpu_accelerator

from buffer import NDBuffer
from buffer.dimlist import DimList
from comm.allgather import allgather
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from testing import assert_equal, assert_true
from utils.index import IndexList


def all_gather_test[
    dtype: DType, rank: Int, ngpus: Int
](list_of_ctx: List[DeviceContext], lengths: List[Int]) -> None:
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
        var host_nd_buf = NDBuffer[dtype, rank](host_buffer, DimList(length))
        for j in range(length):
            host_nd_buf[j] = Scalar[dtype](
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

    # Create input NDBuffers.
    var in_bufs = InlineArray[NDBuffer[dtype, rank, ImmutAnyOrigin], ngpus](
        fill={}
    )

    for i in range(ngpus):
        in_bufs[i] = NDBuffer[dtype, rank](
            in_bufs_list[i].unsafe_ptr(), DimList(lengths[i])
        )

    # Create flat output buffer array (ngpus * ngpus).
    var out_bufs = InlineArray[
        NDBuffer[dtype, rank, MutAnyOrigin], ngpus * ngpus
    ](fill={})

    for device_idx in range(ngpus):
        for input_idx in range(ngpus):
            var output_idx = device_idx * ngpus + input_idx
            out_bufs[output_idx] = NDBuffer[dtype, rank](
                out_bufs_list[device_idx][input_idx].unsafe_ptr(),
                DimList(lengths[input_idx]),
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

        var flat_out = InlineArray[
            NDBuffer[dtype, rank, MutAnyOrigin], ngpus * ngpus
        ](fill={})
        for device_idx in range(ngpus):
            for input_idx in range(ngpus):
                var idx = device_idx * ngpus + input_idx
                flat_out[idx] = NDBuffer[dtype, rank](
                    out_bufs_list[device_idx][input_idx].unsafe_ptr(),
                    DimList(lengths[input_idx]),
                )

        try:
            print("  Testing vendor CCL allgather (uniform counts)")
            vendor_ccl.allgather[dtype=dtype, rank=rank, ngpus=ngpus](
                in_bufs, flat_out, list_of_ctx
            )

            for i in range(ngpus):
                list_of_ctx[i].synchronize()
            _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)
        except:
            pass

    # Test the naive implementation explicitly.
    print("  Testing backward compatible implementation (naive path)")
    allgather(in_bufs, out_bufs, list_of_ctx)

    # Synchronize all devices.
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results for old implementation.
    _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)

    # Reset output buffers for second test.
    for device_idx in range(ngpus):
        for input_idx in range(ngpus):
            list_of_ctx[device_idx].enqueue_memset[dtype](
                out_bufs_list[device_idx][input_idx], val=0
            )

    # Test the implementation with rank_sigs (P2P-capable).
    print("  Testing new implementation with rank_sigs (P2P-capable)")
    allgather(in_bufs, out_bufs, rank_sigs, list_of_ctx)

    # Synchronize all devices.
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results for new implementation.
    _verify_results[dtype](out_bufs_list, list_of_ctx, lengths, ngpus)

    # Clean up.
    for i in range(ngpus):
        host_buffers[i].free()


fn _verify_results[
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
            var host_output_managed = ManagedLayoutTensor[
                dtype, Layout(UNKNOWN_VALUE)
            ](
                RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                    IndexList[1](length)
                ),
                list_of_ctx[device_idx],
            )
            var host_output = host_output_managed.tensor[update=False]().ptr

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


def main() -> None:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )

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
        all_gather_test[DType.bfloat16, rank=1, ngpus=num_gpus](
            ctx, materialize[lengths]()
        )
