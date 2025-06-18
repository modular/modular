# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.comm.allgather import allgather
from gpu.host import DeviceBuffer, DeviceContext
from testing import assert_equal


def all_gather_test[
    type: DType, rank: Int, ngpus: Int
](list_of_ctx: List[DeviceContext], lengths: List[Int]) -> None:
    """Test allgather with new variadic output semantics.

    Each device should receive individual copies of all inputs,
    not a single concatenated buffer.
    """
    # Create device buffers for all GPUs.
    var in_bufs_list = List[DeviceBuffer[type]](capacity=ngpus)
    var out_bufs_list = List[List[DeviceBuffer[type]]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[type]]](capacity=ngpus)

    # Initialize input buffers.
    for i in range(ngpus):
        var length = lengths[i]

        # Create device buffer.
        in_bufs_list.append(list_of_ctx[i].create_buffer_sync[type](length))

        # Create host buffer with test data.
        var host_buffer = UnsafePointer[Scalar[type]].alloc(length)
        host_buffers.append(host_buffer)

        # Initialize with unique values per device.
        var host_nd_buf = NDBuffer[type, rank](host_buffer, DimList(length))
        for j in range(length):
            host_nd_buf[j] = Scalar[type](
                i * 1000 + j
            )  # Device i has values i*1000 + index

        # Copy to device.
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create output buffers - each device needs ngpus output buffers.
    for device_idx in range(ngpus):
        var device_outputs = List[DeviceBuffer[type]](capacity=ngpus)
        for input_idx in range(ngpus):
            var length = lengths[input_idx]
            device_outputs.append(
                list_of_ctx[device_idx].create_buffer_sync[type](length)
            )
        out_bufs_list.append(device_outputs)

    # Create input NDBuffers.
    var in_bufs = InlineArray[NDBuffer[type, rank, MutableAnyOrigin], ngpus](
        NDBuffer[type, rank, MutableAnyOrigin]()
    )

    for i in range(ngpus):
        in_bufs[i] = NDBuffer[type, rank](
            in_bufs_list[i]._unsafe_ptr(), DimList(lengths[i])
        )

    # Create flat output buffer array (ngpus * ngpus).
    var out_bufs = InlineArray[
        NDBuffer[type, rank, MutableAnyOrigin], ngpus * ngpus
    ](NDBuffer[type, rank, MutableAnyOrigin]())

    for device_idx in range(ngpus):
        for input_idx in range(ngpus):
            var output_idx = device_idx * ngpus + input_idx
            out_bufs[output_idx] = NDBuffer[type, rank](
                out_bufs_list[device_idx][input_idx]._unsafe_ptr(),
                DimList(lengths[input_idx]),
            )

    allgather(in_bufs, out_bufs, list_of_ctx)

    # Synchronize all devices.
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results - each device should have copies of all inputs.
    for device_idx in range(ngpus):
        for input_idx in range(ngpus):
            var length = lengths[input_idx]
            var host_output = UnsafePointer[Scalar[type]].alloc(length)

            # Copy output back to host.
            list_of_ctx[device_idx].enqueue_copy(
                host_output, out_bufs_list[device_idx][input_idx]
            )
            list_of_ctx[device_idx].synchronize()

            # Verify this matches the original input from device input_idx.
            for j in range(length):
                var expected = Scalar[type](input_idx * 1000 + j)
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
                    raise e

            host_output.free()

    # Clean up.
    for i in range(ngpus):
        host_buffers[i].free()


def main() -> None:
    # Test configurations.
    alias test_lengths = (
        List[Int](8 * 1024, 8 * 1024),
        List[Int](128 * 1024, 8 * 1024),
        List[Int](8 * 1024, 256 * 1024),
        List[Int](8 * 1024, 8 * 1024, 8 * 1024, 8 * 1024),
        List[Int](128 * 1024, 256 * 1024, 8 * 1024, 64 * 1024),
        # Test uneven shapes.
        List[Int](37919, 37919, 37918, 37918),
        # Simple uneven case.
        List[Int](4, 3, 3),
        # Another uneven case with 2 GPUs.
        List[Int](1025, 1024),
    )

    @parameter
    for test_idx in range(len(test_lengths)):
        alias lengths = test_lengths[test_idx]
        alias num_gpus = len(lengths)

        if DeviceContext.number_of_devices() < num_gpus:
            continue

        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        print("  Testing configuration:", test_idx, "with", num_gpus, "GPUs")
        all_gather_test[DType.bfloat16, rank=1, ngpus=num_gpus](ctx, lengths)
