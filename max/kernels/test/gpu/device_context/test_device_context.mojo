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

from math import iota

from gpu import *
from gpu.host import DeviceBuffer, DeviceContext, DeviceFunction
from testing import assert_equal


# A Simple Kernel performing the sum of two arrays
fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
    supplement: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = in0[tid] + in1[tid] + supplement


def test_is_compatible(ctx: DeviceContext):
    assert_equal(ctx.is_compatible(), True)


fn test_basic(ctx: DeviceContext) raises:
    alias length = 1024

    # Host memory buffers for input and output data
    var in0_host = UnsafePointer[Float32].alloc(length)
    var in1_host = UnsafePointer[Float32].alloc(length)
    var out_host = UnsafePointer[Float32].alloc(length)

    # Initialize inputs
    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    # Device memory buffers for the kernel input and output
    var in0_device = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    # Copy the input data from the Host to the Device memory
    ctx.enqueue_copy(in0_device, in0_host)
    ctx.enqueue_copy(in1_device, in1_host)

    var block_dim = 32
    var supplement = 5

    # Execute the kernel on the device.
    #  - notice the simple function call like invocation
    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        length,
        supplement,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Copy the results back from the device to the host
    ctx.enqueue_copy(out_host, out_device)

    # Wait for the computation to be completed
    ctx.synchronize()

    var expected: List[Float32] = [
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    for i in range(10):
        print("at index", i, "the value is", out_host[i])
        assert_equal(out_host[i], expected[i])

    # Release the Host buffers
    in0_host.free()
    in1_host.free()
    out_host.free()


def test_move(ctx: DeviceContext):
    var b = ctx
    var c = b^
    c.synchronize()


def test_id(ctx: DeviceContext):
    # CPU always gets id 0 so test for that.
    assert_equal(ctx.id(), 0)


def test_print(ctx: DeviceContext):
    alias size = 15

    var host_buffer = ctx.enqueue_create_host_buffer[DType.uint16](size)
    ctx.synchronize()

    iota(host_buffer.unsafe_ptr(), size)

    var expected_host = (
        "HostBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])"
    )
    assert_equal(String(host_buffer), expected_host)

    var dev_buffer = ctx.enqueue_create_buffer[DType.uint16](size)
    host_buffer.enqueue_copy_to(dev_buffer)
    ctx.synchronize()

    var expected_dev = (
        "DeviceBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])"
    )
    assert_equal(String(dev_buffer), expected_dev)

    alias large_size = 1001
    var large_buffer = ctx.enqueue_create_host_buffer[DType.float32](large_size)
    ctx.synchronize()

    iota(large_buffer.unsafe_ptr(), large_size)

    var expected_large = (
        "HostBuffer([0.0, 1.0, 2.0, ..., 998.0, 999.0, 1000.0])"
    )
    assert_equal(String(large_buffer), expected_large)


def main():
    # Create an instance of the DeviceContext
    with DeviceContext() as ctx:
        # Execute our test with the context
        test_is_compatible(ctx)
        test_basic(ctx)
        test_move(ctx)
        test_id(ctx)
        test_print(ctx)
