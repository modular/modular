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

from gpu import *
from gpu.host import DeviceBuffer, DeviceContext, DeviceFunction
from memory import UnsafePointer
from testing import assert_equal


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


fn test(ctx: DeviceContext) raises:
    alias length = 1024

    # Allocate the input buffers as sub buffers of a bigger one
    var in_host = UnsafePointer[Float32].alloc(2 * length)
    var out_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in_host[i] = i
        in_host[i + length] = 2

    var in_device = ctx.enqueue_create_buffer[DType.float32](2 * length)
    var in0_device = in_device.create_sub_buffer[DType.float32](0, length)
    var in1_device = in_device.create_sub_buffer[DType.float32](length, length)

    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    ctx.enqueue_copy(in_device, in_host)

    var block_dim = 32
    var supplement = 5

    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        length,
        supplement,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Make sure our main input device tensor doesn't disappear
    _ = in_device

    ctx.enqueue_copy(out_host, out_device)

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

    in_host.free()
    out_host.free()


def main():
    with DeviceContext() as ctx:
        test(ctx)
