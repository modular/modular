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

from gpu import warp, block, global_idx
from gpu.host import DeviceContext
from gpu.globals import WARP_SIZE
from math import ceildiv
from testing import assert_equal

alias type = DType.uint64


fn warp_sum_kernel[
    type: DType,
](
    output: UnsafePointer[Scalar[type]],
    input: UnsafePointer[Scalar[type]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = warp.sum(input[tid])


def test_warp_sum(ctx: DeviceContext):
    alias size = WARP_SIZE
    alias BLOCK_SIZE = WARP_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[type]].alloc(size)
    var out_host = UnsafePointer[Scalar[type]].alloc(size)

    for i in range(size):
        in_host[i] = i
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[type](size)
    var out_device = ctx.enqueue_create_buffer[type](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    ctx.enqueue_function[warp_sum_kernel[type=type]](
        out_device.unsafe_ptr(),
        in_device.unsafe_ptr(),
        size,
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    for i in range(size):
        var expected: Scalar[type] = size * (size - 1) // 2

        assert_equal(
            out_host[i],
            expected,
            msg=String(
                "out_host[", i, "] = ", out_host[i], " expected = ", expected
            ),
        )

    # Cleanup
    in_host.free()
    out_host.free()


fn block_sum_kernel[
    type: DType,
    block_size: Int,
](
    output: UnsafePointer[Scalar[type]],
    input: UnsafePointer[Scalar[type]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = block.sum[block_size=block_size, broadcast=True](input[tid])


def test_block_sum(ctx: DeviceContext):
    # Initialize a block with several warps. The sum for each warp is tested
    # above.
    alias BLOCK_SIZE = WARP_SIZE * 2
    alias size = BLOCK_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[type]].alloc(size)
    var out_host = UnsafePointer[Scalar[type]].alloc(size)

    for i in range(size):
        in_host[i] = i
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[type](size)
    var out_device = ctx.enqueue_create_buffer[type](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    ctx.enqueue_function[block_sum_kernel[type=type, block_size=BLOCK_SIZE]](
        out_device.unsafe_ptr(),
        in_device.unsafe_ptr(),
        size,
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    for i in range(size):
        var expected: Scalar[type] = size * (size - 1) // 2

        assert_equal(
            out_host[i],
            expected,
            msg=String(
                "out_host[", i, "] = ", out_host[i], " expected = ", expected
            ),
        )

    # Cleanup
    in_host.free()
    out_host.free()


def main():
    with DeviceContext() as ctx:
        test_warp_sum(ctx)

        test_block_sum(ctx)
