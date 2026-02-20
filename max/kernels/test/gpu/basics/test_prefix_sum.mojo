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

from math import ceildiv

from gpu import global_idx
from gpu.primitives import block, warp
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from testing import assert_equal

comptime dtype = DType.uint64


fn warp_prefix_sum_kernel[
    dtype: DType,
    exclusive: Bool,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= UInt(size):
        return
    output[tid] = warp.prefix_sum[exclusive=exclusive](input[tid])


def test_warp_prefix_sum[exclusive: Bool](ctx: DeviceContext):
    comptime size = WARP_SIZE
    comptime BLOCK_SIZE = WARP_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[dtype]].alloc(size)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(size)

    for i in range(size):
        in_host[i] = UInt64(i)
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    comptime kernel = warp_prefix_sum_kernel[dtype=dtype, exclusive=exclusive]
    ctx.enqueue_function_experimental[kernel](
        out_device,
        in_device,
        size,
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    for i in range(size):
        var expected: Scalar[dtype]

        comptime if exclusive:
            expected = UInt64(i * (i - 1) // 2)
        else:
            expected = UInt64(i * (i + 1) // 2)

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


fn block_prefix_sum_kernel[
    dtype: DType,
    block_size: Int,
    exclusive: Bool,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= UInt(size):
        return
    output[tid] = block.prefix_sum[exclusive=exclusive, block_size=block_size](
        input[tid]
    )


def test_block_prefix_sum[exclusive: Bool](ctx: DeviceContext):
    # Initialize a block with several warps. The prefix sum for each warp is
    # tested above.
    comptime BLOCK_SIZE = WARP_SIZE * 13
    comptime size = BLOCK_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[dtype]].alloc(size)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(size)

    for i in range(size):
        in_host[i] = UInt64(i)
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    comptime kernel = block_prefix_sum_kernel[
        dtype=dtype, block_size=BLOCK_SIZE, exclusive=exclusive
    ]
    ctx.enqueue_function_experimental[kernel](
        out_device,
        in_device,
        size,
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    for i in range(size):
        var expected: Scalar[dtype]

        comptime if exclusive:
            expected = UInt64(i * (i - 1) // 2)
        else:
            expected = UInt64(i * (i + 1) // 2)

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
        test_warp_prefix_sum[exclusive=True](ctx)
        test_warp_prefix_sum[exclusive=False](ctx)

        test_block_prefix_sum[exclusive=True](ctx)
        test_block_prefix_sum[exclusive=False](ctx)
