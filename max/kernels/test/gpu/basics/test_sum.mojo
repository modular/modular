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

from std.math import ceildiv

from std.gpu import global_idx
from std.gpu.primitives import block, warp
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext
from std.testing import assert_equal

comptime dtype = DType.uint64


def warp_sum_kernel[
    dtype: DType,
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = warp.sum(input[tid])


def test_warp_sum(ctx: DeviceContext) raises:
    comptime size = WARP_SIZE
    comptime BLOCK_SIZE = WARP_SIZE

    # Allocate and initialize host memory
    var in_host = List(range(UInt64(0), UInt64(size)))
    var out_host = List(length=size, fill=Scalar[dtype](0))

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    comptime kernel = warp_sum_kernel[dtype=dtype]
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
        var expected: Scalar[dtype] = UInt64(size * (size - 1) // 2)

        assert_equal(
            out_host[i],
            expected,
            msg=String(t"out_host[{i}] = {out_host[i]} expected = {expected}"),
        )
    _ = out_host^
    _ = in_host^


def block_sum_kernel[
    dtype: DType,
    block_size: Int,
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = block.sum[block_size=block_size, broadcast=True](input[tid])


def test_block_sum(ctx: DeviceContext) raises:
    # Initialize a block with several warps. The sum for each warp is tested
    # above.
    comptime BLOCK_SIZE = WARP_SIZE * 2
    comptime size = BLOCK_SIZE

    # Allocate and initialize host memory
    var in_host = List(range(UInt64(0), UInt64(size)))
    var out_host = List(length=size, fill=Scalar[dtype](0))

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    comptime kernel = block_sum_kernel[dtype=dtype, block_size=BLOCK_SIZE]
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
        var expected: Scalar[dtype] = UInt64(size * (size - 1) // 2)

        assert_equal(
            out_host[i],
            expected,
            msg=String(t"out_host[{i}] = {out_host[i]} expected = {expected}"),
        )
    _ = out_host^
    _ = in_host^


def main() raises:
    with DeviceContext() as ctx:
        test_warp_sum(ctx)

        test_block_sum(ctx)
