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

from math import ceildiv

from gpu import block, global_idx, warp, lane_id
from gpu.host import DeviceContext
from gpu.host import DeviceContext
from gpu.globals import WARP_SIZE
from math import ceildiv
from testing import assert_equal
from sys.intrinsics import lanemask_lt

alias dtype = DType.uint64


fn warp_prefix_sum_kernel[
    dtype: DType,
    exclusive: Bool,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = warp.prefix_sum[exclusive=exclusive](input[tid])


def test_warp_prefix_sum[exclusive: Bool](ctx: DeviceContext):
    alias size = WARP_SIZE
    alias BLOCK_SIZE = WARP_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[dtype]].alloc(size)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(size)

    for i in range(size):
        in_host[i] = i
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    ctx.enqueue_function[
        warp_prefix_sum_kernel[dtype=dtype, exclusive=exclusive]
    ](
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
        var expected: Scalar[dtype]

        @parameter
        if exclusive:
            expected = i * (i - 1) // 2
        else:
            expected = i * (i + 1) // 2

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
    if tid >= size:
        return
    output[tid] = block.prefix_sum[exclusive=exclusive, block_size=block_size](
        input[tid]
    )


def test_block_prefix_sum[exclusive: Bool](ctx: DeviceContext):
    # Initialize a block with several warps. The prefix sum for each warp is
    # tested above.
    alias BLOCK_SIZE = WARP_SIZE * 13
    alias size = BLOCK_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[dtype]].alloc(size)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(size)

    for i in range(size):
        in_host[i] = i
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    ctx.enqueue_function[
        block_prefix_sum_kernel[
            dtype=dtype, block_size=BLOCK_SIZE, exclusive=exclusive
        ]
    ](
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
        var expected: Scalar[dtype]

        @parameter
        if exclusive:
            expected = i * (i - 1) // 2
        else:
            expected = i * (i + 1) // 2

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


fn warp_rank_kernel[
    dtype: DType,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
):
    var tid = lane_id()
    var v = input[tid]
    var condition = v % 4 == 1
    var rank = warp.rank(condition)
    if condition:
      print("tid", tid, "rank", rank, "value", v + 100, "lanemask_lt", lanemask_lt())
      output[rank] = v + 100

def test_warp_rank(ctx: DeviceContext):
    alias size = WARP_SIZE

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[dtype]].alloc(size)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(size)

    for i in range(size):
        in_host[i] = i
        out_host[i] = 0

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(out_device, out_host)

    # Launch kernel
    ctx.enqueue_function[warp_rank_kernel[dtype=dtype]](
        out_device.unsafe_ptr(),
        in_device.unsafe_ptr(),
        block_dim=WARP_SIZE,
        grid_dim=1,
    )

    # Copy results back and verify
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var expected = List[Scalar[dtype]](length=size, fill=0)

    for i in range(size // 4):
      expected[i] = 100 + 1 + i * 4

    for i in range(size):
        assert_equal(
            out_host[i],
            expected[i],
            msg=String(
                "out_host[", i, "] = ", out_host[i], " expected = ", expected[i]
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

        test_warp_rank(ctx)
        test_warp_rank(ctx)
