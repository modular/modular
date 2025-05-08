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

from gpu import warp, global_idx
from memory import UnsafePointer, memset
from gpu.host import DeviceContext
from math import ceildiv
from testing import assert_equal


fn prefix_sum[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = warp.prefix_sum[exclusive=True](input[tid])


def test_prefix_sum(ctx: DeviceContext):
    alias dtype = DType.uint64
    alias size = 32
    alias BLOCK_SIZE = 32

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
    ctx.enqueue_function[prefix_sum[dtype=dtype]](
        out_device.unsafe_ptr(),
        in_device.unsafe_ptr(),
        size,
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var expected = Scalar[dtype](0)
    for i in range(size):
        assert_equal(
            out_host[i],
            expected,
            msg=String(
                "out_host[", i, "] = ", out_host[i], " expected = ", expected
            ),
        )
        expected += in_host[i]

    # Cleanup
    in_host.free()
    out_host.free()


def main():
    with DeviceContext() as ctx:
        test_prefix_sum(ctx)
