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

import gpu.warp as warp
from gpu import barrier, global_idx
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext
from testing import assert_equal


fn kernel[
    dtype: DType
](
    input: UnsafePointer[Scalar[dtype]],
    output: UnsafePointer[Scalar[dtype]],
    shared_data: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var global_tid = global_idx.x
    if global_tid >= UInt(size):
        return
    shared_data[global_tid] = input[global_tid]

    barrier()

    var result = shared_data[global_tid] + shared_data[0]

    output[global_tid] = result


fn test_barrier[dtype: DType](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size
    alias constant_add: Scalar[dtype] = 42
    var input_host = UnsafePointer[Scalar[dtype]].alloc(buffer_size)
    var output_host = UnsafePointer[Scalar[dtype]].alloc(buffer_size)
    var shared_host = UnsafePointer[Scalar[dtype]].alloc(buffer_size)

    for i in range(buffer_size):
        input_host[i] = i + constant_add
        output_host[i] = -1.0
        shared_host[i] = -999.0

    var input_buffer = ctx.enqueue_create_buffer[dtype](buffer_size)
    var output_buffer = ctx.enqueue_create_buffer[dtype](buffer_size)
    var shared_buffer = ctx.enqueue_create_buffer[dtype](buffer_size)

    ctx.enqueue_copy(input_buffer, input_host)
    ctx.enqueue_copy(output_buffer, output_host)
    ctx.enqueue_copy(shared_buffer, shared_host)

    ctx.enqueue_function_checked[kernel[dtype], kernel[dtype]](
        input_buffer,
        output_buffer,
        shared_buffer,
        buffer_size,
        grid_dim=1,
        block_dim=block_size,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.enqueue_copy(shared_host, shared_buffer)
    ctx.synchronize()

    for i in range(buffer_size):
        assert_equal(output_host[i], 2 * constant_add + i)
        assert_equal(shared_host[i], constant_add + i)

    _ = input_buffer
    _ = shared_buffer

    input_host.free()
    output_host.free()
    shared_host.free()


fn main() raises:
    with DeviceContext() as ctx:
        test_barrier[DType.float32](ctx)
