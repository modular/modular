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

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext


fn vec_func[
    op: fn (Float32, Float32) capturing [_] -> Float32
](
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = op(in0[tid], in1[tid])


@no_inline
fn run_captured_func(ctx: DeviceContext, captured: Float32) raises:
    print("-")
    print("run_captured_func(", captured, "):")

    alias length = 1024

    var in0 = ctx.enqueue_create_buffer[DType.float32](length)
    var in1 = ctx.enqueue_create_buffer[DType.float32](length).enqueue_fill(2)
    var out = ctx.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i

    @parameter
    fn add_with_captured(left: Float32, right: Float32) -> Float32:
        return left + right + captured

    var block_dim = 32

    alias kernel = vec_func[add_with_captured]
    # TODO(MAXPLAT-335): Make compile_function_experimental support this case.
    var kernel_func = ctx.compile_function_checked[kernel, kernel]()
    ctx.enqueue_function_checked(
        kernel_func,
        in0,
        in1,
        out,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 2 + captured,
                String("at index ", i, " the value is ", out_host[i]),
            )


fn main() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_capture(" + ctx.name() + "):")

    run_captured_func(ctx, 2.5)
    run_captured_func(ctx, -1.5)

    print("Done.")
