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

# COM: Note: CPU function compilation not supported
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext

alias T = DType.float32
alias S = Scalar[T]


fn vec_func(
    in0: UnsafePointer[S],
    in1: UnsafePointer[S],
    out: UnsafePointer[S],
    s: S,
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + s


fn test_function_unchecked(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function_unchecked(" + ctx.name() + "):")

    alias length = 1024
    alias block_dim = 32

    var scalar: S = 2

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    ctx.enqueue_function[vec_func](
        in0,
        in1,
        out,
        scalar + 1,
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
                i + 5,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )

    print("Done.")


fn test_function_checked(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function_checked(" + ctx.name() + "):")

    alias length = 1024
    alias block_dim = 32

    var scalar: S = 2

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    var compiled_vec_func = ctx.compile_function_checked[vec_func, vec_func]()
    ctx.enqueue_function_checked(
        compiled_vec_func,
        in0,
        in1,
        out,
        scalar + 1,
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
                i + 5,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )

    print("Done.")


fn main() raises:
    var ctx = create_test_device_context()
    test_function_unchecked(ctx)
    test_function_checked(ctx)
