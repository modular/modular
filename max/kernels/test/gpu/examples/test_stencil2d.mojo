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

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import barrier, block_dim, block_idx, global_idx, thread_idx
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import UnsafePointer, stack_allocation

from utils.index import Index

alias BLOCK_DIM = 4


fn stencil2d(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    arr_size: Int,
    num_rows: Int,
    num_cols: Int,
    coeff0: Int,
    coeff1: Int,
    coeff2: Int,
    coeff3: Int,
    coeff4: Int,
):
    var tidx = global_idx.x
    var tidy = global_idx.y

    var a = NDBuffer[DType.float32, 1](a_ptr, Index(arr_size))
    var b = NDBuffer[DType.float32, 1](b_ptr, Index(arr_size))

    if tidy > 0 and tidx > 0 and tidy < num_rows - 1 and tidx < num_cols - 1:
        b[tidy * num_cols + tidx] = (
            coeff0 * a[tidy * num_cols + tidx - 1]
            + coeff1 * a[tidy * num_cols + tidx]
            + coeff2 * a[tidy * num_cols + tidx + 1]
            + coeff3 * a[(tidy - 1) * num_cols + tidx]
            + coeff4 * a[(tidy + 1) * num_cols + tidx]
        )


fn stencil2d_smem(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    arr_size: Int,
    num_rows: Int,
    num_cols: Int,
    coeff0: Int,
    coeff1: Int,
    coeff2: Int,
    coeff3: Int,
    coeff4: Int,
):
    var tidx = global_idx.x
    var tidy = global_idx.y
    var lindex_x = thread_idx.x + 1
    var lindex_y = thread_idx.y + 1

    var a = NDBuffer[DType.float32, 1](a_ptr, Index(arr_size))
    var b = NDBuffer[DType.float32, 1](b_ptr, Index(arr_size))

    var a_shared = NDBuffer[
        DType.float32,
        2,
        MutableAnyOrigin,
        DimList(BLOCK_DIM + 2, BLOCK_DIM + 2),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Each element is loaded in shared memory.
    a_shared[Index(lindex_y, lindex_x)] = a[tidy * num_cols + tidx]

    # First column also loads elements left and right to the block.
    if thread_idx.x == 0:
        a_shared[Index(lindex_y, 0)] = (
            a[tidy * num_cols + (tidx - 1)] if 0
            <= tidy * num_cols + (tidx - 1)
            < arr_size else 0
        )
        a_shared[Index(Int(lindex_y), BLOCK_DIM + 1)] = (
            a[tidy * num_cols + tidx + BLOCK_DIM] if 0
            <= tidy * num_cols + tidx + BLOCK_DIM
            < arr_size else 0
        )

    # First row also loads elements above and below the block.
    if thread_idx.y == 0:
        a_shared[Index(0, lindex_x)] = (
            a[(tidy - 1) * num_cols + tidx] if 0
            < (tidy - 1) * num_cols + tidx
            < arr_size else 0
        )
        a_shared[Index(BLOCK_DIM + 1, lindex_x)] = (
            a[(tidy + BLOCK_DIM) * num_cols + tidx] if 0
            <= (tidy + BLOCK_DIM) * num_cols + tidx
            < arr_size else 0
        )

    barrier()

    if tidy > 0 and tidx > 0 and tidy < num_rows - 1 and tidx < num_cols - 1:
        b[tidy * num_cols + tidx] = (
            coeff0 * a_shared[Index(lindex_y, lindex_x - 1)]
            + coeff1 * a_shared[Index(lindex_y, lindex_x)]
            + coeff2 * a_shared[Index(lindex_y, lindex_x + 1)]
            + coeff3 * a_shared[Index(lindex_y - 1, lindex_x)]
            + coeff4 * a_shared[Index(lindex_y + 1, lindex_x)]
        )


# CHECK-LABEL: run_stencil2d
fn run_stencil2d[smem: Bool](ctx: DeviceContext) raises:
    print("== run_stencil2d")

    alias m = 64
    alias coeff0 = 3
    alias coeff1 = 2
    alias coeff2 = 4
    alias coeff3 = 1
    alias coeff4 = 5
    alias iterations = 4

    alias num_rows = 8
    alias num_cols = 8

    var a_host = UnsafePointer[Float32].alloc(m)
    var b_host = UnsafePointer[Float32].alloc(m)

    for i in range(m):
        a_host[i] = i
        b_host[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](m)
    var b_device = ctx.enqueue_create_buffer[DType.float32](m)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias func_select = stencil2d_smem if smem == True else stencil2d

    for _ in range(iterations):
        ctx.enqueue_function[func_select](
            a_device,
            b_device,
            m,
            num_rows,
            num_cols,
            coeff0,
            coeff1,
            coeff2,
            coeff3,
            coeff4,
            grid_dim=(
                ceildiv(num_rows, BLOCK_DIM),
                ceildiv(num_cols, BLOCK_DIM),
            ),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

        var tmp_ptr = b_device
        b_device = a_device
        a_device = tmp_ptr

    ctx.enqueue_copy(b_host, b_device)
    ctx.synchronize()

    # CHECK: 37729.0 ,52628.0 ,57021.0 ,60037.0 ,58925.0 ,39597.0 ,
    # CHECK: 57888.0 ,80505.0 ,86322.0 ,89682.0 ,86994.0 ,57818.0 ,
    # CHECK: 76680.0 ,106488.0 ,113400.0 ,116775.0 ,112182.0 ,73933.0 ,
    # CHECK: 95424.0 ,132408.0 ,140400.0 ,143775.0 ,137262.0 ,89925.0 ,
    # CHECK: 91968.0 ,135753.0 ,144450.0 ,147450.0 ,138642.0 ,81842.0 ,
    # CHECK: 50277.0 ,73628.0 ,81985.0 ,83565.0 ,71417.0 ,43229.0 ,
    for i in range(1, num_rows - 1):
        for j in range(1, num_cols - 1):
            print(b_host[i * num_cols + j], ",", end="")
        print()

    _ = a_device
    _ = b_device

    _ = a_host
    _ = b_host


def main():
    with DeviceContext() as ctx:
        run_stencil2d[False](ctx)
        run_stencil2d[True](ctx)
