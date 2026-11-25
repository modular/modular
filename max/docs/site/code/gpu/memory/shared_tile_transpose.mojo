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
# DOC: gpu/memory.mdx

"""Transposes a matrix twice, once straight from global memory and once
through a shared-memory tile, and checks both results against the CPU.

The two kernels show the three memory spaces a kernel names: arguments in
global memory, a tile in shared memory, and locals in registers. The staged
kernel reads and writes global memory along rows, so neighboring threads touch
neighboring addresses on both sides of the transpose.
"""

from std.sys import has_accelerator

from layout import ImmTileTensor, MutTileTensor, TileTensor
from layout import row_major, stack_allocation
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

comptime dtype = DType.float32
comptime SIZE = 1024
comptime TILE = 32
# Each block has TILE x ROWS_PER_PASS threads, and each thread moves
# TILE / ROWS_PER_PASS elements, so the block stays small on every target.
comptime ROWS_PER_PASS = 8
comptime matrix_layout = row_major[SIZE, SIZE]()


def naive_transpose(
    input: ImmTileTensor[dtype, type_of(matrix_layout), ImmUntrackedOrigin],
    output: MutTileTensor[dtype, type_of(matrix_layout), MutUntrackedOrigin],
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    var col = block_idx.x * TILE + thread_idx.x
    var row = block_idx.y * TILE + thread_idx.y
    comptime for i in range(0, TILE, ROWS_PER_PASS):
        # The read walks a row, but the write walks a column: consecutive
        # threads store SIZE elements apart.
        output[col, row + i] = input[row + i, col]


def staged_transpose(
    input: ImmTileTensor[dtype, type_of(matrix_layout), ImmUntrackedOrigin],
    output: MutTileTensor[dtype, type_of(matrix_layout), MutUntrackedOrigin],
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    # One extra element per row shifts each row of the tile by one bank, which
    # spreads the column reads in the second loop across banks.
    var tile = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE, TILE + 1]()
    )
    comptime assert tile.flat_rank == 2

    var tx = thread_idx.x
    var ty = thread_idx.y
    var in_col = block_idx.x * TILE + tx
    var in_row = block_idx.y * TILE + ty
    comptime for i in range(0, TILE, ROWS_PER_PASS):
        tile[ty + i, tx] = input[in_row + i, in_col]

    # Every thread reads values that other threads wrote.
    barrier()

    # Swap the block coordinates so the write also walks a row of `output`.
    var out_col = block_idx.y * TILE + tx
    var out_row = block_idx.x * TILE + ty
    comptime for i in range(0, TILE, ROWS_PER_PASS):
        output[out_row + i, out_col] = tile[tx, ty + i]


def check_transpose(
    input: ImmTileTensor[dtype, type_of(matrix_layout), _],
    output: ImmTileTensor[dtype, type_of(matrix_layout), _],
    name: String,
) raises:
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    for row in range(SIZE):
        for col in range(SIZE):
            var got = output[col, row]
            var expected = input[row, col]
            if got != expected:
                raise Error(
                    t"{name}: output[{col}, {row}] is {got}, expected"
                    t" {expected}"
                )
    print(name, "matches the CPU transpose")


def main() raises:
    comptime assert has_accelerator(), "This example requires a GPU"
    comptime assert SIZE % TILE == 0, "SIZE must be a multiple of TILE"
    comptime assert TILE % ROWS_PER_PASS == 0

    var ctx = DeviceContext()
    var host_in = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
    var dev_in = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    var dev_out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    ctx.synchronize()
    for i in range(SIZE * SIZE):
        host_in[i] = Float32(i)
    ctx.enqueue_copy(dev_in, host_in)

    var input = TileTensor(dev_in, matrix_layout)
    var output = TileTensor(dev_out, matrix_layout)
    var host_input = TileTensor(host_in, matrix_layout)

    comptime grid = (SIZE // TILE, SIZE // TILE)
    comptime block = (TILE, ROWS_PER_PASS)

    ctx.enqueue_memset(dev_out, 0.0)
    ctx.enqueue_function[naive_transpose](
        input.as_imm(), output, grid_dim=grid, block_dim=block
    )
    with dev_out.map_to_host() as host_out:
        check_transpose(
            host_input, TileTensor(host_out, matrix_layout), "naive_transpose"
        )

    ctx.enqueue_memset(dev_out, 0.0)
    ctx.enqueue_function[staged_transpose](
        input.as_imm(), output, grid_dim=grid, block_dim=block
    )
    with dev_out.map_to_host() as host_out:
        check_transpose(
            host_input, TileTensor(host_out, matrix_layout), "staged_transpose"
        )
    # `map_to_host()` synchronized the context, so the kernels are done with
    # `dev_in`. Using it here keeps it alive until then.
    _ = dev_in^
