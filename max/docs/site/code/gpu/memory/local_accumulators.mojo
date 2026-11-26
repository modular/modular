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

"""Sums the columns of a matrix, keeping each thread's partial sums in a small
per-thread array, and checks the result against the CPU.

Each thread sums several columns. Its accumulators are a `LOCAL` allocation,
private to the thread, which the compiler keeps in registers because every
index into it is known at compile time.
"""

from std.sys import has_accelerator

from layout import ImmTileTensor, MutTileTensor, TileTensor
from layout import row_major, stack_allocation
from max.gpu import thread_idx
from max.gpu.host import DeviceContext

comptime dtype = DType.float32
comptime THREADS = 256
comptime COLS_PER_THREAD = 4
comptime ROWS = 512
comptime COLS = THREADS * COLS_PER_THREAD
comptime matrix_layout = row_major[ROWS, COLS]()
comptime sums_layout = row_major[COLS]()


def column_sums(
    input: ImmTileTensor[dtype, type_of(matrix_layout), ImmUntrackedOrigin],
    output: MutTileTensor[dtype, type_of(sums_layout), MutUntrackedOrigin],
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 1
    var acc = stack_allocation[dtype, address_space=.LOCAL](
        row_major[COLS_PER_THREAD]()
    )
    comptime assert acc.flat_rank == 1
    comptime for k in range(COLS_PER_THREAD):
        acc[k] = 0
    for row in range(ROWS):
        comptime for k in range(COLS_PER_THREAD):
            # Neighboring threads read neighboring columns, so loads coalesce.
            acc[k] = acc[k] + input[row, k * THREADS + thread_idx.x]
    comptime for k in range(COLS_PER_THREAD):
        output[k * THREADS + thread_idx.x] = acc[k]


def main() raises:
    comptime assert has_accelerator(), "This example requires a GPU"

    var ctx = DeviceContext()
    var host_in = ctx.enqueue_create_host_buffer[dtype](ROWS * COLS)
    var host_out = ctx.enqueue_create_host_buffer[dtype](COLS)
    var dev_in = ctx.enqueue_create_buffer[dtype](ROWS * COLS)
    var dev_out = ctx.enqueue_create_buffer[dtype](COLS)
    ctx.synchronize()
    for row in range(ROWS):
        for col in range(COLS):
            host_in[row * COLS + col] = Float32(col % 7)
    ctx.enqueue_copy(dev_in, host_in)

    ctx.enqueue_function[column_sums](
        TileTensor(dev_in, matrix_layout).as_imm(),
        TileTensor(dev_out, sums_layout),
        grid_dim=1,
        block_dim=THREADS,
    )
    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()

    for col in range(COLS):
        var expected = Float32(ROWS * (col % 7))
        if host_out[col] != expected:
            raise Error(
                t"sum of column {col} is {host_out[col]}, expected {expected}"
            )
    print("Column sums match the CPU")
    _ = dev_in^
    _ = dev_out^
