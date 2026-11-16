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

import extensibility

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.atomic import Atomic

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from std.memory import unsafe_stack_allocation

from extensibility import Input, Output, Tensor, TileTensorable
from layout import TileTensor

from std.utils import StaticTuple
from std.utils.coord import Coord

comptime bin_width = Int(UInt8.MAX)


def _histogram_cpu(
    output: TileTensor[mut=True, .int64, ...], input: TileTensor[.uint8, ...]
):
    for i in range(Int(output.dim[0]())):
        output.store[width=1](Coord(i), 0)

    for i in range(Int(input.dim[0]())):
        var bin = Coord(Int(input.load[width=1](Coord(i))))
        output.store[width=1](bin, output.load[width=1](bin) + 1)


def _histogram_gpu(
    output: TileTensor[mut=True, .int64, ...],
    input: TileTensor[.uint8, ...],
    ctx_ptr: DeviceContext,
) raises:
    comptime bin_width = Int(UInt8.MAX) + 1
    comptime block_dim = bin_width

    # Set the maximum number of threads per block to the block dimension.
    # This is equivalent to the `__launch_bounds__` attribute in CUDA.
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_dim))
    )
    def kernel(
        output: Pointer[Int64, MutUntrackedOrigin],
        input: Pointer[UInt8, ImmUntrackedOrigin],
        n_dev: Int32,
    ):
        var n = Int(n_dev)
        var tid = global_idx.x

        if tid >= n:
            return

        # Allocate shared memory for the histogram
        var shared_mem = unsafe_stack_allocation[
            bin_width, Int64, address_space=.SHARED
        ]()

        # Initialize the shared memory to 0
        shared_mem[unsafe_offset=thread_idx.x] = 0

        # Synchronize all threads to ensure that the shared memory is initialized
        barrier()

        # Increment the shared memory for the current thread
        _ = Atomic.fetch_add(
            shared_mem.unsafe_offset(Int(input[unsafe_offset=tid])), 1
        )

        # Synchronize all threads to ensure that the shared memory is updated
        barrier()

        # Increment the output for the current thread
        _ = Atomic.fetch_add(
            output.unsafe_offset(thread_idx.x),
            shared_mem[unsafe_offset=thread_idx.x],
        )

    var n = Int(input.dim[0]())

    var grid_dim = ceildiv(n, block_dim)

    var ctx = ctx_ptr

    # Zero initialize the output buffer
    ctx.enqueue_memset(
        DeviceBuffer(
            ctx, output.unsafe_ptr(), Int(output.dim[0]()), owning=False
        ),
        0,
    )

    ctx.enqueue_function[kernel](
        output.unsafe_ptr(),
        input.unsafe_ptr(),
        Int32(n),
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@extensibility.register("histogram")
struct Histogram:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert type_of(output).rank == 1
        comptime assert type_of(input).rank == 1
        var output_tile = output.to_tile_tensor[.int64]()
        var input_tile = input.to_tile_tensor[.uint8]().as_imm()

        comptime if ctx.target.is_cpu():
            _histogram_cpu(output_tile, input_tile)
        else:
            _histogram_gpu(output_tile, input_tile, ctx)
