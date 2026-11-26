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

"""Moves data from the host to the device and back, and scales it in between.

The data stays in `bfloat16` in host and global memory, which halves the bytes
each copy and each global load moves compared with `float32`. The kernel
widens each value to `float32` in a register, computes there, and narrows the
result when it stores it.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import assert_almost_equal

from layout import ImmTileTensor, MutTileTensor, TileTensor, row_major
from max.gpu import global_idx
from max.gpu.host import DeviceContext

comptime storage_dtype = DType.bfloat16
comptime compute_dtype = DType.float32
comptime SIZE = 4096
comptime BLOCK_SIZE = 256
comptime layout = row_major[SIZE]()
comptime SCALE: Scalar[compute_dtype] = 0.5


def scale(
    input: ImmTileTensor[storage_dtype, type_of(layout), ImmUntrackedOrigin],
    output: MutTileTensor[storage_dtype, type_of(layout), MutUntrackedOrigin],
):
    comptime assert input.flat_rank == 1 and output.flat_rank == 1
    var i = global_idx.x
    if i < SIZE:
        var x = input[i].cast[compute_dtype]()
        output[i] = (x * SCALE).cast[storage_dtype]()


def main() raises:
    comptime assert has_accelerator(), "This example requires a GPU"

    var ctx = DeviceContext()
    var device = ctx.device()
    print("Device:", device.get_name())
    print("Host and device share memory:", device.is_host_unified())
    print("Shared memory per block:", device.max_shared_memory_per_block())

    # Host buffers are pinned, so the device can copy them without staging.
    var host_in = ctx.enqueue_create_host_buffer[storage_dtype](SIZE)
    var host_out = ctx.enqueue_create_host_buffer[storage_dtype](SIZE)
    var dev_in = ctx.enqueue_create_buffer[storage_dtype](SIZE)
    var dev_out = ctx.enqueue_create_buffer[storage_dtype](SIZE)

    # The allocations above are queued; wait for them before the host writes.
    ctx.synchronize()
    for i in range(SIZE):
        host_in[i] = Scalar[storage_dtype](i % 256)

    # Copies and the kernel run in order on the context's stream.
    ctx.enqueue_copy(dev_in, host_in)
    ctx.enqueue_function[scale](
        TileTensor(dev_in, layout).as_imm(),
        TileTensor(dev_out, layout),
        grid_dim=ceildiv(SIZE, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    ctx.enqueue_copy(host_out, dev_out)
    # The host must wait before it reads what the device wrote.
    ctx.synchronize()
    # Mojo destroys a value after its last use. Using the device buffers here
    # keeps them alive until the queued work no longer needs them.
    _ = dev_in^
    _ = dev_out^

    for i in range(SIZE):
        var expected = host_in[i].cast[compute_dtype]() * SCALE
        assert_almost_equal(
            host_out[i].cast[compute_dtype](), expected, rtol=1e-2
        )
    print("All", SIZE, "values match")
