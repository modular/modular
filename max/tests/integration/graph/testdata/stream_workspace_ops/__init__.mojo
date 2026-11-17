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
"""A kernel pair staged through a workspace allocated at run time.

The workspace comes from the op's `DeviceContext`, as a library's scratch buffer
does, so a side-stream op allocates through its stream's context.
"""

from std.math import ceildiv

import extensibility
from extensibility import Input, Output, Tensor, TileTensorable
from max.gpu import global_idx
from max.gpu.host import DeviceContext

comptime BLOCK_THREADS = 256
comptime MIB_FLOATS = (1 << 20) // 4


def _fill(
    workspace: Pointer[Float32, MutUntrackedOrigin],
    input: Pointer[Float32, MutUntrackedOrigin],
    n_dev: Int32,
    total_dev: Int64,
):
    """Tiles the input over the whole workspace."""
    var i = global_idx.x
    if i < Int(total_dev):
        workspace[unsafe_offset=i] = input[unsafe_offset=i % Int(n_dev)]


def _read(
    output: Pointer[Float32, MutUntrackedOrigin],
    workspace: Pointer[Float32, MutUntrackedOrigin],
    n_dev: Int32,
    total_dev: Int64,
):
    """Averages the workspace's first and last copies of the input."""
    var i = global_idx.x
    var n = Int(n_dev)
    if i < n:
        var last = Int(total_dev) - n + i
        output[unsafe_offset=i] = 0.5 * (
            workspace[unsafe_offset=i] + workspace[unsafe_offset=last]
        )


@extensibility.register("stream_workspace")
struct StreamWorkspace:
    """Copies the input to the output through an `mib` MiB device workspace."""

    @staticmethod
    def execute[
        mib: Int
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert type_of(output).dtype == .float32
        comptime assert type_of(input).dtype == .float32
        comptime if not ctx.target.is_gpu():
            raise Error("stream_workspace runs on GPUs only")
        var output_tile = output.to_tile_tensor[.float32]()
        var input_tile = input.to_tile_tensor[.float32]()
        var n = Int(input.shape()[0].value()) * Int(input.shape()[1].value())
        if n == 0:
            return
        var total = max(mib * MIB_FLOATS, n)
        var workspace = ctx.enqueue_create_buffer[.float32](total)
        var ws = workspace.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        ctx.enqueue_function[_fill](
            ws,
            input_tile.ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            Int32(n),
            Int64(total),
            grid_dim=ceildiv(total, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )
        ctx.enqueue_function[_read](
            output_tile.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
            ws,
            Int32(n),
            Int64(total),
            grid_dim=ceildiv(n, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )
        # Freed once the kernels are queued, as a library frees its scratch.
        _ = workspace^
