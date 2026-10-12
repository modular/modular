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

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def read_dim[grid: Bool, axis: Int](output: MutPointer[Int32, MutAnyOrigin]):
    # Read one axis per kernel: when a kernel reads several axes, their loads
    # can merge into one wide load and hide a wrong load of a single axis.
    if (
        block_idx.x == 0
        and block_idx.y == 0
        and block_idx.z == 0
        and thread_idx.x == 0
        and thread_idx.y == 0
        and thread_idx.z == 0
    ):
        comptime if grid:
            comptime if axis == 0:
                output[] = Int32(grid_dim.x)
            elif axis == 1:
                output[] = Int32(grid_dim.y)
            else:
                output[] = Int32(grid_dim.z)
        else:
            comptime if axis == 0:
                output[] = Int32(block_dim.x)
            elif axis == 1:
                output[] = Int32(block_dim.y)
            else:
                output[] = Int32(block_dim.z)


def check_axis[
    grid: Bool, axis: Int
](ctx: DeviceContext, expected: Int32) raises:
    var output_host = ctx.enqueue_create_host_buffer[.int32](1)
    var output_buffer = ctx.enqueue_create_buffer[.int32](1)
    output_buffer.enqueue_fill(-1)

    # Distinct sizes for every axis of the grid and the block, so reading the
    # wrong axis cannot pass.
    ctx.enqueue_function[read_dim[grid, axis]](
        output_buffer,
        grid_dim=(6, 5, 3),
        block_dim=(8, 4, 2),
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()
    assert_equal(output_host[0], expected)


def test_block_dim(ctx: DeviceContext) raises:
    check_axis[False, 0](ctx, 8)
    check_axis[False, 1](ctx, 4)
    check_axis[False, 2](ctx, 2)


def test_grid_dim(ctx: DeviceContext) raises:
    check_axis[True, 0](ctx, 6)
    check_axis[True, 1](ctx, 5)
    check_axis[True, 2](ctx, 3)


def main() raises:
    with DeviceContext() as ctx:
        test_block_dim(ctx)
        test_grid_dim(ctx)
