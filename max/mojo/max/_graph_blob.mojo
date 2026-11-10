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
"""Provides the primitives a graph blob's kernel calls.

A graph blob runs the device code of several kernels in one launch, on every
core and thread of the device. The graph compiler generates its kernel: each
member runs only on the threads `is_thread_active` accepts for its
`active_threads`, and every thread waits at a `barrier` over the whole launch
between members. Each target implements the two device primitives through its
MAX plugin. On a GPU that doesn't, using them is a compile-time error;
elsewhere, such as in a kernel body elaborated for the host, they abort.
"""

from std.os import abort
from std.sys.info import is_gpu

from max._plugin import CurrentMaxPlugin
from max.gpu.host import Dim

comptime _UNSUPPORTED = "graph blobs are not supported on this target"


# TODO(GEX-4497): Once a member can run on blocks with gaps between them, or on
# some of each block's threads, the encoding depends on the target. Take it
# from the members' configuration type, which knows its target, rather than
# from a lookup of the accelerator at run time.
def active_threads(
    grid_dim: Dim, block_dim: Dim, launch_grid_dim: Dim, launch_block_dim: Dim
) raises -> Int32:
    """Returns what a graph blob's kernel needs to know about a member
    launched on `grid_dim` and `block_dim`, as a kernel argument.

    The host computes it, where the device's MAX plugin doesn't apply. For
    now every target encodes it the same way, because a member runs on the
    first blocks of the launch, as many as its own grid has, and on all
    threads of each.

    Args:
        grid_dim: The grid of the member's own launch configuration.
        block_dim: The block dims of the member's own launch configuration.
        launch_grid_dim: The grid of the blob's launch.
        launch_block_dim: The block dims of the blob's launch.

    Returns:
        The number of blocks the member runs on.

    Raises:
        If the member's grid isn't one-dimensional or has more blocks than the
        launch, or if its block dims aren't the launch's.
    """
    if grid_dim.y() * grid_dim.z() != 1:
        raise Error(
            "an op in a graph blob runs on a one-dimensional grid, but got ",
            grid_dim,
        )
    if grid_dim.x() > launch_grid_dim.x():
        raise Error(
            "an op in a graph blob runs on at most ",
            launch_grid_dim.x(),
            " blocks, but got ",
            grid_dim.x(),
        )
    if (
        block_dim.x() != launch_block_dim.x()
        or block_dim.y() != launch_block_dim.y()
        or block_dim.z() != launch_block_dim.z()
    ):
        raise Error(
            "an op in a graph blob runs every thread of its blocks, ",
            launch_block_dim,
            ", but got ",
            block_dim,
        )
    return Int32(grid_dim.x())


# TODO(GEX-4497): A member runs on the first blocks of the launch, as many as
# its own grid has, with every thread of each, so a block count describes it.
# Pass each member's own thread mask instead once an op needs a core set with
# gaps or only some of a core's threads, as NeoDriver's configurations can
# describe.
@always_inline
def is_thread_active(active: Int32) -> Bool:
    """Returns whether the calling thread runs the member with `active`.

    Args:
        active: The member's `active_threads`: the number of blocks it runs
            on, from the start of the launch's grid.

    Returns:
        Whether the calling thread's block is one the member runs on.
    """
    comptime if CurrentMaxPlugin._handles_graph_blobs:
        return CurrentMaxPlugin.graph_blob_is_thread_active(active)
    else:
        comptime assert not is_gpu(), _UNSUPPORTED
        abort(_UNSUPPORTED)


@always_inline
def barrier():
    """Waits until every thread of every block in the launch reaches it.

    Writes made before the barrier are visible to every thread after it. Every
    thread the launch runs must call it, so it can't sit under a condition
    that some threads skip.
    """
    comptime if CurrentMaxPlugin._handles_graph_blobs:
        CurrentMaxPlugin.graph_blob_barrier()
    else:
        comptime assert not is_gpu(), _UNSUPPORTED
        abort(_UNSUPPORTED)
