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

"""This module provides GPU thread and block indexing functionality.

It defines aliases and functions for accessing GPU grid, block, thread and cluster
dimensions and indices. These are essential primitives for GPU programming that allow
code to determine its position and dimensions within the GPU execution hierarchy.

Most functionality is architecture-agnostic, with some NVIDIA-specific features clearly marked.
The module is designed to work seamlessly across different GPU architectures while providing
optimal performance through hardware-specific optimizations where applicable."""

from std._plugin import CurrentPlugin
from std.sys import llvm_intrinsic
from std.sys.info import (
    CompilationTarget,
    _is_sm_9x_or_newer,
    is_gpu,
)


comptime _GPUIdPluginHookFnType = def() thin -> Int
"""Plugin-hook signature for `PluginHooks.gpu_id_fn`."""


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


# Check that the dimension is either x, y, or z.
# TODO: Some day we should use typed string literals or 'requires' clauses to
#       enforce this at the type system level.
# https://github.com/modular/modular/issues/1278
def _verify_xyz[dim: StaticString]():
    comptime assert (
        dim == "x" or dim == "y" or dim == "z"
    ), "the dimension must be x, y, or z"


comptime _NO_GPU_ID_HOOK_NOTE = (
    "the target's stdlib plugin does not provide `gpu_id_fn`. A GPU target"
    " built without a plugin, such as one named only by its triple, gets the"
    " default plugin; use a target from `get_gpu_target()` instead."
)


# ===-----------------------------------------------------------------------===#
# lane_id
# ===-----------------------------------------------------------------------===#


@inline(.nodebug)
def lane_id() -> Int:
    """Returns the lane ID of the current thread within its warp.

    The lane ID is a unique identifier for each thread within a warp, ranging from 0 to
    WARP_SIZE-1. This ID is commonly used for warp-level programming and thread
    synchronization within a warp.

    Returns:
        The lane ID (0 to WARP_SIZE-1) of the current thread.
    """
    return _lane_id()


@inline(.nodebug)
def _lane_id() -> Int:
    comptime assert is_gpu(), "This function only applies to GPUs."

    comptime if case .Some(f) = CurrentPlugin.gpu_id_fn["lane_id", ""]:
        return f()
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note=_NO_GPU_ID_HOOK_NOTE,
        ]()


# ===-----------------------------------------------------------------------===#
# thread_idx
# ===-----------------------------------------------------------------------===#


struct _ThreadIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a thread within a block.
    """

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    def __getattr_param__[dim: StringLiteral](self) -> Int:
        """Gets the `x`, `y`, or `z` coordinates of a thread within a block.

        Returns:
            The `x`, `y`, or `z` coordinates of a thread within a block.
        """
        _verify_xyz[dim]()
        comptime if case .Some(f) = CurrentPlugin.gpu_id_fn["thread_idx", dim]:
            return f()
        else:
            CompilationTarget.unsupported_target_error[
                operation=String("thread_idx.", dim),
                note=_NO_GPU_ID_HOOK_NOTE,
            ]()


comptime thread_idx = _ThreadIdx()
"""Contains the thread index in the block, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_idx
# ===-----------------------------------------------------------------------===#


struct _BlockIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a block within a grid.
    """

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    def __getattr_param__[dim: StringLiteral](self) -> Int:
        """Gets the `x`, `y`, or `z` coordinates of a block within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a block within a grid.
        """
        _verify_xyz[dim]()
        comptime if case .Some(f) = CurrentPlugin.gpu_id_fn["block_idx", dim]:
            return f()
        else:
            CompilationTarget.unsupported_target_error[
                operation=String("block_idx.", dim),
                note=_NO_GPU_ID_HOOK_NOTE,
            ]()


comptime block_idx = _BlockIdx()
"""Contains the block index in the grid, as `x`, `y`, and `z` values."""

# ===-----------------------------------------------------------------------===#
# block_dim
# ===-----------------------------------------------------------------------===#


struct _BlockDim(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    block."""

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    def __getattr_param__[dim: StaticString](self) -> Int:
        """Gets the `x`, `y`, or `z` dimension of the block.

        Returns:
            The `x`, `y`, or `z` dimension of the block.
        """
        _verify_xyz[dim]()

        comptime if case .Some(f) = CurrentPlugin.gpu_id_fn["block_dim", dim]:
            return f()
        else:
            CompilationTarget.unsupported_target_error[
                operation=String("block_dim.", dim),
                note=_NO_GPU_ID_HOOK_NOTE,
            ]()


comptime block_dim = _BlockDim()
"""Contains the dimensions of the block as `x`, `y`, and `z` values.

For example: `block_dim.y`."""


# ===-----------------------------------------------------------------------===#
# grid_dim
# ===-----------------------------------------------------------------------===#


struct _GridDim(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    grid."""

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    def __getattr_param__[dim: StaticString](self) -> Int:
        """Gets the `x`, `y`, or `z` dimension of the grid.

        Returns:
            The `x`, `y`, or `z` dimension of the grid.
        """
        _verify_xyz[dim]()

        comptime if case .Some(f) = CurrentPlugin.gpu_id_fn["grid_dim", dim]:
            return f()
        else:
            CompilationTarget.unsupported_target_error[
                operation=String("grid_dim.", dim),
                note=_NO_GPU_ID_HOOK_NOTE,
            ]()


comptime grid_dim = _GridDim()
"""Provides accessors for getting the `x`, `y`, and `z`
dimensions of a grid."""


# ===-----------------------------------------------------------------------===#
# global_idx
# ===-----------------------------------------------------------------------===#


struct _GlobalIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` global offset of
    the kernel launch."""

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    def __getattr_param__[dim: StringLiteral](self) -> Int:
        """Gets the `x`, `y`, or `z` dimension of the program.

        Returns:
            The `x`, `y`, or `z` dimension of the program.
        """
        _verify_xyz[dim]()
        var t_idx = thread_idx.__getattr_param__[dim]()
        var b_idx = block_idx.__getattr_param__[dim]()
        var b_dim = block_dim.__getattr_param__[dim]()

        return Int(b_idx * b_dim + t_idx)


comptime global_idx = _GlobalIdx()
"""Contains the global offset of the kernel launch, as `x`, `y`, and `z`
values."""


# ===-----------------------------------------------------------------------===#
# cluster_dim
# ===-----------------------------------------------------------------------===#


struct _ClusterDim(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    cluster."""

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    def __getattr_param__[dim: StaticString](self) -> Int:
        """Gets the `x`, `y`, or `z` dimension of the cluster.

        Returns:
            The `x`, `y`, or `z` dimension of the cluster.
        """
        comptime assert (
            _is_sm_9x_or_newer()
        ), "cluster_id is only supported on NVIDIA SM90+ GPUs"
        _verify_xyz[dim]()

        comptime intrinsic_name = "llvm.nvvm.read.ptx.sreg.cluster.nctaid." + dim
        return Int(
            llvm_intrinsic[intrinsic_name, Int32, has_side_effect=False]()
        )


comptime cluster_dim = _ClusterDim()
"""Contains the dimensions of the cluster, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# cluster_idx
# ===-----------------------------------------------------------------------===#


struct _ClusterIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a cluster within a grid."""

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    @staticmethod
    def _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        return "llvm.nvvm.read.ptx.sreg.clusterid." + dim

    @inline(.nodebug)
    def __getattr_param__[dim: StringLiteral](self) -> Int:
        """Gets the `x`, `y`, or `z` coordinates of a cluster within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a cluster within a grid.
        """
        comptime assert (
            _is_sm_9x_or_newer()
        ), "cluster_id is only supported on NVIDIA SM90+ GPUs"
        _verify_xyz[dim]()
        comptime intrinsic_name = Self._get_intrinsic_name[dim]()
        return Int(
            llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]()
        )


comptime cluster_idx = _ClusterIdx()
"""Contains the cluster index in the grid, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_id_in_cluster
# ===-----------------------------------------------------------------------===#


struct _ClusterBlockIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a threadblock within a cluster."""

    @inline(.nodebug)
    def __init__(out self):
        return

    @inline(.nodebug)
    @staticmethod
    def _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        return "llvm.nvvm.read.ptx.sreg.cluster.ctaid." + dim

    @inline(.nodebug)
    def __getattr_param__[dim: StringLiteral](self) -> Int:
        """Gets the `x`, `y`, or `z` coordinates of a threadblock within a cluster.

        Returns:
            The `x`, `y`, or `z` coordinates of a threadblock within a cluster.
        """
        comptime assert (
            _is_sm_9x_or_newer()
        ), "cluster_id is only supported on NVIDIA SM90+ GPUs"
        _verify_xyz[dim]()
        comptime intrinsic_name = Self._get_intrinsic_name[dim]()
        return Int(
            llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]()
        )


comptime block_id_in_cluster = _ClusterBlockIdx()
"""Contains the block id of the threadblock within a cluster, as `x`, `y`, and `z` values."""
