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

import math
from sys import llvm_intrinsic
from sys.info import (
    CompilationTarget,
    _is_sm_9x_or_newer,
    is_amd_gpu,
    is_apple_gpu,
    is_gpu,
    is_nvidia_gpu,
)
from memory import AddressSpace

from ..globals import WARP_SIZE
from .warp import broadcast


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


# Check that the dimension is either x, y, or z.
# TODO: Some day we should use typed string literals or 'requires' clauses to
#       enforce this at the type system level.
# https://github.com/modular/modular/issues/1278
fn _verify_xyz[dim: StaticString]():
    comptime assert (
        dim == "x" or dim == "y" or dim == "z"
    ), "the dimension must be x, y, or z"


@always_inline
fn _get_gcn_idx[offset: Int, dtype: DType]() -> UInt:
    var ptr = llvm_intrinsic[
        "llvm.amdgcn.implicitarg.ptr",
        UnsafePointer[
            Scalar[dtype],
            MutExternalOrigin,
            address_space = AddressSpace.CONSTANT,
        ],
        has_side_effect=False,
    ]()
    return UInt(ptr.load[alignment=4](offset))


# ===-----------------------------------------------------------------------===#
# lane_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn lane_id() -> UInt:
    """Returns the lane ID of the current thread within its warp.

    The lane ID is a unique identifier for each thread within a warp, ranging from 0 to
    WARP_SIZE-1. This ID is commonly used for warp-level programming and thread
    synchronization within a warp.

    Returns:
        The lane ID (0 to WARP_SIZE-1) of the current thread.
    """
    comptime assert is_gpu(), "This function only applies to GPUs."

    comptime if is_nvidia_gpu():
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.laneid",
                    Int32,
                    has_side_effect=False,
                ]().cast[DType.uint32]()
            )
        )

    elif is_amd_gpu():
        comptime none = Int32(-1)
        comptime zero = Int32(0)
        var t = llvm_intrinsic[
            "llvm.amdgcn.mbcnt.lo", Int32, has_side_effect=False
        ](none, zero)
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.amdgcn.mbcnt.hi", Int32, has_side_effect=False
                ](none, t).cast[DType.uint32]()
            )
        )

    elif is_apple_gpu():
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.air.thread_index_in_simdgroup",
                    Int32,
                    has_side_effect=False,
                ]().cast[DType.uint32]()
            )
        )

    else:
        return CompilationTarget.unsupported_target_error[
            UInt,
            operation = __get_current_function_name(),
        ]()


# ===-----------------------------------------------------------------------===#
# warp_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn warp_id() -> UInt:
    """Returns the warp ID of the current thread within its block.
    The warp ID is a unique identifier for each warp within a block, ranging
    from 0 to BLOCK_SIZE/WARP_SIZE-1. This ID is commonly used for warp-level
    programming and synchronization within a block.

    Returns:
        The warp ID (0 to BLOCK_SIZE/WARP_SIZE-1) of the current thread.
    """

    return thread_idx.x // UInt(WARP_SIZE)


# ===-----------------------------------------------------------------------===#
# sm_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn sm_id() -> UInt:
    """Returns the Streaming Multiprocessor (SM) ID of the current thread.

    The SM ID uniquely identifies which physical streaming multiprocessor the thread is
    executing on. This is useful for SM-level optimizations and understanding hardware
    utilization.

    If called on non-NVIDIA GPUs, this function aborts as this functionality
    is only supported on NVIDIA hardware.

    Returns:
        The SM ID of the current thread.
    """

    comptime if is_nvidia_gpu():
        return broadcast(
            UInt(
                Int(
                    llvm_intrinsic[
                        "llvm.nvvm.read.ptx.sreg.smid",
                        Int32,
                        has_side_effect=False,
                    ]().cast[DType.uint32]()
                )
            )
        )
    else:
        return CompilationTarget.unsupported_target_error[
            UInt,
            operation = __get_current_function_name(),
            note="sm_id() is only supported when targeting NVIDIA GPUs.",
        ]()


# ===-----------------------------------------------------------------------===#
# thread_idx
# ===-----------------------------------------------------------------------===#


struct _ThreadIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a thread within a block."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        comptime if is_nvidia_gpu():
            return "llvm.nvvm.read.ptx.sreg.tid." + dim
        elif is_amd_gpu():
            return "llvm.amdgcn.workitem.id." + dim
        elif is_apple_gpu():
            return "llvm.air.thread_position_in_threadgroup." + dim
        else:
            return CompilationTarget.unsupported_target_error[
                StaticString,
                operation = __get_current_function_name(),
            ]()

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a thread within a block.

        Returns:
            The `x`, `y`, or `z` coordinates of a thread within a block.
        """
        _verify_xyz[dim]()
        comptime intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]()
        )


comptime thread_idx = _ThreadIdx()
"""Contains the thread index in the block, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_idx
# ===-----------------------------------------------------------------------===#


struct _BlockIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a block within a grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        comptime if is_nvidia_gpu():
            return "llvm.nvvm.read.ptx.sreg.ctaid." + dim
        elif is_amd_gpu():
            return "llvm.amdgcn.workgroup.id." + dim
        elif is_apple_gpu():
            return "llvm.air.threadgroup_position_in_grid." + dim
        else:
            return CompilationTarget.unsupported_target_error[
                StaticString,
                operation = __get_current_function_name(),
            ]()

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a block within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a block within a grid.
        """
        _verify_xyz[dim]()
        comptime intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]()
        )


comptime block_idx = _BlockIdx()
"""Contains the block index in the grid, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_dim
# ===-----------------------------------------------------------------------===#


struct _BlockDim(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    block."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StaticString](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the block.

        Returns:
            The `x`, `y`, or `z` dimension of the block.
        """
        _verify_xyz[dim]()

        comptime if is_nvidia_gpu():
            comptime intrinsic_name = "llvm.nvvm.read.ptx.sreg.ntid." + dim
            return UInt(
                Int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
        elif is_apple_gpu():
            return UInt(
                Int(
                    llvm_intrinsic[
                        "llvm.air.threads_per_threadgroup." + dim,
                        Int32,
                        has_side_effect=False,
                    ]()
                )
            )
        elif is_amd_gpu():

            @parameter
            fn _get_offset() -> Int:
                comptime if dim == "x":
                    return 6
                elif dim == "y":
                    return 7
                else:
                    comptime assert dim == "z"
                    return 8

            return _get_gcn_idx[_get_offset(), DType.uint16]()

        else:
            return CompilationTarget.unsupported_target_error[
                UInt,
                operation = __get_current_function_name(),
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

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StaticString](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the grid.

        Returns:
            The `x`, `y`, or `z` dimension of the grid.
        """
        _verify_xyz[dim]()

        comptime if is_nvidia_gpu():
            comptime intrinsic_name = "llvm.nvvm.read.ptx.sreg.nctaid." + dim
            return UInt(
                Int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
        elif is_amd_gpu():

            @parameter
            fn _get_offset() -> Int:
                comptime if dim == "x":
                    return 0
                elif dim == "y":
                    return 1
                else:
                    comptime assert dim == "z"
                    return 2

            return _get_gcn_idx[_get_offset(), DType.uint32]()
        elif is_apple_gpu():
            comptime intrinsic_name = "llvm.air.threads_per_grid." + dim
            var gridDim = UInt(
                Int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
            # Metal passes grid dimension as a gridDim.dim * blockDim.dim.
            # To make things compatible with NVidia and AMDGPU, divide result
            # by block_dim.dim
            return gridDim // block_dim.__getattr__[dim]()
        else:
            return CompilationTarget.unsupported_target_error[
                UInt,
                operation = __get_current_function_name(),
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

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the program.

        Returns:
            The `x`, `y`, or `z` dimension of the program.
        """
        _verify_xyz[dim]()
        var t_idx = thread_idx.__getattr__[dim]()
        var b_idx = block_idx.__getattr__[dim]()
        var b_dim = block_dim.__getattr__[dim]()

        return math.fma(b_idx, b_dim, t_idx)


comptime global_idx = _GlobalIdx()
"""Contains the global offset of the kernel launch, as `x`, `y`, and `z`
values."""


# ===-----------------------------------------------------------------------===#
# cluster_dim
# ===-----------------------------------------------------------------------===#


struct _ClusterDim(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    cluster."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StaticString](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the cluster.

        Returns:
            The `x`, `y`, or `z` dimension of the cluster.
        """
        comptime assert (
            _is_sm_9x_or_newer()
        ), "cluster_id is only supported on NVIDIA SM90+ GPUs"
        _verify_xyz[dim]()

        comptime intrinsic_name = "llvm.nvvm.read.ptx.sreg.cluster.nctaid." + dim
        return UInt(
            Int(llvm_intrinsic[intrinsic_name, Int32, has_side_effect=False]())
        )


comptime cluster_dim = _ClusterDim()
"""Contains the dimensions of the cluster, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# cluster_idx
# ===-----------------------------------------------------------------------===#


struct _ClusterIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a cluster within a grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        return "llvm.nvvm.read.ptx.sreg.clusterid." + dim

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a cluster within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a cluster within a grid.
        """
        comptime assert (
            _is_sm_9x_or_newer()
        ), "cluster_id is only supported on NVIDIA SM90+ GPUs"
        _verify_xyz[dim]()
        comptime intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            Int(llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]())
        )


comptime cluster_idx = _ClusterIdx()
"""Contains the cluster index in the grid, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_id_in_cluster
# ===-----------------------------------------------------------------------===#


struct _ClusterBlockIdx(Defaultable, TrivialRegisterPassable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a threadblock within a cluster."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        return "llvm.nvvm.read.ptx.sreg.cluster.ctaid." + dim

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a threadblock within a cluster.

        Returns:
            The `x`, `y`, or `z` coordinates of a threadblock within a cluster.
        """
        comptime assert (
            _is_sm_9x_or_newer()
        ), "cluster_id is only supported on NVIDIA SM90+ GPUs"
        _verify_xyz[dim]()
        comptime intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            Int(llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]())
        )


comptime block_id_in_cluster = _ClusterBlockIdx()
"""Contains the block id of the threadblock within a cluster, as `x`, `y`, and `z` values."""
