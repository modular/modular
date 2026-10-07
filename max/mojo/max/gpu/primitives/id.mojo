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

It defines aliases and functions for accessing GPU grid, block, and thread
indices and dimensions.
"""

from std.math.uutils import ufloordiv
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget, is_amd_gpu, is_nvidia_gpu
from std.sys.intrinsics import readfirstlane

from max.gpu.globals import WARP_SIZE

from . import warp


@__doc_inline
from std._gpu.primitives.id import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    cluster_dim,
    cluster_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)


# ===-----------------------------------------------------------------------===#
# warp_id
# ===-----------------------------------------------------------------------===#


@inline(.nodebug)
def warp_id[*, broadcast: Bool = False]() -> Int:
    """Returns the warp ID of the current thread within its block.
    The warp ID is a unique identifier for each warp within a block, ranging
    from 0 to BLOCK_SIZE/WARP_SIZE-1. This ID is commonly used for warp-level
    programming and synchronization within a block.

    Parameters:
        broadcast: If true, broadcasts the warp ID to all threads in the warp,
                   ensuring that all threads in the same warp have the same
                   value. This can be useful for certain warp-level algorithms.

    Returns:
        The warp ID (0 to BLOCK_SIZE/WARP_SIZE-1) of the current thread.
    """
    return _warp_id[broadcast=broadcast]()


@inline(.nodebug)
def _warp_id[
    *,
    broadcast: Bool = False,
]() -> Int:
    var res = ufloordiv(thread_idx.x, WARP_SIZE)
    comptime if broadcast:
        comptime if is_amd_gpu():
            res = readfirstlane(res)
        else:
            res = warp.broadcast(res)
    return Int(res)


# ===-----------------------------------------------------------------------===#
# sm_id
# ===-----------------------------------------------------------------------===#


@inline(.nodebug)
def sm_id() -> Int:
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
        return warp.broadcast(
            Int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="sm_id() is only supported when targeting NVIDIA GPUs.",
        ]()
