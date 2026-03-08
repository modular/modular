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
"""GPU implementation of elementwise functions."""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from std.gpu.primitives.grid_controls import (
    pdl_launch_attributes,
)  # @doc_private
from std.gpu.host import DeviceContext
from std.gpu.host.info import B200
from std.math import ceildiv, clamp
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import _get_start_indices_of_nth_subvolume_uint


# ===-----------------------------------------------------------------------===#
# Elementwise GPU implementation
# ===-----------------------------------------------------------------------===#


@always_inline
fn _elementwise_impl_gpu[
    rank: Int,
    //,
    func: fn[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: UInt,
](shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func[width, rank](indices)` as sub-tasks for a suitable
    combination of width and indices so as to cover shape on the GPU.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.

    Args:
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    # optimized implementation inspired by https://archive.md/Tye9y#selection-1101.2-1151.3

    comptime hw_info = ctx.default_device_info

    comptime registers_per_thread = 255
    comptime num_waves = 32
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count = UInt(hw_info.sm_count)
    comptime threads_per_multiprocessor = UInt(
        hw_info.threads_per_multiprocessor
    )

    comptime assert (
        sm_count > 0 and threads_per_multiprocessor > 0
    ), "the sm_count and thread_count must be known"

    # split between packed and tail regions of input
    var length = UInt(shape.flattened_length())
    var num_packed_elems = length // simd_width
    var unpacked_tail_length = length % simd_width
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    comptime block_size_unrounded = registers_per_block // registers_per_thread

    # when testing other elementwise kernels, they appear to also use 128 as the block size on blackwell specifically
    comptime block_size = 128 if ctx.default_device_info == B200 else block_size_unrounded - (
        block_size_unrounded % 2
    )

    var num_blocks = clamp(
        ceildiv(num_packed_elems, UInt(block_size)),
        1,
        sm_count * threads_per_multiprocessor // UInt(block_size) * num_waves,
    )

    @__copy_capture(
        num_packed_elems, unpacked_tail_length, packed_region_length
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
    )
    fn _elementwise_gpu_kernel[*, block_size: UInt, handle_uneven_simd: Bool]():
        # process the packed region
        var tid = thread_idx.x + block_size * block_idx.x

        comptime if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
            launch_dependent_grids()

        comptime if PDLLevel() > PDLLevel.OFF:
            wait_on_dependent_grids()

        for idx in range(
            tid,
            num_packed_elems,
            block_size * grid_dim.x,
        ):
            var start_indices = _get_start_indices_of_nth_subvolume_uint[0](
                idx * simd_width, shape
            )

            comptime if handle_uneven_simd:
                if start_indices[rank - 1] + Int(simd_width) >= shape[rank - 1]:
                    comptime for off in range(Int(simd_width)):
                        func[1, rank](
                            _get_start_indices_of_nth_subvolume_uint[0](
                                idx * simd_width + UInt(off),
                                shape,
                            ).canonicalize()
                        )
                else:
                    func[Int(simd_width), rank](start_indices.canonicalize())
            else:
                # The alignment is by number of elements, which will be converted to
                # number of bytes by graph compiler.
                func[Int(simd_width), rank, Int(simd_width)](
                    start_indices.canonicalize()
                )

        # process the tail region
        if tid < unpacked_tail_length:
            var index_tup = _get_start_indices_of_nth_subvolume_uint[0](
                packed_region_length + tid, shape
            ).canonicalize()
            func[1, rank](index_tup)

        comptime if PDLLevel() == PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

    if shape[rank - 1] % Int(simd_width) == 0:
        comptime kernel = _elementwise_gpu_kernel[
            block_size=UInt(block_size), handle_uneven_simd=False
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_blocks),
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )
    else:
        comptime kernel = _elementwise_gpu_kernel[
            block_size=UInt(block_size), handle_uneven_simd=True
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_blocks),
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )
