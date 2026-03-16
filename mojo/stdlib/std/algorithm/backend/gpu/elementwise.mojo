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
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from std.gpu.primitives.cluster import (
    cluster_sync_acquire,
    cluster_sync_release,
    clusterlaunchcontrol_try_cancel,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_query_cancel_get_first_ctaid,
    elect_one_sync_with_mask,
)
from std.gpu.primitives.grid_controls import (
    pdl_launch_attributes,
)  # @doc_hidden
from std.gpu.host import DeviceContext
from std.gpu.host.info import B200
from std.gpu.sync import mbarrier_init, mbarrier_arrive_expect_tx_relaxed
from std.math import ceildiv, clamp
from std.memory import stack_allocation
from std.sys._assembly import inlined_assembly
from std.sys.defines import get_defined_bool
from std.sys.info import _is_sm_100x_or_newer
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import _get_start_indices_of_nth_subvolume_uint


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#

comptime _USE_CLC_WORK_STEALING = get_defined_bool[
    "USE_CLC_WORK_STEALING", True
]()


@always_inline("nodebug")
def _mbarrier_wait_acquire_cta(
    mbar: UnsafePointer[mut=True, Int64, _, address_space=AddressSpace.SHARED],
    phase: UInt32,
):
    """Spin-waits on an mbarrier until the given phase completes, with acquire
    semantics at CTA scope."""
    inlined_assembly[
        """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
        }""",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](Int32(Int(mbar)), phase)


# ===-----------------------------------------------------------------------===#
# Work-stealing elementwise (SM100+ CLC)
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu_clc[
    rank: Int,
    //,
    func: fn[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: UInt,
    block_size: Int,
](shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func` over `shape` on SM100+ GPUs using Cluster Launch Control
    work-stealing.

    Each thread block processes one tile of `block_size` packed elements, then
    attempts to cancel and steal the work of a not-yet-launched block. This
    gives the reduced overhead of persistent kernels with the preemption and
    load-balancing benefits of one-block-per-tile launches.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.

    Args:
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    var length = UInt(shape.flattened_length())
    var num_packed_elems, unpacked_tail_length = divmod(length, simd_width)
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    var num_tiles = ceildiv(num_packed_elems, UInt(block_size))
    if num_tiles == 0:
        num_tiles = 1

    @__copy_capture(
        num_packed_elems, unpacked_tail_length, packed_region_length
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
    )
    def _kernel[*, block_size: UInt, handle_uneven_simd: Bool]():
        var result = stack_allocation[
            1,
            UInt128,
            address_space=AddressSpace.SHARED,
            alignment=16,
        ]()
        var mbar = stack_allocation[
            1,
            Int64,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()

        var tile_id = UInt(block_idx.x)
        var phase: UInt32 = 0

        comptime if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
            launch_dependent_grids()

        comptime if PDLLevel() > PDLLevel.OFF:
            wait_on_dependent_grids()

        # Initialize mbarrier and kick-start CLC pipeline.
        if thread_idx.x == 0:
            mbarrier_init(mbar, Int32(1))
            if elect_one_sync_with_mask(mask=1):
                clusterlaunchcontrol_try_cancel(result, mbar)
            _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

        # Work-stealing loop.
        while True:
            # Process current tile.
            var global_packed_idx = tile_id * UInt(block_size) + UInt(
                thread_idx.x
            )
            if global_packed_idx < num_packed_elems:
                var start_indices = _get_start_indices_of_nth_subvolume_uint[0](
                    global_packed_idx * simd_width, shape
                )

                comptime if handle_uneven_simd:
                    if (
                        start_indices[rank - 1] + Int(simd_width)
                        >= shape[rank - 1]
                    ):
                        comptime for off in range(Int(simd_width)):
                            func[1, rank](
                                _get_start_indices_of_nth_subvolume_uint[0](
                                    global_packed_idx * simd_width + UInt(off),
                                    shape,
                                ).canonicalize()
                            )
                    else:
                        func[Int(simd_width), rank](
                            start_indices.canonicalize()
                        )
                else:
                    func[Int(simd_width), rank, Int(simd_width)](
                        start_indices.canonicalize()
                    )

            # Leader waits for cancel result.
            if thread_idx.x == 0:
                _mbarrier_wait_acquire_cta(mbar, phase)
                phase ^= 1

            # All threads sync — result now visible.
            barrier()

            # Check if we stole another tile.
            if not Bool(clusterlaunchcontrol_query_cancel_is_canceled(result)):
                break

            # Read stolen CTA's block index as next tile.
            tile_id = UInt(
                clusterlaunchcontrol_query_cancel_get_first_ctaid["x"](result)
            )

            # All threads must read result before leader reuses buffer.
            barrier()

            # Leader: fence and issue next cancel request.
            if thread_idx.x == 0:
                cluster_sync_release()
                cluster_sync_acquire()
                if elect_one_sync_with_mask(mask=1):
                    clusterlaunchcontrol_try_cancel(result, mbar)
                _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

        # Tail: only the first block handles remainder elements.
        if UInt(block_idx.x) == 0 and UInt(thread_idx.x) < (
            unpacked_tail_length
        ):
            var index_tup = _get_start_indices_of_nth_subvolume_uint[0](
                packed_region_length + UInt(thread_idx.x), shape
            ).canonicalize()
            func[1, rank](index_tup)

        comptime if PDLLevel() == PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

    if shape[rank - 1] % Int(simd_width) == 0:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=False
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_tiles),
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )
    else:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=True
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_tiles),
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )


# ===-----------------------------------------------------------------------===#
# Grid-stride elementwise (pre-SM100)
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu_grid_stride[
    rank: Int,
    //,
    func: fn[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: UInt,
    block_size: Int,
    num_waves: Int,
    sm_count: UInt,
    threads_per_multiprocessor: UInt,
](shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func` over `shape` using a grid-stride loop.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        num_waves: The number of waves to saturate SMs.
        sm_count: The number of streaming multiprocessors.
        threads_per_multiprocessor: The number of threads per SM.

    Args:
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    # optimized implementation inspired by https://archive.md/Tye9y#selection-1101.2-1151.3

    var length = UInt(shape.flattened_length())
    var num_packed_elems, unpacked_tail_length = divmod(length, simd_width)
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    var num_blocks = clamp(
        ceildiv(num_packed_elems, UInt(block_size)),
        1,
        sm_count
        * threads_per_multiprocessor
        // UInt(block_size)
        * UInt(num_waves),
    )

    @__copy_capture(
        num_packed_elems, unpacked_tail_length, packed_region_length
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
    )
    def _kernel[*, block_size: UInt, handle_uneven_simd: Bool]():
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
                # The alignment is by number of elements, which will be
                # converted to number of bytes by graph compiler.
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
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=False
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_blocks),
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )
    else:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=True
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_blocks),
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu[
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

    comptime block_size_unrounded = registers_per_block // registers_per_thread

    # when testing other elementwise kernels, they appear to also use 128 as
    # the block size on blackwell specifically
    comptime block_size = 128 if ctx.default_device_info == B200 else block_size_unrounded - (
        block_size_unrounded % 2
    )

    comptime if _is_sm_100x_or_newer() and _USE_CLC_WORK_STEALING:
        _elementwise_impl_gpu_clc[func, simd_width, block_size](shape, ctx)
    else:
        _elementwise_impl_gpu_grid_stride[
            func,
            simd_width,
            block_size,
            num_waves,
            sm_count,
            threads_per_multiprocessor,
        ](shape, ctx)
