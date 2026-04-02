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
    PDL,
    PDLLevel,
    WARP_SIZE,
    barrier,
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    grid_dim_uint as grid_dim,
    thread_idx_uint as thread_idx,
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
from std.sys.defines import get_defined_bool, get_defined_int
from std.sys.info import _is_sm_100x_or_newer, has_nvidia_gpu_accelerator
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import _get_start_indices_of_nth_subvolume


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


@always_inline("nodebug")
def _advance_indices[
    rank: Int
](mut idx: IndexList[rank, ...], shape: IndexList[rank, ...]):
    """Advances a multi-dimensional index by one element in row-major order."""
    idx[rank - 1] += 1
    comptime for d in reversed(range(1, rank)):
        if idx[d] >= shape[d]:
            idx[d] = 0
            idx[d - 1] += 1


# ===-----------------------------------------------------------------------===#
# Work-stealing elementwise (SM100+ CLC)
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu_clc[
    rank: Int,
    //,
    simd_width: UInt,
    block_size: Int,
    FuncType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) unified register_passable -> None,
    elems_per_thread: UInt,
    pdl_level: PDLLevel,
](func: FuncType, shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func` over `shape` on SM100+ GPUs using Cluster Launch Control
    work-stealing.

    Each thread block processes one tile of `block_size * elems_per_thread`
    packed elements, then attempts to cancel and steal the work of a
    not-yet-launched block. This gives the reduced overhead of persistent
    kernels with the preemption and load-balancing benefits of one-block-per-
    tile launches.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        FuncType: The body function type.
        elems_per_thread: Number of packed elements each thread processes per
            tile. Higher values increase instruction-level parallelism and
            reduce CLC cancel frequency.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        func: The closure carrying the captured state of the body function.
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

    var num_tiles = ceildiv(
        num_packed_elems, UInt(block_size) * elems_per_thread
    )
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
        # Shared variables for single-barrier broadcast of cancel results.
        var canceled = stack_allocation[
            1,
            UInt32,
            address_space=AddressSpace.SHARED,
        ]()
        var next_tile = stack_allocation[
            1,
            UInt32,
            address_space=AddressSpace.SHARED,
        ]()

        var tile_id = UInt(block_idx.x)
        var phase: UInt32 = 0

        with PDL():
            # Initialize mbarrier and kick-start CLC pipeline.
            if thread_idx.x == 0:
                mbarrier_init(mbar, Int32(1))
                if elect_one_sync_with_mask(mask=1):
                    clusterlaunchcontrol_try_cancel(result, mbar)
                _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

            # Work-stealing loop.
            while True:
                # Process current tile — each thread handles multiple packed
                # elements at stride block_size for coalesced access.
                var base = tile_id * UInt(block_size * elems_per_thread) + UInt(
                    thread_idx.x
                )

                @parameter
                @always_inline
                def _process_elem(start_indices: IndexList[rank, ...]):
                    comptime if handle_uneven_simd:
                        if (
                            start_indices[rank - 1] + Int(simd_width)
                            > shape[rank - 1]
                        ):
                            func[1, rank](start_indices.canonicalize())
                            var si = start_indices
                            comptime for _off in range(1, Int(simd_width)):
                                _advance_indices(si, shape)
                                func[1, rank](si.canonicalize())
                        else:
                            func[Int(simd_width), rank](
                                start_indices.canonicalize()
                            )
                    else:
                        func[Int(simd_width), rank, Int(simd_width)](
                            start_indices.canonicalize()
                        )

                comptime for e in range(elems_per_thread):
                    var global_packed_idx = base + UInt(e * block_size)
                    if global_packed_idx < num_packed_elems:
                        _process_elem(
                            _get_start_indices_of_nth_subvolume[0](
                                Int(global_packed_idx * simd_width), shape
                            )
                        )

                # Leader: wait for cancel result, extract values, then
                # immediately issue the next cancel before the barrier.
                # This eliminates one barrier per iteration by letting
                # thread 0 read and reuse the result buffer in the same
                # critical section, publishing via separate shared vars.
                if thread_idx.x == 0:
                    _mbarrier_wait_acquire_cta(mbar, phase)
                    phase ^= 1

                    var is_canceled = (
                        clusterlaunchcontrol_query_cancel_is_canceled(result)
                    )
                    var ctaid = (
                        clusterlaunchcontrol_query_cancel_get_first_ctaid["x"](
                            result
                        )
                    )

                    # Issue next cancel only if this one succeeded.
                    if Bool(is_canceled):
                        cluster_sync_release()
                        cluster_sync_acquire()
                        if elect_one_sync_with_mask(mask=1):
                            clusterlaunchcontrol_try_cancel(result, mbar)
                        _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

                    # Publish extracted values for all threads.
                    canceled[0] = is_canceled
                    next_tile[0] = ctaid

                # Single barrier — all threads see broadcast values.
                barrier()

                if canceled[0] == 0:
                    break

                tile_id = UInt(next_tile[0])

            # Tail: only the first block handles remainder elements.
            if UInt(block_idx.x) == 0 and UInt(thread_idx.x) < (
                unpacked_tail_length
            ):
                func[1, rank](
                    _get_start_indices_of_nth_subvolume[0](
                        Int(packed_region_length + UInt(thread_idx.x)), shape
                    ).canonicalize()
                )

    if shape[rank - 1] % Int(simd_width) == 0:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=False
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_tiles),
            block_dim=block_size,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=True
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_tiles),
            block_dim=block_size,
            attributes=pdl_launch_attributes(pdl_level),
        )


# ===-----------------------------------------------------------------------===#
# Grid-stride elementwise (pre-SM100)
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu_grid_stride[
    rank: Int,
    //,
    simd_width: UInt,
    block_size: Int,
    num_waves: Int,
    sm_count: UInt,
    threads_per_multiprocessor: UInt,
    FuncType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) unified register_passable -> None,
    elems_per_thread: UInt,
    pdl_level: PDLLevel,
](func: FuncType, shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func` over `shape` using a grid-stride loop.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        num_waves: The number of waves to saturate SMs.
        sm_count: The number of streaming multiprocessors.
        threads_per_multiprocessor: The number of threads per SM.
        FuncType: The body function type.
        elems_per_thread: Number of packed elements each thread processes per
            stride iteration for instruction-level parallelism.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        func: The closure carrying the captured state of the body function.
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

    # Grid is sized to saturate SMs — do NOT divide by elems_per_thread
    # here, as that would reduce occupancy. The unrolling only helps when
    # threads have multiple grid-stride iterations to process.
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
        # process the packed region — each thread handles multiple packed
        # elements at stride block_size for coalesced access and ILP.
        var tid = thread_idx.x + block_size * block_idx.x
        var stride = block_size * grid_dim.x

        with PDL():

            @parameter
            @always_inline
            def _process_elem(start_indices: IndexList[rank, ...]):
                comptime if handle_uneven_simd:
                    if (
                        start_indices[rank - 1] + Int(simd_width)
                        > shape[rank - 1]
                    ):
                        func[1, rank](start_indices.canonicalize())
                        var si = start_indices
                        comptime for _off in range(1, Int(simd_width)):
                            _advance_indices(si, shape)
                            func[1, rank](si.canonicalize())
                    else:
                        func[Int(simd_width), rank](
                            start_indices.canonicalize()
                        )
                else:
                    func[Int(simd_width), rank, Int(simd_width)](
                        start_indices.canonicalize()
                    )

            for base_idx in range(
                tid,
                num_packed_elems,
                stride * elems_per_thread,
            ):
                comptime for e in range(elems_per_thread):
                    var idx = base_idx + UInt(e) * stride
                    if idx < num_packed_elems:
                        _process_elem(
                            _get_start_indices_of_nth_subvolume[0](
                                Int(idx * simd_width), shape
                            )
                        )

            # process the tail region
            if tid < unpacked_tail_length:
                func[1, rank](
                    _get_start_indices_of_nth_subvolume[0](
                        Int(packed_region_length + tid), shape
                    ).canonicalize()
                )

    if shape[rank - 1] % Int(simd_width) == 0:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=False
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_blocks),
            block_dim=block_size,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        comptime kernel = _kernel[
            block_size=UInt(block_size), handle_uneven_simd=True
        ]
        ctx.enqueue_function[kernel, kernel](
            grid_dim=Int(num_blocks),
            block_dim=block_size,
            attributes=pdl_launch_attributes(pdl_level),
        )


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu[
    rank: Int,
    //,
    *,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: UInt,
    pdl_level: PDLLevel = PDLLevel(1),
](*, shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func[width, rank](indices)` as sub-tasks for a suitable
    combination of width and indices so as to cover shape on the GPU.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    def func_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](indices: IndexList[rank]) unified register_passable {}:
        func[width, rank, alignment](indices)

    comptime hw_info = ctx.default_device_info

    comptime registers_per_thread = 255
    comptime num_waves = get_defined_int["MOJO_ELEMENTWISE_NUM_WAVES", 32]()
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count = UInt(hw_info.sm_count)
    comptime threads_per_multiprocessor = UInt(
        hw_info.threads_per_multiprocessor
    )

    comptime assert (
        sm_count > 0 and threads_per_multiprocessor > 0
    ), "the sm_count and thread_count must be known"

    comptime block_size_unrounded = registers_per_block // registers_per_thread

    # Round down to the warp/wavefront size for correct hardware alignment
    # on both NVIDIA (warp=32) and AMD (wavefront=64).
    comptime warp_size = WARP_SIZE
    comptime default_block_size = (
        128 if ctx.default_device_info
        == B200 else block_size_unrounded - (block_size_unrounded % warp_size)
    )
    comptime block_size = get_defined_int[
        "MOJO_ELEMENTWISE_BLOCK_SIZE", default_block_size
    ]()
    comptime short_row_block_size = get_defined_int[
        "MOJO_ELEMENTWISE_SHORT_ROW_BLOCK_SIZE",
        64 if ctx.default_device_info == B200 else default_block_size,
    ]()
    comptime elems_per_thread = UInt(
        get_defined_int[
            "MOJO_ELEMENTWISE_ELEMS_PER_THREAD",
            4 if has_nvidia_gpu_accelerator() else 1,
        ]()
    )
    comptime clc_min_packed_per_row = UInt(
        get_defined_int["MOJO_ELEMENTWISE_CLC_MIN_PACKED_PER_ROW", 9]()
    )
    var packed_elems_per_row = UInt(shape[rank - 1]) // simd_width

    var length = UInt(shape.flattened_length())
    var use_32bit = length <= UInt(UInt32.MAX)

    if length == 0:
        return

    comptime if _is_sm_100x_or_newer() and _USE_CLC_WORK_STEALING:
        var num_packed = length // simd_width
        var num_tiles = ceildiv(num_packed, UInt(block_size) * elems_per_thread)

        if packed_elems_per_row < clc_min_packed_per_row or num_tiles <= 1:
            # Short rows or single-tile workloads: use grid-stride to avoid
            # CLC synchronization overhead (mbarrier, cancel, cluster fences)
            # when there is nothing to steal.
            if use_32bit:
                _elementwise_impl_gpu_grid_stride[
                    simd_width=simd_width,
                    block_size=short_row_block_size,
                    num_waves=num_waves,
                    sm_count=sm_count,
                    threads_per_multiprocessor=threads_per_multiprocessor,
                    elems_per_thread=elems_per_thread,
                    pdl_level=pdl_level,
                ](func=func_unified, shape=shape.cast[DType.uint32](), ctx=ctx)
            else:
                _elementwise_impl_gpu_grid_stride[
                    simd_width=simd_width,
                    block_size=short_row_block_size,
                    num_waves=num_waves,
                    sm_count=sm_count,
                    threads_per_multiprocessor=threads_per_multiprocessor,
                    elems_per_thread=elems_per_thread,
                    pdl_level=pdl_level,
                ](func=func_unified, shape=shape.cast[DType.uint64](), ctx=ctx)
        else:
            if use_32bit:
                _elementwise_impl_gpu_clc[
                    simd_width=simd_width,
                    block_size=block_size,
                    elems_per_thread=elems_per_thread,
                    pdl_level=pdl_level,
                ](func=func_unified, shape=shape.cast[DType.uint32](), ctx=ctx)
            else:
                _elementwise_impl_gpu_clc[
                    simd_width=simd_width,
                    block_size=block_size,
                    elems_per_thread=elems_per_thread,
                    pdl_level=pdl_level,
                ](func=func_unified, shape=shape.cast[DType.uint64](), ctx=ctx)
    else:
        if use_32bit:
            _elementwise_impl_gpu_grid_stride[
                simd_width=simd_width,
                block_size=block_size,
                num_waves=num_waves,
                sm_count=sm_count,
                threads_per_multiprocessor=threads_per_multiprocessor,
                elems_per_thread=elems_per_thread,
                pdl_level=pdl_level,
            ](func=func_unified, shape=shape.cast[DType.uint32](), ctx=ctx)
        else:
            _elementwise_impl_gpu_grid_stride[
                simd_width=simd_width,
                block_size=block_size,
                num_waves=num_waves,
                sm_count=sm_count,
                threads_per_multiprocessor=threads_per_multiprocessor,
                elems_per_thread=elems_per_thread,
                pdl_level=pdl_level,
            ](func=func_unified, shape=shape.cast[DType.uint64](), ctx=ctx)
