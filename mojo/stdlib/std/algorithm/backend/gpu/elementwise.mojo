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
    block_idx,
    grid_dim,
    thread_idx,
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
from std.math.uutils import ufloordiv, uceildiv, udivmod
from std.memory import stack_allocation
from std.sys._assembly import inlined_assembly
from std.sys.defines import get_defined_bool, get_defined_int
from std.sys.info import _has_sm_100x_or_newer, has_nvidia_gpu_accelerator
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import _get_start_indices_of_nth_subvolume


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#

comptime _USE_CLC_WORK_STEALING = get_defined_bool[
    "USE_CLC_WORK_STEALING", True
]()

comptime _PDL_LEVEL = PDLLevel(1)


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
    simd_width: Int,
    block_size: Int,
    FuncType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    elems_per_thread: Int,
    *,
    trace_description: StaticString = "",
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
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    var length = shape.flattened_length()
    var num_packed_elems, unpacked_tail_length = udivmod(length, simd_width)
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    var num_tiles = uceildiv(num_packed_elems, block_size * elems_per_thread)
    if num_tiles == 0:
        num_tiles = 1

    @__copy_capture(
        num_packed_elems, unpacked_tail_length, packed_region_length
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
    )
    @__name(
        t"{trace_description}_r{rank}_w{simd_width}_b{block_size}.clc.{handle_uneven_simd}",
        mangle=True,
    )
    def _kernel[*, handle_uneven_simd: Bool]():
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

        var tile_id = block_idx.x
        var phase: UInt32 = 0

        # Initialize mbarrier and kick-start CLC pipeline.
        if thread_idx.x == 0:
            mbarrier_init(mbar, Int32(1))
            if elect_one_sync_with_mask(mask=1):
                clusterlaunchcontrol_try_cancel(result, mbar)
            _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

        with PDL():
            # Work-stealing loop.
            while True:
                # Process current tile — each thread handles multiple packed
                # elements at stride block_size for coalesced access.
                var base = (
                    tile_id * block_size * elems_per_thread + thread_idx.x
                )

                @parameter
                @always_inline
                def _process_elem(start_indices: IndexList[rank, ...]):
                    comptime if handle_uneven_simd:
                        if (
                            start_indices[rank - 1] + simd_width
                            > shape[rank - 1]
                        ):
                            func[1, rank](start_indices.canonicalize())
                            var si = start_indices
                            comptime for _off in range(1, simd_width):
                                _advance_indices(si, shape)
                                func[1, rank](si.canonicalize())
                        else:
                            func[simd_width, rank](start_indices.canonicalize())
                    else:
                        func[simd_width, rank, simd_width](
                            start_indices.canonicalize()
                        )

                comptime for e in range(elems_per_thread):
                    var global_packed_idx = base + e * block_size
                    if global_packed_idx < num_packed_elems:
                        _process_elem(
                            _get_start_indices_of_nth_subvolume[0](
                                global_packed_idx * simd_width, shape
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

                tile_id = Int(next_tile[0])

            # Tail: only the first block handles remainder elements.
            if block_idx.x == 0 and thread_idx.x < unpacked_tail_length:
                func[1, rank](
                    _get_start_indices_of_nth_subvolume[0](
                        packed_region_length + thread_idx.x, shape
                    ).canonicalize()
                )

    if shape[rank - 1] % simd_width == 0:
        comptime kernel = _kernel[handle_uneven_simd=False]
        ctx.enqueue_function[kernel](
            grid_dim=num_tiles,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )
    else:
        comptime kernel = _kernel[handle_uneven_simd=True]
        ctx.enqueue_function[kernel](
            grid_dim=num_tiles,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )


# ===-----------------------------------------------------------------------===#
# Grid-stride elementwise (pre-SM100)
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu_grid_stride[
    rank: Int,
    //,
    simd_width: Int,
    block_size: Int,
    num_waves: Int,
    sm_count: Int,
    threads_per_multiprocessor: Int,
    FuncType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    elems_per_thread: Int,
    *,
    trace_description: StaticString = "",
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
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    # optimized implementation inspired by https://archive.md/Tye9y#selection-1101.2-1151.3

    var length = shape.flattened_length()
    var num_packed_elems, unpacked_tail_length = udivmod(length, simd_width)
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    # Grid is sized to saturate SMs — do NOT divide by elems_per_thread
    # here, as that would reduce occupancy. The unrolling only helps when
    # threads have multiple grid-stride iterations to process.
    var num_blocks: Int = clamp(
        uceildiv(num_packed_elems, block_size),
        1,
        sm_count
        * ufloordiv(threads_per_multiprocessor, block_size)
        * num_waves,
    )

    @__copy_capture(
        num_packed_elems, unpacked_tail_length, packed_region_length
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
    )
    @__name(
        t"{trace_description}_r{rank}_w{simd_width}_b{block_size}.gs.{handle_uneven_simd}",
        mangle=True,
    )
    def _kernel[*, handle_uneven_simd: Bool]():
        # process the packed region — each thread handles multiple packed
        # elements at stride block_size for coalesced access and ILP.
        var tid = thread_idx.x + block_size * block_idx.x
        var stride = block_size * grid_dim.x

        with PDL():

            @parameter
            @always_inline
            def _process_elem(start_indices: IndexList[rank, ...]):
                comptime if handle_uneven_simd:
                    if start_indices[rank - 1] + simd_width > shape[rank - 1]:
                        func[1, rank](start_indices.canonicalize())
                        var si = start_indices
                        comptime for _off in range(1, simd_width):
                            _advance_indices(si, shape)
                            func[1, rank](si.canonicalize())
                    else:
                        func[simd_width, rank](start_indices.canonicalize())
                else:
                    func[simd_width, rank, simd_width](
                        start_indices.canonicalize()
                    )

            for base_idx in range(
                tid,
                num_packed_elems,
                stride * elems_per_thread,
            ):
                comptime for e in range(elems_per_thread):
                    var idx = base_idx + e * stride
                    if idx < num_packed_elems:
                        _process_elem(
                            _get_start_indices_of_nth_subvolume[0](
                                idx * simd_width, shape
                            )
                        )

            # process the tail region
            if tid < unpacked_tail_length:
                func[1, rank](
                    _get_start_indices_of_nth_subvolume[0](
                        Int(packed_region_length + tid), shape
                    ).canonicalize()
                )

    if shape[rank - 1] % simd_width == 0:
        comptime kernel = _kernel[handle_uneven_simd=False]
        ctx.enqueue_function[kernel](
            grid_dim=num_blocks,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )
    else:
        comptime kernel = _kernel[handle_uneven_simd=True]
        ctx.enqueue_function[kernel](
            grid_dim=num_blocks,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )


# ===-----------------------------------------------------------------------===#
# Dual-elementwise grid-stride
# ===-----------------------------------------------------------------------===#


@always_inline
def _dual_elementwise_impl_gpu_grid_stride[
    rank: Int,
    //,
    simd_width: Int,
    block_size: Int,
    num_waves: Int,
    sm_count: Int,
    threads_per_multiprocessor: Int,
    Func0Type: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    Func1Type: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    elems_per_thread: Int,
    *,
    trace_description: StaticString = "",
](
    func_0: Func0Type,
    func_1: Func1Type,
    *,
    shape_0: IndexList[rank, ...],
    shape_1: IndexList[rank, ...],
    ctx: DeviceContext,
) raises:
    """Executes two elementwise functions over their respective shapes in a
    single kernel launch using a grid-stride loop.

    Both shapes must have the same element type (the dispatcher casts them
    before calling this function).

    Parameters:
        rank: The rank of the buffers.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        num_waves: The number of waves to saturate SMs.
        sm_count: The number of streaming multiprocessors.
        threads_per_multiprocessor: The number of threads per SM.
        Func0Type: The first body function type.
        Func1Type: The second body function type.
        elems_per_thread: Number of packed elements each thread processes per
            stride iteration for instruction-level parallelism.
        trace_description: Description of the trace.

    Args:
        func_0: The first body function closure.
        func_1: The second body function closure.
        shape_0: The shape for the first function.
        shape_1: The shape for the second function.
        ctx: The device context.

    Raises:
        If the GPU kernel launch fails.
    """

    var length_0 = shape_0.flattened_length()
    var length_1 = shape_1.flattened_length()
    var num_packed_0, unpacked_tail_0 = udivmod(length_0, simd_width)
    var num_packed_1, unpacked_tail_1 = udivmod(length_1, simd_width)
    var packed_region_0 = length_0 - unpacked_tail_0
    var packed_region_1 = length_1 - unpacked_tail_1
    var max_packed = (
        num_packed_0 if num_packed_0 > num_packed_1 else num_packed_1
    )

    if length_0 == 0 and length_1 == 0:
        return

    var num_blocks: Int = clamp(
        uceildiv(max_packed, block_size),
        1,
        sm_count
        * ufloordiv(threads_per_multiprocessor, block_size)
        * num_waves,
    )

    @__copy_capture(
        num_packed_0,
        unpacked_tail_0,
        packed_region_0,
        num_packed_1,
        unpacked_tail_1,
        packed_region_1,
        max_packed,
    )
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
    )
    @__name(
        t"{trace_description}_r{rank}_w{simd_width}_b{block_size}.dual_gs.{handle_uneven_simd}",
        mangle=True,
    )
    def _kernel[*, handle_uneven_simd: Bool]():
        var tid = thread_idx.x + block_size * block_idx.x
        var stride = block_size * grid_dim.x

        with PDL():

            @parameter
            @always_inline
            def _process_0(start_indices: IndexList[rank, ...]):
                comptime if handle_uneven_simd:
                    if start_indices[rank - 1] + simd_width > shape_0[rank - 1]:
                        func_0[1, rank](start_indices.canonicalize())
                        var si = start_indices
                        comptime for _off in range(1, simd_width):
                            _advance_indices(si, shape_0)
                            func_0[1, rank](si.canonicalize())
                    else:
                        func_0[simd_width, rank](start_indices.canonicalize())
                else:
                    func_0[simd_width, rank, simd_width](
                        start_indices.canonicalize()
                    )

            @parameter
            @always_inline
            def _process_1(start_indices: IndexList[rank, ...]):
                comptime if handle_uneven_simd:
                    if start_indices[rank - 1] + simd_width > shape_1[rank - 1]:
                        func_1[1, rank](start_indices.canonicalize())
                        var si = start_indices
                        comptime for _off in range(1, simd_width):
                            _advance_indices(si, shape_1)
                            func_1[1, rank](si.canonicalize())
                    else:
                        func_1[simd_width, rank](start_indices.canonicalize())
                else:
                    func_1[simd_width, rank, simd_width](
                        start_indices.canonicalize()
                    )

            for base_idx in range(
                tid,
                max_packed,
                stride * elems_per_thread,
            ):
                comptime for e in range(elems_per_thread):
                    var idx = base_idx + e * stride
                    if idx < num_packed_0:
                        _process_0(
                            _get_start_indices_of_nth_subvolume[0](
                                idx * simd_width, shape_0
                            )
                        )
                    if idx < num_packed_1:
                        _process_1(
                            _get_start_indices_of_nth_subvolume[0](
                                idx * simd_width, shape_1
                            )
                        )

            if tid < unpacked_tail_0:
                func_0[1, rank](
                    _get_start_indices_of_nth_subvolume[0](
                        Int(packed_region_0 + tid), shape_0
                    ).canonicalize()
                )
            if tid < unpacked_tail_1:
                func_1[1, rank](
                    _get_start_indices_of_nth_subvolume[0](
                        Int(packed_region_1 + tid), shape_1
                    ).canonicalize()
                )

    var both_aligned = (
        shape_0[rank - 1] % simd_width == 0
        and shape_1[rank - 1] % simd_width == 0
    )
    if both_aligned:
        comptime kernel = _kernel[handle_uneven_simd=False]
        ctx.enqueue_function[kernel](
            grid_dim=num_blocks,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )
    else:
        comptime kernel = _kernel[handle_uneven_simd=True]
        ctx.enqueue_function[kernel](
            grid_dim=num_blocks,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu[
    rank: Int,
    //,
    simd_width: Int,
    FuncType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    *,
    trace_description: StaticString = "",
](func: FuncType, *, shape: IndexList[rank, ...], ctx: DeviceContext) raises:
    """Executes `func[width, rank](indices)` as sub-tasks for a suitable
    combination of width and indices so as to cover shape on the GPU.

    Parameters:
        rank: The rank of the buffer.
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    comptime hw_info = ctx.default_device_info

    comptime registers_per_thread = 255
    comptime num_waves = get_defined_int["MOJO_ELEMENTWISE_NUM_WAVES", 32]()
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count: Int = hw_info.sm_count
    comptime threads_per_multiprocessor = hw_info.threads_per_multiprocessor

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
    comptime elems_per_thread = get_defined_int[
        "MOJO_ELEMENTWISE_ELEMS_PER_THREAD",
        4 if has_nvidia_gpu_accelerator() else 1,
    ]()
    comptime clc_min_packed_per_row = get_defined_int[
        "MOJO_ELEMENTWISE_CLC_MIN_PACKED_PER_ROW", 9
    ]()
    var packed_elems_per_row = ufloordiv(shape[rank - 1], simd_width)

    var length = UInt(shape.flattened_length())
    var use_32bit = length <= UInt(UInt32.MAX)

    if length == 0:
        return

    comptime if _has_sm_100x_or_newer() and _USE_CLC_WORK_STEALING:
        var num_packed = ufloordiv(Int(length), simd_width)
        var num_tiles = uceildiv(num_packed, block_size * elems_per_thread)

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
                    trace_description=trace_description,
                ](func=func, shape=shape.cast[DType.uint32](), ctx=ctx)
            else:
                _elementwise_impl_gpu_grid_stride[
                    simd_width=simd_width,
                    block_size=short_row_block_size,
                    num_waves=num_waves,
                    sm_count=sm_count,
                    threads_per_multiprocessor=threads_per_multiprocessor,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape.cast[DType.uint64](), ctx=ctx)
        else:
            if use_32bit:
                _elementwise_impl_gpu_clc[
                    simd_width=simd_width,
                    block_size=block_size,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape.cast[DType.uint32](), ctx=ctx)
            else:
                _elementwise_impl_gpu_clc[
                    simd_width=simd_width,
                    block_size=block_size,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape.cast[DType.uint64](), ctx=ctx)
    else:
        if use_32bit:
            _elementwise_impl_gpu_grid_stride[
                simd_width=simd_width,
                block_size=block_size,
                num_waves=num_waves,
                sm_count=sm_count,
                threads_per_multiprocessor=threads_per_multiprocessor,
                elems_per_thread=elems_per_thread,
                trace_description=trace_description,
            ](func=func, shape=shape.cast[DType.uint32](), ctx=ctx)
        else:
            _elementwise_impl_gpu_grid_stride[
                simd_width=simd_width,
                block_size=block_size,
                num_waves=num_waves,
                sm_count=sm_count,
                threads_per_multiprocessor=threads_per_multiprocessor,
                elems_per_thread=elems_per_thread,
                trace_description=trace_description,
            ](func=func, shape=shape.cast[DType.uint64](), ctx=ctx)


@always_inline
def _dual_elementwise_impl_gpu[
    rank: Int,
    //,
    simd_width: Int,
    Func0Type: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    Func1Type: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    *,
    trace_description: StaticString = "",
](
    func_0: Func0Type,
    func_1: Func1Type,
    *,
    shape_0: IndexList[rank],
    shape_1: IndexList[rank],
    ctx: DeviceContext,
) raises:
    """Executes two elementwise functions over their respective shapes in a
    single GPU kernel launch.

    Parameters:
        rank: The rank of the buffers.
        simd_width: The SIMD vector width to use.
        Func0Type: The first body function type.
        Func1Type: The second body function type.
        trace_description: Description of the trace.

    Args:
        func_0: The first body function closure.
        func_1: The second body function closure.
        shape_0: The shape for the first function.
        shape_1: The shape for the second function.
        ctx: The device context.

    Raises:
        If the GPU kernel launch fails.
    """

    comptime hw_info = ctx.default_device_info

    comptime registers_per_thread = 255
    comptime num_waves = get_defined_int["MOJO_ELEMENTWISE_NUM_WAVES", 32]()
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count: Int = hw_info.sm_count
    comptime threads_per_multiprocessor = hw_info.threads_per_multiprocessor

    comptime assert (
        sm_count > 0 and threads_per_multiprocessor > 0
    ), "the sm_count and thread_count must be known"

    comptime block_size_unrounded = registers_per_block // registers_per_thread
    comptime warp_size = WARP_SIZE
    comptime default_block_size = (
        128 if ctx.default_device_info
        == B200 else block_size_unrounded - (block_size_unrounded % warp_size)
    )
    comptime block_size = get_defined_int[
        "MOJO_ELEMENTWISE_BLOCK_SIZE", default_block_size
    ]()
    comptime elems_per_thread = get_defined_int[
        "MOJO_ELEMENTWISE_ELEMS_PER_THREAD",
        4 if has_nvidia_gpu_accelerator() else 1,
    ]()

    var max_length = UInt(shape_0.flattened_length())
    var len_1 = UInt(shape_1.flattened_length())
    if len_1 > max_length:
        max_length = len_1
    var use_32bit = max_length <= UInt(UInt32.MAX)

    if max_length == 0:
        return

    if use_32bit:
        _dual_elementwise_impl_gpu_grid_stride[
            simd_width=simd_width,
            block_size=block_size,
            num_waves=num_waves,
            sm_count=sm_count,
            threads_per_multiprocessor=threads_per_multiprocessor,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func_0=func_0,
            func_1=func_1,
            shape_0=shape_0.cast[DType.uint32](),
            shape_1=shape_1.cast[DType.uint32](),
            ctx=ctx,
        )
    else:
        _dual_elementwise_impl_gpu_grid_stride[
            simd_width=simd_width,
            block_size=block_size,
            num_waves=num_waves,
            sm_count=sm_count,
            threads_per_multiprocessor=threads_per_multiprocessor,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func_0=func_0,
            func_1=func_1,
            shape_0=shape_0.cast[DType.uint64](),
            shape_1=shape_1.cast[DType.uint64](),
            ctx=ctx,
        )
