# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from collections import OptionalReg
from math import ceildiv
from sys import align_of, simd_width_of, size_of

from buffer.buffer import NDBuffer
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.device_context import DeviceBuffer
from gpu.id import (
    block_id_in_cluster,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.id import warp_id as get_warp_id
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import (
    AddressSpace,
    async_copy,
    fence_mbarrier_init,
    fence_async_view_proxy,
    external_memory,
)
from gpu.mma import st_matrix
from gpu.sync import async_copy_arrive, named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    st_matrix_n_layout,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
)
from memory import bitcast, stack_allocation
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import block_swizzle
from ..tile_scheduler import MatmulSchedule, TileScheduler, RasterOrder
from ..tile_scheduler_splitk import SplitKTileScheduler
from ....structuring import NVIDIASharedMemoryManager as SharedMemoryManager
from ....structuring import (
    SMemTileType,
    SMemTileIterType,
    RegTileType,
    SMemBarrier,
)


# Ring buffer abstraction for producer-consumer synchronization
@register_passable("trivial")
struct RingBuffer[
    num_pipeline_stages: Int,
    num_consumers: Int,
    cluster_size: Int,
]:
    """Ring buffer for managing pipeline synchronization between producers and consumers.

    This struct encapsulates the synchronization logic for a multi-stage pipeline
    with one producer and multiple consumers, supporting both single-block and
    multi-cluster configurations.
    """

    # Barriers for synchronization
    var full_mbar: SMemBarrier
    var empty_mbar: SMemBarrier

    fn __init__(
        out self,
        full_mbar: SMemBarrier,
        empty_mbar: SMemBarrier,
    ):
        """Initialize ring buffer with barrier pointers."""
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar

    @always_inline
    fn init_barriers(self, thread_idx: UInt):
        """Initialize the ring buffer barriers."""
        if thread_idx == 0:

            @parameter
            for i in range(num_pipeline_stages):
                self.full_mbar[i].init(1)
                self.empty_mbar[i].init(num_consumers * cluster_size)

    @always_inline
    fn init_barriers_cpasync(self, thread_idx: UInt):
        """Initialize barriers for cp.async variant."""
        if thread_idx == 0:

            @parameter
            for i in range(num_pipeline_stages):
                self.full_mbar[i].init(WARPGROUP_SIZE)
                self.empty_mbar[i].init(num_consumers * cluster_size)

    @always_inline
    fn producer_wait_for_empty[
        pipeline_stages: Int,
        expected_bytes: Int = 0,
    ](self, mut write_state: PipelineState[pipeline_stages]) -> UInt32:
        """Producer waits for empty buffer slot and prepares for loading."""
        var write_idx = write_state.index()
        self.empty_mbar[Int(write_idx)].wait(write_state.phase())
        if expected_bytes > 0:
            self.full_mbar[Int(write_idx)].expect_bytes(expected_bytes)
        return write_idx

    @always_inline
    fn producer_signal_full[
        pipeline_stages: Int,
    ](self, mut write_state: PipelineState[pipeline_stages]):
        """Producer signals that buffer is full and advances to next stage."""
        write_state.step()

    @always_inline
    fn producer_signal_full_and_arrive[
        pipeline_stages: Int,
    ](self, mut write_state: PipelineState[pipeline_stages]):
        """Producer signals for cp.async operations.

        This handles the specific signaling pattern needed for cp.async:
        1. Signal async copy arrival
        2. Arrive at the barrier
        3. Advance to next stage
        """
        var write_idx = write_state.index()
        async_copy_arrive(self.full_mbar[write_idx].unsafe_ptr())
        _ = self.full_mbar[write_idx].arrive()
        write_state.step()

    @always_inline
    fn consumer_wait_for_full[
        pipeline_stages: Int,
    ](self, mut read_state: PipelineState[pipeline_stages]) -> UInt32:
        """Consumer waits for full buffer slot."""
        var read_idx = read_state.index()
        self.full_mbar[Int(read_idx)].wait(read_state.phase())
        return read_idx

    @always_inline
    fn consumer_signal_empty(
        self,
        read_idx: UInt32,
        warp_group_thread_idx: UInt,
    ):
        """Consumer signals that buffer slot is empty."""

        @parameter
        if cluster_size > 1:
            if warp_group_thread_idx < UInt(cluster_size):
                _ = self.empty_mbar[Int(read_idx)].arrive_cluster(
                    warp_group_thread_idx
                )
        else:
            if warp_group_thread_idx == 0:
                _ = self.empty_mbar[Int(read_idx)].arrive()

    @always_inline
    fn consumer_advance[
        pipeline_stages: Int,
    ](self, mut read_state: PipelineState[pipeline_stages]):
        """Consumer advances to next pipeline stage."""
        read_state.step()

    @always_inline
    fn arrive_empty_barriers(
        self,
        warp_group_thread_idx: UInt,
    ):
        """Helper to arrive at empty barriers during consumer initialization."""

        @parameter
        for i in range(num_pipeline_stages):

            @parameter
            if cluster_size > 1:
                if warp_group_thread_idx < UInt(cluster_size):
                    _ = self.empty_mbar[i].arrive_cluster(warp_group_thread_idx)
            else:
                if warp_group_thread_idx == 0:
                    _ = self.empty_mbar[i].arrive()

    # Storage size is now handled by the parent struct


@always_inline
fn find_K_alignment_upto_16B(row_bytes_arg: Int) -> Int:
    """Find alignment among 1B, 2B, 4B, 16B based on the row's bytes."""

    var row_bytes = row_bytes_arg
    var alignment = 1

    @parameter
    for i in range(4):
        if row_bytes & 1 == 1:
            return alignment
        row_bytes >>= 1
        alignment <<= 1

    return alignment


# Shared memory structure for Hopper SM90 kernel
@register_passable("trivial")
struct HopperMatmulSM90Kernel_SMem[
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    c_type: DType,
    c_layout: Layout,
    num_pipeline_stages: Int,
    num_consumers: Int,
    cluster_size: Int,
]:
    alias SMM = SharedMemoryManager[]

    # Tile iterator types
    alias ATileIterType = Self.SMM.TileIter[
        a_type, a_layout, num_pipeline_stages
    ]
    alias BTileIterType = Self.SMM.TileIter[
        b_type, b_layout, num_pipeline_stages
    ]
    alias CTileType = Self.SMM.Tile[c_type, c_layout]

    # Pipeline barrier types
    alias PipelineBarrierType = Self.SMM.Array[
        SharedMemBarrier, num_pipeline_stages
    ]

    # Tile iterators
    var a_tiles: Self.ATileIterType.T
    var b_tiles: Self.BTileIterType.T
    var c_tile: Self.CTileType.T

    # Pipeline barriers (full and empty)
    var full_mbar: Self.PipelineBarrierType.T
    var empty_mbar: Self.PipelineBarrierType.T

    fn __init__(out self):
        var smem_mgr = Self.SMM()
        # Initialize tile iterators
        self.a_tiles = Self.ATileIterType.build(smem_mgr)
        self.b_tiles = Self.BTileIterType.build(smem_mgr)
        self.c_tile = Self.CTileType.build(smem_mgr)
        # Initialize barriers
        self.full_mbar = Self.PipelineBarrierType.build(smem_mgr)
        self.empty_mbar = Self.PipelineBarrierType.build(smem_mgr)

    @staticmethod
    @always_inline
    fn pipeline_storage_size() -> Int:
        """Calculate the memory size for all pipeline stages."""
        var a_size = Self.ATileIterType.storage_size
        var b_size = Self.BTileIterType.storage_size

        return (
            # A and B tile iterators with padding
            a_size
            + b_size
            # Pipeline barriers (full + empty)
            + 2 * Self.PipelineBarrierType.storage_size
        )

    @staticmethod
    @always_inline
    fn output_storage_size() -> Int:
        """Calculate the memory size for output tile."""
        return Self.CTileType.storage_size

    @staticmethod
    @always_inline
    fn storage_size() -> Int:
        """Calculate the total storage size."""
        return Self.pipeline_storage_size() + Self.output_storage_size()


@register_passable("trivial")
struct ScatterGather:
    @staticmethod
    @always_inline
    fn load_tile[
        dtype: DType,
        tile_layout: Layout,
        desc_layout: Layout,
        dst_layout: Layout, //,
        cluster_size: Int,
        use_partitioned_multicast: Bool,
    ](
        tma_op: TMATensorTile[dtype, tile_layout, desc_layout],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            address_space = AddressSpace.SHARED,
            alignment=128,
            *_, **_,
        ],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
        rank: UInt,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        alias tma_load_size = desc_layout.size()
        alias tma_rows = desc_layout.shape[0].value()

        @parameter
        if cluster_size > 1:

            @parameter
            if use_partitioned_multicast:
                tma_op.async_multicast_load_partitioned[
                    tma_rows, tma_load_size
                ](
                    dst,
                    mem_barrier,
                    rank,
                    coords,
                    multicast_mask,
                )

            else:
                if rank == 0:
                    tma_op.async_multicast_load(
                        dst,
                        mem_barrier,
                        coords,
                        multicast_mask,
                    )

        else:
            tma_op.async_copy(
                dst,
                mem_barrier,
                coords,
            )

    @staticmethod
    @always_inline("nodebug")
    fn load_tile[
        dtype: DType,
        src_layout: Layout,
        dst_layout: Layout, //,
        thread_layout: Layout,
        swizzle_mode: TensorMapSwizzle,
        vector_size: Int,
    ](
        src: LayoutTensor[
            dtype,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            *_, **_,
        ],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            *_, **_,
        ],
        tile_idx_m: Int,
        tile_idx_n: Int,
    ):
        alias BM = dst_layout.shape[0].value()
        alias BN = dst_layout.shape[1].value()
        var a_gmem_tile = src.tile[BM, BN](
            tile_idx_m,
            tile_idx_n,
        ).vectorize[1, vector_size]()
        Self.async_copy_with_bound_check[
            thread_layout,
            swizzle_mode,
        ](a_gmem_tile, dst.vectorize[1, vector_size]())

    @staticmethod
    @always_inline
    fn async_copy_with_bound_check[
        dtype: DType,
        src_layout: Layout,
        dst_layout: Layout, //,
        thread_layout: Layout,
        swizzle_mode: TensorMapSwizzle,
    ](
        src: LayoutTensor[
            dtype,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            *_, **_,
        ],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            *_, **_,
        ],
    ):
        """Helper function for cp.async with bound checking."""
        constrained[
            src.layout.rank() == 2, "Global memory tile must be rank 2."
        ]()

        constrained[
            src_layout.shape == dst_layout.shape,
            "Global memory tile must match source layout: "
            + String(src_layout)
            + " != "
            + String(dst_layout),
        ]()

        alias src_shape1 = src.layout.shape[1].value()
        alias swizzle_bytes = swizzle_mode.bytes()
        constrained[
            src_shape1 * src.element_size * size_of[src.dtype]()
            == swizzle_bytes,
            String(
                "Global memory tile shape-1 ",
                src_shape1 * src.element_size,
                "must match swizzle bytes.",
                swizzle_bytes,
            ),
        ]()

        var src_frag = src.distribute[thread_layout](thread_idx.x)
        var dst_frag = dst.distribute[thread_layout](thread_idx.x)

        alias src_stride0 = src.layout.stride[0].value()
        var src_bound0 = Int32(src.runtime_layout.shape.value[0])
        var src_bound1 = (
            Int32(src.runtime_layout.shape.value[1]) * dst.element_size
        )

        var dst_frag_offset = dst_frag.distance(dst.ptr)
        alias dst_stride0 = dst.layout.stride[0].value()
        var dst_frag_base_coord0 = Int32(dst_frag_offset // dst_stride0)
        var dst_frag_base_coord1 = Int32(dst_frag_offset % dst_stride0)
        alias swizzle = make_swizzle[
            8,
            Int(swizzle_bytes // size_of[dst.dtype]()),
            Int(simd_width_of[dst.dtype]()),
        ]()

        alias num_vecs = dst_frag.layout.size()

        @parameter
        for i in range(num_vecs):
            alias dst_idx = dst_frag.layout(i)
            alias dst_idx_base = dst_idx % swizzle.size()
            alias dst_idx_diff = dst_idx - dst_idx_base
            var dst_swizzled_idx = Int32(
                swizzle(dst_frag_offset + dst_idx_base) + dst_idx_diff
            )
            var dst_ptr = dst.ptr + Int(dst_swizzled_idx)

            # TODO: we should be able to use idx2crd for this.
            alias dst_shifted_coord0 = dst_idx // dst_stride0
            alias dst_shifted_coord1 = dst_idx % dst_stride0
            var dst_coord0 = dst_shifted_coord0 + dst_frag_base_coord0
            var dst_coord1 = dst_shifted_coord1 + dst_frag_base_coord1

            alias size_bytes = dst.element_size * size_of[dst.dtype]()

            var src_ptr = (
                src.ptr.address_space_cast[AddressSpace.GLOBAL]()
                + dst_coord1
                + dst_coord0 * src_stride0
            )

            if dst_coord0 < src_bound0 and dst_coord1 < src_bound1:
                async_copy[
                    size_bytes,
                    bypass_L1_16B=False,
                    fill = Scalar[dst.dtype](0),
                ](src_ptr, dst_ptr, src_size=size_bytes)
            else:
                # Zero-fill the OOB address
                async_copy[
                    size_bytes, bypass_L1_16B=False, fill = Scalar[dst.dtype](0)
                ](src_ptr, dst_ptr, src_size=0)


struct HopperMatmulSM90Kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: Int,
    num_threads: Int = 128,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    partitioned_multicast: Bool = False,
    use_tma_store: Bool = False,
    promotion_frequency: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    hilbert_swizzle: Bool = False,
]:
    """Hopper SM90 Matmul kernel with structured shared memory management."""

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_consumer = (num_threads // 128) - 1
    alias num_consumer_threads = Self.num_consumer * 128

    alias num_m_mmas = Self.BM // wgmma_shape[0] // Self.num_consumer
    alias num_n_mmas = Self.BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias a_smem_layout = tile_layout_k_major[
        a_type, Self.BM, Self.BK, a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, Self.BN, Self.BK, b_swizzle
    ]()

    alias AccumRegTileType = RegTileType[
        Self.accum_type,
        Layout.row_major(Self.num_m_mmas * Self.num_n_mmas, Self.c_frag_size),
    ]

    alias cluster_size = Int(
        cluster_shape[0] * cluster_shape[1] * cluster_shape[2]
    )

    alias SMem = HopperMatmulSM90Kernel_SMem[
        a_type,
        Self.a_smem_layout,
        b_type,
        Self.b_smem_layout,
        c_type,
        c_smem_layout,
        num_pipeline_stages,
        Self.num_consumer,
        Self.cluster_size,
    ]

    @staticmethod
    @always_inline
    fn num_regs() -> Int:
        if Self.num_consumer == 1:
            return 256
        if Self.num_consumer == 2:
            return 240
        return 160

    @staticmethod
    @always_inline
    fn validate_constraints():
        """Validate common constraints for all kernel variants."""
        constrained[transpose_b, "Only support transposed B in layout"]()

        constrained[
            not partitioned_multicast
            or a_swizzle.bytes() // size_of[a_type]() == Self.BK,
            (
                "Currently partitioned multi-casting is only supported when BK"
                " == (a_swizzle.bytes // size_of[a_type])"
            ),
        ]()
        constrained[
            not partitioned_multicast
            or b_swizzle.bytes() // size_of[b_type]() == Self.BK,
            (
                "Currently partitioned multi-casting is only supported when BK"
                " == (b_swizzle.bytes // size_of[b_type])"
            ),
        ]()

    @staticmethod
    @always_inline
    fn async_load_AB_tma[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        /,
        *,
        num_k_iters: Int,
        tile_shape: IndexList[3],
        cluster_dims: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
        use_partitioned_multicast: Bool = False,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        a_smem_iter: Self.SMem.ATileIterType.T,
        b_smem_iter: Self.SMem.BTileIterType.T,
        m_coord: UInt,
        n_coord: UInt,
        k_coord: UInt,
        rank_n: UInt,
        rank_m: UInt,
        ring_buffer: RingBuffer,
        mut write_state: PipelineState[num_pipeline_stages],
    ):
        """Load A and B tiles using TMA (Tensor Memory Accelerator)."""
        alias a_expected_bytes = a_smem_iter.layout.size() * size_of[a_type]()
        alias b_expected_bytes = b_smem_iter.layout.size() * size_of[b_type]()
        alias expected_bytes = a_expected_bytes + b_expected_bytes

        alias CLUSTER_N = UInt(cluster_dims[0])
        alias CLUSTER_M = UInt(cluster_dims[1])

        alias BM = tile_shape[0]
        alias BN = tile_shape[1]
        alias BK = tile_shape[2]

        var multicast_column_mask = 0

        @parameter
        for i in range(CLUSTER_M):
            multicast_column_mask |= Int(1 << (i * CLUSTER_N))

        var multicast_row_mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N)

        alias num_full_k_iters = ceildiv(
            num_k_iters, ring_buffer.num_pipeline_stages
        )
        alias num_remaining_k_iters = num_k_iters % ring_buffer.num_pipeline_stages

        # `num_pipeline_stages_to_unroll` determines how many pipeline stages should be unroll in the producer loop;
        # if num_k_iters % pipeline_stages != 0 then for the last loop, we only unroll (num_k_iters % pipeline_stages) pipeline stages
        @always_inline
        @parameter
        fn producer_loop[
            num_pipeline_stages_to_unroll: Int,
        ](k_iter: Int):
            @parameter
            for j in range(num_pipeline_stages_to_unroll):
                var write_idx = ring_buffer.producer_wait_for_empty[
                    num_pipeline_stages,
                    expected_bytes,
                ](write_state)

                var k_offset = UInt(
                    k_coord
                    + UInt(k_iter * ring_buffer.num_pipeline_stages)
                    + UInt(j)
                ) * UInt(BK)

                var a_smem_tile = a_smem_iter.next(write_idx)[]
                ScatterGather.load_tile[
                    Int(CLUSTER_N), use_partitioned_multicast
                ](
                    a_tma_op,
                    a_smem_tile,
                    ring_buffer.full_mbar[write_idx],
                    rank_n,
                    (k_offset, m_coord),
                    multicast_row_mask,
                )

                var b_smem_tile = b_smem_iter.next(write_idx)[]
                ScatterGather.load_tile[
                    Int(CLUSTER_M), use_partitioned_multicast
                ](
                    b_tma_op,
                    b_smem_tile,
                    ring_buffer.full_mbar[write_idx],
                    rank_m,
                    (k_offset, n_coord),
                    multicast_column_mask << rank_n,
                )

                ring_buffer.producer_signal_full[num_pipeline_stages](
                    write_state
                )

        @parameter
        if num_remaining_k_iters == 0:
            for k_iter in range(num_full_k_iters):
                producer_loop[ring_buffer.num_pipeline_stages](k_iter)
        else:
            for k_iter in range(num_full_k_iters - 1):
                producer_loop[ring_buffer.num_pipeline_stages](k_iter)
            producer_loop[num_remaining_k_iters](num_full_k_iters - 1)

    @staticmethod
    @always_inline
    fn async_load_AB_cpasync[
        a_mem_layout: Layout,
        b_mem_layout: Layout, //,
        /,
        *,
        pipeline_stages: Int,
        swizzle_mode: TensorMapSwizzle,
        vector_size: Int,
        num_k_iters: Int,
        tile_shape: IndexList[3],
    ](
        a: LayoutTensor[
            a_type,
            a_mem_layout,
            MutableAnyOrigin,
        ],
        b: LayoutTensor[
            b_type,
            b_mem_layout,
            MutableAnyOrigin,
        ],
        block_idx_m: UInt,
        block_idx_n: UInt,
        a_smem_iter: Self.SMem.ATileIterType.T,
        b_smem_iter: Self.SMem.BTileIterType.T,
        ring_buffer: RingBuffer,
        mut write_state: PipelineState[pipeline_stages],
    ):
        """Load A and B tiles using cp.async for unaligned memory access."""
        alias BM = tile_shape[0]
        alias BN = tile_shape[1]
        alias BK = tile_shape[2]

        alias num_full_k_iters = ceildiv(num_k_iters, pipeline_stages)
        alias num_remaining_k_iters = num_k_iters % pipeline_stages

        alias num_threads_per_row = BK // vector_size
        alias thread_layout = Layout.row_major(
            WARPGROUP_SIZE // num_threads_per_row, num_threads_per_row
        )

        @always_inline
        @parameter
        fn producer_loop[
            num_pipeline_stages_to_unroll: Int,
        ](k_iter: Int):
            @parameter
            for j in range(num_pipeline_stages_to_unroll):
                var write_idx = ring_buffer.producer_wait_for_empty[
                    pipeline_stages,
                    0,  # expected_bytes
                ](write_state)

                # FIXME: Without coalesce this crashes inside IntTuple
                var a_smem_tile = a_smem_iter.next(write_idx)[].coalesce()
                ScatterGather.load_tile[
                    thread_layout,
                    swizzle_mode,
                    vector_size,
                ](
                    a,
                    a_smem_tile,
                    Int(block_idx_m),
                    k_iter * pipeline_stages + j,
                )

                var b_smem_tile = b_smem_iter.next(write_idx)[].coalesce()
                ScatterGather.load_tile[
                    thread_layout,
                    swizzle_mode,
                    vector_size,
                ](
                    b,
                    b_smem_tile,
                    Int(block_idx_n),
                    k_iter * pipeline_stages + j,
                )

                ring_buffer.producer_signal_full_and_arrive[pipeline_stages](
                    write_state
                )

        @parameter
        if num_remaining_k_iters == 0:

            @parameter
            for k_iter in range(num_full_k_iters):
                producer_loop[pipeline_stages](k_iter)
        else:

            @parameter
            for k_iter in range(num_full_k_iters - 1):
                producer_loop[pipeline_stages](k_iter)
            producer_loop[num_remaining_k_iters](num_full_k_iters - 1)

    @staticmethod
    @always_inline
    fn finalize_kernel():
        """Common finalization for all kernel variants."""

        @parameter
        if pdl_level >= PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

        # Ensure SMEM destruction doesn't happen
        @parameter
        if Self.cluster_size > 1:
            cluster_sync()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        lut_ptr: DeviceBuffer[DType.uint32],
    ):
        Self.validate_constraints()

        alias K = b_layout.shape[1].value()
        alias simd_size = simd_width_of[c_type]()

        alias use_cluster = Self.cluster_size > 1
        var block_idx_swizzle: IndexList[2, element_type = DType.uint32]

        @parameter
        if not use_cluster:

            @parameter
            if hilbert_swizzle:
                # a 32-bit (UInt32) value that encodes a block's Hilbert-swizzled coordinates as
                # upper 16 bits = y, lower 16 bits = x
                var linear = UInt32(block_idx.y * grid_dim.x + block_idx.x)
                var packed = lut_ptr.unsafe_ptr()[linear]
                var new_x = packed & 0xFFFF
                var new_y = packed >> 16
                block_idx_swizzle = Index[dtype = DType.uint32](new_x, new_y)
            else:
                block_idx_swizzle = block_swizzle(
                    Index[dtype = DType.uint32](block_idx.x, block_idx.y),
                    Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
                )
        else:
            block_idx_swizzle = Index[dtype = DType.uint32](
                block_idx.x, block_idx.y
            )

        var wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        # Initialize Shared Memory
        var smem = Self.SMem()

        var warp_group_idx, warp_group_thread_idx = divmod(
            thread_idx.x, UInt(WARPGROUP_SIZE)
        )

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        @parameter
        if (
            pdl_level > PDLLevel.OFF
            and pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
        ):
            wait_on_dependent_grids()

        var lane_predicate = elect_one_sync()
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

        # Create local ring buffer
        var ring_buffer = RingBuffer[
            num_pipeline_stages,
            Self.num_consumer,
            Self.cluster_size,
        ](smem.full_mbar, smem.empty_mbar)
        ring_buffer.init_barriers(thread_idx.x)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        alias num_k_iters = ceildiv(K, Self.BK)

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            alias num_regs = 24 if Self.num_consumer <= 2 else 32
            warpgroup_reg_dealloc[num_regs]()

            if warp_id == 0 and lane_predicate:
                var m_coord = block_idx_swizzle[1] * Self.BM
                var n_coord = block_idx_swizzle[0] * Self.BN
                with PipelineState[num_pipeline_stages]() as write_state:
                    Self.async_load_AB_tma[
                        tile_shape=block_tile_shape,
                        cluster_dims=cluster_shape,
                        use_partitioned_multicast=partitioned_multicast,
                        num_k_iters=num_k_iters,
                    ](
                        a_tma_op,
                        b_tma_op,
                        smem.a_tiles,
                        smem.b_tiles,
                        UInt(m_coord),
                        UInt(n_coord),
                        0,
                        rank_n,
                        rank_m,
                        ring_buffer,
                        write_state,
                    )
        else:
            warpgroup_reg_alloc[Self.num_regs()]()

            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            ring_buffer.arrive_empty_barriers(warp_group_thread_idx)

            with PipelineState[num_pipeline_stages]() as read_state:
                Self.consumer_main_loop[num_k_iters=num_k_iters](
                    final_c_reg_tile,
                    c_reg_tile,
                    smem.a_tiles,
                    smem.b_tiles,
                    ring_buffer,
                    read_state,
                    wgmma_op,
                    UInt(local_warp_group_idx),
                    UInt(warp_group_thread_idx),
                )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            warp_specialized_gemm_output[
                c_tile_shape = Index(Self.BM, Self.BN),
                c_swizzle=c_swizzle,
                wgmma_shape=wgmma_shape,
                num_consumer = Self.num_consumer,
                use_tma_store=use_tma_store,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            ](
                c_tma_op,
                c,
                smem.c_tile,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                UInt(thread_idx.x - UInt(WARPGROUP_SIZE)),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_persistent[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
        grid_shape: IndexList[2],
        schedule: MatmulSchedule,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        problem_shape: IndexList[3],
    ):
        Self.validate_constraints()

        alias K = b_layout.shape[1].value()
        alias N = b_layout.shape[0].value()
        alias M = a_layout.shape[0].value()
        alias simd_size = simd_width_of[c_type]()

        var scheduler = TileScheduler[
            Index(M, N, K), block_tile_shape, grid_shape, schedule=schedule
        ](problem_shape)

        alias use_cluster = Self.cluster_size > 1

        var wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        var work_info = scheduler.get_current_work_info()

        # Initialize Shared Memory
        var smem = Self.SMem()

        var warp_group_idx, warp_group_thread_idx = divmod(
            thread_idx.x, UInt(WARPGROUP_SIZE)
        )

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        @parameter
        if (
            pdl_level > PDLLevel.OFF
            and pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
        ):
            wait_on_dependent_grids()

        var lane_predicate = elect_one_sync()
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

        # Create local ring buffer
        var ring_buffer = RingBuffer[
            num_pipeline_stages,
            Self.num_consumer,
            Self.cluster_size,
        ](smem.full_mbar, smem.empty_mbar)
        ring_buffer.init_barriers(thread_idx.x)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        alias num_k_iters = ceildiv(K, Self.BK)

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            alias num_regs = 24 if Self.num_consumer <= 2 else 32
            warpgroup_reg_dealloc[num_regs]()
            if warp_id == 0 and lane_predicate:
                with PipelineState[num_pipeline_stages]() as write_state:
                    while work_info.is_valid():
                        var m_coord = work_info.m
                        var n_coord = work_info.n

                        Self.async_load_AB_tma[
                            tile_shape=block_tile_shape,
                            cluster_dims=cluster_shape,
                            use_partitioned_multicast=partitioned_multicast,
                            num_k_iters=num_k_iters,
                        ](
                            a_tma_op,
                            b_tma_op,
                            smem.a_tiles,
                            smem.b_tiles,
                            UInt(m_coord),
                            UInt(n_coord),
                            0,
                            rank_n,
                            rank_m,
                            ring_buffer,
                            write_state,
                        )
                        work_info = scheduler.fetch_next_work()
        else:
            warpgroup_reg_alloc[Self.num_regs()]()

            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            @parameter
            if a_type is DType.float8_e4m3fn:
                _ = final_c_reg_tile.fill(0.0)
            else:
                _ = c_reg_tile.fill(0.0)

            ring_buffer.arrive_empty_barriers(warp_group_thread_idx)

            with PipelineState[num_pipeline_stages]() as read_state:
                while work_info.is_valid():
                    Self.consumer_main_loop[num_k_iters=num_k_iters](
                        final_c_reg_tile,
                        c_reg_tile,
                        smem.a_tiles,
                        smem.b_tiles,
                        ring_buffer,
                        read_state,
                        wgmma_op,
                        UInt(local_warp_group_idx),
                        UInt(warp_group_thread_idx),
                    )

                    var block_y = UInt(ceildiv(work_info.m, Self.BM))
                    var block_x = UInt(ceildiv(work_info.n, Self.BN))
                    var output_reg_tile = (
                        final_c_reg_tile if a_type
                        is DType.float8_e4m3fn else c_reg_tile
                    )

                    warp_specialized_gemm_output[
                        c_tile_shape = Index(Self.BM, Self.BN),
                        c_swizzle=c_swizzle,
                        wgmma_shape=wgmma_shape,
                        num_consumer = Self.num_consumer,
                        use_tma_store=use_tma_store,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    ](
                        c_tma_op,
                        c,
                        smem.c_tile,
                        output_reg_tile,
                        UInt(warp_group_thread_idx),
                        UInt(local_warp_group_idx),
                        thread_idx.x - UInt(WARPGROUP_SIZE),
                        Int(block_y),
                        Int(block_x),
                    )
                    work_info = scheduler.fetch_next_work()

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_unaligned[
        c_desc_layout: Layout,
        c_tma_layout: Layout,
        pipeline_stages: Int = 7,
    ](
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    ):
        """Kernel using cp.async for A/B loading when K alignment doesn't meet TMA requirements.
        """
        Self.validate_constraints()

        alias K = b_layout.shape[1].value()
        alias simd_size = simd_width_of[c_type]()

        alias use_cluster = Self.cluster_size > 1
        var block_idx_swizzle: IndexList[2, element_type = DType.uint32]

        @parameter
        if not use_cluster:
            block_idx_swizzle = block_swizzle(
                Index[dtype = DType.uint32](block_idx.x, block_idx.y),
                Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
            )
        else:
            block_idx_swizzle = Index[dtype = DType.uint32](
                block_idx.x, block_idx.y
            )

        var wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        # Initialize Shared Memory using SharedMemoryManager
        var smem = HopperMatmulSM90Kernel_SMem[
            a_type,
            Self.a_smem_layout,
            b_type,
            Self.b_smem_layout,
            c_type,
            c_smem_layout,
            pipeline_stages,
            Self.num_consumer,
            Self.cluster_size,
        ]()

        alias k_align = find_K_alignment_upto_16B(K * size_of[a_type]())

        var warp_group_idx, warp_group_thread_idx = divmod(
            thread_idx.x, UInt(WARPGROUP_SIZE)
        )
        alias num_k_iters = ceildiv(K, Self.BK)

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        @parameter
        if (
            pdl_level > PDLLevel.OFF
            and pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
        ):
            wait_on_dependent_grids()

        var lane_predicate = elect_one_sync()

        # Create local ring buffer for cp.async
        var ring_buffer = RingBuffer[
            pipeline_stages,
            Self.num_consumer,
            Self.cluster_size,
        ](smem.full_mbar, smem.empty_mbar)
        ring_buffer.init_barriers_cpasync(thread_idx.x)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            warpgroup_reg_dealloc[32]()

            var m_coord = block_idx_swizzle[1]
            var n_coord = block_idx_swizzle[0]

            with PipelineState[pipeline_stages]() as write_state:
                Self.async_load_AB_cpasync[
                    pipeline_stages=pipeline_stages,
                    swizzle_mode = Self.a_swizzle,
                    vector_size = k_align // size_of[Self.a_type](),
                    num_k_iters=num_k_iters,
                    tile_shape = Index(Self.BM, Self.BN, Self.BK),
                ](
                    a,
                    b,
                    UInt(block_idx_swizzle[1]),
                    UInt(block_idx_swizzle[0]),
                    smem.a_tiles,
                    smem.b_tiles,
                    ring_buffer,
                    write_state,
                )

        else:
            constrained[
                Self.num_consumer <= 2, "Only support 1 or 2 consumer"
            ]()
            warpgroup_reg_alloc[232]()

            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            ring_buffer.arrive_empty_barriers(warp_group_thread_idx)

            with PipelineState[pipeline_stages]() as read_state:
                Self.consumer_main_loop[num_k_iters=num_k_iters](
                    final_c_reg_tile,
                    c_reg_tile,
                    smem.a_tiles,
                    smem.b_tiles,
                    ring_buffer,
                    read_state,
                    wgmma_op,
                    UInt(local_warp_group_idx),
                    UInt(warp_group_thread_idx),
                )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            warp_specialized_gemm_output[
                c_tile_shape = Index(Self.BM, Self.BN),
                c_swizzle=c_swizzle,
                wgmma_shape=wgmma_shape,
                num_consumer = Self.num_consumer,
                use_tma_store=use_tma_store,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            ](
                c_tma_op,
                c,
                smem.c_tile,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                thread_idx.x - UInt(WARPGROUP_SIZE),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_splitk[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
        splits: Int,
        raster_order: RasterOrder,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        workspace_buffer: NDBuffer[Self.accum_type, 3, MutableAnyOrigin],
        locks_ptr: UnsafePointer[NoneType],
        problem_shape: IndexList[3],
    ):
        """Split-K variant of the kernel for better load balancing on small problems.
        """

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])

        alias K = b_layout.shape[1].value()
        alias N = b_layout.shape[0].value()
        alias M = a_layout.shape[0].value()
        alias NUM_TILES = ceildiv(M, Self.BM) * ceildiv(N, Self.BN)

        Self.validate_constraints()
        constrained[a_type == b_type, "A and B must have the same type"]()

        @parameter
        if splits > 1:
            # This static constraint only needs to apply if splitk is actually used
            constrained[(K % Self.BK) == 0, "K must be divisible by BK"]()

        alias workspace_layout = Layout.row_major(NUM_TILES, Self.BM, Self.BN)
        var reduction_workspace = LayoutTensor(
            workspace_buffer.data,
            RuntimeLayout[workspace_layout].row_major(
                IndexList[3](NUM_TILES, Self.BM, Self.BN)
            ),
        )

        alias num_k_iters = K // Self.BK
        alias simd_size = simd_width_of[c_type]()

        alias use_cluster = Self.cluster_size > 1

        var wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        # Initialize Shared Memory
        var smem = Self.SMem()

        var warp_group_idx, warp_group_thread_idx = divmod(
            thread_idx.x, UInt(WARPGROUP_SIZE)
        )

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        @parameter
        if (
            pdl_level > PDLLevel.OFF
            and pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
        ):
            wait_on_dependent_grids()

        var lane_predicate = elect_one_sync()
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

        # Create local ring buffer
        var ring_buffer = RingBuffer[
            num_pipeline_stages,
            Self.num_consumer,
            Self.cluster_size,
        ](smem.full_mbar, smem.empty_mbar)
        ring_buffer.init_barriers(thread_idx.x)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        var scheduler = SplitKTileScheduler[
            Index(N, K),
            block_tile_shape,
            splits,
            Self.num_consumer,
            num_pipeline_stages,
            Index(CLUSTER_M, CLUSTER_N),
            raster_order,
        ](
            problem_shape,
            Index(rank_m, rank_n),
            locks_ptr,
        )

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            alias num_regs = 24 if Self.num_consumer <= 2 else 32
            var work_tile_info = scheduler.initial_work_tile_info()

            warpgroup_reg_dealloc[num_regs]()
            if warp_id == 0 and lane_predicate:
                with PipelineState[num_pipeline_stages]() as write_state:
                    while work_tile_info.is_valid():
                        var m_coord = work_tile_info.m * Self.BM
                        var n_coord = work_tile_info.n * Self.BN

                        alias work_k_tile_count = num_k_iters // splits
                        var work_k_tile_start = work_tile_info.get_k_start()

                        Self.async_load_AB_tma[
                            tile_shape=block_tile_shape,
                            cluster_dims=cluster_shape,
                            use_partitioned_multicast=partitioned_multicast,
                            num_k_iters=work_k_tile_count,
                        ](
                            a_tma_op,
                            b_tma_op,
                            smem.a_tiles,
                            smem.b_tiles,
                            UInt(m_coord),
                            UInt(n_coord),
                            UInt(work_k_tile_start),
                            rank_n,
                            rank_m,
                            ring_buffer,
                            write_state,
                        )

                        # Get next work tile
                        work_tile_info = scheduler.fetch_next_work(
                            work_tile_info
                        )
        else:
            warpgroup_reg_alloc[Self.num_regs()]()

            var work_tile_info = scheduler.initial_work_tile_info()
            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            @parameter
            if a_type is DType.float8_e4m3fn:
                _ = final_c_reg_tile.fill(0.0)
            else:
                _ = c_reg_tile.fill(0.0)

            ring_buffer.arrive_empty_barriers(warp_group_thread_idx)

            with PipelineState[num_pipeline_stages]() as read_state:
                while work_tile_info.is_valid():
                    alias work_k_tile_count = num_k_iters // splits

                    Self.consumer_main_loop[num_k_iters=work_k_tile_count](
                        final_c_reg_tile,
                        c_reg_tile,
                        smem.a_tiles,
                        smem.b_tiles,
                        ring_buffer,
                        read_state,
                        wgmma_op,
                        UInt(local_warp_group_idx),
                        UInt(warp_group_thread_idx),
                    )

                    var output_reg_tile = (
                        final_c_reg_tile if a_type
                        is DType.float8_e4m3fn else c_reg_tile
                    )

                    scheduler.reduction(
                        reduction_workspace,
                        output_reg_tile,
                        work_tile_info,
                        Self.num_consumer,
                        local_warp_group_idx,
                    )

                    # check if this is the reduction tile
                    if scheduler.is_last_split(work_tile_info):
                        var block_y = UInt(work_tile_info.m)
                        var block_x = UInt(work_tile_info.n)

                        warp_specialized_gemm_output[
                            c_tile_shape = Index(Self.BM, Self.BN),
                            c_swizzle=c_swizzle,
                            wgmma_shape=wgmma_shape,
                            num_consumer = Self.num_consumer,
                            use_tma_store=use_tma_store,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                        ](
                            c_tma_op,
                            c,
                            smem.c_tile,
                            output_reg_tile,
                            UInt(warp_group_thread_idx),
                            UInt(local_warp_group_idx),
                            thread_idx.x - UInt(WARPGROUP_SIZE),
                            Int(block_y),
                            Int(block_x),
                        )

                    # Get next work tile
                    work_tile_info = scheduler.fetch_next_work(work_tile_info)

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_grouped[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_smem_layout, c_desc_layout],
        a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    ):
        """Grouped matmul variant for MoE (Mixture of Experts) models.

        This variant handles multiple experts where each expert processes a subset of tokens.
        The a_offsets array indicates token boundaries for each expert.
        """
        Self.validate_constraints()

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])

        alias K = b_layout.shape[1].value()
        alias N = c_layout.shape[1].value()
        alias simd_size = simd_width_of[c_type]()

        alias use_cluster = Self.cluster_size > 1
        var block_idx_swizzle = block_swizzle(
            Index[dtype = DType.uint32](block_idx.x, block_idx.y),
            Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
        ) if not use_cluster else Index[dtype = DType.uint32](
            block_idx.x, block_idx.y
        )

        # The block may be OOB because we create blocks based the maximum
        # number of tokens per expert.
        M = a_offsets[Int(block_idx.z + 1)] - a_offsets[Int(block_idx.z)]
        if UInt32(block_idx_swizzle[1] * Self.BM) >= M:
            return

        a_start_row = a_offsets[Int(block_idx.z)]

        expert = expert_ids[Int(block_idx.z)]
        # We use -1 to indicate that the block is not active for LoRA use cases.
        # but we still need to zero out the output for this case.
        skip_matmul = expert < 0

        b_start_row = expert * N

        wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        # Initialize Shared Memory
        var smem = Self.SMem()

        var warp_group_idx, warp_group_thread_idx = divmod(
            thread_idx.x, UInt(WARPGROUP_SIZE)
        )

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        var lane_predicate = elect_one_sync()
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

        # Create local ring buffer
        var ring_buffer = RingBuffer[
            num_pipeline_stages,
            Self.num_consumer,
            Self.cluster_size,
        ](smem.full_mbar, smem.empty_mbar)
        ring_buffer.init_barriers(thread_idx.x)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        alias num_k_iters = K // Self.BK

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            alias num_regs = 24 if Self.num_consumer <= 2 else 32
            warpgroup_reg_dealloc[num_regs]()

            if (
                warp_group_thread_idx == 0
                and lane_predicate
                and not skip_matmul
            ):
                var m_coord = block_idx.y * UInt(
                    Self.BM
                ) if CLUSTER_N > 1 else UInt(Int(a_start_row)) + UInt(
                    block_idx_swizzle[1]
                ) * UInt(
                    Self.BM
                )

                var n_coord = block_idx.x * UInt(
                    Self.BN
                ) if CLUSTER_M > 1 else UInt(Int(b_start_row)) + UInt(
                    block_idx_swizzle[0]
                ) * UInt(
                    Self.BN
                )

                with PipelineState[num_pipeline_stages]() as write_state:
                    Self.async_load_AB_tma[
                        tile_shape=block_tile_shape,
                        cluster_dims=cluster_shape,
                        use_partitioned_multicast=partitioned_multicast,
                        num_k_iters=num_k_iters,
                    ](
                        a_tma_op,
                        b_tma_op,
                        smem.a_tiles,
                        smem.b_tiles,
                        UInt(m_coord),
                        UInt(n_coord),
                        0,
                        rank_n,
                        rank_m,
                        ring_buffer,
                        write_state,
                    )

        else:
            warpgroup_reg_alloc[Self.num_regs()]()

            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            _ = c_reg_tile.fill(0.0)

            if not skip_matmul:
                with PipelineState[num_pipeline_stages]() as read_state:
                    ring_buffer.arrive_empty_barriers(warp_group_thread_idx)

                    Self.consumer_main_loop[num_k_iters=num_k_iters](
                        final_c_reg_tile,
                        c_reg_tile,
                        smem.a_tiles,
                        smem.b_tiles,
                        ring_buffer,
                        read_state,
                        wgmma_op,
                        UInt(local_warp_group_idx),
                        UInt(warp_group_thread_idx),
                    )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            # C layout for current expert
            alias c_gmem_layout = Layout(
                IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1)
            )
            alias c_gmem_type = LayoutTensor[
                c_type,
                c_gmem_layout,
                MutableAnyOrigin,
                layout_int_type = DType.int32,
                address_space = AddressSpace.GENERIC,
            ]

            # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
            var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
                Index(M, N), Index(N, 1)
            )

            var c_by_expert = c_gmem_type(
                c.ptr + a_start_row * N, c_gmem_runtime_layout
            )

            @always_inline
            @parameter
            fn elementwise_epilogue_fn_wrapper[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                @parameter
                if elementwise_lambda_fn:
                    alias elementwise_epilogue = elementwise_lambda_fn.value()
                    var batch_idx = IndexList[2](
                        Int(a_start_row + idx[0]), idx[1]
                    )
                    elementwise_epilogue(batch_idx, val)

            warp_specialized_gemm_output[
                c_tile_shape = Index(Self.BM, Self.BN),
                c_swizzle=c_swizzle,
                wgmma_shape=wgmma_shape,
                num_consumer = Self.num_consumer,
                use_tma_store=use_tma_store,
                elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
                    elementwise_epilogue_fn_wrapper
                ) if elementwise_lambda_fn else None,
            ](
                c_tma_op,
                c_by_expert,
                smem.c_tile,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                thread_idx.x - UInt(WARPGROUP_SIZE),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @always_inline
    fn consumer_main_loop[
        accum_type: DType,
        c_reg_layout: Layout,
        a_smem_layout: Layout,
        b_smem_layout: Layout,
        pipeline_stages: Int,
        num_consumer: Int, //,
        /,
        *,
        num_k_iters: Int,
    ](
        final_c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
        c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
        a_smem_iter: SMemTileIterType[a_type, a_smem_layout, _, _],
        b_smem_iter: SMemTileIterType[b_type, b_smem_layout, _, _],
        ring_buffer: RingBuffer[
            pipeline_stages, num_consumer, Self.cluster_size
        ],
        mut read_state: PipelineState[pipeline_stages],
        wgmma_op: TensorCoreAsync[
            accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle,
            b_swizzle,
            transpose_b,
        ],
        local_warp_group_idx: UInt,
        warp_group_thread_idx: UInt,
    ):
        @parameter
        if a_type is DType.float8_e4m3fn:
            _ = final_c_reg_tile.fill(0.0)
        else:
            _ = c_reg_tile.fill(0.0)

        var fp8_promotion_iter = 0

        alias num_full_k_iters = ceildiv(num_k_iters, pipeline_stages)
        alias num_remaining_k_iters = num_k_iters % pipeline_stages

        # `num_pipeline_stages_to_unroll` determines how many pipeline stages should be unroll in the consumer loop;
        # if num_k_iters % pipeline_stages != 0 then for the last loop, we only unroll (num_k_iters % pipeline_stages) pipeline stages
        @always_inline
        @parameter
        fn consumer_loop[
            num_pipeline_stages_to_unroll: Int,
        ](k_iter: Int):
            @parameter
            for j in range(num_pipeline_stages_to_unroll):
                var read_idx = ring_buffer.consumer_wait_for_full[
                    pipeline_stages
                ](read_state)

                var a_smem_tile = a_smem_iter.next(read_idx)[]
                var b_smem_tile = b_smem_iter.next(read_idx)[]

                warpgroup_fence(c_reg_tile)
                wgmma_op.arrive()
                alias scale_c = 0 if a_type is DType.float8_e4m3fn else 1
                wgmma_op.wgmma[num_consumer, scale_c=scale_c](
                    a_smem_tile,
                    b_smem_tile,
                    c_reg_tile,
                    Int(local_warp_group_idx),
                )
                wgmma_op.commit_group()
                warpgroup_fence(c_reg_tile)
                wgmma_op.wait_group()

                ring_buffer.consumer_signal_empty(
                    read_idx, warp_group_thread_idx
                )

                @parameter
                if a_type is DType.float8_e4m3fn:
                    fp8_promotion_iter += 1
                    if fp8_promotion_iter == promotion_frequency:
                        Self.promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)
                        fp8_promotion_iter -= promotion_frequency

                ring_buffer.consumer_advance[pipeline_stages](read_state)

        @parameter
        if num_remaining_k_iters == 0:
            for k_iter in range(num_full_k_iters):
                consumer_loop[pipeline_stages](k_iter)
        else:
            for k_iter in range(num_full_k_iters - 1):
                consumer_loop[pipeline_stages](k_iter)
            consumer_loop[num_remaining_k_iters](num_full_k_iters - 1)

        # Final promotion for fp8 data type if num_k_iters % promotion_frequency != 0
        @parameter
        if a_type is DType.float8_e4m3fn:
            if fp8_promotion_iter != 0:
                Self.promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)

    @staticmethod
    @always_inline
    fn promote_to_cuda_cores[
        accum_type: DType, layout: Layout
    ](
        c_reg_tile: RegTileType[accum_type, layout, _],
        final_c_reg_tile: RegTileType[accum_type, layout, _],
    ):
        constrained[
            accum_type in (DType.float32, DType.float16),
            "Only support fp32 and fp16 data type in CUDA Core promotion",
        ]()
        constrained[
            len(layout) == 2, "Only support 2D layout in CUDA Core promotion"
        ]()

        alias num_mma = c_reg_tile.layout.shape[0].value()
        alias c_frag_size = c_reg_tile.layout.shape[1].value()

        # CUDA Core promotion for fp8 data type increases the precision of the result.
        # This is a workaround used by cutlass and cuBLAS to ensure the results are as precise as possible.
        @parameter
        for mma_id in range(num_mma):

            @parameter
            for i in range(c_frag_size):
                final_c_reg_tile[mma_id, i] = rebind[Scalar[accum_type]](
                    final_c_reg_tile[mma_id, i]
                ) + rebind[Scalar[accum_type]](c_reg_tile[mma_id, i])


# Helper functions for st.matrix output processing
# These functions break down the complex handle_stmatrix_output function into
# smaller, more manageable pieces for better readability and maintainability.


@always_inline
fn _calculate_output_bounds[
    c_layout: Layout, //, BM: Int, BN: Int
](
    c: LayoutTensor[_, c_layout, MutableAnyOrigin, *_, **_],
    block_y: Int,
    block_x: Int,
) -> (UInt32, UInt32):
    """Calculate the output bounds for the current thread block."""
    alias N = c_layout.shape[1].value()
    var M = c.dim[0]()
    var M_bound = min(UInt32((block_y + 1) * BM), UInt32(M))
    var N_bound = min(UInt32((block_x + 1) * BN), UInt32(N))
    return M_bound, N_bound


@always_inline
fn _store_fragments_to_smem[
    c_type: DType,
    c_tile_layout: Layout,
    accum_type: DType,
    c_reg_layout: Layout, //,
    wgmma_shape: IndexList[3],
    BN: Int,
    WG_BM: Int,
    WG_BN: Int,
    TMA_BN: Int,
    num_m_mmas: Int,
    sub_wg_bn_id: Int,
    use_x2_for_last_iter: Bool,
    num_consumer: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    st_matrix_swizzle: Swizzle,
    st_matrix_rt_layout: RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ],
):
    """Store fragments from registers to shared memory using st.matrix instructions.
    """

    @parameter
    for tma_n in range(WG_BN // TMA_BN):

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for i in range(TMA_BN // 16):
                var st_matrix_args = RuntimeTuple[
                    IntTuple(
                        UNKNOWN_VALUE,
                        IntTuple(
                            i,
                            m_mma,
                            UNKNOWN_VALUE,
                        ),
                    )
                ](
                    Int(warp_group_thread_idx),
                    i,
                    m_mma,
                    Int(local_warp_group_idx),
                )
                var n_mma = (
                    i + tma_n * (TMA_BN // 16) + sub_wg_bn_id * (WG_BN // 16)
                )
                var offset = c_tile.ptr.offset(
                    st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    + WG_BM * TMA_BN * tma_n
                )

                @always_inline
                @parameter
                fn st_matrix_frag[x2: Bool = False]():
                    alias xn = 1 if x2 else 2
                    alias xf = 2 if x2 else 1
                    var c_frag = c_reg_tile.tile[1, 4 * xf](
                        m_mma,
                        xn * n_mma,
                    )
                    var d_reg = c_frag.load[4 * xf](0, 0).cast[DType.bfloat16]()
                    var d_reg_f32_packed = bitcast[DType.float32, 2 * xf](d_reg)
                    st_matrix[simd_width = 2 * xf](offset, d_reg_f32_packed)

                @parameter
                if use_x2_for_last_iter:
                    st_matrix_frag()
                else:
                    st_matrix_frag[True]()


@always_inline
fn _apply_compute_lambda[
    c_type: DType,
    c_tile_layout: Layout, //,
    elementwise_compute_lambda_fn: elementwise_compute_lambda_type,
    N: Int,
    WG_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_wg_coord_m: Int,
    c_gmem_wg_coord_n: Int,
    local_thread_idx: UInt,
    M_bound: UInt32,
    N_bound: UInt32,
):
    """Apply the compute lambda function to the output data."""
    alias compute_lambda = elementwise_compute_lambda_fn
    alias st_matrix_vec_swizzle = make_ldmatrix_swizzle[c_type, WG_BN]()
    alias thread_layout = Layout.row_major(
        num_consumer_threads // (WG_BN // simd_size),
        WG_BN // simd_size,
    )

    var c_gmem_frag_with_offsets = c_gmem_wg_tile.vectorize[
        1, simd_size
    ]().distribute_with_offset[thread_layout](local_thread_idx)
    var c_gmem_frag = c_gmem_frag_with_offsets[0]
    var c_gmem_offset_coords = c_gmem_frag_with_offsets[1]
    var coord_m = c_gmem_wg_coord_m + c_gmem_offset_coords[0]
    var coord_n = c_gmem_wg_coord_n + c_gmem_offset_coords[1] * simd_size

    var c_smem_frag = c_tile.vectorize[1, simd_size]().distribute[
        thread_layout, swizzle=st_matrix_vec_swizzle
    ](local_thread_idx)

    alias num_stores_per_thread = c_gmem_frag.layout.size()

    @parameter
    for i in range(num_stores_per_thread):
        alias src_idx = c_smem_frag.layout(i)
        alias dst_idx = c_gmem_frag.layout(i)
        alias dst_m_offset = dst_idx // N
        alias dst_n_offset = dst_idx % N
        var m = UInt32(coord_m + dst_m_offset)
        var n = UInt32(coord_n + dst_n_offset)
        alias alignment = align_of[SIMD[c_type, simd_size]]()

        if m < M_bound and n < N_bound:
            var reg_val = compute_lambda[alignment=alignment](
                (Int(m), Int(n)),
                c_smem_frag[i, 0],
            )
            c_smem_frag[i, 0] = reg_val


@always_inline
fn _apply_epilogue_lambda[
    c_type: DType,
    c_tile_layout: Layout, //,
    elementwise_lambda_fn: elementwise_epilogue_type,
    N: Int,
    WG_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_wg_coord_m: Int,
    c_gmem_wg_coord_n: Int,
    local_thread_idx: UInt,
    M_bound: UInt32,
    N_bound: UInt32,
):
    """Apply the epilogue lambda function to the output data."""
    alias epilogue = elementwise_lambda_fn
    alias st_matrix_vec_swizzle = make_ldmatrix_swizzle[c_type, WG_BN]()
    alias thread_layout = Layout.row_major(
        num_consumer_threads // (WG_BN // simd_size),
        WG_BN // simd_size,
    )

    var c_gmem_frag_with_offsets = c_gmem_wg_tile.vectorize[
        1, simd_size
    ]().distribute_with_offset[thread_layout](local_thread_idx)
    var c_gmem_frag = c_gmem_frag_with_offsets[0]
    var c_gmem_offset_coords = c_gmem_frag_with_offsets[1]
    var coord_m = c_gmem_wg_coord_m + c_gmem_offset_coords[0]
    var coord_n = c_gmem_wg_coord_n + c_gmem_offset_coords[1] * simd_size

    var c_smem_frag = c_tile.vectorize[1, simd_size]().distribute[
        thread_layout, swizzle=st_matrix_vec_swizzle
    ](local_thread_idx)

    alias num_stores_per_thread = c_gmem_frag.layout.size()

    @parameter
    for i in range(num_stores_per_thread):
        alias src_idx = c_smem_frag.layout(i)
        alias dst_idx = c_gmem_frag.layout(i)
        alias dst_m_offset = dst_idx // N
        alias dst_n_offset = dst_idx % N
        var m = UInt32(coord_m + dst_m_offset)
        var n = UInt32(coord_n + dst_n_offset)
        alias alignment = align_of[SIMD[c_type, simd_size]]()

        if m < M_bound and n < N_bound:
            epilogue[alignment=alignment](
                (Int(m), Int(n)),
                c_smem_frag[i, 0].cast[c_type](),
            )


@always_inline
fn _perform_output_store[
    c_type: DType,
    c_tma_layout: Layout,
    c_desc_layout: Layout,
    c_tile_layout: Layout, //,
    use_tma_store: Bool,
    use_x2_for_last_iter: Bool,
    BM: Int,
    BN: Int,
    WG_BM: Int,
    WG_BN: Int,
    TMA_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
    st_matrix_swizzle: Swizzle,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    local_thread_idx: UInt,
    block_x: Int,
    block_y: Int,
    sub_wg_bn_id: Int,
):
    """Perform the final output store operation."""

    @parameter
    if use_tma_store and not use_x2_for_last_iter:
        fence_async_view_proxy()

        if local_thread_idx < UInt(WG_BN // TMA_BN):
            var smem_offset = c_tile.ptr.offset(
                WG_BM * TMA_BN * local_thread_idx
            )
            var c_tma_tile = SMemTileType[
                c_type,
                c_tma_layout,
                alignment=128,
            ](smem_offset)

            c_tma_op.async_store(
                c_tma_tile,
                (
                    UInt(
                        block_x * BN
                        + sub_wg_bn_id * WG_BN
                        + local_thread_idx * UInt(TMA_BN)
                    ),
                    UInt(block_y * BM),
                ),
            )
            c_tma_op.commit_group()
            c_tma_op.wait_group()

    else:
        alias thread_layout = Layout.row_major(
            num_consumer_threads // (WG_BN // simd_size),
            WG_BN // simd_size,
        )

        @parameter
        if use_x2_for_last_iter:
            var masked_c_tile = c_tile.slice[
                Slice(0, Int(c_tile.layout.shape[0])),
                Slice(0, Int(c_tile.layout.shape[1]) // 2),
            ]()
            var masked_c_gmem_wg_tile = c_gmem_wg_tile.slice[
                Slice(0, Int(c_gmem_wg_tile.layout.shape[0])),
                Slice(0, Int(c_gmem_wg_tile.layout.shape[1]) // 2),
            ]()
            alias thread_layout_v2 = Layout.row_major(
                num_consumer_threads // (WG_BN // simd_size),
                WG_BN // simd_size // 2,
            )
            if local_thread_idx < UInt(num_consumer_threads // 2):
                copy_sram_to_dram[
                    thread_layout=thread_layout_v2,
                    swizzle=st_matrix_swizzle,
                ](
                    masked_c_gmem_wg_tile.vectorize[1, simd_size](),
                    masked_c_tile.vectorize[1, simd_size](),
                )
        else:
            copy_sram_to_dram[
                thread_layout=thread_layout,
                swizzle=st_matrix_swizzle,
            ](
                c_gmem_wg_tile.vectorize[1, simd_size](),
                c_tile.vectorize[1, simd_size](),
            )


@always_inline
fn handle_stmatrix_output[
    c_type: DType,
    c_tma_layout: Layout,
    c_desc_layout: Layout,
    accum_type: DType,
    c_layout: Layout,
    c_tile_layout: Layout,
    c_reg_layout: Layout, //,
    wgmma_shape: IndexList[3],
    num_consumer: Int,
    use_tma_store: Bool,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type],
    elementwise_compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type],
    BM: Int,
    BN: Int,
    num_m_mmas: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    c_gmem_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_corner_coords: c.CornerCoordsType,
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Handle output using st.matrix instructions for optimized bf16 output."""
    # Calculate output bounds
    var M_bound, N_bound = _calculate_output_bounds[BM, BN](c, block_y, block_x)

    # Layout dimensions
    alias WG_BM = c_tile.layout.shape[0].value()
    alias WG_BN = c_tile.layout.shape[1].value()
    alias TMA_BN = c_tma_op.layout.shape[1].value() if use_tma_store else WG_BN

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    alias st_matrix_swizzle = make_ldmatrix_swizzle[
        c_type, TMA_BN, log2_floor(16 // size_of[c_type]())
    ]()

    var lane = lane_id()
    alias num_sub_wg_bn_iters = ceildiv(BN, WG_BN)
    alias last_iter = BN // WG_BN
    alias needs_x2 = BN % WG_BN != 0
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE

    constrained[
        needs_x2 == (c_frag_size % 4 == 0 and c_frag_size % 8 != 0),
        "stmatrix and wgmma register count conflict: needs_x2 = "
        + String(needs_x2)
        + " c_frag_size ="
        + String(c_frag_size),
    ]()

    @parameter
    for sub_wg_bn_id in range(num_sub_wg_bn_iters):
        alias use_x2_for_last_iter = needs_x2 and sub_wg_bn_id == last_iter

        # Store fragments to shared memory
        _store_fragments_to_smem[
            wgmma_shape,
            BN,
            WG_BM,
            WG_BN,
            TMA_BN,
            num_m_mmas,
            sub_wg_bn_id,
            use_x2_for_last_iter,
            num_consumer,
        ](
            c_tile,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            st_matrix_swizzle,
            st_matrix_rt_layout,
        )

        named_barrier[num_consumer_threads](10)

        alias thread_layout = Layout.row_major(
            num_consumer_threads // (WG_BN // simd_size),
            WG_BN // simd_size,
        )

        var c_gmem_wg_tile_crd_idx = c_gmem_tile.tile_with_offset[BM, WG_BN](
            0, sub_wg_bn_id
        )
        var c_gmem_wg_tile = c_gmem_wg_tile_crd_idx[0]
        var c_gmem_wg_coords = rebind[c.CornerCoordsType](
            c_gmem_wg_tile_crd_idx[1]
        )
        c_gmem_wg_coords = c_gmem_wg_coords + c_gmem_corner_coords

        # Handle compute lambda
        @parameter
        if elementwise_compute_lambda_fn:
            _apply_compute_lambda[
                elementwise_compute_lambda_fn.value(),
                c_layout.shape[1].value(),
                WG_BN,
                num_consumer_threads,
                simd_size,
            ](
                c_tile,
                c_gmem_wg_tile,
                c_gmem_wg_coords[0],
                c_gmem_wg_coords[1],
                local_thread_idx,
                M_bound,
                N_bound,
            )
            named_barrier[num_consumer_threads](10)

        # Handle epilogue lambda
        @parameter
        if elementwise_lambda_fn:
            _apply_epilogue_lambda[
                elementwise_lambda_fn.value(),
                c_layout.shape[1].value(),
                WG_BN,
                num_consumer_threads,
                simd_size,
            ](
                c_tile,
                c_gmem_wg_tile,
                c_gmem_wg_coords[0],
                c_gmem_wg_coords[1],
                local_thread_idx,
                M_bound,
                N_bound,
            )

        else:
            # Regular store path
            _perform_output_store[
                use_tma_store,
                use_x2_for_last_iter,
                BM,
                BN,
                WG_BM,
                WG_BN,
                TMA_BN,
                num_consumer_threads,
                simd_size,
                st_matrix_swizzle,
            ](
                c_tma_op,
                c_tile,
                c_gmem_wg_tile,
                local_thread_idx,
                block_x,
                block_y,
                sub_wg_bn_id,
            )

        named_barrier[num_consumer_threads](10)


@always_inline
fn warp_specialized_gemm_output[
    c_type: DType,
    accum_type: DType,
    c_layout: Layout,
    c_tile_layout: Layout,
    c_tma_layout: Layout,
    c_reg_layout: Layout,
    c_desc_layout: Layout, //,
    /,
    *,
    c_tile_shape: IndexList[2],
    c_swizzle: TensorMapSwizzle,
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_tile: SMemTileType[
        c_type,
        c_tile_layout,
        alignment=128,
    ],
    c_reg_tile: RegTileType[
        accum_type,
        c_reg_layout,
        _,
    ],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Write matrix multiplication output from registers to global memory.

    This function handles three main paths:
    1. Using st.matrix instructions for optimized bf16 output
    2. Aligned case when N is divisible by tile size
    3. General case with bounds checking
    """
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]
    alias num_consumer_threads = num_consumer * WARPGROUP_SIZE
    alias simd_size = simd_width_of[c_type]()
    alias BM = c_tile_shape[0]
    alias BN = c_tile_shape[1]

    # Use helper to compute tile coordinates
    var tile_crd_idx = c.tile_with_offset[BM, BN](block_y, block_x)
    var c_gmem_tile = tile_crd_idx[0]
    var c_gmem_corner_coords = tile_crd_idx[1]
    var c_gmem_offset = tile_crd_idx[2]
    var c_gmem_split_crd_idx = c_gmem_tile.tile_with_offset[
        BM // num_consumer, BN
    ](Int(local_warp_group_idx), 0)
    var c_gmem_split = c_gmem_split_crd_idx[0]

    alias c_coord_type = c.CornerCoordsType
    var warp_id = warp_group_thread_idx // UInt(WARP_SIZE)

    alias N = c_layout.shape[1].value()
    alias is_N_multiple_of_16B = N * size_of[c_type]() % 16 == 0
    alias WG_BM = c_tile.layout.shape[0].value()
    alias WG_BN = c_tile.layout.shape[1].value()
    alias TMA_BN = c_tma_op.layout.shape[1].value() if use_tma_store else WG_BN
    # fmt: off
    alias use_stmatrix = accum_type is DType.float32 \
            and c_type is DType.bfloat16 \
            and c_frag_size % 4 == 0 \
            and BM % wgmma_shape[0] == 0 \
            and WG_BN % 16 == 0 \
            and num_consumer <= 2 \
            and BN == wgmma_shape[1] \
            and BM == WG_BM \
            and N * size_of[c_type]() % 16 == 0
    # fmt: on

    @parameter
    if use_stmatrix:
        # Use the extracted st_matrix handler
        handle_stmatrix_output[
            wgmma_shape,
            num_consumer,
            use_tma_store,
            elementwise_lambda_fn,
            elementwise_compute_lambda_fn,
            BM,
            BN,
            num_m_mmas,
            num_consumer_threads,
            simd_size,
        ](
            c_tma_op,
            c,
            c_tile,
            c_reg_tile,
            c_gmem_tile,
            c_gmem_corner_coords,
            warp_group_thread_idx,
            local_warp_group_idx,
            local_thread_idx,
            block_y,
            block_x,
        )

    # N dim doesn't need bound check.
    elif N % BN == 0:

        @parameter
        if (
            elementwise_lambda_fn is not None
            or elementwise_compute_lambda_fn is not None
        ):
            # Output dimensions in global memory.
            alias N = c_layout.shape[1].value()
            var M: UInt = UInt(c.dim[0]())

            lane = lane_id()

            c_frag_vec2 = c_reg_tile.vectorize[1, 2]()

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    var warp_tile_crd_idx = c_gmem_split.tile_with_offset[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](Int(m_mma * 4 + warp_id), n_mma)
                    var warp_tile = warp_tile_crd_idx[0]
                    var warp_tile_coords = rebind[c_coord_type](
                        warp_tile_crd_idx[1]
                    )
                    warp_tile_coords = (
                        warp_tile_coords
                        + c_gmem_corner_coords
                        + rebind[c_coord_type](c_gmem_split_crd_idx[1])
                    )
                    warp_tile_offset = (
                        warp_tile_crd_idx[2]
                        + c_gmem_offset
                        + c_gmem_split_crd_idx[2]
                    )

                    gmem_frag_with_offsets = warp_tile.vectorize[
                        1, 2
                    ]().distribute_with_offset[Layout.row_major(8, 4)](lane)
                    gmem_frag = gmem_frag_with_offsets[0]
                    gmem_offset_coords = rebind[c_coord_type](
                        gmem_frag_with_offsets[1]
                    )
                    gmem_offset_coords[1] *= 2
                    coords = gmem_offset_coords + warp_tile_coords
                    c_block_offset = (
                        gmem_frag_with_offsets[2] + warp_tile_offset
                    )

                    alias num_vecs = gmem_frag.layout.size()

                    @parameter
                    for i in range(num_vecs):
                        alias dst_idx = gmem_frag.layout(i)
                        alias dst_m_offset = dst_idx // N
                        alias dst_n_offset = dst_idx % N
                        var m = Int(coords[0] + dst_m_offset)
                        var n = Int(coords[1] + dst_n_offset)

                        alias alignment = align_of[SIMD[c_type, 2]]()
                        if m < Int(M) and n < N:

                            @parameter
                            if elementwise_lambda_fn:
                                alias epilogue = elementwise_lambda_fn.value()
                                epilogue[alignment=alignment](
                                    (m, n),
                                    c_frag_vec2[mma_id, i].cast[c_type](),
                                )
                            else:
                                alias compute_lambda = elementwise_compute_lambda_fn.value()
                                var reg_val = compute_lambda[
                                    alignment=alignment
                                ](
                                    (m, n),
                                    c_frag_vec2[mma_id, i].cast[c_type](),
                                )
                                c.ptr.store[alignment=alignment](
                                    c_block_offset + dst_idx, reg_val
                                )

        else:

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # (m_mma, n_mma) is coordinates for a warp group's tile.
                    # A warp group is 4x1 warps.
                    warp_tile = c_gmem_split.tile[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](Int(m_mma * 4 + warp_id), n_mma)

                    # Tile at (mma_id, 0) is a long vector containing all fragments
                    # for this warp.
                    c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

                    # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
                    # elementwise. This pattern repeats to fill the warp tile.
                    copy_local_to_dram[Layout.row_major(8, 4)](
                        warp_tile.vectorize[1, 2](),
                        c_frag.vectorize[1, 2](),
                    )

    else:
        constrained[
            elementwise_compute_lambda_fn is None,
            (
                "compute_lambda_fn is not supported when N % BN != 0 and TMA is"
                " not used"
            ),
        ]()

        # Lane's coordinate is in 8x4 warp.
        var lane_coord0 = UInt32(lane_id() // 4)
        var lane_coord1 = UInt32(lane_id() % 4)

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                # (m_mma, n_mma) is coordinates for a warp group's tile.
                # A warp group is 4x1 warps.
                var warp_tile_crd_idx = c_gmem_split.tile_with_offset[
                    wgmma_shape[0] // 4, wgmma_shape[1]
                ](Int(m_mma * 4 + warp_id), n_mma, 0, 0)
                var warp_tile = warp_tile_crd_idx[0]
                var warp_tile_coords = rebind[c_coord_type](
                    warp_tile_crd_idx[1]
                )
                warp_tile_coords = (
                    warp_tile_coords
                    + c_gmem_corner_coords
                    + c_coord_type(
                        wgmma_shape[0] * Int(local_warp_group_idx), 0
                    )
                )

                # A single fragment matrix is the 8x8 output shard by a warp.
                alias num_n_frag_mat = wgmma_shape[1] // 8
                alias num_m_frag_mat = wgmma_shape[0] // 4 // 8

                @parameter
                for n_frag in range(num_n_frag_mat):

                    @parameter
                    for m_frag in range(num_m_frag_mat):
                        alias frag_mat_id = n_frag * num_m_frag_mat + m_frag

                        var frag_mat_gmem = warp_tile.tile[8, 8](m_frag, n_frag)

                        var bound0 = UInt32(
                            frag_mat_gmem.runtime_layout.shape[0].value[0]
                        )
                        var bound1 = UInt32(
                            frag_mat_gmem.runtime_layout.shape[1].value[0]
                        )

                        @parameter
                        for i in range(2):
                            if (
                                lane_coord0 < bound0
                                and lane_coord1 * 2 + i < bound1
                            ):

                                @parameter
                                if elementwise_lambda_fn:
                                    alias epilogue = elementwise_lambda_fn.value()

                                    var frag_mat_coords = (
                                        warp_tile_coords
                                        + c_coord_type(
                                            Int(m_frag * 8), Int(n_frag * 8)
                                        )
                                    )
                                    var frag_coords = (
                                        frag_mat_coords
                                        + c_coord_type(
                                            Int(lane_coord0),
                                            Int(lane_coord1 * 2 + i),
                                        )
                                    )

                                    epilogue[
                                        alignment = align_of[Scalar[c_type]]()
                                    ](
                                        Index(frag_coords[0], frag_coords[1]),
                                        c_reg_tile[
                                            mma_id, frag_mat_id * 2 + i
                                        ].cast[c_type](),
                                    )

                                else:
                                    frag_mat_gmem[
                                        Int(lane_coord0),
                                        Int(lane_coord1 * 2 + i),
                                    ] = rebind[frag_mat_gmem.element_type](
                                        c_reg_tile[
                                            mma_id, frag_mat_id * 2 + i
                                        ].cast[c_type]()
                                    )
