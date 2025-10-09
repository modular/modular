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
from buffer.dimlist import DimList
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.device_context import DeviceBuffer
from gpu.host.info import H100
from gpu.id import (
    block_dim,
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
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from gpu.mma import st_matrix
from gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import make_ldmatrix_swizzle
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
    create_tma_tile,
    create_tma_tile_template,
)
from logger import Logger
from memory import bitcast, stack_allocation
from memory.pointer import _GPUAddressSpace
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import (
    MatmulConfig,
    block_swizzle,
    get_hilbert_lut_with_cache,
)
from ..tile_scheduler import MatmulSchedule, TileScheduler
from .loadop import async_load_AB
from ....structuring import NVIDIASharedMemoryManager as SharedMemoryManager
from ....structuring import SMemTileType, SMemTileIterType, RegTileType


# Shared memory structure for Hopper SM90 kernel
struct HopperMatmulSM90Kernel_SMem[
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    c_type: DType,
    c_layout: Layout,
    num_pipeline_stages: Int,
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
    var c_smem: Self.CTileType.T

    # Pipeline barriers (full and empty)
    var full_mbar: Self.PipelineBarrierType.T
    var empty_mbar: Self.PipelineBarrierType.T

    fn __init__(out self):
        var smem_mgr = Self.SMM()
        # Initialize tile iterators
        self.a_tiles = Self.ATileIterType.build(smem_mgr)
        self.b_tiles = Self.BTileIterType.build(smem_mgr)
        self.c_smem = Self.CTileType.build(smem_mgr)
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

    alias SMem = HopperMatmulSM90Kernel_SMem[
        a_type,
        Self.a_smem_layout,
        b_type,
        Self.b_smem_layout,
        c_type,
        c_smem_layout,
        num_pipeline_stages,
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
        constrained[transpose_b, "Only support transposed B in layout"]()

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])
        alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

        alias K = b_layout.shape[1].value()
        alias simd_size = simd_width_of[c_type]()

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

        alias use_cluster = cluster_size[cluster_shape]() > 1
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

        wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        # Initialize Shared Memory using SharedMemoryManager
        var smem = Self.SMem()

        full = smem.full_mbar
        empty = smem.empty_mbar

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
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

            @parameter
            for i in range(num_pipeline_stages):
                full[i].init(1)
                empty[i].init(Self.num_consumer * CLUSTER_SIZE)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if cluster_size[cluster_shape]() > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            alias num_regs = 24 if Self.num_consumer <= 2 else 32
            warpgroup_reg_dealloc[num_regs]()

            if warp_id == 0 and lane_predicate:
                var write_pipeline_states = PipelineState[num_pipeline_stages]()

                var m_coord = block_idx_swizzle[1] * Self.BM
                var n_coord = block_idx_swizzle[0] * Self.BN

                async_load_AB[
                    block_tile_shape=block_tile_shape,
                    cluster_shape=cluster_shape,
                    partitioned_multicast=partitioned_multicast,
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
                    write_pipeline_states,
                    empty,
                    full,
                )

        else:
            warpgroup_reg_alloc[Self.num_regs()]()

            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            @parameter
            for i in range(num_pipeline_stages):

                @parameter
                if cluster_size[cluster_shape]() > 1:
                    if warp_group_thread_idx < CLUSTER_SIZE:
                        _ = empty[i].arrive_cluster(warp_group_thread_idx)
                else:
                    if warp_group_thread_idx == 0:
                        _ = empty[i].arrive()

            var read_pipeline_states = PipelineState[num_pipeline_stages]()

            consumer_main_loop[
                cluster_shape=cluster_shape,
                promotion_frequency=promotion_frequency,
                num_consumer = Self.num_consumer,
                num_k_iters=num_k_iters,
            ](
                final_c_reg_tile,
                c_reg_tile,
                smem.a_tiles,
                smem.b_tiles,
                read_pipeline_states,
                full,
                empty,
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
                smem.c_smem,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                UInt(thread_idx.x - UInt(WARPGROUP_SIZE)),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        @parameter
        if pdl_level >= PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

        # TO ensure SMEM destruction doesn't happen
        @parameter
        if cluster_size[cluster_shape]() > 1:
            cluster_sync()

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
        constrained[transpose_b, "Only support transposed B in layout"]()

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])
        alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

        alias K = b_layout.shape[1].value()
        alias N = b_layout.shape[0].value()
        alias M = a_layout.shape[0].value()
        alias simd_size = simd_width_of[c_type]()

        var scheduler = TileScheduler[
            Index(M, N, K), block_tile_shape, grid_shape, schedule=schedule
        ](problem_shape)

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

        alias use_cluster = cluster_size[cluster_shape]() > 1

        wgmma_op = TensorCoreAsync[
            Self.accum_type,
            a_type,
            b_type,
            wgmma_shape,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=transpose_b,
        ]()

        var work_info = scheduler.get_current_work_info()

        # Initialize Shared Memory using SharedMemoryManager
        var smem = Self.SMem()

        full = smem.full_mbar
        empty = smem.empty_mbar

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
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

            @parameter
            for i in range(num_pipeline_stages):
                full[i].init(1)
                empty[i].init(Self.num_consumer * CLUSTER_SIZE)

        @parameter
        if cluster_size[cluster_shape]() > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            alias num_regs = 24 if Self.num_consumer <= 2 else 32
            warpgroup_reg_dealloc[num_regs]()
            if warp_id == 0 and lane_predicate:
                var write_pipeline_states = PipelineState[num_pipeline_stages]()

                while work_info.is_valid():
                    var m_coord = work_info.m
                    var n_coord = work_info.n

                    async_load_AB[
                        block_tile_shape=block_tile_shape,
                        cluster_shape=cluster_shape,
                        partitioned_multicast=partitioned_multicast,
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
                        write_pipeline_states,
                        empty,
                        full,
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

            @parameter
            for i in range(num_pipeline_stages):

                @parameter
                if cluster_size[cluster_shape]() > 1:
                    if warp_group_thread_idx < CLUSTER_SIZE:
                        _ = empty[i].arrive_cluster(warp_group_thread_idx)
                else:
                    if warp_group_thread_idx == 0:
                        _ = empty[i].arrive()

            var read_pipeline_states = PipelineState[num_pipeline_stages]()

            while work_info.is_valid():
                consumer_main_loop[
                    cluster_shape=cluster_shape,
                    promotion_frequency=promotion_frequency,
                    num_consumer = Self.num_consumer,
                    num_k_iters=num_k_iters,
                ](
                    final_c_reg_tile,
                    c_reg_tile,
                    smem.a_tiles,
                    smem.b_tiles,
                    read_pipeline_states,
                    full,
                    empty,
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
                    smem.c_smem,
                    output_reg_tile,
                    UInt(warp_group_thread_idx),
                    UInt(local_warp_group_idx),
                    UInt(thread_idx.x - UInt(WARPGROUP_SIZE)),
                    Int(block_y),
                    Int(block_x),
                )
                work_info = scheduler.fetch_next_work()

        @parameter
        if pdl_level >= PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

        # TO ensure SMEM destruction doesn't happen
        @parameter
        if cluster_size[cluster_shape]() > 1:
            cluster_sync()

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
        constrained[transpose_b, "Only support transposed B in layout"]()

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])
        alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

        alias K = b_layout.shape[1].value()
        alias simd_size = simd_width_of[c_type]()

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

        alias use_cluster = cluster_size[cluster_shape]() > 1
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

        wgmma_op = TensorCoreAsync[
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
        ]()

        alias k_align = find_K_alignment_upto_16B(K * size_of[a_type]())
        full = smem.full_mbar
        empty = smem.empty_mbar

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

        if thread_idx.x == 0:

            @parameter
            for i in range(pipeline_stages):
                full[i].init(WARPGROUP_SIZE)
                empty[i].init(Self.num_consumer * CLUSTER_SIZE)

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the cluster
        @parameter
        if cluster_size[cluster_shape]() > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

        var warp_id = get_warp_id()
        if warp_group_idx == 0:
            warpgroup_reg_dealloc[32]()

            var write_pipeline_states = PipelineState[pipeline_stages]()

            var m_coord = block_idx_swizzle[1]
            var n_coord = block_idx_swizzle[0]

            async_load_AB[
                swizzle_mode=a_swizzle,
                cp_size = k_align // size_of[a_type](),
                num_k_iters=num_k_iters,
                block_tile_shape=block_tile_shape,
            ](
                a,
                b,
                UInt(block_idx_swizzle[1]),
                UInt(block_idx_swizzle[0]),
                smem.a_tiles,
                smem.b_tiles,
                write_pipeline_states,
                empty,
                full,
            )

        else:
            constrained[
                Self.num_consumer <= 2, "Only support 1 or 2 consumer"
            ]()
            warpgroup_reg_alloc[232]()

            var local_warp_group_idx = warp_group_idx - 1

            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            @parameter
            for i in range(pipeline_stages):

                @parameter
                if cluster_size[cluster_shape]() > 1:
                    if warp_group_thread_idx < CLUSTER_SIZE:
                        _ = empty[i].arrive_cluster(warp_group_thread_idx)
                else:
                    if warp_group_thread_idx == 0:
                        _ = empty[i].arrive()

            var read_pipeline_states = PipelineState[pipeline_stages]()

            consumer_main_loop[
                cluster_shape=cluster_shape,
                promotion_frequency=promotion_frequency,
                num_consumer = Self.num_consumer,
                num_k_iters=num_k_iters,
            ](
                final_c_reg_tile,
                c_reg_tile,
                smem.a_tiles,
                smem.b_tiles,
                read_pipeline_states,
                full,
                empty,
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
                smem.c_smem,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                UInt(thread_idx.x - UInt(WARPGROUP_SIZE)),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        @parameter
        if pdl_level >= PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

        # TO ensure SMEM destruction doesn't happen
        @parameter
        if cluster_size[cluster_shape]() > 1:
            cluster_sync()


@always_inline
fn cluster_size[cluster_shape: StaticTuple[Int32, 3]]() -> Int32:
    var size: Int32 = 1

    @parameter
    for i in range(3):
        size *= cluster_shape[i]
    return size


@always_inline
fn promote_to_cuda_cores[
    accum_type: DType, layout: Layout
](
    c_reg_tile: LayoutTensor[
        accum_type,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    final_c_reg_tile: LayoutTensor[
        accum_type,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
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

    # CUD Core promotion for fp8 data type increases the precision of the result.
    # This is a workaround used by cutlass and cuBLAS to ensure the results are as precise as possible.
    @parameter
    for mma_id in range(num_mma):

        @parameter
        for i in range(c_frag_size):
            final_c_reg_tile[mma_id, i] = rebind[Scalar[accum_type]](
                final_c_reg_tile[mma_id, i]
            ) + rebind[Scalar[accum_type]](c_reg_tile[mma_id, i])


@always_inline
fn consumer_main_loop[
    accum_type: DType,
    a_type: DType,
    b_type: DType,
    c_reg_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    wgmma_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    num_k_iters: Int,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    promotion_frequency: Int = 1,
    num_consumer: Int = 1,
](
    final_c_reg_tile: LayoutTensor[
        accum_type,
        c_reg_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    c_reg_tile: LayoutTensor[
        accum_type,
        c_reg_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        *_, **_,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        *_, **_,
    ],
    mut read_pipeline_states: PipelineState[pipeline_stages],
    full: UnsafePointer[SharedMemBarrier, address_space = AddressSpace.SHARED],
    empty: UnsafePointer[SharedMemBarrier, address_space = AddressSpace.SHARED],
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
    alias CLUSTER_SIZE = cluster_size[cluster_shape]()

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
            var read_idx = read_pipeline_states.index()

            full[read_idx].wait(read_pipeline_states.phase())

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

            @parameter
            if cluster_size[cluster_shape]() > 1:
                if warp_group_thread_idx < UInt(CLUSTER_SIZE):
                    _ = empty[read_idx].arrive_cluster(warp_group_thread_idx)
            else:
                if warp_group_thread_idx == 0:
                    _ = empty[read_idx].arrive()

            @parameter
            if a_type is DType.float8_e4m3fn:
                fp8_promotion_iter += 1
                if fp8_promotion_iter == promotion_frequency:
                    promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)
                    fp8_promotion_iter -= promotion_frequency

            read_pipeline_states.step()

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
            promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)


@always_inline
fn warp_specialized_gemm_output[
    c_type: DType,
    accum_type: DType,
    c_layout: Layout,
    c_smem_layout: Layout,
    c_tma_layout: Layout,
    c_reg_layout: Layout,
    c_desc_layout: Layout,
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
    c_smem_tile: LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        *_, **_,
    ],
    c_reg_tile: LayoutTensor[
        accum_type,
        c_reg_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]
    alias num_consumer_threads = num_consumer * WARPGROUP_SIZE
    alias simd_size = simd_width_of[c_type]()
    alias BM = c_tile_shape[0]
    alias BN = c_tile_shape[1]

    var tile_crd_idx = c.tile_with_offset[BM, BN](block_y, block_x)
    var c_gmem_tile = tile_crd_idx[0]
    var c_gmem_corner_coords = tile_crd_idx[1]
    var c_gmem_offset = tile_crd_idx[2]
    var c_gmem_split_crd_idx = c_gmem_tile.tile_with_offset[
        BM // num_consumer, BN
    ](Int(local_warp_group_idx), 0)
    var c_gmem_split = c_gmem_split_crd_idx[0]
    alias c_coord_type = __type_of(c_gmem_corner_coords)
    var warp_id = warp_group_thread_idx // UInt(WARP_SIZE)

    alias N = c_layout.shape[1].value()
    alias is_N_multiple_of_16B = N * size_of[c_type]() % 16 == 0
    alias WG_BM = c_smem_tile.layout.shape[0].value()
    alias WG_BN = c_smem_tile.layout.shape[1].value()
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
        # Output dimensions in global memory.
        var M = c.dim[0]()
        var M_bound = min(UInt32((block_y + 1) * BM), UInt32(M))
        var N_bound = min(UInt32((block_x + 1) * BN), UInt32(N))
        var st_matrix_rt_layout = RuntimeLayout[
            st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ]()
        alias st_matrix_swizzle = make_ldmatrix_swizzle[
            c_type, TMA_BN, log2_floor(16 // size_of[c_type]())
        ]()

        alias N = c_layout.shape[1].value()
        lane = lane_id()
        alias num_sub_wg_bn_iters = ceildiv(BN, WG_BN)
        alias last_iter = BN // WG_BN
        alias needs_x2 = BN % WG_BN != 0
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
                        var offset = c_smem_tile.ptr.offset(
                            st_matrix_swizzle(
                                st_matrix_rt_layout(st_matrix_args)
                            )
                            + WG_BM * TMA_BN * tma_n
                        )

                        @parameter
                        if use_x2_for_last_iter:
                            var c_frag = c_reg_tile.tile[1, 4](
                                m_mma,
                                2
                                * (
                                    i
                                    + tma_n * (TMA_BN // 16)
                                    + sub_wg_bn_id * (WG_BN // 16)
                                ),
                            )

                            var d_reg_x2 = c_frag.load[4](0, 0).cast[
                                DType.bfloat16
                            ]()
                            var d_reg_f32_packed_x2 = bitcast[DType.float32, 2](
                                d_reg_x2
                            )
                            st_matrix[simd_width=2](offset, d_reg_f32_packed_x2)
                        else:
                            var c_frag = c_reg_tile.tile[1, 8](
                                m_mma,
                                i
                                + tma_n * (TMA_BN // 16)
                                + sub_wg_bn_id * (WG_BN // 16),
                            )

                            var d_reg = c_frag.load[8](0, 0).cast[
                                DType.bfloat16
                            ]()
                            var d_reg_f32_packed = bitcast[DType.float32, 4](
                                d_reg
                            )
                            st_matrix[simd_width=4](offset, d_reg_f32_packed)

            named_barrier[num_consumer_threads,](10)

            alias thread_layout = Layout.row_major(
                num_consumer_threads // (WG_BN // simd_size),
                WG_BN // simd_size,
            )

            var c_gmem_wg_tile_crd_idx = c_gmem_tile.tile_with_offset[
                BM, WG_BN
            ](0, sub_wg_bn_id)
            var c_gmem_wg_tile = c_gmem_wg_tile_crd_idx[0]
            var c_gmem_wg_coords = rebind[c_coord_type](
                c_gmem_wg_tile_crd_idx[1]
            )
            c_gmem_wg_coords = c_gmem_wg_coords + c_gmem_corner_coords

            @parameter
            if elementwise_compute_lambda_fn:
                alias compute_lambda = elementwise_compute_lambda_fn.value()
                alias st_matrix_vec_swizzle = make_ldmatrix_swizzle[
                    c_type, WG_BN
                ]()

                var c_gmem_frag_with_offsets = c_gmem_wg_tile.vectorize[
                    1, simd_size
                ]().distribute_with_offset[thread_layout](local_thread_idx)
                var c_gmem_frag = c_gmem_frag_with_offsets[0]
                var c_gmem_offset_coords = rebind[c_coord_type](
                    c_gmem_frag_with_offsets[1]
                )
                c_gmem_offset_coords[1] *= simd_size
                var coords = c_gmem_offset_coords + c_gmem_wg_coords

                var c_smem_frag = c_smem_tile.vectorize[
                    1, simd_size
                ]().distribute[thread_layout, swizzle=st_matrix_vec_swizzle](
                    local_thread_idx
                )

                alias num_stores_per_thread = c_gmem_frag.layout.size()

                @parameter
                for i in range(num_stores_per_thread):
                    alias src_idx = c_smem_frag.layout(i)
                    alias dst_idx = c_gmem_frag.layout(i)
                    alias dst_m_offset = dst_idx // N
                    alias dst_n_offset = dst_idx % N
                    var m = UInt32(coords[0] + dst_m_offset)
                    var n = UInt32(coords[1] + dst_n_offset)
                    alias alignment = align_of[SIMD[c_type, simd_size]]()

                    if m < M_bound and n < N_bound:
                        var reg_val = compute_lambda[alignment=alignment](
                            (Int(m), Int(n)),
                            c_smem_frag[i, 0],
                        )
                        c_smem_frag[i, 0] = reg_val

                named_barrier[num_consumer_threads](10)

            @parameter
            if elementwise_lambda_fn:
                alias epilogue = elementwise_lambda_fn.value()
                alias st_matrix_vec_swizzle = make_ldmatrix_swizzle[
                    c_type, WG_BN
                ]()

                var c_gmem_frag_with_offsets = c_gmem_wg_tile.vectorize[
                    1, simd_size
                ]().distribute_with_offset[thread_layout](local_thread_idx)
                var c_gmem_frag = c_gmem_frag_with_offsets[0]
                var c_gmem_offset_coords = rebind[c_coord_type](
                    c_gmem_frag_with_offsets[1]
                )
                c_gmem_offset_coords[1] *= simd_size
                var coords = c_gmem_offset_coords + c_gmem_wg_coords

                var c_smem_frag = c_smem_tile.vectorize[
                    1, simd_size
                ]().distribute[thread_layout, swizzle=st_matrix_vec_swizzle](
                    local_thread_idx
                )

                alias num_stores_per_thread = c_gmem_frag.layout.size()

                @parameter
                for i in range(num_stores_per_thread):
                    alias src_idx = c_smem_frag.layout(i)
                    alias dst_idx = c_gmem_frag.layout(i)

                    alias dst_m_offset = dst_idx // N
                    alias dst_n_offset = dst_idx % N
                    var m = UInt32(coords[0] + dst_m_offset)
                    var n = UInt32(coords[1] + dst_n_offset)

                    alias alignment = align_of[SIMD[c_type, simd_size]]()

                    if m < M_bound and n < N_bound:
                        epilogue[alignment=alignment](
                            (Int(m), Int(n)),
                            c_smem_frag[i, 0].cast[c_type](),
                        )

            else:

                @parameter
                if use_tma_store and not needs_x2:
                    fence_async_view_proxy()

                    if local_thread_idx < UInt(WG_BN // TMA_BN):
                        var smem_offset = c_smem_tile.ptr.offset(
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

                    @parameter
                    if use_x2_for_last_iter:
                        var masked_c_smem_tile = c_smem_tile.slice[
                            Slice(0, Int(c_smem_tile.layout.shape[0])),
                            Slice(0, Int(c_smem_tile.layout.shape[1]) // 2),
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
                                masked_c_smem_tile.vectorize[1, simd_size](),
                            )
                    else:
                        copy_sram_to_dram[
                            thread_layout=thread_layout,
                            swizzle=st_matrix_swizzle,
                        ](
                            c_gmem_wg_tile.vectorize[1, simd_size](),
                            c_smem_tile.vectorize[1, simd_size](),
                        )
            named_barrier[num_consumer_threads,](10)

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


fn _is_valid_cluster_shape[
    cluster_shape: IndexList[3]
](grid_shape: IndexList[2], num_tiles_n: Int) -> Bool:
    if num_tiles_n % cluster_shape[0] != 0:
        return False

    @parameter
    for i in range(2):
        if (
            grid_shape[i] < cluster_shape[i]
            or grid_shape[i] % cluster_shape[i] != 0
        ):
            return False

    return True


fn _get_grid_shape[
    cluster_shape: IndexList[3] = Index(1, 1, 1)
](num_tiles_n: Int) -> IndexList[2]:
    # Hardcode values on purpose until we move this inside tile scheduler
    # in a more robust way.
    alias h100_num_SMs = H100.sm_count
    num_blocks_n = min(num_tiles_n, h100_num_SMs)
    adjusted_grid_shape = Index(
        num_blocks_n,
        h100_num_SMs // num_blocks_n,
    )

    # A Naive heuristic to select grid shape based on number of tile in N.
    if num_tiles_n % 8 == 0 or not _is_valid_cluster_shape[cluster_shape](
        adjusted_grid_shape, num_tiles_n
    ):
        return Index(8, 16)

    return adjusted_grid_shape


fn _is_valid_grid_shape[
    grid_shape: IndexList[2], cluster_shape: IndexList[3]
](num_tiles_n: Int) -> Bool:
    constrained[
        grid_shape[0] * grid_shape[1] <= H100.sm_count,
        "Total grid size exceed number of SMs in H100.",
    ]()

    if not _is_valid_cluster_shape[cluster_shape](grid_shape, num_tiles_n):
        return False

    if grid_shape[0] <= num_tiles_n:
        return num_tiles_n % grid_shape[0] == 0

    return grid_shape[0] % num_tiles_n == 0


fn _get_c_smem_layout[
    block_tile_shape: IndexList[3],
    a_type: DType,
    b_type: DType,
    c_type: DType,
    num_pipeline_stages: Int,
]() -> Layout:
    alias BM = Int(block_tile_shape[0])
    alias BN = Int(block_tile_shape[1])
    alias BK = Int(block_tile_shape[2])

    alias WG_BM = BM
    alias MAX_WG_BN = 128

    alias available_smem_size = Int(
        H100.shared_memory_per_multiprocessor - 1024
    )
    alias pipeline_smem_size = Int(
        num_pipeline_stages
        * (
            BM * BK * size_of[a_type]()
            + BN * BK * size_of[b_type]()
            + (size_of[Int64]() * 2)
        )
    )

    alias available_c_smem_size = Int(available_smem_size - pipeline_smem_size)
    # We want the shared memory N to be at least 16 when using `stmatrix`
    # (c_type = bf16) because it would make TMA and masked copy from shared
    # memory to global memory easier.
    alias MIN_WG_BN = 16 if size_of[c_type]() == 2 else BN // 4

    @parameter
    if available_smem_size > (
        pipeline_smem_size + (WG_BM * MIN_WG_BN * size_of[c_type]())
    ):

        fn _get_max_wg_bn() capturing -> Int:
            var WG_BN = MAX_WG_BN
            while (
                available_c_smem_size < WG_BM * WG_BN * size_of[c_type]()
                or BN % WG_BN != 0
            ) and WG_BN > MIN_WG_BN:
                WG_BN //= 2
            return WG_BN

        alias max_wg_bn = _get_max_wg_bn()
        return Layout.row_major(WG_BM, max_wg_bn)
    else:
        constrained[
            False,
            "There is not enough SMEM to fit the pipeline yet alone the"
            " output tile!"
            + " available_smem_size: "
            + String(available_smem_size)
            + " pipeline_smem_size + WG_BM * MIN_WG_BN * size_of[c_type](): "
            + String(
                pipeline_smem_size + WG_BM * MIN_WG_BN * size_of[c_type]()
            ),
        ]()
        return Layout.row_major(0, 0)


fn warp_specialize_gemm_with_multicasting[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    hilbert_swizzle: Bool = False,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    alias N_static = c_shape.get[1]()
    alias K_static = a_shape.get[1]()
    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var K = a_device.dim[1]()

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    constrained[
        (a_type == b_type is DType.float8_e4m3fn)
        or (a_type == b_type and a_type in (DType.bfloat16, DType.float32)),
        "Unsupported input dtype",
    ]()

    constrained[
        a_type != DType.float8_e4m3fn or BK == 128,
        "BK must be 128 for fp8 data type for numerical accuracy correctness",
    ]()

    constrained[
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None,
        "Either the epilogue lambda or the compute lambda can be used",
    ]()

    constrained[
        BM > 64 or (BM == 64 and config.num_consumer == 1),
        "Only support 1 consumer for BM=64",
    ]()

    alias k_align = find_K_alignment_upto_16B(K_static * size_of[a_type]())
    constrained[
        k_align in (4, 8, 16), "H100 matmul K dim must be multiple of 4B"
    ]()

    var logger = Logger()

    logger.info("Executing Warp Specialized Gemm with Multicasting")
    logger.info("block_tile_shape:", config.block_tile_shape)
    logger.info("cluster_shape:", config.cluster_shape)
    logger.info("mma_shape:", config.mma_shape)

    @parameter
    if schedule == MatmulSchedule.NONE:
        pass
    elif schedule == MatmulSchedule.DS_SCHEDULER:
        constrained[
            grid_shape is not None,
            "Grid shape must be provided for DS scheduler",
        ]()
        alias ds_grid_shape = grid_shape.value()
        constrained[
            ds_grid_shape[0] <= H100.sm_count and ds_grid_shape[1] == 1,
            "Deepseek scheduler only accepts grid shape with 1 column",
        ]()

    elif grid_shape:
        constrained[
            _is_valid_grid_shape[grid_shape.value(), config.cluster_shape](
                ceildiv(N_static, BN)
            ),
            String(
                "grid shape:",
                grid_shape.value(),
                "is not compatible with cluster shape:",
                config.cluster_shape,
                "and static N:",
                N_static,
                sep=" ",
            ),
        ]()

    alias grid_shape_adjusted = grid_shape.value() if grid_shape else _get_grid_shape[
        config.cluster_shape
    ](
        ceildiv(N_static, BN)
    )

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        Int(config.num_pipeline_stages),
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(),
        c_smem_layout.shape[1].value() // config.num_consumer,
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # make sure TMA_BN = 64 -> 128B swizzle, 32 -> 64B swizzle and etc.
    alias c_swizzle = TensorMapSwizzle(
        min(log2_floor(c_smem_tile[1] // 8), 3)
    ) if use_tma_store else TensorMapSwizzle.SWIZZLE_NONE

    var c_tma_op = create_tma_tile_template[
        c_type,
        2,
        c_smem_tile,
        swizzle_mode=c_swizzle,
        __desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1]),
    ]()

    @parameter
    if use_tma_store:
        c_tma_op = create_tma_tile[
            c_smem_tile,
            swizzle_mode=c_swizzle,
            __desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1]),
        ](ctx, c)

    var lut_ptr = ctx.enqueue_create_buffer[DType.uint32](0)

    @parameter
    if hilbert_swizzle:
        var grid_x = ceildiv(N, BN)
        var grid_y = ceildiv(M, BM)
        lut_ptr = get_hilbert_lut_with_cache(ctx, grid_x, grid_y)

    alias num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE

    # Use the shared memory struct to calculate the size
    alias SMem = HopperMatmulSM90Kernel_SMem[
        a_type,
        tile_layout_k_major[a_type, BM, BK, a_swizzle](),
        b_type,
        tile_layout_k_major[b_type, BN, BK, b_swizzle](),
        c_type,
        c_smem_layout,
        Int(config.num_pipeline_stages),
    ]
    alias smem_size = SMem.storage_size()

    constrained[
        smem_size <= H100.shared_memory_per_multiprocessor - 1024,
        "requested SMEM size exceeds 227KB limit.",
    ]()

    alias matmul_kernel[hilbert_swizzle: Bool = False] = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        a.layout,
        b.layout,
        c.layout,
        c_smem_layout,
        config.block_tile_shape,
        config.mma_shape,
        cluster_shape,
        Int(config.num_pipeline_stages),
        Int(num_threads),
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast = config.partitioned_multicast,
        use_tma_store=use_tma_store,
        promotion_frequency=1,
        pdl_level = config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        hilbert_swizzle=hilbert_swizzle,
    ]

    # TMA requires stride (K) multiple of 16B. If not satisfied,
    # we need to use cp.async.ca for 4B and 8B access, and ld for
    # 2B or smaller access.
    # Note that K * size_of[a_type]() decides the 2nd row's alignment
    # and Nvidia requires access alignment by access size.
    # Dispatch kernel using TMA load when the stride is multiple of 16B.
    @parameter
    if k_align == 16:
        var a_tma_op = create_tma_tile[
            Index(
                BM // CLUSTER_N, BK
            ) if config.partitioned_multicast else Index(BM, BK),
            swizzle_mode=a_swizzle,
        ](ctx, a)

        var b_tma_op = create_tma_tile[
            Index(
                BN // CLUSTER_M, BK
            ) if config.partitioned_multicast else Index(BN, BK),
            swizzle_mode=b_swizzle,
        ](ctx, b)

        @parameter
        if schedule != MatmulSchedule.NONE:
            alias kernel = matmul_kernel[].run_persistent[
                a_tma_op.layout,
                b_tma_op.layout,
                c_tma_op.layout,
                a_tma_op.desc_layout,
                b_tma_op.desc_layout,
                c_tma_op.desc_layout,
                grid_shape=grid_shape_adjusted,
                schedule=schedule,
            ]

            ctx.enqueue_function[kernel](
                a_tma_op,
                b_tma_op,
                c_tma_op,
                c,
                Index(M, N, K),
                grid_dim=(grid_shape_adjusted[0], grid_shape_adjusted[1]),
                block_dim=(num_threads),
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_size
                ),
                attributes=pdl_launch_attributes(config.pdl_level()),
            )
        else:
            alias kernel = matmul_kernel[hilbert_swizzle=hilbert_swizzle].run[
                a_tma_op.layout,
                b_tma_op.layout,
                c_tma_op.layout,
                a_tma_op.desc_layout,
                b_tma_op.desc_layout,
                c_tma_op.desc_layout,
            ]

            ctx.enqueue_function[kernel](
                a_tma_op,
                b_tma_op,
                c_tma_op,
                a,
                b,
                c,
                lut_ptr,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(num_threads),
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_size
                ),
                attributes=pdl_launch_attributes(config.pdl_level()),
            )

    # Dispatch kernel using cp.async.ca when the stride is not multiple of 4B or 8B..
    else:
        alias kernel = matmul_kernel[].run_unaligned[
            c_tma_op.desc_layout,
            c_tma_op.layout,
            pipeline_stages = config.num_pipeline_stages,
        ]

        ctx.enqueue_function[kernel](
            c_tma_op,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(num_threads),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_size
            ),
            attributes=pdl_launch_attributes(config.pdl_level()),
        )
