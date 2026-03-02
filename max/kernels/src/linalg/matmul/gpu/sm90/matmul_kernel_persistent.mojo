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

from math import ceildiv

from gpu import MAX_THREADS_PER_BLOCK_METADATA
from gpu.globals import WARPGROUP_SIZE
from gpu import thread_idx, block_idx
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile
from gpu.memory import external_memory, AddressSpace

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.matmul.gpu.tile_scheduler import MatmulSchedule, TileScheduler
from linalg.matmul.gpu.sm100_structured.structured_kernels.pipeline import (
    ProducerConsumerPipeline,
)
from .matmul_kernels import find_K_alignment_upto_16B, HopperMatmulSM90Kernel
from .tile_loader import CPAsyncBarrierHandler


__extension HopperMatmulSM90Kernel:
    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(num_threads)
        ),
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
        c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
        problem_shape: IndexList[3],
    ):
        comptime K = b_layout.shape[1].value()
        comptime num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SMem]()[]

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        var pipeline = smem.create_pipeline()
        var barrier_handler = Self.TMABarrier(
            pipeline, Self.num_consumer, Self.cluster_size
        )

        # Create TileLoaderTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        comptime N = b_layout.shape[0].value()
        comptime M = a_layout.shape[0].value()
        var scheduler = TileScheduler[
            Index(M, N, K), block_tile_shape, grid_shape, schedule=schedule
        ](problem_shape)
        var work_info = scheduler.get_current_work_info()

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate:
                while work_info.is_valid():
                    var m_coord = work_info.m
                    var n_coord = work_info.n

                    Self.producer_main_loop_pipeline[num_k_iters=num_k_iters](
                        UInt(m_coord),
                        UInt(n_coord),
                        0,
                        a_loader,
                        b_loader,
                        barrier_handler,
                        pipeline,
                        smem.a_tiles(),
                        smem.b_tiles(),
                    )
                    work_info = scheduler.fetch_next_work()
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            Self.consumer_arrive_empty_barriers(warp_group_thread_idx, pipeline)

            while work_info.is_valid():
                Self.consumer_main_loop_pipeline[num_k_iters=num_k_iters](
                    wgmma_op,
                    local_warp_group_idx,
                    final_c_reg_tile,
                    c_reg_tile,
                    pipeline,
                    smem.a_tiles(),
                    smem.b_tiles(),
                    warp_group_thread_idx,
                )

                var block_y = UInt(ceildiv(work_info.m, UInt32(Self.BM)))
                var block_x = UInt(ceildiv(work_info.n, UInt32(Self.BN)))
                var output_reg_tile = (
                    final_c_reg_tile if a_type
                    == DType.float8_e4m3fn else c_reg_tile
                )

                Self.consumer_output(
                    c_tma_op,
                    c,
                    smem.c_tile(),
                    output_reg_tile,
                    warp_group_thread_idx,
                    local_warp_group_idx,
                    thread_idx.x - UInt(WARPGROUP_SIZE),
                    Int(block_y),
                    Int(block_x),
                )
                work_info = scheduler.fetch_next_work()

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(num_threads)
        ),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_unaligned[
        c_desc_layout: Layout,
        c_tma_layout: Layout,
    ](
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
        b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    ):
        """Kernel using cp.async for A/B loading when K alignment doesn't meet TMA requirements.
        """
        comptime K = b_layout.shape[1].value()
        comptime num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SMem]()[]

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        var pipeline = smem.create_pipeline()
        var barrier_handler = CPAsyncBarrierHandler(
            pipeline, Self.num_consumer, Self.cluster_size
        )

        # Create TileLoaderCPAsync loaders
        comptime k_align = find_K_alignment_upto_16B(K * size_of[a_type]())
        var a_loader, b_loader = Self.build_cpasync_loaders[k_align](a, b)

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle()

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group (all threads participate for cp.async)
            warpgroup_reg_dealloc[32]()

            Self.producer_main_loop_pipeline[num_k_iters=num_k_iters](
                UInt(block_idx_swizzle[1]),
                UInt(block_idx_swizzle[0]),
                0,
                a_loader,
                b_loader,
                barrier_handler,
                pipeline,
                smem.a_tiles(),
                smem.b_tiles(),
            )
        else:
            # Consumer warp groups
            comptime assert (
                Self.num_consumer <= 2
            ), "Only support 1 or 2 consumer"
            warpgroup_reg_alloc[232]()

            var local_warp_group_idx = warp_group_idx - 1
            var c_reg_tile = Self.AccumRegTile.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTile.stack_allocation()

            Self.consumer_arrive_empty_barriers(warp_group_thread_idx, pipeline)

            Self.consumer_main_loop_pipeline[num_k_iters=num_k_iters](
                wgmma_op,
                local_warp_group_idx,
                final_c_reg_tile,
                c_reg_tile,
                pipeline,
                smem.a_tiles(),
                smem.b_tiles(),
                warp_group_thread_idx,
            )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                == DType.float8_e4m3fn else c_reg_tile
            )

            Self.consumer_output(
                c_tma_op,
                c,
                smem.c_tile(),
                output_reg_tile,
                warp_group_thread_idx,
                local_warp_group_idx,
                thread_idx.x - UInt(WARPGROUP_SIZE),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()
