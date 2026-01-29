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
"""RAII warp context managers for SM100 matmul kernel.

MmaWarpContext: MMA warp - allocates TMEM, deallocates on exit
EpilogueWarpContext: Epilogue warp - consumes TMEM, signals completion on exit
"""

from .barriers import TmemDeallocBarrier, WarpGroupBarrier
from .tile_pipeline import (
    OutputTilePipeline,
    EpilogueKContext,
    MmaKStage,
    InputTilePipeline,
    TilePayload,
)
from .pipeline import ProducerConsumerPipeline
from .tmem import TmemAllocation


# =============================================================================
# Shared type aliases for warp contexts
# =============================================================================


struct _WarpContextTypes[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
](TrivialRegisterType):
    """Shared type definitions for MMA and Epilogue warp contexts."""

    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime Pipeline = OutputTilePipeline[
        Self.num_accum_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Dealloc = TmemDeallocBarrier[Self.cta_group]
    comptime Sync = WarpGroupBarrier[
        Self.mma_threads + Self.epilogue_threads, 1
    ]


# =============================================================================
# MmaWarpContext
# =============================================================================


struct MmaWarpContext[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
](TrivialRegisterType):
    """MMA warp context - owns TMEM lifecycle and output pipeline.

    __enter__: Signals epilogue that TMEM is allocated
    __exit__: Waits for epilogue, deallocates TMEM
    """

    comptime _Types = _WarpContextTypes[
        Self.num_accum_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.mma_threads,
        Self.epilogue_threads,
    ]
    comptime Tmem = Self._Types.Tmem
    comptime Pipeline = Self._Types.Pipeline
    comptime Dealloc = Self._Types.Dealloc
    comptime Sync = Self._Types.Sync

    var tmem: Self.Tmem
    var output_pipeline: Self.Pipeline
    var dealloc_barrier: Self.Dealloc

    fn __init__(
        out self,
        tmem: Self.Tmem,
        output_pipeline: Self.Pipeline,
        dealloc_barrier: Self.Dealloc,
    ):
        self.tmem = tmem
        self.output_pipeline = output_pipeline
        self.dealloc_barrier = dealloc_barrier

    @staticmethod
    @always_inline
    fn create(
        tmem_addr_storage: Self.Tmem.SmemAddrStorage,
        accum_barriers: Self.Pipeline.BarrierArray,
        dealloc_mbar: Self.Dealloc.BarrierStorage,
        mma_complete_mask: UInt16,
    ) -> Self:
        """Create MMA warp context with all necessary components.

        Allocates TMEM and creates output pipeline internally.

        Args:
            tmem_addr_storage: Shared storage for TMEM address communication.
            accum_barriers: Barrier array for accumulator pipeline.
            dealloc_mbar: Barrier for TMEM deallocation synchronization.
            mma_complete_mask: Multicast mask for MMA completion signaling.

        Returns:
            Fully initialized MmaWarpContext.
        """
        var tmem = Self.Tmem.allocate(tmem_addr_storage)
        var output_pipeline = Self.Pipeline(
            accum_barriers, tmem, mma_complete_mask
        )
        return Self(tmem, output_pipeline, Self.Dealloc(dealloc_mbar))

    fn __enter__(self) -> Self:
        Self.Sync.arrive()
        return self

    fn __exit__(self):
        self.dealloc_barrier.complete_dealloc(self.tmem)

    @always_inline
    fn per_k_stage(
        mut self,
    ) -> MmaKStage[
        origin_of(self.output_pipeline),
        Self.num_accum_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]:
        """Get per-K stage for blockwise FP8 MMA loop.

        Returns a context manager that acquires an output stage and
        signals mma_arrive on exit.

        Example:
            for i in range(num_iters):
                with mma_ctx.per_k_stage() as mma_stage:
                    mma(input_tiles, mma_op, AccumTensor(mma_stage.tmem.offset()))
                # __exit__ signals mma_arrive automatically
        """
        return self.output_pipeline.per_k().produce()


# =============================================================================
# EpilogueWarpContext
# =============================================================================


struct EpilogueWarpContext[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
](TrivialRegisterType):
    """Epilogue warp context - consumes TMEM data, signals completion.

    IMPORTANT: Call Sync.wait() BEFORE constructing to ensure TMEM address
    is visible from shared memory.
    """

    comptime _Types = _WarpContextTypes[
        Self.num_accum_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.mma_threads,
        Self.epilogue_threads,
    ]
    comptime Tmem = Self._Types.Tmem
    comptime Pipeline = Self._Types.Pipeline
    comptime Dealloc = Self._Types.Dealloc
    comptime Sync = Self._Types.Sync

    var tmem: Self.Tmem
    var output_pipeline: Self.Pipeline
    var dealloc_barrier: Self.Dealloc

    fn __init__(
        out self,
        tmem: Self.Tmem,
        output_pipeline: Self.Pipeline,
        dealloc_barrier: Self.Dealloc,
    ):
        self.tmem = tmem
        self.output_pipeline = output_pipeline
        self.dealloc_barrier = dealloc_barrier

    @staticmethod
    @always_inline
    fn create(
        tmem_addr_storage: Self.Tmem.SmemAddrStorage,
        accum_barriers: Self.Pipeline.BarrierArray,
        dealloc_mbar: Self.Dealloc.BarrierStorage,
        mma_complete_mask: UInt16,
    ) -> Self:
        """Create Epilogue warp context with all necessary components.

        Reads TMEM address from shared memory and creates output pipeline.
        IMPORTANT: Call Sync.wait() BEFORE calling this to ensure TMEM
        address is visible.

        Args:
            tmem_addr_storage: Shared storage containing TMEM address.
            accum_barriers: Barrier array for accumulator pipeline.
            dealloc_mbar: Barrier for TMEM deallocation synchronization.
            mma_complete_mask: Multicast mask for MMA completion signaling.

        Returns:
            Fully initialized EpilogueWarpContext.
        """
        var tmem = Self.Tmem.from_shared(tmem_addr_storage)
        var output_pipeline = Self.Pipeline(
            accum_barriers, tmem, mma_complete_mask
        )
        return Self(tmem, output_pipeline, Self.Dealloc(dealloc_mbar))

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        self.dealloc_barrier.signal_complete()

    @always_inline
    fn per_k_stage[
        input_origin: MutOrigin,
        Payload: TilePayload,
        num_group_stages: Int,
        k_group_size: Int,
    ](
        mut self,
        ref[input_origin] input_pipeline: InputTilePipeline[
            Payload, num_group_stages, k_group_size
        ],
    ) -> EpilogueKContext[
        origin_of(self.output_pipeline),
        origin_of(input_pipeline.pipeline),
        Self.num_accum_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        num_group_stages,
    ]:
        """Get per-K stage context for blockwise FP8 epilogue.

        Bundles output pipeline (MMA→Epilogue sync) and input pipeline
        (A-scales consumption) into a single context manager.

        Example:
            for k_iter in range(num_iters):
                with epi_ctx.per_k_stage(input_pipeline) as epi_stage:
                    accum.promote(epi_stage, ...)
                # Both pipelines signaled automatically

        Args:
            input_pipeline: The InputTilePipeline (extracts .pipeline internally).

        Returns:
            EpilogueKContext context manager that handles both pipelines.
        """
        return self.output_pipeline.per_k_epilogue(input_pipeline.pipeline)
