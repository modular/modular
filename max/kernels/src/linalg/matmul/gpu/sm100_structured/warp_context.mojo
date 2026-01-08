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
from .tile_pipeline import OutputTilePipeline
from .tmem import TmemAllocation


# =============================================================================
# Shared type aliases for warp contexts
# =============================================================================


@register_passable("trivial")
struct _WarpContextTypes[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
]:
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


@register_passable("trivial")
struct MmaWarpContext[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
]:
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

    fn __enter__(self) -> Self:
        Self.Sync.arrive()
        return self

    fn __exit__(self):
        self.dealloc_barrier.complete_dealloc(self.tmem)


# =============================================================================
# EpilogueWarpContext
# =============================================================================


@register_passable("trivial")
struct EpilogueWarpContext[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
]:
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

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        self.dealloc_barrier.signal_complete()
