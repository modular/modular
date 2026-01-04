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

Provides lifecycle management for MMA and Epilogue warps:

- MmaWarpContext: Allocates TMEM, manages output pipeline, deallocates on exit
- EpilogueWarpContext: Consumes TMEM data, signals completion on exit

Usage (MMA warp):
    var tmem = Tmem.allocate(smem.tmem_addr())
    var ctx = MmaWarpContext(tmem, OutputPipeline(...), TmemDeallocBarrier(...))
    with ctx:
        with ctx.output_pipeline.producer() as stage:
            # ... MMA work ...
    # __exit__ waits for epilogue and deallocates TMEM

Usage (Epilogue warp):
    EpilogueCtx.Sync.wait()  # MUST wait before reading TMEM address!
    var tmem = Tmem.from_shared(smem.tmem_addr())
    var ctx = EpilogueWarpContext(tmem, OutputPipeline(...), TmemDeallocBarrier(...))
    with ctx:
        with ctx.output_pipeline.consumer() as stage:
            # ... store results ...
    # __exit__ signals MMA that epilogue is complete
"""

from .barriers import TmemDeallocBarrier, WarpGroupBarrier
from .tile_pipeline import OutputTilePipeline
from .tmem import TmemAllocation


@register_passable("trivial")
struct MmaWarpContext[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
]:
    """MMA warp context - owns TMEM lifecycle and output pipeline.

    __enter__: Signals epilogue that TMEM is allocated (non-blocking arrive)
    __exit__: Waits for epilogue completion, then deallocates TMEM
    """

    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime Pipeline = OutputTilePipeline[
        Self.num_accum_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Dealloc = TmemDeallocBarrier[Self.cta_group]
    comptime Sync = WarpGroupBarrier[
        Self.mma_threads + Self.epilogue_threads, 1
    ]

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


@register_passable("trivial")
struct EpilogueWarpContext[
    num_accum_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    mma_threads: Int,
    epilogue_threads: Int,
]:
    """Epilogue warp context - consumes TMEM data, signals completion.

    IMPORTANT: Call Sync.wait() BEFORE constructing! The wait must happen
    before reading the TMEM address that MMA stored in shared memory.

    __enter__: Returns self (wait already done by caller)
    __exit__: Signals MMA warp that epilogue is complete
    """

    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime Pipeline = OutputTilePipeline[
        Self.num_accum_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Dealloc = TmemDeallocBarrier[Self.cta_group]
    comptime Sync = WarpGroupBarrier[
        Self.mma_threads + Self.epilogue_threads, 1
    ]

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
