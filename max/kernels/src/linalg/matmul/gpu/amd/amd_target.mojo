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
"""AMD GPU target definitions for the pipeline scheduling framework.

Platform-specific hardware descriptions for AMD CDNA/RDNA GPUs. Provides:
  - MI355X cost model: per-op latencies, resources, and roles
  - MI355X pipeline configurations: double-buffer and single-buffer
  - MI355X target profiles: unified cost model + pipeline config
  - AMD schedule_group_barrier hints for instruction interleaving

The generic framework (schedule_framework.mojo) has no AMD-specific
references. All hardware knowledge lives here.
"""

from std.collections import List

from pipeline.types import (
    OpCost,
    OpDesc,
    OpRole,
    Phase,
    ResourceKind,
    TargetCostModel,
    _Ops,
)
from pipeline.config import (
    BlockSizing,
    FragOrder,
    LoopCarriedSpec,
    PipelineConfig,
    TargetProfile,
    WarpStaggerRule,
)
from pipeline.pipeline_dsl import ScheduleEntry


# =============================================================================
# MI355X Cost Model
# =============================================================================


def mi355x_cost_model() -> TargetCostModel:
    """MI355X cost model: production-tuned latencies.

    Global loads (LOAD_A, LOAD_B): GLOBAL_MEM, 200 cycles, GLOBAL_LOAD
    Fragment loads (MMA_LOAD_A, MMA_LOAD_B): LDS, 20 cycles, FRAGMENT_LOAD
    MMA compute (COMPUTE, MMA): MMA_UNIT, 16 cycles, COMPUTE

    Op tags are kernel-specific (defined in PingPongOps / DefaultMatmulOps):
      Ping-pong: 0=LOAD_A, 1=LOAD_B, 2=COMPUTE, 3=MMA_LOAD_A, 4=MMA_LOAD_B, 5=MMA
      Default:   0=LOAD_DRAM, 1=STORE_SMEM, 2=LOAD_FRAG, 3=COMPUTE
    """
    var model = TargetCostModel()
    # Global loads: DRAM → LDS buffer
    model.set_cost(0, OpCost(ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD))
    model.set_cost(1, OpCost(ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD))
    # Compute (tag 2 is legacy COMPUTE alias)
    model.set_cost(2, OpCost(ResourceKind.MMA_UNIT, 16, OpRole.COMPUTE))
    # Fragment loads: LDS → registers
    model.set_cost(3, OpCost(ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    model.set_cost(4, OpCost(ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    # MMA execution
    model.set_cost(5, OpCost(ResourceKind.MMA_UNIT, 16, OpRole.COMPUTE))
    return model^


# =============================================================================
# MI355X Pipeline Configurations
# =============================================================================


def mi355x_double_buffer(
    vm_per_load_a: Int = 4,
    vm_per_load_b: Int = 4,
    max_globals: Int = 1,
) -> PipelineConfig:
    """MI355X ping-pong: double buffer, 2 warp groups, 2x2 MMA grid."""
    return PipelineConfig(
        depth=2,
        prefetch=1,
        drain_passes=2,
        prologue_fill=1,
        loop_carried=LoopCarriedSpec.none(),
        block_sizing=BlockSizing(heavy=4, light=2, max_globals=max_globals),
        frag_order=FragOrder(b_before_a=True),
        m_mmas=2,
        n_mmas=2,
        num_halves=2,
        mma_serial=True,
        mma_latency=16,
        vm_per_load_a=vm_per_load_a,
        vm_per_load_b=vm_per_load_b,
        ch0_match_field=0,  # A: frag.subtile matches compute.stage (row)
        ch1_match_field=1,  # B: frag.subtile matches compute.subtile (col)
        warp_stagger=WarpStaggerRule.auto(),
    )


def mi355x_single_buffer() -> PipelineConfig:
    """MI355X default matmul: single buffer, barrier-gated."""
    return PipelineConfig(
        depth=1,
        prefetch=1,
        drain_passes=2,
        prologue_fill=0,
        loop_carried=LoopCarriedSpec(role=OpRole.FRAGMENT_LOAD, selector=0),
        block_sizing=BlockSizing(heavy=4, light=2, max_globals=0),
        frag_order=FragOrder(b_before_a=True),
        m_mmas=2,
        n_mmas=2,
        num_halves=1,
        mma_serial=True,
        mma_latency=16,
        vm_per_load_a=4,
        vm_per_load_b=4,
        ch0_match_field=0,
        ch1_match_field=1,
        warp_stagger=WarpStaggerRule.none(),
    )


# =============================================================================
# MI355X Target Profiles
# =============================================================================


def mi355x_target(
    vm_per_load_a: Int = 4,
    vm_per_load_b: Int = 4,
    max_globals: Int = 1,
) -> TargetProfile:
    """MI355X target: ping-pong double-buffer with production-tuned costs.

    Provides both the per-op cost model (latencies, resources, roles)
    and the pipeline structure (depth=2, 2x2 MMA grid, warp groups)
    from a single call.
    """
    return TargetProfile(
        cost_model=mi355x_cost_model(),
        pipeline=mi355x_double_buffer(
            vm_per_load_a=vm_per_load_a,
            vm_per_load_b=vm_per_load_b,
            max_globals=max_globals,
        ),
    )


def mi355x_single_buffer_cost_model() -> TargetCostModel:
    """MI355X cost model for single-buffer matmul (DefaultMatmulOps tags).

    Tag mapping (from DefaultMatmulOps):
      0=LOAD_DRAM:  GLOBAL_MEM, 200 cycles, GLOBAL_LOAD
      1=STORE_SMEM: LDS, 20 cycles, SHARED_STORE
      2=LOAD_FRAG:  LDS, 20 cycles, FRAGMENT_LOAD
      3=COMPUTE:    MMA_UNIT, 64 cycles, COMPUTE
    """
    var model = TargetCostModel()
    model.set_cost(0, OpCost(ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD))
    model.set_cost(1, OpCost(ResourceKind.LDS, 20, OpRole.SHARED_STORE))
    model.set_cost(2, OpCost(ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    model.set_cost(3, OpCost(ResourceKind.MMA_UNIT, 64, OpRole.COMPUTE))
    return model^


def mi355x_single_buffer_target() -> TargetProfile:
    """MI355X target: single-buffer with production-tuned costs."""
    return TargetProfile(
        cost_model=mi355x_single_buffer_cost_model(),
        pipeline=mi355x_single_buffer(),
    )


# =============================================================================
# AMD Schedule Group Barrier Hints
# =============================================================================


@fieldwise_init
struct AMDScheduleHints(ImplicitlyCopyable, Movable):
    """Hardware expansion factors for AMD schedule_group_barrier hints.

    Each body op (COMPUTE, FRAGMENT_LOAD, STORE_SMEM) expands to multiple
    hardware instructions. These factors tell the framework how many, so
    it can compute the interleaving ratios automatically.

    Example for a 4x4x2 MMA config with 16x16 tiles:
        AMDScheduleHints(m_mmas=4, n_mmas=4, k_mmas=2,
                         mma_m=16, mma_n=16, a_loads=2, b_loads=2)
    """

    var m_mmas: Int  # M-dimension MMA tiles (DS_READ per fragment = m+n)
    var n_mmas: Int  # N-dimension MMA tiles
    var k_mmas: Int  # K-dimension MMA tiles per COMPUTE body op
    var mma_m: Int  # MMA instruction M dimension (for 32x32 special case)
    var mma_n: Int  # MMA instruction N dimension
    var a_loads_per_thread: Int  # DS_WRITE ops per STORE_SMEM (A tile)
    var b_loads_per_thread: Int  # DS_WRITE ops per STORE_SMEM (B tile)


def append_amd_hints(
    mut ker: List[ScheduleEntry],
    body: List[OpDesc],
    config: PipelineConfig,
    hints: AMDScheduleHints,
):
    """Append AMD schedule_group_barrier hints to a kernel entry list."""
    var lc_sel = config.loop_carried.selector
    var num_non_lc_frags = 0
    var num_lc_frags = 0
    for i in range(len(body)):
        if body[i].role == OpRole.FRAGMENT_LOAD:
            if body[i].subtile == lc_sel:
                num_lc_frags += 1
            else:
                num_non_lc_frags += 1
    var num_frags = num_lc_frags + num_non_lc_frags

    var reads_per_frag = hints.m_mmas + hints.n_mmas
    var total_mmas = hints.m_mmas * hints.n_mmas * hints.k_mmas
    var total_stores = hints.a_loads_per_thread + hints.b_loads_per_thread
    var total_reads = reads_per_frag * num_frags

    var mmas_per_read = min(
        1 if hints.mma_m == hints.mma_n == 32 else 2,
        total_mmas // total_reads,
    )
    var remaining_mmas = total_mmas - total_reads * mmas_per_read
    var mmas_per_store = 0
    var mmas_per_store_extra = 0
    if total_stores > 0:
        mmas_per_store, mmas_per_store_extra = divmod(
            remaining_mmas, total_stores
        )

    @always_inline
    def _hint(mut ker: List[ScheduleEntry], mask: Int, count: Int):
        var slot = len(ker)
        ker.append(
            ScheduleEntry(
                op=OpDesc(
                    tag=_Ops.SCHED_GROUP_BARRIER.value,
                    subtile=mask,
                    wait_value=count,
                    resource=ResourceKind.SCALAR,
                    role=OpRole.FENCE,
                ),
                time_slot=slot,
                phase=Phase.KERNEL,
                is_prefetch=False,
            )
        )

    comptime DS_READ = 0
    comptime DS_WRITE = 1
    comptime VMEM_READ = 2
    comptime MFMA = 3

    for _ in range(reads_per_frag * num_non_lc_frags):
        _hint(ker, DS_READ, 1)
        _hint(ker, MFMA, mmas_per_read)

    for i in range(total_stores):
        var mmas_this = mmas_per_store
        if i < mmas_per_store_extra:
            mmas_this += 1
        _hint(ker, DS_WRITE, 1)
        _hint(ker, MFMA, mmas_this // 2)
        _hint(ker, VMEM_READ, 1)
        _hint(ker, MFMA, mmas_this - mmas_this // 2)

    for _ in range(reads_per_frag * num_lc_frags):
        _hint(ker, DS_READ, 1)
        _hint(ker, MFMA, mmas_per_read)
