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
"""MMABlockSpec and PipelineProgram — declarative pipeline program representation.

These structures hold the compiled schedule as a sequence of MMA blocks,
separating schedule definition from schedule expansion.
"""

from std.collections import List

from .pipeline_dsl import EntryBuilder, ScheduleEntry
from .types import OpDesc, Phase, _Ops

# =============================================================================
# MMA Block Specification & Pipeline Program
# =============================================================================


struct MMABlockSpec(ImplicitlyCopyable, Movable):
    """Declarative specification for one MMA block in a pipeline schedule.

    Each MMA block follows a fixed pattern with optional elements:
      [pre_op_0?], [pre_op_1?], [global_load?], [global_load_1?],
      [pre_sync?], barrier, [schedule_barrier?], [wait_lgkm(0)?],
      set_prio(1), mma, [fused_mma?], set_prio(0),
      barrier, [schedule_barrier?]

    Optional elements are controlled by sentinel values:
      - OpDesc fields: _Ops.NONE.value means skip
      - Bool flags: False means skip the corresponding element

    All operations are stored as pre-built OpDesc values, avoiding any
    runtime-to-comptime conversion.
    """

    var mma: OpDesc
    var pre_op_0: OpDesc
    var pre_op_1: OpDesc
    var global_load: OpDesc
    var global_load_1: OpDesc
    var pre_sync: OpDesc
    var fused_mma: OpDesc
    var post_barrier_lgkm: Bool
    var post_barrier_sched: Bool
    var trailing_sched_barrier: Bool
    var global_load_prefetch: Bool
    var global_load_1_prefetch: Bool
    var drain_lgkm_before_loads: Bool

    @always_inline
    def __init__(
        out self,
        *,
        mma: OpDesc,
        pre_op_0: OpDesc = OpDesc.none(),
        pre_op_1: OpDesc = OpDesc.none(),
        global_load: OpDesc = OpDesc.none(),
        global_load_1: OpDesc = OpDesc.none(),
        pre_sync: OpDesc = OpDesc.none(),
        fused_mma: OpDesc = OpDesc.none(),
        post_barrier_lgkm: Bool = True,
        post_barrier_sched: Bool = False,
        trailing_sched_barrier: Bool = False,
        global_load_prefetch: Bool = False,
        global_load_1_prefetch: Bool = False,
        drain_lgkm_before_loads: Bool = False,
    ):
        self.mma = mma
        self.pre_op_0 = pre_op_0
        self.pre_op_1 = pre_op_1
        self.global_load = global_load
        self.global_load_1 = global_load_1
        self.pre_sync = pre_sync
        self.fused_mma = fused_mma
        self.post_barrier_lgkm = post_barrier_lgkm
        self.post_barrier_sched = post_barrier_sched
        self.trailing_sched_barrier = trailing_sched_barrier
        self.global_load_prefetch = global_load_prefetch
        self.global_load_1_prefetch = global_load_1_prefetch
        self.drain_lgkm_before_loads = drain_lgkm_before_loads

    @always_inline
    def _n_pre_barrier_ops(self) -> Int:
        """Count optional ops before the barrier (pre_op_0/1, drain, global loads, pre_sync).
        """
        var n = 0
        if self.pre_op_0.is_present():
            n += 1
        if self.pre_op_1.is_present():
            n += 1
        if self.drain_lgkm_before_loads:
            n += 1
        if self.global_load.is_present():
            n += 1
        if self.global_load_1.is_present():
            n += 1
        if self.pre_sync.is_present():
            n += 1
        return n

    @always_inline
    def entry_count(self) -> Int:
        """Count the number of schedule entries this block will expand to.

        5 mandatory entries (barrier, set_prio(1), mma, set_prio(0), barrier)
        plus one for each optional field that is present.
        """
        var n = 5 + self._n_pre_barrier_ops()
        if self.post_barrier_sched:
            n += 1
        if self.post_barrier_lgkm:
            n += 1
        if self.fused_mma.is_present():
            n += 1
        if self.trailing_sched_barrier:
            n += 1
        return n

    @always_inline
    def mma_position(self) -> Int:
        """Return the offset of the MMA op within this block's entries."""
        var n = self._n_pre_barrier_ops() + 1  # +1 for barrier
        if self.post_barrier_sched:
            n += 1
        if self.post_barrier_lgkm:
            n += 1
        return n + 1  # +1 for set_prio(1)

    @always_inline
    def post_barrier_lgkm_position(self) -> Int:
        """Return the offset of the post-barrier wait_lgkm(0) within entries.

        Returns -1 if post_barrier_lgkm is False.
        """
        if not self.post_barrier_lgkm:
            return -1
        var n = self._n_pre_barrier_ops() + 1  # +1 for barrier
        if self.post_barrier_sched:
            n += 1
        return n

    @always_inline
    def expand[N: Int, phase: Phase](self, mut b: EntryBuilder[N, phase]):
        """Expand this block spec into schedule entries via an EntryBuilder."""
        b.emit_if(self.pre_op_0)
        b.emit_if(self.pre_op_1)
        b.emit_flag(OpDesc.wait_lgkm[0](), self.drain_lgkm_before_loads)
        b.emit_if(self.global_load, self.global_load_prefetch)
        b.emit_if(self.global_load_1, self.global_load_1_prefetch)
        b.emit_if(self.pre_sync)
        b.emit(OpDesc.barrier())
        b.emit_flag(OpDesc.schedule_barrier(), self.post_barrier_sched)
        b.emit_flag(OpDesc.wait_lgkm[0](), self.post_barrier_lgkm)
        b.emit(OpDesc.set_prio[1]())
        b.emit(self.mma)
        b.emit_if(self.fused_mma)
        b.emit(OpDesc.set_prio[0]())
        b.emit(OpDesc.barrier())
        b.emit_flag(OpDesc.schedule_barrier(), self.trailing_sched_barrier)

    @always_inline
    def expand_to_list(self, mut out: List[ScheduleEntry], phase: Phase):
        """Expand this block spec by appending to a List."""

        @always_inline
        def _e(
            mut out: List[ScheduleEntry],
            op: OpDesc,
            phase: Phase,
            prefetch: Bool = False,
        ):
            out.append(
                ScheduleEntry(
                    op=op,
                    time_slot=len(out),
                    phase=phase,
                    is_prefetch=prefetch,
                )
            )

        if self.pre_op_0.is_present():
            _e(out, self.pre_op_0, phase)
        if self.pre_op_1.is_present():
            _e(out, self.pre_op_1, phase)
        if self.drain_lgkm_before_loads:
            _e(out, OpDesc.wait_lgkm[0](), phase)
        if self.global_load.is_present():
            _e(out, self.global_load, phase, self.global_load_prefetch)
        if self.global_load_1.is_present():
            _e(out, self.global_load_1, phase, self.global_load_1_prefetch)
        if self.pre_sync.is_present():
            _e(out, self.pre_sync, phase)
        _e(out, OpDesc.barrier(), phase)
        if self.post_barrier_sched:
            _e(out, OpDesc.schedule_barrier(), phase)
        if self.post_barrier_lgkm:
            _e(out, OpDesc.wait_lgkm[0](), phase)
        _e(out, OpDesc.set_prio[1](), phase)
        _e(out, self.mma, phase)
        if self.fused_mma.is_present():
            _e(out, self.fused_mma, phase)
        _e(out, OpDesc.set_prio[0](), phase)
        _e(out, OpDesc.barrier(), phase)
        if self.trailing_sched_barrier:
            _e(out, OpDesc.schedule_barrier(), phase)


struct PipelineProgram(Copyable, Movable):
    """A pipeline schedule phase as a sequence of MMA block specifications.

    Separates schedule *definition* (what blocks to emit) from schedule
    *expansion* (writing ScheduleEntry values into a List). This makes
    schedules declarative data rather than imperative code.
    """

    var blocks: List[MMABlockSpec]
    var trailing_barrier: Bool

    @always_inline
    def __init__(
        out self, num_blocks: Int = 0, *, trailing_barrier: Bool = False
    ):
        self.blocks = List[MMABlockSpec]()
        for _ in range(num_blocks):
            self.blocks.append(MMABlockSpec(mma=OpDesc.none()))
        self.trailing_barrier = trailing_barrier

    @always_inline
    def total_entries(self) -> Int:
        """Count total schedule entries this program will expand to."""
        var n = 0
        for i in range(len(self.blocks)):
            n += self.blocks[i].entry_count()
        if self.trailing_barrier:
            n += 1
        return n

    @always_inline
    def block_start(self, block_idx: Int) -> Int:
        """Return the starting entry index for the given block."""
        var n = 0
        for i in range(block_idx):
            n += self.blocks[i].entry_count()
        return n

    @always_inline
    def mma_entry(self, block_idx: Int) -> Int:
        """Return the entry index of the MMA op in the given block."""
        return (
            self.block_start(block_idx) + self.blocks[block_idx].mma_position()
        )

    @always_inline
    def expand_to_list(self, phase: Phase) -> List[ScheduleEntry]:
        """Expand all blocks into a List of schedule entries."""
        var out = List[ScheduleEntry]()
        for i in range(len(self.blocks)):
            self.blocks[i].expand_to_list(out, phase)
        if self.trailing_barrier:
            out.append(
                ScheduleEntry(
                    op=OpDesc.barrier(),
                    time_slot=len(out),
                    phase=phase,
                    is_prefetch=False,
                )
            )
        return out^
