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
"""Pipeline DSL: ScheduleEntry, EntryBuilder, Pipe, pipe, annotate_pipe."""

from std.collections import InlineArray

from .dependency_graph import LoopBody, OpNode
from .types import (
    OpDesc,
    Phase,
    TargetCostModel,
)

# =============================================================================
# Schedule Entry
# =============================================================================


struct ScheduleEntry(ImplicitlyCopyable, Movable):
    """An operation placed at a specific position in the schedule.

    Fields:
        op: The operation descriptor.
        time_slot: Ordinal position within the phase (0-indexed).
        phase: Which phase this entry belongs to (PROLOGUE, KERNEL, EPILOGUE).
        is_prefetch: If True, this op is only emitted when there is a next
            iteration (i.e., conditional on `k < K - 2*BK`).
    """

    var op: OpDesc
    var time_slot: Int
    var phase: Phase
    var is_prefetch: Bool

    @always_inline
    def __init__(
        out self,
        *,
        op: OpDesc,
        time_slot: Int,
        phase: Phase,
        is_prefetch: Bool,
    ):
        self.op = op
        self.time_slot = time_slot
        self.phase = phase
        self.is_prefetch = is_prefetch


# =============================================================================
# Entry Builder
# =============================================================================


struct EntryBuilder[N: Int, phase: Phase]:
    """Appends ScheduleEntry values into a fixed-size array.

    Encapsulates the repeated `entries[t] = ScheduleEntry(...); t += 1`
    pattern, auto-tracking the write position and phase.
    """

    var entries: InlineArray[ScheduleEntry, Self.N]
    var pos: Int

    @always_inline
    def __init__(out self, pos: Int = 0):
        self.entries = InlineArray[ScheduleEntry, Self.N](uninitialized=True)
        self.pos = pos

    @always_inline
    def __init__(
        out self, entries: InlineArray[ScheduleEntry, Self.N], pos: Int = 0
    ):
        """Wrap an existing entries array (e.g. from a caller)."""
        self.entries = entries
        self.pos = pos

    @always_inline
    def emit(mut self, op: OpDesc, prefetch: Bool = False):
        """Append one entry and advance the position."""
        self.entries[self.pos] = ScheduleEntry(
            op=op,
            time_slot=self.pos,
            phase=Self.phase,
            is_prefetch=prefetch,
        )
        self.pos += 1

    @always_inline
    def emit_if(mut self, op: OpDesc, prefetch: Bool = False):
        """Emit only if op is present (not the NONE sentinel)."""
        if op.is_present():
            self.emit(op, prefetch)

    @always_inline
    def emit_flag(mut self, op: OpDesc, flag: Bool):
        """Emit only if flag is True."""
        if flag:
            self.emit(op)


# =============================================================================
# Pipe DSL — declarative pipeline sequence construction
# =============================================================================


struct Pipe[N: Int](ImplicitlyCopyable, Movable):
    """A compile-time sequence of N pipeline operations.

    Build sequences using the >> operator to chain ops into a pipeline:

        pipe(ld()) >> st() >> br() >> frag[0]() >> sb()

    Convert to schedule entries for a given phase:

        my_pipe.as_schedule[Phase.PROLOGUE]()

    Concatenate two pipes:

        pipe_a >> pipe_b   # Pipe[A+B]
    """

    var ops: InlineArray[OpDesc, Self.N]

    @always_inline
    def __init__(out self):
        self.ops = InlineArray[OpDesc, Self.N](uninitialized=True)

    @always_inline
    def __rshift__(self, op: OpDesc) -> Pipe[Self.N + 1]:
        """Append a single op: pipe >> op -> Pipe[N+1]."""
        var result = Pipe[Self.N + 1]()
        for i in range(Self.N):
            result.ops[i] = self.ops[i]
        result.ops[Self.N] = op
        return result^

    @always_inline
    def __rshift__[M: Int](self, other: Pipe[M]) -> Pipe[Self.N + M]:
        """Concatenate two pipes: Pipe[N] >> Pipe[M] -> Pipe[N+M]."""
        var result = Pipe[Self.N + M]()
        for i in range(Self.N):
            result.ops[i] = self.ops[i]
        for i in range(M):
            result.ops[Self.N + i] = other.ops[i]
        return result^

    @always_inline
    def as_schedule[phase: Phase](self) -> InlineArray[ScheduleEntry, Self.N]:
        """Convert to schedule entries with sequential time slots."""
        var entries = InlineArray[ScheduleEntry, Self.N](uninitialized=True)
        for i in range(Self.N):
            entries[i] = ScheduleEntry(
                op=self.ops[i],
                time_slot=i,
                phase=phase,
                is_prefetch=False,
            )
        return entries^

    @always_inline
    def as_schedule[
        phase: Phase
    ](self, offset: Int) -> InlineArray[ScheduleEntry, Self.N]:
        """Convert to schedule entries with time slots starting at offset."""
        var entries = InlineArray[ScheduleEntry, Self.N](uninitialized=True)
        for i in range(Self.N):
            entries[i] = ScheduleEntry(
                op=self.ops[i],
                time_slot=offset + i,
                phase=phase,
                is_prefetch=False,
            )
        return entries^

    @always_inline
    def emit_into[
        MaxN: Int, phase: Phase
    ](
        self, mut entries: InlineArray[ScheduleEntry, MaxN], offset: Int = 0
    ) -> Int:
        """Write ops into a larger schedule array starting at offset.

        Returns the new offset (offset + N), allowing chained writes.
        """
        for i in range(Self.N):
            entries[offset + i] = ScheduleEntry(
                op=self.ops[i],
                time_slot=offset + i,
                phase=phase,
                is_prefetch=False,
            )
        return offset + Self.N

    @always_inline
    def emit_into_body(
        self,
        mut body: LoopBody,
        offset: Int = 0,
    ) -> Int:
        """Write ops into a LoopBody's ops array as OpNodes.

        Returns the new offset (offset + N).
        """
        for i in range(Self.N):
            body.ops[offset + i] = OpNode(op=self.ops[i])
        return offset + Self.N


@always_inline
def pipe(op: OpDesc) -> Pipe[1]:
    """Create a single-element pipe. Use as the start of a >> chain."""
    var p = Pipe[1]()
    p.ops[0] = op
    return p^


def annotate_pipe[
    N: Int
](logical: Pipe[N], model: TargetCostModel,) -> Pipe[N]:
    """Apply a target cost model to a Pipe of logical ops.

    Same as annotate_ops but operates on Pipe[N] for compile-time-sized
    pipelines (e.g., single_buffer_schedule input).
    """
    var result = Pipe[N]()
    for i in range(N):
        var op = logical.ops[i]
        if op.tag < 128:
            var cost = model.get_cost(op.tag)
            op.resource = cost.resource
            op.latency = cost.latency
            op.role = cost.role
        result.ops[i] = op
    return result^
