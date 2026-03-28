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
"""Scheduling algorithms: greedy and within-iteration optimal schedulers.

These are free functions that consume a LoopBody and return a permutation.
The graph (LoopBody) never schedules itself — scheduling is an external
concern. This follows the LLVM pattern where ScheduleDAG is pure data and
MachineSchedStrategy is a separate pluggable component.

Note: the "optimal" scheduler minimizes within-iteration makespan (single
iteration time span), not the initiation interval (II) used in modulo
scheduling. Inter-iteration overlap is handled by the prologue/epilogue
structure in program_builder, not by the scheduler. See DESIGN.md for
scope and limitations.
"""

from std.collections import List

from .dependency_graph import LoopBody
from .types import OpRole, ResourceKind


def _is_ready(
    body: LoopBody,
    op_idx: Int,
    scheduled: List[Bool],
) -> Bool:
    """Check if all d=0 FLOW and ANTI predecessors are scheduled."""
    for e in range(len(body.edges)):
        var edge = body.edges[e]
        if edge.loop_distance >= 1:
            continue
        # Enforce both FLOW (RAW) and ANTI (WAR) d=0 edges.
        # ANTI edges model LDS read-before-write: mma_load must
        # complete before a prefetch global_load overwrites the
        # same LDS buffer. Without enforcing ANTI edges, the
        # scheduler can move prefetch writes before reads, causing
        # LDS corruption.
        if edge.consumer_idx == op_idx and not scheduled[edge.producer_idx]:
            return False
    return True


def _evaluate_makespan(
    body: LoopBody, order: List[Int], lds_contention_penalty: Int = 0
) -> Int:
    """Simulate execution of an ordering on a simple resource model.

    Models hardware with serial MMA unit and optional LDS port contention:
      - MMA_UNIT: serial (capacity 1) — MFMA ops serialize
      - LDS port (when lds_contention_penalty > 0): shared between LDS reads
        (fragment loads, resource=LDS) and LDS writes (global loads,
        resource=GLOBAL_MEM). Overlapping ops incur a penalty.
      - All other resources: unlimited capacity

    The LDS contention model reflects that on AMD CDNA3, global→LDS stores
    and LDS→register reads share physical LDS ports. When both types overlap,
    the hardware serializes port access, adding latency. The penalty parameter
    controls how many extra cycles each overlap costs (0 = no modeling).

    For each op in the given order:
      start = max(dep_ready_time, mma_unit_free, lds_port_free)
      finish = start + latency

    Returns the makespan (max finish time across all ops).
    """
    var finish_time = List[Int]()
    for _ in range(len(body.ops)):
        finish_time.append(0)
    var mma_free = 0
    var lds_port_free = 0

    for pos in range(len(body.ops)):
        var op_idx = order[pos]
        var op = body.ops[op_idx]
        var lat = op.latency
        var is_mma = op.op.role == OpRole.COMPUTE
        var uses_lds_port = lds_contention_penalty > 0 and (
            op.resource == ResourceKind.LDS
            or op.resource == ResourceKind.GLOBAL_MEM
        )

        # Earliest start: max of d=0 predecessor finish times.
        var start = 0
        for e in range(len(body.edges)):
            var edge = body.edges[e]
            if edge.loop_distance >= 1:
                continue
            if edge.consumer_idx != op_idx:
                continue
            var pred_finish = finish_time[edge.producer_idx]
            if pred_finish > start:
                start = pred_finish

        # Serial MMA resource constraint.
        if is_mma and mma_free > start:
            start = mma_free

        # LDS port contention: fragment loads (LDS reads) and global loads
        # (LDS writes) share the port. Apply penalty when they overlap.
        if uses_lds_port and lds_port_free > start:
            start = lds_port_free

        var end = start + lat
        finish_time[op_idx] = end
        if is_mma:
            mma_free = end
        if uses_lds_port:
            # The port is "busy" for penalty cycles, not the full latency.
            # This models partial contention rather than full serialization.
            lds_port_free = start + lds_contention_penalty

    # Makespan = max finish time.
    var makespan = 0
    for i in range(len(body.ops)):
        if finish_time[i] > makespan:
            makespan = finish_time[i]
    return makespan


def _asap_lower_bound(body: LoopBody, lds_contention_penalty: Int = 0) -> Int:
    """Compute a lower bound on the achievable within-iteration makespan.

    Returns max(critical_path, mma_serial_bound, lds_serial_bound) where:
      critical_path = max(ASAP[i] + latency[i]) — longest dep chain
      mma_serial_bound = sum(latency[i]) for MMA/COMPUTE ops — serial MMA
      lds_serial_bound = sum(penalty) for LDS-port ops (when penalty > 0)

    Note: this is a within-iteration bound, not the MII (minimum initiation
    interval) from modulo scheduling theory. If a schedule's makespan
    equals this bound, no better within-iteration ordering exists.
    """
    var asap = body.compute_asap()

    # Critical path bound: longest ASAP completion time.
    var critical_path = 0
    for i in range(len(body.ops)):
        var completion = asap[i] + body.ops[i].latency
        if completion > critical_path:
            critical_path = completion

    # MMA serial bound: total latency of serial-resource ops.
    var mma_total = 0
    for i in range(len(body.ops)):
        if body.ops[i].op.role == OpRole.COMPUTE:
            mma_total += body.ops[i].latency

    var bound = critical_path if critical_path > mma_total else mma_total

    # LDS port serial bound: total penalty for ops that use the LDS port.
    if lds_contention_penalty > 0:
        var lds_count = 0
        for i in range(len(body.ops)):
            var res = body.ops[i].resource
            if res == ResourceKind.LDS or res == ResourceKind.GLOBAL_MEM:
                lds_count += 1
        var lds_total = lds_count * lds_contention_penalty
        if lds_total > bound:
            bound = lds_total

    return bound


def greedy_schedule(body: LoopBody) -> List[Int]:
    """Constrained list scheduler for MMA-centered block structure.

    Derives a valid execution order from the dependency graph,
    respecting structural constraints of the interleaved ping-pong kernel:

      1. Half isolation: ops 0..num_ops/2 are scheduled first (blocks 0-3),
         then ops num_ops/2..num_ops (blocks 4-7). This preserves the
         two-half warp-group structure that build_program_from_ldg_ordered()
         requires (first 12 ops → blocks 0-3, last 12 → blocks 4-7).
      2. Data dependencies: all d=0 predecessors (FLOW and ANTI) must be
         scheduled before the consumer. FLOW edges enforce RAW (register
         and accumulator) deps. ANTI edges enforce WAR (LDS buffer) deps:
         all mma_loads reading from an LDS buffer must complete before any
         prefetch global_load writes to that buffer.
      3. MMA-centered blocks: each block terminates with exactly 1 MMA.
      4. Priority: lowest op index wins among ready ops. This reproduces
         the declaration order from define_interleaved_loop_body(), which
         is the correct execution order. The ScheduleConfig wait counts
         (vmcnt, lgkmcnt) are calibrated for this specific ordering, so
         any reordering requires recalculating those counts.

    Returns a permutation of [0, num_ops) suitable for
    build_program_from_ldg_ordered(). When ops are defined in execution
    order, this produces the identity permutation — validating that the
    dependency graph is sufficient to derive the schedule.
    """
    var scheduled = List[Bool]()
    for _ in range(len(body.ops)):
        scheduled.append(False)
    var order = List[Int]()
    var pos = 0

    # Schedule each half independently.
    var half_size = len(body.ops) // 2
    for half in range(2):
        var half_lo = half * half_size
        var half_hi = half_lo + half_size

        # Process 4 MMA-centered blocks per half.
        for _block in range(4):
            while pos < len(body.ops):
                # Pick lowest-index ready op in this half.
                var best = -1
                for i in range(half_lo, half_hi):
                    if scheduled[i]:
                        continue
                    if not _is_ready(body, i, scheduled):
                        continue
                    best = i
                    break  # Lowest index wins.

                debug_assert(best >= 0, "no ready op found for block")

                order.append(best)
                pos += 1
                scheduled[best] = True
                if body.ops[best].op.role == OpRole.COMPUTE:
                    break  # Block terminated by MMA/compute.

    debug_assert(pos == len(body.ops), "all ops must be scheduled")
    return order^


def optimal_schedule(body: LoopBody) -> List[Int]:
    """Find the minimum-makespan execution ordering via backtracking search.

    Explores all valid orderings (respecting d=0 dependency edges) and
    returns the one with minimum simulated within-iteration makespan.
    This is an exhaustive CSP solver that runs entirely at compile time.

    Note: "optimal" refers to within-iteration makespan, not the
    initiation interval (II) from modulo scheduling theory.

    For small graphs (8 ops, default matmul), the search tree has ~50-200
    leaves and completes quickly without pruning. Early termination kicks
    in when the lower bound is achieved — provably optimal, stop searching.

    Uses iterative backtracking with explicit state — no recursion.
    Same comptime patterns as greedy_schedule() (while loops,
    mutable List, _is_ready checks).

    Note: This generic solver does NOT call greedy_schedule() for
    seeding because greedy_schedule() assumes the 24-op interleaved
    ping-pong structure (MMA-terminated blocks). For the halved variant
    used with 24-op graphs, see optimal_schedule_with_halves().

    Returns:
        Permutation of [0, num_ops) with minimum makespan.
    """
    var placed = List[Bool]()
    for _ in range(len(body.ops)):
        placed.append(False)
    var order = List[Int]()
    for _ in range(len(body.ops)):
        order.append(0)
    var best_order = List[Int]()
    for _ in range(len(body.ops)):
        best_order.append(0)
    var best_cost = 999999
    var pos = 0
    var lb = _asap_lower_bound(body)

    # last_tried[d] = last op index tried at depth d (-1 = none yet).
    var last_tried = List[Int]()
    for _ in range(len(body.ops)):
        last_tried.append(-1)

    while pos >= 0:
        if pos == len(body.ops):
            # Complete ordering — evaluate and potentially update best.
            var cost = _evaluate_makespan(body, order)
            if cost < best_cost:
                best_cost = cost
                for i in range(len(body.ops)):
                    best_order[i] = order[i]
                # Early termination: provably optimal.
                if best_cost == lb:
                    return best_order^
            # Backtrack.
            pos -= 1
            placed[order[pos]] = False
            continue

        # Find next ready op with index > last_tried[pos].
        var found = False
        for i in range(last_tried[pos] + 1, len(body.ops)):
            if placed[i]:
                continue
            if not _is_ready(body, i, placed):
                continue
            # Place it.
            order[pos] = i
            placed[i] = True
            last_tried[pos] = i
            pos += 1
            found = True
            break

        if not found:
            # No valid candidate at this depth — backtrack.
            last_tried[pos] = -1
            if pos > 0:
                pos -= 1
                placed[order[pos]] = False
            else:
                break  # Exhausted all possibilities.

    return best_order^


def optimal_schedule_with_halves(
    body: LoopBody,
    max_globals_per_block: Int = 0,
    lds_contention_penalty: Int = 0,
) -> List[Int]:
    """Optimal scheduler with half-isolation constraint.

    Positions [0, N/2) draw from ops [0, N/2), positions [N/2, N)
    draw from ops [N/2, N). This mirrors the greedy_schedule()
    half-isolation for the ping-pong kernel but uses exhaustive
    backtracking instead of greedy selection.

    Half isolation preserves the warp-group structure that
    build_program_from_ldg_ordered() requires: first N/2 ops map to
    blocks 0-3 (warp group 0), last N/2 to blocks 4-7 (warp group 1).

    Pruning optimizations for compile-time feasibility:
      1. Seed best_cost from greedy greedy_schedule() — provides
         a strong upper bound that prunes most branches immediately.
      2. Early termination when best_cost == lower_bound — provably
         optimal, stop searching.

    When max_globals_per_block > 0, an additional structural constraint
    limits how many GLOBAL_LOAD ops can appear between consecutive
    COMPUTE (MMA) ops. This ensures uniform global load distribution
    across MMA blocks after build_double_buffer_program groups by MMA
    delimiters. For example, max_globals_per_block=1 yields a [1,1,1,1]
    distribution instead of the unconstrained [1,0,2,1].

    When lds_contention_penalty > 0, the resource model adds a shared
    LDS port constraint: fragment loads (LDS reads) and global loads
    (LDS writes) contend on the same port, with the given penalty per
    overlap. This biases the solver toward separating LDS reads from
    LDS writes in the schedule.
    """
    # Seed from greedy schedule — provides a strong initial upper bound.
    var greedy = greedy_schedule(body)
    var best_cost = _evaluate_makespan(body, greedy, lds_contention_penalty)
    var best_order = greedy^

    # Early exit: if greedy already achieves the lower bound, it's optimal.
    var lb = _asap_lower_bound(body, lds_contention_penalty)
    if best_cost == lb:
        return best_order^

    var placed = List[Bool]()
    for _ in range(len(body.ops)):
        placed.append(False)
    var order = List[Int]()
    for _ in range(len(body.ops)):
        order.append(0)
    var pos = 0

    var last_tried = List[Int]()
    for _ in range(len(body.ops)):
        last_tried.append(-1)

    var half_size = len(body.ops) // 2

    while pos >= 0:
        if pos == len(body.ops):
            # Both halves complete — evaluate.
            var cost = _evaluate_makespan(body, order, lds_contention_penalty)
            if cost < best_cost:
                best_cost = cost
                for i in range(len(body.ops)):
                    best_order[i] = order[i]
                # Early termination: provably optimal.
                if best_cost == lb:
                    return best_order^
            # Backtrack.
            pos -= 1
            placed[order[pos]] = False
            continue

        # Determine which half based on current position.
        var half_lo = 0 if pos < half_size else half_size
        var half_hi = half_size if pos < half_size else len(body.ops)

        # Ensure search starts at half_lo.
        var start_from = last_tried[pos] + 1
        if start_from < half_lo:
            start_from = half_lo

        # Count globals in current block (since last COMPUTE) for the
        # max_globals_per_block constraint.
        var globals_in_block = 0
        if max_globals_per_block > 0:
            var half_start_pos = 0 if pos < half_size else half_size
            for j in range(pos - 1, half_start_pos - 1, -1):
                var placed_role = body.ops[order[j]].op.role
                if placed_role == OpRole.COMPUTE:
                    break
                if placed_role == OpRole.GLOBAL_LOAD:
                    globals_in_block += 1

        # Find next ready op in [half_lo, half_hi).
        var found = False
        for i in range(start_from, half_hi):
            if placed[i]:
                continue
            if not _is_ready(body, i, placed):
                continue
            # Enforce max globals per block: skip GLOBAL_LOAD if at limit.
            if (
                max_globals_per_block > 0
                and body.ops[i].op.role == OpRole.GLOBAL_LOAD
                and globals_in_block >= max_globals_per_block
            ):
                continue
            order[pos] = i
            placed[i] = True
            last_tried[pos] = i
            pos += 1
            found = True
            break

        if not found:
            last_tried[pos] = -1
            if pos > 0:
                pos -= 1
                placed[order[pos]] = False
            else:
                break  # Exhausted all possibilities.

    return best_order^
