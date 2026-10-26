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
#
# Negative control for the compute-sanitizer `synccheck` lane.
#
# Both kernels here are CORRECT. They rendezvous on a `named_barrier` from two
# distinct program points, which is what a warp-specialized kernel does and
# what PTX permits: `named_barrier[N](id)` lowers to `bar.sync id, N`, and the
# participating warps may reach it from different instruction addresses so long
# as they all arrive and all name the same count. synccheck models a barrier as
# a single instruction address, so it reports
#
#   Barrier error detected. Divergent thread(s) in block
#
# for both of them. That report is the tool's limitation, not a defect: this
# file is the standing proof of it, so the SM100 attention/MLA findings do not
# have to be re-argued from source comments each time the sweep runs. The
# findings classifier in `run_sanitizer.sh` keys off exactly this distinction --
# a divergence attributed to a `named_barrier` line is noise, a divergence
# attributed to a plain `barrier()` is not.
#
# Run it explicitly and expect BOTH halves:
#
#   CS=max/kernels/test/gpu/compute_sanitizer
#   $CS/run_sanitizer.sh synccheck \
#     //$CS:negative_control_synccheck_named_barrier.mojo.test
#
#   1. the raw `bazel-testlogs/.../test.log` carries "Divergent thread(s) in
#      block" and the bazel target FAILS, because `--error-exitcode 1` turns
#      the report into a non-zero exit -- that is synccheck flagging a kernel
#      this file proves correct; and
#   2. the sweep's findings summary is nevertheless EMPTY and the script exits
#      0, because `classify_synccheck.py` resolved the report to a
#      `named_barrier` line and dropped it.
#
# Half 1 failing to appear means the tool changed and the filter may now be
# suppressing something real. Half 2 failing to appear means the filter has
# stopped recognizing the pattern and the lane is about to go red on noise.
#
# Measured on B200, compute-sanitizer 2025.4.1: `two_warpgroup_handoff` is
# reported 64 times (ERROR SUMMARY: 64) with no assertion failure, and the
# classifier suppresses all of it. `partial_cta_handoff` is NOT reported --
# see its docstring. A run where NEITHER kernel is reported means this control
# has stopped working, not that synccheck has been fixed; the two look
# identical from the outside and only the first is evidence.
#
# The host-side checks below assert the handoffs actually happened, so "correct
# by construction" is verified rather than asserted: if synccheck ever stops
# reporting these, the lane's noise classification should be revisited, and if
# these kernels ever stop producing the right answer the control itself is
# broken.
#
# Tagged `manual` so it is excluded from `//...` and
# `//max/kernels/test/gpu/...` wildcard expansion.

from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from max.gpu import thread_idx
from max.gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.sync import named_barrier

# ===-----------------------------------------------------------------------===#
# Arm 1: two warpgroups, whole-CTA rendezvous from two program points
# ===-----------------------------------------------------------------------===#

comptime PAIR_THREADS = 2 * WARPGROUP_SIZE
comptime PAIR_BAR: Int32 = 2


def two_warpgroup_handoff(dst: MutPointer[Int32, MutAnyOrigin]):
    """WG1 publishes and leaves; WG0 rendezvouses later and consumes.

    This mirrors the SM100 softmax epilogue, where WG1 takes the pair barrier
    at an early-out and returns while WG0 reaches the same barrier further down
    after its own epilogue. Both warpgroups always arrive and both name
    `2 * WARPGROUP_SIZE`, so the rendezvous is well defined.
    """
    var tid = Int(thread_idx.x)
    var wg = tid // WARPGROUP_SIZE
    var slot = tid % WARPGROUP_SIZE

    var published = unsafe_stack_allocation[
        WARPGROUP_SIZE, Int32, address_space=.SHARED
    ]()

    if wg == 1:
        published[unsafe_offset=slot] = Int32(slot + 1)
        # Program point A.
        named_barrier[Int32(PAIR_THREADS)](PAIR_BAR)
        dst[unsafe_offset=tid] = Int32(slot + 1)
        return

    # WG0 does work of its own before the rendezvous, so the two arrivals are
    # genuinely different program points rather than one the compiler folded.
    var acc = Int32(0)
    for i in range(slot + 1):
        acc += Int32(i)

    # Program point B.
    named_barrier[Int32(PAIR_THREADS)](PAIR_BAR)

    # Only reachable after WG1's publish, which is the point of the barrier.
    dst[unsafe_offset=tid] = published[unsafe_offset=slot] + acc


def check_two_warpgroup_handoff() raises:
    with DeviceContext() as ctx:
        var dst = ctx.enqueue_create_buffer[.int32](PAIR_THREADS)
        ctx.enqueue_function[two_warpgroup_handoff](
            dst,
            grid_dim=(1,),
            block_dim=(PAIR_THREADS,),
        )
        var host = ctx.enqueue_create_host_buffer[.int32](PAIR_THREADS)
        ctx.enqueue_copy(host, dst)
        ctx.synchronize()

        for tid in range(PAIR_THREADS):
            var slot = tid % WARPGROUP_SIZE
            var expected = Int32(slot + 1)
            if tid < WARPGROUP_SIZE:
                expected += Int32(slot * (slot + 1) // 2)
            assert_equal(host[tid], expected)


# ===-----------------------------------------------------------------------===#
# Arm 2: partial-CTA rendezvous, consumers plus one extra warp
# ===-----------------------------------------------------------------------===#

comptime CONS_WARPS = 4
comptime CONS_THREADS = CONS_WARPS * WARP_SIZE
comptime MMA_WARP = CONS_WARPS
comptime TMA_WARP = CONS_WARPS + 1
comptime PARTIAL_THREADS = (CONS_WARPS + 2) * WARP_SIZE
comptime PARTIAL_BAR: Int32 = 1


def partial_cta_handoff(dst: MutPointer[Int32, MutAnyOrigin]):
    """Consumers and one producer warp rendezvous; a third role never joins.

    This mirrors `sparse_index_fp8_sm100_prefill`: the consumer warps stage
    their inputs and the MMA warp waits for that staging, both naming
    `CONS_THREADS + WARP_SIZE`, while the TMA warp does independent work and
    never touches this barrier id. Only the named participants arrive, so the
    count is exact.

    NOT the proof, and does not currently carry any: measured on B200 with
    compute-sanitizer 2025.4.1, synccheck reports nothing on this kernel,
    while `two_warpgroup_handoff` above is reported 64 times. So the proof
    rests on that one, and this is a correctness test of the partial-CTA
    shape rather than a second demonstration.

    Worth someone's time, because it is unexplained: the ticket records
    `sparse_index_fp8_sm100_prefill` being flagged at its own partial-CTA
    rendezvous, so the tool clearly does report this shape somewhere, and
    the difference between that kernel and this one is not the barrier
    count. Do not read a silent run here as evidence the tool stopped
    reporting -- check `two_warpgroup_handoff` for that.
    """
    var tid = Int(thread_idx.x)
    var wid = tid // WARP_SIZE

    var staged = unsafe_stack_allocation[
        CONS_THREADS, Int32, address_space=.SHARED
    ]()
    var ready = unsafe_stack_allocation[1, Int32, address_space=.SHARED]()

    if wid < CONS_WARPS:
        staged[unsafe_offset=tid] = Int32(tid + 1)
        # Program point A: the consumers' side of the rendezvous.
        named_barrier[Int32(CONS_THREADS + WARP_SIZE)](PARTIAL_BAR)
        # Reading what the producer published makes the handoff bidirectional,
        # so neither side's pre-barrier store can be treated as dead.
        dst[unsafe_offset=tid] = Int32(tid + 1) + ready[unsafe_offset=0]
        return

    if wid == MMA_WARP:
        # Work of its own before the rendezvous. Without it the two arrivals
        # are textually identical in both arms and the compiler tail-merges
        # them into ONE instruction address -- at which point synccheck sees a
        # single arrival site, reports nothing, and this arm silently proves
        # nothing. (Measured: it did exactly that before this was added.) The
        # real kernel this mirrors allocates TMEM here, which is what keeps
        # its two arrivals distinct.
        if tid == MMA_WARP * WARP_SIZE:
            ready[unsafe_offset=0] = Int32(CONS_THREADS)

        # Program point B: the producer's side, reached from a different arm.
        named_barrier[Int32(CONS_THREADS + WARP_SIZE)](PARTIAL_BAR)
        # One lane folds what the consumers staged, which is only defined
        # because of the barrier above.
        var lane = tid - MMA_WARP * WARP_SIZE
        var total = Int32(0)
        for i in range(lane, CONS_THREADS, WARP_SIZE):
            total += staged[unsafe_offset=i]
        dst[unsafe_offset=tid] = total
        return

    # The TMA warp shares the CTA but not the barrier.
    dst[unsafe_offset=tid] = Int32(-1)


def check_partial_cta_handoff() raises:
    with DeviceContext() as ctx:
        var dst = ctx.enqueue_create_buffer[.int32](PARTIAL_THREADS)
        ctx.enqueue_function[partial_cta_handoff](
            dst,
            grid_dim=(1,),
            block_dim=(PARTIAL_THREADS,),
        )
        var host = ctx.enqueue_create_host_buffer[.int32](PARTIAL_THREADS)
        ctx.enqueue_copy(host, dst)
        ctx.synchronize()

        for tid in range(CONS_THREADS):
            assert_equal(host[tid], Int32(tid + 1) + Int32(CONS_THREADS))

        for lane in range(WARP_SIZE):
            var expected = Int32(0)
            for i in range(lane, CONS_THREADS, WARP_SIZE):
                expected += Int32(i + 1)
            assert_equal(host[MMA_WARP * WARP_SIZE + lane], expected)

        for lane in range(WARP_SIZE):
            assert_equal(host[TMA_WARP * WARP_SIZE + lane], Int32(-1))


def main() raises:
    check_two_warpgroup_handoff()
    check_partial_cta_handoff()
