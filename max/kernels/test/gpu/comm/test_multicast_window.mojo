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
"""The multicast boundary's symmetric window: allocation, shape and start state.

The window is the session-scoped resource the allreduce dispatch takes its
multicast arm through, so what has to hold before any kernel reads it is that it
allocates once per device set and geometry, that it is sized for the band, and
that it starts in the state the protocol reads: this rank's call state zeroed
(call 0, nothing written, nothing to clear), the payload holding the `-0.0`
"nothing arrived" sentinel, and the arrival words at zero.

The fourth claim is the capture one. Serving captures every decode batch size
before it runs one, so a geometry's first call lands inside a device graph, and
the window has to be created there with the arm in that graph rather than
deferred past it.
"""

from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.testing import assert_equal, assert_true

from comm.allreduce_residual_rmsnorm import (
    MCAR_GROUPED_MAX_ROWS,
    MCAR_NORM_MAX_ROWS,
    mcar_rmsnorm_fits,
)
from comm.lamport import LAMPORT_SENTINEL_U32
from comm.multicast_window import (
    MCAR_BAND_BYTES,
    MCAR_GENERATIONS,
    MCAR_STATE_WORDS,
    McarWindow,
    mcar_payload_index,
    mcar_window,
    mcar_window_key,
    mcar_window_rows,
)
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)

comptime NG = 2

comptime DTYPE = DType.bfloat16

comptime COLS = 4096
"""MiMo-V2's hidden size: 128 rows of bf16 fill the band exactly."""

comptime NSWORDS = 4
"""`uint32` words per 16-byte pack: how much of one slot to check."""


def _window_inside_a_capture[
    ngpus: Int
](ctxs: Array[DeviceContext[], ngpus], cols: Int) raises -> Optional[
    Pointer[McarWindow[ngpus], MutUntrackedOrigin]
]:
    """The window the team gets while a device graph is recording, if any.

    The window is host work -- a symmetric allocation across the team, and one
    synchronize per rank to order its fills -- but none of it is work a capture
    forbids: capture is RELAXED, and the allocation runs on the contexts
    `mcar_window` creates for itself rather than on the recording stream. So the
    first call for a geometry may land inside a capture, and creating the window
    there is what keeps the arm in that graph. This pins the don't-refuse rule:
    rank 0 asks through the recording context, which reports
    `is_graph_capturing()`, and the call still has to produce the window.

    A device graph built node by node is the closest stand-in for a capture a
    kernel test can make: it is not a stream capture, so it cannot show that the
    real capture mode tolerates the allocation -- the shipped traces do -- only
    that nothing here consults the capture state to refuse one.

    Args:
        ctxs: One context per rank, in rank order.
        cols: Row width to ask for, at a geometry nothing has created yet.

    Returns:
        The window the capture got, or `None` when the device set cannot map
        multicast at all.
    """
    var got = List[Pointer[McarWindow[ngpus], MutUntrackedOrigin]](capacity=1)

    def build(
        mut builder: DeviceGraphBuilder[_],
    ) raises {mut got, imm ctxs, imm cols,}:
        # A copy per rank, and rank 0's replaced by the recording context while
        # the capture is live: `ctxs.copy()` then one assignment, which is how
        # the other sites put a borrowed context into a team array.
        var team = ctxs.copy()
        with builder.recording_context() as rec:
            team[0] = rec.copy()
            if case .Some(w) = mcar_window[ngpus, DTYPE](team, cols):
                got.append(w)

    _ = DeviceGraph.create(ctxs[0], build)
    if len(got) == 1:
        return Optional(got[0])
    return None


def _read_words(
    ctx: DeviceContext, ptr: MutPointer[UInt32, MutUntrackedOrigin], count: Int
) raises -> List[UInt32]:
    """Copies `count` words of one rank's device memory back to the host.

    The window keeps its buffers out of reach of any destructor, so the test
    reads through a non-owning view: dropping it frees nothing.
    """
    var buffer = DeviceBuffer[DType.uint32](ctx, ptr, count, owning=False)
    var host = unsafe_alloc[UInt32](count)
    ctx.enqueue_copy(host, buffer)
    ctx.synchronize()
    var out = List[UInt32](capacity=count)
    for i in range(count):
        out.append(host[unsafe_offset=i])
    host.unsafe_free()
    return out^


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() >= NG,
        "the test needs one device per rank",
    )

    # The band: 128 rows of 4096 bf16 are one 1 MiB message, the bucket the
    # multicast arm replaces.
    var rows = mcar_window_rows[DTYPE](COLS)
    assert_equal(rows, MCAR_BAND_BYTES // (COLS * size_of[DTYPE]()))
    assert_equal(rows, 128)
    # A row wider than the band slot has no capacity at all, which is the shape
    # the dispatch has to keep off this arm.
    assert_equal(mcar_window_rows[DTYPE](MCAR_BAND_BYTES), 0)
    # Both epilogues are capped below the band, each for its own reason: a
    # shipped arm is cheaper at equal work for the normed row, and the arm costs
    # draft acceptance for the grouped one. So the band is not what bounds an
    # epilogue, its cap is -- a row count the window holds is still not served.
    assert_true(
        mcar_rmsnorm_fits[DTYPE, NG, quantized=False](MCAR_NORM_MAX_ROWS, COLS)
    )
    assert_true(
        not mcar_rmsnorm_fits[DTYPE, NG, quantized=False](
            MCAR_NORM_MAX_ROWS + 1, COLS
        )
    )
    assert_true(
        mcar_rmsnorm_fits[DTYPE, NG, quantized=True](
            MCAR_GROUPED_MAX_ROWS, COLS
        )
    )
    assert_true(
        not mcar_rmsnorm_fits[DTYPE, NG, quantized=True](
            MCAR_GROUPED_MAX_ROWS + 1, COLS
        )
    )
    assert_true(MCAR_NORM_MAX_ROWS < rows)
    assert_true(MCAR_GROUPED_MAX_ROWS < rows)
    # One pack a thread: a row past the geometry's ceiling is not served.
    assert_true(
        not mcar_rmsnorm_fits[DTYPE, NG, quantized=True](1, COLS * 1024)
    )
    # fp32 transport needs twice the slot for the same rows.
    assert_equal(mcar_window_rows[DType.float32](COLS), rows // 2)

    var dev_ctxs = List[DeviceContext[]](capacity=NG)
    for i in range(NG):
        dev_ctxs.append(DeviceContext(device_id=i))
    var ctxs = Array[DeviceContext[], NG](
        fill_with_unrolled=lambda [i: Int]() -> DeviceContext[]: dev_ctxs[
            i
        ].copy()
    )
    if case .Some(w) = mcar_window[NG, DTYPE](ctxs, COLS):
        assert_equal(w[].rows(), rows)
        assert_equal(w[].cols, COLS)
        assert_equal(w[].dtype, DTYPE)

        # One allocation per device set and geometry, call state included.
        if case .Some(w2) = mcar_window[NG, DTYPE](ctxs, COLS):
            assert_true(
                Int(w2[].rank(0).state) == Int(w[].rank(0).state),
                "the window is not memoized",
            )
        else:
            assert_true(False, "the memoized window went missing")

        var ctx0 = ctxs[0]
        # Start state: zeroed call state (call 0, nothing to clear).
        var rank0 = w[].rank(0)
        var state = _read_words(ctx0, rank0.state, MCAR_STATE_WORDS)
        for i in range(MCAR_STATE_WORDS):
            assert_equal(state[i], UInt32(0))
        # Start state: "nothing arrived", in every rank slot of the window.
        var slots = _read_words(ctx0, rank0.payload, NG * NSWORDS)
        for i in range(len(slots)):
            assert_equal(slots[i], LAMPORT_SENTINEL_U32)
        # The two addresses a kernel uses are different memory: the multicast
        # view is not this rank's copy.
        assert_true(
            Int(rank0.payload_multicast) != Int(rank0.payload),
            "the window's multicast view aliases the local copy",
        )

        # A capture opened before any eager call at a geometry is where that
        # geometry's window is created: capture is RELAXED and the allocation
        # runs on the window's own contexts, so the arm is in the graph. Serving
        # captures every decode batch size before it runs one
        # (`warmup_pre_ready`), so this is the ordinary case, not a corner.
        if case .Some(captured) = _window_inside_a_capture[NG](ctxs, COLS // 2):
            assert_equal(captured[].cols, COLS // 2)
            # ... and the capture did not cost the team the window: the same
            # allocation answers an eager call, so nothing was recorded as
            # unavailable.
            if case .Some(eager) = mcar_window[NG, DTYPE](ctxs, COLS // 2):
                assert_true(
                    Int(eager[].rank(0).state) == Int(captured[].rank(0).state),
                    "a capture created the window but the team lost it",
                )
            else:
                assert_true(False, "the window a capture created went missing")
            print(
                "multicast window: a capture creates the window, arm stays in"
                " the graph"
            )
        else:
            assert_true(False, "no window while a device graph was recording")
        print("multicast window: PASS at", NG, "ranks,", rows, "rows")
    else:
        # A device set without NVLS, and every HAL build, must fall back to the
        # existing paths rather than fail.
        print("multicast window: NOT SUPPORTED on this device set (fallback)")

    # The layout the kernel indexes with: `[gen][row][rank][col]`.
    assert_equal(mcar_payload_index(0, 0, 0, 0, rows, NG, COLS), 0)
    assert_equal(mcar_payload_index(0, 1, 0, 0, rows, NG, COLS), NG * COLS)
    assert_equal(mcar_payload_index(0, 0, 1, 0, rows, NG, COLS), COLS)
    assert_equal(
        mcar_payload_index(1, 0, 0, 0, rows, NG, COLS), rows * NG * COLS
    )
    assert_true(
        MCAR_GENERATIONS >= 2, "a rotating window needs two generations"
    )

    # The window is keyed by the ordered device ids, not just the team size: two
    # device sets of the same size must not share one window's mapping or its
    # call state.
    var ids_a = Array[Int32, NG](
        fill_with_unrolled=lambda [i: Int]() -> Int32: Int32(i)
    )
    var ids_b = Array[Int32, NG](
        fill_with_unrolled=lambda [i: Int]() -> Int32: Int32(i + 2)
    )
    var ids_swapped = Array[Int32, NG](
        fill_with_unrolled=lambda [i: Int]() -> Int32: Int32(NG - 1 - i)
    )
    assert_equal(
        mcar_window_key[DTYPE](ids_a, COLS),
        mcar_window_key[DTYPE](ids_a.copy(), COLS),
    )
    assert_true(
        mcar_window_key[DTYPE](ids_a, COLS)
        != mcar_window_key[DTYPE](ids_b, COLS)
    )
    assert_true(
        mcar_window_key[DTYPE](ids_a, COLS)
        != mcar_window_key[DTYPE](ids_swapped, COLS)
    )
    assert_true(
        mcar_window_key[DTYPE](ids_a, COLS)
        != mcar_window_key[DTYPE](ids_a, COLS // 2)
    )
    # The message dtype is part of the geometry: a wider element type holds
    # fewer rows in the same slot, so the two windows cannot be one allocation.
    assert_true(
        mcar_window_key[DTYPE](ids_a, COLS)
        != mcar_window_key[DType.float32](ids_a, COLS)
    )
