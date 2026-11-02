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

"""Stage 2 must emit exactly the picks its greedy selection emitted.

`_topk_stage2` used to select its picks one at a time: a block-wide reduction
for the best candidate (larger value, ties to the smaller candidate slot), then
that slot overwritten with the dead sentinel. It now orders the candidate set
once, so the values it emits -- and the order it emits them in, equal values
included -- have to stay identical, and so do the indices it reports for them.

The reference is that greedy selection, replayed on the host over the exact
stage-1 candidates the kernel produced. Comparing against the stage-1 output
rather than against the input keeps the check on stage 2: whichever partition
stage 1 chose, and however it ranked ties inside a block, the two selection
rules have to agree on the result.

The rows are built so the check can fail:

- dense ties: a handful of distinct values over the whole row, so the candidate
  set is mostly ties and the slot tie-break decides most of the output order;
- a per-row `k` below `max_k`, so the emission stops on `k` rather than on the
  candidate set;
- rows with fewer live candidates than `max_k` (dead-sentinel holes), so the
  tail padding is compared too;
- rows with NaN holes, which must be skipped rather than ranked.

`largest=False` runs the same check with the comparison flipped, and the
`max_k` sweep covers candidate counts that pad to different power-of-two
networks.
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.topk import (
    _topk_gpu,
    _topk_stage2_smem_budget,
    _topk_stage2_smem_bytes,
    topk_gpu,
)
from std.testing import assert_equal, assert_true
from std.utils.numerics import max_or_inf, min_or_neg_inf, nan

comptime DTYPE = DType.float32
comptime IDX = DType.int64
comptime BLOCK_SIZE = 256
comptime NUM_BLOCKS = 8
# One stage-1 block per 256 elements of the row, so every block has live
# candidates and ties can span blocks.
comptime N = BLOCK_SIZE * NUM_BLOCKS
comptime MAX_K = 32

# How many distinct values the dense-tie rows draw from. Few enough that a
# tie-break disagreement moves the output.
comptime DENSE_VALUES = 7

# The rows `check_rows` puts in one batch.
comptime NUM_TEST_ROWS = 4


def stage2_greedy_reference[
    largest: Bool = True
](
    cand_vals: List[Float32],
    cand_idxs: List[Int64],
    k_batch: Int,
    max_k: Int,
    mut out_vals: List[Float32],
    mut out_idxs: List[Int64],
):
    """The selection `_topk_stage2` used to run, over one row's candidates.

    Fills `out_vals` and `out_idxs` with the row's `max_k` outputs: the best
    remaining candidate over and over (larger value, or smaller for
    `largest=False`, equal values to the smaller slot, so ties come out in slot
    order), each pick's slot taken out of the running, then the sentinel and
    index `-1` for the slots past the `k` picks asked for.

    A candidate the sentinel does not strictly beat -- stage-1 padding, or NaN,
    as neither `x > -inf` nor `x < +inf` holds for it -- is stored as the
    sentinel first. The kernel does the same before sorting, so those
    candidates all tie at the sentinel and an exhausted row emits them in slot
    order.
    """
    var dead = min_or_neg_inf[.float32]()
    comptime if not largest:
        dead = max_or_inf[.float32]()

    var num_cand = len(cand_vals)
    var num_picks = min(k_batch, num_cand)

    var remaining = List[Float32](capacity=num_cand)
    for i in range(num_cand):
        var val = cand_vals[i]
        comptime if largest:
            remaining.append(val if val > dead else dead)
        else:
            remaining.append(val if val < dead else dead)

    # The sentinel is a real value here (`-inf` ties with `-inf`), so picks are
    # taken out of the running by slot rather than by overwriting them.
    var taken = List[Bool](length=num_cand, fill=False)
    var done = False
    for k in range(max_k):
        if done or k >= num_picks:
            out_vals.append(dead)
            out_idxs.append(-1)
            continue
        var best = -1
        var best_val = dead
        for i in range(num_cand):
            if taken[i]:
                continue
            var val = remaining[i]
            comptime if largest:
                if best < 0 or val > best_val:
                    best = i
                    best_val = val
            else:
                if best < 0 or val < best_val:
                    best = i
                    best_val = val
        out_vals.append(best_val)
        out_idxs.append(cand_idxs[best])
        taken[best] = True
        # Once the picked value is the sentinel the row is exhausted: the pick
        # itself still reports the index of the slot it came from, and every
        # slot after it is padding with index `-1`.
        done = best_val == dead


def dense_tie_row(seed: Int) -> List[Float32]:
    """A row of `N` values drawn from `DENSE_VALUES` distinct values."""
    var row = List[Float32](capacity=N)
    var state = seed
    for _ in range(N):
        state = (state * 1103515245 + 12345) % 2147483648
        # Positive and small, so the maximum is unambiguous while most of the
        # row is tied.
        row.append(Float32(Int(state) % DENSE_VALUES))
    return row^


def short_row() -> List[Float32]:
    """A row with three live candidates, spread over three blocks."""
    var dead = min_or_neg_inf[DTYPE]()
    var row = List[Float32](capacity=N)
    for _ in range(N):
        row.append(dead)
    row[900] = 3.0
    row[5] = 2.0
    row[1300] = 1.0
    return row^


def nan_row() -> List[Float32]:
    """A row with NaN holes around a descending ramp."""
    var row = List[Float32](capacity=N)
    for i in range(N):
        if i % 97 == 0:
            row.append(nan[DTYPE]())
        else:
            row.append(Float32(N - i))
    return row^


def bottom_row() -> List[Float32]:
    """A row of small finite values for `largest=False`, plus `+inf` holes."""
    var dead = max_or_inf[DTYPE]()
    var row = List[Float32](capacity=N)
    for i in range(N):
        if i % 89 == 0:
            row.append(dead)
        else:
            row.append(Float32(Int(i) % 13) - 6.0)
    return row^


def test_row_values(kind: Int, largest: Bool) -> List[Float32]:
    """Input row `kind` of the batch."""
    if kind == 0:
        return dense_tie_row(1)
    if kind == 1:
        return dense_tie_row(7)
    if kind == 2:
        return short_row() if largest else bottom_row()
    return nan_row()


def test_row_k(kind: Int, max_k: Int) -> Int64:
    """The `k` asked for on row `kind`."""
    if kind == 1:
        return Int64(max_k // 2)
    return Int64(max_k)


def test_row_label(kind: Int) -> String:
    """What row `kind` is, for failure messages."""
    if kind == 0:
        return "dense ties"
    if kind == 1:
        return "dense ties, half k"
    if kind == 2:
        return "few live candidates"
    return "NaN holes"


def check_rows[
    largest: Bool
](ctx: DeviceContext, max_k: Int, label: String) raises:
    """Runs the two-stage kernel over one batch and diffs every output slot."""
    var batch_size = NUM_TEST_ROWS
    var num_cand = NUM_BLOCKS * max_k

    var in_buffer = ctx.enqueue_create_buffer[DTYPE](batch_size * N)
    with in_buffer.map_to_host() as host:
        var t = TileTensor(host, row_major(batch_size, N))
        for b in range(batch_size):
            var row = test_row_values(b, largest)
            for i in range(N):
                t[b, i] = row[i]

    var k_buffer = ctx.enqueue_create_buffer[.int64](batch_size)
    with k_buffer.map_to_host() as host:
        for b in range(batch_size):
            host[b] = test_row_k(b, max_k)

    var local_vals_buf = ctx.enqueue_create_buffer[DTYPE](batch_size * num_cand)
    var local_idxs_buf = ctx.enqueue_create_buffer[IDX](batch_size * num_cand)
    var out_vals_buf = ctx.enqueue_create_buffer[DTYPE](batch_size * max_k)
    var out_idxs_buf = ctx.enqueue_create_buffer[IDX](batch_size * max_k)

    _topk_gpu[sampling=False, largest=largest](
        ctx,
        max_k,
        TileTensor(in_buffer, row_major(batch_size, N))
        .as_unsafe_any_origin()
        .as_imm(),
        TileTensor(local_vals_buf, row_major(batch_size, num_cand)),
        TileTensor(local_idxs_buf, row_major(batch_size, num_cand)),
        TileTensor(out_vals_buf, row_major(batch_size, max_k)),
        TileTensor(out_idxs_buf, row_major(batch_size, max_k)),
        k=TileTensor(k_buffer, row_major(batch_size))
        .as_unsafe_any_origin()
        .as_imm(),
        block_size=BLOCK_SIZE,
        num_blocks_per_input=NUM_BLOCKS,
    )

    var host_local_vals = List[Float32](
        length=batch_size * num_cand, fill=Float32(0)
    )
    var host_local_idxs = List[Int64](
        length=batch_size * num_cand, fill=Int64(0)
    )
    var host_out_vals = List[Float32](
        length=batch_size * max_k, fill=Float32(0)
    )
    var host_out_idxs = List[Int64](length=batch_size * max_k, fill=Int64(0))
    ctx.enqueue_copy(host_local_vals, local_vals_buf)
    ctx.enqueue_copy(host_local_idxs, local_idxs_buf)
    ctx.enqueue_copy(host_out_vals, out_vals_buf)
    ctx.enqueue_copy(host_out_idxs, out_idxs_buf)
    ctx.synchronize()

    for b in range(batch_size):
        var cand_vals = List[Float32](capacity=num_cand)
        var cand_idxs = List[Int64](capacity=num_cand)
        for i in range(num_cand):
            cand_vals.append(host_local_vals[b * num_cand + i])
            cand_idxs.append(host_local_idxs[b * num_cand + i])
        var want_vals = List[Float32](capacity=max_k)
        var want_idxs = List[Int64](capacity=max_k)
        stage2_greedy_reference[largest](
            cand_vals,
            cand_idxs,
            Int(test_row_k(b, max_k)),
            max_k,
            want_vals,
            want_idxs,
        )
        for k in range(max_k):
            var context = (
                label
                + ", row "
                + String(b)
                + " ("
                + test_row_label(b)
                + "), slot "
                + String(k)
            )
            assert_equal(host_out_vals[b * max_k + k], want_vals[k], context)
            assert_equal(host_out_idxs[b * max_k + k], want_idxs[k], context)

    _ = in_buffer^
    _ = k_buffer^
    _ = local_vals_buf^
    _ = local_idxs_buf^
    _ = out_vals_buf^
    _ = out_idxs_buf^


def check_apple_static_budget(ctx: DeviceContext) raises:
    """A candidate set that pads past the Apple kernel's static allocation.

    `max_k = 65` over eight stage-1 blocks is 520 candidates, which the sort
    pads to 1024 -- 12 KiB of candidate values and slots. That fits the SM's
    shared memory, but on Apple stage 2 carves everything out of one static
    allocation of `_topk_stage2_apple_static_bytes` (7680 B for `float32`), so
    the partition count has to come down before the launch and the request has
    to be measured against that allocation rather than the SM's.

    Two checks: the invariant that makes the shrink necessary and sufficient
    (on Apple the eight-block request is past the budget), and the results,
    which an overrun would corrupt. The row values are distinct and descending,
    so the true picks are indices `0..max_k` in order.
    """
    comptime max_k = 65
    comptime N = 4096
    var batch_size = 2

    comptime if ctx.target.is_apple_gpu():
        assert_true(
            _topk_stage2_smem_bytes[DTYPE](max_k, NUM_BLOCKS)
            > _topk_stage2_smem_budget[DTYPE](ctx),
            "an eight-block request of "
            + String(_topk_stage2_smem_bytes[DTYPE](max_k, NUM_BLOCKS))
            + " bytes is supposed to be past the Apple budget of "
            + String(_topk_stage2_smem_budget[DTYPE](ctx))
            + " bytes, which is what makes the shrink necessary",
        )
        # bfloat16 halves the allocation (the value count is fixed, each value
        # is half the size) and its candidates cost 2 + 8 bytes instead of
        # 4 + 8, so its padded request overruns by more and has to trip the
        # same check.
        assert_true(
            _topk_stage2_smem_bytes[.bfloat16](max_k, NUM_BLOCKS)
            > _topk_stage2_smem_budget[.bfloat16](ctx),
            "the bfloat16 eight-block request of "
            + String(_topk_stage2_smem_bytes[.bfloat16](max_k, NUM_BLOCKS))
            + " bytes is supposed to be past the Apple budget of "
            + String(_topk_stage2_smem_budget[.bfloat16](ctx))
            + " bytes",
        )

    var in_buffer = ctx.enqueue_create_buffer[DTYPE](batch_size * N)
    with in_buffer.map_to_host() as host:
        var t = TileTensor(host, row_major(batch_size, N))
        for b in range(batch_size):
            for i in range(N):
                t[b, i] = Float32((b + 1) * (N - i))

    var out_vals_buf = ctx.enqueue_create_buffer[DTYPE](batch_size * max_k)
    var out_idxs_buf = ctx.enqueue_create_buffer[IDX](batch_size * max_k)

    topk_gpu[sampling=False, largest=True](
        ctx,
        max_k,
        TileTensor(in_buffer, row_major(batch_size, N))
        .as_unsafe_any_origin()
        .as_imm(),
        TileTensor(out_vals_buf, row_major(batch_size, max_k)),
        TileTensor(out_idxs_buf, row_major(batch_size, max_k)),
    )

    var host_vals = List[Float32](length=batch_size * max_k, fill=Float32(0))
    var host_idxs = List[Int64](length=batch_size * max_k, fill=Int64(0))
    ctx.enqueue_copy(host_vals, out_vals_buf)
    ctx.enqueue_copy(host_idxs, out_idxs_buf)
    ctx.synchronize()

    for b in range(batch_size):
        for k in range(max_k):
            var context = (
                "budget edge, row " + String(b) + ", slot " + String(k)
            )
            assert_equal(
                host_vals[b * max_k + k], Float32((b + 1) * (N - k)), context
            )
            assert_equal(host_idxs[b * max_k + k], Int64(k), context)

    _ = in_buffer^
    _ = out_vals_buf^
    _ = out_idxs_buf^


def main() raises:
    with DeviceContext() as ctx:
        # `max_k` values whose candidate counts pad to different networks:
        # 32 * 8 is an exact power of two, 20 * 8 pads 160 up to 256, 8 * 8 is
        # the smallest candidate set spanning more than a warp, and 1 * 8 is
        # the greedy case.
        for max_k in [32, 20, 8, 1]:
            check_rows[True](ctx, max_k, "largest, k=" + String(max_k))
        check_rows[False](ctx, MAX_K, "largest=False, k=" + String(MAX_K))
        check_apple_static_budget(ctx)
        print("test_topk_stage2_order: OK")
