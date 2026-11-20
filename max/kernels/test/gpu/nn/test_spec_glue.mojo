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
"""Tests for the folded glue of the speculative-decode accept path.

`first_rejected_index` and `accept_pair` each replace a chain of whole-tensor
graph ops over a tensor of a few dozen entries with one launch. Every value
involved is an integer or an elementwise float with no reordering, so the bar
here is equality with the chain, not a tolerance: the first rejected position
must be the index the sentinel-concat/multiply/argmax chain produced, and the
accept pair must be the rejection bit and the fallback token the gather /
compare / select / multiply chain produced -- including the arm where the row
carried no proposal and the residual subtracts a one-hot at the drafted token.
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.sampling.spec_glue import accept_pair, first_rejected_index
from std.testing import assert_equal
from std.utils import StaticTuple


def _reference_first_rejected(row: List[Bool]) -> Int:
    """The index the graph chain produced: first True, else the row length.

    That chain appended a sentinel True, weighted position `i` by
    `len(row) + 1 - i`, and took the argmax, so the smallest True index wins
    and "none rejected" lands on the row length.
    """
    for i in range(len(row)):
        if row[i]:
            return i
    return len(row)


def _check_first_rejected(
    ctx: DeviceContext, label: String, mask: List[List[Bool]]
) raises:
    var rows = len(mask)
    var steps = len(mask[0])
    var mask_host = ctx.enqueue_create_host_buffer[DType.bool](rows * steps)
    # Two guard elements past the output so an overrun is visible: they must
    # still hold the poison value afterwards.
    var out_host = ctx.enqueue_create_host_buffer[DType.int64](rows + 2)
    ctx.synchronize()

    var mask_rows = TileTensor(mask_host, row_major(rows, steps))
    for r in range(rows):
        for c in range(steps):
            mask_rows[r, c] = Scalar[.bool](mask[r][c])

    var mask_dev = ctx.enqueue_create_buffer[DType.bool](rows * steps)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](rows + 2)
    ctx.enqueue_copy(mask_dev, mask_host)
    ctx.enqueue_memset(out_dev, -7)

    first_rejected_index(
        TileTensor(mask_dev, row_major(rows, steps)),
        TileTensor(out_dev, row_major(rows, 1)),
        ctx,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for r in range(rows):
        assert_equal(
            Int(out_host[r]),
            _reference_first_rejected(mask[r]),
            msg=String(t"{label}: row {r} first rejected"),
        )
    for tail in range(rows, rows + 2):
        assert_equal(
            Int(out_host[tail]),
            -7,
            msg=String(t"{label}: wrote past the output"),
        )

    _ = mask_dev^
    _ = out_dev^


def test_first_rejected_cases(ctx: DeviceContext) raises:
    """Every accept verdict the sampler can reach, at K3's decode shape."""
    # K3 drafts 7 positions and verifies 8, at batch 4: the mask the verdict
    # builds is [batch, 7].
    var batch = 4
    var steps = 7
    var all_accepted = List[List[Bool]]()
    var first_position = List[List[Bool]]()
    var last_position = List[List[Bool]]()
    var alternating = List[List[Bool]]()
    for b in range(batch):
        var accepted_row = List[Bool]()
        var first_row = List[Bool]()
        var last_row = List[Bool]()
        var alt_row = List[Bool]()
        for s in range(steps):
            accepted_row.append(False)
            first_row.append(s == 0)
            last_row.append(s == steps - 1)
            alt_row.append((s + b) % 2 == 0)
        all_accepted.append(accepted_row^)
        first_position.append(first_row^)
        last_position.append(last_row^)
        alternating.append(alt_row^)

    _check_first_rejected(ctx, "all accepted", all_accepted)
    _check_first_rejected(ctx, "first position", first_position)
    _check_first_rejected(ctx, "last position", last_position)
    _check_first_rejected(ctx, "alternating", alternating)


def test_first_rejected_odd_shapes(ctx: DeviceContext) raises:
    """One row, one position, and a row count past a thread block."""
    _check_first_rejected(ctx, "single accepted", [[False]])
    _check_first_rejected(ctx, "single rejected", [[True]])

    var accepted_row = List[Bool]()
    var rejected_row = List[Bool]()
    for _ in range(8):
        accepted_row.append(False)
        rejected_row.append(True)
    var one_accepted = List[List[Bool]]()
    one_accepted.append(accepted_row^)
    var one_rejected = List[List[Bool]]()
    one_rejected.append(rejected_row^)
    _check_first_rejected(ctx, "one accepted row", one_accepted)
    _check_first_rejected(ctx, "one rejected row", one_rejected)

    # 513 rows: more than one thread block on every target (the block size is
    # 128 on B200 and 256 on CDNA4), so a second block has to run.
    var wide = List[List[Bool]]()
    for r in range(513):
        var row = List[Bool]()
        row.append(r % 3 == 0)
        wide.append(row^)
    _check_first_rejected(ctx, "multi block", wide)


comptime _PA_ROWS = 2
comptime _PA_STEPS = 3
comptime _PA_VOCAB = 4
comptime _PA_FLAT = _PA_ROWS * _PA_STEPS * _PA_VOCAB

# The masked target mass at each drafted token, with 9.0 everywhere else so a
# read at the wrong vocabulary index changes the verdict.
comptime _PA_PROBS = StaticTuple[Float32, _PA_FLAT](
    9.0,
    0.9,
    9.0,
    9.0,
    9.0,
    9.0,
    0.2,
    9.0,
    9.0,
    9.0,
    9.0,
    0.3,
    0.1,
    9.0,
    9.0,
    9.0,
    9.0,
    0.5,
    9.0,
    9.0,
    9.0,
    0.4,
    9.0,
    9.0,
)
# The proposal's mass there. Two positions carry none, which is the arm where
# the accept test takes q = 1 and the residual subtracts a one-hot instead.
comptime _PA_DISTS = StaticTuple[Float32, _PA_FLAT](
    9.0,
    0.5,
    9.0,
    9.0,
    9.0,
    9.0,
    1.0,
    9.0,
    9.0,
    9.0,
    9.0,
    0.0,
    0.0,
    9.0,
    9.0,
    9.0,
    9.0,
    0.5,
    9.0,
    9.0,
    9.0,
    0.25,
    9.0,
    9.0,
)
comptime _PA_TOKENS = StaticTuple[Int64, 6](1, 2, 3, 0, 1, 1)
comptime _PA_COINS = StaticTuple[Float32, 6](0.1, 0.5, 0.9, 0.05, 1.0, 0.4)
# coin * q_eff >= p_target, with q_eff = 1 where the row has no proposal.
comptime _PA_WANT_REJECTED = StaticTuple[Bool, 6](
    False, True, True, False, True, False
)
comptime _PA_WANT_FALLBACK = StaticTuple[Int64, 6](-1, -1, 3, 0, -1, -1)


def test_accept_pair_arms(ctx: DeviceContext) raises:
    """Both accept outcomes, both residual fallbacks, one launch.

    Row 0's last position and row 1's first carry no proposal (`q == 0`), so
    those must take q = 1 in the test and fall back to a one-hot at the drafted
    token in the residual. Every other vocabulary entry holds 9.0, so reading
    the distributions at the wrong index flips the expected verdict.
    """
    var probs_host = ctx.enqueue_create_host_buffer[DType.float32](_PA_FLAT)
    var dists_host = ctx.enqueue_create_host_buffer[DType.float32](_PA_FLAT)
    var tokens_host = ctx.enqueue_create_host_buffer[DType.int64](
        _PA_ROWS * _PA_STEPS
    )
    var coins_host = ctx.enqueue_create_host_buffer[DType.float32](
        _PA_ROWS * _PA_STEPS
    )
    var rejected_host = ctx.enqueue_create_host_buffer[DType.bool](
        _PA_ROWS * _PA_STEPS
    )
    var fallback_host = ctx.enqueue_create_host_buffer[DType.int64](
        _PA_ROWS * _PA_STEPS
    )
    ctx.synchronize()

    for k in range(_PA_FLAT):
        probs_host[k] = _PA_PROBS[k]
        dists_host[k] = _PA_DISTS[k]
    for k in range(_PA_ROWS * _PA_STEPS):
        tokens_host[k] = _PA_TOKENS[k]
        coins_host[k] = _PA_COINS[k]

    var probs_dev = ctx.enqueue_create_buffer[DType.float32](_PA_FLAT)
    var dists_dev = ctx.enqueue_create_buffer[DType.float32](_PA_FLAT)
    var tokens_dev = ctx.enqueue_create_buffer[DType.int64](
        _PA_ROWS * _PA_STEPS
    )
    var coins_dev = ctx.enqueue_create_buffer[DType.float32](
        _PA_ROWS * _PA_STEPS
    )
    var rejected_dev = ctx.enqueue_create_buffer[DType.bool](
        _PA_ROWS * _PA_STEPS
    )
    var fallback_dev = ctx.enqueue_create_buffer[DType.int64](
        _PA_ROWS * _PA_STEPS
    )
    ctx.enqueue_copy(probs_dev, probs_host)
    ctx.enqueue_copy(dists_dev, dists_host)
    ctx.enqueue_copy(tokens_dev, tokens_host)
    ctx.enqueue_copy(coins_dev, coins_host)

    var probs_view = TileTensor(
        probs_dev, row_major(_PA_ROWS, _PA_STEPS, _PA_VOCAB)
    )
    var dists_view = TileTensor(
        dists_dev, row_major(_PA_ROWS, _PA_STEPS, _PA_VOCAB)
    )
    var tokens_view = TileTensor(tokens_dev, row_major(_PA_ROWS, _PA_STEPS))
    var coins_view = TileTensor(coins_dev, row_major(_PA_ROWS, _PA_STEPS))
    accept_pair(
        probs_view.as_imm(),
        dists_view.as_imm(),
        tokens_view.as_imm(),
        coins_view.as_imm(),
        TileTensor(rejected_dev, row_major(_PA_ROWS, _PA_STEPS)),
        TileTensor(fallback_dev, row_major(_PA_ROWS, _PA_STEPS)),
        ctx,
    )
    ctx.enqueue_copy(rejected_host, rejected_dev)
    ctx.enqueue_copy(fallback_host, fallback_dev)
    ctx.synchronize()

    for k in range(_PA_ROWS * _PA_STEPS):
        assert_equal(
            Bool(rejected_host[k]),
            _PA_WANT_REJECTED[k],
            msg=String(t"accept_pair rejection bit at position {k}"),
        )
        assert_equal(
            Int(fallback_host[k]),
            Int(_PA_WANT_FALLBACK[k]),
            msg=String(t"accept_pair fallback token at position {k}"),
        )
    _ = probs_dev^
    _ = dists_dev^
    _ = tokens_dev^
    _ = coins_dev^
    _ = rejected_dev^
    _ = fallback_dev^


def main() raises:
    with DeviceContext() as ctx:
        test_first_rejected_cases(ctx)
        test_first_rejected_odd_shapes(ctx)
        test_accept_pair_arms(ctx)
