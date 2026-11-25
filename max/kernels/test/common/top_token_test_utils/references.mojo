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
"""Float64 references and logit fillers shared by the top-token tests."""

from std.math import exp, log
from std.testing import assert_equal
from std.utils.numerics import inf, min_finite, neg_inf

from builtin_kernels.monoids import TopTokenLogit, TopTokenPieces


def scrambled_logit(row: Int, col: Int, d: Int) -> Float64:
    """A deterministic, non-monotonic fill spanning a few logit decades."""
    var h = (col * 2654435761 + row * 97) % 1_000_003
    return Float64(h) / 1_000_003.0 * 12.0 - 6.0


def reference_score(logits: List[Float64]) -> Float64:
    """The score from the definition, in float64.

    A row whose only finite logit is the top has no excluded mass and
    scores `+inf`; a row with no finite logit scores `+inf` too.
    """
    var d = len(logits)
    var l_max = neg_inf[DType.float64]()
    for i in range(d):
        if logits[i] > l_max:
            l_max = logits[i]

    if l_max == neg_inf[DType.float64]():
        return inf[DType.float64]()
    var excluded = 0.0
    var ties = 0
    for i in range(d):
        if logits[i] == l_max:
            ties += 1
        elif logits[i] > neg_inf[DType.float64]():
            excluded += exp(logits[i] - l_max)

    var mass = excluded + Float64(ties - 1)
    if mass <= 0.0:
        return inf[DType.float64]()
    return -log(mass)


def reference_pieces(
    logits: List[Float64], start: Int
) -> Tuple[Float64, Float64, Float64]:
    """The pieces from the definition, in float64.

    The index is the lowest among the tied maxima, plus `start`.
    """
    var d = len(logits)
    var l_max = neg_inf[DType.float64]()
    for i in range(d):
        if logits[i] > l_max:
            l_max = logits[i]

    var excluded = 0.0
    var ties = 0
    var top_idx = d
    for i in range(d):
        if logits[i] == l_max:
            ties += 1
            if i < top_idx:
                top_idx = i
        elif logits[i] > neg_inf[DType.float64]():
            excluded += exp(logits[i] - l_max)

    var mass = excluded + Float64(ties - 1)
    return (l_max, Float64(top_idx + start), mass)


def confident_filler(row: Int, col: Int, d: Int) -> Float64:
    """Row 0's top sits 40 logits above the rest, so the excluded mass
    is near 1e-11: the `sum_all - 1` form cancels to zero in fp32."""
    if row == 0:
        return 40.0 if col == 0 else scrambled_logit(row, col, d)
    return scrambled_logit(row, col, d)


def two_tied_filler(row: Int, col: Int, d: Int) -> Float64:
    """Two columns share each row's maximum; the lower index is the top."""
    var v = scrambled_logit(row, col, d)
    if col == (row * 7919) % d or col == (row * 104729) % d:
        return 10.0
    return v


def three_tied_filler(row: Int, col: Int, d: Int) -> Float64:
    """Three columns share each row's maximum."""
    var v = scrambled_logit(row, col, d)
    if col % 3 == 0 and col < 9:
        return 10.0
    return v


def topk_masked_filler(row: Int, col: Int, d: Int) -> Float64:
    """All but 20 entries masked to `-inf` (a top-k mask)."""
    if col < 20:
        return scrambled_logit(row, col, d)
    return neg_inf[DType.float64]()


def only_top_finite_filler(row: Int, col: Int, d: Int) -> Float64:
    """Only the top is finite: zero excluded mass, a `+inf` score."""
    if col == 7:
        return 5.0
    return neg_inf[DType.float64]()


def min_finite_filler(row: Int, col: Int, d: Int) -> Float64:
    """Ties the first two logits at the lowest finite accumulator value."""
    if col >= 2:
        return neg_inf[DType.float64]()
    return Float64(min_finite[.float32]())


def all_masked_filler(row: Int, col: Int, d: Int) -> Float64:
    """No finite logit in any row; the score is `+inf`, not NaN."""
    return neg_inf[DType.float64]()


def tail_winner_filler(row: Int, col: Int, d: Int) -> Float64:
    """Places the winners in separate scalar tail lanes."""
    if row == 0 and col == d - 2:
        return 16.0
    if row == 1 and col == d - 1:
        return 16.0
    return scrambled_logit(row, col, d)


def masked_ties_filler(row: Int, col: Int, d: Int) -> Float64:
    """Combines two- and three-way ties with masks and a singleton."""
    if row == 2:
        return only_top_finite_filler(row, col, d)
    if col >= 20:
        return neg_inf[DType.float64]()
    if row == 0 and (col == 5 or col == 9):
        return 6.0
    if row == 1 and (col == 3 or col == 11 or col == 19):
        return 6.0
    return scrambled_logit(row, col, d)


def top_token_state_checks() -> SIMD[.int32, 4]:
    """Evaluates the same join and padding invariants on CPU and GPU."""
    comptime minimum = min_finite[.float32]()
    comptime empty = neg_inf[.float32]()
    var scores = TopTokenLogit[.float32, 1]()
    var singleton = TopTokenLogit[.float32, 1]()
    singleton.accumulate[.float32, 1](SIMD[.float32, 1](minimum))
    scores.join(singleton)

    var pieces = TopTokenPieces[.float32, 1]()
    var right = TopTokenPieces[.float32, 1]()
    right.accumulate[.float32, 1](
        SIMD[.float32, 1](minimum), SIMD[.int64, 1](Int64(5))
    )
    pieces.join(right)
    var singleton_kept = (
        scores.t[0] == 1.0 and pieces.t[0] == 1.0 and pieces.i[0] == 5
    )

    var lower_index = TopTokenPieces[.float32, 1]()
    lower_index.accumulate[.float32, 1](
        SIMD[.float32, 1](minimum), SIMD[.int64, 1](Int64(2))
    )
    scores.join(singleton)
    pieces.join(lower_index)
    var tie_kept = (
        scores.t[0] == 2.0 and pieces.t[0] == 2.0 and pieces.i[0] == 2
    )

    var padded = TopTokenLogit[.float32, 2]()
    padded.accumulate[.float32, 1](SIMD[.float32, 1](minimum))
    var padded_pieces = TopTokenPieces[.float32, 2]()
    padded_pieces.accumulate[.float32, 1](
        SIMD[.float32, 1](minimum), SIMD[.int64, 1](Int64(2))
    )
    var padding_neutral = (
        padded.t[0] == 1.0
        and padded.t[1] == 0.0
        and padded.b[1] == 0.0
        and padded.m[1] == empty
        and padded_pieces.t[0] == 1.0
        and padded_pieces.i[0] == 2
        and padded_pieces.t[1] == 0.0
        and padded_pieces.b[1] == 0.0
        and padded_pieces.m[1] == empty
        and padded_pieces.i[1] == Int32.MAX
    )

    var both_empty = TopTokenLogit[.float32, 1]()
    both_empty.join(TopTokenLogit[.float32, 1]())
    var join_neutral = (
        both_empty.m[0] == empty
        and both_empty.t[0] == 0.0
        and both_empty.b[0] == 0.0
    )
    return SIMD[.int32, 4](
        Int32(1 if singleton_kept else 0),
        Int32(1 if tie_kept else 0),
        Int32(1 if padding_neutral else 0),
        Int32(1 if join_neutral else 0),
    )


def assert_top_token_states(checks: SIMD[.int32, 4]) raises:
    """Reports which invariant failed after the fixture executes."""
    assert_equal(checks[0], 1, "Empty-left join lost the finite singleton")
    assert_equal(checks[1], 1, "Tie join lost the lowest index or count")
    assert_equal(checks[2], 1, "Padding contributed to a reduction")
    assert_equal(checks[3], 1, "Joining two empty states changed the identity")
