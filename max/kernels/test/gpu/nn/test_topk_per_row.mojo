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
"""What `topk_per_row_gpu` adds to the block-wide top-k: a pick count per row,
int64 columns next to the values, and a `(-inf, -1)` tail.

The selection itself is covered by `test_topk_bitonic`. Equal scores may come
out in any order, so each row's values must match a CPU sort position by
position, and each column must be in range, unique within the row, and score
exactly the value in its slot. Slots past `k[r]` hold `(-inf, -1)`.

`masked` rows are `-inf` wherever a candidate has not closed, up to and
including a row with no live candidate, as the indexer's are. `tied` rows draw
from a handful of values so the cutoff falls inside a run of equal scores.
`edges` rows keep a live prefix of 0, 1, 511, 512 or 513 columns, around
`k = 512`. `masked` and `edges` rows under `all` picks ask for more slots than
the row has finite entries, so the picked range itself carries `-inf` slots
whose column may be a sort padding lane (negative).

Rows of 1030 and 4097 columns put odd rows' bases off a 16 byte boundary.
Past 2048 columns, the shapes reach each select the ordered top-k picks:
register-resident (5000), streaming with fewer rows than SMs (12000), and
streaming with more (300 rows).
"""

from std.builtin.sort import sort
from std.testing import assert_equal, assert_true
from std.utils.numerics import min_or_neg_inf

from max.gpu.host import DeviceContext
from nn.topk_per_row import topk_per_row_gpu


def _next(mut state: UInt64) -> UInt64:
    state = state * 6364136223846793005 + 1442695040888963407
    return state >> 33


def check[
    dtype: DType
](
    ctx: DeviceContext,
    mut state: UInt64,
    rows: Int,
    n: Int,
    max_k: Int,
    fill: String,
    picks_mode: String,
) raises:
    var label = String(
        dtype, " ", rows, "x", n, " k=", max_k, " ", fill, " ", picks_mode
    )
    var neg_inf = min_or_neg_inf[dtype]()
    var scores = List[Scalar[dtype]](capacity=rows * n)
    var edge_lives = [0, 1, 511, 512, 513]
    for r in range(rows):
        # Masked share grows with the row, so the batch spans fully live rows
        # through one with nothing live.
        var masked_pct = (r * 100) // max(rows - 1, 1)
        for c in range(n):
            var v = Scalar[dtype](Float32(Int(_next(state) % 1048576)) / 1e6)
            if fill == "tied":
                v = Scalar[dtype](Float32(Int(_next(state) % 5)))
            if fill == "masked" and Int(_next(state) % 100) < masked_pct:
                v = neg_inf
            if fill == "edges" and c >= edge_lives[r % len(edge_lives)]:
                v = neg_inf
            scores.append(v)

    var picks = List[Int64](capacity=rows)
    for r in range(rows):
        var finite = 0
        for c in range(n):
            finite += Int(scores[r * n + c] > neg_inf)
        if picks_mode == "ragged":
            picks.append(Int64((r * 37) % (max_k + 1)))
        elif picks_mode == "live":
            picks.append(Int64(min(finite, max_k)))
        else:
            picks.append(Int64(max_k))

    var expected = List[Float32](length=rows * max_k, fill=Float32(neg_inf))
    for r in range(rows):
        var row = List[Float32](capacity=n)
        for c in range(n):
            row.append(Float32(scores[r * n + c]))
        sort(row)
        for j in range(min(Int(picks[r]), n)):
            expected[r * max_k + j] = row[n - 1 - j]

    var scores_dev = ctx.enqueue_create_buffer[dtype](rows * n)
    var k_dev = ctx.enqueue_create_buffer[.int64](rows)
    var vals_dev = ctx.enqueue_create_buffer[dtype](rows * max_k)
    var idxs_dev = ctx.enqueue_create_buffer[.int64](rows * max_k)
    # Sentinels no slot can legitimately hold, so an unwritten one fails.
    vals_dev.enqueue_fill(Scalar[dtype](7.0))
    idxs_dev.enqueue_fill(Int64(-7))
    with scores_dev.map_to_host() as h:
        for i in range(rows * n):
            h[i] = scores[i]
    with k_dev.map_to_host() as h:
        for r in range(rows):
            h[r] = picks[r]

    topk_per_row_gpu(
        ctx,
        scores_dev.unsafe_ptr().as_unsafe_any_origin().as_imm(),
        k_dev.unsafe_ptr().as_unsafe_any_origin().as_imm(),
        vals_dev.unsafe_ptr().as_unsafe_any_origin(),
        idxs_dev.unsafe_ptr().as_unsafe_any_origin(),
        rows,
        n,
        max_k,
    )
    ctx.synchronize()

    with vals_dev.map_to_host() as gv:
        with idxs_dev.map_to_host() as gi:
            for r in range(rows):
                var kept = min(Int(picks[r]), n)
                var seen = List[Bool](length=n, fill=False)
                for j in range(max_k):
                    var at = r * max_k + j
                    var slot = String("row ", r, " slot ", j, " ", label)
                    assert_equal(
                        Float32(gv[at]),
                        expected[at],
                        String("value mismatch at ", slot),
                    )
                    var col = Int(gi[at])
                    if j >= kept:
                        assert_equal(col, -1, String("tail at ", slot))
                    elif col < 0:
                        # A padding column of the single-block sort can land
                        # on a slot that scores -inf anyway.
                        assert_true(
                            gv[at] == neg_inf,
                            String("missing column at ", slot),
                        )
                    else:
                        assert_true(col < n, String("bad column ", slot))
                        assert_true(not seen[col], String("repeat ", slot))
                        seen[col] = True
                        assert_equal(
                            scores[r * n + col],
                            gv[at],
                            String("column scores another value at ", slot),
                        )


def main() raises:
    var state: UInt64 = 0x9E3779B97F4A7C15
    with DeviceContext() as ctx:
        check[.float32](ctx, state, 32, 1024, 512, "masked", "live")
        check[.float32](ctx, state, 88, 2048, 512, "masked", "live")
        check[.float32](ctx, state, 16, 1024, 512, "random", "ragged")
        check[.float32](ctx, state, 8, 2048, 512, "tied", "all")
        check[.float32](ctx, state, 4, 100, 37, "random", "all")
        check[.float32](ctx, state, 4, 5000, 512, "masked", "live")
        check[.bfloat16](ctx, state, 16, 2048, 512, "masked", "live")
        check[.float32](ctx, state, 10, 2048, 512, "edges", "live")
        check[.float32](ctx, state, 10, 5000, 512, "edges", "live")
        check[.float32](ctx, state, 7, 1030, 512, "random", "all")
        check[.float32](ctx, state, 32, 1024, 512, "masked", "all")
        check[.float32](ctx, state, 10, 2048, 512, "edges", "all")
        check[.float32](ctx, state, 7, 1030, 512, "edges", "all")
        check[.float32](ctx, state, 5, 4097, 512, "random", "all")
        check[.float32](ctx, state, 4, 12000, 512, "masked", "live")
        check[.float32](ctx, state, 300, 5000, 512, "masked", "live")
        check[.float32](ctx, state, 4, 20000, 512, "masked", "live")
        check[.bfloat16](ctx, state, 16, 2048, 512, "tied", "all")
        check[.float32](ctx, state, 4, 5000, 512, "tied", "all")
        check[.float32](ctx, state, 4, 12000, 512, "tied", "all")
        check[.float32](ctx, state, 4, 20000, 512, "tied", "all")
        check[.float32](ctx, state, 300, 5000, 512, "tied", "all")
        print("PASS test_topk_per_row")
