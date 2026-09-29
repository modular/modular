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

"""The single-warp top-k path must select exactly what the two-stage path does.

`topk_gpu` routes short rows with a small k to `_topk_warp`, one launch and one
warp instead of `_topk_stage1` + `_topk_stage2`. That path is on every MoE
router, so a divergence is not a tolerance question: picking a different expert
on a tie is a different model.

Each case runs the same input twice -- once letting the dispatch choose (warp
path) and once with an explicit `num_blocks_per_input`, which the dispatch
honors by falling back to the two-stage path -- and requires bit-identical
values *and* indices. `block_size=256` with `num_blocks_per_input =
ceildiv(N, 256)` reproduces exactly what the default would have picked below
the warp path's `N <= 2048` bound, so the fallback arm is the real previous
behavior rather than an unusual configuration of it.

Ties are the interesting input. `fill_tied` draws from a handful of distinct
values so most of the row is exact duplicates, which is what makes the tie
order -- descending value, smallest index first -- observable at all. Iota and
random rows have essentially no ties and would pass under a broken tie rule.
"""

from std.math import ceildiv

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.topk import topk_gpu
from std.random import rand, seed
from std.testing import assert_equal
from std.utils.numerics import min_or_neg_inf

comptime IDX = DType.int64

# Kimi K3's router: top-16 of 896 experts, one token per decode step.
comptime K3_N = 896
comptime K3_K = 16


def fill_row_major[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    batch_size: Int,
    n: Int,
    fill: String,
    num_distinct: Int,
):
    """Writes the row pattern named by `fill` into a `[batch_size, n]` buffer.

    "tied" draws from `num_distinct` values so nearly every element is an exact
    duplicate; that is the pattern that makes the tie order observable.
    """
    if fill == "random":
        rand(ptr, batch_size * n)
        return
    for b in range(batch_size):
        for i in range(n):
            var v: Int
            if fill == "iota":
                v = (b * n + i) % 1024
            else:
                v = (i * 7 + b * 3) % num_distinct
            ptr[b * n + i] = Scalar[dtype](Float64(v))


def check_matches_two_stage[
    dtype: DType, largest: Bool = True
](
    ctx: DeviceContext,
    batch_size: Int,
    n: Int,
    k: Int,
    fill: String,
    num_distinct: Int = 4,
) raises:
    """Runs one shape through both paths and requires identical output."""
    var in_buf = ctx.enqueue_create_buffer[dtype](batch_size * n)
    var warp_vals = ctx.enqueue_create_buffer[dtype](batch_size * k)
    var warp_idxs = ctx.enqueue_create_buffer[IDX](batch_size * k)
    var ref_vals = ctx.enqueue_create_buffer[dtype](batch_size * k)
    var ref_idxs = ctx.enqueue_create_buffer[IDX](batch_size * k)

    with in_buf.map_to_host() as h:
        fill_row_major[dtype](h.unsafe_ptr(), batch_size, n, fill, num_distinct)

    var in_t = TileTensor(in_buf, row_major(Coord(batch_size, n)))

    topk_gpu[sampling=False, largest=largest](
        ctx,
        k,
        in_t.as_unsafe_any_origin().as_imm(),
        TileTensor(warp_vals, row_major(Coord(batch_size, k))),
        TileTensor(warp_idxs, row_major(Coord(batch_size, k))),
    )

    topk_gpu[sampling=False, largest=largest](
        ctx,
        k,
        in_t.as_unsafe_any_origin().as_imm(),
        TileTensor(ref_vals, row_major(Coord(batch_size, k))),
        TileTensor(ref_idxs, row_major(Coord(batch_size, k))),
        block_size=256,
        num_blocks_per_input=ceildiv(n, 256),
    )
    ctx.synchronize()

    var label = String(
        "batch=",
        batch_size,
        " N=",
        n,
        " k=",
        k,
        " fill=",
        fill,
        " largest=",
        largest,
    )
    with warp_vals.map_to_host() as wv:
        with ref_vals.map_to_host() as rv:
            for i in range(batch_size * k):
                assert_equal(
                    wv[i], rv[i], String("value mismatch at ", i, " ", label)
                )
    with warp_idxs.map_to_host() as wi:
        with ref_idxs.map_to_host() as ri:
            for i in range(batch_size * k):
                assert_equal(
                    wi[i], ri[i], String("index mismatch at ", i, " ", label)
                )

    _ = in_buf^
    _ = warp_vals^
    _ = warp_idxs^
    _ = ref_vals^
    _ = ref_idxs^


def check_per_row_k(ctx: DeviceContext) raises:
    """Per-row k, including the k=0 and k=-1 (means max_k) encodings.

    `_topk_warp` reads the same per-row `K` buffer the two-stage kernels do,
    and must fill positions at or past that row's k with the dead value and
    index -1, exactly as `_topk_stage2` does.
    """
    comptime dtype = DType.float32
    comptime n = 512
    comptime max_k = 8
    comptime batch_size = 4
    comptime dead = min_or_neg_inf[dtype]()

    var in_buf = ctx.enqueue_create_buffer[dtype](batch_size * n)
    var k_buf = ctx.enqueue_create_buffer[.int64](batch_size)
    var vals = ctx.enqueue_create_buffer[dtype](batch_size * max_k)
    var idxs = ctx.enqueue_create_buffer[IDX](batch_size * max_k)

    with in_buf.map_to_host() as h:
        for b in range(batch_size):
            for i in range(n):
                # Descending, so the top-k of row b is indices 0..k-1.
                h[b * n + i] = Scalar[dtype](Float64(n - i))
    with k_buf.map_to_host() as h:
        h[0] = Int64(0)
        h[1] = Int64(3)
        h[2] = Int64(-1)
        h[3] = Int64(max_k)

    var in_t = TileTensor(in_buf, row_major(Coord(batch_size, n)))
    topk_gpu[sampling=False, largest=True](
        ctx,
        max_k,
        in_t.as_unsafe_any_origin().as_imm(),
        TileTensor(vals, row_major(Coord(batch_size, max_k))),
        TileTensor(idxs, row_major(Coord(batch_size, max_k))),
        k=TileTensor(k_buf, row_major(Coord(batch_size)))
        .as_unsafe_any_origin()
        .as_imm(),
    )
    ctx.synchronize()

    var expect_k = [0, 3, max_k, max_k]
    with vals.map_to_host() as v:
        with idxs.map_to_host() as ix:
            for b in range(batch_size):
                for j in range(max_k):
                    var at = b * max_k + j
                    if j < expect_k[b]:
                        assert_equal(v[at], Scalar[dtype](Float64(n - j)))
                        assert_equal(Int(ix[at]), j)
                    else:
                        assert_equal(v[at], dead)
                        assert_equal(Int(ix[at]), -1)

    _ = in_buf^
    _ = k_buf^
    _ = vals^
    _ = idxs^


def check_bound_edges(ctx: DeviceContext) raises:
    """The rows either side of the `N <= 2048` dispatch bound must agree.

    2048 is the last length whose two-stage block partition is still
    contiguous, which is what makes the two tie orders the same; 2049 falls
    back. Both are checked against a tie-heavy row so the comparison has
    something to disagree about.
    """
    check_matches_two_stage[.float32](ctx, 2, 2047, 16, "tied", num_distinct=6)
    check_matches_two_stage[.float32](ctx, 2, 2048, 16, "tied", num_distinct=6)


def check_neg_inf_rows(ctx: DeviceContext) raises:
    """A -inf element is invisible to both paths, and must stay that way.

    `TopK_2.insert` folds with a strict `>` against a -inf seed, so a -inf
    element is never selected -- the same rule in both paths. The row here has
    exactly k finite values so the selection is still well defined.
    """
    comptime dtype = DType.float32
    comptime n = 256
    comptime k = 4
    comptime batch_size = 2
    comptime dead = min_or_neg_inf[dtype]()

    var in_buf = ctx.enqueue_create_buffer[dtype](batch_size * n)
    var vals = ctx.enqueue_create_buffer[dtype](batch_size * k)
    var idxs = ctx.enqueue_create_buffer[IDX](batch_size * k)

    with in_buf.map_to_host() as h:
        for b in range(batch_size):
            for i in range(n):
                h[b * n + i] = dead
            for j in range(k):
                h[b * n + 3 + j * 17] = Scalar[dtype](Float64(k - j))

    var in_t = TileTensor(in_buf, row_major(Coord(batch_size, n)))
    topk_gpu[sampling=False, largest=True](
        ctx,
        k,
        in_t.as_unsafe_any_origin().as_imm(),
        TileTensor(vals, row_major(Coord(batch_size, k))),
        TileTensor(idxs, row_major(Coord(batch_size, k))),
    )
    ctx.synchronize()

    with idxs.map_to_host() as ix:
        for b in range(batch_size):
            for j in range(k):
                assert_equal(Int(ix[b * k + j]), 3 + j * 17)

    _ = in_buf^
    _ = vals^
    _ = idxs^


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        # The shape this path exists for.
        check_matches_two_stage[.float32](ctx, 1, K3_N, K3_K, "random")
        check_matches_two_stage[.float32](ctx, 1, K3_N, K3_K, "iota")
        check_matches_two_stage[.float32](
            ctx, 1, K3_N, K3_K, "tied", num_distinct=8
        )
        # Prefill-shaped batches of the same row length.
        check_matches_two_stage[.float32](ctx, 64, K3_N, K3_K, "random")
        check_matches_two_stage[.float32](
            ctx, 64, K3_N, K3_K, "tied", num_distinct=32
        )

        # Rows shorter than a warp, and lengths that are not a multiple of it.
        check_matches_two_stage[.float32](ctx, 3, 7, 3, "tied", num_distinct=3)
        check_matches_two_stage[.float32](ctx, 3, 100, 5, "random")
        check_matches_two_stage[.float32](ctx, 3, 100, 5, "tied")
        # k == N, the degenerate "select everything" request.
        check_matches_two_stage[.float32](ctx, 2, 33, 33, "tied")

        # Other expert counts in use across MoE routers.
        check_matches_two_stage[.float32](ctx, 8, 128, 8, "random")
        check_matches_two_stage[.float32](ctx, 8, 256, 8, "tied")
        check_matches_two_stage[.float32](ctx, 4, 1024, 2, "random")

        check_bound_edges(ctx)

        # bottom-k shares the kernel through the `largest` parameter.
        check_matches_two_stage[.float32, largest=False](
            ctx, 2, K3_N, K3_K, "random"
        )
        check_matches_two_stage[.float32, largest=False](ctx, 2, 256, 8, "tied")

        check_matches_two_stage[.bfloat16](ctx, 4, 512, 8, "random")
        check_matches_two_stage[.bfloat16](ctx, 4, 512, 8, "tied")

        check_per_row_k(ctx)
        check_neg_inf_rows(ctx)
