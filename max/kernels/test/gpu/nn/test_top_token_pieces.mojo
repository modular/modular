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
"""Correctness tests for the top-token pieces kernel.

The kernel reduces each row of a `[rows, vocab]` logits tensor to the
three pieces a sharded greedy drafter allgathers — the row max, the
lowest index of the max plus `start`, and the excluded mass
`sum_{j != top} exp(l_j - l_max) + (ties - 1)` — in one pass over the
row. The reference here computes the same values in float64 from the
definition; the excluded mass is summed directly at the max's scale,
never as `sum_all - 1`, which cancels in fp32 on a confident row.
"""

from std.sys import align_of

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from std.testing import assert_almost_equal, assert_equal, assert_true

from builtin_kernels.rowwise_reductions import (
    TopTokenPiecesColumns,
    top_token_columns,
)
from top_token_test_utils.references import (
    min_finite_filler,
    only_top_finite_filler,
    reference_pieces,
    scrambled_logit,
    three_tied_filler,
    topk_masked_filler,
    two_tied_filler,
)


def run_case[
    dtype: DType,
    start: Int,
](
    ctx: DeviceContext,
    rows: Int,
    d: Int,
    *,
    filler: def(Int, Int, Int) thin -> Float64 = scrambled_logit,
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-4,
) raises:
    var logits_h = ctx.enqueue_create_host_buffer[dtype](rows * d)
    for row in range(rows):
        for col in range(d):
            logits_h[row * d + col] = filler(row, col, d).cast[dtype]()

    var logits_dev = ctx.enqueue_create_buffer[dtype](rows * d)
    ctx.enqueue_copy(logits_dev, logits_h)
    # The launch may run outside the context's queue; the copy must
    # complete before the kernel reads the buffer.
    ctx.synchronize()
    var pieces_dev = ctx.enqueue_create_buffer[.float32](rows * 3)

    var logits_buf = TileTensor(logits_dev, row_major(Coord(rows, d)))
    var pieces_buf = TileTensor(pieces_dev, row_major(Coord(rows, 3)))

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var logits_buf} -> SIMD[dtype, width]:
        return logits_buf.load[
            width=width, alignment=alignment * align_of[dtype]()
        ](coords)

    @inline(.always)
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[.float32, width]) {var pieces_buf}:
        pieces_buf.store[width=width, alignment=align_of[DType.float32]()](
            coords, val
        )

    top_token_columns[
        dtype, Columns=TopTokenPiecesColumns[start], reduce_dim=1
    ](input_fn, output_fn, Coord(rows, d), ctx)

    var pieces_h = ctx.enqueue_create_host_buffer[.float32](rows * 3)
    ctx.enqueue_copy(pieces_h, pieces_dev)
    ctx.synchronize()

    for row in range(rows):
        var logits = List[Float64]()
        for col in range(d):
            logits.append(Float64(logits_h[row * d + col]))
        var l_max, top_idx, mass = reference_pieces(logits, start)
        var got_max = Float64(pieces_h[row * 3])
        var got_idx = Float64(pieces_h[row * 3 + 1])
        var got_mass = Float64(pieces_h[row * 3 + 2])
        assert_almost_equal(
            got_max,
            l_max,
            rtol=rtol,
            atol=atol,
            msg=String(
                t"row={row} d={d}: max mismatch, got {got_max}, want {l_max}"
            ),
        )
        # The index is exact: the lowest tied max, plus the start.
        assert_equal(
            got_idx,
            top_idx,
            msg=String(
                t"row={row} d={d}: index mismatch, got {got_idx}, want"
                t" {top_idx}"
            ),
        )
        assert_almost_equal(
            got_mass,
            mass,
            rtol=rtol,
            atol=atol,
            msg=String(
                t"row={row} d={d}: excluded mass mismatch, got"
                t" {got_mass}, want {mass}"
            ),
        )

    _ = logits_dev^
    _ = pieces_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Vocabulary-scale rows at a nonzero shard start.
        run_case[.float32, 40960](ctx, 2, 129280)
        run_case[.float32, 0](ctx, 2, 154880)
        # A size that divides no SIMD width or tile.
        run_case[.float32, 101](ctx, 3, 4099)
        # Rows past the SM count with a SIMD-divisible vocab: the block
        # tier dispatches at W = 4 lanes per thread, the multi-lane
        # partial-join path.
        run_case[.float32, 512](ctx, 148, 4096)
        # Tied maxima: two and three; the lowest index must win.
        run_case[.float32, 7](ctx, 2, 4099, filler=two_tied_filler)
        run_case[.float32, 13](ctx, 2, 4099, filler=three_tied_filler)
        # Top-k masked rows and a row with only the top finite.
        run_case[.float32, 3](ctx, 2, 4099, filler=topk_masked_filler)
        run_case[.float32, 5](ctx, 2, 4099, filler=only_top_finite_filler)
        run_case[.float32, 3](ctx, 1, 2, filler=min_finite_filler)
        # bfloat16 input, fp32 accumulation.
        run_case[.bfloat16, 17](ctx, 2, 4099)
        run_case[.bfloat16, 65536](ctx, 2, 129280)
        # Zero rows (no launch, no crash) and one row.
        run_case[.float32, 0](ctx, 0, 4099)
        run_case[.float32, 2048](ctx, 1, 4099)
