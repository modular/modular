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
"""Runs the head-free draft score body on CPU.

The GPU test (`test/gpu/nn/test_top_token_logit.mojo`) covers the same
cases at vocabulary scale; this one proves the CPU tier path (cooperative
and wide-accumulator) with the same float64 reference.
"""

from std.sys import align_of, simd_width_of
from std.utils.numerics import inf

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from std.testing import assert_almost_equal, assert_true

from builtin_kernels.rowwise_reductions import (
    TopTokenLogitColumns,
    top_token_columns,
)
from top_token_test_utils.references import (
    all_masked_filler,
    assert_top_token_states,
    confident_filler,
    masked_ties_filler,
    min_finite_filler,
    only_top_finite_filler,
    reference_score,
    scrambled_logit,
    tail_winner_filler,
    top_token_state_checks,
    two_tied_filler,
)


def run_case[
    dtype: DType
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
    # A CPU rowwise launch runs on this thread, outside ctx's queue, so
    # the copy must complete before the kernel reads the buffer.
    ctx.synchronize()
    var scores_dev = ctx.enqueue_create_buffer[.float32](rows)

    var logits_buf = TileTensor(logits_dev, row_major(Coord(rows, d)))
    var scores_buf = TileTensor(scores_dev, row_major(Coord(rows, 1)))

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
    ](coords: Coord, val: SIMD[.float32, width]) {var scores_buf}:
        scores_buf.store[width=width, alignment=align_of[DType.float32]()](
            coords, val
        )

    top_token_columns[dtype, Columns=TopTokenLogitColumns, reduce_dim=1](
        input_fn, output_fn, Coord(rows, d), ctx
    )

    var scores_h = ctx.enqueue_create_host_buffer[.float32](rows)
    ctx.enqueue_copy(scores_h, scores_dev)
    ctx.synchronize()

    for row in range(rows):
        var logits = List[Float64]()
        for col in range(d):
            logits.append(Float64(logits_h[row * d + col]))
        var expected = reference_score(logits)
        var got = Float64(scores_h[row])
        if expected == inf[DType.float64]():
            assert_true(
                got == inf[DType.float64](),
                msg=String(t"row={row} d={d}: expected +inf score, got {got}"),
            )
        else:
            assert_almost_equal(
                got,
                expected,
                rtol=rtol,
                atol=atol,
                msg=String(
                    t"row={row} d={d}: score mismatch, got {got}, want"
                    t" {expected}"
                ),
            )

    _ = logits_dev^
    _ = scores_dev^


def main() raises:
    with DeviceContext.cpu() as ctx:
        # Odd size: no SIMD width or split divides it.
        run_case[.float32](ctx, 3, 4099)
        run_case[.float32](ctx, 2, 4099, filler=tail_winner_filler)
        run_case[.bfloat16](ctx, 2, 4099, filler=tail_winner_filler)
        run_case[.float32](ctx, 3, 64, filler=masked_ties_filler)
        # A row long enough that the CPU wide-accumulator path fires.
        run_case[.float32](ctx, 2, 65536)
        # Tied maxima.
        run_case[.float32](ctx, 2, 4099, filler=two_tied_filler)
        # Only the top finite: `+inf`.
        run_case[.float32](ctx, 2, 4099, filler=only_top_finite_filler)
        run_case[.float32](ctx, 1, 2, filler=min_finite_filler)
        run_case[.float32](
            ctx, 1, simd_width_of[DType.float32](), filler=min_finite_filler
        )
        # All rows fully masked: `+inf`, not NaN.
        run_case[.float32](ctx, 2, 4099, filler=all_masked_filler)
        # A confident row: the score must hold 1e-4 relative.
        run_case[.float32](ctx, 1, 65536, filler=confident_filler, rtol=1e-4)
        # bfloat16 input, fp32 accumulation.
        run_case[.bfloat16](ctx, 2, 4099)
        # Zero rows (no launch, no crash) and one row.
        run_case[.float32](ctx, 0, 4099)
        run_case[.float32](ctx, 1, 4099)
    assert_top_token_states(top_token_state_checks())
