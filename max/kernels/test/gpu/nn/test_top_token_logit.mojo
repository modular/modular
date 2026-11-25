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
"""Correctness tests for the head-free draft score kernel.

The kernel scores each row of a `[rows, vocab]` logits tensor with
`-log(excluded_mass)`, `excluded_mass = sum_{j != top} exp(l_j - l_max)
+ (ties - 1)`, in one pass over the row. The reference here computes the
same value in float64 from the definition: the excluded mass is summed
directly at the max's scale, never as `sum_all - 1`, which cancels in
fp32 on a confident row.
"""

from std.sys import align_of
from std.testing import assert_almost_equal, assert_true
from std.utils.numerics import inf

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major

from builtin_kernels.rowwise_reductions import (
    TopTokenLogitColumns,
    top_token_columns,
)
from top_token_test_utils.references import (
    all_masked_filler,
    assert_top_token_states,
    confident_filler,
    min_finite_filler,
    only_top_finite_filler,
    reference_score,
    scrambled_logit,
    three_tied_filler,
    top_token_state_checks,
    topk_masked_filler,
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


def _write_state_checks(output: MutPointer[Int32, MutAnyOrigin]):
    var checks = top_token_state_checks()
    comptime for i in range(4):
        output[i] = checks[i]


def _check_states_on_device(ctx: DeviceContext) raises:
    var device_checks = ctx.enqueue_create_buffer[.int32](4)
    var host_checks = ctx.enqueue_create_host_buffer[.int32](4)
    ctx.enqueue_function[_write_state_checks](
        device_checks, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(host_checks, device_checks)
    ctx.synchronize()
    var checks = SIMD[.int32, 4](
        host_checks[0], host_checks[1], host_checks[2], host_checks[3]
    )
    assert_top_token_states(checks)


def main() raises:
    with DeviceContext() as ctx:
        # Vocabulary-scale rows: the launch shapes the score runs at.
        run_case[.float32](ctx, 2, 129280)
        run_case[.float32](ctx, 2, 154880)
        # A size that divides no SIMD width or tile.
        run_case[.float32](ctx, 3, 4099)
        # Rows past the SM count with a SIMD-divisible vocab: the block
        # tier dispatches at W = 4 lanes per thread, the multi-lane
        # partial-join path.
        run_case[.float32](ctx, 148, 4096)
        # Tied maxima: two and three.
        run_case[.float32](ctx, 2, 4099, filler=two_tied_filler)
        run_case[.float32](ctx, 2, 4099, filler=three_tied_filler)
        # Top-k masked rows and a row with only the top finite.
        run_case[.float32](ctx, 2, 4099, filler=topk_masked_filler)
        run_case[.float32](ctx, 2, 4099, filler=only_top_finite_filler)
        run_case[.float32](ctx, 1, 2, filler=min_finite_filler)
        # All rows fully masked: `+inf`, not NaN.
        run_case[.float32](ctx, 2, 4099, filler=all_masked_filler)
        # A confident row: the excluded mass is near 1e-11, so the
        # score must hold 1e-4 relative against the float64 reference.
        run_case[.float32](ctx, 1, 129280, filler=confident_filler, rtol=1e-4)
        # bfloat16 input, fp32 accumulation.
        run_case[.bfloat16](ctx, 2, 4099)
        run_case[.bfloat16](ctx, 2, 129280)
        # Zero rows (no launch, no crash) and one row.
        run_case[.float32](ctx, 0, 4099)
        run_case[.float32](ctx, 1, 4099)
        _check_states_on_device(ctx)
