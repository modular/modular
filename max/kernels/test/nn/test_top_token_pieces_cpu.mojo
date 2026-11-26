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
"""Runs the top-token pieces body on CPU.

The GPU test (`test/gpu/nn/test_top_token_pieces.mojo`) covers the same
cases at vocabulary scale; this one proves the CPU tier path
(cooperative and wide-accumulator) with the same float64 reference.
"""

from std.sys import align_of

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from std.testing import assert_almost_equal, assert_equal

from builtin_kernels.rowwise_reductions import (
    TopTokenPiecesColumns,
    top_token_columns,
)
from top_token_test_utils.references import (
    confident_filler,
    masked_ties_filler,
    min_finite_filler,
    only_top_finite_filler,
    reference_pieces,
    scrambled_logit,
    tail_winner_filler,
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
    # A CPU rowwise launch runs on this thread, outside ctx's queue, so
    # the copy must complete before the kernel reads the buffer.
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
    # A CPU rowwise launch runs outside the context's queue, so the
    # copies must complete before the host reads (see
    # rowwise_reduce_test_utils/public_reduce.mojo:41).
    ctx.synchronize()

    for row in range(rows):
        var logits = List[Float64]()
        for col in range(d):
            logits.append(Float64(logits_h[row * d + col]))
        var l_max, top_idx, mass = reference_pieces(logits, start)
        var got_max = Float64(pieces_h[row * 3])
        var got_idx = Float64(pieces_h[row * 3 + 1])
        var got_mass = Float64(pieces_h[row * 3 + 2])
        assert_almost_equal(got_max, l_max, rtol=rtol, atol=atol)
        assert_equal(got_idx, top_idx)
        assert_almost_equal(got_mass, mass, rtol=rtol, atol=atol)

    _ = logits_dev^
    _ = pieces_dev^


def main() raises:
    with DeviceContext.cpu() as ctx:
        run_case[.float32, 11](ctx, 3, 4099)
        run_case[.float32, 101](ctx, 2, 4099, filler=tail_winner_filler)
        run_case[.bfloat16, 101](ctx, 2, 4099, filler=tail_winner_filler)
        run_case[.float32, 101](ctx, 3, 64, filler=masked_ties_filler)
        run_case[.float32, 0](
            ctx, 1, 2048, filler=confident_filler, rtol=1e-4, atol=0
        )
        run_case[.float32, 40960](ctx, 2, 65536)
        run_case[.float32, 7](ctx, 2, 4099, filler=two_tied_filler)
        run_case[.float32, 5](ctx, 2, 4099, filler=only_top_finite_filler)
        run_case[.float32, 3](ctx, 1, 2, filler=min_finite_filler)
        run_case[.bfloat16, 13](ctx, 2, 4099)
        run_case[.float32, 0](ctx, 0, 4099)
        run_case[.float32, 2048](ctx, 1, 4099)
