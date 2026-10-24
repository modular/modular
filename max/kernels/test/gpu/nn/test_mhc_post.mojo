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

from std.math import abs, max
from std.random import randn, random_float64, seed

from layout import Coord, TileTensor, row_major
from layout.tile_layout import Layout
from max.gpu.host import DeviceAttribute, DeviceContext, default_accelerator
from nn.mhc import mhc_post, mhc_post_block_size, mhc_post_non_temporal
from std.testing import assert_raises, assert_true
from std.utils.numerics import FPUtils


def _ulp[dtype: DType](x: Float64) -> Float64:
    """Returns the `dtype` unit in the last place at magnitude `|x|`."""
    var a = abs(x)
    if a < 1e-30:
        return 0
    var ulp = Float64(1)
    while ulp > a:
        ulp /= 2
    while ulp * 2 <= a:
        ulp *= 2
    comptime mantissa_steps = 1 << FPUtils[dtype].mantissa_width()
    return ulp / Float64(mantissa_steps)


def run_mhc_post[
    hc: Int, hidden: Int, dtype: DType = .bfloat16
](ctx: DeviceContext, num_rows: Int) raises:
    var n_streams = num_rows * hc * hidden
    var n_y = num_rows * hidden
    var n_post = num_rows * hc
    var n_comb = num_rows * hc * hc

    var streams_h = ctx.enqueue_create_host_buffer[dtype](n_streams)
    var y_h = ctx.enqueue_create_host_buffer[dtype](n_y)
    var post_h = ctx.enqueue_create_host_buffer[.float32](n_post)
    var comb_h = ctx.enqueue_create_host_buffer[.float32](n_comb)
    var out_h = ctx.enqueue_create_host_buffer[dtype](n_streams)

    # Residual streams grow through the network, so a few units of scale.
    randn[dtype](streams_h.unsafe_ptr(), n_streams, 0.0, 3.0)
    randn[dtype](y_h.unsafe_ptr(), n_y, 0.0, 1.0)
    for i in range(n_post):
        post_h[i] = Float32(random_float64(0, 2))
    # Roughly doubly stochastic: a random positive matrix, one row and one
    # column normalization, so no entry is symmetric with its transpose.
    for t in range(num_rows):
        var base = t * hc * hc
        for i in range(hc * hc):
            comb_h[base + i] = Float32(random_float64(0.01, 1))
        for j in range(hc):
            var s = Float32(0)
            for k in range(hc):
                s += comb_h[base + j * hc + k]
            for k in range(hc):
                comb_h[base + j * hc + k] /= s
        for k in range(hc):
            var s = Float32(0)
            for j in range(hc):
                s += comb_h[base + j * hc + k]
            for j in range(hc):
                comb_h[base + j * hc + k] /= s

    var streams_d = ctx.enqueue_create_buffer[dtype](n_streams)
    var y_d = ctx.enqueue_create_buffer[dtype](n_y)
    var post_d = ctx.enqueue_create_buffer[.float32](n_post)
    var comb_d = ctx.enqueue_create_buffer[.float32](n_comb)
    var out_d = ctx.enqueue_create_buffer[dtype](n_streams)
    ctx.enqueue_copy(streams_d, streams_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.enqueue_copy(post_d, post_h)
    ctx.enqueue_copy(comb_d, comb_h)

    var streams_shape = row_major((num_rows, hc, hidden))
    mhc_post[
        hc_mult=hc,
        hidden_size=hidden,
        target="gpu",
        block_size=mhc_post_block_size[dtype, hidden](),
    ](
        TileTensor(out_d, streams_shape),
        TileTensor(streams_d, streams_shape).as_imm(),
        TileTensor(y_d, row_major((num_rows, hidden))).as_imm(),
        TileTensor(post_d, row_major((num_rows, hc))).as_imm(),
        TileTensor(comb_d, row_major((num_rows, hc, hc))).as_imm(),
        ctx,
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # fp64 truth. The kernel rounds to `dtype` once, so it may be off by one
    # ulp, plus fp32 accumulation error (a few fp32 eps of the summed
    # term magnitudes) where the terms cancel and the ulp gets tiny.
    comptime fp32_accum_rel_tol = 2.0**-21
    var max_err = Float64(0)
    var max_err_over_tol = Float64(0)
    var bad = 0
    for t in range(num_rows):
        for k in range(hc):
            var p = Float64(post_h[t * hc + k])
            var c = Array[Float64, hc](fill=0)
            for j in range(hc):
                c[j] = Float64(comb_h[t * hc * hc + j * hc + k])
            for i in range(hidden):
                var ref_v = p * Float64(y_h[t * hidden + i])
                var mag = abs(ref_v)
                for j in range(hc):
                    var term = c[j] * Float64(
                        streams_h[(t * hc + j) * hidden + i]
                    )
                    ref_v += term
                    mag += abs(term)
                var got = Float64(out_h[(t * hc + k) * hidden + i])
                var err = abs(got - ref_v)
                var tol = _ulp[dtype](ref_v) + mag * fp32_accum_rel_tol
                max_err = max(max_err, err)
                max_err_over_tol = max(max_err_over_tol, err / tol)
                # Negated so a NaN output counts as a mismatch.
                if not (err <= tol):
                    bad += 1
    print(
        "mhc_post ",
        dtype,
        " T=",
        num_rows,
        " max_abs_err=",
        max_err,
        " max_err/tol=",
        max_err_over_tol,
        " mismatches=",
        bad,
    )
    assert_true(bad == 0, "mhc_post differs from the fp64 reference")


def test_mhc_post_rejects_bad_layouts(ctx: DeviceContext) raises:
    """A strided or misaligned view must raise before anything launches."""
    comptime hc = 4
    comptime hidden = 4096
    comptime num_rows = 2
    var streams_d = ctx.enqueue_create_buffer[.bfloat16](
        2 * num_rows * hc * hidden + 8
    )
    var y_d = ctx.enqueue_create_buffer[.bfloat16](num_rows * hidden)
    var post_d = ctx.enqueue_create_buffer[.float32](num_rows * hc)
    var comb_d = ctx.enqueue_create_buffer[.float32](num_rows * hc * hc)
    var out_d = ctx.enqueue_create_buffer[.bfloat16](num_rows * hc * hidden)
    var streams_shape = row_major((num_rows, hc, hidden))
    var output = TileTensor(out_d, streams_shape)
    var y = TileTensor(y_d, row_major((num_rows, hidden))).as_imm()
    var post = TileTensor(post_d, row_major((num_rows, hc))).as_imm()
    var comb = TileTensor(comb_d, row_major((num_rows, hc, hc))).as_imm()

    # Every other element along the hidden dimension.
    var strided = TileTensor(
        streams_d,
        Layout(
            Coord(num_rows, hc, hidden),
            Coord(2 * hc * hidden, 2 * hidden, 2),
        ),
    ).as_imm()
    with assert_raises(contains="unit inner strides"):
        mhc_post[hc_mult=hc, hidden_size=hidden, target="gpu"](
            output, strided, y, post, comb, ctx
        )

    # Shifted by one element, so no row starts on a 16-byte boundary.
    var misaligned = TileTensor(
        streams_d.unsafe_ptr().unsafe_offset(1), streams_shape
    ).as_imm()
    with assert_raises(contains="unit inner strides"):
        mhc_post[hc_mult=hc, hidden_size=hidden, target="gpu"](
            output, misaligned, y, post, comb, ctx
        )


def test_mhc_post_zero_rows(ctx: DeviceContext) raises:
    """An empty batch passes the checks and launches nothing."""
    comptime hc = 4
    comptime hidden = 4096
    var streams_d = ctx.enqueue_create_buffer[.bfloat16](hc * hidden)
    var y_d = ctx.enqueue_create_buffer[.bfloat16](hidden)
    var post_d = ctx.enqueue_create_buffer[.float32](hc)
    var comb_d = ctx.enqueue_create_buffer[.float32](hc * hc)
    var out_d = ctx.enqueue_create_buffer[.bfloat16](hc * hidden)
    var streams_shape = row_major((0, hc, hidden))
    mhc_post[hc_mult=hc, hidden_size=hidden, target="gpu"](
        TileTensor(out_d, streams_shape),
        TileTensor(streams_d, streams_shape).as_imm(),
        TileTensor(y_d, row_major((0, hidden))).as_imm(),
        TileTensor(post_d, row_major((0, hc))).as_imm(),
        TileTensor(comb_d, row_major((0, hc, hc))).as_imm(),
        ctx,
    )
    ctx.synchronize()


def test_mhc_post_non_temporal_threshold(ctx: DeviceContext) raises:
    # Both arms compute the same values, so only this pins the switch point.
    var num_cus = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var first_nt_row: Int
    comptime if default_accelerator().is_amd_gpu():
        first_nt_row = num_cus
    else:
        first_nt_row = 8 * num_cus + 1
    assert_true(not mhc_post_non_temporal(first_nt_row - 1, num_cus))
    assert_true(mhc_post_non_temporal(first_nt_row, num_cus))


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        test_mhc_post_rejects_bad_layouts(ctx)
        test_mhc_post_zero_rows(ctx)
        test_mhc_post_non_temporal_threshold(ctx)
        # The loads and stores turn non-temporal from 1 token per CU on AMD
        # and past 8 per SM on NVIDIA.
        for num_rows in [1, 4, 17, 64, 255, 256, 1024, 4096, 16384]:
            run_mhc_post[hc=4, hidden=4096](ctx, num_rows)
        # 256-thread blocks do not split 5120 bf16 columns; 128 do.
        for num_rows in [1, 17, 1185, 8168]:
            run_mhc_post[hc=4, hidden=5120](ctx, num_rows)
        # Other element sizes change the vector width, chunk and k_blocks.
        for num_rows in [1, 17, 256]:
            run_mhc_post[hc=4, hidden=4096, dtype=.float32](ctx, num_rows)
            run_mhc_post[hc=4, hidden=4096, dtype=.float16](ctx, num_rows)
