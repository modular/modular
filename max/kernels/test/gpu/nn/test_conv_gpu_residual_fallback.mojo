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
"""`conv_gpu(has_residual=True)` on shapes the fused SM100 routes decline.

`D = Conv(A, B) + beta * C` must hold on every path `conv_gpu` can take, not
only the SM100 kernels that fuse it. These shapes (64 -> 64 and 24 -> 32
channels, fp16, bf16 and fp32, both filter layouts) miss the dedicated
routes and land on cuDNN, the im2col matmul or the naive kernel, so the test
catches a path that never reads `source_ptr`.

The expected output is `conv_gpu` without the residual followed by the
residual and epilogue arithmetic on the host, rounding to the output dtype
after the convolution and after each operation, which is the order the fused
kernels use. Operands are small integers, so the convolution is exact.
"""

from std.sys import has_accelerator
from std.testing import assert_equal

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext, HostBuffer
from nn.conv.conv import conv_gpu

comptime PLAIN = 0
comptime STORE = 1
comptime COMPUTE = 2


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def run_case[
    dt: DType, fcrs: Bool, epilogue: Int, residual: Bool = True
](
    ctx: DeviceContext,
    h: Int,
    w: Int,
    cin: Int,
    cout: Int,
    beta: Float32,
    name: StringLiteral,
) raises -> Int:
    """Runs one 3x3 pad-1 problem with a residual.

    Parameters:
        dt: Element type of activation, filter, output and source.
        fcrs: Hand `conv_gpu` an FCRS filter instead of RSCF.
        epilogue: `PLAIN` (store `conv + beta * source`), `STORE` (a storing
            epilogue adding a per-channel bias to that sum, what the
            `conv2d_residual_add` graph op passes) or `COMPUTE` (`relu` of
            that sum).
        residual: Bank the residual: `conv_gpu` gets a source and `beta`.

    Args:
        ctx: Device context.
        h: Input height (output height with pad 1).
        w: Input width.
        cin: Input channels.
        cout: Output channels.
        beta: Residual scale factor (exact in the dtype).
        name: Label printed with the result.

    Returns:
        The number of output elements that differ from the expectation.
    """
    var n_act = h * w * cin
    var n_filt = cout * 9 * cin
    var n_out = h * w * cout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](cout)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var plain_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _hash_value(i, h + 1).cast[dt]()
    for i in range(cout):
        bias_host[i] = _hash_value(i, 5).cast[dt]()
    for i in range(n_out):
        src_host[i] = _hash_value(i, 9).cast[dt]()
    # The filter values are defined in FCRS order; RSCF stores the same ones.
    var filt_view = TileTensor(
        filt_host,
        row_major(cout, cin, 3, 3) if fcrs else row_major(3, 3, cin, cout),
    )
    for f in range(cout):
        for c in range(cin):
            for r in range(3):
                for s in range(3):
                    var v = _hash_value(
                        ((f * cin + c) * 3 + r) * 3 + s, 3
                    ).cast[dt]()
                    comptime if fcrs:
                        filt_view[f, c, r, s] = v
                    else:
                        filt_view[r, s, c, f] = v

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](cout)
    var src_dev = ctx.enqueue_create_buffer[dt](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var plain_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(src_dev, src_host)
    # Poison the output: a path that never stores must not match.
    out_dev.enqueue_fill(Scalar[dt](-7))

    var act_tt = TileTensor(act_dev, row_major(1, h, w, cin))
    var filt_tt = TileTensor(
        filt_dev,
        row_major(cout, cin, 3, 3) if fcrs else row_major(3, 3, cin, cout),
    )
    var out_tt = TileTensor(out_dev, row_major(1, h, w, cout))
    var plain_tt = TileTensor(plain_dev, row_major(1, h, w, cout))
    var bias_tt = TileTensor(bias_dev, row_major(cout))
    var src_p: MutPointer[Scalar[dt], MutAnyOrigin] = src_dev.unsafe_ptr()
    # `out_tt` is also the output, so the closure writes through an untracked
    # view of it.
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        out_tt.layout,
    )

    @always_inline
    @__allow_legacy_any_origin_fields
    def store_bias[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_view}:
        comptime assert idx.rank == 4, "conv_gpu closures take NHWC"
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        out_view.store[width=width, alignment=alignment](
            idx, (val + b).cast[dt]()
        )

    @always_inline
    def relu[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        return max(val, SIMD[_dtype, width](0))

    var strides = Coord(1, 1)
    var pads = Coord(1, 1, 1, 1)

    # The convolution alone, through the same dispatcher.
    conv_gpu[dt, dt, dt, filter_is_fcrs=fcrs](
        act_tt, filt_tt, plain_tt, strides, strides, pads, 1, ctx
    )

    comptime match epilogue:
        case == PLAIN:
            conv_gpu[dt, dt, dt, filter_is_fcrs=fcrs, has_residual=True](
                act_tt,
                filt_tt,
                out_tt,
                strides,
                strides,
                pads,
                1,
                ctx,
                src_p,
                beta,
            )
        case == STORE:
            conv_gpu[
                dt,
                dt,
                dt,
                has_epilogue_fn=True,
                filter_is_fcrs=fcrs,
                has_residual=residual,
            ](
                act_tt,
                filt_tt,
                out_tt,
                strides,
                strides,
                pads,
                1,
                ctx,
                src_p,
                beta,
                epilogue_fn=store_bias,
            )
        case _:
            conv_gpu[
                dt,
                dt,
                dt,
                filter_is_fcrs=fcrs,
                has_residual=True,
                has_compute_fn=True,
            ](
                act_tt,
                filt_tt,
                out_tt,
                strides,
                strides,
                pads,
                1,
                ctx,
                src_p,
                beta,
                compute_fn=relu,
            )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(plain_host, plain_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var c = i % cout
        var x = plain_host[i].cast[.float32]()
        var beta_used = beta if residual else Float32(0)
        var scaled = (beta_used * src_host[i].cast[.float32]()).cast[dt]()
        var sum = (x + scaled.cast[.float32]()).cast[dt]()
        var want: Float32
        comptime match epilogue:
            case == PLAIN:
                want = sum.cast[.float32]()
            case == STORE:
                want = (
                    (sum.cast[.float32]() + bias_host[c].cast[.float32]())
                    .cast[dt]()
                    .cast[.float32]()
                )
            case _:
                want = max(sum.cast[.float32](), Float32(0))
        if out_host[i].cast[.float32]() != want:
            if bad < 3:
                print(
                    t"     first mismatches: element {i} got {out_host[i]} want"
                    t" {want} conv {plain_host[i]} source {src_host[i]} bias"
                    t" {bias_host[c]}"
                )
            bad += 1
    comptime layout_name = "fcrs" if fcrs else "rscf"
    print(
        t"   {name} {dt} {layout_name} epilogue {epilogue} mismatches:"
        t" {bad} of {n_out}"
    )

    _ = act_dev^
    _ = filt_dev^
    _ = bias_dev^
    _ = src_dev^
    _ = out_dev^
    _ = plain_dev^
    return bad


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    comptime f16 = DType.float16
    comptime f32 = DType.float32
    comptime bf16 = DType.bfloat16
    var failures = 0
    with DeviceContext() as ctx:
        print("--- fp16, 64 -> 64 channels ---")
        failures += run_case[f16, True, PLAIN](ctx, 12, 16, 64, 64, 1.0, "c64")
        failures += run_case[f16, True, STORE](ctx, 12, 16, 64, 64, 1.0, "c64")
        failures += run_case[f16, True, COMPUTE](
            ctx, 12, 16, 64, 64, 0.5, "c64"
        )
        failures += run_case[f16, False, PLAIN](ctx, 12, 16, 64, 64, 1.0, "c64")
        failures += run_case[f16, False, STORE](ctx, 12, 16, 64, 64, 1.0, "c64")
        print("--- fp32 ---")
        failures += run_case[f32, True, PLAIN](ctx, 12, 16, 64, 64, 0.5, "c64")
        failures += run_case[f32, True, STORE](ctx, 12, 16, 64, 64, 1.0, "c64")
        failures += run_case[f32, False, PLAIN](ctx, 12, 16, 64, 64, 1.0, "c64")
        print("--- bf16, 24 -> 32 channels (not 64-byte aligned) ---")
        failures += run_case[bf16, True, PLAIN](ctx, 12, 16, 24, 32, 1.0, "c24")
        failures += run_case[bf16, True, STORE](ctx, 12, 16, 24, 32, 1.0, "c24")
        failures += run_case[bf16, False, COMPUTE](
            ctx, 12, 16, 24, 32, 1.0, "c24"
        )
    assert_equal(failures, 0, "paths that drop or mis-apply the residual")
    print("ALL CONV_GPU RESIDUAL FALLBACK TESTS PASSED")
