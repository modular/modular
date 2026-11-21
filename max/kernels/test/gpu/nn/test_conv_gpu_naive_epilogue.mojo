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
"""A storing epilogue that adds a loaded bias, on the naive conv kernels.

`conv_gpu` with an RSCF filter and a storing epilogue on 64 -> 64 channels
runs `conv2d_gpu_naive_nhwc_rscf_epilogue_fn`, and a 3-D NDHWC convolution
with one runs `conv3d_gpu_naive_ndhwc_qrscf_epilogue_fn`. That kernel used to call the
epilogue with the default alignment of 1, so a half-precision store in the
closure was split into two byte stores, and `ld + add.f16 + st.b8` produced a
wrong low byte on the remote B200 executors (9,426 of 12,288 outputs, non-
integer values from integer operands) while it was exact on other B200 boxes.
The expectation is the plain convolution followed by the bias add on the
host. Operands are small integers, so everything is exact.
"""

from std.sys import align_of, has_accelerator
from std.testing import assert_equal

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.conv.conv import conv_gpu

comptime H = 12
comptime W = 16
comptime C = 64
comptime D3 = 4
comptime H3 = 6
comptime W3 = 8
comptime C3 = 16


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def run_case[
    dt: DType, fcrs: Bool = False, cin: Int = C, cout: Int = C
](ctx: DeviceContext, name: StringLiteral) raises -> Int:
    comptime n_act = H * W * cin
    comptime n_filt = 9 * cin * cout
    comptime n_out = H * W * cout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](cout)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var plain_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _hash_value(i, 1).cast[dt]()
    for i in range(n_filt):
        filt_host[i] = _hash_value(i, 3).cast[dt]()
    for i in range(cout):
        bias_host[i] = _hash_value(i, 5).cast[dt]()

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](cout)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var plain_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    out_dev.enqueue_fill(Scalar[dt](-7))

    var act_tt = TileTensor(act_dev, row_major(1, H, W, cin))
    var filt_tt = TileTensor(
        filt_dev,
        row_major(cout, cin, 3, 3) if fcrs else row_major(3, 3, cin, cout),
    )
    var out_tt = TileTensor(out_dev, row_major(1, H, W, cout))
    var plain_tt = TileTensor(plain_dev, row_major(1, H, W, cout))
    var bias_tt = TileTensor(bias_dev, row_major[cout]())
    # `out_tt` is also the output, so the closure writes through an untracked
    # view of it.
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        out_tt.layout,
    )

    @always_inline
    @__allow_legacy_any_origin_fields
    def add_bias[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view, var bias_tt}:
        var b = bias_tt.load[width=width, alignment=align_of[dt]()]((idx[3],))
        out_view.store[width=width, alignment=alignment](
            idx, (val + b.cast[_dtype]()).cast[dt]()
        )

    var strides = Coord(1, 1)
    var pads = Coord(1, 1, 1, 1)
    conv_gpu[dt, dt, dt, filter_is_fcrs=fcrs](
        act_tt, filt_tt, plain_tt, strides, strides, pads, 1, ctx
    )
    conv_gpu[dt, dt, dt, has_epilogue_fn=True, filter_is_fcrs=fcrs](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        strides,
        pads,
        1,
        ctx,
        None,
        Float32(0),
        add_bias,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(plain_host, plain_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var want = (
            plain_host[i].cast[.float32]()
            + bias_host[i % cout].cast[.float32]()
        ).cast[dt]()
        if out_host[i] != want:
            if bad < 3:
                print(t"   element {i} got {out_host[i]} want {want}")
            bad += 1
    print(t"   {name} mismatches: {bad} of {n_out}")

    _ = act_dev^
    _ = filt_dev^
    _ = bias_dev^
    _ = out_dev^
    _ = plain_dev^
    return bad


def run_case_3d[
    dt: DType, k: Int = 3, cin: Int = C3, cout: Int = C3
](ctx: DeviceContext, name: StringLiteral) raises -> Int:
    comptime n_act = D3 * H3 * W3 * cin
    comptime n_filt = k * k * k * cin * cout
    comptime n_out = D3 * H3 * W3 * cout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](cout)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var plain_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _hash_value(i, 1).cast[dt]()
    for i in range(n_filt):
        filt_host[i] = _hash_value(i, 3).cast[dt]()
    for i in range(cout):
        bias_host[i] = _hash_value(i, 5).cast[dt]()

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](cout)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var plain_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    out_dev.enqueue_fill(Scalar[dt](-7))

    var act_tt = TileTensor(act_dev, row_major(1, D3, H3, W3, cin))
    var filt_tt = TileTensor(filt_dev, row_major[k, k, k, cin, cout]())
    var out_tt = TileTensor(out_dev, row_major(1, D3, H3, W3, cout))
    var plain_tt = TileTensor(plain_dev, row_major(1, D3, H3, W3, cout))
    var bias_tt = TileTensor(bias_dev, row_major[cout]())
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        out_tt.layout,
    )

    @always_inline
    @__allow_legacy_any_origin_fields
    def add_bias[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view, var bias_tt}:
        comptime assert idx.rank == 5, "3-D conv closures take NDHWC"
        var b = bias_tt.load[width=width, alignment=align_of[dt]()]((idx[4],))
        out_view.store[width=width, alignment=alignment](
            idx, (val + b.cast[_dtype]()).cast[dt]()
        )

    var strides = Coord(1, 1, 1)
    comptime pad = 1 if k == 3 else 0
    var pads = Coord(pad, pad, pad, pad, pad, pad)
    conv_gpu[dt, dt, dt, filter_is_fcrs=False](
        act_tt, filt_tt, plain_tt, strides, strides, pads, 1, ctx
    )
    conv_gpu[dt, dt, dt, has_epilogue_fn=True, filter_is_fcrs=False](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        strides,
        pads,
        1,
        ctx,
        None,
        Float32(0),
        add_bias,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(plain_host, plain_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var want = (
            plain_host[i].cast[.float32]()
            + bias_host[i % cout].cast[.float32]()
        ).cast[dt]()
        if out_host[i] != want:
            if bad < 3:
                print(t"   element {i} got {out_host[i]} want {want}")
            bad += 1
    print(t"   {name} mismatches: {bad} of {n_out}")

    _ = act_dev^
    _ = filt_dev^
    _ = bias_dev^
    _ = out_dev^
    _ = plain_dev^
    return bad


def run_case_conv1d[
    dt: DType
](ctx: DeviceContext, name: StringLiteral) raises -> Int:
    """A strided 1 x 5 conv (the fp16 im2col class) whose 1,020 channels are not
    16-byte aligned, so the matmul epilogue stores one element at a time."""
    comptime W1 = 59
    comptime C1 = 512
    comptime F1 = 1020
    comptime W1_out = (W1 - 5) // 2 + 1
    comptime n_act = W1 * C1
    comptime n_filt = F1 * C1 * 5
    comptime n_out = W1_out * F1

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](F1)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var plain_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _hash_value(i, 1).cast[dt]()
    for i in range(n_filt):
        filt_host[i] = _hash_value(i, 3).cast[dt]()
    for i in range(F1):
        bias_host[i] = _hash_value(i, 5).cast[dt]()

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](F1)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var plain_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    out_dev.enqueue_fill(Scalar[dt](-7))

    var act_tt = TileTensor(act_dev, row_major(1, 1, W1, C1))
    var filt_tt = TileTensor(filt_dev, row_major(F1, C1, 1, 5))
    var out_tt = TileTensor(out_dev, row_major(1, 1, W1_out, F1))
    var plain_tt = TileTensor(plain_dev, row_major(1, 1, W1_out, F1))
    var bias_tt = TileTensor(bias_dev, row_major[F1]())
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        out_tt.layout,
    )

    @always_inline
    @__allow_legacy_any_origin_fields
    def add_bias[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view, var bias_tt}:
        var b = bias_tt.load[width=width, alignment=align_of[dt]()]((idx[3],))
        out_view.store[width=width, alignment=alignment](
            idx, (val + b.cast[_dtype]()).cast[dt]()
        )

    var strides = Coord(1, 2)
    var ones = Coord(1, 1)
    var pads = Coord(0, 0, 0, 0)
    conv_gpu[dt, dt, dt, filter_is_fcrs=True](
        act_tt, filt_tt, plain_tt, strides, ones, pads, 1, ctx
    )
    conv_gpu[dt, dt, dt, has_epilogue_fn=True, filter_is_fcrs=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        ones,
        pads,
        1,
        ctx,
        None,
        Float32(0),
        add_bias,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(plain_host, plain_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var want = (
            plain_host[i].cast[.float32]() + bias_host[i % F1].cast[.float32]()
        ).cast[dt]()
        if out_host[i] != want:
            if bad < 3:
                print(t"   element {i} got {out_host[i]} want {want}")
            bad += 1
    print(t"   {name} mismatches: {bad} of {n_out}")

    _ = act_dev^
    _ = filt_dev^
    _ = bias_dev^
    _ = out_dev^
    _ = plain_dev^
    return bad


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var failures = 0
    with DeviceContext() as ctx:
        failures += run_case[.float16](ctx, "fp16 rscf bias epilogue")
        failures += run_case[.bfloat16](ctx, "bf16 rscf bias epilogue")
        failures += run_case[.float32](ctx, "fp32 rscf bias epilogue")
        # 20 output channels leave a tail in the epilogue's vector loop; FCRS
        # runs the cuDNN fallback and its elementwise epilogue pass.
        failures += run_case[.float16, True, 24, 20](
            ctx, "fp16 fcrs 24 -> 20 bias epilogue"
        )
        failures += run_case[.float32, True, 24, 20](
            ctx, "fp32 fcrs 24 -> 20 bias epilogue"
        )
        failures += run_case_3d[.float16](ctx, "fp16 3-D bias epilogue")
        failures += run_case_3d[.bfloat16](ctx, "bf16 3-D bias epilogue")
        failures += run_case_3d[.float32](ctx, "fp32 3-D bias epilogue")
        # Tails in the matmul epilogue: bf16 3x3x3 (im2col) and 1x1x1 (matmul).
        failures += run_case_3d[.bfloat16, 3, 24, 20](
            ctx, "bf16 3-D im2col 24 -> 20 bias epilogue"
        )
        failures += run_case_3d[.bfloat16, 1, 24, 20](
            ctx, "bf16 3-D 1x1x1 24 -> 20 bias epilogue"
        )
        failures += run_case_conv1d[.float16](
            ctx, "fp16 strided conv1d, 1020 channels"
        )
        failures += run_case_conv1d[.bfloat16](
            ctx, "bf16 strided conv1d, 1020 channels"
        )
    assert_equal(failures, 0, "naive conv storing epilogue")
    print("PASSED")
