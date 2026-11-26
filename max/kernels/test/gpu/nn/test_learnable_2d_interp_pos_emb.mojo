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

"""Tests for learnable 2D interpolated position embedding GPU kernel.

Verifies correct behavior for:
  - No interpolation (h==H, w==W) with t=1
  - No interpolation with temporal embedding (t>1)
  - Bicubic interpolation (constant field preserved)
  - Multiple videos with mixed shapes (GPU vs host reference)
  - Bilinear interpolation, including temporal and 1x1 grids and captured
    BF16 rounding ties
"""

from std.math import clamp, floor, sin, cos, max, min
from std.memory import bitcast

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.learnable_2d_interp_pos_emb import learnable_2d_interp_pos_emb
from std.testing import assert_almost_equal, assert_equal


def test_no_interp_no_temporal[dtype: DType](ctx: DeviceContext) raises:
    """When h==H, w==W, t=1: output = x + weight.flatten()."""
    comptime H = 4
    comptime W = 4
    comptime dim = 8
    comptime N = 1
    comptime L = H * W
    comptime num_frames = 4

    # Allocate host buffers.
    var x_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var w_ptr = ctx.enqueue_create_host_buffer[dtype](H * W * dim)
    var g_ptr = ctx.enqueue_create_host_buffer[.int64](N * 3)
    var tw_ptr = ctx.enqueue_create_host_buffer[.float32](num_frames * dim)
    var out_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)

    # Fill via TileTensors.
    var x_tt = TileTensor(x_ptr, row_major[L, dim]())
    var w_tt = TileTensor(w_ptr, row_major[H, W, dim]())
    var g_tt = TileTensor(g_ptr, row_major[N, 3]())
    var tw_tt = TileTensor(tw_ptr, row_major[num_frames, dim]())

    for l in range(L):
        for d in range(dim):
            x_tt[l, d] = Scalar[dtype](Float32(l * dim + d) * 0.001)

    for h in range(H):
        for w in range(W):
            for d in range(dim):
                w_tt[h, w, d] = Scalar[dtype](
                    Float32(h * W + w) + Float32(d) * 0.01
                )

    g_tt[0, 0] = 1  # t
    g_tt[0, 1] = Int64(H)
    g_tt[0, 2] = Int64(W)

    # Fill time_weight (won't be used since t=1, but must be valid).
    var half = dim // 2
    for t in range(num_frames):
        for d in range(half):
            var omega = 1.0 / (
                Float32(10000.0) ** (Float32(2 * d) / Float32(dim))
            )
            var angle = Float32(t) * Float32(omega)
            tw_tt[t, d] = sin(angle)
            tw_tt[t, half + d] = cos(angle)

    # Run GPU kernel.
    var x_dev = ctx.enqueue_create_buffer[dtype](L * dim)
    var w_dev = ctx.enqueue_create_buffer[dtype](H * W * dim)
    var g_dev = ctx.enqueue_create_buffer[.int64](N * 3)
    var tw_dev = ctx.enqueue_create_buffer[.float32](num_frames * dim)
    var out_dev = ctx.enqueue_create_buffer[dtype](L * dim)

    ctx.enqueue_copy(x_dev, x_ptr)
    ctx.enqueue_copy(w_dev, w_ptr)
    ctx.enqueue_copy(g_dev, g_ptr)
    ctx.enqueue_copy(tw_dev, tw_ptr)

    var x_d = TileTensor(x_dev, row_major(L, dim))
    var w_d = TileTensor(w_dev, row_major(H, W, dim))
    var g_d = TileTensor(g_dev, row_major(N, Idx[3]))
    var tw_d = TileTensor(tw_dev, row_major(num_frames, dim))
    var out_d = TileTensor(out_dev, row_major(L, dim))

    learnable_2d_interp_pos_emb(out_d, x_d, w_d, g_d, tw_d, ctx)
    ctx.enqueue_copy(out_ptr, out_dev)
    ctx.synchronize()

    var out_tt = TileTensor(out_ptr, row_major[L, dim]())
    for h in range(H):
        for w in range(W):
            var pos = h * W + w
            for d in range(dim):
                var expected = Scalar[dtype](
                    Float32(x_tt[pos, d]) + Float32(w_tt[h, w, d])
                )
                assert_almost_equal(
                    out_tt[pos, d],
                    expected,
                    atol=1e-5,
                    msg="no_interp pos=" + String(pos) + " d=" + String(d),
                )


def test_no_interp_with_temporal[dtype: DType](ctx: DeviceContext) raises:
    """When h==H, w==W, t>1: temporal sincos embedding is added."""
    comptime H = 4
    comptime W = 4
    comptime dim = 8
    comptime N = 1
    comptime t_val = 3
    comptime L = t_val * H * W
    comptime num_frames = 4

    var x_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var w_ptr = ctx.enqueue_create_host_buffer[dtype](H * W * dim)
    var g_ptr = ctx.enqueue_create_host_buffer[.int64](N * 3)
    var tw_ptr = ctx.enqueue_create_host_buffer[.float32](num_frames * dim)
    var out_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)

    # x_ptr is consumed by the kernel as input; zero-init since the test
    # doesn't fill it explicitly.
    for i in range(L * dim):
        x_ptr[i] = Scalar[dtype](0)

    # var x_tt = TileTensor(x_ptr, row_major[L, dim]())
    var w_tt = TileTensor(w_ptr, row_major[H, W, dim]())
    var g_tt = TileTensor(g_ptr, row_major[N, 3]())
    var tw_tt = TileTensor(tw_ptr, row_major[num_frames, dim]())

    for h in range(H):
        for w in range(W):
            for d in range(dim):
                w_tt[h, w, d] = Scalar[dtype](
                    Float32(h * W + w) + Float32(d) * 0.01
                )

    g_tt[0, 0] = Int64(t_val)
    g_tt[0, 1] = Int64(H)
    g_tt[0, 2] = Int64(W)

    var half = dim // 2
    for t in range(num_frames):
        for d in range(half):
            var omega = 1.0 / (
                Float32(10000.0) ** (Float32(2 * d) / Float32(dim))
            )
            var angle = Float32(t) * Float32(omega)
            tw_tt[t, d] = sin(angle)
            tw_tt[t, half + d] = cos(angle)

    # GPU.
    var x_dev = ctx.enqueue_create_buffer[dtype](L * dim)
    var w_dev = ctx.enqueue_create_buffer[dtype](H * W * dim)
    var g_dev = ctx.enqueue_create_buffer[.int64](N * 3)
    var tw_dev = ctx.enqueue_create_buffer[.float32](num_frames * dim)
    var out_dev = ctx.enqueue_create_buffer[dtype](L * dim)

    ctx.enqueue_copy(x_dev, x_ptr)
    ctx.enqueue_copy(w_dev, w_ptr)
    ctx.enqueue_copy(g_dev, g_ptr)
    ctx.enqueue_copy(tw_dev, tw_ptr)

    var x_d = TileTensor(x_dev, row_major(L, dim))
    var w_d = TileTensor(w_dev, row_major(H, W, dim))
    var g_d = TileTensor(g_dev, row_major(N, Idx[3]))
    var tw_d = TileTensor(tw_dev, row_major(num_frames, dim))
    var out_d = TileTensor(out_dev, row_major(L, dim))

    learnable_2d_interp_pos_emb(out_d, x_d, w_d, g_d, tw_d, ctx)
    ctx.enqueue_copy(out_ptr, out_dev)
    ctx.synchronize()

    # Verify: output[frame*H*W + h*W + w, d] == weight[h,w,d] + time_weight[frame,d]
    var out_tt = TileTensor(out_ptr, row_major[L, dim]())
    for frame in range(t_val):
        for h in range(H):
            for w in range(W):
                var pos = frame * (H * W) + h * W + w
                for d in range(dim):
                    var expected = Scalar[dtype](
                        Float32(w_tt[h, w, d]) + Float32(tw_tt[frame, d])
                    )
                    assert_almost_equal(
                        out_tt[pos, d],
                        expected,
                        atol=1e-5,
                        msg="temporal frame="
                        + String(frame)
                        + " pos="
                        + String(h * W + w)
                        + " d="
                        + String(d),
                    )


def test_bicubic_constant_field[dtype: DType](ctx: DeviceContext) raises:
    """Bicubic interpolation of a constant field preserves the constant."""
    comptime H = 4
    comptime W = 4
    comptime dim = 8
    comptime N = 1
    comptime h_out = 8
    comptime w_out = 6
    comptime L = h_out * w_out
    comptime num_frames = 4

    var x_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var w_ptr = ctx.enqueue_create_host_buffer[dtype](H * W * dim)
    var g_ptr = ctx.enqueue_create_host_buffer[.int64](N * 3)
    var tw_ptr = ctx.enqueue_create_host_buffer[.float32](num_frames * dim)
    var out_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)

    # x_ptr is consumed by the kernel as input; zero-init since the test
    # doesn't fill it explicitly.
    for i in range(L * dim):
        x_ptr[i] = Scalar[dtype](0)

    # var x_tt = TileTensor(x_ptr, row_major[L, dim]())
    var w_tt = TileTensor(w_ptr, row_major[H, W, dim]())
    var g_tt = TileTensor(g_ptr, row_major[N, 3]())
    var tw_tt = TileTensor(tw_ptr, row_major[num_frames, dim]())

    # weight[:, :, d] = d + 1.0 — spatially constant per channel.
    for h in range(H):
        for w in range(W):
            for d in range(dim):
                w_tt[h, w, d] = Scalar[dtype](Float32(d) + 1.0)

    g_tt[0, 0] = 1
    g_tt[0, 1] = Int64(h_out)
    g_tt[0, 2] = Int64(w_out)

    var half = dim // 2
    for t in range(num_frames):
        for d in range(half):
            var omega = 1.0 / (
                Float32(10000.0) ** (Float32(2 * d) / Float32(dim))
            )
            var angle = Float32(t) * Float32(omega)
            tw_tt[t, d] = sin(angle)
            tw_tt[t, half + d] = cos(angle)

    # GPU.
    var x_dev = ctx.enqueue_create_buffer[dtype](L * dim)
    var w_dev = ctx.enqueue_create_buffer[dtype](H * W * dim)
    var g_dev = ctx.enqueue_create_buffer[.int64](N * 3)
    var tw_dev = ctx.enqueue_create_buffer[.float32](num_frames * dim)
    var out_dev = ctx.enqueue_create_buffer[dtype](L * dim)

    ctx.enqueue_copy(x_dev, x_ptr)
    ctx.enqueue_copy(w_dev, w_ptr)
    ctx.enqueue_copy(g_dev, g_ptr)
    ctx.enqueue_copy(tw_dev, tw_ptr)

    var x_d = TileTensor(x_dev, row_major(L, dim))
    var w_d = TileTensor(w_dev, row_major(H, W, dim))
    var g_d = TileTensor(g_dev, row_major(N, Idx[3]))
    var tw_d = TileTensor(tw_dev, row_major(num_frames, dim))
    var out_d = TileTensor(out_dev, row_major(L, dim))

    learnable_2d_interp_pos_emb(out_d, x_d, w_d, g_d, tw_d, ctx)
    ctx.enqueue_copy(out_ptr, out_dev)
    ctx.synchronize()

    var out_tt = TileTensor(out_ptr, row_major[L, dim]())
    for pos in range(L):
        for d in range(dim):
            var expected = Scalar[dtype](Float32(d) + 1.0)
            assert_almost_equal(
                out_tt[pos, d],
                expected,
                atol=1e-4,
                msg="bicubic constant pos=" + String(pos) + " d=" + String(d),
            )


@inline(.always)
def _cubic_weight(x: Float32) -> Float32:
    """Catmull-Rom cubic weight for host-side reference computation."""
    var a: Float32 = -0.75
    var ax = abs(x)
    var ax2 = ax * ax
    var ax3 = ax2 * ax
    if ax <= 1:
        return (a + 2) * ax3 - (a + 3) * ax2 + 1
    elif ax < 2:
        return a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a
    else:
        return 0


def test_multi_video[dtype: DType](ctx: DeviceContext) raises:
    """Multiple videos with mixed shapes: GPU matches host reference."""
    comptime H = 4
    comptime W = 4
    comptime dim = 8
    comptime N = 3
    comptime num_frames = 4
    # Video 0: t=1, h=4, w=4  -> 16  (no interp)
    # Video 1: t=2, h=6, w=8  -> 96  (interp + temporal)
    # Video 2: t=1, h=3, w=5  -> 15  (interp)
    comptime L = 16 + 96 + 15

    var x_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var w_ptr = ctx.enqueue_create_host_buffer[dtype](H * W * dim)
    var g_ptr = ctx.enqueue_create_host_buffer[.int64](N * 3)
    var tw_ptr = ctx.enqueue_create_host_buffer[.float32](num_frames * dim)
    var out_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var exp_ptr = ctx.enqueue_create_host_buffer[dtype](L * dim)

    var x_tt = TileTensor(x_ptr, row_major[L, dim]())
    var w_tt = TileTensor(w_ptr, row_major[H, W, dim]())
    var g_tt = TileTensor(g_ptr, row_major[N, 3]())
    var tw_tt = TileTensor(tw_ptr, row_major[num_frames, dim]())
    var exp_tt = TileTensor(exp_ptr, row_major[L, dim]())

    for l in range(L):
        for d in range(dim):
            x_tt[l, d] = Scalar[dtype](Float32((l * dim + d) % 100) * 0.01)

    for h in range(H):
        for w in range(W):
            for d in range(dim):
                w_tt[h, w, d] = Scalar[dtype](
                    Float32(h * W + w) + Float32(d) * 0.01
                )

    g_tt[0, 0] = 1
    g_tt[0, 1] = 4
    g_tt[0, 2] = 4
    g_tt[1, 0] = 2
    g_tt[1, 1] = 6
    g_tt[1, 2] = 8
    g_tt[2, 0] = 1
    g_tt[2, 1] = 3
    g_tt[2, 2] = 5

    var half = dim // 2
    for t in range(num_frames):
        for d in range(half):
            var omega = 1.0 / (
                Float32(10000.0) ** (Float32(2 * d) / Float32(dim))
            )
            var angle = Float32(t) * Float32(omega)
            tw_tt[t, d] = sin(angle)
            tw_tt[t, half + d] = cos(angle)

    # Host reference computation.
    var offset: Int = 0
    for img in range(N):
        var t = Int(g_tt[img, 0])
        var h = Int(g_tt[img, 1])
        var w = Int(g_tt[img, 2])
        var no_interp = h == H and w == W
        var scale_h = Float32(H) / Float32(h)
        var scale_w = Float32(W) / Float32(w)

        for tt in range(t):
            for hh in range(h):
                for ww in range(w):
                    var ih_floor: Int = 0
                    var iw_floor: Int = 0
                    var dy: Float32 = 0
                    var dx: Float32 = 0
                    if not no_interp:
                        var in_h = (Float32(hh) + 0.5) * scale_h - 0.5
                        var in_w = (Float32(ww) + 0.5) * scale_w - 0.5
                        ih_floor = Int(floor(in_h))
                        iw_floor = Int(floor(in_w))
                        dy = in_h - Float32(ih_floor)
                        dx = in_w - Float32(iw_floor)

                    for d in range(dim):
                        var pos_val: Float32
                        if no_interp:
                            pos_val = Float32(w_tt[hh, ww, d])
                        else:
                            var val: Float32 = 0
                            for i in range(4):
                                var yp = clamp(ih_floor + i - 1, 0, H - 1)
                                var wy = _cubic_weight(Float32(i) - 1.0 - dy)
                                for j in range(4):
                                    var xp = clamp(iw_floor + j - 1, 0, W - 1)
                                    var wx = _cubic_weight(
                                        Float32(j) - 1.0 - dx
                                    )
                                    val += Float32(w_tt[yp, xp, d]) * wy * wx
                            pos_val = val

                        var time_val: Float32 = 0
                        if t > 1:
                            time_val = Float32(tw_tt[tt, d])

                        exp_tt[offset, d] = Scalar[dtype](
                            Float32(x_tt[offset, d]) + pos_val + time_val
                        )
                    offset += 1

    # GPU.
    var x_dev = ctx.enqueue_create_buffer[dtype](L * dim)
    var w_dev = ctx.enqueue_create_buffer[dtype](H * W * dim)
    var g_dev = ctx.enqueue_create_buffer[.int64](N * 3)
    var tw_dev = ctx.enqueue_create_buffer[.float32](num_frames * dim)
    var out_dev = ctx.enqueue_create_buffer[dtype](L * dim)

    ctx.enqueue_copy(x_dev, x_ptr)
    ctx.enqueue_copy(w_dev, w_ptr)
    ctx.enqueue_copy(g_dev, g_ptr)
    ctx.enqueue_copy(tw_dev, tw_ptr)

    var x_d = TileTensor(x_dev, row_major(L, dim))
    var w_d = TileTensor(w_dev, row_major(H, W, dim))
    var g_d = TileTensor(g_dev, row_major(N, Idx[3]))
    var tw_d = TileTensor(tw_dev, row_major(num_frames, dim))
    var out_d = TileTensor(out_dev, row_major(L, dim))

    learnable_2d_interp_pos_emb(out_d, x_d, w_d, g_d, tw_d, ctx)
    ctx.enqueue_copy(out_ptr, out_dev)
    ctx.synchronize()

    var out_tt = TileTensor(out_ptr, row_major[L, dim]())
    for i in range(L):
        for d in range(dim):
            assert_almost_equal(
                out_tt[i, d],
                exp_tt[i, d],
                atol=1e-4,
                msg="multi_video ref mismatch at i="
                + String(i)
                + " d="
                + String(d),
            )


def test_sincos_embed(ctx: DeviceContext) raises:
    """Verify sincos positional embedding values."""
    comptime dim = 4
    comptime num_frames = 2
    var tw_ptr = ctx.enqueue_create_host_buffer[.float32](num_frames * dim)
    var tw_tt = TileTensor(tw_ptr, row_major[num_frames, dim]())

    var half = dim // 2
    for t in range(num_frames):
        for d in range(half):
            var omega = 1.0 / (
                Float32(10000.0) ** (Float32(2 * d) / Float32(dim))
            )
            var angle = Float32(t) * Float32(omega)
            tw_tt[t, d] = sin(angle)
            tw_tt[t, half + d] = cos(angle)

    # t=0: sin(0)=0, cos(0)=1 for all d.
    assert_almost_equal(tw_tt[0, 0], Float32(0), atol=1e-6, msg="sin(0) d=0")
    assert_almost_equal(tw_tt[0, 1], Float32(0), atol=1e-6, msg="sin(0) d=1")
    assert_almost_equal(tw_tt[0, 2], Float32(1), atol=1e-6, msg="cos(0) d=0")
    assert_almost_equal(tw_tt[0, 3], Float32(1), atol=1e-6, msg="cos(0) d=1")

    # t=1: omega_0 = 1/10000^0 = 1.0, omega_1 = 1/10000^0.5 = 0.01
    assert_almost_equal(
        tw_tt[1, 0], sin(Float32(1.0)), atol=1e-6, msg="sin(omega_0)"
    )
    assert_almost_equal(
        tw_tt[1, 1], sin(Float32(0.01)), atol=1e-6, msg="sin(omega_1)"
    )
    assert_almost_equal(
        tw_tt[1, 2], cos(Float32(1.0)), atol=1e-6, msg="cos(omega_0)"
    )
    assert_almost_equal(
        tw_tt[1, 3], cos(Float32(0.01)), atol=1e-6, msg="cos(omega_1)"
    )


def test_bilinear_rectangular[dtype: DType](ctx: DeviceContext) raises:
    """Checks rectangular, native, temporal, and 1x1 grids."""
    comptime H = 4
    comptime W = 6
    comptime dim = 7
    comptime N = 5
    comptime num_frames = 2
    comptime L = 7 * 9 + 2 * 3 + H * W + num_frames * 3 * 5 + 1
    var x = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var weight = ctx.enqueue_create_host_buffer[dtype](H * W * dim)
    var grids = ctx.enqueue_create_host_buffer[.int64](N * 3)
    var time = ctx.enqueue_create_host_buffer[.float32](num_frames * dim)
    var actual = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var expected = ctx.enqueue_create_host_buffer[dtype](L * dim)
    var x_tt = TileTensor(x, row_major[L, dim]())
    var weight_tt = TileTensor(weight, row_major[H, W, dim]())
    var grids_tt = TileTensor(grids, row_major[N, 3]())
    var expected_tt = TileTensor(expected, row_major[L, dim]())
    for i in range(L * dim):
        x[i] = Scalar[dtype](Float32(i % 17 - 8) / 32.0)
    for y in range(H):
        for x_pos in range(W):
            for d in range(dim):
                weight_tt[y, x_pos, d] = Scalar[dtype](
                    Float32((y * 7 + x_pos * x_pos + d * 3) % 23) / 16.0
                )
    for i in range(num_frames * dim):
        time[i] = Float32(i % 5 - 2) / 8.0
    grids_tt[0, 0] = 1
    grids_tt[0, 1] = 7
    grids_tt[0, 2] = 9
    grids_tt[1, 0] = 1
    grids_tt[1, 1] = 2
    grids_tt[1, 2] = 3
    grids_tt[2, 0] = 1
    grids_tt[2, 1] = H
    grids_tt[2, 2] = W
    grids_tt[3, 0] = num_frames
    grids_tt[3, 1] = 3
    grids_tt[3, 2] = 5
    grids_tt[4, 0] = 1
    grids_tt[4, 1] = 1
    grids_tt[4, 2] = 1
    var offset = 0
    for image in range(N):
        var t = Int(grids_tt[image, 0])
        var h = Int(grids_tt[image, 1])
        var w = Int(grids_tt[image, 2])
        # Scale first, as the kernel and ATen do, so inexact scales round
        # the same way.
        var scale_h = Float32(H) / Float32(h)
        var scale_w = Float32(W) / Float32(w)
        for frame in range(t):
            for y in range(h):
                for x_pos in range(w):
                    var sy = max((Float32(y) + 0.5) * scale_h - 0.5, 0.0)
                    var sx = max((Float32(x_pos) + 0.5) * scale_w - 0.5, 0.0)
                    var y0 = Int(floor(sy))
                    var x0 = Int(floor(sx))
                    var y1 = min(y0 + 1, H - 1)
                    var x1 = min(x0 + 1, W - 1)
                    var dy = sy - Float32(y0)
                    var dx = sx - Float32(x0)
                    for d in range(dim):
                        var w00 = Float32(weight_tt[y0, x0, d])
                        var w01 = Float32(weight_tt[y0, x1, d])
                        var w10 = Float32(weight_tt[y1, x0, d])
                        var w11 = Float32(weight_tt[y1, x1, d])
                        var top = (w01 - w00).fma(dx, w00)
                        var bottom = (w11 - w10).fma(dx, w10)
                        var position = Scalar[dtype](
                            (bottom - top).fma(dy, top)
                        )
                        var time_val: Float32 = 0
                        if t > 1:
                            time_val = time[frame * dim + d]
                        expected_tt[offset, d] = Scalar[dtype](
                            Float32(x_tt[offset, d])
                            + Float32(position)
                            + time_val
                        )
                    offset += 1
    var x_dev = ctx.enqueue_create_buffer[dtype](L * dim)
    var weight_dev = ctx.enqueue_create_buffer[dtype](H * W * dim)
    var grids_dev = ctx.enqueue_create_buffer[.int64](N * 3)
    var time_dev = ctx.enqueue_create_buffer[.float32](num_frames * dim)
    var output_dev = ctx.enqueue_create_buffer[dtype](L * dim)
    ctx.enqueue_copy(x_dev, x)
    ctx.enqueue_copy(weight_dev, weight)
    ctx.enqueue_copy(grids_dev, grids)
    ctx.enqueue_copy(time_dev, time)
    learnable_2d_interp_pos_emb[bilinear=True](
        TileTensor(output_dev, row_major[L, dim]()),
        TileTensor(x_dev, row_major[L, dim]()),
        TileTensor(weight_dev, row_major[H, W, dim]()),
        TileTensor(grids_dev, row_major[N, 3]()),
        TileTensor(time_dev, row_major[num_frames, dim]()),
        ctx,
    )
    ctx.enqueue_copy(actual, output_dev)
    ctx.synchronize()
    for i in range(L * dim):
        assert_almost_equal(actual[i], expected[i], atol=1e-5)


def test_bilinear_bfloat16_rounding(ctx: DeviceContext) raises:
    """Requires interpolation to round before cancellation with the input."""
    comptime dtype = DType.bfloat16
    comptime H = 2
    comptime W = 2
    comptime dim = 1
    comptime L = 16
    var x = ctx.enqueue_create_buffer[dtype](L)
    x.enqueue_fill(-1)
    var weight = ctx.enqueue_create_host_buffer[dtype](H * W)
    weight[0] = 1
    weight[1] = 1.0078125
    weight[2] = 1
    weight[3] = 1.0078125
    var grids = ctx.enqueue_create_host_buffer[.int64](3)
    grids[0] = 1
    grids[1] = 4
    grids[2] = 4
    var weight_dev = ctx.enqueue_create_buffer[dtype](H * W)
    var grids_dev = ctx.enqueue_create_buffer[.int64](3)
    var time = ctx.enqueue_create_buffer[.float32](dim)
    time.enqueue_fill(0)
    var output = ctx.enqueue_create_buffer[dtype](L)
    var actual = ctx.enqueue_create_host_buffer[dtype](L)
    ctx.enqueue_copy(weight_dev, weight)
    ctx.enqueue_copy(grids_dev, grids)
    learnable_2d_interp_pos_emb[bilinear=True](
        TileTensor(output, row_major[L, dim]()),
        TileTensor(x, row_major[L, dim]()),
        TileTensor(weight_dev, row_major[H, W, dim]()),
        TileTensor(grids_dev, row_major[1, 3]()),
        TileTensor(time, row_major[1, dim]()),
        ctx,
    )
    ctx.enqueue_copy(actual, output)
    ctx.synchronize()
    for row in range(4):
        for col in range(4):
            var expected = Scalar[dtype](0.0 if col < 2 else 0.0078125)
            assert_equal(actual[row * 4 + col], expected)


def test_bilinear_bfloat16_lerp_order[
    source_height: Int, source_width: Int, target_height: Int, target_width: Int
](
    ctx: DeviceContext,
    source_col: Int,
    output_index: Int,
    corners: SIMD[DType.float32, 4],
    input_value: Float32,
    expected_bits: UInt16,
) raises:
    """Checks BF16 interpolation ties against captured reference outputs."""
    comptime dtype = DType.bfloat16
    comptime L = target_height * target_width
    var x = ctx.enqueue_create_buffer[dtype](L)
    x.enqueue_fill(Scalar[dtype](input_value))
    var weight = ctx.enqueue_create_host_buffer[dtype](
        source_height * source_width
    )
    for i in range(source_height * source_width):
        weight[i] = 0
    weight[source_col] = Scalar[dtype](corners[0])
    weight[source_col + 1] = Scalar[dtype](corners[1])
    weight[source_width + source_col] = Scalar[dtype](corners[2])
    weight[source_width + source_col + 1] = Scalar[dtype](corners[3])
    var grids = ctx.enqueue_create_host_buffer[.int64](3)
    grids[0] = 1
    grids[1] = Int64(target_height)
    grids[2] = Int64(target_width)
    var weight_dev = ctx.enqueue_create_buffer[dtype](
        source_height * source_width
    )
    var grids_dev = ctx.enqueue_create_buffer[.int64](3)
    var time = ctx.enqueue_create_buffer[.float32](1)
    time.enqueue_fill(0)
    var output = ctx.enqueue_create_buffer[dtype](L)
    var actual = ctx.enqueue_create_host_buffer[dtype](L)
    ctx.enqueue_copy(weight_dev, weight)
    ctx.enqueue_copy(grids_dev, grids)
    learnable_2d_interp_pos_emb[bilinear=True](
        TileTensor(output, row_major[L, 1]()),
        TileTensor(x, row_major[L, 1]()),
        TileTensor(weight_dev, row_major[source_height, source_width, 1]()),
        TileTensor(grids_dev, row_major[1, 3]()),
        TileTensor(time, row_major[1, 1]()),
        ctx,
    )
    ctx.enqueue_copy(actual, output)
    ctx.synchronize()
    assert_equal(bitcast[DType.uint16](actual[output_index]), expected_bits)


def main() raises:
    with DeviceContext() as ctx:
        test_sincos_embed(ctx)
        test_no_interp_no_temporal[.float32](ctx)
        test_no_interp_with_temporal[.float32](ctx)
        test_bicubic_constant_field[.float32](ctx)
        test_multi_video[.float32](ctx)
        test_bilinear_rectangular[.float32](ctx)
        test_bilinear_rectangular[.bfloat16](ctx)
        test_bilinear_rectangular[.float16](ctx)
        test_bilinear_bfloat16_rounding(ctx)
        # Captured PyTorch GPU outputs; weighted sums round across these ties.
        test_bilinear_bfloat16_lerp_order[4, 2, 5, 1](
            ctx,
            0,
            1,
            SIMD[DType.float32, 4](
                0.0267333984375, 0.0169677734375, 0.038818359375, 0.015869140625
            ),
            -0.1376953125,
            0xBDE6,
        )
        test_bilinear_bfloat16_lerp_order[2, 4, 1, 3](
            ctx,
            2,
            2,
            SIMD[DType.float32, 4](
                0.0244140625,
                0.0206298828125,
                -0.0115966796875,
                -0.00946044921875,
            ),
            -0.0634765625,
            0xBD6C,
        )
    print("All learnable_2d_interp_pos_emb tests passed!")
