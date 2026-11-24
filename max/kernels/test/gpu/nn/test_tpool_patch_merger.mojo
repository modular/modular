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
"""Test tpool_patch_merger kernel against CPU reference behavior.

Python reference: `tpool_patch_merger` in max/nn/_graph_ops/__init__.py
  - x: (total_tokens, D), grid_thws: (n_videos, 3) [T,H,W], merge (kH,kW)
  - Output: contiguous tensor (total_output_patches, D) with temporal mean.
This test implements the same logic in Mojo (CPU reference) and compares
GPU kernel output to it.
"""


from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.tpool_patch_merger import (
    tpool_patch_merger,
)
from std.random import rand, seed
from std.testing import assert_almost_equal


def cpu_reference_one_video[
    dtype: DType,
](
    x: Pointer[Scalar[dtype], _],
    in_offset: Int,
    out_base: MutPointer[Scalar[dtype], _],
    t: Int,
    h: Int,
    w: Int,
    kH: Int,
    kW: Int,
    D: Int,
) -> None:
    """CPU reference for one video: same math as Python tpool_patch_merger."""
    var new_H = h // kH
    var new_W = w // kW
    var n_pat = new_H * new_W * kH * kW
    for pat_idx in range(n_pat):
        var sp_idx, ker_idx = divmod(pat_idx, (kH * kW))
        var nh, nw = divmod(sp_idx, new_W)
        var ph, pw = divmod(ker_idx, kW)
        var h_src = nh * kH + ph
        var w_src = nw * kW + pw
        var spatial_flat = h_src * w + w_src
        for d in range(D):
            var acc = Scalar[dtype](0)
            for ti in range(t):
                var row = in_offset + ti * (h * w) + spatial_flat
                acc += x[row * D + d]
            acc = acc / Scalar[dtype](t)
            out_base.unsafe_store(pat_idx * D + d, acc)


def test_tpool_patch_merger(
    ctx: DeviceContext,
    hidden_dim: Int = 8,
    t0: Int = 2,
    h0: Int = 4,
    w0: Int = 4,
    t1: Int = 3,
    h1: Int = 6,
    w1: Int = 6,
) raises:
    """Compare GPU kernel output to CPU reference (same math as Python)."""
    comptime dtype = DType.bfloat16
    comptime kH = 2
    comptime kW = 2
    var max_h = max(h0, h1)
    var max_w = max(w0, w1)
    var len0 = t0 * h0 * w0
    var len1 = t1 * h1 * w1
    var total_in = len0 + len1
    var n_videos: Int = 2

    # Each video outputs H*W rows (temporal dim averaged out).
    var out0_rows = h0 * w0
    var out1_rows = h1 * w1
    var total_out = out0_rows + out1_rows

    # Host buffers for input and grid_thws
    var x_host = ctx.enqueue_create_host_buffer[dtype](total_in * hidden_dim)
    var bounds_host = ctx.enqueue_create_host_buffer[.int64](n_videos * 3)
    ctx.synchronize()

    seed(42)
    rand[dtype](x_host.as_span(), min=0.0, max=1.0)

    bounds_host[0] = Int64(t0)
    bounds_host[1] = Int64(h0)
    bounds_host[2] = Int64(w0)
    bounds_host[3] = Int64(t1)
    bounds_host[4] = Int64(h1)
    bounds_host[5] = Int64(w1)

    # Device buffers
    var x_dev = ctx.enqueue_create_buffer[dtype](total_in * hidden_dim)
    var out_dev = ctx.enqueue_create_buffer[dtype](total_out * hidden_dim)
    var bounds = ctx.enqueue_create_buffer[.int64](n_videos * 3)
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(bounds, bounds_host)
    ctx.enqueue_memset(out_dev, 0)
    ctx.synchronize()

    # CPU reference: write to host buffers
    var ref_host = ctx.enqueue_create_host_buffer[dtype](total_out * hidden_dim)

    cpu_reference_one_video[dtype](
        x_host.as_span().unsafe_ptr(),
        0,
        ref_host.as_span().unsafe_ptr(),
        t0,
        h0,
        w0,
        kH,
        kW,
        hidden_dim,
    )
    cpu_reference_one_video[dtype](
        x_host.as_span().unsafe_ptr(),
        len0,
        ref_host.as_span().unsafe_ptr().unsafe_offset(out0_rows * hidden_dim),
        t1,
        h1,
        w1,
        kH,
        kW,
        hidden_dim,
    )

    # GPU kernel: contiguous output
    var x_tile = TileTensor[mut=False, Engine=_](
        x_dev,
        row_major(total_in, hidden_dim),
    )
    var out_tile = TileTensor(
        out_dev,
        row_major(total_out, hidden_dim),
    )
    var bounds_tensor = TileTensor[mut=False, Engine=_](
        bounds,
        row_major(n_videos, Idx[3]),
    )

    tpool_patch_merger(
        out_tile,
        x_tile,
        bounds_tensor,
        kH,
        kW,
        max_h,
        max_w,
        ctx,
    )

    ctx.synchronize()

    # Copy GPU output back
    var out_host = ctx.enqueue_create_host_buffer[dtype](total_out * hidden_dim)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # Compare to CPU reference; relax for bfloat16 precision
    var atol = 1e-2
    for i in range(total_out * hidden_dim):
        assert_almost_equal(out_host[i], ref_host[i], atol=atol)

    print(
        t"test_tpool_patch_merger: PASSED hidden_dim={hidden_dim}"
        t" t0={t0} h0={h0} w0={w0} t1={t1} h1={h1} w1={w1}"
    )


def main() raises:
    with DeviceContext() as ctx:
        test_tpool_patch_merger(ctx)
        # Both launch paths must support at least 65,536 pooled spatial rows.
        test_tpool_patch_merger(ctx, hidden_dim=7, t0=1, h0=256, w0=256)
        test_tpool_patch_merger(ctx, hidden_dim=32, t0=1, h0=256, w0=256)
        test_tpool_patch_merger(ctx, hidden_dim=32, t0=1, h0=138, w0=482)
        # Independent maxima produce a larger grid than either video's rows.
        test_tpool_patch_merger(
            ctx, hidden_dim=7, t0=1, h0=512, w0=2, t1=1, h1=2, w1=512
        )
        test_tpool_patch_merger(
            ctx, hidden_dim=32, t0=1, h0=512, w0=2, t1=1, h1=2, w1=512
        )
        # Raw temporal tokens exceed 65,535 while pooled spatial rows do not.
        test_tpool_patch_merger(ctx, hidden_dim=7, t0=2, h0=128, w0=256)
        # Exercise more than one channel tile in each launch path.
        test_tpool_patch_merger(ctx, hidden_dim=4097)
        test_tpool_patch_merger(ctx, hidden_dim=8192)
