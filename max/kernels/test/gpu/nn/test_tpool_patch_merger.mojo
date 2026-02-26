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
"""Test tpool_patch_merger kernel against Python reference behavior.

Python reference: tpool_path_merger.py
  - x: (total_tokens, D), grid_thws: (n_videos, 3) [T,H,W], merge (kH,kW)
  - Output: list of tensors, each (new_H*new_W, kH*kW, D) with temporal mean.
This test implements the same logic in Mojo (CPU reference) and compares
GPU kernel output to it.
"""

from math import ceildiv

from gpu.host import DeviceContext
from layout._coord import Coord, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.tpool_patch_merger import (
    tpool_patch_merger,
)
from random import rand, seed
from testing import assert_almost_equal


fn cpu_reference_one_video[
    dtype: DType,
](
    x: UnsafePointer[Scalar[dtype]],
    in_offset: Int,
    out_base: UnsafePointer[mut=True, Scalar[dtype]],
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
        var sp_idx = pat_idx // (kH * kW)
        var ker_idx = pat_idx % (kH * kW)
        var nh = sp_idx // new_W
        var nw = sp_idx % new_W
        var ph = ker_idx // kW
        var pw = ker_idx % kW
        var h_src = nh * kH + ph
        var w_src = nw * kW + pw
        var spatial_flat = h_src * w + w_src
        for d in range(D):
            var acc = Scalar[dtype](0)
            for ti in range(t):
                var row = in_offset + ti * (h * w) + spatial_flat
                acc += x[row * D + d]
            acc = acc / Scalar[dtype](t)
            out_base.store(pat_idx * D + d, acc)


fn test_tpool_patch_merger(ctx: DeviceContext) raises:
    """Compare GPU kernel output to CPU reference (same math as Python)."""
    comptime dtype = DType.bfloat16
    comptime kH = 2
    comptime kW = 2
    comptime D = 8

    # Two videos: (T=2, H=4, W=4) and (T=3, H=6, W=6)
    var t0: Int = 2
    var h0: Int = 4
    var w0: Int = 4
    var t1: Int = 3
    var h1: Int = 6
    var w1: Int = 6
    var max_h = max(h0, h1)
    var max_w = max(w0, w1)
    var len0 = t0 * h0 * w0
    var len1 = t1 * h1 * w1
    var total_in = len0 + len1
    var n_videos: Int = 2

    var out0_rows = (h0 // kH) * (w0 // kW) * kH * kW
    var out1_rows = (h1 // kH) * (w1 // kW) * kH * kW

    # Host buffers for input and grid_thws
    var x_host = ctx.enqueue_create_host_buffer[dtype](total_in * D)
    var bounds_host = ctx.enqueue_create_host_buffer[DType.int32](n_videos * 3)
    ctx.synchronize()

    seed(42)
    rand[dtype](x_host.unsafe_ptr(), total_in * D, min=0.0, max=1.0)

    bounds_host[0] = Int32(t0)
    bounds_host[1] = Int32(h0)
    bounds_host[2] = Int32(w0)
    bounds_host[3] = Int32(t1)
    bounds_host[4] = Int32(h1)
    bounds_host[5] = Int32(w1)

    # Device buffers
    var x_dev = ctx.enqueue_create_buffer[dtype](total_in * D)
    var out0_dev = ctx.enqueue_create_buffer[dtype](out0_rows * D)
    var out1_dev = ctx.enqueue_create_buffer[dtype](out1_rows * D)
    var bounds = ctx.enqueue_create_buffer[DType.int32](n_videos * 3)
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(bounds, bounds_host)
    ctx.enqueue_memset(out0_dev, 0)
    ctx.enqueue_memset(out1_dev, 0)
    ctx.synchronize()

    # CPU reference (same math as Python tpool_patch_merger): write to host buffers
    var ref0_host = ctx.enqueue_create_host_buffer[dtype](out0_rows * D)
    var ref1_host = ctx.enqueue_create_host_buffer[dtype](out1_rows * D)

    # Video 0: (T, H, W) = (2, 4, 4) → len0 = 32 tokens in x. in_offset = 0
    # Video 1: (T, H, W) = (3, 6, 6) → len1 = 108 tokens in x. in_offset = len0

    cpu_reference_one_video[dtype](
        x_host.unsafe_ptr(),
        0,
        ref0_host.unsafe_ptr(),
        t0,
        h0,
        w0,
        kH,
        kW,
        D,
    )
    cpu_reference_one_video[dtype](
        x_host.unsafe_ptr(),
        len0,
        ref1_host.unsafe_ptr(),
        t1,
        h1,
        w1,
        kH,
        kW,
        D,
    )

    # GPU kernel: out_ptrs device array
    var host_out_ptrs = ctx.enqueue_create_host_buffer[DType.uint64](n_videos)
    host_out_ptrs[0] = UInt64(Int(out0_dev.unsafe_ptr()))
    host_out_ptrs[1] = UInt64(Int(out1_dev.unsafe_ptr()))
    var out_ptrs_dev = ctx.enqueue_create_buffer[DType.uint64](n_videos)
    ctx.enqueue_copy(out_ptrs_dev, host_out_ptrs)
    ctx.synchronize()

    var max_pat = out0_rows if out0_rows > out1_rows else out1_rows
    var x_tile = TileTensor(
        x_dev.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        row_major(Coord(Idx(total_in), Idx(D))),
    )
    var out_ptrs_tensor = TileTensor(
        out_ptrs_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        row_major(Coord(Idx(n_videos))),
    )
    var bounds_tensor = TileTensor(
        bounds.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        row_major(Coord(Idx(n_videos), Idx(3))),
    )

    tpool_patch_merger(
        out_ptrs_tensor,
        x_tile,
        bounds_tensor,
        kH,
        kW,
        max_h,
        max_w,
        ctx,
    )

    ctx.synchronize()

    # Copy GPU outputs back
    var out0_host = ctx.enqueue_create_host_buffer[dtype](out0_rows * D)
    var out1_host = ctx.enqueue_create_host_buffer[dtype](out1_rows * D)
    ctx.enqueue_copy(out0_host, out0_dev)
    ctx.enqueue_copy(out1_host, out1_dev)
    ctx.synchronize()

    # Compare to CPU reference (Python-equivalent); relax for bfloat16 precision
    var atol = 1e-2
    for i in range(out0_rows * D):
        assert_almost_equal(out0_host[i], ref0_host[i], atol=atol)
    for i in range(out1_rows * D):
        assert_almost_equal(out1_host[i], ref1_host[i], atol=atol)

    print("test_tpool_patch_merger: PASSED")


def main():
    with DeviceContext() as ctx:
        test_tpool_patch_merger(ctx)
