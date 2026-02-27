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

from math import ceildiv, divmod
from sys.info import simd_width_of

from gpu import block_idx, thread_idx
from gpu.host import DeviceContext
from layout.coord import Coord, Idx, coord
from layout._layout import TensorLayout, row_major
from layout.tile_tensor import TileTensor
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, StaticTensorSpec


# ------------------------------------------------------------------------------
# GPU kernel
# ------------------------------------------------------------------------------


fn tpool_patch_merger_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    x_origin: ImmutOrigin,
    OutLayout: TensorLayout,
    out_origin: MutOrigin,
    GridThwLayout: TensorLayout,
    grid_thw_origin: ImmutOrigin,
    vec_width: Int,
    num_threads: Int,
](
    x_tile: TileTensor[dtype, XLayout, x_origin],
    out_tile: TileTensor[dtype, OutLayout, out_origin],
    grid_thws: TileTensor[DType.int32, GridThwLayout, grid_thw_origin],
    kH: Int,
    kW: Int,
    D: Int,
    n_vids: Int,
):
    """Temporal pooling patch merger kernel.

    Averages x across the temporal dimension for each video, rearranging
    spatially according to the (kH, kW) merge kernel.  Each video's output
    occupies H_i * W_i contiguous rows in the flat output tensor.

    Grid mapping:
        block_idx.z  = video index
        block_idx.y  = patch index within the video (max_pat upper bound)
        block_idx.x  = tile index along D
        thread_idx.x = lane within D tile

    Args:
        x_tile: Input tensor [n_tokens, D].
        out_tile: Contiguous output tensor [total_output_patches, D].
        grid_thws: Grid dimensions tensor [n_vids, 3] with (T, H, W) per video.
        kH: Merge kernel height.
        kW: Merge kernel width.
        D: Hidden dimension.
        n_vids: Number of videos.
    """
    comptime assert x_tile.flat_rank == 2, "x_tile must be rank 2"
    comptime assert out_tile.flat_rank == 2, "out_tile must be rank 2"
    comptime assert grid_thws.flat_rank == 2, "grid_thws must be rank 2"

    var vid = Int(block_idx.z)
    var pat_idx = Int(block_idx.y)
    var d_tile = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    if vid >= n_vids:
        return

    var t = Int(grid_thws[Coord(Idx(vid), Idx(0))])
    var h = Int(grid_thws[Coord(Idx(vid), Idx(1))])
    var w = Int(grid_thws[Coord(Idx(vid), Idx(2))])

    var new_H = h // kH
    var new_W = w // kW
    var n_patches = new_H * new_W
    var n_kernel = kH * kW
    var n_pat_total = n_patches * n_kernel

    if pat_idx >= n_pat_total:
        return

    # Scan grid_thws to compute input and output offsets for this video.
    var in_offset: Int = 0
    var out_offset: Int = 0
    for i in range(vid):
        var ti = Int(grid_thws[Coord(Idx(i), Idx(0))])
        var hi = Int(grid_thws[Coord(Idx(i), Idx(1))])
        var wi = Int(grid_thws[Coord(Idx(i), Idx(2))])
        in_offset += ti * hi * wi
        out_offset += hi * wi

    var sp_idx, ker_idx = divmod(pat_idx, n_kernel)
    var nh, nw = divmod(sp_idx, new_W)
    var ph, pw = divmod(ker_idx, kW)
    var h_src = nh * kH + ph
    var w_src = nw * kW + pw
    var spatial_flat = h_src * w + w_src

    comptime if vec_width == 1:
        var d = d_tile * num_threads + tid
        if d >= D:
            return
        var acc = Scalar[dtype](0)
        for t_i in range(t):
            var row = in_offset + t_i * (h * w) + spatial_flat
            acc += x_tile[Coord(Idx(row), Idx(d))]
        acc /= Scalar[dtype](t)
        out_tile.store(Coord(Idx(out_offset + pat_idx), Idx(d)), acc)
    else:
        var d_start = (d_tile * num_threads + tid) * vec_width
        if d_start >= D:
            return
        var acc = SIMD[dtype, vec_width](0)
        for t_i in range(t):
            var row = in_offset + t_i * (h * w) + spatial_flat
            acc += x_tile.load[width=vec_width](Coord(Idx(row), Idx(d_start)))
        acc /= Scalar[dtype](t)
        out_tile.store[width=vec_width](
            Coord(Idx(out_offset + pat_idx), Idx(d_start)), acc
        )


# ------------------------------------------------------------------------------
# Host launch (enqueue)
# ------------------------------------------------------------------------------


fn tpool_patch_merger[
    dtype: DType = DType.bfloat16,
](
    output: TileTensor[dtype, _, MutAnyOrigin],
    x: TileTensor[dtype, _, ImmutAnyOrigin],
    bounds: TileTensor[DType.int32, _, ImmutAnyOrigin],
    kH: Int,
    kW: Int,
    max_h: Int,
    max_w: Int,
    ctx: DeviceContext,
) raises:
    """Temporal pooling patch merger entry point.

    Args:
        output: Contiguous output tensor [total_output_patches, D].
        x: Input tensor [n_tokens, D].
        bounds: Grid dimensions tensor [n_vids, 3] with (T, H, W) per video.
        kH: Merge kernel height.
        kW: Merge kernel width.
        max_h: Maximum H across all videos (for grid sizing).
        max_w: Maximum W across all videos (for grid sizing).
        ctx: Device context.
    """
    var D = Int(x.dim[1]())
    var n_vids = Int(bounds.dim[0]())
    var max_pat = max_h // kH * max_w // kW * kH * kW

    comptime simd_width = simd_width_of[
        dtype, target = ctx.default_device_info.target()
    ]()
    comptime num_threads = 256

    if D % simd_width == 0:
        var grid_x = ceildiv(D, num_threads * simd_width)
        comptime kernel = tpool_patch_merger_kernel[
            dtype,
            x.LayoutType,
            ImmutOrigin(x.origin),
            output.LayoutType,
            output.origin,
            bounds.LayoutType,
            ImmutOrigin(bounds.origin),
            simd_width,
            num_threads,
        ]
        ctx.enqueue_function_experimental[kernel](
            x.as_immut(),
            output,
            bounds.as_immut(),
            kH,
            kW,
            D,
            n_vids,
            grid_dim=(grid_x, max_pat, n_vids),
            block_dim=(num_threads, 1, 1),
        )
    else:
        var grid_x = ceildiv(D, num_threads * 1)
        comptime kernel = tpool_patch_merger_kernel[
            dtype,
            x.LayoutType,
            ImmutOrigin(x.origin),
            output.LayoutType,
            output.origin,
            bounds.LayoutType,
            ImmutOrigin(bounds.origin),
            1,
            num_threads,
        ]
        ctx.enqueue_function_experimental[kernel](
            x.as_immut(),
            output,
            bounds.as_immut(),
            kH,
            kW,
            D,
            n_vids,
            grid_dim=(grid_x, max_pat, n_vids),
            block_dim=(num_threads, 1, 1),
        )
