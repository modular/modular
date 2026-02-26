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
from layout._coord import Coord, Idx, coord
from layout._layout import TensorLayout, row_major
from layout._tile_tensor import TileTensor
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, StaticTensorSpec


# ------------------------------------------------------------------------------
# GPU kernel
# ------------------------------------------------------------------------------


@always_inline
fn _reduce_t_and_store[
    dtype: DType,
](
    x_tile: TileTensor[dtype, _, MutExternalOrigin],
    out_base_tensor: TileTensor[dtype, _, MutAnyOrigin],
    in_offset: Int,
    out_offset: Int,
    t: Int,
    h: Int,
    w: Int,
    kH: Int,
    kW: Int,
    D: Int,
    pat_idx: Int,
    d: Int,
) where (x_tile.flat_rank == 2 and out_base_tensor.flat_rank == 1):
    var new_W = w // kW
    var n_kernel = kH * kW

    var sp_idx, ker_idx = divmod(pat_idx, n_kernel)
    var nh, nw = divmod(sp_idx, new_W)
    var ph, pw = divmod(ker_idx, kW)

    var h_src = nh * kH + ph
    var w_src = nw * kW + pw
    var spatial_flat = h_src * w + w_src

    var acc = Scalar[dtype](0)
    for t_i in range(t):
        var row = in_offset + t_i * (h * w) + spatial_flat
        acc += x_tile[Coord(Idx(row), Idx(d))]
    acc = acc / Scalar[dtype](t)
    var out_row = out_offset + pat_idx
    out_base_tensor.store(Coord(Idx(out_row * D + d)), acc)


fn tpool_patch_merger_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    OutLayout: TensorLayout,
    GridThwLayout: TensorLayout,
    vec_width: Int,
    num_threads: Int,
](
    x_tile: TileTensor[dtype, XLayout, MutExternalOrigin],
    out_ptrs: TileTensor[DType.uint64, OutLayout, MutAnyOrigin],
    grid_thws: TileTensor[DType.int32, GridThwLayout, MutAnyOrigin],
    kH: Int,
    kW: Int,
    D: Int,
    n_vids: Int,
):
    """
    Temporal pooling patch merger kernel.

    Args:
        x_tile: Input tensor [n_tokens, D].
        out_ptrs: Output tensor [n_vids] of [D] tensors, each video contains a pointer to a [D] tensor.
        grid_thws: Grid dimensions tensor [n_vids, 3].
        kH: Kernel height.
        kW: Kernel width.
        D: Input dimension.
        n_vids: Number of videos.
    """
    comptime assert x_tile.flat_rank == 2, "x_tile must be rank 2"
    comptime assert out_ptrs.flat_rank == 1, "out_ptrs must be rank 1"
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

    var in_offset: Int = 0
    for i in range(vid):
        var ti = Int(grid_thws[Coord(Idx(i), Idx(0))])
        var hi = Int(grid_thws[Coord(Idx(i), Idx(1))])
        var wi = Int(grid_thws[Coord(Idx(i), Idx(2))])
        in_offset += ti * hi * wi

    var addr = Int(out_ptrs[Coord(Idx(vid))])
    var out_base = UnsafePointer[mut=True, Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=addr
    )
    var out_base_tensor = TileTensor(
        out_base,
        row_major(Coord(Idx(D))),
    )

    comptime if vec_width == 1:
        var d = d_tile * num_threads + tid
        if d >= D:
            return
        _reduce_t_and_store[dtype](
            x_tile,
            out_base_tensor,
            in_offset,
            0,
            t,
            h,
            w,
            kH,
            kW,
            D,
            pat_idx,
            d,
        )
    else:
        var d_start = (d_tile * num_threads + tid) * vec_width
        if d_start >= D:
            return

        var sp_idx, ker_idx = divmod(pat_idx, n_kernel)
        var nh, nw = divmod(sp_idx, new_W)
        var ph, pw = divmod(ker_idx, kW)
        var h_src = nh * kH + ph
        var w_src = nw * kW + pw
        var spatial_flat = h_src * w + w_src

        var acc = SIMD[dtype, vec_width](0)
        for t_i in range(t):
            var row = in_offset + t_i * (h * w) + spatial_flat
            acc += x_tile.load[width=vec_width](Coord(Idx(row), Idx(d_start)))

        acc /= Scalar[dtype](t)
        var out_row = pat_idx
        out_base_tensor.store[width=vec_width](
            Coord(Idx(out_row * D + d_start)), acc
        )


# ------------------------------------------------------------------------------
# Host launch (enqueue)
# ------------------------------------------------------------------------------


fn tpool_patch_merger[
    dtype: DType = DType.bfloat16,
](
    output: TileTensor[
        DType.uint64, _, MutAnyOrigin
    ],  # [n_vids] of [D] tensors,
    x: TileTensor[dtype, _, MutExternalOrigin],  # [n_tokens, D],
    bounds: TileTensor[DType.int32, _, MutAnyOrigin],  # [n_vids, 3],
    kH: Int,
    kW: Int,
    max_h: Int,
    max_w: Int,
    ctx: DeviceContext,
) raises where (
    x.flat_rank == 2 and output.flat_rank == 1 and bounds.flat_rank == 2
):
    """
    Temporal pooling patch merger kernel entry point.

    Args:
        output: Output tensor [n_vids] of [D] tensors, each video contains a pointer to a [D] tensor.
        x: Input tensor [n_tokens, D].
        bounds: Bounds tensor [n_vids, 3].
        kH: Kernel height.
        kW: Kernel width.
        max_h: Maximum height.
        max_w: Maximum width.
        ctx: Device context.
    """
    var D = Int(x.dim[1]())
    var n_vids = Int(bounds.dim[0]())
    var n_tokens = Int(x.dim[0]())
    var max_pat = max_h // kH * max_w // kW * kH * kW

    comptime simd_width = simd_width_of[
        dtype, target = ctx.default_device_info.target()
    ]()
    comptime num_threads = 256

    if D % simd_width == 0:
        var grid_x = ceildiv(D, num_threads * simd_width)
        comptime kernel = tpool_patch_merger_kernel[
            dtype,
            type_of(x).LayoutType,
            type_of(output).LayoutType,
            type_of(bounds).LayoutType,
            simd_width,
            num_threads,
        ]
        ctx.enqueue_function[kernel, kernel](
            x,
            output,
            bounds,
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
            type_of(x).LayoutType,
            type_of(output).LayoutType,
            type_of(bounds).LayoutType,
            1,
            num_threads,
        ]
        ctx.enqueue_function[kernel, kernel](
            x,
            output,
            bounds,
            kH,
            kW,
            D,
            n_vids,
            grid_dim=(grid_x, max_pat, n_vids),
            block_dim=(num_threads, 1, 1),
        )
