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
"""Pieces the shape-gated SM100 conv2d fprop kernels share.

Target family: Blackwell sm_100a. `conv2d_f96_fprop`, `conv2d_f192_fprop` and
`conv2d_k1x5_fprop` each have their own pipeline, but they read the same
operands, build the same TMA descriptors, and move an accumulator from tensor
memory to a swizzled shared-memory tile the same way. That common part lives
here so a fix lands in one place.
"""

from std.collections import Array
from std.memory import bitcast
from std.sys import size_of

from layout import TileTensor, coord
from layout.swizzle import make_swizzle
from layout.tma_async import TMATensorTileIm2col, create_tensor_tile_im2col
from max.gpu import lane_id
from max.gpu.compute.arch.tcgen05 import tcgen05_ld, tcgen05_load_wait
from max.gpu.compute.mma import st_matrix
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from structured_kernels.tile_types import (
    TmaOpType,
    create_tma_tile,
    static_row_major,
)

from .conv_config import Conv2dProblemShape


def dedicated_problem_shape[
    filter_is_fcrs: Bool
](
    input: TileTensor[...],
    filter: TileTensor[...],
    output: TileTensor[...],
    pad_h: Int,
    pad_w: Int,
) -> Conv2dProblemShape:
    """The geometry of a stride 1, dilation 1, single group convolution.

    Parameters:
        filter_is_fcrs: Whether `filter` is FCRS `[F, C, R, S]` rather than
            RSCF `[R, S, C, F]`.

    Args:
        input: Input activation, NHWC.
        filter: Filter weights.
        output: Output, NHWC.
        pad_h: Symmetric padding along the height.
        pad_w: Symmetric padding along the width.

    Returns:
        The problem shape the kernels' `*_supported` predicates take.
    """
    var filter_h: Int
    var filter_w: Int
    comptime if filter_is_fcrs:
        filter_h = Int(filter.dim[2]())
        filter_w = Int(filter.dim[3]())
    else:
        filter_h = Int(filter.dim[0]())
        filter_w = Int(filter.dim[1]())
    return Conv2dProblemShape(
        batch=Int(input.dim[0]()),
        in_height=Int(input.dim[1]()),
        in_width=Int(input.dim[2]()),
        in_channels=Int(input.dim[3]()),
        out_channels=Int(output.dim[3]()),
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )


@always_inline
def assert_dedicated_operands[
    has_epilogue_fn: Bool, has_compute_fn: Bool, has_residual: Bool
](
    output: TileTensor[...],
    activation: TileTensor[...],
    filter: TileTensor[...],
):
    """Checks the epilogue flags and operand layouts the kernels accept.

    The caller asserts that the three tensors share its element type; the
    compiler needs that evidence where it converts the tensors' types.

    Parameters:
        has_epilogue_fn: A storing epilogue is fused.
        has_compute_fn: A transforming epilogue is fused.
        has_residual: A residual is fused.

    Args:
        output: Output tensor, NHWC.
        activation: Input activation, NHWC.
        filter: Filter, KRSC or RSCF.
    """
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"
    comptime assert not (
        has_residual and has_compute_fn
    ), "the residual kernel has no compute epilogue"
    comptime assert (
        output.flat_rank == 4
        and activation.flat_rank == 4
        and filter.flat_rank == 4
    ), "activation, filter and output are rank 4 (NHWC, KRSC, NHWC)"
    comptime assert (
        output.is_row_major and activation.is_row_major and filter.is_row_major
    ), (
        "the TMA descriptors address dense row-major tensors; a strided view"
        " would read the wrong elements"
    )


def create_act_im2col_tma[
    act_type: DType,
    bm: Int,
    bk: Int,
    pix: Int,
    swizzle: TensorMapSwizzle,
](
    ctx: DeviceContext,
    activation: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    problem: Conv2dProblemShape,
) raises -> TMATensorTileIm2col[act_type, coord[bm, bk], coord[pix, bk]]:
    """The im2col descriptor for an NHWC activation.

    Parameters:
        act_type: Element type of the activation.
        bm: Output pixels of one tile.
        bk: Channels of one K tile.
        pix: Output pixels one transaction loads.
        swizzle: Swizzle mode of the shared-memory tile.

    Args:
        ctx: Device context for the descriptor.
        activation: Input activation, NHWC.
        problem: The convolution geometry.

    Returns:
        A descriptor whose box is `[pix, bk]`.
    """
    return create_tensor_tile_im2col[
        act_type,
        coord[bm, bk],
        swizzle_mode=swizzle,
        __tile_shape=coord[bm, bk],
        __desc_shape=coord[pix, bk],
    ](
        ctx,
        activation,
        -problem.pad_h,
        -problem.pad_w,
        problem.pad_h - (problem.filter_h - 1),
        problem.pad_w - (problem.filter_w - 1),
        problem.out_height(),
        problem.out_width(),
        problem.filter_h,
        problem.filter_w,
    )


def create_box_tma[
    rows: Int, cols: Int, swizzle: TensorMapSwizzle
](ctx: DeviceContext, tensor: TileTensor[...]) raises -> TmaOpType[
    tensor.dtype, static_row_major[rows, cols], static_row_major[rows, cols]
]:
    """A descriptor over a row-major 2-D view, with a `[rows, cols]` box.

    Parameters:
        rows: Box rows.
        cols: Box columns.
        swizzle: Swizzle mode of the shared-memory box.

    Args:
        ctx: Device context for the descriptor.
        tensor: The 2-D view.

    Returns:
        The TMA descriptor.
    """
    return create_tma_tile[
        static_row_major[rows, cols],
        static_row_major[rows, cols],
        coord[rows, cols],
        swizzle_mode=swizzle,
    ](ctx, tensor)


@always_inline
def stage_accumulator[
    dtype: DType, cols: Int, rows: Int, swizzle_mode: TensorMapSwizzle
](
    taddr: UInt32,
    dst: MutPointer[Scalar[dtype], _, address_space=.SHARED],
    warp: Int,
):
    """Moves `cols` accumulator columns of one warp from tensor memory to a
    swizzled shared-memory tile, converted to `dtype`.

    A warp holds `rows` live accumulator rows: 32 for a 128-row tile, 16 for
    a 64-row one, whose second set of 16 lanes is empty. The tile is written
    with `stmatrix` in `swizzle_mode`, which must be the swizzle of the TMA
    descriptor that stores the tile, so the caller passes the same constant.

    Parameters:
        dtype: Output element type.
        cols: Accumulator columns, 32 or 64.
        rows: Live rows per warp, 16 or 32.
        swizzle_mode: Swizzle of the shared-memory tile; one row of `cols`
            elements spans exactly its width.

    Args:
        taddr: Tensor-memory address of the first column.
        dst: The shared-memory tile.
        warp: Warp index within the epilogue warp group.
    """
    comptime assert cols == 32 or cols == 64, "whole 32 or 64 column pieces"
    comptime assert rows == 16 or rows == 32, "16 or 32 live rows per warp"
    comptime assert (
        swizzle_mode.bytes() == cols * size_of[dtype]()
    ), "a tile row must span the swizzle width"
    comptime rep = cols // 8
    var upper = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=rep,
        dtype=.float32,
        pack=False,
    ](taddr)
    comptime if rows == 32:
        var lower = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat=rep,
            dtype=.float32,
            pack=False,
        ](taddr + UInt32(16 << 16))
        tcgen05_load_wait()
        var base = dst.unsafe_offset(warp * 32 * cols)
        _stsm[dtype, cols, swizzle_mode=swizzle_mode](upper, base)
        _stsm[dtype, cols, swizzle_mode=swizzle_mode](
            lower,
            base.unsafe_offset(16 * cols),
        )
    else:
        tcgen05_load_wait()
        _stsm[dtype, cols, swizzle_mode=swizzle_mode](
            upper,
            dst.unsafe_offset(warp * 16 * cols),
        )


@always_inline
def _stsm[
    dtype: DType, cols: Int, n: Int, swizzle_mode: TensorMapSwizzle
](
    vec: Array[Scalar[.float32], n],
    dst: MutPointer[Scalar[dtype], _, address_space=.SHARED],
):
    comptime swizzle = make_swizzle[dtype, swizzle_mode]()
    comptime row = 32 // size_of[dtype]()
    comptime per_lane = 16 // size_of[dtype]()
    var lane = Int(lane_id())
    var lane_off = (lane & 15) * cols + (lane >> 4) * 8
    comptime for i in range(cols // row):
        var off = swizzle(lane_off + i * row)
        var v = SIMD[dtype, per_lane]()
        comptime for k in range(per_lane // 2):
            var pair = SIMD[.float32, 2](
                vec[i * per_lane + 2 * k], vec[i * per_lane + 2 * k + 1]
            )
            var c = pair.cast[dtype]()
            v[2 * k] = c[0]
            v[2 * k + 1] = c[1]
        st_matrix[simd_width=4](dst.unsafe_offset(off), bitcast[.float32, 4](v))
