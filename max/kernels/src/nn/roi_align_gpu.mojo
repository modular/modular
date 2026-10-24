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
"""Implements the GPU RoI Align kernel.

Each output element ``output[ri, ph, pw, c]`` is the pooled value of the
``roi_bin_grid_h x roi_bin_grid_w`` bilinearly-sampled points inside the
region's bin.  The kernel assigns one thread per output element: the thread
recomputes the region's bin geometry (cheap integer/float arithmetic on the
ROI row) and accumulates the four bilinear corner contributions.  This
mirrors the CPU ``roi_align_nhwc`` loop; the two agree within float32
rounding tolerance (the GPU compiler may reorder/contract the same
operations).
"""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceAttribute
from layout import TensorEngine, TensorLayout, TileTensor
from std.math import ceil, ceildiv
from std.utils.numerics import min_or_neg_inf


@__name(t"roi_align_{dtype}_{mode}")
def roi_align_kernel[
    InputLayoutType: TensorLayout,
    input_origin: ImmOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    RoisLayoutType: TensorLayout,
    rois_origin: ImmOrigin,
    dtype: DType,
    InputEngine: TensorEngine,
    OutputEngine: TensorEngine,
    RoisEngine: TensorEngine,
    aligned: Bool,
    mode: StaticString,
](
    input_tensor: TileTensor[
        dtype, InputLayoutType, input_origin, Engine=InputEngine
    ],
    output_tensor: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    rois_tensor: TileTensor[
        dtype, RoisLayoutType, rois_origin, Engine=RoisEngine
    ],
    pooled_height: Int32,
    pooled_width: Int32,
    spatial_scale: Float32,
    sampling_ratio: Float32,
    rows_per_block: Int32,
    total_rows: Int32,
):
    """One thread per output element ``[ri, ph, pw, c]``.

    The block's ``x`` dimension strides the channel axis (32 channels per
    pass); the ``y`` dimension strides the pooled ``(ph, pw)`` rows assigned
    to the block.  The grid spans the region axis, chunked by
    ``rows_per_block``; each block grid-strides its assigned rows and
    channels so the launch covers every output element.
    """
    comptime assert mode == "AVG" or mode == "MAX", "mode must be AVG or MAX"

    var _pooled_height = Int(pooled_height)
    var _pooled_width = Int(pooled_width)
    var _rows_per_block = Int(rows_per_block)
    var _total_rows = Int(total_rows)

    var channels = Int(input_tensor.dim[3]())
    var height = Int(input_tensor.dim[1]())
    var width = Int(input_tensor.dim[2]())

    var rows_per_iter = block_dim.y

    var start_row = block_idx.x * _rows_per_block
    var end_row = min(start_row + _rows_per_block, _total_rows)

    for row in range(start_row + thread_idx.y, end_row, rows_per_iter):
        var ri = row // (_pooled_height * _pooled_width)
        var rem = row % (_pooled_height * _pooled_width)
        var ph = rem // _pooled_width
        var pw = rem % _pooled_width

        # Region coordinates and batch index.
        var roi_batch_idx = Int(rois_tensor[ri, 0][0])
        var roi_start_w = Float32(rois_tensor[ri, 1][0]) * spatial_scale
        var roi_start_h = Float32(rois_tensor[ri, 2][0]) * spatial_scale
        var roi_end_w = Float32(rois_tensor[ri, 3][0]) * spatial_scale
        var roi_end_h = Float32(rois_tensor[ri, 4][0]) * spatial_scale
        comptime offset = Float32(0.5 if aligned else 0.0)
        roi_start_w -= offset
        roi_start_h -= offset
        roi_end_w -= offset
        roi_end_h -= offset

        var roi_height = roi_end_h - roi_start_h if aligned else max(
            roi_end_h - roi_start_h, 1.0
        )
        var roi_width = roi_end_w - roi_start_w if aligned else max(
            roi_end_w - roi_start_w, 1.0
        )

        var bin_size_h = roi_height / Float32(_pooled_height)
        var bin_size_w = roi_width / Float32(_pooled_width)

        var roi_bin_grid_h = Int(
            sampling_ratio if sampling_ratio > 0 else ceil(bin_size_h)
        )
        var roi_bin_grid_w = Int(
            sampling_ratio if sampling_ratio > 0 else ceil(bin_size_w)
        )
        var pool_elemn_num = max(roi_bin_grid_h * roi_bin_grid_w, 1)

        for c in range(thread_idx.x, channels, block_dim.x):
            var pool_val: Scalar[dtype]
            comptime if mode == "AVG":
                pool_val = 0
            else:
                pool_val = min_or_neg_inf[dtype]()

            for iy in range(roi_bin_grid_h):
                for ix in range(roi_bin_grid_w):
                    # Map (ph, pw, iy, ix) into the input plane.
                    var y = (
                        roi_start_h
                        + Float32(ph) * bin_size_h
                        + (Float32(iy) + Float32(0.5))
                        * bin_size_h
                        / Float32(roi_bin_grid_h)
                    )
                    var x = (
                        roi_start_w
                        + Float32(pw) * bin_size_w
                        + (Float32(ix) + Float32(0.5))
                        * bin_size_w
                        / Float32(roi_bin_grid_w)
                    )

                    if not (Float32(-1.0) <= y <= Float32(height)) or not (
                        Float32(-1.0) <= x <= Float32(width)
                    ):
                        continue

                    y = max(y, 0)
                    x = max(x, 0)

                    var y_low = min(Int(y), height - 1)
                    var x_low = min(Int(x), width - 1)
                    var y_high = min(y_low + 1, height - 1)
                    var x_high = min(x_low + 1, width - 1)

                    var ly = y - Float32(y_low)
                    var lx = x - Float32(x_low)
                    var hy = 1.0 - ly
                    var hx = 1.0 - lx

                    var w1 = hy * hx
                    var w2 = hy * lx
                    var w3 = ly * hx
                    var w4 = ly * lx

                    var v1 = rebind[Scalar[dtype]](
                        input_tensor.load((roi_batch_idx, y_low, x_low, c))
                    )
                    var v2 = rebind[Scalar[dtype]](
                        input_tensor.load((roi_batch_idx, y_low, x_high, c))
                    )
                    var v3 = rebind[Scalar[dtype]](
                        input_tensor.load((roi_batch_idx, y_high, x_low, c))
                    )
                    var v4 = rebind[Scalar[dtype]](
                        input_tensor.load((roi_batch_idx, y_high, x_high, c))
                    )

                    comptime if mode == "AVG":
                        pool_val = pool_val + (Scalar[dtype](w1) * v1)
                        pool_val = pool_val + (Scalar[dtype](w2) * v2)
                        pool_val = pool_val + (Scalar[dtype](w3) * v3)
                        pool_val = pool_val + (Scalar[dtype](w4) * v4)
                    else:
                        pool_val = max(
                            pool_val,
                            max(
                                max(
                                    Scalar[dtype](w1) * v1,
                                    Scalar[dtype](w2) * v2,
                                ),
                                max(
                                    Scalar[dtype](w3) * v3,
                                    Scalar[dtype](w4) * v4,
                                ),
                            ),
                        )

            comptime if mode == "AVG":
                pool_val = pool_val / Scalar[dtype](pool_elemn_num)
            output_tensor.store((ri, ph, pw, c), pool_val)


def _roi_align_impl[
    dtype: DType,
    aligned: Bool,
    mode: StaticString,
    max_threads: Int = 256,
    threads_per_row: Int = 32,
](
    input_tensor: TileTensor[mut=False, dtype, ...],
    output_tensor: TileTensor[mut=True, dtype, ...],
    rois_tensor: TileTensor[mut=False, dtype, ...],
    pooled_height: Int,
    pooled_width: Int,
    spatial_scale: Scalar,
    sampling_ratio: Scalar,
    ctx: DeviceContext,
) raises:
    """Launches :func:`roi_align_kernel` over the output's pooled grid."""
    if output_tensor.num_elements() == 0:
        return

    var total_rows = (
        Int(pooled_height) * Int(pooled_width) * Int(rois_tensor.dim(0))
    )

    comptime assert threads_per_row > 0 and max_threads % threads_per_row == 0

    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var linear_block_count: Int
    var rows_per_block: Int
    if sm_count > total_rows:
        linear_block_count = total_rows
        rows_per_block = 1
    else:
        linear_block_count = sm_count
        rows_per_block = ceildiv(total_rows, sm_count)

    comptime block_rows = max_threads // threads_per_row
    comptime kernel = roi_align_kernel[
        input_origin=ImmOrigin(input_tensor.origin),
        InputLayoutType=input_tensor.LayoutType,
        output_origin=output_tensor.origin,
        OutputLayoutType=output_tensor.LayoutType,
        rois_origin=ImmOrigin(rois_tensor.origin),
        RoisLayoutType=rois_tensor.LayoutType,
        dtype=dtype,
        InputEngine=type_of(input_tensor.as_imm()).Engine,
        OutputEngine=output_tensor.Engine,
        RoisEngine=type_of(rois_tensor.as_imm()).Engine,
        aligned=aligned,
        mode=mode,
    ]

    ctx.enqueue_function[kernel](
        input_tensor.as_imm(),
        output_tensor,
        rois_tensor.as_imm(),
        Int32(pooled_height),
        Int32(pooled_width),
        spatial_scale.cast[.float32](),
        sampling_ratio.cast[.float32](),
        Int32(rows_per_block),
        Int32(total_rows),
        grid_dim=(linear_block_count),
        block_dim=(threads_per_row, block_rows),
    )


def roi_align_gpu[
    dtype: DType,
    aligned: Bool,
    mode: StaticString = "AVG",
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    rois: TileTensor[mut=False, dtype, ...],
    output_height: Int,
    output_width: Int,
    in_spatial_scale: Scalar,
    in_sampling_ratio: Scalar,
    ctx: DeviceContext,
) raises:
    """GPU RoI Align for NHWC input.

    Args:
        output: Pre-allocated output tensor ``[M, pooled_h, pooled_w, C]``.
        input: Batched images ``[N, H, W, C]``.
        rois: ROI box coordinates ``[M, 5]``.
        output_height: Pooled output height.
        output_width: Pooled output width.
        in_spatial_scale: Scale factor remapping ROI coordinates to input.
        in_sampling_ratio: Sampling points per pooled bin.
        ctx: Device context for the participating GPU.
    """
    _roi_align_impl[dtype, aligned, mode](
        input,
        output,
        rois,
        output_height,
        output_width,
        in_spatial_scale,
        in_sampling_ratio,
        ctx,
    )
