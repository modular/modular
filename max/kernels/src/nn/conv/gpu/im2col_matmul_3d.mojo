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
"""Explicit im2col + `_matmul_gpu` dispatch for 3D convolution.

Mirrors the AMD RDNA 2D pattern in
`max/kernels/src/nn/conv/gpu/amd/rdna/dispatch.mojo`, extended to 3D
(NDHWC input, QRSCF or FCQRS filter) and bounded by an M-tile loop so
large video resolutions do not blow the scratch budget.

The generic `_matmul_gpu` auto-routes to SM100 UMMA on Blackwell for
bf16, so this path gives the native 3D conv access to tensor cores
without touching the TMA im2col descriptor layer.
"""

from std.math import ceildiv, divmod, gcd
from std.math.uutils import udivmod
from std.sys import align_of, simd_width_of, size_of
from max.gpu import block_dim, block_idx, global_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu import _matmul_gpu
from std.utils import IndexList
from nn.conv.conv_utils import (
    ConvSimdComputeFn,
    ConvSimdEpilogueFn,
    no_conv_simd_compute_fn,
    no_conv_simd_epilogue_fn,
)


# =========================================================================
# GPU kernels: im2col materialization and filter transpose
# =========================================================================


@__name(t"conv3d_im2col_ndhwc_{input_dtype}")
def _im2col_ndhwc_kernel[
    input_dtype: DType,
    filter_dtype: DType,
    output_dtype: DType,
    input_layout_type: TensorLayout,
    filter_layout_type: TensorLayout,
    output_layout_type: TensorLayout,
    input_engine: TensorEngine,
    filter_engine: TensorEngine,
    output_engine: TensorEngine,
    filter_is_fcrs: Bool,
](
    im2col_ptr: MutPointer[Scalar[input_dtype], MutAnyOrigin],
    input: TileTensor[
        input_dtype, input_layout_type, ImmutAnyOrigin, Engine=input_engine
    ],
    filter: TileTensor[
        filter_dtype, filter_layout_type, ImmutAnyOrigin, Engine=filter_engine
    ],
    output: TileTensor[
        output_dtype, output_layout_type, ImmutAnyOrigin, Engine=output_engine
    ],
    pad_d: Int32,
    pad_h: Int32,
    pad_w: Int32,
    stride_d: Int32,
    stride_h: Int32,
    stride_w: Int32,
    m_offset: Int32,
    m_count: Int32,
):
    """Write rows [m_offset, m_offset + m_count) of the 5D im2col matrix.

    M = batch * D_out * H_out * W_out (linearized output voxel).
    K = Q * R * S * C (filter-flattened reduction axis).
    Output is laid out [m_count, K] row-major starting at `output_ptr`.
    Dilation is assumed to be 1 (enforced at dispatch time).

    Block-per-row layout: one block handles a single output voxel (row of
    the im2col matrix); threads within the block cooperate on the K axis.
    This amortizes the (batch, d_out, h_out, w_out) decomposition and the
    input-base-offset math across all threads in the block instead of
    paying them per-element.
    """
    var _pad_d = Int(pad_d)
    var _pad_h = Int(pad_h)
    var _pad_w = Int(pad_w)
    var _stride_d = Int(stride_d)
    var _stride_h = Int(stride_h)
    var _stride_w = Int(stride_w)
    var _m_offset = Int(m_offset)
    var _m_count = Int(m_count)
    var local_m = block_idx.x
    if local_m >= _m_count:
        return

    var batch_size = Int(input.dim[0]())
    var D = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())

    # The filter and output are passed to this kernel to provide static shape
    # information.
    comptime assert filter.shape_known, "filter shape must be static"

    comptime Q = filter.static_shape[2 if filter_is_fcrs else 0]
    comptime R = filter.static_shape[3 if filter_is_fcrs else 1]
    comptime S = filter.static_shape[4 if filter_is_fcrs else 2]
    comptime C = filter.static_shape[1 if filter_is_fcrs else 3]
    comptime F = filter.static_shape[0 if filter_is_fcrs else 4]

    comptime simd_width = gcd(simd_width_of[input_dtype](), C)

    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())

    var K = Q * R * S * C
    var m = _m_offset + local_m

    # Per-block decomposition (amortized across block_dim threads).
    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out
    var batch, spatial = udivmod(m, DHW_out)
    var d_out, rem = udivmod(spatial, HW_out)
    var h_out, w_out = udivmod(rem, W_out)

    # Precompute the (stride, pad)-dependent input bases; within a block
    # only (q, r, s, c) change across threads.
    var d_in_base = d_out * _stride_d - _pad_d
    var h_in_base = h_out * _stride_h - _pad_h
    var w_in_base = w_out * _stride_w - _pad_w

    var RSC = R * S * C
    var SC = S * C

    var row_base = local_m * K
    var k = thread_idx.x * simd_width
    while k < K:
        var q, rsc = udivmod(k, RSC)
        var r, sc = udivmod(rsc, SC)
        var s, c = udivmod(sc, C)

        var d_in = d_in_base + q
        var h_in = h_in_base + r
        var w_in = w_in_base + s

        var val = SIMD[input_dtype, simd_width](0)
        if 0 <= d_in < D and 0 <= h_in < H and 0 <= w_in < W:
            val = input.load[
                width=simd_width, alignment=align_of[input_dtype]()
            ](Coord(batch, d_in, h_in, w_in, c))

        im2col_ptr.unsafe_store(row_base + k, val)
        k += block_dim.x * simd_width


@__name(t"conv3d_transpose_qrscf_to_nk_{dtype}")
def _transpose_qrscf_to_nk[
    dtype: DType,
    filter_layout_type: TensorLayout,
    filter_engine: TensorEngine,
](
    filter: TileTensor[
        dtype, filter_layout_type, ImmutAnyOrigin, Engine=filter_engine
    ],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """QRSCF [Q,R,S,C,F] -> [F, Q*R*S*C] row-major for matmul transpose_b."""
    comptime assert filter.shape_known, "filter shape must be static"

    comptime Q = filter.static_shape[0]
    comptime R = filter.static_shape[1]
    comptime S = filter.static_shape[2]
    comptime C = filter.static_shape[3]
    comptime F = filter.static_shape[4]

    comptime K = Q * R * S * C
    comptime total = F * K
    var tid = global_idx.x
    if tid >= total:
        return

    var f, k = udivmod(tid, K)

    comptime RSC = R * S * C
    comptime SC = S * C
    var q, rsc = udivmod(k, RSC)
    var r, sc = udivmod(rsc, SC)
    var s, c = udivmod(sc, C)

    dst_ptr.unsafe_store(tid, filter[q, r, s, c, f])


@__name(t"conv3d_transpose_fcqrs_to_nk_{dtype}")
def _transpose_fcqrs_to_nk[
    dtype: DType,
    filter_layout_type: TensorLayout,
    filter_engine: TensorEngine,
](
    filter: TileTensor[
        dtype, filter_layout_type, ImmutAnyOrigin, Engine=filter_engine
    ],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """FCQRS [F,C,Q,R,S] -> [F, Q*R*S*C] row-major for matmul transpose_b."""
    comptime assert filter.shape_known, "filter shape must be static"

    comptime F = filter.static_shape[0]
    comptime C = filter.static_shape[1]
    comptime Q = filter.static_shape[2]
    comptime R = filter.static_shape[3]
    comptime S = filter.static_shape[4]

    comptime K = Q * R * S * C
    comptime total = F * K
    var tid = global_idx.x
    if tid >= total:
        return

    var f, k = udivmod(tid, K)

    comptime RSC = R * S * C
    comptime SC = S * C
    var q, rsc = udivmod(k, RSC)
    var r, sc = udivmod(rsc, SC)
    var s, c = udivmod(sc, C)

    dst_ptr.unsafe_store(tid, filter[f, c, q, r, s])


# =========================================================================
# Public dispatch entry point
# =========================================================================

comptime _DEFAULT_M_TILE_BYTE_BUDGET = 256 * 1024 * 1024
comptime _MIN_M_TILE = 1024


def dispatch_im2col_matmul_conv3d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    EpilogueFnType: ConvSimdEpilogueFn = type_of(no_conv_simd_epilogue_fn),
    ComputeFnType: ConvSimdComputeFn = type_of(no_conv_simd_compute_fn),
    //,
    filter_is_fcrs: Bool = False,
    has_epilogue_fn: Bool = False,
    m_tile_byte_budget: Int = _DEFAULT_M_TILE_BYTE_BUDGET,
    has_compute_fn: Bool = False,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    symmetric_padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_conv_simd_epilogue_fn,
    compute_fn: ComputeFnType = no_conv_simd_compute_fn,
) raises -> Bool:
    """Try to dispatch a 3-D conv as explicit im2col + generic matmul.

    Returns True if the conv was handled; False if the caller should fall
    back to another implementation (naive Mojo kernel, cuDNN, etc.).

    Skips on: non-bf16 dtype, grouped conv, dilation != 1, kernel size
    1x1x1 (the vectorized naive kernel wins on tiny shapes), and K too
    small for the matmul fast path.

    Parameters:
        input_type: Element type of the input tensor; must be
            `DType.bfloat16`.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        filter_is_fcrs: True if the filter is in `FCQRS` layout, False if
            in `QRSCF` layout (inferred, defaults to False).
        has_epilogue_fn: True when `epilogue_fn` stores the output instead
            of the matmul (defaults to False).
        m_tile_byte_budget: Upper bound on bytes allocated for one M-tile
            of the im2col matrix, controlling scratch memory usage
            (inferred, defaults to 256 MiB).
        has_compute_fn: True when the matmul stores `compute_fn` of each
            output value (defaults to False).

    Args:
        input: Input tensor in `NDHWC` layout with shape
            `[batch, D, H, W, C]`.
        filter: Filter tensor with static shape, either `FCQRS` or
            `QRSCF` depending on `filter_is_fcrs`.
        output: Output tensor in `NDHWC` layout with shape
            `[batch, D_out, H_out, W_out, F]`.
        stride: Spatial strides for the depth, height, and width axes.
        dilation: Spatial dilation for the depth, height, and width
            axes; must be 1 for each.
        symmetric_padding: Symmetric padding for the depth, height, and
            width axes.
        num_groups: Number of convolution groups; must be 1.
        ctx: Device context for enqueueing GPU kernels and allocating
            scratch buffers.
        epilogue_fn: Elementwise epilogue applied to each output element
            with 5-D coordinates; used only when `has_epilogue_fn` is true.
        compute_fn: Elementwise transform of each output element with 5-D
            coordinates; used only when `has_compute_fn` is true.

    Returns:
        True if the conv was handled, False if the caller should fall
        back to another implementation.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"

    comptime if input_type != .bfloat16:
        return False
    comptime if not filter.shape_known:
        return False

    if num_groups != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1:
        return False

    var batch = Int(input.dim[0]())

    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())

    comptime Q = filter.static_shape[2 if filter_is_fcrs else 0]
    comptime R = filter.static_shape[3 if filter_is_fcrs else 1]
    comptime S = filter.static_shape[4 if filter_is_fcrs else 2]
    comptime C = filter.static_shape[1 if filter_is_fcrs else 3]
    comptime F = filter.static_shape[0 if filter_is_fcrs else 4]

    # Bail on 1x1x1: the vectorized naive kernel already beats cuDNN there,
    # and K is too small to amortize the matmul launch overhead.
    if Q == 1 and R == 1 and S == 1:
        return False

    var full_M = batch * D_out * H_out * W_out
    comptime K = Q * R * S * C
    comptime N = F

    # Minimum sane K. _matmul_gpu's SM100 path handles small K fine, but
    # K < 16 is below MMA_K even for bf16 and not worth the scratch.
    if K < 16:
        return False

    # --- Transpose filter to [N, K] once, before the M-tile loop. ---
    var filter_size = filter.num_elements()
    var filter_nk_buf = ctx.enqueue_create_buffer[filter_type](filter_size)

    comptime transpose_block = 256
    var transpose_grid = ceildiv(filter_size, transpose_block)

    comptime if filter_is_fcrs:
        ctx.enqueue_function[
            _transpose_fcqrs_to_nk[
                filter_type, filter.LayoutType, filter.Engine
            ]
        ](
            filter.as_imm(),
            filter_nk_buf,
            grid_dim=transpose_grid,
            block_dim=transpose_block,
        )
    else:
        ctx.enqueue_function[
            _transpose_qrscf_to_nk[
                filter_type, filter.LayoutType, filter.Engine
            ]
        ](
            filter.as_imm(),
            filter_nk_buf,
            grid_dim=transpose_grid,
            block_dim=transpose_block,
        )

    # --- Decide M-tile size so scratch stays bounded. ---
    # First pick a cap from the byte budget. Then equalize: if multiple
    # tiles are needed, prefer ceildiv(full_M, num_tiles) over the cap to
    # avoid a tiny ragged tail tile that under-uses the matmul.
    var bytes_per_row = K * size_of[input_type]()
    var m_tile_by_budget = m_tile_byte_budget // bytes_per_row
    var m_tile_cap = max(m_tile_by_budget, _MIN_M_TILE)
    var m_tile: Int
    if full_M <= m_tile_cap:
        m_tile = full_M
    else:
        var num_tiles = ceildiv(full_M, m_tile_cap)
        m_tile = ceildiv(full_M, num_tiles)

    var im2col_buf = ctx.enqueue_create_buffer[input_type](m_tile * K)

    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out

    # --- M-tile loop. ---
    var _m_offset = 0
    while _m_offset < full_M:
        var remaining = full_M - _m_offset
        var _m_count = min(m_tile, remaining)

        # Block-per-row: one block per output voxel, threads cooperate on K.
        comptime im2col_block = 256
        comptime im2col_kernel = _im2col_ndhwc_kernel[
            input_type,
            filter_type,
            output_type,
            input.LayoutType,
            filter.LayoutType,
            output.LayoutType,
            input.Engine,
            filter.Engine,
            output.Engine,
            filter_is_fcrs,
        ]
        ctx.enqueue_function[im2col_kernel](
            im2col_buf,
            input.as_imm(),
            filter.as_imm(),
            output.as_imm(),
            Int32(symmetric_padding[0]),
            Int32(symmetric_padding[1]),
            Int32(symmetric_padding[2]),
            Int32(stride[0]),
            Int32(stride[1]),
            Int32(stride[2]),
            Int32(_m_offset),
            Int32(_m_count),
            grid_dim=_m_count,
            block_dim=im2col_block,
        )

        var a_tt = TileTensor(im2col_buf, row_major(_m_count, K))
        var b_tt = TileTensor(filter_nk_buf, row_major(N, K))
        # Output is NDHWC = [batch, D_out, H_out, W_out, C_out]; rows in the
        # flattened [M, N] layout are contiguous, so we advance by _m_offset * N.
        var c_engine = output._offset_storage(_m_offset * N)
        var c_tt = TileTensor(c_engine, row_major(_m_count, N))

        comptime if has_epilogue_fn:

            @inline(.always)
            def _gemm_epilogue[
                _dtype: DType,
                _width: SIMDLength,
                *,
                alignment: Int,
            ](coords_2d: Coord, val: SIMD[_dtype, _width]) {
                var DHW_out,
                var HW_out,
                var W_out,
                var _m_offset,
                var epilogue_fn,
            }:
                comptime assert (
                    coords_2d.rank == 2
                ), "the matmul epilogue contract is rank 2"
                var full_m = _m_offset + Int(coords_2d[0])
                var n_idx = Int(coords_2d[1])
                var batch_idx, sp = divmod(full_m, DHW_out)
                var d_idx, rem2 = divmod(sp, HW_out)
                var h_idx, w_idx = divmod(rem2, W_out)
                epilogue_fn(
                    (batch_idx, d_idx, h_idx, w_idx, n_idx),
                    rebind[SIMD[output_type, _width]](val),
                )

            _matmul_gpu[use_tensor_core=True, transpose_b=True](
                c_tt, a_tt.as_imm(), b_tt.as_imm(), _gemm_epilogue, ctx
            )
        elif has_compute_fn:

            @inline(.always)
            def _gemm_compute[
                _dtype: DType,
                _width: SIMDLength,
                *,
                alignment: Int,
            ](coords_2d: Coord, val: SIMD[_dtype, _width]) {
                var DHW_out,
                var HW_out,
                var W_out,
                var _m_offset,
                var compute_fn,
            } -> SIMD[_dtype, _width]:
                comptime assert (
                    coords_2d.rank == 2
                ), "the matmul epilogue contract is rank 2"
                var full_m = _m_offset + Int(coords_2d[0])
                var n_idx = Int(coords_2d[1])
                var batch_idx, sp = divmod(full_m, DHW_out)
                var d_idx, rem2 = divmod(sp, HW_out)
                var h_idx, w_idx = divmod(rem2, W_out)
                return rebind[SIMD[_dtype, _width]](
                    compute_fn(
                        (batch_idx, d_idx, h_idx, w_idx, n_idx),
                        rebind[SIMD[output_type, _width]](val),
                    )
                )

            _matmul_gpu[use_tensor_core=True, transpose_b=True](
                c_tt, a_tt.as_imm(), b_tt.as_imm(), _gemm_compute, ctx
            )
        else:
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
            ](c_tt, a_tt.as_imm(), b_tt.as_imm(), ctx)

        _m_offset += _m_count

    return True
