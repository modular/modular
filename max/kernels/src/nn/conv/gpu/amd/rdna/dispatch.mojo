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
"""RDNA dispatch for 2-D convolution.

Provides two paths for 2-D convolution on RDNA 3+:

1. **Implicit GEMM** (preferred): Fuses im2col into the WMMA matmul kernel's
   A-tile loader, eliminating the large intermediate buffer. Requires
   C_in % BLOCK_K == 0 for vectorized im2col loads.

2. **Explicit im2col + matmul** (fallback): Materializes the im2col buffer
   then calls the standard RDNA matmul. Used when C_in alignment requirements
   aren't met.
"""

from std.math import ceildiv
from max.gpu import global_idx, WARP_SIZE
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.utils.index import IndexList
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from nn.conv.conv_utils import (
    ConvSimdComputeFn,
    ConvSimdEpilogueFn,
    no_conv_simd_compute_fn,
    no_conv_simd_epilogue_fn,
)
from .conv2d_kernel import conv2d_kernel_rdna


# =========================================================================
# GPU kernels for explicit im2col fallback path
# =========================================================================


@__name(t"rdna_im2col_nhwc_{dtype}")
def _im2col_nhwc_kernel[
    dtype: DType,
](
    output_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    batch_size: Int32,
    H: Int32,
    W: Int32,
    C: Int32,
    R: Int32,
    S: Int32,
    H_out: Int32,
    W_out: Int32,
    pad_h: Int32,
    pad_w: Int32,
):
    """Transform NHWC input to im2col [M, K] matrix on GPU (stride=1, dilation=1).
    """
    var _batch_size = Int(batch_size)
    var _H = Int(H)
    var _W = Int(W)
    var _C = Int(C)
    var _R = Int(R)
    var _S = Int(S)
    var _H_out = Int(H_out)
    var _W_out = Int(W_out)
    var _pad_h = Int(pad_h)
    var _pad_w = Int(pad_w)
    var K = _R * _S * _C
    var HW_out = _H_out * _W_out
    var total = _batch_size * HW_out * K

    var tid = global_idx.x
    if tid >= total:
        return

    var m, k_idx = divmod(tid, K)

    var batch, spatial = divmod(m, HW_out)
    var h_out, w_out = divmod(spatial, _W_out)

    var SC = _S * _C
    var r, sc = divmod(k_idx, SC)
    var s, c = divmod(sc, _C)

    var h_in = h_out - _pad_h + r
    var w_in = w_out - _pad_w + s

    var val = Scalar[dtype](0)
    if h_in >= 0 and h_in < _H and w_in >= 0 and w_in < _W:
        val = input_ptr[batch * _H * _W * _C + h_in * _W * _C + w_in * _C + c]

    output_ptr.unsafe_store(tid, val)


@__name(t"rdna_transpose_rscf_to_nk_{dtype}")
def _transpose_rscf_to_nk[
    dtype: DType,
](
    src_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    R: Int32,
    S: Int32,
    C: Int32,
    F: Int32,
):
    """GPU kernel: transpose filter RSCF [K,N] -> [N,K] for transpose_b matmul.
    """
    var _R = Int(R)
    var _S = Int(S)
    var _C = Int(C)
    var _F = Int(F)
    var K = _R * _S * _C
    var total = K * _F
    var tid = global_idx.x
    if tid >= total:
        return
    var f, k = divmod(tid, K)
    dst_ptr.unsafe_store(tid, src_ptr.unsafe_load(k * _F + f))


@__name(t"rdna_transpose_fcrs_to_nk_{dtype}")
def _transpose_fcrs_to_nk[
    dtype: DType,
](
    src_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    F: Int32,
    C: Int32,
    R: Int32,
    S: Int32,
):
    """GPU kernel: transpose filter FCRS -> [N,K] for transpose_b matmul."""
    var _F = Int(F)
    var _C = Int(C)
    var _R = Int(R)
    var _S = Int(S)
    var K = _R * _S * _C
    var total = K * _F
    var tid = global_idx.x
    if tid >= total:
        return
    var f, k = divmod(tid, K)
    var r, sc = divmod(k, _S * _C)
    var s, c = divmod(sc, _C)
    dst_ptr.unsafe_store(
        tid, src_ptr.unsafe_load(f * _C * _R * _S + c * _R * _S + r * _S + s)
    )


# =========================================================================
# RDNA Conv2D dispatch
# =========================================================================


def dispatch_rdna_conv2d[
    EpilogueFnType: ConvSimdEpilogueFn = type_of(no_conv_simd_epilogue_fn),
    ComputeFnType: ConvSimdComputeFn = type_of(no_conv_simd_compute_fn),
    //,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    has_epilogue_fn: Bool = False,
    has_residual: Bool = False,
    has_compute_fn: Bool = False,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    symmetric_padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
    source_ptr: MutPointer[Scalar[output_type], MutAnyOrigin] = MutPointer[
        Scalar[output_type], MutAnyOrigin
    ].unsafe_dangling(),
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_conv_simd_epilogue_fn,
    compute_fn: ComputeFnType = no_conv_simd_compute_fn,
) raises -> Bool:
    """Try to dispatch Conv2D on RDNA via implicit GEMM (im2col fused into WMMA).

    Returns True if the convolution was handled, False if the caller should
    fall back to another implementation (e.g. MIOpen).

    Uses the implicit GEMM kernel when C_in is aligned to BLOCK_K (covers all
    FLUX VAE shapes), falling back to explicit im2col + matmul otherwise.

    When `has_residual=True`, folds `output = conv + beta * source` into the
    conv epilogue (the RDNA implicit-GEMM/im2col kernels have no native
    residual path). `source_ptr` is NHWC-contiguous, same shape as output:
    e.g. ResNet skip connections that the graph compiler fuses into the conv.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        input_type: `DType` of the input tensor elements.
        filter_type: `DType` of the filter tensor elements.
        output_type: `DType` of the output tensor elements.
        filter_is_fcrs: True if `filter` is laid out as FCRS, False for RSCF.
        has_epilogue_fn: True when `epilogue_fn` stores the output instead
            of the GEMM (defaults to `False`).
        has_residual: True to fold a scaled residual `beta * source_ptr` into
            the conv epilogue (defaults to `False`).
        has_compute_fn: True when the GEMM stores `compute_fn` of each
            post-residual output value (defaults to `False`).

    Args:
        input: Rank-4 NHWC input tensor.
        filter: Rank-4 filter tensor in FCRS or RSCF layout per
            `filter_is_fcrs`.
        output: Rank-4 NHWC output tensor written by the convolution.
        stride: Spatial stride `[stride_h, stride_w]`; only `[1, 1]` is
            supported.
        dilation: Spatial dilation `[dilation_h, dilation_w]`; only
            `[1, 1]` is supported.
        symmetric_padding: Symmetric spatial padding `[pad_h, pad_w]`
            applied to the input.
        num_groups: Number of convolution groups; only `1` is supported.
        ctx: `DeviceContext` used to enqueue kernels and synchronize.
        source_ptr: NHWC-contiguous residual source with the same shape as
            `output`; read only when `has_residual=True` (defaults to a
            dangling pointer).
        beta: Scale factor applied to the residual source when
            `has_residual=True` (defaults to `0.0`).
        epilogue_fn: Elementwise epilogue applied to each output element;
            used only when `has_epilogue_fn` is true.
        compute_fn: Elementwise transform of each post-residual output
            element; used only when `has_compute_fn` is true.
    """
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"

    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    comptime if input_type in (DType.bfloat16, DType.float16):
        # Check runtime constraints
        if (
            stride[0] != 1
            or stride[1] != 1
            or dilation[0] != 1
            or dilation[1] != 1
            or num_groups != 1
        ):
            return False

        # Extract dimensions
        var batch = Int(input.dim[0]())
        var in_h = Int(input.dim[1]())
        var in_w = Int(input.dim[2]())
        var in_c = Int(input.dim[3]())
        var out_h = Int(output.dim[1]())
        var out_w = Int(output.dim[2]())
        var out_c = Int(output.dim[3]())

        var fh: Int
        var fw: Int
        comptime if filter_is_fcrs:
            fh = Int(filter.dim[2]())
            fw = Int(filter.dim[3]())
        else:
            fh = Int(filter.dim[0]())
            fw = Int(filter.dim[1]())

        var M = batch * out_h * out_w
        var K = fh * fw * in_c
        var N = out_c

        if K % 16 != 0:
            return False

        # --- Transpose filter to [N, K] for transpose_b=True ---
        var filter_size = filter.num_elements()
        var filter_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
        var filter_nk_ptr = filter_buf.unsafe_ptr()

        comptime transpose_block = 256
        var transpose_grid = ceildiv(filter_size, transpose_block)

        comptime if filter_is_fcrs:
            ctx.enqueue_function[_transpose_fcrs_to_nk[filter_type]](
                filter.ptr,
                filter_nk_ptr,
                Int32(filter.dim[0]()),
                Int32(filter.dim[1]()),
                Int32(filter.dim[2]()),
                Int32(filter.dim[3]()),
                grid_dim=transpose_grid,
                block_dim=transpose_block,
            )
        else:
            ctx.enqueue_function[_transpose_rscf_to_nk[filter_type]](
                filter.ptr,
                filter_nk_ptr,
                Int32(filter.dim[0]()),
                Int32(filter.dim[1]()),
                Int32(filter.dim[2]()),
                Int32(filter.dim[3]()),
                grid_dim=transpose_grid,
                block_dim=transpose_block,
            )

        # Choose BLOCK_K based on K alignment
        comptime BLOCK_K = 32

        # --- Helper to launch the implicit GEMM kernel ---
        @inline(.always)
        def _launch_implicit_gemm[
            EpilogueFnType: ElementwiseEpilogueFn,
            ComputeFnType: ElementwiseComputeFn,
            //,
            has_epilogue_fn: Bool,
            has_compute_fn: Bool,
        ](epilogue_fn: EpilogueFnType, compute_fn: ComputeFnType) raises {imm}:
            comptime NUM_WARPS = 16  # 8x2 warp grid
            comptime BLOCK_M = 128
            comptime BLOCK_N = 128

            var filter_nk_tt = TileTensor(filter_nk_ptr, row_major(N, K))
            var out_tt = output.reshape(row_major(M, N))

            comptime conv_kernel = conv2d_kernel_rdna[
                output_type,
                input_type,
                filter_type,
                type_of(out_tt).LayoutType,
                type_of(filter_nk_tt).LayoutType,
                EpilogueFnType=EpilogueFnType,
                ComputeFnType=ComputeFnType,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                BLOCK_K=BLOCK_K,
                out_engine=type_of(out_tt).Engine,
                filter_nk_engine=type_of(filter_nk_tt).Engine,
            ]

            ctx.enqueue_function[conv_kernel](
                out_tt,
                input.ptr,
                filter_nk_tt,
                Int32(M),
                Int32(N),
                Int32(K),
                Int32(out_h * out_w),
                Int32(out_w),
                Int32(in_h),
                Int32(in_w),
                Int32(in_c),
                Int32(fh),
                Int32(fw),
                Int32(symmetric_padding[0]),
                Int32(symmetric_padding[1]),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=(ceildiv(N, BLOCK_N), ceildiv(M, BLOCK_M)),
                block_dim=(NUM_WARPS * WARP_SIZE,),
            )

        # --- Helper to launch explicit im2col + matmul fallback ---
        @inline(.always)
        def _launch_explicit_im2col[
            EpilogueFnType: ElementwiseEpilogueFn,
            ComputeFnType: ElementwiseComputeFn,
            //,
            has_epilogue_fn: Bool,
            has_compute_fn: Bool,
        ](epilogue_fn: EpilogueFnType, compute_fn: ComputeFnType) raises {imm}:
            var im2col_size = M * K
            var im2col_buf = ctx.enqueue_create_buffer[input_type](im2col_size)
            var im2col_ptr = im2col_buf.unsafe_ptr()

            comptime im2col_block = 256
            var im2col_grid = ceildiv(im2col_size, im2col_block)
            ctx.enqueue_function[_im2col_nhwc_kernel[input_type]](
                im2col_ptr,
                input.ptr,
                Int32(batch),
                Int32(in_h),
                Int32(in_w),
                Int32(in_c),
                Int32(fh),
                Int32(fw),
                Int32(out_h),
                Int32(out_w),
                Int32(symmetric_padding[0]),
                Int32(symmetric_padding[1]),
                grid_dim=(im2col_grid,),
                block_dim=(im2col_block,),
            )

            var a_tt = TileTensor(im2col_ptr, row_major(M, K))
            var b_tt = TileTensor(filter_nk_ptr, row_major(N, K))
            var c_tt = output.reshape(row_major(M, N))

            comptime if has_epilogue_fn:
                _matmul_gpu[use_tensor_core=True, transpose_b=True](
                    c_tt, a_tt.as_imm(), b_tt.as_imm(), epilogue_fn, ctx
                )
            elif has_compute_fn:
                _matmul_gpu[use_tensor_core=True, transpose_b=True](
                    c_tt, a_tt.as_imm(), b_tt.as_imm(), compute_fn, ctx
                )
            else:
                _matmul_gpu[use_tensor_core=True, transpose_b=True](
                    c_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
                )
            _ = im2col_buf^

        @inline(.always)
        def _launch[
            EpilogueFnType: ElementwiseEpilogueFn,
            ComputeFnType: ElementwiseComputeFn,
            //,
            has_epilogue_fn: Bool,
            has_compute_fn: Bool,
        ](epilogue_fn: EpilogueFnType, compute_fn: ComputeFnType) raises {imm}:
            if K % BLOCK_K == 0 and in_c % BLOCK_K == 0:
                _launch_implicit_gemm[
                    has_epilogue_fn=has_epilogue_fn,
                    has_compute_fn=has_compute_fn,
                ](epilogue_fn, compute_fn)
            else:
                _launch_explicit_im2col[
                    has_epilogue_fn=has_epilogue_fn,
                    has_compute_fn=has_compute_fn,
                ](epilogue_fn, compute_fn)

        # --- Select implicit vs explicit im2col ---
        # The GEMM output `C[m, n]` is the NHWC tensor `[batch, out_h, out_w,
        # N]` flattened to `[M, N]`, so `m` already encodes `(batch, h, w)` and
        # the contiguous output (and same-shape `source`) index is `m*N + n`.
        comptime if has_epilogue_fn:
            var hw = out_h * out_w
            var resid_src = source_ptr
            var resid_beta = beta

            @inline(.always)
            def _gemm_epilogue[
                _dtype: DType,
                _width: SIMDLength,
                *,
                alignment: Int,
            ](coords_2d: Coord, val: SIMD[_dtype, _width]) {
                var hw,
                var out_w,
                var N,
                var resid_src,
                var resid_beta,
                var epilogue_fn,
            }:
                comptime assert (
                    coords_2d.rank == 2
                ), "the matmul epilogue contract is rank 2"
                var m_idx = Int(coords_2d[0])
                var n_idx = Int(coords_2d[1])
                var out_val = val

                # Fold the fused skip connection: out = conv + beta*source.
                comptime if has_residual:
                    var s = (
                        (resid_src.unsafe_offset(m_idx * N + n_idx))
                        .unsafe_load[width=_width]()
                        .cast[_dtype]()
                    )
                    out_val = out_val + s * resid_beta.cast[_dtype]()

                var batch_idx: Int
                var rem_idx: Int
                var h_idx: Int
                var w_idx: Int
                batch_idx, rem_idx = divmod(m_idx, hw)
                h_idx, w_idx = divmod(rem_idx, out_w)
                epilogue_fn(
                    (batch_idx, h_idx, w_idx, n_idx),
                    rebind[SIMD[output_type, _width]](out_val),
                )

            _launch[has_epilogue_fn=True, has_compute_fn=False](
                _gemm_epilogue, no_compute_fn
            )
        elif has_residual or has_compute_fn:
            var hw = out_h * out_w
            var resid_src = source_ptr
            var resid_beta = beta

            # The kernel stores the result, so the closure doesn't capture
            # `output`.
            @inline(.always)
            def _residual_compute[
                _dtype: DType,
                _width: SIMDLength,
                *,
                alignment: Int,
            ](coords_2d: Coord, val: SIMD[_dtype, _width]) {
                var hw,
                var out_w,
                var N,
                var resid_src,
                var resid_beta,
                var compute_fn,
            } -> SIMD[_dtype, _width]:
                comptime assert (
                    coords_2d.rank == 2
                ), "the matmul epilogue contract is rank 2"
                var m_idx = Int(coords_2d[0])
                var n_idx = Int(coords_2d[1])
                var out_val = val

                comptime if has_residual:
                    var s = (
                        (resid_src.unsafe_offset(m_idx * N + n_idx))
                        .unsafe_load[width=_width]()
                        .cast[_dtype]()
                    )
                    out_val = out_val + s * resid_beta.cast[_dtype]()

                comptime if has_compute_fn:
                    var batch_idx, rem_idx = divmod(m_idx, hw)
                    var h_idx, w_idx = divmod(rem_idx, out_w)
                    out_val = rebind[SIMD[_dtype, _width]](
                        compute_fn(
                            (batch_idx, h_idx, w_idx, n_idx),
                            rebind[SIMD[output_type, _width]](out_val),
                        )
                    )
                return out_val

            _launch[has_epilogue_fn=False, has_compute_fn=True](
                no_epilogue_fn, _residual_compute
            )
        else:
            _launch[has_epilogue_fn=False, has_compute_fn=False](
                no_epilogue_fn, no_compute_fn
            )

        # Full sync required to keep filter_buf alive until the kernel completes.
        # TODO: attach buffer to a stream callback to allow pipelining.
        ctx.synchronize()
        _ = filter_buf^
        return True

    return False
