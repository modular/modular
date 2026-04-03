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
from std.gpu import global_idx_uint as global_idx, WARP_SIZE
from std.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.utils.index import IndexList
from linalg.utils import elementwise_epilogue_type
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from .conv2d_kernel import conv2d_kernel_rdna


# =========================================================================
# GPU kernels for explicit im2col fallback path
# =========================================================================


def _im2col_nhwc_kernel[
    dtype: DType,
](
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    batch_size: Int,
    H: Int,
    W: Int,
    C: Int,
    R: Int,
    S: Int,
    H_out: Int,
    W_out: Int,
    pad_h: Int,
    pad_w: Int,
):
    """Transform NHWC input to im2col [M, K] matrix on GPU (stride=1, dilation=1).
    """
    var K = R * S * C
    var HW_out = H_out * W_out
    var total = batch_size * HW_out * K

    var tid = Int(global_idx.x)
    if tid >= total:
        return

    var m = tid // K
    var k_idx = tid - m * K

    var batch = m // HW_out
    var spatial = m - batch * HW_out
    var h_out = spatial // W_out
    var w_out = spatial - h_out * W_out

    var SC = S * C
    var r = k_idx // SC
    var sc = k_idx - r * SC
    var s = sc // C
    var c = sc - s * C

    var h_in = h_out - pad_h + r
    var w_in = w_out - pad_w + s

    var val = Scalar[dtype](0)
    if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
        val = input_ptr[batch * H * W * C + h_in * W * C + w_in * C + c]

    output_ptr.store(tid, val)


def _transpose_rscf_to_nk[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    R: Int,
    S: Int,
    C: Int,
    F: Int,
):
    """GPU kernel: transpose filter RSCF [K,N] -> [N,K] for transpose_b matmul.
    """
    var K = R * S * C
    var total = K * F
    var tid = Int(global_idx.x)
    if tid >= total:
        return
    var f = tid // K
    var k = tid - f * K
    dst_ptr.store(tid, src_ptr.load(k * F + f))


def _transpose_fcrs_to_nk[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    F: Int,
    C: Int,
    R: Int,
    S: Int,
):
    """GPU kernel: transpose filter FCRS -> [N,K] for transpose_b matmul."""
    var K = R * S * C
    var total = K * F
    var tid = Int(global_idx.x)
    if tid >= total:
        return
    var f = tid // K
    var k = tid - f * K
    var r = k // (S * C)
    var sc = k - r * (S * C)
    var s = sc // C
    var c = sc - s * C
    dst_ptr.store(tid, src_ptr.load(f * C * R * S + c * R * S + r * S + s))


# =========================================================================
# RDNA Conv2D dispatch
# =========================================================================


def dispatch_rdna_conv2d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    symmetric_padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Try to dispatch Conv2D on RDNA via implicit GEMM (im2col fused into WMMA).

    Returns True if the convolution was handled, False if the caller should
    fall back to another implementation (e.g. MIOpen).

    Uses the implicit GEMM kernel when C_in is aligned to BLOCK_K (covers all
    FLUX VAE shapes), falling back to explicit im2col + matmul otherwise.
    """

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
            ctx.enqueue_function[
                _transpose_fcrs_to_nk[filter_type],
                _transpose_fcrs_to_nk[filter_type],
            ](
                filter.ptr,
                filter_nk_ptr,
                Int(filter.dim[0]()),
                Int(filter.dim[1]()),
                Int(filter.dim[2]()),
                Int(filter.dim[3]()),
                grid_dim=transpose_grid,
                block_dim=transpose_block,
            )
        else:
            ctx.enqueue_function[
                _transpose_rscf_to_nk[filter_type],
                _transpose_rscf_to_nk[filter_type],
            ](
                filter.ptr,
                filter_nk_ptr,
                Int(filter.dim[0]()),
                Int(filter.dim[1]()),
                Int(filter.dim[2]()),
                Int(filter.dim[3]()),
                grid_dim=transpose_grid,
                block_dim=transpose_block,
            )

        # Choose BLOCK_K based on K alignment
        comptime BLOCK_K = 32

        # --- Helper to launch the implicit GEMM kernel ---
        @parameter
        @always_inline
        def _launch_implicit_gemm[
            _epilogue: Optional[elementwise_epilogue_type] = None,
        ]() raises:
            comptime NUM_WARPS = 16  # 8x2 warp grid
            comptime BLOCK_M = 128
            comptime BLOCK_N = 128

            var filter_nk_tt = TileTensor(
                filter_nk_ptr, row_major(Coord(Idx(N), Idx(K)))
            )
            var out_tt = TileTensor(
                output.ptr, row_major(Coord(Idx(M), Idx(N)))
            )

            comptime conv_kernel = conv2d_kernel_rdna[
                output_type,
                input_type,
                filter_type,
                type_of(out_tt).LayoutType,
                type_of(filter_nk_tt).LayoutType,
                elementwise_lambda_fn=_epilogue,
                BLOCK_K=BLOCK_K,
            ]

            ctx.enqueue_function[conv_kernel, conv_kernel](
                out_tt,
                input.ptr,
                filter_nk_tt,
                M,
                N,
                K,
                out_h * out_w,
                out_w,
                in_h,
                in_w,
                in_c,
                fh,
                fw,
                symmetric_padding[0],
                symmetric_padding[1],
                grid_dim=(ceildiv(N, BLOCK_N), ceildiv(M, BLOCK_M)),
                block_dim=(NUM_WARPS * WARP_SIZE,),
            )

        # --- Helper to launch explicit im2col + matmul fallback ---
        @parameter
        @always_inline
        def _launch_explicit_im2col[
            _epilogue: Optional[elementwise_epilogue_type] = None,
        ]() raises:
            var im2col_size = M * K
            var im2col_buf = ctx.enqueue_create_buffer[input_type](im2col_size)
            var im2col_ptr = im2col_buf.unsafe_ptr()

            comptime im2col_block = 256
            var im2col_grid = ceildiv(im2col_size, im2col_block)
            ctx.enqueue_function[
                _im2col_nhwc_kernel[input_type],
                _im2col_nhwc_kernel[input_type],
            ](
                im2col_ptr,
                input.ptr,
                batch,
                in_h,
                in_w,
                in_c,
                fh,
                fw,
                out_h,
                out_w,
                symmetric_padding[0],
                symmetric_padding[1],
                grid_dim=(im2col_grid,),
                block_dim=(im2col_block,),
            )

            var a_tt = TileTensor(im2col_ptr, row_major(Coord(Idx(M), Idx(K))))
            var b_tt = TileTensor(
                filter_nk_ptr, row_major(Coord(Idx(N), Idx(K)))
            )
            var c_tt = TileTensor(output.ptr, row_major(Coord(Idx(M), Idx(N))))

            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
                elementwise_lambda_fn=_epilogue,
            ](
                c_tt,
                a_tt.as_immut(),
                b_tt.as_immut(),
                ctx,
            )
            _ = im2col_buf^

        # --- Select implicit vs explicit im2col ---
        comptime if maybe_epilogue_func:
            comptime epilogue_4d = maybe_epilogue_func.value()
            var hw = out_h * out_w

            @parameter
            @always_inline
            @__copy_capture(hw, out_w)
            def _gemm_epilogue[
                _dtype: DType,
                _width: Int,
                *,
                alignment: Int = 1,
            ](coords_2d: IndexList[2], val: SIMD[_dtype, _width]):
                var m_idx = coords_2d[0]
                var n_idx = coords_2d[1]
                var batch_idx: Int
                var rem_idx: Int
                var h_idx: Int
                var w_idx: Int
                batch_idx, rem_idx = divmod(m_idx, hw)
                h_idx, w_idx = divmod(rem_idx, out_w)
                epilogue_4d(
                    IndexList[4](batch_idx, h_idx, w_idx, n_idx),
                    rebind[SIMD[output_type, _width]](val),
                )

            if K % BLOCK_K == 0 and in_c % BLOCK_K == 0:
                _launch_implicit_gemm[
                    Optional[elementwise_epilogue_type](_gemm_epilogue)
                ]()
            else:
                _launch_explicit_im2col[
                    Optional[elementwise_epilogue_type](_gemm_epilogue)
                ]()
        else:
            if K % BLOCK_K == 0 and in_c % BLOCK_K == 0:
                _launch_implicit_gemm[]()
            else:
                _launch_explicit_im2col[]()

        # Full sync required to keep filter_buf alive until the kernel completes.
        # TODO: attach buffer to a stream callback to allow pipelining.
        ctx.synchronize()
        _ = filter_buf^
        return True

    return False
