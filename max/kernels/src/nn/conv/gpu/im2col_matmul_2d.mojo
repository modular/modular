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
"""Explicit im2col + `_matmul_gpu` dispatch for 2D convolution.

Materialises an im2col `[M, K]` scratch into global memory and calls the
generic `_matmul_gpu` on it. `_matmul_gpu` auto-routes to SM100 UMMA on
Blackwell for bf16, giving non-128-aligned-channel 2-D convs access to
tensor cores without the TMA im2col descriptor layer.

- M = batch * H_out * W_out  (linearized output pixel)
- K = R * S * C_in            (filter-flattened reduction axis)
- N = C_out                   (output channels)

Gate: bf16, groups=1, dilation=1, kernel > 1×1 (the vectorized naive
kernel wins on 1×1), K >= 16 (below MMA_K).
"""

from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys.info import has_apple_gpu_accelerator, size_of
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.apple import ConvIm2colParams, enqueue_apple_conv2d
from std.utils.index import IndexList
from linalg.utils import elementwise_epilogue_type
from nn.conv.conv_utils import elementwise_simd_epilogue_type


# =========================================================================
# GPU kernels: im2col materialisation and filter transpose
# =========================================================================


@__name(t"conv2d_im2col_nhwc_{dtype}")
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
    stride_h: Int,
    stride_w: Int,
    m_offset: Int,
    m_count: Int,
):
    """Write rows [m_offset, m_offset + m_count) of the 4D im2col matrix.

    M = batch * H_out * W_out (linearized output pixel).
    K = R * S * C (filter-flattened reduction axis).
    Output is laid out [m_count, K] row-major starting at `output_ptr`.
    Dilation is assumed to be 1 (enforced at dispatch time).

    Block-per-row layout: one block handles a single output pixel (row of
    the im2col matrix); threads within the block cooperate on the K axis.
    """
    var local_m = block_idx.x
    if local_m >= m_count:
        return

    var K = R * S * C
    var m = m_offset + local_m

    # Per-block decomposition (amortized across block_dim threads).
    var HW_out = H_out * W_out
    var batch, spatial = udivmod(m, HW_out)
    var h_out, w_out = udivmod(spatial, W_out)

    var h_in_base = h_out * stride_h - pad_h
    var w_in_base = w_out * stride_w - pad_w
    var batch_base = batch * H * W * C
    var hw_stride = W * C
    var w_stride = C

    var SC = S * C

    var row_base = local_m * K
    var k = thread_idx.x
    while k < K:
        var r, sc = udivmod(k, SC)
        var s, c = udivmod(sc, C)

        var h_in = h_in_base + r
        var w_in = w_in_base + s

        var val = Scalar[dtype](0)
        if 0 <= h_in < H and 0 <= w_in < W:
            var in_idx = batch_base + h_in * hw_stride + w_in * w_stride + c
            val = input_ptr[in_idx]

        output_ptr.store(row_base + k, val)
        k += block_dim.x


@__name(t"conv2d_transpose_filter_to_nk_{dtype}_{filter_is_fcrs}")
def _transpose_filter_to_nk[
    dtype: DType,
    filter_is_fcrs: Bool,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    R: Int,
    S: Int,
    C: Int,
    F: Int,
):
    """Transpose RSCF or FCRS filter layout to [F, R*S*C] for matmul transpose_b.
    """
    var K = R * S * C
    var total = F * K
    var tid = global_idx.x
    if tid >= total:
        return

    var f, k = udivmod(tid, K)

    var SC = S * C
    var r, sc = udivmod(k, SC)
    var s, c = udivmod(sc, C)

    var src_idx: Int
    comptime if filter_is_fcrs:
        src_idx = f * C * R * S + c * R * S + r * S + s
    else:
        src_idx = (r * S + s) * C * F + c * F + f
    dst_ptr.store(tid, src_ptr.load(src_idx))


# =========================================================================
# Public dispatch entry point
# =========================================================================

comptime _DEFAULT_M_TILE_BYTE_BUDGET = 256 * 1024 * 1024
comptime _MIN_M_TILE = 1024


def dispatch_im2col_matmul_conv2d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
    m_tile_byte_budget: Int = _DEFAULT_M_TILE_BYTE_BUDGET,
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
    """Try to dispatch a 2-D conv as explicit im2col + generic matmul.

    Returns True if the conv was handled; False if the caller should fall
    back to another implementation (naive Mojo kernel, cuDNN, etc.).

    Skips on: non-bf16 dtype, grouped conv, dilation != 1, kernel size
    1x1 (the vectorized naive kernel wins on tiny shapes), and K too
    small for the matmul fast path.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    comptime if input_type != DType.bfloat16:
        return False

    if num_groups != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1:
        return False

    var batch = Int(input.dim[0]())
    var H = Int(input.dim[1]())
    var W = Int(input.dim[2]())
    var C_in = Int(input.dim[3]())

    var H_out = Int(output.dim[1]())
    var W_out = Int(output.dim[2]())
    var C_out = Int(output.dim[3]())

    var R: Int
    var S: Int
    comptime if filter_is_fcrs:
        R = Int(filter.dim[2]())
        S = Int(filter.dim[3]())
    else:
        R = Int(filter.dim[0]())
        S = Int(filter.dim[1]())

    # The vectorized naive kernel beats cuDNN on 1x1, and K is too small
    # to amortize the matmul launch overhead there.
    if R == 1 and S == 1:
        return False

    var full_M = batch * H_out * W_out
    var K = R * S * C_in
    var N = C_out

    # Minimum sane K. _matmul_gpu's SM100 path handles small K fine, but
    # K < 16 is below MMA_K even for bf16 and not worth the scratch.
    if K < 16:
        return False

    # Degenerate N shapes (e.g. conv_out 96->3) don't amortize the
    # matmul launch cost; the naive kernel wins on these. On B200 we
    # measured naive at 0.21 ms vs im2col at 0.66 ms for C_out=3.
    # Apple has no naive FCRS path, so the matmul must take small N too.
    comptime if not has_apple_gpu_accelerator():
        if N < 16:
            return False

    # Filter transpose runs once, before the M-tile loop.
    var filter_size = filter.num_elements()
    var filter_nk_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_nk_ptr = filter_nk_buf.unsafe_ptr()

    comptime transpose_block = 256
    var transpose_grid = ceildiv(filter_size, transpose_block)

    # The kernel expects (R, S, C, F) — reorder FCRS dims to match.
    var R_dim: Int
    var S_dim: Int
    var C_dim: Int
    var F_dim: Int
    comptime if filter_is_fcrs:
        F_dim = Int(filter.dim[0]())
        C_dim = Int(filter.dim[1]())
        R_dim = Int(filter.dim[2]())
        S_dim = Int(filter.dim[3]())
    else:
        R_dim = Int(filter.dim[0]())
        S_dim = Int(filter.dim[1]())
        C_dim = Int(filter.dim[2]())
        F_dim = Int(filter.dim[3]())

    ctx.enqueue_function[_transpose_filter_to_nk[filter_type, filter_is_fcrs]](
        filter.ptr,
        filter_nk_ptr,
        R_dim,
        S_dim,
        C_dim,
        F_dim,
        grid_dim=transpose_grid,
        block_dim=transpose_block,
    )

    # If multiple tiles are needed, equalize to avoid a tiny ragged tail
    # tile that would under-use the matmul.
    var bytes_per_row = K * size_of[input_type]()
    var m_tile_by_budget = m_tile_byte_budget // bytes_per_row
    var m_tile_cap = (
        m_tile_by_budget if m_tile_by_budget > _MIN_M_TILE else _MIN_M_TILE
    )
    var m_tile: Int
    if full_M <= m_tile_cap:
        m_tile = full_M
    else:
        var num_tiles = ceildiv(full_M, m_tile_cap)
        m_tile = ceildiv(full_M, num_tiles)

    var im2col_buf = ctx.enqueue_create_buffer[input_type](m_tile * K)
    var im2col_ptr = im2col_buf.unsafe_ptr()

    var HW_out = H_out * W_out

    var m_offset = 0
    while m_offset < full_M:
        var remaining = full_M - m_offset
        var m_count = m_tile if remaining > m_tile else remaining

        # Block-per-row: one block per output pixel, threads cooperate on K.
        comptime im2col_block = 256
        ctx.enqueue_function[_im2col_nhwc_kernel[input_type]](
            im2col_ptr,
            input.ptr,
            batch,
            H,
            W,
            C_in,
            R,
            S,
            H_out,
            W_out,
            symmetric_padding[0],
            symmetric_padding[1],
            stride[0],
            stride[1],
            m_offset,
            m_count,
            grid_dim=m_count,
            block_dim=im2col_block,
        )

        var a_tt = TileTensor(im2col_ptr, row_major(Coord(m_count, K)))
        var b_tt = TileTensor(filter_nk_ptr, row_major(Coord(N, K)))
        # NHWC rows are contiguous in the flattened [M, N] layout.
        var c_ptr = output.ptr + m_offset * N
        var c_tt = TileTensor(c_ptr, row_major(Coord(m_count, N)))

        comptime if maybe_epilogue_func:
            comptime epilogue_4d = maybe_epilogue_func.value()

            @parameter
            @always_inline
            @__copy_capture(HW_out, W_out, m_offset)
            def _gemm_epilogue[
                _dtype: DType,
                _width: SIMDSize,
                *,
                alignment: Int = 1,
            ](coords_2d: IndexList[2], val: SIMD[_dtype, _width]):
                var full_m = m_offset + coords_2d[0]
                var n_idx = coords_2d[1]
                var batch_idx = full_m // HW_out
                var sp = full_m - batch_idx * HW_out
                var h_idx = sp // W_out
                var w_idx = sp - h_idx * W_out
                epilogue_4d(
                    IndexList[4](batch_idx, h_idx, w_idx, n_idx),
                    rebind[SIMD[output_type, _width]](val),
                )

            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
                elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                    _gemm_epilogue
                ),
            ](c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx)
        else:
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
            ](c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx)

        m_offset += m_count

    # Synchronize so scratch stays alive until kernels finish.
    # TODO: stream-callback lifetime management would allow pipelining.
    ctx.synchronize()
    _ = filter_nk_buf^
    _ = im2col_buf^
    return True


def dispatch_fused_im2col_conv2d_apple[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
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
    """Apple M5 fused online-im2col conv2d (no `[M, K]` scratch materialised).

    Apple silicon (`compute_capability == 5`). Mirrors the MI355 conv pattern:
    the filter is transposed to the `(C_out, K=R*S*C_in)` NK matrix once (the
    same kernel the materialised path uses), then `enqueue_apple_conv2d` runs the
    simdgroup-tiled GEMM with the A operand gathered from the NHWC input on the
    fly -- so the im2col matrix never touches global memory. This wins across
    both compute- and memory-bound regimes, removing the materialised path's
    memory-bound loss (and the `conv_gpu` Apple memory-bound naive guard).

    Returns True if handled; False to fall back. Self-gates: bf16, groups=1,
    dilation=1, kernel > 1x1, K=R*S*C_in >= 16. Unlike the materialised
    dispatcher, there is NO N=C_out >= 16 gate: the fused path has no `[M, K]`
    scratch round-trip, so small C_out (down to 1) takes the MMA path and beats
    the naive thread-per-pixel conv (which is also broken on Metal for the
    C_out=3 VAE->RGB shape). The MMA handles tiny N via the existing edge-tile
    mask (`b_valid_cols` zero-fill in the B load + `acol < n` in the epilogue),
    all comptime-fixed by SG_N=32 and independent of the runtime N.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    comptime if input_type != DType.bfloat16:
        return False

    if num_groups != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1:
        return False

    var batch = Int(input.dim[0]())
    var H = Int(input.dim[1]())
    var W = Int(input.dim[2]())
    var C_in = Int(input.dim[3]())

    var H_out = Int(output.dim[1]())
    var W_out = Int(output.dim[2]())
    var C_out = Int(output.dim[3]())

    var R: Int
    var S: Int
    comptime if filter_is_fcrs:
        R = Int(filter.dim[2]())
        S = Int(filter.dim[3]())
    else:
        R = Int(filter.dim[0]())
        S = Int(filter.dim[1]())

    if R == 1 and S == 1:
        return False

    var full_M = batch * H_out * W_out
    var K = R * S * C_in
    var N = C_out

    if K < 16:
        return False
    # No N>=16 gate: the fused MMA path has no `[M, K]` scratch round-trip, so
    # it handles small C_out (down to 1) on the MMA path -- edge-masked by
    # `b_valid_cols` (B-load zero-fill) and `acol < n` (epilogue). The
    # materialised dispatcher keeps N>=16 because for tiny C_out naive beat its
    # scratch round-trip; that tradeoff does not apply here, and naive is broken
    # on Metal for the C_out=3 VAE->RGB conv this routes around.
    if N < 1:
        return False

    # The 16x16 simdgroup MMA needs M5+, fall back to the materialised matmul path.
    if ctx.compute_capability() < 5:
        return False

    # Filter transpose to (C_out, K) -- identical to the materialised path so
    # the K-ordering (r, s, c) matches the gather's (M, K) -> NHWC map.
    var filter_size = filter.num_elements()
    var filter_nk_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_nk_ptr = filter_nk_buf.unsafe_ptr()

    comptime transpose_block = 256
    var transpose_grid = ceildiv(filter_size, transpose_block)

    var R_dim: Int
    var S_dim: Int
    var C_dim: Int
    var F_dim: Int
    comptime if filter_is_fcrs:
        F_dim = Int(filter.dim[0]())
        C_dim = Int(filter.dim[1]())
        R_dim = Int(filter.dim[2]())
        S_dim = Int(filter.dim[3]())
    else:
        R_dim = Int(filter.dim[0]())
        S_dim = Int(filter.dim[1]())
        C_dim = Int(filter.dim[2]())
        F_dim = Int(filter.dim[3]())

    ctx.enqueue_function[_transpose_filter_to_nk[filter_type, filter_is_fcrs]](
        filter.ptr,
        filter_nk_ptr,
        R_dim,
        S_dim,
        C_dim,
        F_dim,
        grid_dim=transpose_grid,
        block_dim=transpose_block,
    )

    # The Apple conv launcher takes a single `in_type` for input and filter
    # (the GEMM's A and B). Under the bf16 gate above, filter_type == input_type;
    # rebind the NK pointer so the type matches the input's.
    comptime assert (
        filter_type == input_type
    ), "Apple fused conv expects filter dtype == input dtype (bf16)"
    var filter_nk_in_ptr = rebind[
        UnsafePointer[Scalar[input_type], MutAnyOrigin]
    ](filter_nk_ptr)
    var filter_nk = TileTensor(filter_nk_in_ptr, row_major(Coord(N, K)))
    # Flat (M, N) view of the NHWC output buffer (NHWC rows are contiguous).
    var c_tt = TileTensor(output.ptr, row_major(Coord(full_M, N)))

    var conv = ConvIm2colParams(
        H=Int32(H),
        W=Int32(W),
        C=Int32(C_in),
        R=Int32(R),
        S=Int32(S),
        H_out=Int32(H_out),
        W_out=Int32(W_out),
        pad_h=Int32(symmetric_padding[0]),
        pad_w=Int32(symmetric_padding[1]),
        stride_h=Int32(stride[0]),
        stride_w=Int32(stride[1]),
    )

    comptime if maybe_epilogue_func:
        comptime epilogue_4d = maybe_epilogue_func.value()
        var HW_out = H_out * W_out

        @parameter
        @always_inline
        @__copy_capture(HW_out, W_out)
        def _gemm_epilogue[
            _dtype: DType,
            _width: SIMDSize,
            *,
            alignment: Int = 1,
        ](coords_2d: IndexList[2], val: SIMD[_dtype, _width]):
            var full_m = coords_2d[0]
            var n_idx = coords_2d[1]
            var batch_idx = full_m // HW_out
            var sp = full_m - batch_idx * HW_out
            var h_idx = sp // W_out
            var w_idx = sp - h_idx * W_out
            epilogue_4d(
                IndexList[4](batch_idx, h_idx, w_idx, n_idx),
                rebind[SIMD[output_type, _width]](val),
            )

        enqueue_apple_conv2d[
            in_type=input_type,
            c_type=output_type,
            elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                _gemm_epilogue
            ),
        ](c_tt, input, filter_nk, conv, ctx)
    else:
        enqueue_apple_conv2d[
            in_type=input_type,
            c_type=output_type,
        ](c_tt, input, filter_nk, conv, ctx)

    ctx.synchronize()
    _ = filter_nk_buf^
    return True
