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

from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys.info import size_of
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.utils.index import IndexList
from linalg.utils import elementwise_epilogue_type
from nn.conv.conv_utils import elementwise_simd_epilogue_type


# =========================================================================
# GPU kernels: im2col materialization and filter transpose
# =========================================================================


@__name(t"conv3d_im2col_ndhwc_{dtype}", mangle=True)
def _im2col_ndhwc_kernel[
    dtype: DType,
](
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    batch_size: Int,
    D: Int,
    H: Int,
    W: Int,
    C: Int,
    Q: Int,
    R: Int,
    S: Int,
    D_out: Int,
    H_out: Int,
    W_out: Int,
    pad_d: Int,
    pad_h: Int,
    pad_w: Int,
    stride_d: Int,
    stride_h: Int,
    stride_w: Int,
    m_offset: Int,
    m_count: Int,
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
    var local_m = block_idx.x
    if local_m >= m_count:
        return

    var K = Q * R * S * C
    var m = m_offset + local_m

    # Per-block decomposition (amortized across block_dim threads).
    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out
    var batch, spatial = udivmod(m, DHW_out)
    var d_out, rem = udivmod(spatial, HW_out)
    var h_out, w_out = udivmod(rem, W_out)

    # Precompute input-base offsets and (batch, stride, pad)-dependent
    # addends; within a block only (q, r, s, c) change across threads.
    var d_in_base = d_out * stride_d - pad_d
    var h_in_base = h_out * stride_h - pad_h
    var w_in_base = w_out * stride_w - pad_w
    var batch_base = batch * D * H * W * C
    var dhw_stride = H * W * C
    var hw_stride = W * C
    var w_stride = C

    var RSC = R * S * C
    var SC = S * C

    var row_base = local_m * K
    var k = thread_idx.x
    while k < K:
        var q, rsc = udivmod(k, RSC)
        var r, sc = udivmod(rsc, SC)
        var s, c = udivmod(sc, C)

        var d_in = d_in_base + q
        var h_in = h_in_base + r
        var w_in = w_in_base + s

        var val = Scalar[dtype](0)
        if 0 <= d_in < D and 0 <= h_in < H and 0 <= w_in < W:
            var in_idx = (
                batch_base
                + d_in * dhw_stride
                + h_in * hw_stride
                + w_in * w_stride
                + c
            )
            val = input_ptr[in_idx]

        output_ptr.store(row_base + k, val)
        k += block_dim.x


@__name(t"conv3d_transpose_qrscf_to_nk_{dtype}", mangle=True)
def _transpose_qrscf_to_nk[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    Q: Int,
    R: Int,
    S: Int,
    C: Int,
    F: Int,
):
    """QRSCF [Q,R,S,C,F] -> [F, Q*R*S*C] row-major for matmul transpose_b."""
    var K = Q * R * S * C
    var total = F * K
    var tid = global_idx.x
    if tid >= total:
        return

    var f, k = udivmod(tid, K)

    var RSC = R * S * C
    var SC = S * C
    var q, rsc = udivmod(k, RSC)
    var r, sc = udivmod(rsc, SC)
    var s, c = udivmod(sc, C)

    var src_idx = ((q * R + r) * S + s) * C * F + c * F + f
    dst_ptr.store(tid, src_ptr.load(src_idx))


@__name(t"conv3d_transpose_fcqrs_to_nk_{dtype}", mangle=True)
def _transpose_fcqrs_to_nk[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    F: Int,
    C: Int,
    Q: Int,
    R: Int,
    S: Int,
):
    """FCQRS [F,C,Q,R,S] -> [F, Q*R*S*C] row-major for matmul transpose_b."""
    var K = Q * R * S * C
    var total = F * K
    var tid = global_idx.x
    if tid >= total:
        return

    var f, k = udivmod(tid, K)

    var RSC = R * S * C
    var SC = S * C
    var q, rsc = udivmod(k, RSC)
    var r, sc = udivmod(rsc, SC)
    var s, c = udivmod(sc, C)

    var src_idx = f * C * Q * R * S + c * Q * R * S + q * R * S + r * S + s
    dst_ptr.store(tid, src_ptr.load(src_idx))


# =========================================================================
# Public dispatch entry point
# =========================================================================

comptime _DEFAULT_M_TILE_BYTE_BUDGET = 256 * 1024 * 1024
comptime _MIN_M_TILE = 1024


def dispatch_im2col_matmul_conv3d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
    m_tile_byte_budget: Int = _DEFAULT_M_TILE_BYTE_BUDGET,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    symmetric_padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Try to dispatch a 3-D conv as explicit im2col + generic matmul.

    Returns True if the conv was handled; False if the caller should fall
    back to another implementation (naive Mojo kernel, cuDNN, etc.).

    Skips on: non-bf16 dtype, grouped conv, dilation != 1, kernel size
    1x1x1 (the vectorized naive kernel wins on tiny shapes), and K too
    small for the matmul fast path.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"

    comptime if input_type != DType.bfloat16:
        return False

    if num_groups != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1:
        return False

    var batch = Int(input.dim[0]())
    var D = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())
    var C_in = Int(input.dim[4]())

    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())
    var C_out = Int(output.dim[4]())

    var Q: Int
    var R: Int
    var S: Int
    comptime if filter_is_fcrs:
        Q = Int(filter.dim[2]())
        R = Int(filter.dim[3]())
        S = Int(filter.dim[4]())
    else:
        Q = Int(filter.dim[0]())
        R = Int(filter.dim[1]())
        S = Int(filter.dim[2]())

    # Bail on 1x1x1: the vectorized naive kernel already beats cuDNN there,
    # and K is too small to amortize the matmul launch overhead.
    if Q == 1 and R == 1 and S == 1:
        return False

    var full_M = batch * D_out * H_out * W_out
    var K = Q * R * S * C_in
    var N = C_out

    # Minimum sane K. _matmul_gpu's SM100 path handles small K fine, but
    # K < 16 is below MMA_K even for bf16 and not worth the scratch.
    if K < 16:
        return False

    # --- Transpose filter to [N, K] once, before the M-tile loop. ---
    var filter_size = filter.num_elements()
    var filter_nk_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_nk_ptr = filter_nk_buf.unsafe_ptr()

    comptime transpose_block = 256
    var transpose_grid = ceildiv(filter_size, transpose_block)

    comptime if filter_is_fcrs:
        ctx.enqueue_function[
            _transpose_fcqrs_to_nk[filter_type],
            _transpose_fcqrs_to_nk[filter_type],
        ](
            filter.ptr,
            filter_nk_ptr,
            Int(filter.dim[0]()),
            Int(filter.dim[1]()),
            Int(filter.dim[2]()),
            Int(filter.dim[3]()),
            Int(filter.dim[4]()),
            grid_dim=transpose_grid,
            block_dim=transpose_block,
        )
    else:
        ctx.enqueue_function[
            _transpose_qrscf_to_nk[filter_type],
            _transpose_qrscf_to_nk[filter_type],
        ](
            filter.ptr,
            filter_nk_ptr,
            Int(filter.dim[0]()),
            Int(filter.dim[1]()),
            Int(filter.dim[2]()),
            Int(filter.dim[3]()),
            Int(filter.dim[4]()),
            grid_dim=transpose_grid,
            block_dim=transpose_block,
        )

    # --- Decide M-tile size so scratch stays bounded. ---
    # First pick a cap from the byte budget. Then equalize: if multiple
    # tiles are needed, prefer ceildiv(full_M, num_tiles) over the cap to
    # avoid a tiny ragged tail tile that under-uses the matmul.
    var bytes_per_row = K * size_of[input_type]()
    var m_tile_by_budget = m_tile_byte_budget // bytes_per_row
    var m_tile_cap = (
        m_tile_by_budget if m_tile_by_budget > _MIN_M_TILE else _MIN_M_TILE
    )
    var num_tiles: Int
    var m_tile: Int
    if full_M <= m_tile_cap:
        num_tiles = 1
        m_tile = full_M
    else:
        num_tiles = ceildiv(full_M, m_tile_cap)
        m_tile = ceildiv(full_M, num_tiles)

    var im2col_buf = ctx.enqueue_create_buffer[input_type](m_tile * K)
    var im2col_ptr = im2col_buf.unsafe_ptr()

    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out

    # --- M-tile loop. ---
    var m_offset = 0
    while m_offset < full_M:
        var remaining = full_M - m_offset
        var m_count = m_tile if remaining > m_tile else remaining

        # Block-per-row: one block per output voxel, threads cooperate on K.
        comptime im2col_block = 256
        ctx.enqueue_function[
            _im2col_ndhwc_kernel[input_type],
            _im2col_ndhwc_kernel[input_type],
        ](
            im2col_ptr,
            input.ptr,
            batch,
            D,
            H,
            W,
            C_in,
            Q,
            R,
            S,
            D_out,
            H_out,
            W_out,
            symmetric_padding[0],
            symmetric_padding[1],
            symmetric_padding[2],
            stride[0],
            stride[1],
            stride[2],
            m_offset,
            m_count,
            grid_dim=m_count,
            block_dim=im2col_block,
        )

        var a_tt = TileTensor(
            im2col_ptr, row_major(Coord(Idx(m_count), Idx(K)))
        )
        var b_tt = TileTensor(filter_nk_ptr, row_major(Coord(Idx(N), Idx(K))))
        # Output is NDHWC = [batch, D_out, H_out, W_out, C_out]; rows in the
        # flattened [M, N] layout are contiguous, so we advance by m_offset * N.
        var c_ptr = output.ptr + m_offset * N
        var c_tt = TileTensor(c_ptr, row_major(Coord(Idx(m_count), Idx(N))))

        comptime if maybe_epilogue_func:
            comptime epilogue_5d = maybe_epilogue_func.value()

            @parameter
            @always_inline
            @__copy_capture(DHW_out, HW_out, H_out, W_out, m_offset)
            def _gemm_epilogue[
                _dtype: DType,
                _width: Int,
                *,
                alignment: Int = 1,
            ](coords_2d: IndexList[2], val: SIMD[_dtype, _width]):
                var full_m = m_offset + coords_2d[0]
                var n_idx = coords_2d[1]
                var batch_idx = full_m // DHW_out
                var sp = full_m - batch_idx * DHW_out
                var d_idx = sp // HW_out
                var rem2 = sp - d_idx * HW_out
                var h_idx = rem2 // W_out
                var w_idx = rem2 - h_idx * W_out
                epilogue_5d(
                    IndexList[5](batch_idx, d_idx, h_idx, w_idx, n_idx),
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
