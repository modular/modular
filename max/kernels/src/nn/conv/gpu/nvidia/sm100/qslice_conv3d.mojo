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
"""Q-slice 3-D conv → Q sequential SM100 2-D conv calls with fp32
accumulator.

A 3D conv with stride=1, dilation=1 decomposes exactly as:

    output[n, d_o, h_o, w_o, f]
      = Σ_q  [ Σ_{r,s,c}  input[n, d_o+q, h_o+r-pad_h, w_o+s-pad_w, c]
                          * filter[q, r, s, c, f] ]

The inner Σ_{r,s,c} is a 2-D conv on a single depth slice, so by
invoking the existing `dispatch_sm100_conv2d` Q times we inherit SM100
UMMA performance without forking any TMA code.

ACCUMULATION STRATEGY
---------------------
Each SM100 2-D conv produces a bf16 result (the kernel uses an fp32
accumulator internally but stores bf16). A direct attempt to chain
Q convs via `has_residual=True` / `beta=1.0` round-trips the running
sum through bf16 every step, which compounds per-element rounding
far beyond what a single 3-D conv produces (empirically ~100-1000
diff vs the reference on WAN mid_res shapes).

This dispatcher instead maintains a **dedicated fp32 accumulator
buffer** outside the conv calls:

  1. Allocate `accum_fp32` of output size and zero-fill it.
  2. Allocate one reusable `temp_bf16` buffer of output size.
  3. For q in [0, Q):
        - Call `dispatch_sm100_conv2d(has_residual=False)` to write
          `conv(input[q], filter[q])` into `temp_bf16`.
        - Launch an elementwise kernel that reads `temp_bf16`, casts
          to fp32, and adds into `accum_fp32`.
  4. Launch a final kernel that casts `accum_fp32` → bf16 and writes
     to user `output` (fused with the caller's 5-D epilogue when one
     is provided).

Only the final cast is lossy; all intermediate accumulation is fp32,
matching what a single all-at-once 3-D conv would produce internally.

Gate:
- bf16 input/filter/output dtype.
- SM100 device (`_is_sm10x_gpu`).
- `filter_is_fcrs=False` (QRSCF only — the per-q slab is a
  contiguous RSCF view at offset `q*R*S*C*F`; FCQRS would need a
  separate extraction kernel because a fixed-q FCQRS slice is
  non-contiguous).
- stride=1, dilation=1, groups=1, Q>1.
- Zero temporal padding (WAN's causal 3D convs pre-pad temporal
  externally).
- `C_in % 64 == 0` and `C_out % 64 == 0` (SM100 alignment).

When `C_out` is 64-aligned but not 128-aligned (e.g., C_out=192),
the dispatcher zero-pads the filter's F axis up to the next
multiple of 128 (once, before the q loop), runs the accumulator
at `C_out_padded`, and strides the final fp32→bf16 cast to drop
the padded columns when writing user output. This costs ~33%
extra compute on the 192→192 shape but keeps the MMA at its
native 128-wide N tile.

Declined shapes fall through to `dispatch_im2col_matmul_conv3d`.
"""

from std.collections import OptionalReg
from std.math import ceildiv, gcd
from std.math.uutils import udivmod
from std.gpu import global_idx
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.gpu.host.info import _is_sm10x_gpu
from layout import Coord, Idx, TileTensor, row_major
from std.sys import align_of, simd_width_of
from std.utils import IndexList
from linalg.utils import elementwise_epilogue_type
from nn.conv.conv_utils import elementwise_simd_epilogue_type

from .dispatch import dispatch_sm100_conv2d


# =========================================================================
# Elementwise helper kernels
# =========================================================================


@__name(t"qslice_accum_bf16_to_fp32_{dtype}", mangle=True)
def _accum_bf16_to_fp32_kernel[
    dtype: DType,
    output_simd_width: Int,
](
    accum_fp32_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    src_bf16_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    accum_slice_elems: Int,
):
    """Elementwise `accum_fp32[i] += src_bf16[i].cast[fp32]()`.

    One thread per element; no atomics — each thread owns its slot.
    """
    comptime bf16_alignment = align_of[SIMD[dtype, output_simd_width]]()
    comptime fp32_alignment = align_of[SIMD[DType.float32, output_simd_width]]()

    var accum_idx = global_idx.x * output_simd_width
    if accum_idx >= accum_slice_elems:
        return
    var src_val = src_bf16_ptr.load[
        width=output_simd_width, alignment=bf16_alignment
    ](accum_idx).cast[DType.float32]()
    var accum_val = accum_fp32_ptr.load[
        width=output_simd_width, alignment=fp32_alignment
    ](accum_idx)
    accum_fp32_ptr.store[alignment=fp32_alignment](
        accum_idx, accum_val + src_val
    )


@__name(t"qslice_fp32_to_dtype_plain_{dtype}", mangle=True)
def _fp32_to_dtype_plain_kernel[
    dtype: DType,
    output_simd_width: Int,
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_fp32_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    output_elems: Int,
):
    """Elementwise cast fp32 → `dtype` with no epilogue."""
    var output_idx = global_idx.x * output_simd_width
    if output_idx >= output_elems:
        return
    dst_ptr.store[alignment=align_of[SIMD[dtype, output_simd_width]]()](
        output_idx,
        src_fp32_ptr.load[
            width=output_simd_width,
            alignment=align_of[SIMD[DType.float32, output_simd_width]](),
        ](output_idx).cast[dtype](),
    )


@__name(t"qslice_fp32_to_dtype_epilogue_{dtype}", mangle=True)
def _fp32_to_dtype_epilogue_kernel[
    dtype: DType,
    C_out: Int,
    epilogue: elementwise_simd_epilogue_type,
    output_simd_width: Int,
](
    src_fp32_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch: Int,
    D_out: Int,
    H_out: Int,
    W_out: Int,
    output_elems: Int,
):
    """Elementwise cast fp32 → `dtype`, then call the caller's 5-D
    epilogue. The epilogue is expected to perform the write
    (matching the MOGG `output._lambda_store` contract).
    """
    var output_idx = global_idx.x * output_simd_width
    if output_idx >= output_elems:
        return
    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out
    var b, rem = udivmod(output_idx, DHW_out * C_out)
    var d: Int
    d, rem = udivmod(rem, HW_out * C_out)
    var h: Int
    h, rem = udivmod(rem, W_out * C_out)
    var w: Int
    var c: Int
    w, c = udivmod(rem, C_out)
    var val = src_fp32_ptr.load[
        width=output_simd_width,
        alignment=align_of[SIMD[DType.float32, output_simd_width]](),
    ](output_idx).cast[dtype]()
    epilogue[alignment=output_simd_width](IndexList[5](b, d, h, w, c), val)


@__name(t"qslice_pad_filter_qrscf_{dtype}", mangle=True)
def _pad_filter_qrscf_kernel[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    Q: Int,
    R: Int,
    S: Int,
    C_in: Int,
    C_out: Int,
    C_out_padded: Int,
):
    """Zero-pad a QRSCF filter's F dimension from `C_out` to
    `C_out_padded`. One thread per destination element.
    """
    var tid = global_idx.x
    var total = Q * R * S * C_in * C_out_padded
    if tid >= total:
        return
    var qrsc_stride = C_out_padded
    var qrs_stride = C_in * qrsc_stride
    var qr_stride = S * qrs_stride
    var q_stride = R * qr_stride
    var q: Int
    var rem: Int
    q, rem = udivmod(tid, q_stride)
    var r: Int
    r, rem = udivmod(rem, qr_stride)
    var s: Int
    s, rem = udivmod(rem, qrs_stride)
    var c: Int
    var f: Int
    c, f = udivmod(rem, qrsc_stride)
    if f >= C_out:
        dst_ptr.store(tid, Scalar[dtype](0))
        return
    var src_idx = (((q * R + r) * S + s) * C_in + c) * C_out + f
    dst_ptr.store(tid, src_ptr.load(src_idx))


@__name(t"qslice_copy_filter_{dtype}", mangle=True)
def _copy_filter_kernel[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    total: Int,
):
    """Plain indexed filter copy used when no F-padding is required.

    Avoids the divmod chain in `_pad_filter_qrscf_kernel` for the
    common aligned case. The divmod version still works correctly
    with `C_out == C_out_padded`, but wastes cycles on the five-way
    coord unflatten.
    """
    var tid = global_idx.x
    if tid >= total:
        return
    dst_ptr.store(tid, src_ptr.load(tid))


@__name(t"qslice_fp32_strided_to_dtype_plain_{dtype}", mangle=True)
def _fp32_strided_to_dtype_plain_kernel[
    dtype: DType,
    C_out: Int,
    C_out_padded: Int,
    output_simd_width: Int,
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_fp32_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    output_elems: Int,
):
    """Cast fp32 → `dtype` while stripping padded C channels.

    The accumulator was run at `C_out_padded` columns; write only the
    first `C_out` per pixel to the user output. `total` is the user
    output element count = batch*D_out*H_out*W_out*C_out.
    """
    var output_idx = global_idx.x * output_simd_width
    if output_idx >= output_elems:
        return
    var pixel, c = udivmod(output_idx, C_out)
    var src_idx = pixel * C_out_padded + c
    dst_ptr.store[alignment=align_of[SIMD[dtype, output_simd_width]]()](
        output_idx,
        src_fp32_ptr.load[
            width=output_simd_width,
            alignment=align_of[SIMD[DType.float32, output_simd_width]](),
        ](src_idx).cast[dtype](),
    )


@__name(t"qslice_fp32_strided_to_dtype_epilogue_{dtype}", mangle=True)
def _fp32_strided_to_dtype_epilogue_kernel[
    dtype: DType,
    C_out: Int,
    C_out_padded: Int,
    epilogue: elementwise_simd_epilogue_type,
    output_simd_width: Int,
](
    src_fp32_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch: Int,
    D_out: Int,
    H_out: Int,
    W_out: Int,
    output_elems: Int,
):
    """Strided fp32 → `dtype` with 5-D epilogue.

    The accumulator was run at `C_out_padded` columns; this kernel
    walks only over the first `C_out` per pixel and calls the user's
    5-D epilogue with the de-padded `c` coordinate.
    """
    var output_idx = global_idx.x * output_simd_width
    if output_idx >= output_elems:
        return
    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out
    var b, rem = udivmod(output_idx, DHW_out * C_out)
    var d: Int
    d, rem = udivmod(rem, HW_out * C_out)
    var h: Int
    h, rem = udivmod(rem, W_out * C_out)
    var w: Int
    var c: Int
    w, c = udivmod(rem, C_out)
    var src_pixel = ((b * D_out + d) * H_out + h) * W_out + w
    var src_idx = src_pixel * C_out_padded + c
    var val = src_fp32_ptr.load[
        width=output_simd_width,
        alignment=align_of[SIMD[DType.float32, output_simd_width]](),
    ](src_idx).cast[dtype]()
    epilogue[alignment=output_simd_width](IndexList[5](b, d, h, w, c), val)


# =========================================================================
# Public dispatch entry point
# =========================================================================


def dispatch_qslice_conv3d_sm100[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
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
    """Try to dispatch a 3-D conv as Q × SM100 2-D conv calls with a
    dedicated fp32 accumulator.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"

    comptime if not filter.shape_known:
        return False

    comptime if input_type != DType.bfloat16:
        return False
    comptime if output_type != DType.bfloat16:
        return False

    # FCQRS slab extraction would need a dedicated kernel (a fixed-q
    # slice through FCQRS is non-contiguous). QRSCF slab q is just
    # RSCF at offset q*R*S*C*F so we can lean on dispatch_sm100_conv2d.
    comptime if filter_is_fcrs:
        return False

    if num_groups != 1:
        return False
    if stride[0] != 1 or stride[1] != 1 or stride[2] != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1:
        return False

    comptime _is_sm100 = _is_sm10x_gpu(ctx.default_device_info)
    comptime if not _is_sm100:
        return False

    if symmetric_padding[0] != 0:
        return False

    comptime Q = filter.static_shape[0]
    comptime R = filter.static_shape[1]
    comptime S = filter.static_shape[2]

    comptime if Q <= 1:
        return False

    # R=S=1 degenerates the per-q 2-D conv into a matmul. Running Q
    # sequential SM100 conv2d calls (each with its own filter
    # transpose + internal synchronize) is structurally slower than
    # im2col+matmul's single fused matmul on these shapes — im2col
    # collapses R*S*Q into the K dimension and runs one GEMM. Decline
    # so we fall through to the im2col path.
    comptime if R == 1 and S == 1:
        return False

    var batch = Int(input.dim[0]())
    var D = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())

    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())

    comptime C_in = filter.static_shape[3]
    comptime C_out = filter.static_shape[4]

    comptime if C_in % 64 != 0:
        return False

    # SM100 UMMA requires the N macro-tile (MMA_N=128) to divide C_out.
    # We accept C_out that is 64-aligned but not 128-aligned by zero-padding
    # the F axis of the filter up to the next multiple of 128; the final
    # write strips the padded columns back out.
    comptime if C_out % 64 != 0:
        return False

    if D_out != D - Q + 1:
        return False

    var pad_h = symmetric_padding[1]
    var pad_w = symmetric_padding[2]
    var symmetric_padding_2d = IndexList[2](pad_h, pad_w)

    comptime C_out_padded = ceildiv(C_out, 128) * 128
    comptime padded_run = C_out_padded != C_out

    comptime output_simd_width = gcd(
        simd_width_of[output_type, target=get_gpu_target()](), C_out
    )

    var output_elems = batch * D_out * H_out * W_out * C_out
    var accum_total_elems = batch * D_out * H_out * W_out * C_out_padded
    # Per-batch input stride walks the FULL D axis of the user input
    # (D = D_out + Q - 1), not D_out — the q loop slides a D_out window
    # within each batch's D slab.
    var input_slice_elems = D * H * W * C_in
    var accum_slice_elems = D_out * H_out * W_out * C_out_padded

    # --- 1. Allocate fp32 accumulator (zeroed) + reusable bf16 temp. ---
    # Both sized at C_out_padded so the q loop is uniform regardless of
    # whether padded_run is True.
    var accum_fp32_buf = ctx.enqueue_create_buffer[DType.float32](
        accum_total_elems
    )
    accum_fp32_buf.enqueue_fill(Scalar[DType.float32](0.0))
    var accum_fp32_ptr = accum_fp32_buf.unsafe_ptr()

    var temp_bf16_buf = ctx.enqueue_create_buffer[output_type](
        accum_slice_elems
    )
    var temp_bf16_ptr = temp_bf16_buf.unsafe_ptr()

    # --- 1b. Route the filter through a scratch buffer with F=C_out_padded.
    # Using a single buffer for both paths keeps the q loop free of
    # origin-type branching between the user's filter and the scratch
    # buffer. When padded_run=True we zero-pad via the divmod kernel;
    # when False we use a plain indexed copy which is ~10x faster than
    # the divmod variant for the common aligned case.
    var filter_slab_elems = R * S * C_in * C_out_padded
    var padded_filter_buf = ctx.enqueue_create_buffer[filter_type](
        Q * filter_slab_elems
    )
    var filter_base_ptr = padded_filter_buf.unsafe_ptr()
    comptime pad_block = 256
    var pad_total = Q * filter_slab_elems
    var pad_grid = ceildiv(pad_total, pad_block)
    comptime if padded_run:
        ctx.enqueue_function[
            _pad_filter_qrscf_kernel[filter_type],
            _pad_filter_qrscf_kernel[filter_type],
        ](
            filter.ptr,
            filter_base_ptr,
            Q,
            R,
            S,
            C_in,
            C_out,
            C_out_padded,
            grid_dim=pad_grid,
            block_dim=pad_block,
        )
    else:
        ctx.enqueue_function[
            _copy_filter_kernel[filter_type],
            _copy_filter_kernel[filter_type],
        ](
            filter.ptr,
            filter_base_ptr,
            pad_total,
            grid_dim=pad_grid,
            block_dim=pad_block,
        )

    # --- 2. Per-(n, q) conv into temp_bf16, then accumulate into fp32. ---
    comptime accum_block = 256

    for n_batch in range(batch):
        var input_n_offset = n_batch * input_slice_elems
        var accum_n_offset = n_batch * accum_slice_elems
        var accum_n_ptr = accum_fp32_ptr + accum_n_offset

        for q in range(Q):
            var input_q_offset = input_n_offset + q * H * W * C_in
            var input_q_ptr = input.ptr + input_q_offset
            # QRSCF slab q is RSCF at offset q*R*S*C*F_padded — contiguous
            # in the scratch buffer where the extra F columns are zeroed.
            var filter_q_ptr = filter_base_ptr + q * filter_slab_elems

            var act_tt = TileTensor(
                input_q_ptr,
                row_major(Idx(D_out), Idx(H), Idx(W), Idx(C_in)),
            )
            var filter_rscf_tt = TileTensor(
                filter_q_ptr,
                row_major(Idx(R), Idx(S), Idx(C_in), Idx(C_out_padded)),
            )
            var temp_tt = TileTensor(
                temp_bf16_ptr,
                row_major(
                    Idx(D_out), Idx(H_out), Idx(W_out), Idx(C_out_padded)
                ),
            )

            dispatch_sm100_conv2d[
                input_type,
                filter_type,
                output_type,
                filter_is_fcrs=False,
                has_residual=False,
            ](
                act_tt,
                filter_rscf_tt,
                temp_tt,
                symmetric_padding_2d,
                ctx,
                OptionalReg[UnsafePointer[Scalar[output_type], MutAnyOrigin]](
                    None
                ),
                Float32(0.0),
            )

            # Accumulate: accum_fp32_n += temp_bf16.cast[fp32]().
            var accum_grid = ceildiv(
                accum_slice_elems // output_simd_width, accum_block
            )
            comptime accum_kernel = _accum_bf16_to_fp32_kernel[
                output_type, output_simd_width
            ]
            ctx.enqueue_function[accum_kernel, accum_kernel](
                accum_n_ptr,
                temp_bf16_ptr,
                accum_slice_elems,
                grid_dim=accum_grid,
                block_dim=accum_block,
            )

    # --- 3. Final fp32 → bf16 write to user output (fuse epilogue). ---
    # When padded_run=True the accumulator is C_out_padded-wide; use the
    # strided variants so we only write/call epilogue on the first C_out
    # columns per NDHW pixel.
    comptime final_block = 256
    var final_grid = ceildiv(output_elems // output_simd_width, final_block)

    comptime if maybe_epilogue_func:
        comptime epilogue_5d = maybe_epilogue_func.value()
        comptime if padded_run:
            comptime output_kernel = _fp32_strided_to_dtype_epilogue_kernel[
                output_type, C_out, C_out_padded, epilogue_5d, output_simd_width
            ]
            ctx.enqueue_function[output_kernel, output_kernel](
                accum_fp32_ptr,
                batch,
                D_out,
                H_out,
                W_out,
                output_elems,
                grid_dim=final_grid,
                block_dim=final_block,
            )
        else:
            comptime output_kernel = _fp32_to_dtype_epilogue_kernel[
                output_type, C_out, epilogue_5d, output_simd_width
            ]
            ctx.enqueue_function[output_kernel, output_kernel](
                accum_fp32_ptr,
                batch,
                D_out,
                H_out,
                W_out,
                output_elems,
                grid_dim=final_grid,
                block_dim=final_block,
            )
    else:
        comptime if padded_run:
            comptime output_kernel = _fp32_strided_to_dtype_plain_kernel[
                output_type, C_out, C_out_padded, output_simd_width
            ]
            ctx.enqueue_function[output_kernel, output_kernel](
                output.ptr,
                accum_fp32_ptr,
                output_elems,
                grid_dim=final_grid,
                block_dim=final_block,
            )
        else:
            comptime output_kernel = _fp32_to_dtype_plain_kernel[
                output_type, output_simd_width
            ]
            ctx.enqueue_function[output_kernel, output_kernel](
                output.ptr,
                accum_fp32_ptr,
                output_elems,
                grid_dim=final_grid,
                block_dim=final_block,
            )

    return True
