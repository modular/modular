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
"""AMD 4-wave Conv2D fprop with optional residual add.

Mirrors `max/kernels/src/nn/conv/gpu/nvidia/sm100/conv2d.mojo`'s
`conv2d_fprop_with_residual` API so the production-level dispatcher
can swap SM100 (Blackwell) for AMD MI355X (CDNA4) without touching
the call site. Same `Conv2dProblemShape`, same store and compute
epilogue hooks, same `D = Conv(A,B) + beta*C` semantics.

The residual add fires inside `AMD4WaveMatmul.run_conv2d`'s epilogue,
which bulk-prefetches `source` into a per-lane VGPR cluster before
the main loop so the HBM read latency overlaps with the MFMAs. No
extra HBM round-trip, no separate elementwise launch.

When `has_residual=False` (or `beta == 0.0`), the call routes to
`amd_4wave_conv` directly: same code path, no residual cost.

Epilogue ordering matches SM100: `D = lambda(Conv(A,B)) + beta * C`,
i.e. `compute_fn` (pre-residual: bias / ReLU / SiLU / GELU) fires on
the post-cast `c_type` MMA output before the residual FMA, and
`epilogue_fn` (post-residual store) fires after with the fused value.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import align_up, ceildiv

from layout import Idx, TensorEngine, TensorLayout, TileTensor, row_major

from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape


# ----------------------------------------------------------------------
# K-padding kernel: KRSC [C_out, R*S*C_in] -> [C_out, K_padded]
# ----------------------------------------------------------------------


@__name(t"amd_4wave_conv_residual_kpad_{dtype}")
def _kpad_filter_frsc[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    K_real: Int32,
    K_padded: Int32,
    F: Int32,
):
    """Copy `[F, K_real]` -> `[F, K_padded]` with zero-filled trailing K.

    KRSC flattens to `[C_out, R*S*C_in]` row-major. We just need to
    copy each row to a wider stride and zero-pad the tail.
    """
    var _K_real = Int(K_real)
    var _K_padded = Int(K_padded)
    var _F = Int(F)
    var total = _F * _K_padded
    var tid = global_idx.x
    if tid >= total:
        return
    var f = tid // _K_padded
    var k = tid - f * _K_padded
    if k >= _K_real:
        dst_ptr.unsafe_store(tid, Scalar[dtype](0))
        return
    dst_ptr.unsafe_store(
        tid,
        rebind[Scalar[dtype]](src.reshape(row_major(_F, _K_real)).load((f, k))),
    )


@inline(.always)
def amd_4wave_conv_fprop_with_residual[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    *,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
](
    output: TileTensor[
        mut=True, out_type, ...
    ],  # NHWC — D = Conv(A,B) + beta*C
    activation: TileTensor[act_type, ...],  # NHWC — A
    filter: TileTensor[filter_type, ...],  # KRSC — B
    source: TileTensor[out_type, ...],  # NHWC — C (residual)
    beta: Float32,
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Launch AMD 4-wave Conv2D fprop with optional residual add.

    Computes `D = epilogue_fn(compute_fn(Conv(A,B)) + beta*C)` (both
    epilogues optional, see `Parameters`). Mirrors the SM100
    `conv2d_fprop_with_residual` API so the high-level dispatcher can
    swap platforms without changing call sites.

    Supports common UNet / ResNet patterns:
    - Skip connection: `D = Conv(A,B) + C` (beta=1.0, has_residual=True)
    - Residual scaling: `D = Conv(A,B) + 0.5*C` (beta=0.5)
    - Fused bias + residual: `D = (Conv + bias) + C` (`compute_fn` + beta=1)

    Parameters:
        EpilogueFnType: Type of `epilogue_fn`.
        ComputeFnType: Type of `compute_fn`.
        act_type: Input activation dtype. FP8 / BF16 / FP16.
        filter_type: Filter dtype. Must equal `act_type`.
        out_type: Output dtype. BF16 when `act_type` is FP8;
            matches `act_type` otherwise.
        has_epilogue_fn: Whether `epilogue_fn` stores each output chunk
            instead of the kernel.
        has_compute_fn: Whether `compute_fn` transforms each output
            chunk before the residual add.
        has_residual: If True, fuse residual add. If False, `source`
            is ignored and this routes through plain `amd_4wave_conv`.

    Args:
        output: Output tensor `[N, H_out, W_out, C_out]` in NHWC (D).
        activation: Input activation `[N, H, W, C_in]` in NHWC (A).
        filter: Filter weights `[C_out, R, S, C_in]` in KRSC (B).
        source: Residual input `[N, H_out, W_out, C_out]` in NHWC (C).
            Same shape as `output`. Ignored when `has_residual=False`.
        beta: Residual scale factor. If `0.0`, no residual is applied
            (early-out to plain conv).
        problem: Convolution problem shape (`Conv2dProblemShape` from
            `nvidia/sm100/conv_config.mojo`). The shared shape struct
            lets a single call site target either SM100 or MI355X.
        ctx: Device context.
        epilogue_fn: Stores each output chunk at a 2D coordinate
            `(m, n)` *after* the residual add, where
            `m = b*H_out*W_out + h*W_out + w` and `n` is the output
            channel. Used when `has_epilogue_fn` is set.
        compute_fn: Transforms each output chunk *before* the residual
            add. Use for bias / ReLU / SiLU / GELU fusion in the SM100
            ordering. Used when `has_compute_fn` is set.

    Raises:
        Error if device enqueue fails or source shape doesn't match
        output.
    """
    comptime assert (
        act_type == filter_type
    ), "act_type and filter_type must match (a_type == b_type in 4-wave)"

    # Early-out: no residual or beta=0 → route through plain conv.
    # Same code path as `conv2d_fprop` — no overhead from the residual
    # plumbing.
    comptime if not has_residual:
        _launch_plain_conv[
            has_epilogue_fn=has_epilogue_fn, has_compute_fn=has_compute_fn
        ](output, activation, filter, problem, ctx, epilogue_fn, compute_fn)
        return

    if beta == 0.0:
        _launch_plain_conv[
            has_epilogue_fn=has_epilogue_fn, has_compute_fn=has_compute_fn
        ](output, activation, filter, problem, ctx, epilogue_fn, compute_fn)
        return

    # Validate source shape matches output (NHWC == NHWC).
    if Int(source.dim[0]()) != Int(output.dim[0]()):
        raise Error("source batch must match output batch")
    if Int(source.dim[1]()) != Int(output.dim[1]()):
        raise Error("source H must match output H_out")
    if Int(source.dim[2]()) != Int(output.dim[2]()):
        raise Error("source W must match output W_out")
    if Int(source.dim[3]()) != Int(output.dim[3]()):
        raise Error("source channels must match output channels")

    # 2D view of the 4D NHWC output. Row-major NHWC means
    # `output_2d[m, n] == output[b, h, w, n]` with
    # `m = b * H_out * W_out + h * W_out + w`. The same 2D mapping
    # applies to `source` since it shares output's shape and layout.
    # The in-kernel residual path consumes `source_ptr + (m, n)`
    # directly via the implicit row-major mapping — no 4D divmod.
    var M = Int(output.dim[0]()) * Int(output.dim[1]()) * Int(output.dim[2]())
    comptime _C_out = filter.static_shape[0]
    comptime _R = filter.static_shape[1]
    comptime _S = filter.static_shape[2]
    comptime _C_in = filter.static_shape[3]
    comptime K_real = _R * _S * _C_in
    # K-pad to a multiple of `2*BK = 256`. The kernel asserts on this
    # alignment internally; when `K_real` already meets it we skip
    # the alloc + zero-pad entirely.
    comptime K_padded = align_up(K_real, 256)
    comptime _needs_kpad = K_padded != K_real
    var output_2d = output.reshape(row_major(M, Idx[_C_out]))

    # The 2D residual view's row stride is `C_out` (NHWC contiguous).
    var _source_row_stride = _C_out

    # Both epilogues are wired through `amd_4wave_conv` →
    # `AMD4WaveMatmul.run_conv2d`:
    #   - `compute_fn` fires on the MMA output (post-cast to `c_type`)
    #     BEFORE the residual FMA, matching the SM100
    #     `D = lambda(Conv(A,B)) + beta * C` semantics.
    #   - `epilogue_fn` fires at the store site (via
    #     `RegTileEpilogue.store_with_epilogue_fn`) AFTER the residual
    #     FMA, taking responsibility for the final write.

    if problem.stride_h != problem.stride_w or problem.pad_h != problem.pad_w:
        raise Error(
            "asymmetric stride or pad not supported by"
            " amd_4wave_conv_with_residual"
        )

    # K-padded filter buffer (alloc + zero-pad) when needed; otherwise
    # reuse the caller's buffer directly. Then dispatch (stride, pad)
    # to a comptime-specialized kernel launch with `has_residual=True`.
    comptime if _needs_kpad:
        var filter_padded_buf = ctx.enqueue_create_buffer[filter_type](
            _C_out * K_padded
        )
        comptime _kpad_block = 256
        var _kpad_total = _C_out * K_padded
        var _kpad_grid = ceildiv(_kpad_total, _kpad_block)
        comptime kpad_kernel = _kpad_filter_frsc[
            filter_type,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
        ]
        ctx.enqueue_function[kpad_kernel](
            filter.as_imm(),
            filter_padded_buf,
            Int32(K_real),
            Int32(K_padded),
            Int32(_C_out),
            grid_dim=_kpad_grid,
            block_dim=_kpad_block,
        )
        comptime _filter_2d_layout = row_major[_C_out, K_padded]()
        var filter_2d = TileTensor(filter_padded_buf, _filter_2d_layout)

        @inline(.always)
        def _launch_kpad[stride_v: Int, pad_v: Int]() raises {imm}:
            amd_4wave_conv[
                has_residual=True,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                R=_R,
                S=_S,
                stride_h=stride_v,
                stride_w=stride_v,
                pad_h=pad_v,
                pad_w=pad_v,
                C_in=_C_in,
                use_runtime_hw=True,
            ](
                activation,
                filter_2d,
                output_2d,
                ctx,
                source_ptr=ImmPointer[Scalar[out_type], ImmutAnyOrigin](
                    unsafe_from_address=Int(source.ptr)
                ),
                source_row_stride=_source_row_stride,
                beta=beta,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )

        var s = problem.stride_h
        var p = problem.pad_h
        match (s, p):
            case (1, 0):
                _launch_kpad[1, 0]()
            case (1, 1):
                _launch_kpad[1, 1]()
            case (1, 2):
                _launch_kpad[1, 2]()
            case (2, 0):
                _launch_kpad[2, 0]()
            case (2, 1):
                _launch_kpad[2, 1]()
            case (2, 2):
                _launch_kpad[2, 2]()
            case _:
                raise Error(
                    "unsupported (stride, pad) combination for amd_4wave_conv"
                )
    else:
        comptime _filter_2d_layout_no_pad = row_major[_C_out, K_real]()
        var filter_2d_no_pad = filter.reshape(_filter_2d_layout_no_pad)

        @inline(.always)
        def _launch_no_pad[stride_v: Int, pad_v: Int]() raises {imm}:
            amd_4wave_conv[
                has_residual=True,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                R=_R,
                S=_S,
                stride_h=stride_v,
                stride_w=stride_v,
                pad_h=pad_v,
                pad_w=pad_v,
                C_in=_C_in,
                use_runtime_hw=True,
            ](
                activation,
                filter_2d_no_pad,
                output_2d,
                ctx,
                source_ptr=ImmPointer[Scalar[out_type], ImmutAnyOrigin](
                    unsafe_from_address=Int(source.ptr)
                ),
                source_row_stride=_source_row_stride,
                beta=beta,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )

        var s = problem.stride_h
        var p = problem.pad_h
        match (s, p):
            case (1, 0):
                _launch_no_pad[1, 0]()
            case (1, 1):
                _launch_no_pad[1, 1]()
            case (1, 2):
                _launch_no_pad[1, 2]()
            case (2, 0):
                _launch_no_pad[2, 0]()
            case (2, 1):
                _launch_no_pad[2, 1]()
            case (2, 2):
                _launch_no_pad[2, 2]()
            case _:
                raise Error(
                    "unsupported (stride, pad) combination for amd_4wave_conv"
                )


# ----------------------------------------------------------------------
# Internal: plain conv routing (no residual or beta=0)
# ----------------------------------------------------------------------


@inline(.always)
def _launch_plain_conv[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    output: TileTensor[mut=True, out_type, ...],
    activation: TileTensor[act_type, ...],
    filter: TileTensor[filter_type, ...],
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises:
    """Route through `amd_4wave_conv` when residual is disabled."""
    var M = Int(output.dim[0]()) * Int(output.dim[1]()) * Int(output.dim[2]())
    comptime _C_out = filter.static_shape[0]
    comptime _R = filter.static_shape[1]
    comptime _S = filter.static_shape[2]
    comptime _C_in = filter.static_shape[3]
    comptime K_real = _R * _S * _C_in
    comptime K_padded = align_up(K_real, 256)
    comptime _needs_kpad = K_padded != K_real
    var output_2d = output.reshape(row_major(M, Idx[_C_out]))

    if problem.stride_h != problem.stride_w or problem.pad_h != problem.pad_w:
        raise Error(
            "asymmetric stride or pad not supported by"
            " amd_4wave_conv_with_residual"
        )

    # K-padded filter buffer (alloc + zero-pad) when needed; otherwise
    # reuse the caller's buffer directly. Dispatch (stride, pad) inside
    # the branch so `filter_2d` (whose layout differs per branch) stays
    # in scope across the kernel launch.
    comptime if _needs_kpad:
        var filter_padded_buf = ctx.enqueue_create_buffer[filter_type](
            _C_out * K_padded
        )
        comptime _kpad_block = 256
        var _kpad_total = _C_out * K_padded
        var _kpad_grid = ceildiv(_kpad_total, _kpad_block)
        comptime kpad_kernel = _kpad_filter_frsc[
            filter_type,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
        ]
        ctx.enqueue_function[kpad_kernel](
            filter.as_imm(),
            filter_padded_buf,
            Int32(K_real),
            Int32(K_padded),
            Int32(_C_out),
            grid_dim=_kpad_grid,
            block_dim=_kpad_block,
        )
        comptime _filter_2d_layout = row_major[_C_out, K_padded]()
        var filter_2d = TileTensor(filter_padded_buf, _filter_2d_layout)

        @inline(.always)
        def _launch_kpad[stride_v: Int, pad_v: Int]() raises {imm}:
            amd_4wave_conv[
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                R=_R,
                S=_S,
                stride_h=stride_v,
                stride_w=stride_v,
                pad_h=pad_v,
                pad_w=pad_v,
                C_in=_C_in,
                use_runtime_hw=True,
            ](
                activation,
                filter_2d,
                output_2d,
                ctx,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )

        var s = problem.stride_h
        var p = problem.pad_h
        match (s, p):
            case (1, 0):
                _launch_kpad[1, 0]()
            case (1, 1):
                _launch_kpad[1, 1]()
            case (1, 2):
                _launch_kpad[1, 2]()
            case (2, 0):
                _launch_kpad[2, 0]()
            case (2, 1):
                _launch_kpad[2, 1]()
            case (2, 2):
                _launch_kpad[2, 2]()
            case _:
                raise Error(
                    "unsupported (stride, pad) combination for amd_4wave_conv"
                )
    else:
        comptime _filter_2d_layout_no_pad = row_major[_C_out, K_real]()
        var filter_2d_no_pad = filter.reshape(_filter_2d_layout_no_pad)

        @inline(.always)
        def _launch_no_pad[stride_v: Int, pad_v: Int]() raises {imm}:
            amd_4wave_conv[
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                R=_R,
                S=_S,
                stride_h=stride_v,
                stride_w=stride_v,
                pad_h=pad_v,
                pad_w=pad_v,
                C_in=_C_in,
                use_runtime_hw=True,
            ](
                activation,
                filter_2d_no_pad,
                output_2d,
                ctx,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )

        var s = problem.stride_h
        var p = problem.pad_h
        match (s, p):
            case (1, 0):
                _launch_no_pad[1, 0]()
            case (1, 1):
                _launch_no_pad[1, 1]()
            case (1, 2):
                _launch_no_pad[1, 2]()
            case (2, 0):
                _launch_no_pad[2, 0]()
            case (2, 1):
                _launch_no_pad[2, 1]()
            case (2, 2):
                _launch_no_pad[2, 2]()
            case _:
                raise Error(
                    "unsupported (stride, pad) combination for amd_4wave_conv"
                )
