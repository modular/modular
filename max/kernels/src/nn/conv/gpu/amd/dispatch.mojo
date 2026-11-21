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
"""AMD MI355X (CDNA4, gfx950) conv2d dispatch to `amd_4wave_conv`.

Returns True when the input/filter/output shapes + runtime stride / pad
/ dilation are handled by the 4-wave conv kernel; False to fall back to
the caller's MIOpen path. Acceptance rules:

  - Hardware: MI355X (gfx950) only (the 4-wave kernel inherits the
    chiplet/L2 swizzle and MFMA shapes specific to CDNA4).
  - Input dtype: float8_e4m3fn, bfloat16, or float16. Output dtype:
    bfloat16 for FP8, otherwise tracks input.
  - All input / filter / output spatial shapes must be **static**
    (TileTensor `static_shape[i] >= 0`). Dynamic-shape conv shapes
    fall through to MIOpen.
  - `num_groups == 1` (4-wave conv is single-group).
  - `dilation == (1, 1)`.
  - `stride ∈ {(1, 1), (2, 2)}`, with `stride[0] == stride[1]`.
  - `symmetric_padding ∈ {(0, 0), (1, 1), (2, 2)}`, square pad only.

When accepted, the dispatcher:

  1. Allocates a K-padded FRSC filter buffer (zero-filled trailing
     K columns when `R*S*C_in` isn't a multiple of `2*BK = 256`).
  2. Transposes the caller's filter (FCRS or RSCF) into FRSC.
  3. Comptime-materializes (stride, pad) and calls `amd_4wave_conv`
     with the appropriate kernel template parameters.

Two geometries run on wave-specialized kernels first (`ring_conv2d_arm`): a
bf16 96 -> 96 channel 3x3 conv over one 30x30 image (which the 4-wave
conv declines) and over one 240x416 image, both with an FCRS filter.

Mirrors the structure of `nn.conv.gpu.amd.rdna.dispatch` and
`nn.conv.gpu.nvidia.sm100.dispatch`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, GridBlockLaunchConfig
from std.math import align_up, ceildiv
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import _accelerator_arch

from layout import Coord, TensorEngine, TensorLayout, TileTensor, row_major

from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv
from nn.conv.gpu.amd.conv2d_fcrs_fprop import (
    Conv2dFcrsConfig,
    launch_conv2d_fcrs_fprop,
)
from nn.conv.gpu.amd.conv2d_staged_fprop import (
    Conv2dStagedConfig,
    conv2d_staged_fprop,
)


# ---------------------------------------------------------------------------
# Filter transposes to FRSC, with K-padding zero-fill.
# ---------------------------------------------------------------------------


@__name(t"amd_4wave_transpose_rscf_to_frsc_kpad_{dtype}")
def _transpose_rscf_to_frsc_kpad[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    R: Int,
    S: Int,
    C: Int,
    F: Int,
    K_padded: Int,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """GPU kernel: filter RSCF `[R*S*C, F]` -> FRSC `[F, K_padded]`.

    `K_padded >= R*S*C`; padded trailing columns are zero-filled. Lanes run
    along `F`, so each source row is read coalesced; every thread gathers one
    16-byte K-group for its filter and writes it with a single store.
    """
    comptime W = 16 // size_of[dtype]()
    comptime K_real = R * S * C
    comptime assert K_padded % W == 0, "K_padded must hold whole 16-byte groups"
    var tid = global_idx.x
    if tid >= F * (K_padded // W):
        return
    var g = tid // F
    var f = tid - g * F
    var k0 = g * W
    var v = SIMD[dtype, W](0)
    comptime if K_real % W == 0:
        # A group is all real or all padding, so its loads are unconditional
        # and issue back to back.
        if k0 < K_real:
            comptime for j in range(W):
                var rs, c = divmod(k0 + j, C)
                var r, s = divmod(rs, S)
                v[j] = src.load((r, s, c, f))[0]
    else:
        comptime for j in range(W):
            if k0 + j < K_real:
                var rs, c = divmod(k0 + j, C)
                var r, s = divmod(rs, S)
                v[j] = src.load((r, s, c, f))[0]
    dst_ptr.unsafe_store[alignment=align_of[SIMD[dtype, W]]()](
        f * K_padded + k0, v
    )


@__name(t"amd_4wave_transpose_fcrs_to_frsc_kpad_{dtype}")
def _transpose_fcrs_to_frsc_kpad[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    DstEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: TileTensor[dtype, DstLayoutType, dst_origin, Engine=DstEngine],
    F: Int32,
    C: Int32,
    R: Int32,
    S: Int32,
    K_padded: Int32,
):
    """GPU kernel: filter FCRS `[F, C, R, S]` -> FRSC `[F, K_padded]`.

    `K_padded >= R*S*C`; padded trailing columns are zero-filled.
    """
    var _F = Int(F)
    var _C = Int(C)
    var _R = Int(R)
    var _S = Int(S)
    var _K_padded = Int(K_padded)
    var K_real = _R * _S * _C
    var total = _F * _K_padded
    var tid = global_idx.x
    if tid >= total:
        return
    var f = tid // _K_padded
    var k = tid - f * _K_padded
    if k >= K_real:
        dst.raw_store(tid, Scalar[dtype](0))
        return
    var r = k // (_S * _C)
    var sc = k - r * (_S * _C)
    var s = sc // _C
    var c = sc - s * _C
    dst.raw_store(tid, src.load((f, c, r, s))[0])


# 96 -> 96 channel 3x3 convs use a BM64/BN128 tile up to this many output rows.
# N = 96 fills 75% of a BN128 tile either way, so the smaller BM buys a finer
# grid (less tail) at a modest L2 cost; past this size the doubled B traffic
# wins back the difference (MI355X, bf16: 1.19x at M = 25k, 1.04x at M = 100k,
# 0.9x at M = 400k).
comptime _C96_BM64_MAX_M = 131072


# ---------------------------------------------------------------------------
# Runtime-HW launcher (separated so the dispatcher's static path
# doesn't pay compile-time for runtime-HW kernel specializations).
# ---------------------------------------------------------------------------


@inline(.always)
def _launch_amd_4wave_conv2d_runtime[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    K_padded: Int,
    block_k_override: Int,
    block_m_override: Int,
    block_n_override: Int,
    has_residual: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    b_rscf: Bool = False,
](
    input: TileTensor[input_type, ...],
    filter_frsc_tt: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: Int,
    pad: Int,
    ctx: DeviceContext,
    source_ptr: ImmPointer[Scalar[output_type], ImmutAnyOrigin] = ImmPointer[
        Scalar[output_type], ImmutAnyOrigin
    ].unsafe_dangling(),
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises -> Bool:
    """Runtime-HW launch for the 4-wave conv dispatcher.

    Materializes `(stride, pad)` to a comptime pair and invokes the
    runtime-HW path of `amd_4wave_conv`. Returns False only for
    unreachable (defensive) (stride, pad) combinations; the outer
    dispatcher's runtime gate handles validity.

    When `has_residual=True`, plumbs `source_ptr` (NHWC-contiguous,
    same shape as output) + `beta` through to `amd_4wave_conv` for
    fused in-kernel `D = Conv + beta * source`. The 2D NHWC view's
    row stride is `C_out` (NHWC contiguous).
    """

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def _launch[stride_v: Int, pad_v: Int]() raises {imm}:
        var _rt_N = Int(input.dim[0]())
        var _rt_H = Int(input.dim[1]())
        var _rt_W = Int(input.dim[2]())
        comptime _eff_R = R
        comptime _eff_S = S
        var _rt_H_out = (_rt_H + 2 * pad_v - _eff_R) // stride_v + 1
        var _rt_W_out = (_rt_W + 2 * pad_v - _eff_S) // stride_v + 1
        var _rt_M_total = _rt_N * _rt_H_out * _rt_W_out
        # Dynamic 2D `[M_total, C_out]` output view.
        var _dyn_out_layout = row_major(Coord(_rt_M_total, C_out))
        var output_2d_tt = output.reshape(_dyn_out_layout)
        comptime _use_384_runtime_rule = (
            block_m_override == 0
            and block_n_override == 0
            and input_type in (DType.bfloat16, DType.float16)
            and C_in == 384
            and C_out == 384
            and R == 3
            and S == 3
            and stride_v == 1
            and pad_v == 1
        )

        comptime _use_c96_runtime_rule = (
            block_m_override == 0
            and block_n_override == 0
            and input_type in (DType.bfloat16, DType.float16)
            and C_in == 96
            and C_out == 96
            and R == 3
            and S == 3
            and stride_v == 1
            and pad_v == 1
        )

        @inline(.always)
        def _run[block_m: Int, block_n: Int]() raises {imm}:
            amd_4wave_conv[
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                block_m_override=block_m,
                block_n_override=block_n,
                block_k_override=block_k_override,
                R=R,
                S=S,
                stride_h=stride_v,
                stride_w=stride_v,
                pad_h=pad_v,
                pad_w=pad_v,
                C_in=C_in,
                use_runtime_hw=True,
                has_residual=has_residual,
                b_rscf=b_rscf,
            ](
                input,
                filter_frsc_tt,
                output_2d_tt,
                ctx,
                source_ptr=source_ptr,
                source_row_stride=C_out,
                beta=beta,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )

        comptime if _use_384_runtime_rule:
            if _rt_M_total <= 4096:
                _run[64, 64]()
            else:
                _run[0, 0]()
        elif _use_c96_runtime_rule:
            if _rt_M_total <= _C96_BM64_MAX_M:
                _run[64, 128]()
            else:
                _run[0, 0]()
        else:
            _run[block_m_override, block_n_override]()

    match (stride, pad):
        case (1, 0):
            _launch[1, 0]()
        case (1, 1):
            _launch[1, 1]()
        case (1, 2):
            _launch[1, 2]()
        case (2, 0):
            _launch[2, 0]()
        case (2, 1):
            _launch[2, 1]()
        case (2, 2):
            _launch[2, 2]()
        case _:
            return False
    return True


# ---------------------------------------------------------------------------
# Wave-specialized kernels for two measured geometries.
# ---------------------------------------------------------------------------

# 30x30, 96 -> 96, bf16, FCRS filter: the 4-wave conv declines (`M % 64 != 0`),
# and this kernel reads the caller's filter in place. Six compute waves beat
# three by 0.04 to 0.12 us.
comptime _FCRS_30X30 = Conv2dFcrsConfig(
    C=96,
    F=96,
    R=3,
    S=3,
    H=30,
    W=30,
    pad=1,
    stride=1,
    NL=8,
    NOW=6,
    element_bytes=2,
)

# 240x416, 96 -> 96, bf16, FCRS filter (the WAN VAE level-3 conv): both
# operands DMA'd to LDS, groups of 13 strips (half an output row), one
# workgroup per two groups.
comptime _STAGED_240X416 = Conv2dStagedConfig(
    C=96,
    F=96,
    R=3,
    S=3,
    H=240,
    W=416,
    pad=1,
    G=13,
    NL=2,
    NLB=1,
    NOW=4,
    depth=2,
    depth_b=2,
    max_grid=240,
    element_bytes=2,
)


comptime RING_NONE = 0
"""`ring_conv2d_arm`: no wave-specialized kernel takes the geometry."""
comptime RING_FCRS = 1
"""`ring_conv2d_arm`: `conv2d_fcrs_fprop` takes it."""
comptime RING_STAGED = 2
"""`ring_conv2d_arm`: `conv2d_staged_fprop` takes it."""


def ring_conv2d_arm(
    dtype: DType,
    filter_is_fcrs: Bool,
    n: Int,
    h: Int,
    w: Int,
    c: Int,
    f: Int,
    r: Int,
    s: Int,
) -> Int:
    """Which wave-specialized kernel takes a 3x3 stride-1 pad-1 geometry.

    The two kernels were measured on one geometry each: a bf16 conv over one
    image with a static FCRS filter, 96 -> 96 channels, of 30x30
    (`RING_FCRS`) or of 240x416 (`RING_STAGED`). A dimension of `-1` is
    dynamic and is never taken.

    Args:
        dtype: Element type of input, filter and output.
        filter_is_fcrs: The filter is FCRS, else RSCF.
        n: Batch.
        h: Input rows.
        w: Input columns.
        c: Input channels.
        f: Output channels.
        r: Filter rows.
        s: Filter columns.

    Returns:
        `RING_FCRS`, `RING_STAGED` or `RING_NONE`.
    """
    if (
        dtype != .bfloat16
        or not filter_is_fcrs
        or n != 1
        or c != 96
        or f != 96
        or r != 3
        or s != 3
    ):
        return RING_NONE
    if h == 30 and w == 30:
        return RING_FCRS
    if h == 240 and w == 416:
        return RING_STAGED
    return RING_NONE


def _dispatch_ring_conv2d[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    has_residual: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    symmetric_padding: Coord[Int, Int],
    num_groups: Int,
    ctx: DeviceContext,
    source_ptr: Optional[MutPointer[Scalar[output_type], MutAnyOrigin]],
    beta: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises -> Bool:
    """Runs the wave-specialized kernel `ring_conv2d_arm` picks for a static
    geometry, stride 1, pad 1, dilation 1 and one group. Returns False for
    everything else so the caller keeps its own path."""
    comptime arm = ring_conv2d_arm(
        input_type,
        filter_is_fcrs,
        input.static_shape[0],
        input.static_shape[1],
        input.static_shape[2],
        input.static_shape[3],
        output.static_shape[3],
        filter.static_shape[2],
        filter.static_shape[3],
    )
    comptime if (
        arm == RING_NONE
        or filter_type != input_type
        or output_type != input_type
        or not input.is_row_major
        or not filter.is_row_major
        or not output.is_row_major
    ):
        return False
    else:
        if (
            num_groups != 1
            or stride[0] != 1
            or stride[1] != 1
            or dilation[0] != 1
            or dilation[1] != 1
            or symmetric_padding[0] != 1
            or symmetric_padding[1] != 1
        ):
            return False
        # The kernels read the residual only under `has_residual`.
        var source_base = MutPointer[
            Scalar[output_type], MutUntrackedOrigin
        ].unsafe_dangling()
        comptime if has_residual:
            if case .Some(residual) = source_ptr:
                source_base = residual.unsafe_origin_cast[MutUntrackedOrigin]()
        comptime if arm == RING_FCRS:
            var output_view = output.reshape(row_major[1, 30, 30, 96]())
            launch_conv2d_fcrs_fprop[
                _FCRS_30X30,
                has_residual=has_residual,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
            ](
                input.reshape(row_major[1, 30, 30, 96]()),
                filter.reshape(row_major[96, 96, 3, 3]()),
                output_view,
                ctx,
                source=TileTensor(source_base, output_view.layout),
                beta=beta,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )
        else:
            var output_view = output.reshape(row_major[1, 240, 416, 96]())
            conv2d_staged_fprop[
                _STAGED_240X416,
                True,
                has_residual=has_residual,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
            ](
                input.reshape(row_major[1, 240, 416, 96]()),
                filter.reshape(row_major[96, 96, 3, 3]()),
                output_view,
                ctx,
                source=TileTensor(source_base, output_view.layout),
                beta=beta,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )
        return True


# ---------------------------------------------------------------------------
# Dispatch entry point.
# ---------------------------------------------------------------------------


def dispatch_amd_4wave_conv2d[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    has_residual: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    block_m_override: Int = 0,
    block_n_override: Int = 0,
    block_k_override: Int = 0,
    use_ring_kernels: Bool = True,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    symmetric_padding: Coord[Int, Int],
    num_groups: Int,
    ctx: DeviceContext,
    source_ptr: Optional[MutPointer[Scalar[output_type], MutAnyOrigin]] = None,
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises -> Bool:
    """Try to dispatch a Conv2D to `amd_4wave_conv` on MI355X.

    Returns True if the convolution was handled; False if the caller
    should fall back (typically to MIOpen). See module docstring for the
    full acceptance criteria.

    When `has_residual=True` and `source_ptr` is set, computes
    `D = Conv(A, B) + beta * source` via the in-kernel fused residual
    path (`amd_4wave_conv[has_residual=True]`). The source pointer is
    expected to point to an NHWC-contiguous buffer with the same shape
    as `output`. When `has_residual=False` (default), the call is
    identical to the no-residual variant: no extra ABI overhead beyond
    the launch packet's 16 bytes (DCE'd source_ptr / stride / beta).

    Parameters:
        EpilogueFnType: Type of `epilogue_fn`.
        ComputeFnType: Type of `compute_fn`.
        input_type: `DType` of the input activation tensor; must be
            `float8_e4m3fn`, `bfloat16`, or `float16`.
        filter_type: `DType` of the filter tensor; must equal
            `input_type`.
        output_type: `DType` of the output tensor; `bfloat16` for FP8
            inputs, otherwise tracks `input_type`.
        filter_is_fcrs: `True` when `filter` is in FCRS layout; `False`
            for RSCF. Selects the transpose path and the R/S dim indices.
        has_residual: `True` to fuse `D = Conv + beta * source` in-kernel
            via `source_ptr` (defaults to `False`).
        has_epilogue_fn: `True` to store each output chunk through
            `epilogue_fn` instead of writing `output` (defaults to
            `False`).
        has_compute_fn: `True` to store `compute_fn` of each output
            chunk, applied before the residual add (defaults to `False`).
        block_m_override: Explicit BM override. `0` preserves dispatch
            selection. A nonzero value must be 64 or 128, requires a nonzero
            `block_n_override`, and retains the `M % 64 == 0` acceptance gate.
        block_n_override: Explicit BN override. `0` preserves dispatch
            selection. A nonzero value must be 64, 128, or 256, requires a
            nonzero `block_m_override`, retains the `M % 64 == 0` acceptance
            gate, and must be at least BM.
        block_k_override: Override for the BK tile size. `0` selects BK128
            except for measured shape families. A nonzero value must satisfy
            the four-wave kernel's BK legality checks.
        use_ring_kernels: Try the wave-specialized kernels for the geometries
            they were measured on (30x30 and 240x416, 96 -> 96) before the
            four-wave conv. Ignored when a tile override is set.

    Args:
        input: Input activation `TileTensor`, rank-4 NHWC with static
            `C_in`; `N`, `H`, `W` may be dynamic.
        filter: Filter `TileTensor`, rank-4 FCRS or RSCF per
            `filter_is_fcrs`, with static `R`, `S`, `C_in`, `C_out`.
        output: Output activation `TileTensor`, rank-4 NHWC, mutable,
            with static `C_out`.
        stride: Convolution stride `[stride_h, stride_w]`; must be
            `(1, 1)` or `(2, 2)` with `stride_h == stride_w`.
        dilation: Dilation `[dh, dw]`; must be `(1, 1)`.
        symmetric_padding: Symmetric padding `[ph, pw]`; must be
            `(0, 0)`, `(1, 1)`, or `(2, 2)` with `ph == pw`.
        num_groups: Number of convolution groups; must be `1`.
        ctx: `DeviceContext` for kernel launches and buffer allocation.
        source_ptr: Optional pointer to an NHWC-contiguous residual
            source buffer with the same shape as `output`; read only
            when `has_residual=True` (defaults to `None`).
        beta: Scale factor for the residual term in
            `D = Conv + beta * source` (defaults to `0.0`).
        epilogue_fn: Stores each output chunk at a 2D coordinate
            `(m, n)`, where `m = b*H_out*W_out + h*W_out + w` and `n` is
            the output channel. Used when `has_epilogue_fn` is set.
        compute_fn: Transforms each output chunk at the same 2D
            coordinate, before the residual add. Used when
            `has_compute_fn` is set.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    comptime assert (block_m_override == 0) == (
        block_n_override == 0
    ), "block_m_override and block_n_override must both be zero or both be set"
    comptime assert block_m_override == 0 or block_m_override in (
        64,
        128,
    ), "block_m_override must be 0, 64, or 128"
    comptime assert block_n_override == 0 or block_n_override in (
        64,
        128,
        256,
    ), "block_n_override must be 0, 64, 128, or 256"
    comptime assert block_m_override == 0 or (
        block_n_override >= block_m_override
    ), "block_n_override must be >= block_m_override"

    # -------- Hardware gate --------
    # Host-side check on the build target's accelerator string. The
    # `_is_amd_mi355x()` helper is GPU-only (false in host code); the
    # accelerator-arch string is "amdgpu:gfx950" on MI355X.
    comptime if _accelerator_arch() != "amdgpu:gfx950":
        return False

    # -------- Dtype gate --------
    comptime if input_type not in (
        DType.float8_e4m3fn,
        DType.bfloat16,
        DType.float16,
    ):
        return False
    comptime assert (
        input_type == filter_type
    ), "amd_4wave conv requires input_type == filter_type"

    # -------- Wave-specialized kernels for two measured geometries --------
    comptime if (
        use_ring_kernels
        and block_m_override == 0
        and block_n_override == 0
        and block_k_override == 0
    ):
        if _dispatch_ring_conv2d[
            input_type,
            filter_type,
            output_type,
            filter_is_fcrs,
            has_residual,
            has_epilogue_fn,
            has_compute_fn,
        ](
            input,
            filter,
            output,
            stride,
            dilation,
            symmetric_padding,
            num_groups,
            ctx,
            source_ptr,
            beta,
            epilogue_fn,
            compute_fn,
        ):
            return True

    # -------- Static-shape gate --------
    # Required comptime: C_in, C_out, R, S, stride, dilation, pad — the
    # 4-wave conv kernel uses these in MMA shape selection and the
    # `(kh, kw, c)` substrip decomposition. N_batch, H, W may be
    # dynamic (e.g. FLUX VAE with symbolic image resolution); the
    # kernel then routes through the `use_runtime_hw=True` path which
    # reads them from input.dim() at runtime. C_in / C_out / R / S
    # being dynamic falls back to MIOpen.
    comptime _C_in = input.static_shape[3]
    comptime _C_out = output.static_shape[3]
    comptime _N_static = input.static_shape[0]
    comptime _H_static = input.static_shape[1]
    comptime _W_static = input.static_shape[2]
    comptime _all_hw_static = (
        _N_static >= 0 and _H_static >= 0 and _W_static >= 0
    )
    # Filter R, S come from different filter dims based on layout.
    comptime _R = (
        filter.static_shape[2] if filter_is_fcrs else filter.static_shape[0]
    )
    comptime _S = (
        filter.static_shape[3] if filter_is_fcrs else filter.static_shape[1]
    )
    # The measured FP16 BK64 choices are limited to these two geometries.
    # Explicit tile overrides, residuals, and compute callbacks stay unchanged.
    comptime _bk64_96 = (
        _C_out == 96
        and (_H_static < 0 or _H_static == 240)
        and (_W_static < 0 or _W_static == 416)
    )
    comptime _bk64_192 = (
        _C_out == 192
        and (_H_static < 0 or _H_static == 120)
        and (_W_static < 0 or _W_static == 208)
    )
    comptime if (
        input_type == .float16
        and filter_type == .float16
        and output_type == .float16
        and not has_residual
        and not has_compute_fn
        and block_m_override == 0
        and block_n_override == 0
        and block_k_override == 0
        and _C_in == 192
        and _R == 3
        and _S == 3
        and (_N_static < 0 or _N_static == 1)
        and (_bk64_96 or _bk64_192)
    ):
        comptime _target_h = 240 if _bk64_96 else 120
        comptime _target_w = 416 if _bk64_96 else 208
        if (
            Int(input.dim[0]()) == 1
            and Int(input.dim[1]()) == _target_h
            and Int(input.dim[2]()) == _target_w
            and stride[0] == 1
            and stride[1] == 1
            and dilation[0] == 1
            and dilation[1] == 1
            and symmetric_padding[0] == 1
            and symmetric_padding[1] == 1
            and num_groups == 1
        ):
            return dispatch_amd_4wave_conv2d[
                input_type,
                filter_type,
                output_type,
                filter_is_fcrs,
                has_residual=has_residual,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                block_m_override=block_m_override,
                block_n_override=block_n_override,
                block_k_override=64,
            ](
                input,
                filter,
                output,
                stride,
                dilation,
                symmetric_padding,
                num_groups,
                ctx,
                source_ptr=source_ptr,
                beta=beta,
                epilogue_fn=epilogue_fn,
                compute_fn=compute_fn,
            )

    # `C_in` lower bound (per-lane substrip soundness): the per-lane
    # substrip loader code path emits one `buffer_load_*_lds` per lane
    # of `load_width = simd_width_of[dtype]` contiguous NHWC elements,
    # under the assumption that all `load_width` elements live in the
    # same `(kh, kw)` substrip. When `C_in < load_width` (e.g. FLUX.2
    # VAE stem: C_in=3 with BF16 load_width=8), a lane's 8-element
    # load crosses pixel boundaries — so the input data landing at
    # LDS positions `(k_lane+1 .. k_lane+load_width-1)` does NOT
    # correspond to the `(kh, kw, c)` the schedule assigns those
    # slots, and the MMA reduces against mis-mapped activations.
    # The filter zero-padding only protects positions `k >= K_real`,
    # not in-range positions whose loaded data is from the wrong
    # pixel. Until the per-lane loader is rewritten to align loads
    # to substrip boundaries, gate this case off so the caller falls
    # back to MIOpen.
    comptime _simd_w = simd_width_of[input_type]()
    comptime _shapes_ok = (
        _C_in >= _simd_w and _C_out >= 64 and _R > 0 and _S > 0
    )
    comptime if _shapes_ok:
        # -------- Runtime constraints --------
        if num_groups != 1:
            return False
        if dilation[0] != 1 or dilation[1] != 1:
            return False
        if stride[0] != stride[1]:
            return False
        if symmetric_padding[0] != symmetric_padding[1]:
            return False
        if stride[0] != 1 and stride[0] != 2:
            return False
        var pad = symmetric_padding[0]
        if pad != 0 and pad != 1 and pad != 2:
            return False

        # Use BK64 only for measured 3x3 families. Unmeasured shapes retain
        # the kernel's BK128 default.
        comptime _BK_heuristic = (
            64 if (
                (
                    _C_in == 96
                    and _C_out == 96
                    and input_type in (DType.bfloat16, DType.float16)
                )
                or (
                    _C_in == 192
                    and _C_out in (96, 192)
                    and input_type == .bfloat16
                )
                or (
                    _C_in == 384
                    and _C_out == 384
                    and input_type in (DType.bfloat16, DType.float16)
                )
            )
            and _R == 3
            and _S == 3 else 128
        )
        comptime _BK_chosen = (
            block_k_override if block_k_override > 0 else _BK_heuristic
        )

        @inline(.always)
        def _static_use_384[stride_v: Int, pad_v: Int]() -> Bool:
            comptime h_out = (_H_static + 2 * pad_v - _R) // stride_v + 1
            comptime w_out = (_W_static + 2 * pad_v - _S) // stride_v + 1
            return (
                block_m_override == 0
                and block_n_override == 0
                and input_type in (DType.bfloat16, DType.float16)
                and _C_in == 384
                and _C_out == 384
                and _R == 3
                and _S == 3
                and stride_v == 1
                and pad_v == 1
                and _N_static * h_out * w_out <= 4096
            )

        @inline(.always)
        def _static_use_c96[stride_v: Int, pad_v: Int]() -> Bool:
            comptime h_out = (_H_static + 2 * pad_v - _R) // stride_v + 1
            comptime w_out = (_W_static + 2 * pad_v - _S) // stride_v + 1
            return (
                block_m_override == 0
                and block_n_override == 0
                and input_type in (DType.bfloat16, DType.float16)
                and _C_in == 96
                and _C_out == 96
                and _R == 3
                and _S == 3
                and stride_v == 1
                and pad_v == 1
                and _N_static * h_out * w_out <= _C96_BM64_MAX_M
            )

        @inline(.always)
        def _static_eligible[stride_v: Int, pad_v: Int]() -> Bool:
            comptime h_out = (_H_static + 2 * pad_v - _R) // stride_v + 1
            comptime w_out = (_W_static + 2 * pad_v - _S) // stride_v + 1
            comptime m_total = _N_static * h_out * w_out
            comptime use_384_rule = _static_use_384[stride_v, pad_v]()
            return (
                h_out >= 1
                and w_out >= 1
                and m_total >= 64
                and (m_total % 64 == 0 or use_384_rule)
            )

        comptime if _all_hw_static:
            match (stride[0], pad):
                case (1, 0):
                    if not _static_eligible[1, 0]():
                        return False
                case (1, 1):
                    if not _static_eligible[1, 1]():
                        return False
                case (1, 2):
                    if not _static_eligible[1, 2]():
                        return False
                case (2, 0):
                    if not _static_eligible[2, 0]():
                        return False
                case (2, 1):
                    if not _static_eligible[2, 1]():
                        return False
                case (2, 2):
                    if not _static_eligible[2, 2]():
                        return False
                case _:
                    return False

        # -------- Filter repack: caller layout -> FRSC, K-padded -----
        comptime _K_real = _R * _S * _C_in
        # The four-wave scheduler consumes two BK tiles per K stage. Aligning
        # to that exact quantum avoids the fixed BK=128 padding tax while
        # retaining the existing zero-filled filter-tail contract.
        comptime _K_padded = align_up(_K_real, 2 * _BK_chosen)

        @inline(.always)
        @__allow_legacy_any_origin_fields
        def _launch_all[
            b_rscf: Bool
        ](b_tt: TileTensor[filter_type, ...]) raises {imm} -> Bool:
            # Residual placeholder: when has_residual=False, source_ptr may
            # be None or unused. We materialize a dangling placeholder for
            # the kernel arg (which only reads it when has_residual=True at
            # comptime, so the value is never dereferenced in the
            # has_residual=False path).
            var _src_ptr_for_kernel = ImmPointer[
                Scalar[output_type], ImmutAnyOrigin
            ].unsafe_dangling()
            comptime if has_residual:
                # The caller must supply source_ptr; the kernel only reads it.
                if case .Some(source_ptr) = source_ptr:
                    _src_ptr_for_kernel = ImmPointer[
                        Scalar[output_type], ImmutAnyOrigin
                    ](unsafe_from_address=Int(source_ptr))

            # -------- Static vs runtime-HW dispatch --------------------
            comptime if _all_hw_static:

                @inline(.always)
                @__allow_legacy_any_origin_fields
                def _launch_static[
                    stride_v: Int, pad_v: Int
                ]() raises {imm} -> Bool:
                    comptime _eff_R = _R
                    comptime _eff_S = _S
                    comptime _H_out_v = (
                        _H_static + 2 * pad_v - _eff_R
                    ) // stride_v + 1
                    comptime _W_out_v = (
                        _W_static + 2 * pad_v - _eff_S
                    ) // stride_v + 1
                    comptime _M_total_v = (_N_static * _H_out_v * _W_out_v)
                    comptime _use_384_rule = _static_use_384[stride_v, pad_v]()
                    comptime _use_c96_rule = _static_use_c96[stride_v, pad_v]()
                    comptime if not _static_eligible[stride_v, pad_v]():
                        return False
                    comptime _nhwc_in_layout = row_major[
                        _N_static, _H_static, _W_static, _C_in
                    ]()
                    comptime _output_2d_layout = row_major[_M_total_v, _C_out]()
                    var input_nhwc_tt = input.reshape(_nhwc_in_layout)
                    var output_2d_tt = output.reshape(_output_2d_layout)
                    amd_4wave_conv[
                        has_epilogue_fn=has_epilogue_fn,
                        has_compute_fn=has_compute_fn,
                        block_k_override=_BK_chosen,
                        block_m_override=(
                            64 if (
                                _use_384_rule or _use_c96_rule
                            ) else block_m_override
                        ),
                        block_n_override=(
                            64 if _use_384_rule else (
                                128 if _use_c96_rule else block_n_override
                            )
                        ),
                        H=_H_static,
                        W=_W_static,
                        H_out=_H_out_v,
                        W_out=_W_out_v,
                        R=_R,
                        S=_S,
                        stride_h=stride_v,
                        stride_w=stride_v,
                        pad_h=pad_v,
                        pad_w=pad_v,
                        C_in=_C_in,
                        has_residual=has_residual,
                        b_rscf=b_rscf,
                    ](
                        input_nhwc_tt,
                        b_tt,
                        output_2d_tt,
                        ctx,
                        source_ptr=_src_ptr_for_kernel,
                        source_row_stride=_C_out,
                        beta=beta,
                        epilogue_fn=epilogue_fn,
                        compute_fn=compute_fn,
                    )
                    return True

                match (stride[0], pad):
                    case (1, 0):
                        if not _launch_static[1, 0]():
                            return False
                    case (1, 1):
                        if not _launch_static[1, 1]():
                            return False
                    case (1, 2):
                        if not _launch_static[1, 2]():
                            return False
                    case (2, 0):
                        if not _launch_static[2, 0]():
                            return False
                    case (2, 1):
                        if not _launch_static[2, 1]():
                            return False
                    case (2, 2):
                        if not _launch_static[2, 2]():
                            return False
                    case _:
                        return False
            else:
                if not _launch_amd_4wave_conv2d_runtime[
                    C_in=_C_in,
                    C_out=_C_out,
                    R=_R,
                    S=_S,
                    K_padded=_K_padded,
                    b_rscf=b_rscf,
                    block_m_override=block_m_override,
                    block_n_override=block_n_override,
                    block_k_override=_BK_chosen,
                    has_residual=has_residual,
                    has_epilogue_fn=has_epilogue_fn,
                    has_compute_fn=has_compute_fn,
                ](
                    input,
                    b_tt,
                    output,
                    stride[0],
                    pad,
                    ctx,
                    source_ptr=_src_ptr_for_kernel,
                    beta=beta,
                    epilogue_fn=epilogue_fn,
                    compute_fn=compute_fn,
                ):
                    return False
            return True

        # The filter is consumed as stored (RSCF, row-major) when the kernel
        # can transpose it on the LDS read; that drops the repack launch.
        comptime _use_rscf_direct = (
            not filter_is_fcrs
            and filter.is_row_major
            and input_type in (DType.bfloat16, DType.float16)
            and _C_out % 8 == 0
            and _BK_chosen == 64
        )
        comptime if _use_rscf_direct:
            # RSCF is `[R, S, C, F]` row-major, i.e. `[K_real, F]`.
            comptime assert filter.is_row_major
            return _launch_all[True](
                filter.reshape(row_major[_K_real, _C_out]())
            )
        else:
            var filter_frsc_buf = ctx.enqueue_create_buffer[filter_type](
                _C_out * _K_padded
            )
            var filter_frsc_tt = TileTensor(
                filter_frsc_buf, row_major[_C_out, _K_padded]()
            )
            comptime _transpose_block = 256
            comptime if filter_is_fcrs:
                var transpose_total = _C_out * _K_padded
                var transpose_grid = ceildiv(transpose_total, _transpose_block)
                comptime transpose_kernel = _transpose_fcrs_to_frsc_kpad[
                    filter_type,
                    filter.LayoutType,
                    ImmOrigin(filter.origin),
                    filter.Engine,
                    filter_frsc_tt.LayoutType,
                    filter_frsc_tt.origin,
                    filter_frsc_tt.Engine,
                ]
                ctx.enqueue_function[transpose_kernel](
                    filter.as_imm(),
                    filter_frsc_tt,
                    Int32(_C_out),
                    Int32(_C_in),
                    Int32(_R),
                    Int32(_S),
                    Int32(_K_padded),
                    grid_dim=transpose_grid,
                    block_dim=_transpose_block,
                )
            else:
                comptime transpose_kernel = _transpose_rscf_to_frsc_kpad[
                    filter_type,
                    filter.LayoutType,
                    ImmOrigin(filter.origin),
                    filter.Engine,
                    _R,
                    _S,
                    _C_in,
                    _C_out,
                    _K_padded,
                ]
                comptime _groups = _K_padded // (16 // size_of[filter_type]())
                ctx.enqueue_function[transpose_kernel](
                    filter.as_imm(),
                    filter_frsc_buf.unsafe_ptr(),
                    grid_dim=ceildiv(_C_out * _groups, _transpose_block),
                    block_dim=_transpose_block,
                )

            return _launch_all[False](filter_frsc_tt)
    else:
        return False
