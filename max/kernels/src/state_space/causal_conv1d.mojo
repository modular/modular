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
"""Core Causal Conv1D Kernel Implementations.

This module provides CPU and GPU kernel implementations for causal 1D convolution,
supporting both channel-first and channel-last memory layouts.

Causal Convolution:
    In causal convolution, the output at position `i` depends only on inputs at
    positions `[i - width + 1, i]`. This ensures no information leakage from
    future positions, making it suitable for autoregressive sequence modeling.

    Mathematical form for width=4:
        out[i] = sum(x[i-3:i+1] * w[0:4]) + bias  (with boundary handling)

Kernel inventory:

    1. Forward kernels:
        - `causal_conv1d_fwd_cpu[has_bias, has_seq_idx]` — single stride-driven
          CPU core. Channel-first vs channel-last is expressed purely through the
          stride values supplied by the caller, so one function serves both.
        - `causal_conv1d_channel_first_fwd_gpu[…]` — GPU kernel for (B, C, L);
          one channel per block-row, L-contiguous coalescing. Dense prefill hits
          a vectorized fast path (one width-kNElts vector load per thread + a
          short L2-cached halo, fully comptime-unrolled); seq_idx / non-contiguous
          / ragged-tail inputs fall back to a width-generic scalar accumulation.
        - `causal_conv1d_channel_last_fwd_gpu[…]` — GPU kernel for (B, L, C);
          threads map to channels (a block owns kNThreads contiguous channels),
          so per-position loads coalesce across the warp along the contiguous C
          axis. Each thread does a sliding scan over kNElts positions reusing the
          kWidth-tap window; seq_idx falls back to a per-tap scan.

        Every forward kernel is parameterized by whether a bias is present and
        whether a `seq_idx` packed-sequence mask is present, instead of carrying
        a separate hand-copied function per combination. On the CPU these are
        compile-time `Bool` parameters; on the GPU they are runtime `Int8`
        arguments (mirroring the idiom in `varlen_causal_conv1d.mojo`).

        Supported widths: 1, 2, 3, 4.

    2. Update kernels (for autoregressive decode):
        - `causal_conv1d_update_cpu[has_bias]`
        - `causal_conv1d_update_gpu[…]` (runtime `has_bias: Int8`)

        Incremental update operations that maintain conv state for efficient
        autoregressive token generation.

Memory Layouts:
    - Channel-first (B, C, L): Standard layout, contiguous channels per position.
    - Channel-last (B, L, C): Contiguous positions per channel, used by some frameworks.

seq_idx (packed sequences):
    `seq_idx` is a per-(batch, position) tag of shape (B, L) marking which logical
    sub-sequence each position belongs to inside a packed batch. When present, a
    convolution tap only contributes if its input position shares the same
    `seq_idx` as the output position, so the kernel never reads across a
    sub-sequence boundary.

GPU Optimization Parameters:
    - kNElts=4: sequence positions per thread; also the fast-path vector width
      (float4 for FP32), the lever that makes channel-first beat
      Dao-AILab/causal-conv1d at mamba prefill dims on GB10.
    - kNThreads=64: threads per block; 64*kNElts == 256 keeps every thread busy
      at the common L=256 (vs the half-idle block kNThreads=128 leaves).
    - block_dim.y (optional): channels folded per block to add warps/block; the
      channel-first op launches 1D (block_dim.y == 1), which profiled fastest.

Activation Support:
    - None: Direct convolution output
    - SiLU: Sigmoid Linear Unit activation (x * sigmoid(x))
"""

from std.math import exp
from std.sys.info import align_of

from std.algorithm import sync_parallelize
from std.gpu.host import DeviceContext
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from layout import TensorLayout, TileTensor


# ===----------------------------------------------------------------------=== #
# Activation Functions
# ===----------------------------------------------------------------------=== #


def silu[
    dtype: DType, width: SIMDSize
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Sigmoid Linear Unit (SiLU) activation function.

    Computes x * sigmoid(x) = x / (1 + exp(-x)).

    Args:
        x: Input SIMD vector.

    Returns:
        SiLU activation applied element-wise.

    Constraints:
        The dtype must be a floating-point type.
    """
    # Branchless and width-generic. For very negative x, exp(-x) -> +inf and
    # x / (1 + inf) -> 0, which is exactly the small-x clamp, so no explicit
    # guard is needed (and a scalar `if` would not vectorize).
    return x / (1 + exp(-x))


@always_inline
def apply_silu[
    output_dtype: DType
](val: Scalar[output_dtype], active: Bool) -> Scalar[output_dtype]:
    """Apply SiLU when `active`, handling integer outputs via float32.

    Centralizes the activation branch that every conv1d kernel shares: when the
    output dtype is floating point SiLU is applied directly, otherwise the value
    is promoted to float32, activated, and cast back.
    """
    if not active:
        return val
    comptime if output_dtype.is_floating_point():
        return silu(val)
    else:
        return silu(val.cast[DType.float32]()).cast[output_dtype]()


# ===----------------------------------------------------------------------=== #
# CPU Implementation (layout-agnostic, stride-driven)
# ===----------------------------------------------------------------------=== #


def causal_conv1d_fwd_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    has_bias: Bool,
    has_seq_idx: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[
        mut=False, x_dtype, ...
    ],  # (B, C, L) or (B, L, C) via strides
    weight: TileTensor[mut=False, weight_dtype, ...],  # (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # same layout as x
    bias: TileTensor[
        mut=False, bias_dtype, ...
    ],  # (C,) — ignored unless has_bias
    seq_idx: TileTensor[
        mut=False, seq_idx_dtype, ...
    ],  # (B, L) — ignored unless has_seq_idx
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    bias_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    silu_activation: Bool,
    ctx: Optional[DeviceContext] = None,
):
    """Stride-driven CPU causal conv1d for both channel-first and channel-last.

    Layout is encoded entirely in the stride arguments: channel-first passes
    `x_c_stride = seqlen`, `x_l_stride = 1`; channel-last passes `x_c_stride = 1`,
    `x_l_stride = dim`. The weight tensor is always (C, W). Parallelizes across
    `batch * dim` and pre-loads the per-channel weights into registers.

    Parameters:
        x_dtype: Element type of the input.
        weight_dtype: Element type of the weights.
        output_dtype: Element type of the output.
        bias_dtype: Element type of the bias.
        seq_idx_dtype: Element type of the seq_idx mask.
        has_bias: Whether to add the per-channel bias.
        has_seq_idx: Whether to apply packed-sequence masking.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width.
        x: Input tensor.
        weight: Weight tensor of shape (C, W).
        output: Output tensor (same layout as x).
        bias: Bias tensor of shape (C,); only read when `has_bias`.
        seq_idx: Packed-sequence tags of shape (B, L); only read when `has_seq_idx`.
        x_batch_stride: Stride for the batch dimension of the input tensor.
        x_c_stride: Stride for the channel dimension of the input tensor.
        x_l_stride: Stride for the sequence dimension of the input tensor.
        weight_c_stride: Stride for the channel dimension of the weight tensor.
        weight_width_stride: Stride for the width dimension of the weight tensor.
        out_batch_stride: Stride for the batch dimension of the output tensor.
        out_c_stride: Stride for the channel dimension of the output tensor.
        out_l_stride: Stride for the sequence dimension of the output tensor.
        bias_stride: Stride for the bias tensor.
        seq_idx_batch_stride: Stride for the batch dimension of seq_idx.
        seq_idx_l_stride: Stride for the sequence dimension of seq_idx.
        silu_activation: Whether to apply SiLU activation.
        ctx: Optional context to parallelize the work on.
    """
    var width_minus_1: Int = width - 1
    var total_bc = batch * dim

    @parameter
    def process_bc(bc_idx: Int):
        var b, c = divmod(bc_idx, dim)
        if b >= batch or c >= dim:
            return

        var weight_c_base_offset = UInt32(c) * weight_c_stride

        var cur_bias: Scalar[output_dtype] = 0
        comptime if has_bias:
            cur_bias = Scalar[output_dtype](
                bias.raw_load(UInt32(c) * bias_stride)
            )

        # Pre-load weights for this channel to reduce memory access.
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0
        if width >= 1:
            w0 = Scalar[weight_dtype](weight.raw_load(weight_c_base_offset))
        if width >= 2:
            w1 = Scalar[weight_dtype](
                weight.raw_load(weight_c_base_offset + weight_width_stride)
            )
        if width >= 3:
            w2 = Scalar[weight_dtype](
                weight.raw_load(weight_c_base_offset + 2 * weight_width_stride)
            )
        if width >= 4:
            w3 = Scalar[weight_dtype](
                weight.raw_load(weight_c_base_offset + 3 * weight_width_stride)
            )

        var x_base = UInt32(b) * x_batch_stride + UInt32(c) * x_c_stride
        var out_base = UInt32(b) * out_batch_stride + UInt32(c) * out_c_stride
        var seq_base = UInt32(b) * seq_idx_batch_stride

        for l in range(seqlen):
            var cur_seq_idx: Int32 = 0
            comptime if has_seq_idx:
                cur_seq_idx = Int32(
                    seq_idx.raw_load(seq_base + UInt32(l) * seq_idx_l_stride)
                )

            # Accumulate in float32 (matches the GPU fast path and Dao's
            # reference); a no-op for fp32 output, and keeps bf16/fp16 results
            # high-precision so they round to the same value the GPU produces.
            var conv_sum: Scalar[DType.float32] = Scalar[DType.float32](
                cur_bias
            )
            for w in range(width):
                var input_l: Int = l - (width_minus_1 - w)
                if input_l >= 0:
                    var include: Bool = True
                    comptime if has_seq_idx:
                        var in_seq = Int32(
                            seq_idx.raw_load(
                                seq_base + UInt32(input_l) * seq_idx_l_stride
                            )
                        )
                        if in_seq != cur_seq_idx:
                            include = False
                    if include:
                        var x_offset = x_base + UInt32(input_l) * x_l_stride
                        var input_val: Scalar[x_dtype] = x.raw_load(x_offset)
                        var weight_val: Scalar[
                            weight_dtype
                        ] = w0 if w == 0 else (
                            w1 if w == 1 else (w2 if w == 2 else w3)
                        )
                        conv_sum = conv_sum + Scalar[DType.float32](
                            input_val
                        ) * Scalar[DType.float32](weight_val)

            var out_offset = out_base + UInt32(l) * out_l_stride
            var activated = apply_silu(conv_sum, silu_activation)
            output.raw_store(out_offset, Scalar[output_dtype](activated))

    sync_parallelize[process_bc](total_bc, ctx)


# ===----------------------------------------------------------------------=== #
# GPU Implementations
# ===----------------------------------------------------------------------=== #


def causal_conv1d_channel_first_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[x_dtype, x_LT, ImmutUntrackedOrigin],  # (B, C, L)
    weight: TileTensor[weight_dtype, weight_LT, ImmutUntrackedOrigin],  # (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # (B, C, L)
    bias: TileTensor[bias_dtype, bias_LT, ImmutUntrackedOrigin],  # (C,)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, ImmutUntrackedOrigin
    ],  # (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    bias_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    has_bias: Int8,
    has_seq_idx: Int8,
    silu_activation: Int8,
):
    """GPU causal conv1d for channel-first (B, C, L) layout.

    One channel per block-row; each thread owns `kNElts` consecutive sequence
    positions of a single channel (coalesced along the contiguous L axis).
    `has_bias` / `has_seq_idx` gate the bias add and packed-sequence masking.

    Grid: (ceildiv(seqlen, kNThreads * kNElts), dim, batch). Block: kNThreads.

    Optimization (GB10: 3.17 µs vs Dao-AILab 3.30 µs at mamba prefill, fp32;
    2.82 vs 4.06 µs in bf16). The dense path is **issue-bound on load
    instructions**, not bandwidth-bound — the ~3 MB working set is L2-resident,
    and a naive scalar kernel issues `kNElts*kWidth` global loads per thread. The
    vectorized fast path below replaces those with a single width-`kNElts` vector
    load for the thread's tile plus `kWidth-1` L2-cached halo scalars, computed
    fully unrolled at comptime (so every tap/halo index is a compile-time
    constant) and a single vector store. It engages only when the access is
    L-contiguous, unmasked, full-tile, and `kNElts`-aligned; everything else
    (seq_idx / non-contiguous / ragged tail) falls back to the scalar loop, which
    is what keeps correctness general. The op picks `kNThreads*kNElts == 256` so
    every thread is busy at the common L=256 (Dao's 512-tile idles half the
    block). Accumulation is float32 regardless of storage dtype so bf16/fp16 stay
    within parity. Rationale + what regressed (shmem, warp-shuffle, channel-fold)
    is in `.planning/causal-conv1d-optimization-notes.md`.
    """
    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    # Channels may be folded along the block's y dimension to raise occupancy:
    # each block-row covers block_dim.y channels, one per thread_idx.y. With the
    # default 1D launch (block_dim.y == 1, thread_idx.y == 0) this reduces to the
    # classic one-channel-per-block-row mapping, so existing callers are
    # unaffected. Roofline showed the 1D launch occupancy-bound (~63%); folding
    # 2+ channels per block doubles warps/block and hides latency.
    var channel_id: Int = Int(block_idx.y) * Int(block_dim.y) + Int(
        thread_idx.y
    )
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = Int(x.dim[0]())
    var nChannels: Int = Int(x.dim[1]())
    var nSeqLen: Int = Int(x.dim[2]())

    if batch_id >= nBatches or channel_id >= nChannels or kWidth != width:
        return

    # Null pointer safety for the always-present tensors.
    if Int(x.ptr) == 0 or Int(output.ptr) == 0 or Int(weight.ptr) == 0:
        return

    var cur_bias: Scalar[x_dtype] = 0
    if has_bias != 0 and Int(bias.ptr) != 0:
        var bias_dim = Int(bias.dim[0]())
        if bias_dim > 0 and channel_id < bias_dim:
            cur_bias = Scalar[x_dtype](
                bias.raw_load(UInt32(channel_id) * bias_stride)
            )

    var out_vals: SIMD[output_dtype, kNElts] = 0

    # Load this channel's weights, lane w = weight[channel, w]. Used as a
    # per-tap scalar (W[w]); a single width-generic dot product serves every
    # supported width.
    var weight_c_base: UInt32 = UInt32(channel_id) * weight_c_stride
    var W: SIMD[x_dtype, kWidth] = 0

    comptime for w in range(kWidth):
        W[w] = Scalar[x_dtype](
            weight.raw_load(weight_c_base + UInt32(w) * weight_width_stride)
        )

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)
    if seq_start >= nSeqLen:
        return

    var x_cbase: UInt32 = (
        UInt32(batch_id) * x_batch_stride + UInt32(channel_id) * x_c_stride
    )
    var seq_base: UInt32 = UInt32(batch_id) * seq_idx_batch_stride
    var silu_active = Bool(Int(silu_activation) != 0)
    var use_seq_idx = has_seq_idx != 0

    var out_cbase: UInt32 = (
        UInt32(batch_id) * out_batch_stride + UInt32(channel_id) * out_c_stride
    )

    # ---- Fast vectorized path ------------------------------------------------
    # Hot path for dense prefill (e.g. mamba): L-contiguous x/out, no
    # packed-sequence masking, a full (non-ragged) tile, and kNElts-aligned base
    # offsets. The generic path below issues kNElts*kWidth scalar global loads
    # per thread; here we issue a single width-kNElts vector load for this
    # thread's tile plus kWidth-1 L2-cached halo scalars, then compute fully
    # unrolled at compile time (every tap/halo selection resolves at comptime).
    # This collapses the load-instruction count that bounds the scalar path.
    var aligned = (
        x_c_stride % UInt32(kNElts) == 0
        and x_batch_stride % UInt32(kNElts) == 0
        and out_c_stride % UInt32(kNElts) == 0
        and out_batch_stride % UInt32(kNElts) == 0
    )
    if (
        x_l_stride == 1
        and out_l_stride == 1
        and not use_seq_idx
        and seq_start + kNElts <= nSeqLen
        and aligned
    ):
        comptime vec_align = align_of[SIMD[x_dtype, kNElts]]()
        var xv = x.raw_load[width=kNElts, alignment=vec_align](
            x_cbase + UInt32(seq_start)
        )
        # Halo = the kWidth-1 positions immediately preceding this thread's tile:
        # halo[k] = x[seq_start - (kWidth-1) + k]; causal out-of-range -> 0.
        # These belong to the previous thread's tile and are read from global,
        # but they hit L2 (the neighbour just loaded them), so on GB10 this is
        # cheaper than a warp shuffle to fetch them across lanes (measured: the
        # shuffle variant regressed ~9% from the per-thread branch divergence).
        var halo = InlineArray[Scalar[x_dtype], kWidth](fill=0)
        comptime for k in range(kWidth - 1):
            var input_l: Int = seq_start - (kWidth - 1) + k
            if input_l >= 0:
                halo[k] = Scalar[x_dtype](x.raw_load(x_cbase + UInt32(input_l)))

        # Accumulate in float32 regardless of storage dtype (matches Dao's
        # internal precision, so bf16/fp16 stay within parity tolerance; for
        # fp32 it is a no-op). The vector load/store stay in the native dtype to
        # keep memory traffic minimal.
        var bias_f32 = Scalar[DType.float32](cur_bias)
        var fast_out: SIMD[output_dtype, kNElts] = 0
        comptime for i in range(kNElts):
            var acc: Scalar[DType.float32] = bias_f32
            comptime for w in range(kWidth):
                comptime rel = i - (kWidth - 1 - w)
                comptime if rel >= 0:
                    acc = acc + Scalar[DType.float32](W[w]) * Scalar[
                        DType.float32
                    ](xv[rel])
                else:
                    acc = acc + Scalar[DType.float32](W[w]) * Scalar[
                        DType.float32
                    ](halo[i + w])
            if silu_active:
                acc = silu(acc)
            fast_out[i] = acc.cast[output_dtype]()

        output.raw_store[width=kNElts, alignment=vec_align](
            out_cbase + UInt32(seq_start), fast_out
        )
        return
    # ---- Generic scalar path (seq_idx / non-contiguous / ragged tail) --------

    comptime for i in range(kNElts):
        var pos: Int = seq_start + i
        if pos >= seq_end:
            break

        var cur_seq_idx: Int32 = 0
        if use_seq_idx:
            cur_seq_idx = Int32(
                seq_idx.raw_load(seq_base + UInt32(pos) * seq_idx_l_stride)
            )

        var conv_result: Scalar[x_dtype] = 0
        comptime for w in range(kWidth):
            var input_l: Int = pos - (kWidth - 1 - w)
            if input_l >= 0 and input_l < nSeqLen:
                var ok = True
                if use_seq_idx:
                    var in_seq = Int32(
                        seq_idx.raw_load(
                            seq_base + UInt32(input_l) * seq_idx_l_stride
                        )
                    )
                    if in_seq != cur_seq_idx:
                        ok = False
                if ok:
                    var x_val = Scalar[x_dtype](
                        x.raw_load(x_cbase + UInt32(input_l) * x_l_stride)
                    )
                    conv_result = conv_result + W[w] * x_val

        var out_val: Scalar[output_dtype] = Scalar[output_dtype](
            cur_bias
        ) + Scalar[output_dtype](conv_result)
        out_vals[i] = apply_silu(out_val, silu_active)

    comptime for i in range(kNElts):
        var pos: Int = seq_start + i
        if pos >= seq_end:
            break
        output.raw_store(
            out_cbase + UInt32(pos) * out_l_stride,
            Scalar[output_dtype](out_vals[i]),
        )


def causal_conv1d_channel_last_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[x_dtype, x_LT, ImmutUntrackedOrigin],  # (B, L, C)
    weight: TileTensor[weight_dtype, weight_LT, ImmutUntrackedOrigin],  # (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin
    ],  # (B, L, C)
    bias: TileTensor[bias_dtype, bias_LT, ImmutUntrackedOrigin],  # (C,)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, ImmutUntrackedOrigin
    ],  # (B, L)
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    bias_stride: UInt32,
    seq_idx_batch_stride: UInt32,
    seq_idx_l_stride: UInt32,
    has_bias: Int8,
    has_seq_idx: Int8,
    silu_activation: Int8,
):
    """GPU causal conv1d for channel-last (B, L, C) layout.

    Threads map to channels (not positions): thread t of a block owns channel
    `block_idx.y * kNThreads + t`, and the block covers a contiguous range of
    `kNElts` sequence positions (`block_idx.x`). Because C is the contiguous
    axis, neighbouring threads (neighbouring channels) read neighbouring memory
    at each L position, so the per-position loads coalesce across the warp — the
    key to channel-last throughput. Each thread does a short sliding scan over
    its `kNElts` positions, reusing the kWidth-tap window. `has_bias` /
    `has_seq_idx` gate the bias add and packed-sequence masking.

    Grid: (ceildiv(seqlen, kNElts), ceildiv(dim, kNThreads), batch).
    Block: kNThreads.

    Optimization (GB10: 3.15 µs vs Dao-AILab 3.55 µs at mamba prefill, fp32).
    The thread->channel mapping is the whole game: a thread-per-position mapping
    (the obvious port) reads L positions `dim` elements apart across the warp and
    is **uncoalesced** — that version measured 14.5 µs, 4x slower. Mapping
    threads to channels instead makes each per-position load a coalesced warp
    transaction along the contiguous C axis. Channels are independent, so unlike
    channel-first there is no cross-thread halo to share (no shared memory
    needed); per-thread sliding-window reuse (load each of `kNElts+kWidth-1`
    window positions once) removes the only remaining redundancy. The op uses 64
    channels/block x 8 positions/thread. Trajectory and the failed shared-memory
    /vector-load attempts are in `.planning/causal-conv1d-optimization-notes.md`.
    """
    var batch_id: Int = block_idx.z
    var nSeqLen: Int = seqlen
    var nChannels: Int = dim

    # Thread -> channel (coalesced across the warp); block -> kNElts positions.
    var c: Int = Int(block_idx.y) * Int(block_dim.x) + Int(thread_idx.x)
    var l0: Int = Int(block_idx.x) * kNElts

    if batch_id >= batch or c >= nChannels or kWidth != width or l0 >= nSeqLen:
        return

    # Null pointer safety for the always-present tensors.
    if Int(x.ptr) == 0 or Int(output.ptr) == 0 or Int(weight.ptr) == 0:
        return

    var bias_dim = Int(bias.dim[0]())
    var seq_base: UInt32 = UInt32(batch_id) * seq_idx_batch_stride
    var silu_active = Bool(Int(silu_activation) != 0)
    var use_seq_idx = has_seq_idx != 0

    # Per-channel bias and weights (weight row is kWidth-contiguous).
    var cur_bias: Scalar[DType.float32] = 0
    if has_bias != 0 and Int(bias.ptr) != 0 and c < bias_dim:
        cur_bias = Scalar[DType.float32](bias.raw_load(UInt32(c) * bias_stride))
    var wbase: UInt32 = UInt32(c) * weight_c_stride
    var W: SIMD[DType.float32, kWidth] = 0
    comptime for w in range(kWidth):
        W[w] = Scalar[DType.float32](
            weight.raw_load(wbase + UInt32(w) * weight_width_stride)
        )

    var x_bc: UInt32 = (
        UInt32(batch_id) * x_batch_stride + UInt32(c) * x_c_stride
    )
    var out_bc: UInt32 = (
        UInt32(batch_id) * out_batch_stride + UInt32(c) * out_c_stride
    )

    # Scan this thread's kNElts positions for channel c. Each input element is
    # read with a per-position scalar load that coalesces across the warp
    # (adjacent threads = adjacent channels = adjacent memory, since C is
    # contiguous). Accumulate in float32. Correct for any strides (a
    # non-contiguous layout simply loses the coalescing, not correctness).
    #
    # Fast path (no packed-sequence masking): load each of the kNElts+kWidth-1
    # distinct window positions once and reuse it across outputs, instead of the
    # kNElts*kWidth redundant per-tap loads (adjacent outputs share kWidth-1
    # taps). Output cur_l=l0+i, tap w reads l0-(kWidth-1)+(i+w).
    if not use_seq_idx:
        comptime kWindow = kNElts + kWidth - 1
        var xs = InlineArray[Scalar[DType.float32], kWindow](fill=0)
        comptime for j in range(kWindow):
            var input_l: Int = l0 - (kWidth - 1) + j
            if input_l >= 0 and input_l < nSeqLen:
                xs[j] = Scalar[DType.float32](
                    x.raw_load(x_bc + UInt32(input_l) * x_l_stride)
                )
        comptime for i in range(kNElts):
            var cur_l: Int = l0 + i
            if cur_l >= nSeqLen:
                break
            var acc: Scalar[DType.float32] = cur_bias
            comptime for w in range(kWidth):
                acc = acc + W[w] * xs[i + w]
            if silu_active:
                acc = silu(acc)
            output.raw_store(
                out_bc + UInt32(cur_l) * out_l_stride, acc.cast[output_dtype]()
            )
        return

    comptime for i in range(kNElts):
        var cur_l: Int = l0 + i
        if cur_l >= nSeqLen:
            break

        var cur_seq_idx: Int32 = 0
        if use_seq_idx:
            cur_seq_idx = Int32(
                seq_idx.raw_load(seq_base + UInt32(cur_l) * seq_idx_l_stride)
            )

        var acc: Scalar[DType.float32] = cur_bias
        comptime for w in range(kWidth):
            var input_l: Int = cur_l - (kWidth - 1 - w)
            if input_l >= 0:
                var ok = True
                if use_seq_idx:
                    var in_seq = Int32(
                        seq_idx.raw_load(
                            seq_base + UInt32(input_l) * seq_idx_l_stride
                        )
                    )
                    if in_seq != cur_seq_idx:
                        ok = False
                if ok:
                    acc = acc + W[w] * Scalar[DType.float32](
                        x.raw_load(x_bc + UInt32(input_l) * x_l_stride)
                    )
        if silu_active:
            acc = silu(acc)
        output.raw_store(
            out_bc + UInt32(cur_l) * out_l_stride, acc.cast[output_dtype]()
        )


# ============================================================================
# Causal Conv1D Update Kernels
# ============================================================================
# These kernels implement incremental (step-by-step) convolution for inference,
# maintaining a conv_state buffer that gets updated with each step.


def causal_conv1d_update_cpu[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    has_bias: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,  # seqlen of x (typically 1 for autoregressive inference)
    width: Int,
    state_len: Int,  # state_len of conv_state (>= width - 1)
    x: TileTensor[mut=False, x_dtype, ...],  # (B, C, L) or (B, C) when L=1
    conv_state: TileTensor[mut=True, conv_state_dtype, ...],  # (B, C, S)
    weight: TileTensor[mut=False, weight_dtype, ...],  # (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # (B, C, L)
    bias: TileTensor[
        mut=False, bias_dtype, ...
    ],  # (C,) — ignored unless has_bias
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_c_stride: UInt32,
    conv_state_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    silu_activation: Bool,
):
    """CPU implementation of causal conv1d update for incremental inference.

    Concatenates conv_state with x to form a sliding window, computes the
    convolution output for the new positions, then updates conv_state with the
    newest values from x. `has_bias` gates the per-channel bias add.

    Parameters:
        x_dtype: Element type of the input.
        conv_state_dtype: Element type of the conv state.
        weight_dtype: Element type of the weights.
        output_dtype: Element type of the output.
        bias_dtype: Element type of the bias.
        has_bias: Whether to add the per-channel bias.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of input x (typically 1).
        width: Kernel width.
        state_len: Length of conv_state (>= width - 1).
        x: Input tensor.
        conv_state: Convolution state buffer (modified in-place).
        weight: Convolution weights.
        output: Output tensor.
        bias: Bias tensor; only read when `has_bias`.
        x_batch_stride: Stride for batch dimension in x.
        x_c_stride: Stride for channel dimension in x.
        x_l_stride: Stride for sequence dimension in x.
        conv_state_batch_stride: Stride for batch dimension in conv_state.
        conv_state_c_stride: Stride for channel dimension in conv_state.
        conv_state_l_stride: Stride for state dimension in conv_state.
        weight_c_stride: Stride for channel dimension in weight.
        weight_width_stride: Stride for width dimension in weight.
        out_batch_stride: Stride for batch dimension in output.
        out_c_stride: Stride for channel dimension in output.
        out_l_stride: Stride for sequence dimension in output.
        silu_activation: Whether to apply SiLU activation.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for c in range(dim):
            var weight_c_base = Int(UInt32(c) * weight_c_stride)
            var cur_bias: Scalar[output_dtype] = 0
            comptime if has_bias:
                cur_bias = Scalar[output_dtype](bias.raw_load(c))

            # Process each position in the input sequence.
            for l in range(seqlen):
                var conv_sum: Scalar[output_dtype] = cur_bias

                for w in range(width):
                    # Position in the virtual concatenated sequence [state, x].
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[x_dtype] = 0.0

                    if src_pos >= state_len:
                        var x_l_pos = src_pos - state_len
                        var x_offset = Int(
                            UInt32(b) * x_batch_stride
                            + UInt32(c) * x_c_stride
                            + UInt32(x_l_pos) * x_l_stride
                        )
                        input_val = x.raw_load(x_offset)
                    elif src_pos >= 0:
                        var conv_state_offset = Int(
                            UInt32(b) * conv_state_batch_stride
                            + UInt32(c) * conv_state_c_stride
                            + UInt32(src_pos) * conv_state_l_stride
                        )
                        input_val = Scalar[x_dtype](
                            conv_state.raw_load(conv_state_offset)
                        )
                    # else: src_pos < 0, treat as 0 (zero padding)

                    var weight_offset = weight_c_base + Int(
                        UInt32(w) * weight_width_stride
                    )
                    var weight_val: Scalar[weight_dtype] = weight.raw_load(
                        weight_offset
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

                var out_offset = Int(
                    UInt32(b) * out_batch_stride
                    + UInt32(c) * out_c_stride
                    + UInt32(l) * out_l_stride
                )
                output.raw_store(
                    out_offset, apply_silu(conv_sum, silu_activation)
                )

            # Update conv_state: shift old values and add new x values.
            if seqlen >= state_len:
                # x is longer than state, just copy last state_len values from x.
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_offset = Int(
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(x_l_pos) * x_l_stride
                    )
                    var x_val = x.raw_load(x_offset)
                    var conv_state_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    conv_state.raw_store(
                        conv_state_offset, Scalar[conv_state_dtype](x_val)
                    )
            else:
                # Shift conv_state left by seqlen positions, then append x.
                for s in range(state_len - seqlen):
                    var src_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((s + seqlen)) * conv_state_l_stride
                    )
                    var dst_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32(s) * conv_state_l_stride
                    )
                    var val = conv_state.raw_load(src_offset)
                    conv_state.raw_store(dst_offset, val)

                # Copy x values to the end.
                for l in range(seqlen):
                    var x_offset = Int(
                        UInt32(b) * x_batch_stride
                        + UInt32(c) * x_c_stride
                        + UInt32(l) * x_l_stride
                    )
                    var x_val = x.raw_load(x_offset)
                    var conv_state_offset = Int(
                        UInt32(b) * conv_state_batch_stride
                        + UInt32(c) * conv_state_c_stride
                        + UInt32((state_len - seqlen + l)) * conv_state_l_stride
                    )
                    conv_state.raw_store(
                        conv_state_offset, Scalar[conv_state_dtype](x_val)
                    )


def causal_conv1d_update_gpu[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    kNThreads: Int,
    x_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    x: TileTensor[x_dtype, x_LT, ImmutUntrackedOrigin],
    conv_state: TileTensor[conv_state_dtype, conv_state_LT, MutUntrackedOrigin],
    weight: TileTensor[weight_dtype, weight_LT, ImmutUntrackedOrigin],
    output: TileTensor[output_dtype, output_LT, MutUntrackedOrigin],
    bias: TileTensor[bias_dtype, bias_LT, ImmutUntrackedOrigin],
    x_batch_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_c_stride: UInt32,
    conv_state_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_width_stride: UInt32,
    out_batch_stride: UInt32,
    out_c_stride: UInt32,
    out_l_stride: UInt32,
    has_bias: Int8,
    silu_activation: Int8,
):
    """GPU kernel for causal conv1d update (autoregressive decode).

    Performs incremental updates to maintain convolution state for efficient
    token generation: processes the new input, writes the output, and updates the
    conv state. `has_bias` gates the per-channel bias add.

    Grid: (batch, ceildiv(dim, kNThreads)). Block: kNThreads.
    """
    var b = block_idx.x
    var c_base = block_idx.y * kNThreads
    var c = c_base + thread_idx.x

    if b >= batch or c >= dim:
        return

    var width_minus_1: Int = width - 1
    var weight_c_base = Int(UInt32(c) * weight_c_stride)
    var cur_bias: Scalar[output_dtype] = 0
    if has_bias != 0:
        cur_bias = Scalar[output_dtype](bias.raw_load(c))
    var silu_active = Bool(silu_activation != 0)

    for l in range(seqlen):
        var conv_sum: Scalar[output_dtype] = cur_bias

        for w in range(width):
            var src_pos = state_len + l - (width_minus_1 - w)
            var input_val: Scalar[x_dtype] = 0.0

            if src_pos >= state_len:
                var x_l_pos = src_pos - state_len
                var x_offset = Int(
                    UInt32(b) * x_batch_stride
                    + UInt32(c) * x_c_stride
                    + UInt32(x_l_pos) * x_l_stride
                )
                input_val = x.raw_load(x_offset)
            elif src_pos >= 0:
                var conv_state_offset = Int(
                    UInt32(b) * conv_state_batch_stride
                    + UInt32(c) * conv_state_c_stride
                    + UInt32(src_pos) * conv_state_l_stride
                )
                input_val = Scalar[x_dtype](
                    conv_state.raw_load(conv_state_offset)
                )
            var weight_offset = weight_c_base + Int(
                UInt32(w) * weight_width_stride
            )
            var weight_val: Scalar[weight_dtype] = weight.raw_load(
                weight_offset
            )
            conv_sum = conv_sum + Scalar[output_dtype](
                input_val * Scalar[x_dtype](weight_val)
            )
        var out_offset = Int(
            UInt32(b) * out_batch_stride
            + UInt32(c) * out_c_stride
            + UInt32(l) * out_l_stride
        )
        output.raw_store(out_offset, apply_silu(conv_sum, silu_active))

    # Update conv_state.
    if seqlen >= state_len:
        for s in range(state_len):
            var x_l_pos = seqlen - state_len + s
            var x_offset = Int(
                UInt32(b) * x_batch_stride
                + UInt32(c) * x_c_stride
                + UInt32(x_l_pos) * x_l_stride
            )
            var x_val = x.raw_load(x_offset)
            var conv_state_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32(s) * conv_state_l_stride
            )
            conv_state.raw_store(
                conv_state_offset, Scalar[conv_state_dtype](x_val)
            )
    else:
        for s in range(state_len - seqlen):
            var src_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32((s + seqlen)) * conv_state_l_stride
            )
            var dst_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32(s) * conv_state_l_stride
            )
            var val = conv_state.raw_load(src_offset)
            conv_state.raw_store(dst_offset, val)

        for l in range(seqlen):
            var x_offset = Int(
                UInt32(b) * x_batch_stride
                + UInt32(c) * x_c_stride
                + UInt32(l) * x_l_stride
            )
            var x_val = x.raw_load(x_offset)
            var conv_state_offset = Int(
                UInt32(b) * conv_state_batch_stride
                + UInt32(c) * conv_state_c_stride
                + UInt32((state_len - seqlen + l)) * conv_state_l_stride
            )
            conv_state.raw_store(
                conv_state_offset, Scalar[conv_state_dtype](x_val)
            )
