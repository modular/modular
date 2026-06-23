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
"""Apple (Metal) split-K naive flash-attention DECODE kernels.

Apple silicon GPU (Metal), decode-only (one query token per sequence), paged KV
cache via the `MHAOperand` contract, BF16 storage / FP32 accumulation.

Warp-centric producer: one simdgroup (32 lanes) owns one split of the KV range
for one `(batch, head)`. Lane `L` owns the contiguous head-dim chunk
`[L*EPL, L*EPL+EPL)` where `EPL = head_dim // WARP_SIZE`; the query and running
output stay in registers, `Q.K^T` is reduced across lanes with one `air.simd_sum`
per key, and `P.V` is reduction-free. The inner loop has **no `barrier()` and no
threadgroup memory** — the two levers Apple silicon is most sensitive to.

Two kernels:
  * `naive_fa_decode_apple_core`  — producer. Grid `(num_partitions,
    batch_size, num_heads)`, block = `WARP_SIZE` (one simdgroup). Each block
    writes per-partition partials `(o_partial, m_partial, l_partial)` via online
    softmax over `BN`-wide KV tiles.
  * `naive_fa_decode_apple_stitch` — stitch. Grid `(num_heads, batch_size)`,
    block `depth`. One thread per depth element; combines the contiguous
    per-partition partials into the final `output` with a log-sum-exp (LSE)
    reduction.

The host launcher `naive_fa_decode_apple` allocates the partials and enqueues
both kernels; `flash_attention_dispatch` selects it for Apple decode by default
(set `MODULAR_ENABLE_APPLE_NAIVE_FA_DECODE=0` to opt out). The launcher
dispatches the runtime `depth` to a compile-time `Depth` specialization over the
multiples of `WARP_SIZE` up to `NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM`; the
dispatcher only routes here when `depth % WARP_SIZE == 0` and
`depth <= NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM`, otherwise `mha_gpu_naive` runs.

Partial-buffer layout (partition-last / contiguous):
  * `ml_idx(b, head, split) = (b*num_heads + head)*num_partitions + split`
  * `o_idx(b, head, d, split) = ((b*num_heads + head)*depth + d)*num_partitions
    + split`
"""

from std.collections import OptionalReg
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv, exp
from std.sys import llvm_intrinsic
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from layout import UNKNOWN_VALUE, Layout, LayoutTensor

from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand

comptime BN = 16  # KV keys per producer tile step
comptime NEG_INF = Float32(-3.0e38)

# Dispatcher gate: larger dims (and non-multiples of WARP_SIZE) fall back to
# mha_gpu_naive.
comptime NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM = 256


@always_inline
def _apple_simd_sum(val: Float32) -> Float32:
    """Sum `val` across the simdgroup (broadcast to all lanes).

    One hardware instruction vs. a 5-shuffle butterfly tree.
    """
    return llvm_intrinsic["llvm.air.simd_sum", Float32](val)


@always_inline
def _ml_idx(
    b: Int, head: Int, split: Int, num_heads: Int, num_partitions: Int
) -> Int:
    """Partition-last index for the m/l partials."""
    return (b * num_heads + head) * num_partitions + split


@always_inline
def _o_idx(
    b: Int,
    head: Int,
    d: Int,
    split: Int,
    num_heads: Int,
    depth: Int,
    num_partitions: Int,
) -> Int:
    """Partition-last index for the o partials."""
    return ((b * num_heads + head) * depth + d) * num_partitions + split


# ===-------------------------------------------------------------------=== #
# Producer: warp-centric split-K online-softmax.
# Grid (num_partitions, batch, head); block = WARP_SIZE (one simdgroup).
# ===-------------------------------------------------------------------=== #
def naive_fa_decode_apple_core[
    q_type: DType,
    # `output_type` is unused; the parameter list mirrors `mha_gpu_naive` for
    # dispatch uniformity.
    output_type: DType,
    p_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    valid_length_layout: Layout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    Depth: Int,
    SplitSize: Int,
](
    o_partial: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    m_partial: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    l_partial: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    # `max_cache_size` is unused; mirrors `mha_gpu_naive` for dispatch uniformity.
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    num_partitions: Int,
):
    """Warp-centric split-K online-softmax producer for Apple decode attention.

    Lane `L` of the simdgroup owns the contiguous head-dim chunk
    `[L*EPL, L*EPL+EPL)` where `EPL = Depth // WARP_SIZE`. q and the running
    output stay in registers; `Q.K^T` is reduced across lanes with one
    `air.simd_sum` per key; `P.V` is reduction-free. No barriers, no shared
    memory.

    Constraints:
        `Depth % WARP_SIZE == 0` — the head dim must split evenly across lanes.
    """
    comptime assert (
        Depth % WARP_SIZE == 0
    ), "naive_fa_decode_apple_core requires Depth % WARP_SIZE == 0"
    comptime EPL = Depth // WARP_SIZE
    debug_assert(depth == Depth, "runtime depth must match comptime Depth")

    var split_id = Int(block_idx.x)
    var batch_id = Int(block_idx.y)
    var head_id = Int(block_idx.z)
    var kv_head = head_id // group
    var lane = Int(lane_id())

    # Decode offset math — mirror mha.mojo:5253-5305.
    var seq_start: Int
    var cur_query_len: Int
    var q_offset: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        q_offset = depth * (seq_start * num_heads + head_id)
    elif _use_valid_length:
        # KVCache decode: valid_length holds per-sequence query lengths, not row
        # offsets. Mirror `_bmm0_bs` (mha.mojo:5262-5264).
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        q_offset = depth * (head_id + num_heads * max_prompt_len * batch_id)
    else:
        seq_start = batch_id
        cur_query_len = 1
        q_offset = depth * (head_id + num_heads * max_prompt_len * batch_id)

    # The new token's own KV sits at index `cache_length`, so an inaccurate
    # cache length must include it. Mirror `_bmm0_bs` (mha.mojo:5253-5274).
    var cache_len = k.cache_length(batch_id)
    var cur_cache_len: Int
    comptime if _is_cache_length_accurate:
        cur_cache_len = cur_query_len
    else:
        cur_cache_len = cache_len + cur_query_len
    var seq_len = cur_cache_len

    var start = split_id * SplitSize
    if start >= seq_len:
        return
    var end = min(start + SplitSize, seq_len)

    # Decode token's score-matrix row (== cache_length). Mirror mha.mojo:5324.
    var score_row = cur_cache_len - cur_query_len

    var q_base = q_ptr + q_offset + lane * EPL
    var q_frag = q_base.load[width=EPL]().cast[DType.float32]()

    # Replicated on every lane, so the running softmax needs no cross-lane comms.
    var m = NEG_INF
    var l = Float32(0.0)
    var o_frag = SIMD[DType.float32, EPL](0.0)

    for kv0 in range(start, end, BN):
        var partials = SIMD[DType.float32, BN](0.0)

        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var kptr = k.block_paged_ptr[1](
                    UInt32(batch_id), UInt32(j), UInt32(kv_head), 0
                )
                var kvec = (
                    (kptr + lane * EPL).load[width=EPL]().cast[DType.float32]()
                )
                partials[kk] = (q_frag * kvec).reduce_add()

        # `air.simd_sum` is a warp collective; the `j < end` guard is
        # lane-independent, so all lanes enter it together.
        var scores = SIMD[DType.float32, BN](NEG_INF)

        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var s = _apple_simd_sum(partials[kk]) * scale
                scores[kk] = mask_functor.mask(
                    Index(batch_id, head_id, score_row, j), s
                )

        var m_tile = scores.reduce_max()
        var m_new = max(m, m_tile)
        var alpha = exp(m - m_new)
        var p = exp(scores - m_new)  # OOB keys are NEG_INF -> exp == 0
        l = l * alpha + p.reduce_add()
        m = m_new
        o_frag = o_frag * alpha

        # No cross-lane reduction: each lane accumulates its own output chunk.
        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var vptr = v.block_paged_ptr[1](
                    UInt32(batch_id), UInt32(j), UInt32(kv_head), 0
                )
                var vvec = (
                    (vptr + lane * EPL).load[width=EPL]().cast[DType.float32]()
                )
                o_frag = o_frag + p[kk] * vvec

    comptime for i in range(EPL):
        var d = lane * EPL + i
        o_partial[
            _o_idx(
                batch_id, head_id, d, split_id, num_heads, Depth, num_partitions
            )
        ] = o_frag[i].cast[p_type]()
    if lane == 0:
        var idx = _ml_idx(
            batch_id, head_id, split_id, num_heads, num_partitions
        )
        l_partial[idx] = l.cast[p_type]()
        m_partial[idx] = m.cast[p_type]()


# ===-------------------------------------------------------------------=== #
# Stitch: LSE-combine the per-partition partials. Grid (num_heads, batch),
# block `depth`.
# ===-------------------------------------------------------------------=== #
def naive_fa_decode_apple_stitch[
    output_type: DType,
    p_type: DType,
    k_t: MHAOperand,
    # `v_t` and `mask_t` are unused; the parameter list mirrors `mha_gpu_naive`
    # for dispatch uniformity.
    v_t: MHAOperand,
    mask_t: MHAMask,
    valid_length_layout: Layout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    SplitSize: Int,
](
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    o_partial: UnsafePointer[Scalar[p_type], ImmutAnyOrigin],
    m_partial: UnsafePointer[Scalar[p_type], ImmutAnyOrigin],
    l_partial: UnsafePointer[Scalar[p_type], ImmutAnyOrigin],
    k: k_t,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    num_heads: Int,
    depth: Int,
    num_partitions: Int,
):
    var head_id = Int(block_idx.x)
    var batch_id = Int(block_idx.y)
    var d = Int(thread_idx.x)

    if d >= depth:
        return

    # Output offset — mirror mha.mojo:5390.
    var seq_start: Int
    var cur_query_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
    elif _use_valid_length:
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
    else:
        seq_start = batch_id
        cur_query_len = 1

    # Split count must mirror the producer's attend span (`cur_cache_len`), not
    # the bare cache length, so we read exactly the partials that were written.
    var cache_len = k.cache_length(batch_id)
    var cur_cache_len: Int
    comptime if _is_cache_length_accurate:
        cur_cache_len = cur_query_len
    else:
        cur_cache_len = cache_len + cur_query_len
    var seq_chnks = ceildiv(cur_cache_len, SplitSize)

    # Combine in Float32 (partials cast in on read), matching the producer.
    var m = NEG_INF
    var l = Float32(0.0)
    var acc = Float32(0.0)

    for split in range(seq_chnks):
        var ml = _ml_idx(batch_id, head_id, split, num_heads, num_partitions)
        var m_s = m_partial[ml].cast[DType.float32]()
        var m_new = max(m, m_s)
        var corr = exp(m - m_new)
        # `p` must use the same exp base as the producer for an exact combine.
        var p = exp(m_s - m_new)
        l = l * corr + p * l_partial[ml].cast[DType.float32]()
        acc = (
            acc * corr
            + p
            * o_partial[
                _o_idx(
                    batch_id,
                    head_id,
                    d,
                    split,
                    num_heads,
                    depth,
                    num_partitions,
                )
            ].cast[DType.float32]()
        )
        m = m_new

    var o_off = (seq_start * num_heads + head_id) * depth
    output_ptr[o_off + d] = (acc / l).cast[output_type]()


# ===-------------------------------------------------------------------=== #
# Host launcher. Mirrors `mha_gpu_naive` (MHAOperand overload, mha.mojo:5066)
# signature; enqueues the producer/stitch pair. Dispatches the runtime `depth`
# to a compile-time `Depth` specialization over multiples of WARP_SIZE.
# ===-------------------------------------------------------------------=== #
def naive_fa_decode_apple[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: LayoutTensor[mut=False, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: LayoutTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, q.dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
) raises:
    """Host launcher for the Apple split-K decode attention pair (decode-only).
    """
    # No `is_apple_gpu()` assert here — this launcher compiles for the host
    # target, where that target-query is always False. The Apple gate is the
    # caller's (`has_apple_gpu_accelerator()` in dispatch).
    comptime q_type = q.dtype

    var num_keys = max_cache_size

    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    debug_assert(
        depth % WARP_SIZE == 0 and depth <= NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM,
        (
            "naive_fa_decode_apple requires depth %% WARP_SIZE == 0 and depth"
            " <= NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM; the dispatcher must gate"
            " unsupported head dims to mha_gpu_naive"
        ),
    )

    comptime p_type = get_accum_type[q_type]()

    comptime SplitSize = 32  # per-partition KV span

    # Uniform (not per-sequence) alloc avoids a device->host cache_lengths sync
    # each step. The `+ max_prompt_len` covers the producer's `cur_cache_len`
    # span — without it the alloc is one partition short when the cache length
    # is an exact multiple of SplitSize.
    var partition_keys: Int
    comptime if _is_cache_length_accurate:
        partition_keys = max_cache_size
    else:
        partition_keys = max_cache_size + max_prompt_len
    var num_partitions = ceildiv(partition_keys, SplitSize)

    var o_partial_dev = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * depth * num_partitions
    )
    var m_partial_dev = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * num_partitions
    )
    var l_partial_dev = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * num_partitions
    )

    # Non-owning: q/output reach the kernels as raw pointers (mirror
    # `_bmm0_bs`/`_bmm1_bs`, mha.mojo:5125-5128).
    var q_device = DeviceBuffer[q_type](ctx, q.ptr, q.size(), owning=False)
    var output_device = DeviceBuffer[output_type](
        ctx, output.ptr, output.size(), owning=False
    )

    # The producer needs the head dim at compile time (EPL = Depth //
    # WARP_SIZE), so specialize one kernel per supported `depth` and select at
    # runtime. The dispatcher guarantees `depth` matches one branch.
    comptime MAX_D_STEPS = NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM // WARP_SIZE
    comptime for di in range(1, MAX_D_STEPS + 1):
        comptime D = di * WARP_SIZE
        if depth == D:
            comptime core_kernel = naive_fa_decode_apple_core[
                q_type,
                output_type,
                p_type,
                k_t,
                v_t,
                mask_t,
                type_of(valid_length).layout,
                ragged=ragged,
                sink=sink,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
                Depth=D,
                SplitSize=SplitSize,
            ]
            ctx.enqueue_function[core_kernel](
                o_partial_dev,
                m_partial_dev,
                l_partial_dev,
                q_device,
                k,
                v,
                mask_functor,
                valid_length,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_size,
                num_heads,
                depth,
                group,
                num_partitions,
                grid_dim=(num_partitions, batch_size, num_heads),
                block_dim=WARP_SIZE,
            )

    comptime stitch_kernel = naive_fa_decode_apple_stitch[
        output_type,
        p_type,
        k_t,
        v_t,
        mask_t,
        type_of(valid_length).layout,
        ragged=ragged,
        sink=sink,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
        SplitSize=SplitSize,
    ]
    ctx.enqueue_function[stitch_kernel](
        output_device,
        o_partial_dev,
        m_partial_dev,
        l_partial_dev,
        k,
        valid_length,
        num_heads,
        depth,
        num_partitions,
        grid_dim=(num_heads, batch_size),
        block_dim=depth,
    )

    # Keep the partial buffers alive until both kernels have been enqueued.
    _ = o_partial_dev^
    _ = m_partial_dev^
    _ = l_partial_dev^
