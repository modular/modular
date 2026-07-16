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
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv, exp
from std.sys import llvm_intrinsic
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from layout import UNKNOWN_VALUE, Idx, Layout, LayoutTensor, TileTensor
from layout.coord import Coord
from layout.tile_layout import (
    TensorLayout,
    row_major,
)

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
    p_layout: TensorLayout,
    q_layout: TensorLayout,
    valid_length_layout: TensorLayout,
    sink_layout: TensorLayout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    Depth: Int,
    SplitSize: Int,
](
    o_partial: TileTensor[p_type, p_layout, MutAnyOrigin],
    m_partial: TileTensor[p_type, p_layout, MutAnyOrigin],
    l_partial: TileTensor[p_type, p_layout, MutAnyOrigin],
    q: TileTensor[q_type, q_layout, ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    sink_weights: OptionalReg[TileTensor[q_type, sink_layout, ImmutAnyOrigin]],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    # Full key count for the dense decode path (the K tensor's seq dim); the
    # KVCache/ragged paths derive their key count from `cache_length` +
    # `cur_query_len` instead. See the `cur_cache_len` branch below.
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

    # Decode offset math — mirror `_bmm0_bs` (mha.mojo:5560-5589). The
    # `cur_cache_len` (number of keys to attend) is set PER BRANCH because the
    # dense (`else`) path takes it from `max_cache_size` (the K tensor's full
    # seq dim), NOT from `cur_query_len` / `cache_length` — exactly as the naive
    # fallback `_bmm0_bs` and the Apple prefill producer (`fa_prefill.mojo`) do.
    # The prior shared `cur_cache_len = cur_query_len` (under
    # `_is_cache_length_accurate`) silently attended only 1 key on the dense
    # decode path (`_use_valid_length=False, _is_cache_length_accurate=True` --
    # the `flash_attention` dense overload's ABI), so a dense decode dropped all
    # but the first key. Only the KVCache decode path (`_use_valid_length=True`)
    # was ever exercised.
    var seq_start: Int
    var cur_query_len: Int
    var q_offset: Int
    var cur_cache_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        q_offset = depth * (seq_start * num_heads + head_id)
        # The new token's own KV sits at index `cache_length`, so an inaccurate
        # cache length must include it. Mirror `_bmm0_bs` (mha.mojo:5567-5575).
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    elif _use_valid_length:
        # KVCache decode: valid_length holds per-sequence query lengths, not row
        # offsets. Mirror `_bmm0_bs` (mha.mojo:5576-5582).
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        q_offset = depth * (head_id + num_heads * max_prompt_len * batch_id)
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    else:
        # Dense decode: all sequences share one length and cache length; the
        # full key count is `max_cache_size` (the K tensor's seq dim). Mirror
        # `_bmm0_bs` (mha.mojo:5585-5589).
        seq_start = batch_id
        cur_query_len = max_prompt_len
        q_offset = depth * (head_id + num_heads * max_prompt_len * batch_id)
        cur_cache_len = max_cache_size
    var seq_len = cur_cache_len

    var start = split_id * SplitSize
    if start >= seq_len:
        return
    var end = min(start + SplitSize, seq_len)

    # Decode token's score-matrix row (== cache_length). Mirror mha.mojo:5324.
    var score_row = cur_cache_len - cur_query_len

    # Q is a flat 1D TileTensor over the whole buffer; this lane owns the
    # head-dim chunk [lane*EPL, lane*EPL+EPL) at `q_offset`. Vectorized load
    # through the tile (no raw pointer arithmetic).
    var q_frag = q.load[width=EPL](Coord(q_offset + lane * EPL)).cast[
        DType.float32
    ]()

    # KV sub-tile layout: a 1D (depth,) contiguous view of one token's K/V for
    # `kv_head`, reused for every key in this split. `block_paged_tile` infers
    # the type from this value; each lane loads its `EPL` chunk.
    var kv_token_layout = row_major(Coord(depth))

    # Replicated on every lane, so the running softmax needs no cross-lane comms.
    var m = NEG_INF
    var l = Float32(0.0)
    var o_frag = SIMD[DType.float32, EPL](0.0)

    # Attention sink as init-state: pre-seed (m, l) with a virtual "key -1" of
    # raw score `sink_weight`, contributing `exp(sink - m) = 1` to the running
    # denominator (plain `exp`, no log2e — Apple, like the prefill kernel and
    # nn/softmax.mojo, compares the UNSCALED sink weight against the post-scale
    # row max). Seed ONLY split 0: this is split-K, so the stitch kernel does a
    # cross-split LSE combine; seeding every split would count the sink
    # `num_partitions` times. Split 0 always exists (start=0 < seq_len), so the
    # sink is counted exactly once. Mirrors AppleSoftmax.seed_sink in
    # fa_prefill.mojo and amd-attention-sink-as-init-state.
    comptime if sink:
        if split_id == 0:
            # Per-head sink weight from the nullable `OptionalReg[TileTensor]`
            # (NOT a dangling pointer -- KB `unsafepointer-is-non-nullable`).
            # The deref is comptime-gated on `sink`, so None is never reached.
            m = rebind[Scalar[q_type]](sink_weights.value()[head_id]).cast[
                DType.float32
            ]()
            l = Float32(1.0)

    for kv0 in range(start, end, BN):
        var partials = SIMD[DType.float32, BN](0.0)

        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var k_tile = k.block_paged_tile[1](
                    UInt32(batch_id),
                    UInt32(j),
                    UInt32(kv_head),
                    kv_token_layout,
                )
                var kvec = k_tile.load[width=EPL](Coord(lane * EPL)).cast[
                    DType.float32
                ]()
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
                var v_tile = v.block_paged_tile[1](
                    UInt32(batch_id),
                    UInt32(j),
                    UInt32(kv_head),
                    kv_token_layout,
                )
                var vvec = v_tile.load[width=EPL](Coord(lane * EPL)).cast[
                    DType.float32
                ]()
                o_frag = o_frag + p[kk] * vvec

    comptime assert (
        o_partial.flat_rank == 1 and m_partial.flat_rank == 1
    ), "partials are flat 1D TileTensors"
    comptime for i in range(EPL):
        var d = lane * EPL + i
        var oi = _o_idx(
            batch_id, head_id, d, split_id, num_heads, Depth, num_partitions
        )
        o_partial[oi] = rebind[o_partial.ElementType](
            SIMD[p_type, 1](o_frag[i].cast[p_type]())
        )
    if lane == 0:
        var idx = _ml_idx(
            batch_id, head_id, split_id, num_heads, num_partitions
        )
        l_partial[idx] = rebind[l_partial.ElementType](
            SIMD[p_type, 1](l.cast[p_type]())
        )
        m_partial[idx] = rebind[m_partial.ElementType](
            SIMD[p_type, 1](m.cast[p_type]())
        )


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
    output_layout: TensorLayout,
    p_layout: TensorLayout,
    valid_length_layout: TensorLayout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    SplitSize: Int,
](
    output: TileTensor[output_type, output_layout, MutAnyOrigin],
    o_partial: TileTensor[p_type, p_layout, ImmutAnyOrigin],
    m_partial: TileTensor[p_type, p_layout, ImmutAnyOrigin],
    l_partial: TileTensor[p_type, p_layout, ImmutAnyOrigin],
    k: k_t,
    valid_length: TileTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    max_prompt_len: Int,
    # Full key count for the dense decode path; mirrors the producer so the
    # combine's `active_splits` matches the splits the producer actually wrote.
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    num_partitions: Int,
):
    comptime assert (
        o_partial.flat_rank == 1
        and m_partial.flat_rank == 1
        and output.flat_rank == 1
    ), "partials and output are flat 1D TileTensors"
    var head_id = Int(block_idx.x)
    var batch_id = Int(block_idx.y)
    var d = Int(thread_idx.x)

    if d >= depth:
        return

    # Output offset — mirror mha.mojo:5390. `cur_cache_len` (the attend span)
    # is set PER BRANCH and MUST match the producer's exactly, so the combine
    # reads precisely the splits the producer wrote (the dense path takes it
    # from `max_cache_size`, not `cur_query_len`).
    var seq_start: Int
    var cur_query_len: Int
    var cur_cache_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    elif _use_valid_length:
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    else:
        # Dense decode: full key count is `max_cache_size`.
        seq_start = batch_id
        cur_query_len = max_prompt_len
        cur_cache_len = max_cache_size

    # Split count must mirror the producer's attend span (`cur_cache_len`), not
    # the bare cache length, so we read exactly the partials that were written.
    var active_splits = ceildiv(cur_cache_len, SplitSize)

    # Combine in Float32 (partials cast in on read), matching the producer.
    var m = NEG_INF
    var l = Float32(0.0)
    var acc = Float32(0.0)

    for split in range(active_splits):
        var ml = _ml_idx(batch_id, head_id, split, num_heads, num_partitions)
        var m_s = rebind[Scalar[p_type]](m_partial[ml]).cast[DType.float32]()
        var m_new = max(m, m_s)
        var corr = exp(m - m_new)
        # `p` must use the same exp base as the producer for an exact combine.
        var p = exp(m_s - m_new)
        var l_s = rebind[Scalar[p_type]](l_partial[ml]).cast[DType.float32]()
        l = l * corr + p * l_s
        var oi = _o_idx(
            batch_id, head_id, d, split, num_heads, depth, num_partitions
        )
        var o_s = rebind[Scalar[p_type]](o_partial[oi]).cast[DType.float32]()
        acc = acc * corr + p * o_s
        m = m_new

    var o_off = (seq_start * num_heads + head_id) * depth
    output[o_off + d] = rebind[output.ElementType](
        SIMD[output_type, 1]((acc / l).cast[output_type]())
    )


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

    var o_partial_n = batch_size * num_heads * depth * num_partitions
    var ml_partial_n = batch_size * num_heads * num_partitions
    var o_partial_dev = ctx.enqueue_create_buffer[p_type](o_partial_n)
    var m_partial_dev = ctx.enqueue_create_buffer[p_type](ml_partial_n)
    var l_partial_dev = ctx.enqueue_create_buffer[p_type](ml_partial_n)

    # Flat 1D TileTensor views over the q/output/partial buffers. The kernels
    # bake the per-(batch, head, split, depth) offset into a linear index
    # (`_o_idx`/`_ml_idx`) and the BSHD/ragged q/out offset, so the flat views
    # just carry the device pointers with TileTensor typing (no raw pointers /
    # DeviceBuffer-as-pointer inside the kernels).
    var q_flat = TileTensor(
        q.ptr.as_immutable().as_unsafe_any_origin(),
        row_major(Coord(Int(q.size()))),
    )
    var output_flat = TileTensor(
        output.ptr.as_unsafe_any_origin(),
        row_major(Coord(Int(output.size()))),
    )
    var valid_length_flat = TileTensor(
        valid_length.ptr.as_immutable().as_unsafe_any_origin(),
        row_major(Coord(Int(valid_length.size()))),
    )
    var o_partial_t = TileTensor(
        o_partial_dev.unsafe_ptr(), row_major(Coord(o_partial_n))
    )
    var m_partial_t = TileTensor(
        m_partial_dev.unsafe_ptr(), row_major(Coord(ml_partial_n))
    )
    var l_partial_t = TileTensor(
        l_partial_dev.unsafe_ptr(), row_major(Coord(ml_partial_n))
    )
    var o_partial_imm = TileTensor(
        o_partial_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        row_major(Coord(o_partial_n)),
    )
    var m_partial_imm = TileTensor(
        m_partial_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        row_major(Coord(ml_partial_n)),
    )
    var l_partial_imm = TileTensor(
        l_partial_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        row_major(Coord(ml_partial_n)),
    )

    # Sink weights: a nullable `OptionalReg[TileTensor]` passed by value (NOT a
    # dangling `UnsafePointer` -- KB `unsafepointer-is-non-nullable`). When
    # sink=False this is None and never read (the seed is comptime-gated on
    # `sink` in the producer). The per-head [num_heads] tensor is converted to
    # a TileTensor so the kernel stays TileTensor-only.
    var sink_layout_val = row_major(Coord(num_heads))
    comptime SinkTile = TileTensor[
        q_type, type_of(sink_layout_val), ImmutAnyOrigin
    ]
    var sink_tile: OptionalReg[SinkTile]
    comptime if sink:
        var sw = sink_weights.value()
        sink_tile = OptionalReg[SinkTile](
            SinkTile(
                sw.ptr.as_immutable().as_unsafe_any_origin(),
                sink_layout_val,
            )
        )
    else:
        sink_tile = None

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
                type_of(o_partial_t).LayoutType,
                type_of(q_flat).LayoutType,
                type_of(valid_length_flat).LayoutType,
                type_of(sink_layout_val),
                ragged=ragged,
                sink=sink,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
                Depth=D,
                SplitSize=SplitSize,
            ]
            ctx.enqueue_function[core_kernel](
                o_partial_t,
                m_partial_t,
                l_partial_t,
                q_flat,
                k,
                v,
                mask_functor,
                valid_length_flat,
                sink_tile,
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
        type_of(output_flat).LayoutType,
        type_of(o_partial_imm).LayoutType,
        type_of(valid_length_flat).LayoutType,
        ragged=ragged,
        sink=sink,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
        SplitSize=SplitSize,
    ]
    ctx.enqueue_function[stitch_kernel](
        output_flat,
        o_partial_imm,
        m_partial_imm,
        l_partial_imm,
        k,
        valid_length_flat,
        max_prompt_len,
        max_cache_size,
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
