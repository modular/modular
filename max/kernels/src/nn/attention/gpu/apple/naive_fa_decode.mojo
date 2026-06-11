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
Generalizes the single-head split-K decode prototype in
`.scratch/sdpa-decode/mha_decode_playground.mojo` (the `twoshot` path) to MHA +
GQA against the in-tree `MHAOperand` / `LayoutTensor` / `MHAMask` contract.

Two kernels:
  * `naive_fa_decode_apple_core`  — producer. Grid `(num_partitions,
    batch_size, num_heads)`, 1-D threadgroup. Each block owns one split of the
    KV range for one `(batch, head)` and writes per-partition partials
    `(o_partial, m_partial, l_partial)` via online softmax over `BN`-wide KV
    tiles.
  * `naive_fa_decode_apple_stitch` — stitch. Grid `(num_heads, batch_size)`,
    block `depth`. One thread per depth element; combines the contiguous
    per-partition partials into the final `output` with a log-sum-exp (LSE)
    reduction.

The host launcher `naive_fa_decode_apple` allocates the partials and enqueues
both kernels; `flash_attention_dispatch` selects it for Apple decode behind the
`MODULAR_ENABLE_APPLE_NAIVE_FA_DECODE` env flag (default off).

The online-softmax tiling and the LSE combine mirror the validated scratch
prototype; only the I/O contract and the head/GQA indexing change.

Partial-buffer layout (partition-last / contiguous):
  * `ml_idx(b, head, split) = (b*num_heads + head)*num_partitions + split`
  * `o_idx(b, head, d, split) = ((b*num_heads + head)*depth + d)*num_partitions
    + split`
"""

from std.collections import OptionalReg
from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv, exp
from std.memory import stack_allocation
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from layout import UNKNOWN_VALUE, Layout, LayoutTensor

from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand

comptime BN = 16  # KV keys per producer tile step; must be <= producer block width
comptime NEG_INF = Float32(-3.0e38)

# Max head_dim; routes larger dims to mha_gpu_naive.
comptime NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM = 256


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
# Producer: split-K online-softmax. Grid (num_partitions, batch, head).
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
    MaxDepth: Int,
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
    """Split-K online-softmax producer for Apple decode attention.

    Constraints:
        `MaxDepth >= depth` — shared tiles are sized by `MaxDepth`; a smaller
        `MaxDepth` overruns them.
    """
    debug_assert(
        depth <= MaxDepth,
        "naive_fa_decode_apple_core requires MaxDepth >= depth",
    )

    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype

    var split_id = Int(block_idx.x)
    var batch_id = Int(block_idx.y)
    var head_id = Int(block_idx.z)

    var kv_head = head_id // group

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

    # Attend span. When the cache length excludes the current tokens
    # (`_is_cache_length_accurate=False`), the new token's own KV sits at index
    # `cache_length`, so the span must include it: `cache_length + cur_query_len`.
    # Mirror `_bmm0_bs` (mha.mojo:5253-5274).
    var cache_len = k.cache_length(batch_id)
    var cur_cache_len: Int

    comptime if _is_cache_length_accurate:
        cur_cache_len = cur_query_len
    else:
        cur_cache_len = cache_len + cur_query_len

    var seq_len = cur_cache_len

    # Softmax state in shared memory. Math is Float32 (not the opaque `p_type`)
    # so `exp`/arithmetic are provably floating-point; partials cast to `p_type`
    # only at the final write.
    var m_shared = stack_allocation[
        1, Float32, address_space=AddressSpace.SHARED
    ]()
    var l_shared = stack_allocation[
        1, Float32, address_space=AddressSpace.SHARED
    ]()
    var o_shared = stack_allocation[
        MaxDepth, Float32, address_space=AddressSpace.SHARED
    ]()
    var q_shared = stack_allocation[
        MaxDepth, Scalar[q_type], address_space=AddressSpace.SHARED
    ]()

    if thread_idx.x == 0:
        m_shared[0] = NEG_INF
        l_shared[0] = Float32(0.0)

    var q = q_ptr + q_offset
    for i in range(Int(thread_idx.x), depth, Int(block_dim.x)):
        o_shared[i] = Float32(0.0)
        q_shared[i] = q.load[width=1](i)

    barrier()

    var start = split_id * SplitSize
    if start >= seq_len:
        return
    var end = min(start + SplitSize, seq_len)

    # One tile buffer, reused for the K tile then the V tile (K and V share a
    # storage dtype for supported caches).
    var tile_shared = stack_allocation[
        BN * MaxDepth, Scalar[k_type], address_space=AddressSpace.SHARED
    ]()
    var score_shared = stack_allocation[
        BN, Float32, address_space=AddressSpace.SHARED
    ]()
    var soft_shared = stack_allocation[
        BN, Float32, address_space=AddressSpace.SHARED
    ]()

    # Decode token's score-matrix row (== cache_length). Mirror mha.mojo:5324.
    var score_row = cur_cache_len - cur_query_len

    # `.cast[k_type]()` is a no-op for K and the real cast for V (shared tile
    # buffer); rows past `end` are zero-filled and masked out of the scores.
    @always_inline
    def load_kv_tile[
        operand_t: MHAOperand
    ](operand: operand_t, kv0: Int) {
        read tile_shared, read end, read batch_id, read kv_head, read depth
    }:
        for t in range(BN):
            var j = kv0 + t
            if j < end:
                var ptr = operand.block_paged_ptr[1](
                    UInt32(batch_id), UInt32(j), UInt32(kv_head), 0
                )
                for c in range(Int(thread_idx.x), depth, Int(block_dim.x)):
                    tile_shared[t * MaxDepth + c] = ptr.load[width=1](c).cast[
                        k_type
                    ]()
            else:
                for c in range(Int(thread_idx.x), depth, Int(block_dim.x)):
                    tile_shared[t * MaxDepth + c] = Scalar[k_type](0.0)

    for kv0 in range(start, end, BN):
        # ---------------- Load K tile ---------------- #
        load_kv_tile(k, kv0)
        barrier()

        # ---------------- Scores: Q . K^T ---------------- #
        # Requires `block_dim.x >= BN` so every score/soft slot is written
        # before the `[0, BN)` reductions below read them.
        for j in range(Int(thread_idx.x), BN, Int(block_dim.x)):
            var jg = kv0 + j
            if jg < end:
                var acc = Float32(0.0)
                var tile_row_off = j * MaxDepth
                for c in range(depth):
                    acc += (
                        q_shared[c].cast[DType.float32]()
                        * tile_shared[tile_row_off + c].cast[DType.float32]()
                    )
                score_shared[j] = mask_functor.mask(
                    Index(batch_id, head_id, score_row, jg),
                    (acc * scale),
                )
            else:
                score_shared[j] = NEG_INF
        barrier()

        # ---------------- Softmax part 1: running max + rescale ---------- #
        var m_tile = NEG_INF
        for i in range(0, BN):
            m_tile = max(m_tile, score_shared[i])
        var m_new = max(m_tile, m_shared[0])
        var alpha = exp(m_shared[0] - m_new)

        for j in range(Int(thread_idx.x), BN, Int(block_dim.x)):
            soft_shared[j] = exp(score_shared[j] - m_new)
        barrier()

        # ---------------- Softmax part 2: accumulate l, rescale o -------- #
        var l_tile = Float32(0.0)
        for j in range(0, BN):
            l_tile += soft_shared[j]

        for i in range(Int(thread_idx.x), depth, Int(block_dim.x)):
            o_shared[i] *= alpha
        if thread_idx.x == 0:
            l_shared[0] = alpha * l_shared[0] + l_tile
            m_shared[0] = m_new
        barrier()

        # ---------------- P dot V (load V tile, then accumulate) --------- #
        load_kv_tile(v, kv0)
        barrier()

        for i in range(Int(thread_idx.x), depth, Int(block_dim.x)):
            var acc = Float32(0.0)
            for j in range(BN):
                acc += (
                    soft_shared[j]
                    * tile_shared[j * MaxDepth + i].cast[DType.float32]()
                )
            o_shared[i] += acc
        barrier()

    # ---------------- Write partials (partition-last layout) ------------- #
    for i in range(Int(thread_idx.x), depth, Int(block_dim.x)):
        o_partial[
            _o_idx(
                batch_id, head_id, i, split_id, num_heads, depth, num_partitions
            )
        ] = o_shared[i].cast[p_type]()
    if thread_idx.x == 0:
        var idx = _ml_idx(
            batch_id, head_id, split_id, num_heads, num_partitions
        )
        l_partial[idx] = l_shared[0].cast[p_type]()
        m_partial[idx] = m_shared[0].cast[p_type]()


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
# signature; enqueues the producer/stitch pair instead of the BMM0/softmax/BMM1
# triple.
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

    comptime p_type = get_accum_type[q_type]()

    # `MaxDepth` bounds the shared query/output tiles (see core's Constraints);
    # `CORE_BLOCK` is the producer threadgroup width (>= BN so every score slot
    # is written before the BN-wide reductions); `SplitSize` is the per-partition
    # KV span.
    # head_dim gated at dispatcher; producer asserts invariant.
    comptime MaxDepth = NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM
    comptime SplitSize = 32
    comptime CORE_BLOCK = 64

    # Uniform allocation; per-sequence compaction would need a device->host
    # cache_lengths sync each step. Cover the producer's cur_cache_len span
    # (cache_length + cur_query_len): max_cache_size alone is one partition short
    # when the cache length is a multiple of SplitSize.
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

    # Pass q/output to the device kernels as raw pointers via non-owning
    # DeviceBuffers (mirror `_bmm0_bs`/`_bmm1_bs`, mha.mojo:5125-5128).
    var q_device = DeviceBuffer[q_type](ctx, q.ptr, q.size(), owning=False)
    var output_device = DeviceBuffer[output_type](
        ctx, output.ptr, output.size(), owning=False
    )

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
        MaxDepth=MaxDepth,
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
        block_dim=CORE_BLOCK,
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
