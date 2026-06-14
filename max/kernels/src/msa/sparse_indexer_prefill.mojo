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
"""Prefill-path MiniMax-M3 sparse-attention (MSA) indexer.

For each (ragged) query token and index head, this selects the top-k key
*blocks* to attend to. It runs as two launches:

1. `_prefill_block_score_kernel` -- one CTA per (query token, index head). The
   query token's causal key count is `prefix_len + local_index + 1`, so each
   thread takes the max over a block's in-range causal keys of `q . k * sm_scale`
   (bf16 inputs, f32 accumulation), applies init/local forcing, and writes one
   f32 per block into a caller-owned score buffer. Clamping the key range to the
   causal count makes the diagonal (final) block exact without a separate mask.
2. `_prefill_topk_kernel` -- one CTA per (query token, index head). Selects the
   top-k blocks from the score row via `block_select_topk`.

Queries are ragged: `input_row_offsets[b]` gives the start of batch `b`'s tokens.
Selection-only (M3 disables the index value/output on every sparse layer); score
type is `max`.
"""

from std.gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv, max, min
from std.memory import stack_allocation
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf

from layout import TensorLayout, TileTensor

from nn.attention.gpu.sparse_indexer_common import block_select_topk
from nn.attention.mha_operand import MHAOperand


@always_inline
def _token_batch(
    t: Int,
    batch: Int,
    input_row_offsets: UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin],
) -> Int:
    """Return the batch index owning ragged query token `t`."""
    var b = 0
    for bi in range(batch):
        if t < Int(input_row_offsets[bi + 1]):
            b = bi
            break
    return b


@__name(t"sparse_indexer_prefill_score_{dtype}")
def _prefill_block_score_kernel[
    dtype: DType,
    KOperand: MHAOperand,
    QLT: TensorLayout,
    IROLT: TensorLayout,
    PLLT: TensorLayout,
    ScoreLT: TensorLayout,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[
        dtype, QLT, ImmutAnyOrigin
    ],  # [total_q, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    input_row_offsets: TileTensor[
        DType.uint32, IROLT, ImmutAnyOrigin
    ],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, PLLT, ImmutAnyOrigin],  # [batch]
    score: TileTensor[
        DType.float32, ScoreLT, MutAnyOrigin
    ],  # [num_index_heads, total_q, max_num_blocks]
    batch: Int,
    max_num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
):
    comptime assert (
        input_row_offsets.flat_rank == 1 and prefix_lens.flat_rank == 1
    )
    comptime INIT_SCORE = Float32(1.0e30)
    comptime LOCAL_SCORE = Float32(1.0e29)
    var t = block_idx.x
    var h = block_idx.y
    var tid = thread_idx.x
    var bsize = block_dim.x

    var iro_ptr = input_row_offsets.to_layout_tensor().ptr_at_offset(Index(0))
    var b = _token_batch(
        t,
        batch,
        rebind[UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin]](iro_ptr),
    )
    var local_idx = t - Int(input_row_offsets[b])
    # Causal: this query attends to keys [0, prefix_len + local_idx].
    var num_keys = Int(prefix_lens[b]) + local_idx + 1
    var num_blocks = ceildiv(num_keys, block_size)
    var local_start = max(0, num_blocks - local_blocks)

    var q_base = q.to_layout_tensor().ptr_at_offset(Index(t, h, 0))
    var q_smem = stack_allocation[
        idx_head_dim, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    for d in range(tid, idx_head_dim, bsize):
        q_smem[d] = q_base[d].cast[DType.float32]()
    barrier()

    var score_row = score.to_layout_tensor().ptr_at_offset(Index(h, t, 0))

    for blk in range(tid, num_blocks, bsize):
        var key_start = blk * block_size
        var key_end = min(key_start + block_size, num_keys)
        var blk_max = min_or_neg_inf[DType.float32]()
        for key in range(key_start, key_end):
            var k_ptr = k_operand.block_paged_ptr[1](
                UInt32(b), UInt32(key), UInt32(0), UInt32(0)
            )
            var dot = Float32(0)
            for d in range(idx_head_dim):
                dot += q_smem[d] * k_ptr[d].cast[DType.float32]()
            var s = dot * sm_scale
            if s > blk_max:
                blk_max = s

        # Forcing: local (1e29) takes priority over init (1e30). Selection is by
        # value, so a forced block always wins a top-k slot.
        var val = blk_max
        if blk < init_blocks:
            val = INIT_SCORE
        if blk >= local_start:
            val = LOCAL_SCORE
        score_row[blk] = val


@__name(t"sparse_indexer_prefill_topk")
def _prefill_topk_kernel[
    IROLT: TensorLayout,
    PLLT: TensorLayout,
    ScoreLT: TensorLayout,
    OutLT: TensorLayout,
    block_size: Int,
](
    input_row_offsets: TileTensor[
        DType.uint32, IROLT, ImmutAnyOrigin
    ],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, PLLT, ImmutAnyOrigin],  # [batch]
    score: TileTensor[
        DType.float32, ScoreLT, MutAnyOrigin
    ],  # [num_index_heads, total_q, max_num_blocks]
    out_idxs: TileTensor[
        DType.int32, OutLT, MutAnyOrigin
    ],  # [num_index_heads, total_q, topk]
    batch: Int,
    max_num_blocks: Int,
    topk: Int,
):
    comptime assert (
        input_row_offsets.flat_rank == 1 and prefix_lens.flat_rank == 1
    )
    var t = block_idx.x
    var h = block_idx.y

    var iro_ptr = input_row_offsets.to_layout_tensor().ptr_at_offset(Index(0))
    var b = _token_batch(
        t,
        batch,
        rebind[UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin]](iro_ptr),
    )
    var local_idx = t - Int(input_row_offsets[b])
    var num_keys = Int(prefix_lens[b]) + local_idx + 1
    var num_blocks = ceildiv(num_keys, block_size)

    var score_row = score.to_layout_tensor().ptr_at_offset(Index(h, t, 0))
    var out_row = out_idxs.to_layout_tensor().ptr_at_offset(Index(h, t, 0))
    block_select_topk[DType.float32, DType.int32](
        rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](score_row),
        num_blocks,
        topk,
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](out_row),
    )


@always_inline
def sparse_indexer_prefill_score[
    dtype: DType,
    KOperand: MHAOperand,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[dtype, ...],  # [total_q, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    input_row_offsets: TileTensor[DType.uint32, ...],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, total_q, max_num_blocks]
    batch: Int,
    total_q: Int,
    max_num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Launch the prefill block-scoring kernel into `score`.

    See `sparse_indexer_prefill` for the argument contract. Exposed separately so
    tests can drive scoring and selection independently.
    """
    comptime BLOCK_DIM = 128
    comptime assert (
        BLOCK_DIM % WARP_SIZE == 0
    ), "block_dim must be a multiple of the warp size"
    comptime score_kernel = _prefill_block_score_kernel[
        dtype,
        KOperand,
        type_of(q).LayoutType,
        type_of(input_row_offsets).LayoutType,
        type_of(prefix_lens).LayoutType,
        type_of(score).LayoutType,
        num_index_heads,
        idx_head_dim,
        block_size,
    ]
    ctx.enqueue_function[score_kernel](
        q.as_immut(),
        k_operand,
        input_row_offsets.as_immut(),
        prefix_lens.as_immut(),
        score,
        batch,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        grid_dim=(total_q, num_index_heads),
        block_dim=BLOCK_DIM,
    )


@always_inline
def sparse_indexer_prefill_topk[
    num_index_heads: Int,
    block_size: Int,
](
    input_row_offsets: TileTensor[DType.uint32, ...],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, total_q, max_num_blocks]
    out_idxs: TileTensor[DType.int32, ...],  # [num_index_heads, total_q, topk]
    batch: Int,
    total_q: Int,
    max_num_blocks: Int,
    topk: Int,
    ctx: DeviceContext,
) raises:
    """Launch the prefill top-k selection kernel from `score` into `out_idxs`.

    See `sparse_indexer_prefill` for the argument contract. Exposed separately so
    tests can drive scoring and selection independently.
    """
    comptime BLOCK_DIM = 128
    comptime assert (
        BLOCK_DIM % WARP_SIZE == 0
    ), "block_select_topk requires block_dim to be a multiple of the warp size"
    comptime topk_kernel = _prefill_topk_kernel[
        type_of(input_row_offsets).LayoutType,
        type_of(prefix_lens).LayoutType,
        type_of(score).LayoutType,
        type_of(out_idxs).LayoutType,
        block_size,
    ]
    ctx.enqueue_function[topk_kernel](
        input_row_offsets.as_immut(),
        prefix_lens.as_immut(),
        score,
        out_idxs,
        batch,
        max_num_blocks,
        topk,
        grid_dim=(total_q, num_index_heads),
        block_dim=BLOCK_DIM,
    )


@always_inline
def sparse_indexer_prefill[
    dtype: DType,
    KOperand: MHAOperand,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[dtype, ...],  # [total_q, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    input_row_offsets: TileTensor[DType.uint32, ...],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, total_q, max_num_blocks]
    out_idxs: TileTensor[DType.int32, ...],  # [num_index_heads, total_q, topk]
    batch: Int,
    total_q: Int,
    max_num_blocks: Int,
    topk: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Compute MSA top-k block indices for a prefill step (selection only).

    Args:
        q: Query tensor `[total_q, num_index_heads, idx_head_dim]` (normed +
            roped by the caller), ragged across the batch.
        k_operand: Index-K cache as an `MHAOperand` (single head).
        input_row_offsets: Ragged query-token offsets `[batch + 1]`.
        prefix_lens: Per-batch cached-key count preceding the query tokens
            `[batch]` (0 for a fresh prefill).
        score: Caller-owned scratch `[num_index_heads, total_q, max_num_blocks]`;
            written then consumed (and mutated) by the two launches.
        out_idxs: Output block indices `[num_index_heads, total_q, topk]`, int32,
            `-1`-padded.
        batch: Batch size.
        total_q: Total ragged query tokens (`input_row_offsets[batch]`).
        max_num_blocks: Row stride of `score` (>= every per-token block count).
        topk: Number of blocks to select.
        init_blocks: Always-keep leading blocks (forced score 1e30).
        local_blocks: Always-keep trailing/local blocks (forced score 1e29).
        sm_scale: QK scale.
        ctx: Device context.
    """
    sparse_indexer_prefill_score[
        dtype, KOperand, num_index_heads, idx_head_dim, block_size
    ](
        q,
        k_operand,
        input_row_offsets,
        prefix_lens,
        score,
        batch,
        total_q,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        ctx,
    )
    sparse_indexer_prefill_topk[num_index_heads, block_size](
        input_row_offsets,
        prefix_lens,
        score,
        out_idxs,
        batch,
        total_q,
        max_num_blocks,
        topk,
        ctx,
    )
