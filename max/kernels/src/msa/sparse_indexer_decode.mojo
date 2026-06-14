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
"""Decode-path MiniMax-M3 sparse-attention (MSA) indexer.

For each decode query (one token per batch element) and each index head, this
selects the top-k key *blocks* to attend to. It runs as two launches:

1. `_decode_block_score_kernel` -- one CTA per (batch, index head, block chunk).
   The block range `[0, num_blocks)` is split into `num_chunks` contiguous,
   block-aligned slices across the `block_idx.z` grid axis; each CTA owns one
   slice. Within a slice, each thread owns a strided subset of blocks; for each
   block it takes the max over that block's `q . k * sm_scale` (bf16 inputs, f32
   accumulation) and applies init/local forcing, writing one f32 per block into
   a caller-owned score buffer. Only in-range keys are read, so out-of-range
   keys in a partial final block cannot contaminate the block max. Blocks are
   independent and each is written by exactly one chunk, so there is no
   cross-CTA reduction. `num_chunks` is derived from graph-constant quantities
   (batch, head count) only, so empty chunks early-return and the grid stays
   capture-safe.
2. `_decode_topk_kernel` -- one CTA per (index head, batch). Selects the top-k
   blocks from the score row via `block_select_topk`.

Both grids depend only on graph-constant shapes (batch, head count) -- never on
sequence length -- and nothing is allocated inside the op (the score buffer is
passed in). That keeps the decode path safe inside a CUDA-graph capture region.

Selection-only: M3 disables the index value/output on every sparse layer, so
this emits block indices only (no attention output). Score type is `max`.
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


@__name(t"sparse_indexer_decode_score_{dtype}")
def _decode_block_score_kernel[
    dtype: DType,
    KOperand: MHAOperand,
    QLT: TensorLayout,
    SLLT: TensorLayout,
    ScoreLT: TensorLayout,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[
        dtype, QLT, ImmutAnyOrigin
    ],  # [batch, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    seq_lens: TileTensor[DType.uint32, SLLT, ImmutAnyOrigin],  # [batch]
    score: TileTensor[
        DType.float32, ScoreLT, MutAnyOrigin
    ],  # [num_index_heads, batch, max_num_blocks]
    batch: Int,
    max_num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    num_chunks: Int,
):
    comptime assert seq_lens.flat_rank == 1
    comptime INIT_SCORE = Float32(1.0e30)
    comptime LOCAL_SCORE = Float32(1.0e29)
    var b = block_idx.x
    var h = block_idx.y
    var chunk_id = block_idx.z
    var tid = thread_idx.x
    var bsize = block_dim.x

    var num_keys = Int(seq_lens[b])
    # num_keys == 0 needs no special case: num_blocks is 0, so the chunk range
    # below is empty and nothing is written (the top-k kernel reads 0 blocks).
    var num_blocks = ceildiv(num_keys, block_size)
    var local_start = max(0, num_blocks - local_blocks)

    # Each block belongs to exactly one chunk, so these writes need no cross-CTA
    # reduction. The chunk range is CTA-uniform, so the early-return is uniform
    # (no barrier deadlock) and empty chunks cost ~nothing.
    var chunk_blocks = ceildiv(num_blocks, num_chunks)
    var chunk_start = chunk_id * chunk_blocks
    var chunk_end = min(chunk_start + chunk_blocks, num_blocks)
    if chunk_start >= num_blocks:
        return

    var q_base = q.to_layout_tensor().ptr_at_offset(Index(b, h, 0))
    var q_smem = stack_allocation[
        idx_head_dim, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    for d in range(tid, idx_head_dim, bsize):
        q_smem[d] = q_base[d].cast[DType.float32]()
    barrier()

    var score_row = score.to_layout_tensor().ptr_at_offset(Index(h, b, 0))

    for blk in range(chunk_start + tid, chunk_end, bsize):
        var key_start = blk * block_size
        var key_end = min(key_start + block_size, num_keys)
        # Every block in [0, num_blocks) has >= 1 in-range key, so this max is
        # over real keys only and is always finite.
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


@__name(t"sparse_indexer_decode_topk")
def _decode_topk_kernel[
    SLLT: TensorLayout,
    ScoreLT: TensorLayout,
    OutLT: TensorLayout,
    block_size: Int,
](
    seq_lens: TileTensor[DType.uint32, SLLT, ImmutAnyOrigin],  # [batch]
    score: TileTensor[
        DType.float32, ScoreLT, MutAnyOrigin
    ],  # [num_index_heads, batch, max_num_blocks]
    out_idxs: TileTensor[
        DType.int32, OutLT, MutAnyOrigin
    ],  # [num_index_heads, batch, topk]
    batch: Int,
    max_num_blocks: Int,
    topk: Int,
):
    comptime assert seq_lens.flat_rank == 1
    var h = block_idx.x
    var b = block_idx.y

    var num_keys = Int(seq_lens[b])
    var num_blocks = ceildiv(num_keys, block_size)

    var score_row = score.to_layout_tensor().ptr_at_offset(Index(h, b, 0))
    var out_row = out_idxs.to_layout_tensor().ptr_at_offset(Index(h, b, 0))
    block_select_topk[DType.float32, DType.int32](
        rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](score_row),
        num_blocks,
        topk,
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](out_row),
    )


@always_inline
def sparse_indexer_decode_score[
    dtype: DType,
    KOperand: MHAOperand,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[dtype, ...],  # [batch, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    seq_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, batch, max_num_blocks]
    batch: Int,
    max_num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Launch the decode block-scoring kernel into `score`.

    See `sparse_indexer_decode` for the argument contract. Exposed separately so
    tests can drive scoring and selection independently.
    """
    comptime BLOCK_DIM = 128
    comptime assert (
        BLOCK_DIM % WARP_SIZE == 0
    ), "block_dim must be a multiple of the warp size"

    # Split-K so low-batch launches fill the GPU: the un-split grid is only
    # `batch * num_index_heads` CTAs (4 at batch=1) on a 148-SM B200.
    #
    # CAPTURE-SAFETY: `num_chunks` depends only on graph-constant quantities
    # (`batch`, `num_index_heads`), never on `seq_len`/`num_keys` (which vary
    # call-to-call inside a CUDA-graph capture). The block range is computed
    # on-device and chunks past `num_blocks` early-return, so any chunk count is
    # valid -- each block is written by exactly one chunk, no reduction.
    comptime TARGET_GRID = 512  # ~3.5x the B200 SM count
    comptime MAX_CHUNKS = 128
    var num_chunks = max(
        1, min(MAX_CHUNKS, TARGET_GRID // max(1, batch * num_index_heads))
    )

    comptime score_kernel = _decode_block_score_kernel[
        dtype,
        KOperand,
        type_of(q).LayoutType,
        type_of(seq_lens).LayoutType,
        type_of(score).LayoutType,
        num_index_heads,
        idx_head_dim,
        block_size,
    ]
    ctx.enqueue_function[score_kernel](
        q.as_immut(),
        k_operand,
        seq_lens.as_immut(),
        score,
        batch,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        num_chunks,
        grid_dim=(batch, num_index_heads, num_chunks),
        block_dim=BLOCK_DIM,
    )


@always_inline
def sparse_indexer_decode_topk[
    num_index_heads: Int,
    block_size: Int,
](
    seq_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, batch, max_num_blocks]
    out_idxs: TileTensor[DType.int32, ...],  # [num_index_heads, batch, topk]
    batch: Int,
    max_num_blocks: Int,
    topk: Int,
    ctx: DeviceContext,
) raises:
    """Launch the decode top-k selection kernel from `score` into `out_idxs`.

    See `sparse_indexer_decode` for the argument contract. Exposed separately so
    tests can drive scoring and selection independently.
    """
    comptime BLOCK_DIM = 128
    comptime assert (
        BLOCK_DIM % WARP_SIZE == 0
    ), "block_select_topk requires block_dim to be a multiple of the warp size"
    comptime topk_kernel = _decode_topk_kernel[
        type_of(seq_lens).LayoutType,
        type_of(score).LayoutType,
        type_of(out_idxs).LayoutType,
        block_size,
    ]
    ctx.enqueue_function[topk_kernel](
        seq_lens.as_immut(),
        score,
        out_idxs,
        batch,
        max_num_blocks,
        topk,
        grid_dim=(num_index_heads, batch),
        block_dim=BLOCK_DIM,
    )


@always_inline
def sparse_indexer_decode[
    dtype: DType,
    KOperand: MHAOperand,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[dtype, ...],  # [batch, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    seq_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, batch, max_num_blocks]
    out_idxs: TileTensor[DType.int32, ...],  # [num_index_heads, batch, topk]
    batch: Int,
    max_num_blocks: Int,
    topk: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Compute MSA top-k block indices for a decode step (selection only).

    All buffers are caller-owned; nothing is allocated here, and the launch
    grids depend only on `batch` / `num_index_heads`, so this is safe to call
    inside a CUDA-graph capture region.

    Args:
        q: Query tensor `[batch, num_index_heads, idx_head_dim]` (normed +
            roped by the caller).
        k_operand: Index-K cache as an `MHAOperand` (single head).
        seq_lens: Per-batch key counts `[batch]`.
        score: Caller-owned scratch `[num_index_heads, batch, max_num_blocks]`;
            written then consumed (and mutated) by the two launches.
        out_idxs: Output block indices `[num_index_heads, batch, topk]`, int32,
            `-1`-padded.
        batch: Batch size.
        max_num_blocks: Row stride of `score` (>= every per-row block count).
        topk: Number of blocks to select.
        init_blocks: Always-keep leading blocks (forced score 1e30).
        local_blocks: Always-keep trailing/local blocks (forced score 1e29).
        sm_scale: QK scale.
        ctx: Device context.
    """
    sparse_indexer_decode_score[
        dtype, KOperand, num_index_heads, idx_head_dim, block_size
    ](
        q,
        k_operand,
        seq_lens,
        score,
        batch,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        ctx,
    )
    sparse_indexer_decode_topk[num_index_heads, block_size](
        seq_lens,
        score,
        out_idxs,
        batch,
        max_num_blocks,
        topk,
        ctx,
    )
