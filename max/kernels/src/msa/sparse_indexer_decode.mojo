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
"""Decode-path MiniMax-M3 sparse-attention (MSA) indexer (selection only).

Per decode query (one token per batch element) and index head, selects the top-k
key *blocks* via two launches:

1. `_decode_block_score_kernel` -- block-max `q . k * sm_scale` with init/local
   forcing, split-K over the KV-block dimension; writes a caller-owned scores buffer.
2. `_decode_topk_kernel` -- `block_select_topk` over each score row.

Both grids depend only on graph-constant shapes, never on sequence length, and
nothing is allocated inside the op, so the decode path is safe inside a
CUDA-graph capture region. M3 disables the index value/output, so
this emits block indices only (score type `max`).
"""

from std.collections import InlineArray
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.math import ceildiv, clamp, max, min
from std.memory import stack_allocation
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf
from std.utils.static_tuple import StaticTuple

comptime _SCORE_CTA_SIZE = 128

from layout import TensorLayout, TileTensor

from nn.attention.gpu.sparse_indexer_common import block_select_topk
from nn.attention.mha_operand import MHAOperand


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_SCORE_CTA_SIZE))
)
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
    var chunk_id = block_idx.y
    var tid = thread_idx.x

    var num_keys = Int(seq_lens[b])
    var num_blocks = ceildiv(num_keys, block_size)
    var local_start = max(0, num_blocks - local_blocks)

    # Each block is written by exactly one chunk, so the score writes are race-free
    # without a cross-CTA reduction. chunk_start/chunk_end are CTA-uniform, so this
    # early-return cannot deadlock the barrier below.
    var chunk_blocks = ceildiv(num_blocks, num_chunks)
    var chunk_start = chunk_id * chunk_blocks
    var chunk_end = min(chunk_start + chunk_blocks, num_blocks)
    if chunk_start >= num_blocks:
        return

    var q_base = q.to_layout_tensor().ptr_at_offset(Index(b, 0, 0))
    var q_smem = stack_allocation[
        num_index_heads * idx_head_dim,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    for i in range(tid, num_index_heads * idx_head_dim, _SCORE_CTA_SIZE):
        q_smem[i] = q_base[i].cast[DType.float32]()
    barrier()

    comptime num_warps = _SCORE_CTA_SIZE // WARP_SIZE
    var warp_in_cta = warp_id()
    var lane = lane_id()

    # Splitting each dot across LANES_PER_KEY lanes reorders the f32 accumulation,
    # so scores are within f32 tolerance of (not bit-identical to) a per-lane dot.
    comptime LANES_PER_KEY = max(1, min(WARP_SIZE, idx_head_dim // 16))
    comptime assert (
        WARP_SIZE % LANES_PER_KEY == 0 and idx_head_dim % LANES_PER_KEY == 0
    ), "LANES_PER_KEY must divide both the warp size and idx_head_dim"
    comptime LANE_SIMD = idx_head_dim // LANES_PER_KEY
    comptime KEYS_PER_ITER = WARP_SIZE // LANES_PER_KEY

    var key_in_group = lane // LANES_PER_KEY
    var d0 = (lane % LANES_PER_KEY) * LANE_SIMD
    var q_reg = InlineArray[SIMD[DType.float32, LANE_SIMD], num_index_heads](
        uninitialized=True
    )
    comptime for h in range(num_index_heads):
        q_reg[h] = (q_smem + h * idx_head_dim + d0).load[width=LANE_SIMD]()

    for blk in range(chunk_start + warp_in_cta, chunk_end, num_warps):
        var key_start = blk * block_size
        var key_end = min(key_start + block_size, num_keys)
        var lane_max = SIMD[DType.float32, num_index_heads](
            min_or_neg_inf[DType.float32]()
        )
        var n_iter = ceildiv(key_end - key_start, KEYS_PER_ITER)
        for it in range(n_iter):
            var key = key_start + Int(key_in_group) + it * KEYS_PER_ITER
            var k_vec = SIMD[DType.float32, LANE_SIMD](0)
            if key < key_end:
                var k_ptr = k_operand.block_paged_ptr[1](
                    UInt32(b), UInt32(key), UInt32(0), UInt32(0)
                )
                k_vec = (
                    (k_ptr + d0).load[width=LANE_SIMD]().cast[DType.float32]()
                )
            comptime for h in range(num_index_heads):
                var dot = warp.lane_group_sum[num_lanes=LANES_PER_KEY](
                    SIMD[DType.float32, 1]((q_reg[h] * k_vec).reduce_add())
                )[0]
                if key < key_end:
                    var s = dot * sm_scale
                    if s > lane_max[h]:
                        lane_max[h] = s

        comptime for h in range(num_index_heads):
            var blk_max = warp.max(SIMD[DType.float32, 1](lane_max[h]))[0]

            # When a block is both init- and local-forced, local wins: the second
            # assignment overwriting the first is intentional, not a clobber bug.
            if lane == 0:
                var val = blk_max
                if blk < init_blocks:
                    val = INIT_SCORE
                if blk >= local_start:
                    val = LOCAL_SCORE
                score.to_layout_tensor().ptr_at_offset(Index(h, b, 0))[
                    blk
                ] = val


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
    comptime assert (
        _SCORE_CTA_SIZE % WARP_SIZE == 0
    ), "block_dim must be a multiple of the warp size"

    # CAPTURE-SAFETY: num_chunks must depend only on the graph-constant batch,
    # never on seq_len/num_keys (which vary call-to-call inside a CUDA-graph
    # capture); deriving it from the live block count would silently break capture.
    comptime TARGET_GRID = 512  # ~3.5x the B200 SM count
    comptime MAX_CHUNKS = 512
    var num_chunks = clamp(TARGET_GRID // max(1, batch), 1, MAX_CHUNKS)

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
        grid_dim=(batch, num_chunks),
        block_dim=_SCORE_CTA_SIZE,
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
