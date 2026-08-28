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
"""K-pool compression for the DSA indexer.

A k-pooled indexer stores one candidate key per `kpool` consecutive tokens
instead of one per token. A pooled key is a weighted average of its members,
where the weights come from a softmax over a gate score plus a learned
within-pool position embedding:

    logits[m, c]    = gate[member m, c] + ape[m, c]
    weights[:, c]   = softmax over m         # independently per channel c
    pooled[p, c]    = sum_m weights[m, c] * k[member m, c]

The softmax runs per channel, not per member. One weight per member would be a
different function.

Pool `p` covers absolute positions `[p * kpool, (p + 1) * kpool)` of one
request. Only pools whose members all arrive in the same call are written.
"""

from std.math import exp, max

from layout import TensorLayout, TileTensor

from std.gpu import block_idx, thread_idx


@always_inline
def pool_channel[
    kpool: Int
](logits: Array[Float32, kpool], vals: Array[Float32, kpool]) -> Float32:
    """One pooled channel: softmax over `logits`, weighted sum of `vals`.

    Shared by the prefill and decode writers so the two cannot drift.
    """
    var m_max = -Float32.MAX
    for m in range(kpool):
        m_max = max(m_max, logits[m])

    var acc = Float32(0)
    var denom = Float32(0)
    for m in range(kpool):
        var w = exp(logits[m] - m_max)
        denom += w
        acc += w * vals[m]
    return acc / denom


@always_inline
def _batch_of_pool_row(
    pool_row_offsets: TileTensor[mut=False, .uint32, ...],
    batch_size: Int,
    pool_row: Int,
) -> Int:
    """Batch owning output row `pool_row`."""
    for b in range(batch_size):
        if pool_row < Int(pool_row_offsets.raw_load(b + 1)):
            return b
    return batch_size - 1


@__name(t"mla_kpool_compress_{kpool}_{head_dim}")
def kpool_compress_kernel[
    dtype: DType,
    KLayoutType: TensorLayout,
    k_origin: ImmOrigin,
    GateLayoutType: TensorLayout,
    gate_origin: ImmOrigin,
    ApeLayoutType: TensorLayout,
    ape_origin: ImmOrigin,
    IROLayoutType: TensorLayout,
    iro_origin: ImmOrigin,
    PROLayoutType: TensorLayout,
    pro_origin: ImmOrigin,
    CacheLenLayoutType: TensorLayout,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    head_dim: Int,
    kpool: Int,
](
    pooled: TileTensor[dtype, OutLayoutType, out_origin],
    k: TileTensor[mut=False, dtype, KLayoutType, k_origin],
    gate: TileTensor[mut=False, dtype, GateLayoutType, gate_origin],
    ape: TileTensor[mut=False, .float32, ApeLayoutType, ape_origin],
    input_row_offsets: TileTensor[
        mut=False, .uint32, IROLayoutType, iro_origin
    ],
    pool_row_offsets: TileTensor[mut=False, .uint32, PROLayoutType, pro_origin],
    cache_lengths: TileTensor[
        mut=False, .uint32, CacheLenLayoutType, ImmutAnyOrigin
    ],
):
    """Builds one pooled key per block; one thread per channel.

    Parameters:
        dtype: Element type of `k`, `gate` and `pooled`.
        KLayoutType: Layout of `k`.
        k_origin: Origin of `k`.
        GateLayoutType: Layout of `gate`.
        gate_origin: Origin of `gate`.
        ApeLayoutType: Layout of `ape`.
        ape_origin: Origin of `ape`.
        IROLayoutType: Layout of `input_row_offsets`.
        iro_origin: Origin of `input_row_offsets`.
        PROLayoutType: Layout of `pool_row_offsets`.
        pro_origin: Origin of `pool_row_offsets`.
        CacheLenLayoutType: Layout of `cache_lengths`.
        OutLayoutType: Layout of `pooled`.
        out_origin: Origin of `pooled`.
        head_dim: Channels per key; also the block width.
        kpool: Tokens per pool.

    Args:
        pooled: Output `[total_pools, head_dim]`, where `total_pools` is the
            last entry of `pool_row_offsets`.
        k: Layer-normed indexer keys, `[total_tokens, head_dim]`.
        gate: Per-token gate scores, `[total_tokens, head_dim]`.
        ape: Within-pool position embedding, `[kpool, head_dim]`, f32.
        input_row_offsets: Token row offsets per request, `[batch_size + 1]`.
        pool_row_offsets: Output row offsets per request, `[batch_size + 1]`.
            Request `b` owns output rows `[pool_row_offsets[b],
            pool_row_offsets[b + 1])`, one per pool built here.
        cache_lengths: Cached-prefix length per request, `[batch_size]`. A
            pool covers absolute positions, so this is what places the call's
            tokens on the pool grid.
    """
    var pool_row = block_idx.x
    # The last offset is the exact pool count, so an over-sized grid is
    # harmless.
    var batch_size = Int(pool_row_offsets.dim[0]()) - 1
    if pool_row >= Int(pool_row_offsets.raw_load(batch_size)):
        return

    var c = thread_idx.x
    if c >= head_dim:
        return

    var b = _batch_of_pool_row(pool_row_offsets, batch_size, pool_row)

    var local_pool = pool_row - Int(pool_row_offsets.raw_load(b))
    # A cached prefix ending mid-pool means this call opens partway through a
    # pool it cannot build. Skip those leading tokens; the tail ring carries
    # that pool.
    var cache_len = Int(cache_lengths[b])
    var align = (kpool - cache_len % kpool) % kpool
    var first_row = (
        Int(input_row_offsets.raw_load(b)) + align + local_pool * kpool
    )

    # Every member of a written pool exists, so nothing needs masking.
    var logits = Array[Float32, kpool]()
    var vals = Array[Float32, kpool]()
    for m in range(kpool):
        logits[m] = gate.raw_load((first_row + m) * head_dim + c).cast[
            .float32
        ]() + ape.raw_load(m * head_dim + c)
        vals[m] = k.raw_load((first_row + m) * head_dim + c).cast[.float32]()

    pooled.raw_store(
        pool_row * head_dim + c, pool_channel[kpool](logits, vals).cast[dtype]()
    )


@__name(t"mla_kpool_tail_update_{kpool}_{head_dim}")
def kpool_tail_update_kernel[
    dtype: DType,
    TailLayoutType: TensorLayout,
    tail_origin: MutOrigin,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    ClosedLayoutType: TensorLayout,
    closed_origin: MutOrigin,
    KLayoutType: TensorLayout,
    k_origin: ImmOrigin,
    GateLayoutType: TensorLayout,
    gate_origin: ImmOrigin,
    ApeLayoutType: TensorLayout,
    ape_origin: ImmOrigin,
    PosLayoutType: TensorLayout,
    pos_origin: ImmOrigin,
    SlotLayoutType: TensorLayout,
    slot_origin: ImmOrigin,
    head_dim: Int,
    kpool: Int,
](
    tail: TileTensor[dtype, TailLayoutType, tail_origin],
    pooled: TileTensor[dtype, OutLayoutType, out_origin],
    closed_pool: TileTensor[.int32, ClosedLayoutType, closed_origin],
    k: TileTensor[mut=False, dtype, KLayoutType, k_origin],
    gate: TileTensor[mut=False, dtype, GateLayoutType, gate_origin],
    ape: TileTensor[mut=False, .float32, ApeLayoutType, ape_origin],
    positions: TileTensor[mut=False, .int32, PosLayoutType, pos_origin],
    slot_idx: TileTensor[mut=False, .uint32, SlotLayoutType, slot_origin],
    num_requests: Int32,
):
    """Stashes one decoded token per request, and closes a pool when it fills.

    A decoded token cannot be pooled on arrival, because its pool-mates arrived
    on earlier steps and have left the batch. Each request keeps its
    in-progress pool in `tail`, a ring of `kpool` slots addressed by
    `position % kpool`.

    The ring is indexed by `slot_idx[r]`, not by `r`. A batch reorders between
    steps, so row `r` is not always the same request.

    Every real token stashes, whether or not it closes a pool.

    Handles one token per request per call. A speculative step that verifies
    several at once must walk them in position order, which this does not do.

    Parameters:
        dtype: Element type of `tail`, `k`, `gate` and `pooled`.
        TailLayoutType: Layout of `tail`.
        tail_origin: Origin of `tail`.
        OutLayoutType: Layout of `pooled`.
        out_origin: Origin of `pooled`.
        ClosedLayoutType: Layout of `closed_pool`.
        closed_origin: Origin of `closed_pool`.
        KLayoutType: Layout of `k`.
        k_origin: Origin of `k`.
        GateLayoutType: Layout of `gate`.
        gate_origin: Origin of `gate`.
        ApeLayoutType: Layout of `ape`.
        ape_origin: Origin of `ape`.
        PosLayoutType: Layout of `positions`.
        pos_origin: Origin of `positions`.
        SlotLayoutType: Layout of `slot_idx`.
        slot_origin: Origin of `slot_idx`.
        head_dim: Channels per key; also the block width.
        kpool: Tokens per pool.

    Args:
        tail: Per-slot ring, `[max_slots, 2, kpool, head_dim]`, sized by the
            engine's concurrent-request capacity. Index 0 holds keys, index 1
            holds gate scores. Only slot `slot_idx[r]` is touched for row `r`.
            Persists across steps.
        pooled: Output `[num_requests, head_dim]`, meaningful only where
            `closed_pool` is non-negative.
        closed_pool: Output `[num_requests]`. The pool id this step completed,
            or -1 when the request's pool is still filling.
        k: This step's layer-normed keys, `[num_requests, head_dim]`.
        gate: This step's gate scores, `[num_requests, head_dim]`.
        ape: Within-pool position embedding, `[kpool, head_dim]`, f32.
        positions: Absolute position of each request's new token,
            `[num_requests]`. Negative marks a padded batch entry.
        slot_idx: Ring slot owned by each batch row, `[num_requests]`, `uint32`.
        num_requests: Requests actually present.
    """
    var r = block_idx.x
    if r >= Int(num_requests):
        return

    var c = thread_idx.x
    if c >= head_dim:
        return

    var pos = Int(positions.raw_load(r))
    if pos < 0:
        if c == 0:
            closed_pool.raw_store(r, Int32(-1))
        return

    var slot = pos % kpool
    var tail_base = Int(slot_idx.raw_load(r)) * 2 * kpool * head_dim

    # This write and the reads below are the same thread's, so no barrier is
    # needed.
    tail.raw_store(
        tail_base + slot * head_dim + c, k.raw_load(r * head_dim + c)
    )
    tail.raw_store(
        tail_base + (kpool + slot) * head_dim + c,
        gate.raw_load(r * head_dim + c),
    )

    if slot != kpool - 1:
        if c == 0:
            closed_pool.raw_store(r, Int32(-1))
        return

    # The pool closing here spans positions `pos - kpool + 1` through `pos`,
    # so member `m` sits at slot `m` and pairs with `ape[m]`.
    var logits = Array[Float32, kpool]()
    var vals = Array[Float32, kpool]()
    for m in range(kpool):
        logits[m] = tail.raw_load(tail_base + (kpool + m) * head_dim + c).cast[
            .float32
        ]() + ape.raw_load(m * head_dim + c)
        vals[m] = tail.raw_load(tail_base + m * head_dim + c).cast[.float32]()

    pooled.raw_store(
        r * head_dim + c, pool_channel[kpool](logits, vals).cast[dtype]()
    )
    if c == 0:
        closed_pool.raw_store(r, Int32(pos // kpool))
