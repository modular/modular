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

    # Pass 1: per-channel max, for a stable softmax. Every member of a written
    # pool exists, so nothing needs masking.
    var m_max = -Float32.MAX
    for m in range(kpool):
        var logit = gate.raw_load((first_row + m) * head_dim + c).cast[
            .float32
        ]() + ape.raw_load(m * head_dim + c)
        m_max = max(m_max, logit)

    # Pass 2: weighted sum.
    var acc = Float32(0)
    var denom = Float32(0)
    for m in range(kpool):
        var logit = gate.raw_load((first_row + m) * head_dim + c).cast[
            .float32
        ]() + ape.raw_load(m * head_dim + c)
        var w = exp(logit - m_max)
        denom += w
        acc += w * k.raw_load((first_row + m) * head_dim + c).cast[.float32]()

    pooled.raw_store(pool_row * head_dim + c, (acc / denom).cast[dtype]())
