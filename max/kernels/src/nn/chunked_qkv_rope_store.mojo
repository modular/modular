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
"""Fuses chunked QKV unpacking, partial NeoX RoPE, and paged cache writes."""

from std.builtin.device_passable import DevicePassable
from std.math import ceildiv
from std.sys import align_of
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from kv_cache.types import PagedKVCache, PagedKVCacheCollection
from layout import TileTensor
from nn._ragged_utils import get_batch_from_row_offsets
from nn.fused_qk_rope import get_safetensors_idx, rope_value


# HACK: K/V views share the collection's mutable origins but write disjoint
# planes, which the exclusivity checker cannot prove. As in
# `_rope_split_store_ragged_impl`, disable nested-origin checks until the K/V
# views have provably disjoint origins. This also covers mutable `q_output`,
# which must be disjoint from the cache planes and projection inputs.
@__unsafe_nested_origins_read_only
def chunked_qkv_rope_store_kernel[
    qk_head_dim: Int, v_head_dim: Int, chunk_padding: Int = 0
](
    qkv: TileTensor[mut=False, .bfloat16, ...],
    row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs: TileTensor[mut=False, .bfloat16, ...],
    k_cache: PagedKVCache,
    v_cache: type_of(k_cache),
    q_output: TileTensor[mut=True, .bfloat16, ...],
    value_scale: Float32,
):
    """Rotates Q/K and writes scaled V from contiguous per-chunk projections.

    Query output rows at or past the final ragged offset are left unwritten.

    Parameters:
        qk_head_dim: Unpadded Q/K head width.
        v_head_dim: Unpadded V head width.
        chunk_padding: Trailing projection rows to ignore in each chunk.

    Args:
        qkv: BF16 projection [tokens, chunks, chunk_width].
        row_offsets: Ragged offsets [batch_size + 1].
        freqs: Interleaved BF16 RoPE coefficients for NeoX-paired inputs.
        k_cache: Paged key view for the selected layer.
        v_cache: Disjoint paged value view for the same layer.
        q_output: Padded rotated queries [tokens, query_heads, head_size].
        value_scale: FP32 multiplier applied before V's BF16 rounding.

    Constraints:
        Projections, RoPE coefficients, queries, and cache must use BF16.
        The cache must be unquantized and non-MLA. Projections and queries
        must have rank 3, offsets rank 1, and coefficients rank 2. Chunk,
        projection width, query head, cache head, and rotary extents must be
        static. Chunk and head counts must be positive, and both query and
        KV head counts must be divisible by chunks. Each chunk must contain
        its Q, K, and V head rows, and query output width must match the cache
        head width. Q/K and V widths must be positive and fit the cache head;
        rotary width must be positive and fit the Q/K head. Q/K, V, cache
        head, and rotary widths must be multiples of 8.
    """
    comptime cache_t = type_of(k_cache)
    comptime kv_params = cache_t.kv_params
    comptime assert (
        not cache_t.quantization_enabled
    ), "quantized KV caches are unsupported"
    comptime assert (
        not kv_params.is_mla
    ), "cache must have separate K/V planes; MLA is unsupported"
    comptime assert (
        cache_t.dtype == DType.bfloat16
    ), "kv_blocks must be a rank-6 BF16 cache"
    comptime assert (
        qkv.flat_rank == 3 and q_output.flat_rank == 3
    ), "qkv and query output must have rank 3"
    comptime assert (
        row_offsets.flat_rank == 1 and freqs.flat_rank == 2
    ), "ragged offsets must have rank 1 and freqs_cis must have rank 2"
    comptime assert qkv.LayoutType._shape_types[
        1
    ].is_static_value, "chunk, frequency, and cache extents must be static"
    comptime assert qkv.LayoutType._shape_types[
        2
    ].is_static_value, "chunk, frequency, and cache extents must be static"
    comptime assert q_output.LayoutType._shape_types[
        1
    ].is_static_value, "query head and cache extents must be static"
    comptime assert q_output.LayoutType._shape_types[
        2
    ].is_static_value, "query head and cache extents must be static"
    comptime assert freqs.LayoutType._shape_types[
        1
    ].is_static_value, "chunk, frequency, and cache extents must be static"
    comptime chunks = Int(qkv.static_shape[1])
    comptime num_q_heads = Int(q_output.static_shape[1])
    comptime num_kv_heads = kv_params.num_heads
    comptime head_size = kv_params.head_size
    comptime rope_dim = Int(freqs.static_shape[1])
    comptime assert (
        chunks > 0 and num_q_heads > 0 and num_kv_heads > 0
    ), "chunk and head counts must be positive"
    comptime assert (
        num_q_heads % chunks == 0 and num_kv_heads % chunks == 0
    ), "query and KV heads must be divisible by chunks"
    comptime q_heads_per_chunk = num_q_heads // chunks
    comptime kv_heads_per_chunk = num_kv_heads // chunks
    comptime q_rows = q_heads_per_chunk * qk_head_dim
    comptime k_rows = kv_heads_per_chunk * qk_head_dim
    comptime v_rows = kv_heads_per_chunk * v_head_dim
    comptime assert (
        chunk_padding >= 0 and chunk_padding % 8 == 0
    ), "QKV chunk padding must be nonnegative and aligned to 8 elements"
    comptime assert (
        Int(qkv.static_shape[2]) == q_rows + k_rows + v_rows + chunk_padding
    ), "Invalid QKV chunk width"
    comptime assert (
        Int(q_output.static_shape[2]) == head_size
    ), "Query output width must match cache head size"
    comptime assert (
        qk_head_dim > 0 and v_head_dim > 0
    ), "Q/K and V widths must be positive"
    comptime assert (
        qk_head_dim <= head_size and v_head_dim <= head_size
    ), "Q/K and V widths must fit the padded cache head"
    comptime assert (
        rope_dim > 0 and rope_dim <= qk_head_dim
    ), "rotary width must be positive and fit the Q/K head"
    comptime assert (
        qk_head_dim % 8 == 0 and v_head_dim % 8 == 0
    ), "Q/K and V widths must be aligned to 8 elements"
    comptime assert (
        head_size % 8 == 0 and rope_dim % 8 == 0
    ), "cache head and rotary widths must be aligned to 8 elements"
    comptime total_heads = num_q_heads + 2 * num_kv_heads
    comptime lanes_per_head = head_size // 8
    comptime width = 8
    comptime half = width // 2
    comptime alignment = align_of[SIMD[DType.bfloat16, width]]()
    comptime half_alignment = align_of[SIMD[DType.bfloat16, half]]()
    var flat = global_idx.x
    var token = flat // (total_heads * lanes_per_head)
    if token >= Int(qkv.dim[0]()) or token >= Int(
        row_offsets[Int(row_offsets.dim[0]()) - 1]
    ):
        return
    var token_head, lane = divmod(flat, lanes_per_head)
    var _, head = divmod(token_head, total_heads)
    var dim = lane * width
    var batch = get_batch_from_row_offsets(row_offsets, token)
    var local = Int(UInt32(token) - row_offsets[batch])
    var position = k_cache.cache_length(batch) + local
    var is_q = head < num_q_heads
    var is_k = head >= num_q_heads and head < num_q_heads + num_kv_heads
    var kv_head = (
        head - num_q_heads if is_k else head - num_q_heads - num_kv_heads
    )
    var chunk = (
        head // q_heads_per_chunk if is_q else kv_head // kv_heads_per_chunk
    )
    var source_head = (
        head % q_heads_per_chunk if is_q else kv_head % kv_heads_per_chunk
    )
    var source_base: Int
    if is_q:
        source_base = source_head * qk_head_dim
    elif is_k:
        source_base = q_rows + source_head * qk_head_dim
    else:
        source_base = q_rows + k_rows + source_head * v_head_dim
    var real_depth = qk_head_dim if is_q or is_k else v_head_dim
    var val = SIMD[DType.bfloat16, width](0)
    if dim < real_depth:
        if (is_q or is_k) and dim < rope_dim:
            var re_dim, im_dim = get_safetensors_idx(dim, rope_dim)
            var re = qkv.load[width=half, alignment=half_alignment](
                (token, chunk, source_base + re_dim)
            )
            var im = qkv.load[width=half, alignment=half_alignment](
                (token, chunk, source_base + im_dim)
            )
            var freq = freqs.load[width=width, alignment=alignment](
                (position, dim)
            )
            # Match rope_k_cache: rope_value converts inputs to the BF16
            # frequency dtype before the rotation arithmetic.
            var roped = rope_value(re.interleave(im), freq)
            var out_re, out_im = roped.deinterleave()
            if is_q:
                q_output.store[alignment=half_alignment](
                    (token, head, re_dim), out_re
                )
                q_output.store[alignment=half_alignment](
                    (token, head, im_dim), out_im
                )
            else:
                k_cache.store(
                    batch,
                    kv_head,
                    position,
                    re_dim,
                    out_re.cast[cache_t.dtype](),
                )
                k_cache.store(
                    batch,
                    kv_head,
                    position,
                    im_dim,
                    out_im.cast[cache_t.dtype](),
                )
            return
        val = qkv.load[width=width, alignment=alignment](
            (token, chunk, source_base + dim)
        )
        if not is_q and not is_k:
            val = (val.cast[DType.float32]() * value_scale).cast[
                DType.bfloat16
            ]()
    # Padding is literal zero, independently of V's scale and sign.
    if is_q:
        q_output.store[alignment=alignment]((token, head, dim), val)
    elif is_k:
        k_cache.store(batch, kv_head, position, dim, val.cast[cache_t.dtype]())
    else:
        v_cache.store(batch, kv_head, position, dim, val.cast[cache_t.dtype]())


def chunked_qkv_rope_store_paged_ragged[
    qk_head_dim: Int, v_head_dim: Int, chunk_padding: Int = 0
](
    qkv: TileTensor[mut=False, .bfloat16, ...],
    row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs: TileTensor[mut=False, .bfloat16, ...],
    collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    q_output: TileTensor[mut=True, .bfloat16, ...],
    value_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Launches chunked QKV preparation with separate cache accessors.

    Query output rows at or past the final ragged offset are left unwritten.

    Parameters:
        qk_head_dim: Unpadded query/key width.
        v_head_dim: Unpadded value width.
        chunk_padding: Trailing projection rows to ignore in each chunk.

    Args:
        qkv: BF16 chunked projection [tokens, chunks, chunk_width].
        row_offsets: Ragged offsets [batch_size + 1].
        freqs: BF16 RoPE table [max_position, rotary_width].
        collection: Unquantized, non-MLA paged KV collection.
        layer_idx: Layer to append to in both cache planes.
        q_output: Padded queries [tokens, query_heads, cache_head_size].
        value_scale: FP32 multiplier; V times this scale is rounded to BF16
            before storing.
        ctx: GPU device context.

    Raises:
        Error: If token extents differ, the cache layer is invalid, or
            `row_offsets` is empty.

    Constraints:
        Projections, RoPE coefficients, queries, and cache must use BF16;
        the cache must be unquantized and non-MLA. Projections and queries
        must have rank 3, offsets rank 1, and coefficients rank 2. Chunk,
        projection width, query head, cache head, and rotary extents must be
        static. Chunk and head counts must be positive, and both query and
        KV head counts must be divisible by chunks. Each chunk must contain
        its Q, K, and V head rows, and query output width must match the cache
        head width. Q/K and V widths must be positive and fit the cache head;
        rotary width must be positive and fit the Q/K head. Q/K, V, cache
        head, and rotary widths must be multiples of 8. Launch captures must
        support device encoding.
    """
    if Int(qkv.dim[0]()) != Int(q_output.dim[0]()):
        raise Error("QKV and query output token extents must match")
    if Int(layer_idx) >= Int(collection.blocks.dim[2]()):
        raise Error("Invalid cache layer")
    if Int(row_offsets.dim[0]()) < 1:
        raise Error("Ragged offsets must contain an initial offset")
    var k_cache = collection.get_key_cache(Int(layer_idx))
    var v_cache = collection.get_value_cache(Int(layer_idx))
    comptime cache_t = type_of(k_cache)
    comptime total_heads = Int(
        q_output.static_shape[1]
    ) + 2 * cache_t.kv_params.num_heads
    comptime lanes_per_head = cache_t.kv_params.head_size // 8
    var count = Int(qkv.dim[0]()) * total_heads * lanes_per_head
    if count == 0:
        return

    # The closure carries both disjoint views in one device-encoded payload,
    # as rope_split_store does, rather than passing overlapping launch inputs.
    def kernel() {
        var qkv,
        var row_offsets,
        var freqs,
        var k_cache,
        var v_cache,
        var q_output,
        var value_scale,
    }:
        chunked_qkv_rope_store_kernel[qk_head_dim, v_head_dim, chunk_padding](
            qkv,
            row_offsets,
            freqs,
            k_cache,
            v_cache,
            q_output,
            value_scale,
        )

    comptime assert conforms_to(
        type_of(kernel), DevicePassable
    ), "Chunked QKV launch captures must support device encoding"
    ctx.enqueue_function(
        kernel,
        grid_dim=ceildiv(count, 128),
        block_dim=128,
    )
