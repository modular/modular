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
"""Scores k-pools over the pooled indexer cache as GLM-5.3-Flash lays it out.

GLM-5.3-Flash declares its indexer cache `index_head_dim / kpool` = 32 wide,
one row per token, so the paged cache manager's per-token accounting holds.
Pool `p`'s 128-wide key spreads over the `kpool` token rows it covers,
`kpool * p` through `kpool * p + kpool - 1`, and its scale is replicated on
each of them (see `Indexer.quantize_pooled_keys`). The graph op hands that
cache to `mla_indexer_ragged_float8_paged` as is.

This test builds the cache the same way, asks for the top pools, and compares
them with an exact host top-k. The contexts are long enough that the budget
selects a small fraction of the pools, so a scorer reading the wrong bytes
picks a near-random set instead of passing by selecting everything.

The model's indexer cache is one leaf of a multi-layer cache with a padded page
stride, so some cases score a later layer of a padded cache whose other layers
and padding hold random bytes.
"""

from std.math import ceildiv, log1p
from std.random import rand, random_ui64, seed

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mla_index_fp8 import mla_indexer_ragged_float8_paged
from nn.attention.mha_mask import MaskName
from nn.index_fp8 import _has_gfx950_accelerator
from std.collections import Set
from std.testing import assert_true

comptime NUM_HEADS = 32
comptime DEPTH = 128
comptime KPOOL = 4
comptime ROW_WIDTH = DEPTH // KPOOL
comptime PAGE_SIZE = 128
comptime TOP_K = 64
comptime MIN_MEAN_OVERLAP = 0.95


def run_case[
    one_row_per_pool: Bool,
    page_size: Int = PAGE_SIZE,
    num_heads: Int = NUM_HEADS,
    causal: Bool = False,
    num_layers: Int = 1,
    layer_idx: Int = 0,
    page_pad: Int = 0,
    scale_pad: Int = 0,
](cache_lens: List[Int], seq_lens: List[Int], ctx: DeviceContext) raises:
    """Scores the pools and grades the selection against a host top-k.

    Parameters:
        one_row_per_pool: Lay the cache out the way the scorer's own kpool
            test does, one `DEPTH`-wide row per pool and `page_size` pools per
            page, instead of Flash's per-token rows. The control: it checks
            this harness and its host reference, not Flash's layout.
        page_size: Rows per page. At 128 a page holds 32 of Flash's pools, so
            the SM100 scorer stages its 128-key tile in four copies; at 512 it
            takes one.
        num_heads: Indexer heads. 32 runs the K-streaming kernel for every
            shape; 64 on a short query over a shallow cache runs the
            K-resident one, so the two cover both SM100 kernels.
        causal: Mask each query to the pools whose last token it can see.
        num_layers: Layers in the cache; only `layer_idx` holds keys.
        layer_idx: The layer scored.
        page_pad: Elements past the packed key page, so the page stride is
            padded as in a multi-cache leaf.
        scale_pad: The same padding for the scale pages.
    """
    var batch_size = len(cache_lens)
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for b in range(batch_size):
        total_seq_len += seq_lens[b]
        max_seq_len = max(max_seq_len, seq_lens[b])
        max_cache_len = max(max_cache_len, cache_lens[b])

    var max_tokens = max_cache_len + max_seq_len
    var pages_per_seq = ceildiv(max_tokens, page_size)
    var num_blocks = batch_size * pages_per_seq + 4
    var max_pools = max_tokens // KPOOL

    comptime row_width = DEPTH if one_row_per_pool else ROW_WIDTH
    comptime rows_per_page = page_size
    comptime page_stride = num_layers * rows_per_page * row_width + page_pad
    comptime scale_page_stride = num_layers * rows_per_page + scale_pad
    comptime k_layer_base = layer_idx * rows_per_page * row_width
    comptime ks_layer_base = layer_idx * rows_per_page
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=row_width, is_mla=True
    )

    # Pooled keys and per-pool scales on the host, laid out `[batch, pool, 128]`.
    var pooled = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](
        batch_size * max_pools * DEPTH
    )
    rand(pooled.as_span())
    var pool_scale = ctx.enqueue_create_host_buffer[DType.float32](
        batch_size * max_pools
    )
    rand(pool_scale.as_span())
    # An exponential tail makes the top of the ranking decisive; a uniform
    # scale packs the top-k within a hair of the peak, where fp8 ties decide.
    for i in range(batch_size * max_pools):
        var u = Float64(pool_scale[i])
        pool_scale[i] = Float32(-log1p(-min(u, 0.999999)))

    var q_host = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](
        total_seq_len * num_heads * DEPTH
    )
    rand(q_host.as_span())
    var qs_host = ctx.enqueue_create_host_buffer[DType.float32](
        total_seq_len * num_heads
    )
    rand(qs_host.as_span())

    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size * pages_per_seq
    )
    var used = Set[Int]()
    for i in range(batch_size * pages_per_seq):
        var blk = Int(random_ui64(0, UInt64(num_blocks - 1)))
        while blk in used:
            blk = Int(random_ui64(0, UInt64(num_blocks - 1)))
        used.add(blk)
        lut_host[i] = UInt32(blk)

    # Scatter each pool across its `KPOOL` token rows, the way the pooled
    # cache store writes it. Every other byte, including the other layers and
    # the page padding, is random so a scorer reading off this layer's rows
    # selects noise.
    var k_len = num_blocks * page_stride
    var ks_len = num_blocks * scale_page_stride
    var k_host = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](k_len)
    var ks_host = ctx.enqueue_create_host_buffer[DType.float32](ks_len)
    rand(k_host.as_span())
    rand(ks_host.as_span())
    for b in range(batch_size):
        for page in range(pages_per_seq):
            var blk = Int(lut_host[b * pages_per_seq + page])
            for i in range(rows_per_page * row_width):
                k_host[blk * page_stride + k_layer_base + i] = 0
            for i in range(rows_per_page):
                ks_host[blk * scale_page_stride + ks_layer_base + i] = 0
    for b in range(batch_size):
        var num_pools = (cache_lens[b] + seq_lens[b]) // KPOOL
        for p in range(num_pools):
            comptime if one_row_per_pool:
                var page, row = divmod(p, page_size)
                var blk = Int(lut_host[b * pages_per_seq + page])
                for j in range(DEPTH):
                    k_host[
                        blk * page_stride + k_layer_base + row * DEPTH + j
                    ] = pooled[(b * max_pools + p) * DEPTH + j]
                ks_host[
                    blk * scale_page_stride + ks_layer_base + row
                ] = pool_scale[b * max_pools + p]
            else:
                for member in range(KPOOL):
                    var tok = p * KPOOL + member
                    var page, row = divmod(tok, page_size)
                    var blk = Int(lut_host[b * pages_per_seq + page])
                    for j in range(ROW_WIDTH):
                        k_host[
                            blk * page_stride
                            + k_layer_base
                            + row * ROW_WIDTH
                            + j
                        ] = pooled[
                            (b * max_pools + p) * DEPTH + member * ROW_WIDTH + j
                        ]
                    ks_host[
                        blk * scale_page_stride + ks_layer_base + row
                    ] = pool_scale[b * max_pools + p]

    var offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    offsets_host[0] = 0
    for b in range(batch_size):
        offsets_host[b + 1] = offsets_host[b] + UInt32(seq_lens[b])
    var cache_lens_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for b in range(batch_size):
        cache_lens_host[b] = UInt32(cache_lens[b])

    var q_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        total_seq_len * num_heads * DEPTH
    )
    var qs_dev = ctx.enqueue_create_buffer[DType.float32](
        total_seq_len * num_heads
    )
    var k_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](k_len)
    var ks_dev = ctx.enqueue_create_buffer[DType.float32](ks_len)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * pages_per_seq
    )
    var offsets_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    var cache_lens_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var out_dev = ctx.enqueue_create_buffer[DType.int32](total_seq_len * TOP_K)
    var out_host = ctx.enqueue_create_host_buffer[DType.int32](
        total_seq_len * TOP_K
    )
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(qs_dev, qs_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(ks_dev, ks_host)
    ctx.enqueue_copy(lut_dev, lut_host)
    ctx.enqueue_copy(offsets_dev, offsets_host)
    ctx.enqueue_copy(cache_lens_dev, cache_lens_host)

    # Mirrors the graph op: the cache's own row width, and the caller's
    # quantization granularity of one scale per 128 indexer values.
    comptime Collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        rows_per_page,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=DEPTH,
    ]
    var k_collection = Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                k_dev,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(num_layers),
                    Idx[rows_per_page],
                    Idx[1],
                    Idx[row_width],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(cache_lens_dev, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(Int64(batch_size), Int64(pages_per_seq)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(max_seq_len),
        UInt32(max_cache_len),
        rebind[Collection.scales_tt_type](
            TileTensor(
                ks_dev,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(num_layers),
                    Idx[rows_per_page],
                    Idx[1],
                    Idx[1],
                ),
            ).as_unsafe_any_origin()
        ),
        page_stride=page_stride,
        scales_page_stride=scale_page_stride,
    )

    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        type_of(k_collection),
        num_heads,
        DEPTH,
        TOP_K,
        MaskName.CAUSAL.name if causal else MaskName.NULL.name,
        kpool=KPOOL,
    ](
        TileTensor(out_dev, row_major(total_seq_len, TOP_K)),
        TileTensor(q_dev, row_major(total_seq_len, num_heads, DEPTH)),
        TileTensor(qs_dev, row_major(total_seq_len, num_heads)),
        TileTensor(offsets_dev, row_major(batch_size + 1)),
        k_collection,
        UInt32(layer_idx),
        ctx,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var overlap_sum = 0.0
    var token = 0
    for b in range(batch_size):
        for t in range(seq_lens[b]):
            # The kernel's `indexer_key_bound`: a pool is a candidate once its
            # last token is visible.
            var visible = cache_lens[b] + seq_lens[b]
            comptime if causal:
                visible = cache_lens[b] + t + 1
            var num_pools = visible // KPOOL
            var num_selected = min(TOP_K, num_pools)
            var scores = List[Float32](length=num_pools, fill=0.0)
            for p in range(num_pools):
                var acc: Float32 = 0.0
                for h in range(num_heads):
                    var dot: Float32 = 0.0
                    var q_base = (token * num_heads + h) * DEPTH
                    var k_base = (b * max_pools + p) * DEPTH
                    for d in range(DEPTH):
                        dot += (
                            q_host[q_base + d].cast[DType.float32]()
                            * pooled[k_base + d].cast[DType.float32]()
                        )
                    acc += max(dot, 0.0) * qs_host[token * num_heads + h]
                scores[p] = acc * pool_scale[b * max_pools + p]

            # Exact host top-k by repeated selection; small enough here.
            var expected = Set[Int]()
            var taken = List[Bool](length=num_pools, fill=False)
            for _ in range(num_selected):
                var best = -1
                for p in range(num_pools):
                    if not taken[p] and (best < 0 or scores[p] > scores[best]):
                        best = p
                taken[best] = True
                expected.add(best)

            var hits = 0
            for i in range(TOP_K):
                var pool = Int(out_host[token * TOP_K + i])
                if i >= num_selected:
                    assert_true(pool == -1, "slot past the candidates not -1")
                    continue
                assert_true(
                    pool >= 0 and pool < num_pools,
                    "selected pool id out of range",
                )
                if pool in expected:
                    hits += 1
            overlap_sum += (
                Float64(hits) / Float64(num_selected) if num_selected else 1.0
            )
            token += 1

    var mean_overlap = overlap_sum / Float64(total_seq_len)
    print(
        "  one_row_per_pool=",
        one_row_per_pool,
        " causal=",
        causal,
        " layer=",
        layer_idx,
        "/",
        num_layers,
        " page_pad=",
        page_pad,
        " cache_lens=",
        cache_lens[0],
        "... tokens=",
        total_seq_len,
        " pools/top_k ~",
        (cache_lens[0] + seq_lens[0]) // KPOOL,
        "/",
        TOP_K,
        " mean top-k overlap=",
        mean_overlap,
    )
    assert_true(
        mean_overlap >= MIN_MEAN_OVERLAP,
        "pooled scorer selection disagrees with the host top-k",
    )


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        print("k-pooled indexer over the per-token pooled cache layout")
        print("control: one 128-wide row per pool")
        run_case[True]([4092, 6144], [1, 3], ctx)
        print("GLM-5.3-Flash layout: 32-wide token rows, pool over 4 rows")
        run_case[False]([4092, 6144], [1, 3], ctx)
        run_case[False]([2045, 3000], [4, 2], ctx)
        run_case[False, 512]([4092, 6144], [1, 3], ctx)
        run_case[False, PAGE_SIZE, 64]([4092, 6144], [1, 3], ctx)
        print("multi-layer cache, padded page stride, later layer")
        comptime ML = 3
        comptime LAYER = 2
        # Whole rows of padding that keep each page 16-byte aligned, as TMA
        # requires.
        comptime K_PAD = 1056
        comptime S_PAD = 20
        run_case[False, PAGE_SIZE, NUM_HEADS, False, ML, LAYER, K_PAD, S_PAD](
            [4092, 6144], [1, 3], ctx
        )
        run_case[False, PAGE_SIZE, 64, False, ML, LAYER, K_PAD, S_PAD](
            [4092, 6144], [1, 3], ctx
        )
        run_case[False, 512, NUM_HEADS, False, ML, LAYER, K_PAD, S_PAD](
            [4092, 6144], [1, 3], ctx
        )
        print("causal mask, rows shorter than a pool")
        run_case[False, PAGE_SIZE, NUM_HEADS, True, ML, LAYER, K_PAD, S_PAD](
            [4092, 0, 2, 6144], [4, 3, 1, 2], ctx
        )
        comptime if _has_gfx950_accelerator():
            # A 64-row page holds 16 of Flash's pools, short of the gfx950
            # scorer's 32-pool wave, so the op must refuse it.
            print("gfx950: a page shorter than one 32-pool wave is rejected")
            var rejected = False
            try:
                run_case[False, 64]([253], [3], ctx)
            except e:
                assert_true(
                    "page to hold a whole 32-key wave" in String(e), String(e)
                )
                rejected = True
            assert_true(rejected, "the gfx950 scorer accepted a 64-row page")
