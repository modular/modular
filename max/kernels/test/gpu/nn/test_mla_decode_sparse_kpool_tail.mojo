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

"""Sparse FP8 MLA decode over k-pooled top-k selections, end to end.

A k-pooled indexer (`glm5_next`) selects pools, and `kpool_expand_topk_kernel`
widens each row to `POOL_TOPK * KPOOL + KPOOL - 1` token positions: the
selected pools, the `visible % KPOOL` most recent positions no complete pool
covers yet, then `-1`. The sparse decode bounds its scan by the context
length, so it reads only a row's first `cache_len + q_len` columns. That is
correct only while the expansion keeps every valid entry ahead of the
padding; a tail placed in the last columns was never read while the context
was shorter than the row, dropping the query's newest tokens, its own
position included.

Selections go through the real expansion kernel, so this checks the contract
between the two. The decode launch mirrors the graph op: causal mask, logical
indices for masking, physical indices for the gather, and a constant
per-token `topk_lengths`.
"""

from std.math import ceildiv, exp, gcd, sqrt
from std.random import randn, seed

from max.gpu import *
from max.gpu.host import DeviceContext, HostBuffer
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mla_index_kpool import (
    KPOOL_EXPAND_BLOCK_SIZE,
    kpool_expand_topk_kernel,
)
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig, NonNullPointer
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.utils.index import IndexList
from std.utils.numerics import min_or_neg_inf

comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime NUM_HEADS = 8
comptime KPOOL = 4
comptime POOL_TOPK = 512
comptime TAIL_WIDTH = KPOOL - 1
comptime WIDTH = POOL_TOPK * KPOOL + TAIL_WIDTH  # 2051, as glm5_next sets it

# Row-cosine floors. FP8 P costs accuracy when only a handful of keys are
# visible, so short contexts get a looser floor; dropping one of their keys
# still lands far below it. With ~1000 keys, dropping one moves a row only to
# ~0.999, while a correct row stays near 0.9998.
comptime SHORT_CONTEXT_MIN_COSINE = 0.99
comptime LONG_CONTEXT_MIN_COSINE = 0.9995


def _coprime_at_least(m: Int, n: Int) -> Int:
    var s = m
    while gcd(s, n) != 1:
        s += 1
    return s


def _select_pools(
    cache_len: Int, s: Int, mut pool_ids: HostBuffer[.int32], base: Int
):
    """Writes one query's pool selection the way the indexer top-k does.

    Every complete pool is a candidate. The selected ones come in a
    non-monotonic order, like a score-sorted top-k, and when fewer than
    `POOL_TOPK` exist the rest of the row is `-1`.
    """
    var num_pools = (cache_len + s + 1) // KPOOL
    var num_selected = min(num_pools, POOL_TOPK)
    var mult = _coprime_at_least(7, num_pools) if num_pools > 1 else 1
    for j in range(POOL_TOPK):
        pool_ids[base + j] = Int32(
            (j * mult + s) % num_pools if j < num_selected else -1
        )


def run_case(
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    min_row_cosine: Float64,
    ctx: DeviceContext,
    mut failures: List[String],
) raises:
    """Runs one shape and records `name` in `failures` if any row mismatched."""
    var num_keys = cache_len + q_len
    var total_q_tokens = batch_size * q_len
    comptime scale = Float32(0.125)
    comptime kv_type = DType.float8_e4m3fn
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )

    var pages_per_seq = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * pages_per_seq
    var page_elems = PAGE_SIZE * Q_DEPTH
    var block_elems = total_pages * page_elems

    # Shuffled page table so the physical remap is exercised.
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * pages_per_seq
    )
    var page_mult = _coprime_at_least(3, pages_per_seq)
    for b in range(batch_size):
        for p in range(pages_per_seq):
            lut_host[b * pages_per_seq + p] = UInt32(
                b * pages_per_seq + (p * page_mult + 1) % pages_per_seq
            )

    # K values, quantized to FP8 once so the reference sees the same bytes.
    var k_bf16 = ctx.enqueue_create_host_buffer[.bfloat16](
        batch_size * num_keys * Q_DEPTH
    )
    randn(k_bf16.as_span(), mean=0.0, standard_deviation=0.5)
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = 0
    var k_ref = ctx.enqueue_create_host_buffer[.float64](
        batch_size * num_keys * Q_DEPTH
    )
    for b in range(batch_size):
        for t in range(num_keys):
            var block_id = Int(lut_host[b * pages_per_seq + t // PAGE_SIZE])
            var dst = block_id * page_elems + (t % PAGE_SIZE) * Q_DEPTH
            var src = (b * num_keys + t) * Q_DEPTH
            for d in range(Q_DEPTH):
                var q8 = k_bf16[src + d].cast[kv_type]()
                blocks_host[dst + d] = q8
                k_ref[src + d] = q8.cast[.float64]()

    var q_size = total_q_tokens * NUM_HEADS * Q_DEPTH
    var q_bf16 = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_bf16[i].cast[.float8_e4m3fn]()

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for b in range(batch_size):
            cl_host[b] = UInt32(cache_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    with row_offsets_device.map_to_host() as ro_host:
        for i in range(batch_size + 1):
            ro_host[i] = UInt32(i * q_len)

    var pool_host = ctx.enqueue_create_host_buffer[.int32](
        total_q_tokens * POOL_TOPK
    )
    for b in range(batch_size):
        for s in range(q_len):
            _select_pools(cache_len, s, pool_host, (b * q_len + s) * POOL_TOPK)
    var pool_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * POOL_TOPK
    )
    ctx.enqueue_copy(pool_device, pool_host)
    var logical_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * WIDTH
    )
    var expand_out = TileTensor(
        logical_device, row_major(total_q_tokens, WIDTH)
    )
    var expand_pools = TileTensor(
        pool_device, row_major(total_q_tokens, POOL_TOPK)
    )
    var expand_iro = TileTensor(row_offsets_device, row_major(batch_size + 1))
    var expand_clen = TileTensor(cache_lengths_device, row_major(batch_size))
    comptime expand_kernel = kpool_expand_topk_kernel[
        expand_out.LayoutType,
        expand_out.origin,
        type_of(expand_pools.as_imm()).LayoutType,
        ImmOrigin(expand_pools.origin),
        type_of(expand_iro.as_imm()).LayoutType,
        ImmOrigin(expand_iro.origin),
        type_of(expand_clen.as_imm()).LayoutType,
        expand_out.Engine,
        type_of(expand_pools.as_imm()).Engine,
        type_of(expand_iro.as_imm()).Engine,
        type_of(expand_clen.as_imm()).Engine,
        KPOOL,
        POOL_TOPK,
        True,
    ]
    ctx.enqueue_function[expand_kernel](
        expand_out,
        expand_pools.as_imm(),
        expand_iro.as_imm(),
        expand_clen.as_imm(),
        Int32(total_q_tokens),
        grid_dim=(total_q_tokens, 1, 1),
        block_dim=(KPOOL_EXPAND_BLOCK_SIZE, 1, 1),
    )
    var logical_host = ctx.enqueue_create_host_buffer[.int32](
        total_q_tokens * WIDTH
    )
    ctx.enqueue_copy(logical_host, logical_device)
    ctx.synchronize()

    var physical_host = ctx.enqueue_create_host_buffer[.int32](
        total_q_tokens * WIDTH
    )
    for b in range(batch_size):
        for s in range(q_len):
            var g = b * q_len + s
            for i in range(WIDTH):
                var t = Int(logical_host[g * WIDTH + i])
                if t < 0:
                    physical_host[g * WIDTH + i] = -1
                else:
                    var block_id = Int(
                        lut_host[b * pages_per_seq + t // PAGE_SIZE]
                    )
                    physical_host[g * WIDTH + i] = Int32(
                        block_id * PAGE_SIZE + t % PAGE_SIZE
                    )

    # Reference keys come from the pool selection and the visible count, not
    # from the expansion, so a dropped or misplaced entry shows up as a
    # mismatch rather than agreeing with itself.
    var out_size = total_q_tokens * NUM_HEADS * V_DEPTH
    var ref_out = List(length=out_size, fill=Float64(0))
    for b in range(batch_size):
        for s in range(q_len):
            var g = b * q_len + s
            var visible = cache_len + s + 1
            var keys = List[Int]()
            for j in range(POOL_TOPK):
                var pid = Int(pool_host[g * POOL_TOPK + j])
                if pid >= 0:
                    for c in range(KPOOL):
                        keys.append(pid * KPOOL + c)
            for t in range(visible - visible % KPOOL, visible):
                keys.append(t)
            for h in range(NUM_HEADS):
                var q_base = (g * NUM_HEADS + h) * Q_DEPTH
                var scores = List(length=len(keys), fill=Float64(0))
                var max_s = Float64(min_or_neg_inf[.float64]())
                for j in range(len(keys)):
                    var k_base = (b * num_keys + keys[j]) * Q_DEPTH
                    var dot = Float64(0)
                    for d in range(Q_DEPTH):
                        dot += (
                            q_host[q_base + d].cast[.float64]()
                            * k_ref[k_base + d]
                        )
                    scores[j] = dot * Float64(scale)
                    max_s = max(max_s, scores[j])
                var denom = Float64(0)
                for j in range(len(keys)):
                    scores[j] = exp(scores[j] - max_s)
                    denom += scores[j]
                var o_base = (g * NUM_HEADS + h) * V_DEPTH
                for j in range(len(keys)):
                    var w = scores[j] / denom
                    var k_base = (b * num_keys + keys[j]) * Q_DEPTH
                    for d in range(V_DEPTH):
                        ref_out[o_base + d] += w * k_ref[k_base + d]

    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var lut_device = ctx.enqueue_create_buffer[.uint32](
        batch_size * pages_per_seq
    )
    ctx.enqueue_copy(lut_device, lut_host)
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var physical_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * WIDTH
    )
    ctx.enqueue_copy(physical_device, physical_host)
    # One entry per token, all the selection width, as glm5_next passes it.
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](total_q_tokens)
    with topk_lengths_device.map_to_host() as tl_host:
        for i in range(total_q_tokens):
            tl_host[i] = Int32(WIDTH)
    ctx.synchronize()

    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(
            lut_device,
            row_major(Int64(batch_size), Int64(pages_per_seq)),
        )
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device, row_major(total_q_tokens, Idx[NUM_HEADS], Idx[Q_DEPTH])
    )
    var out_tt = TileTensor(
        out_device, row_major(total_q_tokens, Idx[NUM_HEADS], Idx[V_DEPTH])
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )
    var mla_args = MLADispatchScalarArgs[num_heads=NUM_HEADS, is_fp8_kv=True](
        batch_size, cache_len, q_len, ctx
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[.float8_e4m3fn](NUM_HEADS, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        CausalMask(),
        row_offsets_tt,
        scale,
        ctx,
        mla_args.gpu_tile_tensor(),
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            physical_device.unsafe_ptr()
        ),
        indices_stride=WIDTH,
        topk_lengths=NonNullPointer[DType.int32](topk_lengths_device),
        logical_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            logical_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var min_cos = Float64(1)
    var max_err = Float64(0)
    var worst_row = 0
    with out_device.map_to_host() as out_host:
        for row in range(total_q_tokens * NUM_HEADS):
            var dot = Float64(0)
            var norm_a = Float64(0)
            var norm_r = Float64(0)
            for d in range(V_DEPTH):
                var a = out_host[row * V_DEPTH + d].cast[.float64]()
                var r = ref_out[row * V_DEPTH + d]
                dot += a * r
                norm_a += a * a
                norm_r += r * r
                max_err = max(max_err, abs(a - r))
            var cos = dot / sqrt(
                norm_a * norm_r
            ) if norm_a * norm_r > 0 else Float64(0)
            if cos < min_cos:
                min_cos = cos
                worst_row = row
    var ok = min_cos >= min_row_cosine
    print(
        "PASS" if ok else "FAIL",
        name,
        ": batch_size=",
        batch_size,
        " cache_len=",
        cache_len,
        " q_len=",
        q_len,
        " min_row_cosine=",
        min_cos,
        " (token ",
        worst_row // NUM_HEADS,
        ", visible ",
        cache_len + (worst_row // NUM_HEADS) % q_len + 1,
        ") max_abs_err=",
        max_err,
    )

    _ = mla_args
    _ = blocks_device
    _ = lut_device
    _ = cache_lengths_device
    _ = q_device
    _ = out_device
    _ = physical_device
    _ = pool_device
    _ = logical_device
    _ = topk_lengths_device
    _ = row_offsets_device
    if not ok:
        failures.append(name)


def main() raises:
    seed(0)
    var failures = List[String]()
    with DeviceContext() as ctx:
        # Short contexts whose newest tokens sit only in the tail columns.
        run_case(
            "tail_2_of_6", 1, 5, 1, SHORT_CONTEXT_MIN_COSINE, ctx, failures
        )
        run_case(
            "tail_holds_self",
            1,
            100,
            1,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        # Speculative-decode verify: tail widths 3, 0, 1, 2 across the rows.
        run_case("mtp_q4", 2, 98, 4, SHORT_CONTEXT_MIN_COSINE, ctx, failures)
        run_case(
            "tail_1_long_prefix",
            2,
            1000,
            1,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        # Controls: no tail, and a context longer than the list.
        run_case("no_tail", 1, 2047, 1, LONG_CONTEXT_MIN_COSINE, ctx, failures)
        run_case(
            "context_past_list",
            1,
            4097,
            1,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
    if len(failures) > 0:
        var msg = String("sparse kpool tail cases failed:")
        for f in failures:
            msg += " " + f
        raise Error(msg)
