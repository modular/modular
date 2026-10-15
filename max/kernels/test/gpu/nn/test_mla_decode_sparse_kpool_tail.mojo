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

"""Sparse MLA decode over packed top-k selections, end to end.

A k-pooled indexer (`glm5_next`) selects pools, and `kpool_expand_topk_kernel`
widens each row to `pool_topk * kpool + kpool - 1` token positions: the
selected pools, the `visible % kpool` most recent positions no complete pool
covers yet, then `-1`. With `kpool = 1` the same kernel packs a plain top-k
row (GLM-5.2): up to `pool_topk` keys, then `-1`. Selections go through the
real expansion kernel, so this checks the contract between the producer and
the decode.

The decode scans `min(topk, cache_len + q_len)` slots per sequence and masks
row `s` to slots below `cache_len + s + 1`. A `-1` below that limit gathers
as a zero row and scores 0 unless something masks it. That happens for
k-pool rows once a token sees more than `pool_topk * kpool` keys: the row
holds `pool_topk * kpool + visible % kpool` entries, so its last
`kpool - 1 - visible % kpool` slots are `-1`. It also happens for a row
reused at a later position, which holds the earlier token's count. The
decode bounds each row by that closed-form count (`SparseRowLayout`), so
none of these need logical indices. The cases cover:

- Short contexts, where a tail placed in the last columns was never read.
- Plain top-k lists without logical indices, as GLM-5.2 runs once the
  logical mask is dropped.
- k-pool rows past the selection width without logical indices, as
  `glm5_next` runs through the BF16 op.
- Rows reused one or more positions later, and k-pool rows with no tail.

A `-1` slot that scores 0 adds weight to the softmax denominator and nothing
to its numerator. On random data that shrinks a row only slightly. Every case
therefore adds a probe: Q and K carry a fixed component in rotary dimensions
that moves every real key's score by -32. The shift leaves the reference
unchanged, but an unmasked zero row then outscores every real key and drives
the row toward zero. The row keeps its direction, so the check compares
relative error as well as cosine.
"""

from std.collections import OptionalReg
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
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import SparseRowLayout
from std.utils.numerics import min_or_neg_inf

comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime NUM_HEADS = 8
comptime SCALE = Float32(0.125)

# glm5_next: 512 pools of 4, so rows are 2051 wide.
comptime GLM5_NEXT_KPOOL = 4
comptime GLM5_NEXT_POOL_TOPK = 512
# GLM-5.2: plain top-k.
comptime GLM5_2_TOPK = 2048

# The probe lives in rotary dimensions, which V does not read, and its values
# are exact in FP8. Each real key's score moves by
# `PROBE_DIMS * PROBE_Q * PROBE_K * SCALE` = -32, which a zero row does not.
comptime PROBE_FIRST_DIM = V_DEPTH
comptime PROBE_DIMS = 4
comptime PROBE_Q = Float32(8)
comptime PROBE_K = Float32(-8)

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


def _select_pools[
    kpool: Int, pool_topk: Int
](cache_len: Int, s: Int, mut pool_ids: HostBuffer[.int32], base: Int):
    """Writes one query's pool selection the way the indexer top-k does.

    Every complete pool is a candidate. The selected ones come in a
    non-monotonic order, like a score-sorted top-k, and when fewer than
    `pool_topk` exist the rest of the row is `-1`.
    """
    var num_pools = (cache_len + s + 1) // kpool
    var num_selected = min(num_pools, pool_topk)
    var mult = _coprime_at_least(7, num_pools) if num_pools > 1 else 1
    for j in range(pool_topk):
        pool_ids[base + j] = Int32(
            (j * mult + s) % num_pools if j < num_selected else -1
        )


def run_case[
    kv_type: DType, kpool: Int, pool_topk: Int, select_tail: Bool = True
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    use_logical: Bool,
    min_row_cosine: Float64,
    ctx: DeviceContext,
    mut failures: List[String],
    lag: Int = 0,
) raises:
    """Runs one shape and records `name` in `failures` if any row mismatched.

    The launch mirrors the graph op of the model the parameters describe.
    k-pool rows pass a constant per-token `topk_lengths`, as `glm5_next`
    does; plain top-k rows pass none, as GLM-5.2 does. Q and K share
    `kv_type`, so FP8 runs the native FP8 kernel and BF16 runs the BF16
    kernel. `use_logical` passes the pre-remap indices for the logical mask,
    as the FP8 op does today; the BF16 op never passes them. A nonzero `lag`
    computes each row for the token `lag` positions earlier and reuses it,
    as the GLM-5.2 MTP draft does.
    """
    comptime width = pool_topk * kpool + ((kpool - 1) if select_tail else 0)
    comptime is_fp8 = kv_type == DType.float8_e4m3fn
    var num_keys = cache_len + q_len
    var total_q_tokens = batch_size * q_len
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

    # K values, cast to the cache type once so the reference sees the same
    # bytes.
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
                var k_val = k_bf16[src + d].cast[kv_type]()
                if d >= PROBE_FIRST_DIM and d < PROBE_FIRST_DIM + PROBE_DIMS:
                    k_val = PROBE_K.cast[kv_type]()
                blocks_host[dst + d] = k_val
                k_ref[src + d] = k_val.cast[.float64]()

    var q_size = total_q_tokens * NUM_HEADS * Q_DEPTH
    var q_bf16 = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_bf16.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[kv_type](q_size)
    for i in range(q_size):
        var d = i % Q_DEPTH
        if d >= PROBE_FIRST_DIM and d < PROBE_FIRST_DIM + PROBE_DIMS:
            q_host[i] = PROBE_Q.cast[kv_type]()
        else:
            q_host[i] = q_bf16[i].cast[kv_type]()

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for b in range(batch_size):
            cl_host[b] = UInt32(cache_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    with row_offsets_device.map_to_host() as ro_host:
        for i in range(batch_size + 1):
            ro_host[i] = UInt32(i * q_len)

    var pool_host = ctx.enqueue_create_host_buffer[.int32](
        total_q_tokens * pool_topk
    )
    for b in range(batch_size):
        for s in range(q_len):
            _select_pools[kpool, pool_topk](
                cache_len - lag, s, pool_host, (b * q_len + s) * pool_topk
            )
    var pool_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * pool_topk
    )
    ctx.enqueue_copy(pool_device, pool_host)
    var logical_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * width
    )
    var expand_out = TileTensor(
        logical_device, row_major(total_q_tokens, width)
    )
    var expand_pools = TileTensor(
        pool_device, row_major(total_q_tokens, pool_topk)
    )
    var expand_iro = TileTensor(row_offsets_device, row_major(batch_size + 1))
    var sel_cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        batch_size
    )
    with sel_cache_lengths_device.map_to_host() as sel_host:
        for b in range(batch_size):
            sel_host[b] = UInt32(cache_len - lag)
    var expand_clen = TileTensor(
        sel_cache_lengths_device, row_major(batch_size)
    )
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
        kpool,
        pool_topk,
        select_tail,
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
        total_q_tokens * width
    )
    ctx.enqueue_copy(logical_host, logical_device)
    ctx.synchronize()

    var physical_host = ctx.enqueue_create_host_buffer[.int32](
        total_q_tokens * width
    )
    for b in range(batch_size):
        for s in range(q_len):
            var g = b * q_len + s
            for i in range(width):
                var t = Int(logical_host[g * width + i])
                if t < 0:
                    physical_host[g * width + i] = -1
                else:
                    var block_id = Int(
                        lut_host[b * pages_per_seq + t // PAGE_SIZE]
                    )
                    physical_host[g * width + i] = Int32(
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
            var visible = cache_len + s + 1 - lag
            var keys = List[Int]()
            for j in range(pool_topk):
                var pid = Int(pool_host[g * pool_topk + j])
                if pid >= 0:
                    for c in range(kpool):
                        keys.append(pid * kpool + c)
            comptime if select_tail:
                for t in range(visible - visible % kpool, visible):
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
                    scores[j] = dot * Float64(SCALE)
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
    var q_device = ctx.enqueue_create_buffer[kv_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var physical_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * width
    )
    ctx.enqueue_copy(physical_device, physical_host)
    # One entry per token, all the selection width, as glm5_next passes it.
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](total_q_tokens)
    with topk_lengths_device.map_to_host() as tl_host:
        for i in range(total_q_tokens):
            tl_host[i] = Int32(width)
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
    var mla_args = MLADispatchScalarArgs[num_heads=NUM_HEADS, is_fp8_kv=is_fp8](
        batch_size, cache_len, q_len, ctx
    )

    var logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None
    if use_logical:
        logical_indices = rebind[MutPointer[Int32, MutAnyOrigin]](
            logical_device.unsafe_ptr()
        )
    var d_indices = rebind[MutPointer[Int32, MutAnyOrigin]](
        physical_device.unsafe_ptr()
    )

    comptime if kpool > 1:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[kv_type](NUM_HEADS, Q_DEPTH),
            ragged=True,
            sparse=True,
            sparse_row=SparseRowLayout(
                kpool=kpool, pool_topk=pool_topk, select_tail=select_tail
            ),
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            SCALE,
            ctx,
            mla_args.gpu_tile_tensor(),
            d_indices=d_indices,
            indices_stride=width,
            topk_lengths=NonNullPointer[DType.int32](topk_lengths_device),
            logical_indices=logical_indices,
            selection_lag=lag,
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[kv_type](NUM_HEADS, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            SCALE,
            ctx,
            mla_args.gpu_tile_tensor(),
            d_indices=d_indices,
            indices_stride=width,
            logical_indices=logical_indices,
            selection_lag=lag,
        )
    ctx.synchronize()

    # Cosine misses a row that is right in direction but wrong in scale,
    # which is what a zero row does to it, so the relative error is checked
    # too. A row at the cosine floor has relative error about
    # `sqrt(2 * (1 - cos))`.
    var max_row_rel_err = sqrt(2 * (1 - min_row_cosine))
    var min_cos = Float64(1)
    var max_rel = Float64(0)
    var max_err = Float64(0)
    var worst_row = 0
    with out_device.map_to_host() as out_host:
        for row in range(total_q_tokens * NUM_HEADS):
            var dot = Float64(0)
            var norm_a = Float64(0)
            var norm_r = Float64(0)
            var norm_diff = Float64(0)
            for d in range(V_DEPTH):
                var a = out_host[row * V_DEPTH + d].cast[.float64]()
                var r = ref_out[row * V_DEPTH + d]
                dot += a * r
                norm_a += a * a
                norm_r += r * r
                norm_diff += (a - r) * (a - r)
                max_err = max(max_err, abs(a - r))
            var cos = dot / sqrt(
                norm_a * norm_r
            ) if norm_a * norm_r > 0 else Float64(0)
            var rel = sqrt(norm_diff / norm_r) if norm_r > 0 else Float64(0)
            min_cos = min(min_cos, cos)
            if rel > max_rel:
                max_rel = rel
                worst_row = row
    var ok = min_cos >= min_row_cosine and max_rel <= max_row_rel_err
    print(
        "PASS" if ok else "FAIL",
        name,
        ": batch_size=",
        batch_size,
        " cache_len=",
        cache_len,
        " q_len=",
        q_len,
        " logical=",
        use_logical,
        " lag=",
        lag,
        " min_row_cosine=",
        min_cos,
        " max_row_rel_err=",
        max_rel,
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
    _ = sel_cache_lengths_device
    _ = q_device
    _ = out_device
    _ = physical_device
    _ = pool_device
    _ = logical_device
    _ = topk_lengths_device
    _ = row_offsets_device
    if not ok:
        failures.append(name)


def glm5_next_fp8_case(
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    use_logical: Bool,
    min_row_cosine: Float64,
    ctx: DeviceContext,
    mut failures: List[String],
) raises:
    run_case[.float8_e4m3fn, GLM5_NEXT_KPOOL, GLM5_NEXT_POOL_TOPK](
        name,
        batch_size,
        cache_len,
        q_len,
        use_logical,
        min_row_cosine,
        ctx,
        failures,
    )


def glm5_next_bf16_case(
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    min_row_cosine: Float64,
    ctx: DeviceContext,
    mut failures: List[String],
    lag: Int = 0,
) raises:
    """Runs the BF16 path the way the BF16 graph op does: no logical indices."""
    run_case[.bfloat16, GLM5_NEXT_KPOOL, GLM5_NEXT_POOL_TOPK](
        name,
        batch_size,
        cache_len,
        q_len,
        False,
        min_row_cosine,
        ctx,
        failures,
        lag,
    )


def glm5_2_case(
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    q_len: Int,
    use_logical: Bool,
    min_row_cosine: Float64,
    ctx: DeviceContext,
    mut failures: List[String],
    lag: Int = 0,
) raises:
    run_case[.float8_e4m3fn, 1, GLM5_2_TOPK](
        name,
        batch_size,
        cache_len,
        q_len,
        use_logical,
        min_row_cosine,
        ctx,
        failures,
        lag,
    )


def main() raises:
    seed(0)
    var failures = List[String]()
    with DeviceContext() as ctx:
        # k-pool rows through the FP8 op, logical indices on, as today.
        # Short contexts whose newest tokens sit only in the tail columns.
        glm5_next_fp8_case(
            "tail_2_of_6",
            1,
            5,
            1,
            True,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_next_fp8_case(
            "tail_holds_self",
            1,
            100,
            1,
            True,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        # Speculative-decode verify: tail widths 3, 0, 1, 2 across the rows.
        glm5_next_fp8_case(
            "mtp_q4", 2, 98, 4, True, SHORT_CONTEXT_MIN_COSINE, ctx, failures
        )
        glm5_next_fp8_case(
            "tail_1_long_prefix",
            2,
            1000,
            1,
            True,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        # Controls: no tail, and a context longer than the list, whose last
        # slot is `-1` and is masked by the logical indices.
        glm5_next_fp8_case(
            "no_tail",
            1,
            2047,
            1,
            True,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_next_fp8_case(
            "context_past_list",
            1,
            4097,
            1,
            True,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )

        # GLM-5.2: plain top-k with and without logical indices. A causal,
        # packed list leaves every `-1` past the token's visible count, so
        # the slot-count mask alone must match. Contexts below, across and
        # past `top_k`, at decode and MTP verify shapes.
        glm5_2_case(
            "plain_mtp_q4_short",
            2,
            98,
            4,
            True,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_2_case(
            "plain_mtp_q4_short_no_logical",
            2,
            98,
            4,
            False,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_2_case(
            "plain_mtp_q4_crosses_topk",
            2,
            2046,
            4,
            True,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_2_case(
            "plain_mtp_q4_crosses_topk_no_logical",
            2,
            2046,
            4,
            False,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_2_case(
            "plain_mtp_q4_long_no_logical",
            1,
            4097,
            4,
            False,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_2_case(
            "plain_decode_long_no_logical",
            2,
            3000,
            1,
            False,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )

        # glm5_next through the BF16 op, which passes no logical indices.
        # Short contexts hold exactly `visible` entries, so they pass.
        glm5_next_bf16_case(
            "bf16_tail_holds_self",
            1,
            100,
            1,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        glm5_next_bf16_case(
            "bf16_mtp_q4", 2, 98, 4, SHORT_CONTEXT_MIN_COSINE, ctx, failures
        )
        # Past 2051 visible keys the row ends in `3 - visible % 4` slots of
        # `-1` inside the scan. `visible % 4 == 3` leaves none.
        glm5_next_bf16_case(
            "bf16_full_row", 1, 4098, 1, LONG_CONTEXT_MIN_COSINE, ctx, failures
        )
        glm5_next_bf16_case(
            "bf16_gap_1", 1, 4097, 1, LONG_CONTEXT_MIN_COSINE, ctx, failures
        )
        glm5_next_bf16_case(
            "bf16_gap_2", 1, 4100, 1, LONG_CONTEXT_MIN_COSINE, ctx, failures
        )
        glm5_next_bf16_case(
            "bf16_gap_3", 1, 4099, 1, LONG_CONTEXT_MIN_COSINE, ctx, failures
        )
        # MTP verify: gaps of 1, 0, 3 and 2 across the rows.
        glm5_next_bf16_case(
            "bf16_mtp_q4_long",
            2,
            4097,
            4,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
        # The same rows through the FP8 op once logical indices are dropped.
        glm5_next_fp8_case(
            "fp8_gap_3_no_logical",
            1,
            4099,
            1,
            False,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )

        # A selection reused `lag` positions after the token it was computed
        # for, as the GLM-5.2 MTP draft reuses step 0's row. Below `top_k`
        # its last `lag` slots inside the causal limit are `-1`.
        glm5_2_case(
            "plain_reused_lag_3",
            2,
            98,
            1,
            False,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
            lag=3,
        )
        glm5_2_case(
            "plain_reused_lag_3_long",
            1,
            3000,
            1,
            False,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
            lag=3,
        )
        run_case[.bfloat16, 1, GLM5_2_TOPK](
            "bf16_plain_reused_lag_3",
            2,
            98,
            1,
            False,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
            lag=3,
        )
        glm5_next_bf16_case(
            "bf16_reused_lag_2",
            2,
            98,
            4,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
            lag=2,
        )
        glm5_next_bf16_case(
            "bf16_reused_lag_2_long",
            1,
            4097,
            4,
            LONG_CONTEXT_MIN_COSINE,
            ctx,
            failures,
            lag=2,
        )
        # Without the tail, a row holds whole pools only, so a short context
        # whose `visible % 4` is nonzero ends in `-1` inside the scan.
        run_case[
            .bfloat16, GLM5_NEXT_KPOOL, GLM5_NEXT_POOL_TOPK, select_tail=False
        ](
            "bf16_no_tail_mtp_q4",
            2,
            98,
            4,
            False,
            SHORT_CONTEXT_MIN_COSINE,
            ctx,
            failures,
        )
    if len(failures) > 0:
        var msg = String("sparse kpool tail cases failed:")
        for f in failures:
            msg += " " + f
        raise Error(msg)
