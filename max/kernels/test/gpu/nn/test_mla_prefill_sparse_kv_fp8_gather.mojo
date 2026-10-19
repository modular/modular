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

"""Correctness of AMD sparse MLA prefill's gather-to-scratch composition:
the same gather kernel (`_amd_sparse_mla_gather_kv_rows`), `PaddingMask`, and
`flare_mla_decoding` validated for sparse decode, exercised at PREFILL
geometry (multiple query tokens per sequence, `total_q > batch_size`) --
exactly the call sequence `mla_prefill_branch_sparse_fp8`'s AMD arm performs,
invoked directly at the absorbed-latent level (like
`test_mla_decode_sparse_kv_fp8_gather.mojo` does for decode), so this test is
not entangled with the fused graph op's up-projection/RoPE preamble.

Every query token gathers its OWN top-k selection (`d_indices`/`topk_lengths`
are `[total_q, indices_stride]`/`[total_q]`, per query token, not per
sequence) -- causality is carried entirely by the selection, so this test
models it explicitly: sequence 0 starts fresh (`cache_offset=0`, token `i`'s
topk grows as `i+1`), sequence 1 models a CHUNKED PREFILL continuation
(`cache_offset>0`: token `i`'s topk starts already at `cache_offset+1` and
grows from there), so both sequences carry different, per-token-varying
topk_lengths. (Sparse prefill's `d_indices` are absolute physical KV rows,
unlike dense prefill's `cache_offsets` tensor argument -- there is no such
tensor on this path; `cache_offset` here is purely this test's device for
modeling "this chunk continues a longer existing history" in the topk_lengths
schedule.)

Every KV row NOT named by any of its sequence's query tokens' top-k lists is
poisoned (the same deterministic pattern amplified and saturated to the fp8
max) before the gather runs, so a kernel that silently falls back
to dense (or mis-addresses the gather) fails loudly instead of passing by
coincidence.

A RED control runs the identical (poisoned) cache through a NullMask dense
attention over the whole cache, using each sequence's LAST query token (the
one with the largest topk_lengths, so the strongest oracle signal) -- it must
FAIL the same tolerance check the sparse path passes, or the poison technique
proves nothing.

AMD-only at runtime, dual-arch at compile time: `PaddingMask` is not one of
the mask types the SM100 native prefill kernel accepts, so the `comptime if`
guard blocks elaboration of the whole body on a non-AMD build, not just its
execution (mirrors `test_mla_decode_sparse_kv_fp8_gather.mojo`). The native
SM100 sparse prefill path is already covered by
`test_mla_prefill_sparse_kv_fp8.mojo` and friends.
"""

from std.math import ceildiv, exp, isnan

from max.gpu.host import DeviceContext, default_accelerator
from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout.tile_layout import Layout as TileLayout
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_GATHER_VEC,
    _amd_sparse_mla_gather_kv_rows,
    mla_prefill_branch_sparse_fp8,
)
from nn.attention.mha_mask import NullMask, PaddingMask
from nn.attention.mha_utils import MHAConfig
from std.testing import assert_raises, assert_true
from std.utils import IndexList
from std.utils.numerics import nan

comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime K_CACHE_DIM = KV_LATENT_DIM + ROPE_DEPTH
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1

# Mirrors the production launch in mla_graph.mojo: one block packs
# GATHER_ROWS_PER_BLOCK rows of K_CACHE_DIM // _AMD_SPARSE_MLA_GATHER_VEC
# vector lanes each.
comptime GATHER_ROWS_PER_BLOCK = 7

comptime ATOL = Float64(0.2)
comptime RTOL = Float64(0.2)


def _mismatch(got: Float64, expected: Float64) -> Bool:
    """Returns whether `got` is outside tolerance of `expected`.

    NaN compares false against every bound, so it is checked explicitly;
    otherwise an all-NaN output would pass.
    """
    return isnan(got) or abs(got - expected) > ATOL + RTOL * abs(expected)


# Amplifies (then saturates to the fp8 e4m3 max, ~448) the real-row pattern
# to poison unselected rows -- a flat poison constant is not reliable
# because its dot product with Q can land near zero by cancellation.
comptime POISON_SCALE = Float32(1000.0)


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    """Returns a stride coprime with `n`, so a length-`n` ring is visited
    without repeats."""
    if n <= 1:
        return 1
    for m in [3, 5, 7, 11, 13, 17, 19]:
        if _gcd(m, n) == 1:
            return m
    return 23


@always_inline
def _real_k_pattern(bi: Int, t: Int, d: Int) -> Float32:
    """Deterministic, bounded (well within fp8 e4m3 range) per-(sequence, key
    position, dim) value used for every real (non-poisoned) KV cache row."""
    return Float32(((bi * 131 + t * 17 + d * 7) % 37) - 18) * 0.125


@always_inline
def _q_pattern(token_idx: Int, h: Int, d: Int, num_heads: Int) -> Float32:
    """Deterministic, per-QUERY-TOKEN (not per-sequence) Q value -- prefill
    has `total_q` independent query rows, each with its own top-k list."""
    var i = (token_idx * num_heads + h) * K_CACHE_DIM + d
    return Float32((i % 97) - 48) * 0.05


def _run_case[
    num_heads: Int,
](
    batch_size: Int,
    seq_q_len: Int,
    cache_len: Int,
    cache_offsets_list: List[Int],
    indices_stride: Int,
    out_host: MutPointer[BFloat16, MutAnyOrigin],
    out_red_host: MutPointer[BFloat16, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Runs the gather-to-scratch + dense-decode composition for every query
    token (`out_host`, `[total_q, num_heads, KV_LATENT_DIM]`), plus a RED
    control that runs each sequence's LAST query token through a dense
    NullMask decode directly over the (poisoned) paged cache -- no gather, no
    mask (`out_red_host`, `[batch_size, num_heads, KV_LATENT_DIM]`).
    """
    comptime dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.bfloat16
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )

    var total_q = batch_size * seq_q_len
    var max_pages_per_batch = ceildiv(cache_len, PAGE_SIZE)
    var total_pages = batch_size * max_pages_per_batch

    var block_shape = IndexList[6](
        total_pages,
        1,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = total_pages * NUM_LAYERS * PAGE_SIZE * kv_params.head_size
    var blocks_device = ctx.enqueue_create_buffer[dtype](block_elems)
    var blocks_host = ctx.enqueue_create_host_buffer[dtype](block_elems)

    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for bi in range(batch_size):
        for p in range(max_pages_per_batch):
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                bi * max_pages_per_batch + p
            )
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    # ---- per-query-token sparse indices. Sequence bi's token at local
    # position `p` (0-indexed within the chunk) selects the first
    # `min(indices_stride, cache_offsets_list[bi] + p + 1)` entries of that
    # sequence's fixed selection ring -- a nonzero `cache_offsets_list[bi]`
    # means the token already sees that many pre-existing (earlier-chunk)
    # keys before its own chunk-local causal growth, modeling a continued
    # (chunked) prefill.
    var mult = _coprime_multiplier(cache_len)
    var is_selected = List[Bool](length=batch_size * cache_len, fill=False)
    var token_topk_lengths = List[Int](length=total_q, fill=0)
    var total_indices = total_q * indices_stride
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        var base_offset = cache_offsets_list[bi]
        for p in range(seq_q_len):
            var token_idx = bi * seq_q_len + p
            var tlen = min(indices_stride, base_offset + p + 1)
            token_topk_lengths[token_idx] = tlen
            for i in range(indices_stride):
                if i < tlen:
                    var t = (i * mult + 1 + bi * 7) % cache_len
                    is_selected[bi * cache_len + t] = True
                    var page_idx = t // PAGE_SIZE
                    var tok_in_page = t % PAGE_SIZE
                    var block_id = Int(
                        lookup_table_host[bi * max_pages_per_batch + page_idx]
                    )
                    h_indices[token_idx * indices_stride + i] = Int32(
                        block_id * PAGE_SIZE + tok_in_page
                    )
                else:
                    h_indices[token_idx * indices_stride + i] = Int32(-1)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](total_q)
    for token_idx in range(total_q):
        topk_lengths_host[token_idx] = Int32(token_topk_lengths[token_idx])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](total_q)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    # ---- populate the cache: real values at any position selected by at
    # least one of its sequence's query tokens, poison (amplified +
    # saturated) everywhere else in [0, cache_len); the padded tail
    # [cache_len, max_pages_per_batch*PAGE_SIZE) is zeroed so the RED
    # control's dense NullMask view (which spans the whole padded region)
    # reads defined data rather than uninitialized memory.
    for bi in range(batch_size):
        for t in range(max_pages_per_batch * PAGE_SIZE):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var row = block_id * PAGE_SIZE + tok_in_page
            for d in range(K_CACHE_DIM):
                var v: Float32
                if t >= cache_len:
                    v = 0.0
                elif is_selected[bi * cache_len + t]:
                    v = _real_k_pattern(bi, t, d)
                else:
                    v = _real_k_pattern(bi, t, d) * POISON_SCALE
                blocks_host[
                    row * kv_params.num_heads * K_CACHE_DIM + d
                ] = cast_saturating[dtype](v)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var q_size = total_q * num_heads * K_CACHE_DIM
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for token_idx in range(total_q):
        for h in range(num_heads):
            for d in range(K_CACHE_DIM):
                q_host[
                    (token_idx * num_heads + h) * K_CACHE_DIM + d
                ] = cast_saturating[dtype](
                    _q_pattern(token_idx, h, d, num_heads)
                )
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_size = total_q * num_heads * KV_LATENT_DIM
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)
    ctx.enqueue_memset(out_device, 0)

    var out_red_size = batch_size * num_heads * KV_LATENT_DIM
    var out_red_device = ctx.enqueue_create_buffer[out_dtype](out_red_size)
    ctx.enqueue_memset(out_red_device, 0)

    var scalar_args_dev = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_memset(scalar_args_dev, 0)
    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(0),
        UInt32(cache_len),
    )
    var k_cache = kv_collection.get_key_cache(0)

    var scalar_args_buf_tt = TileTensor(
        scalar_args_dev.unsafe_ptr(), row_major[3]()
    )

    # ---- GREEN: gather-to-scratch + dense decode, one gathered "batch"
    # slot per query token (batch_size = total_q here), mirrors
    # `mla_prefill_branch_sparse_fp8`'s AMD arm exactly.
    var scratch_buf = ctx.enqueue_create_buffer[dtype](
        total_q * indices_stride * K_CACHE_DIM
    )
    var k_cache_blocks_base = k_cache.blocks_raw_ptr()
    ctx.enqueue_function[
        _amd_sparse_mla_gather_kv_rows[
            type_of(k_cache), K_CACHE_DIM, GATHER_ROWS_PER_BLOCK
        ]
    ](
        k_cache,
        k_cache_blocks_base,
        d_indices_device.unsafe_ptr(),
        topk_lengths_device.unsafe_ptr(),
        scratch_buf.unsafe_ptr(),
        Int32(indices_stride),
        grid_dim=(
            ceildiv(indices_stride, GATHER_ROWS_PER_BLOCK),
            total_q,
            1,
        ),
        block_dim=(
            K_CACHE_DIM // _AMD_SPARSE_MLA_GATHER_VEC,
            GATHER_ROWS_PER_BLOCK,
            1,
        ),
    )

    var k_scratch = TileTensor(
        scratch_buf,
        row_major(total_q, indices_stride, Idx[1], Idx[K_CACHE_DIM]),
    )
    var q_padded = TileTensor(
        q_device.unsafe_ptr(),
        TileLayout(
            (total_q, Idx[1], Idx[num_heads], Idx[K_CACHE_DIM]),
            (
                Idx[num_heads * K_CACHE_DIM],
                Idx[num_heads * K_CACHE_DIM],
                Idx[K_CACHE_DIM],
                Idx[1],
            ),
        ),
    )
    var out_padded = TileTensor(
        out_device.unsafe_ptr(),
        TileLayout(
            (total_q, Idx[1], Idx[num_heads], Idx[KV_LATENT_DIM]),
            (
                Idx[num_heads * KV_LATENT_DIM],
                Idx[num_heads * KV_LATENT_DIM],
                Idx[KV_LATENT_DIM],
                Idx[1],
            ),
        ),
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths_device.unsafe_ptr().unsafe_bitcast[UInt32](),
        row_major(total_q),
    )
    var padding_mask = PaddingMask(topk_lengths_tt)

    flare_mla_decoding[
        config=MHAConfig[dtype](num_heads, K_CACHE_DIM),
        decoding_warp_split_k=False,
    ](
        out_padded,
        q_padded,
        k_scratch,
        padding_mask,
        scale,
        ctx,
        scalar_args_buf_tt,
    )

    # ---- RED control: dense decode directly over the (poisoned) paged
    # cache -- no gather, no mask, attends over the whole cache -- using
    # each sequence's LAST query token (largest topk_lengths, so the
    # strongest divergence signal). Sequence bi's last token is row
    # `bi*seq_q_len + (seq_q_len-1)` of `q_device`, a fixed `seq_q_len`
    # apart, so a strided view (not a copy) reaches it directly. Batch bi's
    # blocks are contiguous (`lookup_table` assigns block ids
    # `bi*max_pages_per_batch + p`), so a plain row-major reshape over
    # `blocks_device` is a dense `[batch, num_keys_padded, 1, depth]` view.
    var q_last_padded = TileTensor(
        q_device.unsafe_ptr().unsafe_offset(
            (seq_q_len - 1) * num_heads * K_CACHE_DIM
        ),
        TileLayout(
            (batch_size, Idx[1], Idx[num_heads], Idx[K_CACHE_DIM]),
            (
                seq_q_len * num_heads * K_CACHE_DIM,
                seq_q_len * num_heads * K_CACHE_DIM,
                Idx[K_CACHE_DIM],
                Idx[1],
            ),
        ),
    )
    var out_red_padded = TileTensor(
        out_red_device.unsafe_ptr(),
        TileLayout(
            (batch_size, Idx[1], Idx[num_heads], Idx[KV_LATENT_DIM]),
            (
                Idx[num_heads * KV_LATENT_DIM],
                Idx[num_heads * KV_LATENT_DIM],
                Idx[KV_LATENT_DIM],
                Idx[1],
            ),
        ),
    )
    flare_mla_decoding[
        config=MHAConfig[dtype](num_heads, K_CACHE_DIM),
        decoding_warp_split_k=False,
    ](
        out_red_padded,
        q_last_padded,
        TileTensor(
            blocks_device,
            row_major(
                batch_size,
                max_pages_per_batch * PAGE_SIZE,
                Idx[1],
                Idx[K_CACHE_DIM],
            ),
        ),
        NullMask(),
        scale,
        ctx,
        scalar_args_buf_tt,
    )

    ctx.synchronize()

    var out_host_buf = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(out_host_buf, out_device)
    var out_red_host_buf = ctx.enqueue_create_host_buffer[out_dtype](
        out_red_size
    )
    ctx.enqueue_copy(out_red_host_buf, out_red_device)
    ctx.synchronize()
    for i in range(out_size):
        out_host[i] = out_host_buf[i]
    for i in range(out_red_size):
        out_red_host[i] = out_red_host_buf[i]

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = out_red_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = scalar_args_dev
    _ = scratch_buf


def _oracle[
    num_heads: Int
](
    batch_size: Int,
    seq_q_len: Int,
    cache_len: Int,
    cache_offsets_list: List[Int],
    out_ref: MutPointer[Float64, MutAnyOrigin],
) raises:
    """Restricted-softmax f64 oracle, one row per query token: attends only
    over that token's own selected (real, non-poisoned) key positions."""
    comptime scale_f64 = Float64(0.125)
    var mult = _coprime_multiplier(cache_len)

    for bi in range(batch_size):
        var base_offset = cache_offsets_list[bi]
        for p in range(seq_q_len):
            var token_idx = bi * seq_q_len + p
            var tlen = base_offset + p + 1
            for h in range(num_heads):
                var logits = List[Float64]()
                var toks = List[Int]()
                for i in range(tlen):
                    var t = (i * mult + 1 + bi * 7) % cache_len
                    var dot = Float64(0)
                    for d in range(K_CACHE_DIM):
                        dot += Float64(
                            cast_saturating[.float8_e4m3fn](
                                _q_pattern(token_idx, h, d, num_heads)
                            )
                        ) * Float64(
                            cast_saturating[.float8_e4m3fn](
                                _real_k_pattern(bi, t, d)
                            )
                        )
                    logits.append(dot * scale_f64)
                    toks.append(t)

                assert_true(len(logits) > 0, "oracle: empty softmax")
                var mx = Float64(-1e300)
                for v in logits:
                    mx = max(mx, v)
                var sum_exp = Float64(0)
                for v in logits:
                    sum_exp += exp(v - mx)

                var o_base = (token_idx * num_heads + h) * KV_LATENT_DIM
                for vd in range(KV_LATENT_DIM):
                    var acc = Float64(0)
                    for i in range(len(logits)):
                        var wgt = exp(logits[i] - mx) / sum_exp
                        acc += wgt * Float64(
                            cast_saturating[.float8_e4m3fn](
                                _real_k_pattern(bi, toks[i], vd)
                            )
                        )
                    out_ref[o_base + vd] = acc
                _ = logits
                _ = toks


def _check[
    num_heads: Int,
](
    name: String,
    batch_size: Int,
    seq_q_len: Int,
    cache_len: Int,
    cache_offsets_list: List[Int],
    indices_stride: Int,
    ctx: DeviceContext,
) raises:
    """Runs the gather (GREEN, checked against the restricted oracle for
    every query token) and the dense RED control (checked to diverge from
    the same oracle, at each sequence's last token) for one case."""
    var total_q = batch_size * seq_q_len
    var out_size = total_q * num_heads * KV_LATENT_DIM
    var out_red_size = batch_size * num_heads * KV_LATENT_DIM

    var got = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var got_red = ctx.enqueue_create_host_buffer[.bfloat16](out_red_size)
    ctx.synchronize()
    _run_case[num_heads=num_heads](
        batch_size,
        seq_q_len,
        cache_len,
        cache_offsets_list,
        indices_stride,
        got.unsafe_ptr().as_unsafe_any_origin(),
        got_red.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )

    var ref_vals = List[Float64](length=out_size, fill=0.0)
    _oracle[num_heads=num_heads](
        batch_size,
        seq_q_len,
        cache_len,
        cache_offsets_list,
        ref_vals.unsafe_ptr().as_unsafe_any_origin(),
    )

    # GREEN: every query token against its own restricted oracle row.
    var max_abs = Float64(0)
    var max_rel = Float64(0)
    var num_bad = 0
    for bi in range(batch_size):
        var b_max_abs = Float64(0)
        var b_max_rel = Float64(0)
        for p in range(seq_q_len):
            var token_idx = bi * seq_q_len + p
            for h in range(num_heads):
                for d in range(KV_LATENT_DIM):
                    var idx = (token_idx * num_heads + h) * KV_LATENT_DIM + d
                    var g = Float64(got[idx])
                    var r = ref_vals[idx]
                    var ae = abs(g - r)
                    max_abs = max(max_abs, ae)
                    b_max_abs = max(b_max_abs, ae)
                    if abs(r) > 0.05:
                        max_rel = max(max_rel, ae / abs(r))
                        b_max_rel = max(b_max_rel, ae / abs(r))
                    if _mismatch(g, r):
                        num_bad += 1
        print(
            "  [",
            name,
            "] seq=",
            bi,
            " cache_offset=",
            cache_offsets_list[bi],
            " last_topk_len=",
            min(indices_stride, cache_offsets_list[bi] + seq_q_len),
            " max_abs=",
            b_max_abs,
            " max_rel=",
            b_max_rel,
        )

    print(
        "[",
        name,
        "] GREEN (gather) overall max_abs=",
        max_abs,
        " max_rel=",
        max_rel,
        " mismatches=",
        num_bad,
        " / ",
        out_size,
        " (atol=",
        ATOL,
        " rtol=",
        RTOL,
        ")",
    )
    # RED control for the comparison: a NaN-poisoned output must count.
    got[0] = nan[.bfloat16]()
    assert_true(
        _mismatch(Float64(got[0]), ref_vals[0]),
        "[" + name + "] mismatch check does not flag a NaN output",
    )

    assert_true(
        num_bad == 0,
        "["
        + name
        + "] gather/PaddingMask output outside atol+rtol*|ref| of the"
        " restricted-softmax oracle -- gather/mask regression",
    )
    print("  GREEN PASSED")

    # RED: each sequence's last query token, dense over the whole (poisoned)
    # cache, must diverge from that same token's restricted oracle row.
    var red_num_bad = 0
    var red_max_abs = Float64(0)
    for bi in range(batch_size):
        var token_idx = bi * seq_q_len + (seq_q_len - 1)
        for h in range(num_heads):
            for d in range(KV_LATENT_DIM):
                var idx = (token_idx * num_heads + h) * KV_LATENT_DIM + d
                var red_idx = (bi * num_heads + h) * KV_LATENT_DIM + d
                var g = Float64(got_red[red_idx])
                var r = ref_vals[idx]
                var ae = abs(g - r)
                red_max_abs = max(red_max_abs, ae)
                if _mismatch(g, r):
                    red_num_bad += 1

    print(
        "[",
        name,
        "] RED (dense) max_abs=",
        red_max_abs,
        " mismatches=",
        red_num_bad,
        " / ",
        out_red_size,
    )
    assert_true(
        red_num_bad > 0,
        "["
        + name
        + "] RED CONTROL DID NOT FIRE: dense (attends over the poisoned"
        " cache) matched the restricted oracle -- poisoning is not"
        " effective at this shape",
    )
    print("  RED CONTROL FIRED AS EXPECTED (dense diverges from oracle)")


def _check_attn_sink_guard(ctx: DeviceContext) raises:
    """Exercises the AMD sparse prefill arm's guard directly through
    `mla_prefill_branch_sparse_fp8`: the sink is not read on this path, so an
    undeclared one must raise loudly rather than be silently ignored. A caller
    that knows its sink is inert opts out with `attn_sink_inert` -- left off
    here, which is what makes this a red control.

    This guard runs first in the AMD sparse arm, before the up-projection/
    RoPE preamble launches any device kernel, so the call reaches it
    directly without needing numerically meaningful inputs -- every buffer
    below is zeroed and only its *shape* matters.
    """
    comptime dtype = DType.bfloat16
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime scale_dtype = DType.float32
    comptime qk_nope_head_dim = 128
    comptime q_head_dim = qk_nope_head_dim + ROPE_DEPTH
    comptime n_scale_gran = 128
    comptime k_scale_gran = 128
    comptime num_heads = 1

    var batch_size = 1
    var seq_len = 1
    var cache_len = 1
    var num_keys = cache_len + seq_len
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * max_pages_per_batch
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )

    var block_shape = IndexList[6](
        total_pages,
        1,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = total_pages * NUM_LAYERS * PAGE_SIZE * kv_params.head_size
    var blocks_device = ctx.enqueue_create_buffer[fp8_dtype](block_elems)
    ctx.enqueue_memset(blocks_device, 0)

    var lut_size = batch_size * max_pages_per_batch
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for bi in range(batch_size):
        for p in range(max_pages_per_batch):
            lut_host[bi * max_pages_per_batch + p] = UInt32(
                bi * max_pages_per_batch + p
            )
    var lut_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lut_device, lut_host)

    var cl_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cl_host[i] = UInt32(cache_len)
    var cl_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cl_device, cl_host)
    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(cl_device, row_major(Int64(batch_size)))
    var lookup_table_tt = TileTensor(
        lut_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )

    comptime Collection = PagedKVCacheCollection[
        fp8_dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(cache_len),
    )

    var q_device = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * q_head_dim
    )
    ctx.enqueue_memset(q_device, 0)
    var out_device = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * KV_LATENT_DIM
    )
    ctx.enqueue_memset(out_device, 0)
    var kv_in_device = ctx.enqueue_create_buffer[dtype](seq_len * K_CACHE_DIM)
    ctx.enqueue_memset(kv_in_device, 0)
    var freqs_device = ctx.enqueue_create_buffer[scale_dtype](
        num_keys * ROPE_DEPTH
    )
    ctx.enqueue_memset(freqs_device, 0)
    var gamma_device = ctx.enqueue_create_buffer[dtype](KV_LATENT_DIM)
    ctx.enqueue_memset(gamma_device, 0)
    var w_uk_device = ctx.enqueue_create_buffer[fp8_dtype](
        num_heads * KV_LATENT_DIM * qk_nope_head_dim
    )
    ctx.enqueue_memset(w_uk_device, 0)
    comptime w_uk_scale_n = KV_LATENT_DIM // n_scale_gran
    comptime w_uk_scale_k = ceildiv(qk_nope_head_dim, k_scale_gran)
    var w_uk_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        num_heads * w_uk_scale_n * w_uk_scale_k
    )
    ctx.enqueue_memset(w_uk_scale_device, 0)
    var w_uv_device = ctx.enqueue_create_buffer[fp8_dtype](
        num_heads * KV_LATENT_DIM * KV_LATENT_DIM
    )
    ctx.enqueue_memset(w_uv_device, 0)
    comptime w_uv_scale_n = KV_LATENT_DIM // n_scale_gran
    comptime w_uv_scale_k = KV_LATENT_DIM // k_scale_gran
    var w_uv_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        num_heads * w_uv_scale_n * w_uv_scale_k
    )
    ctx.enqueue_memset(w_uv_scale_device, 0)
    var scalar_args_dev = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_memset(scalar_args_dev, 0)

    comptime indices_stride = 8
    var d_indices_device = ctx.enqueue_create_buffer[.int32](
        seq_len * indices_stride
    )
    ctx.enqueue_memset(d_indices_device, 0)
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](seq_len)
    ctx.enqueue_memset(topk_lengths_device, 0)
    var attn_sink_device = ctx.enqueue_create_buffer[.float32](1)
    ctx.enqueue_memset(attn_sink_device, 0)
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((seq_len, Idx[num_heads], Idx[q_head_dim])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((seq_len, Idx[num_heads], Idx[KV_LATENT_DIM])),
    )
    var kv_in_tt = TileTensor(
        kv_in_device.unsafe_ptr(), row_major((seq_len, Idx[K_CACHE_DIM]))
    )
    var freqs_tt = TileTensor(
        freqs_device.unsafe_ptr(), row_major((num_keys, Idx[ROPE_DEPTH]))
    )
    var gamma_tt = TileTensor(
        gamma_device.unsafe_ptr(), row_major(Idx[KV_LATENT_DIM])
    )
    var w_uk_tt = TileTensor(
        w_uk_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[KV_LATENT_DIM], Idx[qk_nope_head_dim])),
    )
    var w_uk_scale_tt = TileTensor(
        w_uk_scale_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[w_uk_scale_n], Idx[w_uk_scale_k])),
    )
    var w_uv_tt = TileTensor(
        w_uv_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[KV_LATENT_DIM], Idx[KV_LATENT_DIM])),
    )
    var w_uv_scale_tt = TileTensor(
        w_uv_scale_device.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[w_uv_scale_n], Idx[w_uv_scale_k])),
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(), row_major(batch_size + 1)
    )
    var scalar_args_buf_tt = TileTensor(
        scalar_args_dev.unsafe_ptr(), row_major[3]()
    )

    @always_inline
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[dtype, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    with assert_raises(contains="attention sink"):
        mla_prefill_branch_sparse_fp8[
            m_scale_granularity=1,
            n_scale_granularity=n_scale_gran,
            k_scale_granularity=k_scale_gran,
            indices_stride=indices_stride,
            target="gpu",
        ](
            out_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            kv_collection,
            UInt32(0),
            Float32(0.125),
            Float32(1e-6),
            w_uk_tt,
            w_uk_scale_tt,
            w_uv_tt,
            w_uv_scale_tt,
            scalar_args_buf_tt,
            kv_input_fn,
            ctx,
            d_indices_device.unsafe_ptr().as_unsafe_any_origin(),
            topk_lengths_device.unsafe_ptr().as_unsafe_any_origin(),
            attn_sink_device.unsafe_ptr().as_unsafe_any_origin(),
        )
    print("  [attn_sink_guard] raised as expected for a non-null attn_sink_ptr")

    _ = blocks_device
    _ = lut_device
    _ = cl_device
    _ = q_device
    _ = out_device
    _ = kv_in_device
    _ = freqs_device
    _ = gamma_device
    _ = w_uk_device
    _ = w_uk_scale_device
    _ = w_uv_device
    _ = w_uv_scale_device
    _ = scalar_args_dev
    _ = d_indices_device
    _ = topk_lengths_device
    _ = attn_sink_device
    _ = row_offsets_device


def main() raises:
    # `PaddingMask` is AMD-only by design (mirrors `mla_prefill_branch_sparse_fp8`'s
    # own `default_accelerator().is_amd_gpu()` gate): the SM100 native prefill kernels this
    # file's `flare_mla_decoding` calls into on NVIDIA only accept
    # NullMask/CausalMask/SlidingWindowCausalMask. A bare
    # `comptime if not default_accelerator().is_amd_gpu(): return` followed by more
    # code does NOT block elaboration of that code on the other arch (only
    # a real if/else does), so the whole body must be nested inside the
    # `comptime if` below. The native SM100 sparse prefill path is already
    # covered by test_mla_prefill_sparse_kv_fp8.mojo and friends.
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            # GLM-5.3 shape: 8 heads, kv_lora_rank 512, qk_rope 64
            # (depth 576, v 512). batch=2 sequences; seq 0 starts fresh
            # (cache_offset=0), seq 1 models a chunked-prefill continuation
            # (cache_offset>0).
            _check[num_heads=8](
                "prefill_gather_s17",
                batch_size=2,
                seq_q_len=17,
                cache_len=1024,
                cache_offsets_list=[0, 40],
                indices_stride=64,
                ctx=ctx,
            )
            _check[num_heads=8](
                "prefill_gather_s64",
                batch_size=2,
                seq_q_len=64,
                cache_len=4096,
                cache_offsets_list=[0, 200],
                indices_stride=300,
                ctx=ctx,
            )

            _check_attn_sink_guard(ctx)

            print("test_mla_prefill_sparse_kv_fp8_gather: ALL PASSED")
