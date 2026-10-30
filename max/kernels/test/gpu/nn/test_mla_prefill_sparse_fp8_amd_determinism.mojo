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
"""AMD sparse MLA prefill branch: repeated runs on identical inputs must give
bit-identical output.

`mla_prefill_branch_sparse_fp8` hands `valid_lengths_buf` and `zero_row_buf`
to its launches as raw pointers. If they are freed before the decode is
enqueued, the decode's split-K scratch takes their memory and overwrites the
zero row that padded slots read, so whole tokens come out wrong and differ from
run to run. The shape below (top-k 2048, 128 tokens, 8 heads) is one where the
decode picks split-K > 1.
"""

from std.math import ceildiv, isnan

from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    HostBuffer,
    default_accelerator,
)
from nn.attention.gpu.mla_graph import mla_prefill_branch_sparse_fp8
from std.testing import assert_equal, assert_true
from std.utils import IndexList

comptime dtype = DType.bfloat16
comptime fp8 = DType.float8_e4m3fn
comptime num_heads = 8
comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime K_CACHE_DIM = KV_LATENT_DIM + ROPE_DEPTH
comptime qk_nope_head_dim = 128
comptime q_head_dim = qk_nope_head_dim + ROPE_DEPTH
comptime PAGE_SIZE = 128
comptime SCALE_GRAN = 128
comptime LAYERS = 3
comptime LAYER = 1
comptime SENTINEL = Float32(30000.0)
comptime NUM_RUNS = 10


def _pattern(i: Int, period: Int, scale: Float32) -> Float32:
    return Float32((i * 7919 + 13) % period - period // 2) * scale


def _fp8_buffer(
    ctx: DeviceContext, n: Int, period: Int, scale: Float32
) raises -> DeviceBuffer[fp8]:
    var host = ctx.enqueue_create_host_buffer[fp8](n)
    ctx.synchronize()
    for i in range(n):
        host[i] = cast_saturating[fp8](_pattern(i, period, scale))
    var dev = ctx.enqueue_create_buffer[fp8](n)
    ctx.enqueue_copy(dev, host)
    return dev^


def _bf16_buffer(
    ctx: DeviceContext, n: Int, period: Int, scale: Float32
) raises -> DeviceBuffer[dtype]:
    var host = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.synchronize()
    for i in range(n):
        host[i] = BFloat16(_pattern(i, period, scale))
    var dev = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(dev, host)
    return dev^


def _run[
    TOPK: Int
](
    ctx: DeviceContext, tokens_per_seq: List[Int], max_valid: Int
) raises -> HostBuffer[dtype]:
    """Runs the branch `NUM_RUNS` times and asserts bit-identical outputs.

    Token `t` selects `(t * 37 + 11) % (max_valid + 1)` rows of its sequence's
    prefix (first token 0, second `max_valid`); the other slots are -1.
    Returns the first run's output.
    """
    var num_seqs = len(tokens_per_seq)
    var seq_len = 0
    var max_tokens = 0
    for n in tokens_per_seq:
        seq_len += n
        max_tokens = max(max_tokens, n)
    var max_cache_len = 700 + 37 * (num_seqs - 1)
    var pages_per_seq = ceildiv(max_cache_len + max_tokens, PAGE_SIZE)
    var pages = num_seqs * pages_per_seq
    var num_keys = max_cache_len + max_tokens
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=K_CACHE_DIM, is_mla=True
    )

    var blocks = _fp8_buffer(
        ctx, pages * LAYERS * PAGE_SIZE * K_CACHE_DIM, 487, 1.0 / 243.0
    )
    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](pages)
    var cl_host = ctx.enqueue_create_host_buffer[DType.uint32](num_seqs)
    var idx_host = ctx.enqueue_create_host_buffer[DType.int32](seq_len * TOPK)
    var offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        num_seqs + 1
    )
    ctx.synchronize()
    for p in range(pages):
        lut_host[p] = UInt32(p)
    var t = 0
    for b in range(num_seqs):
        var cache_len = 700 + 37 * b
        cl_host[b] = UInt32(cache_len)
        offsets_host[b] = UInt32(t)
        for _ in range(tokens_per_seq[b]):
            var valid = (t * 37 + 11) % (max_valid + 1)
            if t == 0:
                valid = 0
            elif t == 1:
                valid = max_valid
            # The identity LUT puts sequence `b`'s row `r` at
            # `b * pages_per_seq * PAGE_SIZE + r`.
            for k in range(TOPK):
                idx_host[t * TOPK + k] = Int32(
                    b * pages_per_seq * PAGE_SIZE
                    + (k * 2654435761 + t * 97) % cache_len
                ) if k < valid else Int32(-1)
            t += 1
    offsets_host[num_seqs] = UInt32(seq_len)

    var lut = ctx.enqueue_create_buffer[DType.uint32](pages)
    var cache_lengths = ctx.enqueue_create_buffer[DType.uint32](num_seqs)
    var d_indices = ctx.enqueue_create_buffer[DType.int32](seq_len * TOPK)
    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](num_seqs + 1)
    ctx.enqueue_copy(lut, lut_host)
    ctx.enqueue_copy(cache_lengths, cl_host)
    ctx.enqueue_copy(d_indices, idx_host)
    ctx.enqueue_copy(row_offsets, offsets_host)
    var topk_lengths = ctx.enqueue_create_buffer[DType.int32](seq_len)
    topk_lengths.enqueue_fill(Int32(TOPK))

    var q = _bf16_buffer(ctx, seq_len * num_heads * q_head_dim, 509, 0.01)
    var kv_in = _bf16_buffer(ctx, seq_len * K_CACHE_DIM, 487, 0.01)
    var freqs_host = ctx.enqueue_create_host_buffer[DType.float32](
        num_keys * ROPE_DEPTH
    )
    ctx.synchronize()
    for i in range(num_keys * ROPE_DEPTH):
        freqs_host[i] = _pattern(i, 101, 0.01)
    var freqs = ctx.enqueue_create_buffer[DType.float32](num_keys * ROPE_DEPTH)
    ctx.enqueue_copy(freqs, freqs_host)
    var gamma = ctx.enqueue_create_buffer[dtype](KV_LATENT_DIM)
    gamma.enqueue_fill(BFloat16(1.0))

    comptime scale_n = KV_LATENT_DIM // SCALE_GRAN
    comptime w_uk_scale_k = ceildiv(qk_nope_head_dim, SCALE_GRAN)
    comptime w_uv_scale_k = KV_LATENT_DIM // SCALE_GRAN
    var w_uk = _fp8_buffer(
        ctx, num_heads * KV_LATENT_DIM * qk_nope_head_dim, 251, 0.01
    )
    var w_uv = _fp8_buffer(
        ctx, num_heads * KV_LATENT_DIM * KV_LATENT_DIM, 241, 0.01
    )
    var w_uk_scale = ctx.enqueue_create_buffer[DType.float32](
        num_heads * scale_n * w_uk_scale_k
    )
    var w_uv_scale = ctx.enqueue_create_buffer[DType.float32](
        num_heads * scale_n * w_uv_scale_k
    )
    w_uk_scale.enqueue_fill(Float32(1.0))
    w_uv_scale.enqueue_fill(Float32(1.0))
    var scalar_args = ctx.enqueue_create_buffer[DType.int64](3)
    scalar_args.enqueue_fill(Int64(0))

    var out_elems = seq_len * num_heads * KV_LATENT_DIM
    var out = ctx.enqueue_create_buffer[dtype](out_elems)
    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks,
        row_major(
            Int64(pages),
            Idx[1],
            Int64(LAYERS),
            Idx[PAGE_SIZE],
            Idx[1],
            Idx[K_CACHE_DIM],
        ),
    )
    var cache_lengths_tt = TileTensor(cache_lengths, row_major(Int64(num_seqs)))
    var lut_tt = TileTensor(
        lut, row_major(Int64(num_seqs), Int64(pages_per_seq))
    )
    comptime Collection = PagedKVCacheCollection[
        fp8,
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
        lut_tt.as_imm().as_unsafe_any_origin(),
        UInt32(max_tokens),
        UInt32(max_cache_len),
    )

    var q_tt = TileTensor(
        q.unsafe_ptr(), row_major((seq_len, Idx[num_heads], Idx[q_head_dim]))
    )
    var kv_in_tt = TileTensor(
        kv_in.unsafe_ptr(), row_major((seq_len, Idx[K_CACHE_DIM]))
    )
    var freqs_tt = TileTensor(
        freqs.unsafe_ptr(), row_major((num_keys, Idx[ROPE_DEPTH]))
    )
    var gamma_tt = TileTensor(gamma.unsafe_ptr(), row_major(Idx[KV_LATENT_DIM]))
    var w_uk_tt = TileTensor(
        w_uk.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[KV_LATENT_DIM], Idx[qk_nope_head_dim])),
    )
    var w_uk_scale_tt = TileTensor(
        w_uk_scale.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[scale_n], Idx[w_uk_scale_k])),
    )
    var w_uv_tt = TileTensor(
        w_uv.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[KV_LATENT_DIM], Idx[KV_LATENT_DIM])),
    )
    var w_uv_scale_tt = TileTensor(
        w_uv_scale.unsafe_ptr(),
        row_major((Idx[num_heads], Idx[scale_n], Idx[w_uv_scale_k])),
    )
    var row_offsets_tt = TileTensor(
        row_offsets.unsafe_ptr(), row_major(num_seqs + 1)
    )
    var scalar_args_tt = TileTensor(scalar_args.unsafe_ptr(), row_major[3]())
    var out_tt = TileTensor(
        out.unsafe_ptr(),
        row_major((seq_len, Idx[num_heads], Idx[KV_LATENT_DIM])),
    )

    @always_inline
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[dtype, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    var first = ctx.enqueue_create_host_buffer[dtype](out_elems)
    var cur = ctx.enqueue_create_host_buffer[dtype](out_elems)
    var runs_differing = 0
    for r in range(NUM_RUNS):
        out.enqueue_fill(BFloat16(SENTINEL))
        mla_prefill_branch_sparse_fp8[
            m_scale_granularity=1,
            n_scale_granularity=SCALE_GRAN,
            k_scale_granularity=SCALE_GRAN,
            indices_stride=TOPK,
            target="gpu",
            attn_sink_inert=True,
        ](
            out_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            kv_collection,
            UInt32(LAYER),
            Float32(0.0625),
            Float32(1e-6),
            w_uk_tt,
            w_uk_scale_tt,
            w_uv_tt,
            w_uv_scale_tt,
            scalar_args_tt,
            kv_input_fn,
            ctx,
            d_indices.unsafe_ptr().as_unsafe_any_origin(),
            topk_lengths.unsafe_ptr().as_unsafe_any_origin(),
            None,
        )
        ctx.enqueue_copy(cur, out)
        ctx.synchronize()
        if r == 0:
            ctx.enqueue_copy(first, out)
            ctx.synchronize()
            continue
        var num_diff = 0
        for i in range(out_elems):
            if first[i].to_bits() != cur[i].to_bits():
                num_diff += 1
        if num_diff > 0:
            runs_differing += 1
            print(t"  run {r} differs in {num_diff} of {out_elems}")
    print(
        t"topk {TOPK} tokens {seq_len} runs differing from run 0:"
        t" {runs_differing} of {NUM_RUNS - 1}"
    )
    assert_equal(runs_differing, 0, "output differs between identical runs")

    _ = blocks^
    _ = lut^
    _ = cache_lengths^
    _ = d_indices^
    _ = row_offsets^
    _ = topk_lengths^
    _ = q^
    _ = kv_in^
    _ = freqs^
    _ = gamma^
    _ = w_uk^
    _ = w_uv^
    _ = w_uk_scale^
    _ = w_uv_scale^
    _ = scalar_args^
    _ = out^
    return first^


def _check_matches_narrow_topk(ctx: DeviceContext) raises:
    """With at most 256 selected rows per token the same selection runs at
    stride 2048 (padded slots, split-K > 1) and at stride 256; both must
    agree to within FP8 rounding of the attention weights."""
    var wide = _run[2048](ctx, [128], 256)
    var narrow = _run[256](ctx, [128], 256)
    var peak = Float64(0)
    var worst = Float64(0)
    for i in range(len(wide)):
        var a = Float64(wide[i])
        var b = Float64(narrow[i])
        assert_true(not isnan(a) and abs(a) < Float64(SENTINEL))
        peak = max(peak, abs(b))
        worst = max(worst, abs(a - b))
    print("wide vs narrow: worst diff", worst, "peak", peak)
    assert_true(peak > 0, "reference output is all zero")
    assert_true(worst <= 0.1 * peak, "stride 2048 disagrees with stride 256")


def main() raises:
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            _ = _run[2048](ctx, [128], 2048)
            _ = _run[2048](ctx, [40, 17], 2048)
            _check_matches_narrow_topk(ctx)
            print("test_mla_prefill_sparse_fp8_amd_determinism: ALL PASSED")
