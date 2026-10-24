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

"""Correctness of AMD sparse MLA decode's gather-to-scratch machinery:
the gather kernel (`_amd_sparse_mla_gather_kv_rows`) that copies top-k
selected latent rows out of a paged FP8 KV cache, `PaddingMask`, and their
composition with the existing (validated) dense `flare_mla_decoding` kernel
-- exactly the call sequence `mla_decode_branch_fp8`'s AMD sparse branch
performs, invoked directly at the absorbed-latent level (like
`test_mla_decode_sparse_kv_fp8.mojo` does for the SM100 native sparse
kernel), so this test is not entangled with the fused graph op's
up-projection/RoPE preamble.

Every KV row NOT named by a batch's top-k list is poisoned (the same
deterministic pattern amplified and saturated to the fp8 max) before the
gather runs. At smoke sequence lengths `topk >= seq_len`, dense and sparse
compute the same thing and a poison-blind oracle would pass on a dense
fallback; poisoning forces `topk < cache_len` here so a kernel that
silently falls back to dense (or mis-addresses the gather) fails loudly
instead.

A RED control runs the identical (poisoned) cache through a NullMask dense
attention over the whole cache -- it must FAIL the same tolerance check the
sparse path passes, or the poison technique proves nothing.

AMD-only at runtime (`PaddingMask` is not one of the mask types the SM100
native decode kernel accepts, so the `comptime if` guard must block
elaboration of the whole body on a non-AMD build, not just its execution).
The build target stays dual-arch (`//:mi355_gpu` and `//:b200_gpu`) so it
compiles clean on both; the native SM100 sparse decode path is already
covered by `test_mla_decode_sparse_kv_fp8.mojo` and friends.
"""

from std.math import ceildiv, exp, isnan
from std.memory import bitcast

from max.gpu.host import DeviceAttribute, DeviceContext, default_accelerator
from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout.tile_layout import Layout as TileLayout
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_sparse_mla_decode_num_partitions,
)
from nn.attention.gpu.mla import flare_mla_decoding, flare_mla_decoding_rows
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_GATHER_VEC,
    _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS,
    _amd_sparse_mla_gather_kv_rows,
    _amd_sparse_mla_gather_kv_rows_and_lengths,
    _amd_sparse_mla_row_valid_lengths,
    mla_decode_branch_fp8,
)
from nn.attention.mha_mask import NullMask, PaddingMask
from nn.attention.mha_operand import KVCacheMHAOperand, SparseRowsMHAOperand
from nn.attention.mha_utils import MHAConfig, NonNullPointer
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
    """Deterministic, bounded (well within fp8 e4m3 range) per-(batch, key
    position, dim) value used for every real (non-poisoned) KV cache row."""
    return Float32(((bi * 131 + t * 17 + d * 7) % 37) - 18) * 0.125


@always_inline
def _q_pattern(bi: Int, h: Int, d: Int, num_heads: Int) -> Float32:
    var i = (bi * num_heads + h) * K_CACHE_DIM + d
    return Float32((i % 97) - 48) * 0.05


def _run_case[
    num_heads: Int,
    sparse_mla: Bool,
    use_derived_valid_lengths: Bool = False,
    fused: Bool = False,
](
    batch_size: Int,
    cache_len: Int,
    topk_lengths_list: List[Int],
    indices_stride: Int,
    out_host: MutPointer[BFloat16, MutAnyOrigin],
    ctx: DeviceContext,
    num_partitions: Optional[Int] = None,
) raises:
    """Runs the gather-to-scratch + dense-decode composition once (S=1) and
    copies the output to the host. `sparse_mla=False` is the RED-control
    arm: identical (poisoned) cache, but a `NullMask` dense decode directly
    over the paged cache (no gather, attends over the whole cache).

    When `use_derived_valid_lengths=True` (only valid with `sparse_mla=True`),
    the dense decode uses `SparseIndexMask` instead of `PaddingMask`.

    `fused=True` (only with `sparse_mla=True`) reads the selected rows in
    place through `SparseRowsMHAOperand` instead of gathering them first.
    """
    comptime dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.bfloat16
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )

    var q_len = 1
    var num_keys = cache_len + q_len
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
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

    # ---- sparse indices: per-batch distinct positions in [0, cache_len),
    # never touching a would-be "new write" row (none here -- this is a
    # pure decode-read test, the cache is fully pre-populated).
    var total_indices = batch_size * indices_stride
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    var mult = _coprime_multiplier(cache_len)
    var is_selected = List[Bool](length=batch_size * cache_len, fill=False)
    for bi in range(batch_size):
        var tlen = topk_lengths_list[bi]
        for i in range(indices_stride):
            if i < tlen:
                var t = (i * mult + 1 + bi * 7) % cache_len
                is_selected[bi * cache_len + t] = True
                var page_idx = t // PAGE_SIZE
                var tok_in_page = t % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[bi * indices_stride + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
            else:
                h_indices[bi * indices_stride + i] = Int32(-1)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_lengths_list[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    # ---- populate the cache: real values at selected rows, poison
    # (amplified + saturated) everywhere else in [0, cache_len); the padded
    # tail [cache_len, max_pages_per_batch*PAGE_SIZE) is zeroed so the RED
    # control's dense NullMask view (which spans the whole padded region,
    # since num_keys there isn't otherwise threaded per-batch) reads defined
    # data rather than uninitialized memory.
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

    var q_size = batch_size * num_heads * K_CACHE_DIM
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for bi in range(batch_size):
        for h in range(num_heads):
            for d in range(K_CACHE_DIM):
                q_host[
                    (bi * num_heads + h) * K_CACHE_DIM + d
                ] = cast_saturating[dtype](_q_pattern(bi, h, d, num_heads))
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_size = batch_size * num_heads * KV_LATENT_DIM
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)
    ctx.enqueue_memset(out_device, 0)

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

    var q_padded = TileTensor(
        q_device.unsafe_ptr(),
        TileLayout(
            (batch_size, Idx[1], Idx[num_heads], Idx[K_CACHE_DIM]),
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
            (batch_size, Idx[1], Idx[num_heads], Idx[KV_LATENT_DIM]),
            (
                Idx[num_heads * KV_LATENT_DIM],
                Idx[num_heads * KV_LATENT_DIM],
                Idx[KV_LATENT_DIM],
                Idx[1],
            ),
        ),
    )
    var scalar_args_buf_tt = TileTensor(
        scalar_args_dev.unsafe_ptr(), row_major[3]()
    )

    comptime if sparse_mla and fused:
        var zero_row_buf = ctx.enqueue_create_buffer[dtype](K_CACHE_DIM)
        ctx.enqueue_memset(zero_row_buf, 0)
        var k_rows = SparseRowsMHAOperand[row_width=K_CACHE_DIM](
            KVCacheMHAOperand(k_cache),
            k_cache.blocks_raw_ptr(),
            zero_row_buf.unsafe_ptr(),
            d_indices_device.unsafe_ptr(),
            topk_lengths_device.unsafe_ptr().unsafe_bitcast[UInt32](),
            indices_stride,
        )
        var topk_lengths_tt = TileTensor(
            topk_lengths_device.unsafe_ptr().unsafe_bitcast[UInt32](),
            row_major(batch_size),
        )
        flare_mla_decoding_rows[
            config=MHAConfig[dtype](num_heads, K_CACHE_DIM)
        ](
            out_padded,
            q_padded,
            k_rows,
            indices_stride,
            PaddingMask(topk_lengths_tt),
            scale,
            ctx,
            scalar_args_buf_tt,
        )
        _ = zero_row_buf
    elif sparse_mla:
        # Mirrors mla_decode_branch_fp8's AMD sparse branch exactly: gather
        # the selected rows into a dense scratch, then run the validated
        # dense decode kernel over it with a padding-only mask.
        var scratch_buf = ctx.enqueue_create_buffer[dtype](
            batch_size * indices_stride * K_CACHE_DIM
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
                batch_size,
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
            row_major(batch_size, indices_stride, Idx[1], Idx[K_CACHE_DIM]),
        )

        var topk_lengths_tt = TileTensor(
            topk_lengths_device.unsafe_ptr().unsafe_bitcast[UInt32](),
            row_major(batch_size),
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
            num_partitions=num_partitions,
        )
        _ = scratch_buf
    else:
        # RED control: dense decode directly over the (poisoned) paged
        # cache -- no gather, no mask, attends over the whole cache.
        # Batch bi's blocks are contiguous (`lookup_table` assigns block ids
        # bi*max_pages_per_batch + p), so a plain row-major reshape over
        # `blocks_device` is a dense [batch, num_keys_padded, 1, depth] view.
        flare_mla_decoding[
            config=MHAConfig[dtype](num_heads, K_CACHE_DIM),
            decoding_warp_split_k=False,
        ](
            out_padded,
            q_padded,
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
    ctx.synchronize()
    for i in range(out_size):
        out_host[i] = out_host_buf[i]

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = scalar_args_dev


def _oracle[
    num_heads: Int
](
    batch_size: Int,
    cache_len: Int,
    topk_lengths_list: List[Int],
    out_ref: MutPointer[Float64, MutAnyOrigin],
) raises:
    """Restricted-softmax f64 oracle: attends only over each batch's
    selected (real, non-poisoned) key positions."""
    comptime scale_f64 = Float64(0.125)
    var mult = _coprime_multiplier(cache_len)

    for bi in range(batch_size):
        var tlen = topk_lengths_list[bi]
        for h in range(num_heads):
            var logits = List[Float64]()
            var toks = List[Int]()
            for i in range(tlen):
                var t = (i * mult + 1 + bi * 7) % cache_len
                var dot = Float64(0)
                for d in range(K_CACHE_DIM):
                    dot += Float64(
                        cast_saturating[.float8_e4m3fn](
                            _q_pattern(bi, h, d, num_heads)
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

            var o_base = (bi * num_heads + h) * KV_LATENT_DIM
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
    sparse_mla: Bool,
](
    name: String,
    batch_size: Int,
    cache_len: Int,
    topk_lengths_list: List[Int],
    indices_stride: Int,
    ctx: DeviceContext,
    *,
    expect_pass: Bool,
    num_partitions: Optional[Int] = None,
) raises:
    var out_size = batch_size * num_heads * KV_LATENT_DIM

    var got = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.synchronize()
    _run_case[num_heads=num_heads, sparse_mla=sparse_mla](
        batch_size,
        cache_len,
        topk_lengths_list,
        indices_stride,
        got.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
        num_partitions,
    )

    var ref_vals = List[Float64](length=out_size, fill=0.0)
    _oracle[num_heads=num_heads](
        batch_size,
        cache_len,
        topk_lengths_list,
        ref_vals.unsafe_ptr().as_unsafe_any_origin(),
    )

    var max_abs = Float64(0)
    var max_rel = Float64(0)
    var num_bad = 0
    for bi in range(batch_size):
        var b_max_abs = Float64(0)
        var b_max_rel = Float64(0)
        for h in range(num_heads):
            for d in range(KV_LATENT_DIM):
                var idx = (bi * num_heads + h) * KV_LATENT_DIM + d
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
            "] batch=",
            bi,
            " topk_len=",
            topk_lengths_list[bi],
            " max_abs=",
            b_max_abs,
            " max_rel=",
            b_max_rel,
        )

    print(
        "[",
        name,
        "] sparse_mla=",
        sparse_mla,
        " overall max_abs=",
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

    if expect_pass:
        assert_true(
            num_bad == 0,
            "["
            + name
            + "] sparse_mla=True: output outside atol+rtol*|ref| of the"
            " restricted-softmax oracle -- gather/mask regression",
        )
        print("  PASSED")
    else:
        assert_true(
            num_bad > 0,
            "["
            + name
            + "] RED CONTROL DID NOT FIRE: dense (attends over the"
            " poisoned cache) matched the restricted oracle -- poisoning"
            " is not effective at this shape",
        )
        print("  RED CONTROL FIRED AS EXPECTED (dense diverges from oracle)")


def _check_s_gt_1_guard(ctx: DeviceContext) raises:
    """Exercises the AMD sparse decode arm's S=1-only guard directly
    through `mla_decode_branch_fp8`: a ragged batch where every sequence
    has 2 query tokens (S=2, as MTP / speculative-decode verify would
    produce) must raise loudly rather than silently mis-run with the
    wrong batch/head geometry (`batch_size = seq_len` assumes one row per
    sequence).

    This guard runs first in the AMD sparse arm, before the
    up-projection/RoPE preamble launches any device kernel, so the call
    reaches it directly without needing numerically meaningful inputs --
    every buffer below is zeroed and only its *shape* matters.
    """
    comptime dtype = DType.bfloat16
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime scale_dtype = DType.float32
    comptime qk_nope_head_dim = 128
    comptime q_head_dim = qk_nope_head_dim + ROPE_DEPTH
    comptime n_scale_gran = 128
    comptime k_scale_gran = 128
    comptime num_heads = 1

    var batch_size = 2
    var q_len = 2  # S=2: the unsupported case this guard must reject.
    var seq_len = batch_size * q_len
    var cache_len = 1
    var num_keys = cache_len + q_len
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
        UInt32(q_len),
        UInt32(cache_len),
    )

    var q_device = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * q_head_dim
    )
    ctx.enqueue_memset(q_device, 0)
    var kv_in_device = ctx.enqueue_create_buffer[dtype](seq_len * K_CACHE_DIM)
    ctx.enqueue_memset(kv_in_device, 0)
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
    var freqs_device = ctx.enqueue_create_buffer[scale_dtype](
        num_keys * ROPE_DEPTH
    )
    ctx.enqueue_memset(freqs_device, 0)
    var gamma_device = ctx.enqueue_create_buffer[dtype](KV_LATENT_DIM)
    ctx.enqueue_memset(gamma_device, 0)
    var out_device = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * KV_LATENT_DIM
    )
    ctx.enqueue_memset(out_device, 0)

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var indices_stride = 8
    var d_indices_device = ctx.enqueue_create_buffer[.int32](
        seq_len * indices_stride
    )
    ctx.enqueue_memset(d_indices_device, 0)
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_memset(topk_lengths_device, 0)
    var scalar_args_dev = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_memset(scalar_args_dev, 0)
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

    with assert_raises(contains="one query token per sequence"):
        mla_decode_branch_fp8[
            m_scale_granularity=1,
            n_scale_granularity=n_scale_gran,
            k_scale_granularity=k_scale_gran,
            mask_str="causal",
            target="gpu",
            sparse_mla=True,
            fp8_q=True,
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
            d_indices=d_indices_device.unsafe_ptr().as_unsafe_any_origin(),
            indices_stride=indices_stride,
            topk_lengths=NonNullPointer[DType.int32](topk_lengths_device),
        )
    print("  [s_gt_1_guard] raised as expected for S=2 (MTP-shaped) input")

    _ = blocks_device
    _ = lut_device
    _ = cl_device
    _ = q_device
    _ = kv_in_device
    _ = w_uk_device
    _ = w_uk_scale_device
    _ = w_uv_device
    _ = w_uv_scale_device
    _ = freqs_device
    _ = gamma_device
    _ = out_device
    _ = row_offsets_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = scalar_args_dev


def _run_lagged_decode[
    with_topk_lengths: Bool
](
    selection_lag: Int,
    topk_lengths_list: List[Int],
    out_host: MutPointer[BFloat16, MutAnyOrigin],
    ctx: DeviceContext,
    num_partitions_in: Optional[Int] = None,
) raises:
    """Runs the AMD sparse decode arm end to end on rows selected two tokens
    early, through `mla_decode_branch_fp8`, with deterministic non-zero data.
    A partition count takes the in-place arm, as the production op does.

    Row `b` holds the `cache_len + 1 - 2` keys its token saw two positions
    ago, then `-1`. `selection_lag` and `topk_lengths` are the knobs under
    test; everything else is identical across calls.
    """
    comptime dtype = DType.bfloat16
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime scale_dtype = DType.float32
    comptime qk_nope_head_dim = 128
    comptime q_head_dim = qk_nope_head_dim + ROPE_DEPTH
    comptime n_scale_gran = 128
    comptime k_scale_gran = 128
    comptime num_heads = 8
    comptime row_lag = 2
    comptime indices_stride = 128

    # The last row sees no key at all two tokens early.
    var cache_lens: List[Int] = [20, 5, 1]
    var batch_size = len(cache_lens)
    var seq_len = batch_size
    var max_cache_len = 0
    for b in range(batch_size):
        max_cache_len = max(max_cache_len, cache_lens[b])
    var num_keys = max_cache_len + 1
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * max_pages_per_batch
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )

    var block_elems = total_pages * NUM_LAYERS * PAGE_SIZE * K_CACHE_DIM
    var blocks_host = ctx.enqueue_create_host_buffer[fp8_dtype](block_elems)
    for i in range(block_elems):
        blocks_host[i] = cast_saturating[fp8_dtype](
            Float32((i * 37) % 23 - 11) * 0.05
        )
    var blocks_device = ctx.enqueue_create_buffer[fp8_dtype](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var lut_size = batch_size * max_pages_per_batch
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lut_host[i] = UInt32(i)
    var lut_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lut_device, lut_host)

    var cl_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        cl_host[b] = UInt32(cache_lens[b])
    var cl_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cl_device, cl_host)

    var h_indices = ctx.enqueue_create_host_buffer[.int32](
        seq_len * indices_stride
    )
    for b in range(batch_size):
        var packed = min(indices_stride, cache_lens[b] + 1 - row_lag)
        for i in range(indices_stride):
            var value = Int32(-1)
            if i < packed:
                value = Int32(
                    (b * max_pages_per_batch + i // PAGE_SIZE) * PAGE_SIZE
                    + i % PAGE_SIZE
                )
            h_indices[b * indices_stride + i] = value
    var d_indices_device = ctx.enqueue_create_buffer[.int32](
        seq_len * indices_stride
    )
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for b in range(batch_size):
        topk_host[b] = Int32(topk_lengths_list[b])
    var topk_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_device, topk_host)

    var q_elems = seq_len * num_heads * q_head_dim
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_elems)
    for i in range(q_elems):
        q_host[i] = Float32((i * 29) % 17 - 8).cast[dtype]() * 0.05
    var q_device = ctx.enqueue_create_buffer[dtype](q_elems)
    ctx.enqueue_copy(q_device, q_host)

    var kv_in_host = ctx.enqueue_create_host_buffer[dtype](
        seq_len * K_CACHE_DIM
    )
    for i in range(seq_len * K_CACHE_DIM):
        kv_in_host[i] = Float32((i * 13) % 11 - 5).cast[dtype]() * 0.05
    var kv_in_device = ctx.enqueue_create_buffer[dtype](seq_len * K_CACHE_DIM)
    ctx.enqueue_copy(kv_in_device, kv_in_host)

    comptime w_uk_elems = num_heads * KV_LATENT_DIM * qk_nope_head_dim
    comptime w_uv_elems = num_heads * KV_LATENT_DIM * KV_LATENT_DIM
    comptime w_uk_scale_n = KV_LATENT_DIM // n_scale_gran
    comptime w_uk_scale_k = ceildiv(qk_nope_head_dim, k_scale_gran)
    comptime w_uv_scale_n = KV_LATENT_DIM // n_scale_gran
    comptime w_uv_scale_k = KV_LATENT_DIM // k_scale_gran
    var w_uk_host = ctx.enqueue_create_host_buffer[fp8_dtype](w_uk_elems)
    for i in range(w_uk_elems):
        w_uk_host[i] = cast_saturating[fp8_dtype](Float32((i * 7) % 9 - 4))
    var w_uk_device = ctx.enqueue_create_buffer[fp8_dtype](w_uk_elems)
    ctx.enqueue_copy(w_uk_device, w_uk_host)
    var w_uv_host = ctx.enqueue_create_host_buffer[fp8_dtype](w_uv_elems)
    for i in range(w_uv_elems):
        w_uv_host[i] = cast_saturating[fp8_dtype](Float32((i * 5) % 7 - 3))
    var w_uv_device = ctx.enqueue_create_buffer[fp8_dtype](w_uv_elems)
    ctx.enqueue_copy(w_uv_device, w_uv_host)
    var w_uk_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        num_heads * w_uk_scale_n * w_uk_scale_k
    )
    w_uk_scale_device.enqueue_fill(Float32(0.02))
    var w_uv_scale_device = ctx.enqueue_create_buffer[scale_dtype](
        num_heads * w_uv_scale_n * w_uv_scale_k
    )
    w_uv_scale_device.enqueue_fill(Float32(0.02))
    var freqs_device = ctx.enqueue_create_buffer[scale_dtype](
        num_keys * ROPE_DEPTH
    )
    ctx.enqueue_memset(freqs_device, 0)
    var gamma_device = ctx.enqueue_create_buffer[dtype](KV_LATENT_DIM)
    gamma_device.enqueue_fill(Scalar[dtype](1))
    var out_device = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * KV_LATENT_DIM
    )
    ctx.enqueue_memset(out_device, 0)
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    var scalar_args_dev = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_memset(scalar_args_dev, 0)
    ctx.synchronize()

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
        UInt32(1),
        UInt32(max_cache_len),
    )

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

    comptime if with_topk_lengths:
        mla_decode_branch_fp8[
            m_scale_granularity=1,
            n_scale_granularity=n_scale_gran,
            k_scale_granularity=k_scale_gran,
            mask_str="causal",
            target="gpu",
            sparse_mla=True,
            fp8_q=True,
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
            d_indices=d_indices_device.unsafe_ptr().as_unsafe_any_origin(),
            indices_stride=indices_stride,
            topk_lengths=NonNullPointer[DType.int32](topk_device),
            num_partitions_in=num_partitions_in,
            selection_lag=selection_lag,
        )
    else:
        mla_decode_branch_fp8[
            m_scale_granularity=1,
            n_scale_granularity=n_scale_gran,
            k_scale_granularity=k_scale_gran,
            mask_str="causal",
            target="gpu",
            sparse_mla=True,
            fp8_q=True,
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
            d_indices=d_indices_device.unsafe_ptr().as_unsafe_any_origin(),
            indices_stride=indices_stride,
            num_partitions_in=num_partitions_in,
            selection_lag=selection_lag,
        )

    var out_host_buf = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * KV_LATENT_DIM
    )
    ctx.enqueue_copy(out_host_buf, out_device)
    ctx.synchronize()
    for i in range(seq_len * num_heads * KV_LATENT_DIM):
        out_host[i] = out_host_buf[i]

    _ = blocks_device
    _ = lut_device
    _ = cl_device
    _ = d_indices_device
    _ = topk_device
    _ = q_device
    _ = kv_in_device
    _ = w_uk_device
    _ = w_uk_scale_device
    _ = w_uv_device
    _ = w_uv_scale_device
    _ = freqs_device
    _ = gamma_device
    _ = out_device
    _ = row_offsets_device
    _ = scalar_args_dev


def _check_selection_lag_end_to_end(ctx: DeviceContext) raises:
    """The decode arm forwards `selection_lag` to the valid length.

    Rows selected two tokens early hold two fewer keys than the token sees
    now. Passing the lag must equal capping by explicit `topk_lengths` (the
    same lengths a second way), and dropping the lag must not: it admits two
    `-1` slots per row as zero keys.
    """
    comptime out_size = 3 * 8 * KV_LATENT_DIM
    var true_lengths: List[Int] = [19, 4, 0]
    var unused: List[Int] = [0, 0, 0]
    var lagged = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var capped = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var dropped = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.synchronize()
    _run_lagged_decode[with_topk_lengths=False](
        2, unused, lagged.unsafe_ptr().as_unsafe_any_origin(), ctx
    )
    _run_lagged_decode[with_topk_lengths=True](
        0, true_lengths, capped.unsafe_ptr().as_unsafe_any_origin(), ctx
    )
    _run_lagged_decode[with_topk_lengths=False](
        0, unused, dropped.unsafe_ptr().as_unsafe_any_origin(), ctx
    )
    var vs_capped = 0
    var vs_dropped = 0
    for i in range(out_size):
        var bits = bitcast[DType.uint16](lagged[i])
        if bits != bitcast[DType.uint16](capped[i]):
            vs_capped += 1
        if bits != bitcast[DType.uint16](dropped[i]):
            vs_dropped += 1
    print(
        "  [selection_lag] lagged vs capped mismatches:",
        vs_capped,
        " lagged vs dropped-lag mismatches:",
        vs_dropped,
    )
    assert_true(vs_capped == 0, "selection_lag does not reach the valid length")
    assert_true(vs_dropped > 0, "RED control: dropping the lag changed nothing")

    # The in-place arm, with both ways of bounding the rows, attends the same
    # keys as the gather, including the keyless row at one partition, where no
    # reduce runs.
    var in_place = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    for num_partitions in [1, 4]:
        for with_topk_lengths in [False, True]:
            if with_topk_lengths:
                _run_lagged_decode[with_topk_lengths=True](
                    0,
                    true_lengths,
                    in_place.unsafe_ptr().as_unsafe_any_origin(),
                    ctx,
                    num_partitions_in=num_partitions,
                )
            else:
                _run_lagged_decode[with_topk_lengths=False](
                    2,
                    unused,
                    in_place.unsafe_ptr().as_unsafe_any_origin(),
                    ctx,
                    num_partitions_in=num_partitions,
                )
            var bad = 0
            for i in range(out_size):
                if _mismatch(
                    Float64(in_place[i].cast[DType.float64]()),
                    Float64(lagged[i].cast[DType.float64]()),
                ):
                    bad += 1
            print(
                "  [selection_lag] in place (np=",
                num_partitions,
                " topk_lengths=",
                with_topk_lengths,
                ") vs gather mismatches:",
                bad,
            )
            assert_true(
                bad == 0, "the in-place decode arm differs from the gather"
            )


def _run_sentinel_case[
    num_heads: Int,
    use_derived_valid_lengths: Bool = True,
    mask_ignores_topk: Bool = False,
    no_topk_lengths: Bool = False,
](
    batch_size: Int,
    cache_len: Int,
    indices_stride: Int,
    sentinel_positions: List[Int],
    topk_lengths_list: List[Int],
    out_host: MutPointer[BFloat16, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Reproduces the production bug shape: `topk_lengths == indices_stride`
    (the constant passed through Python) while `d_indices` contains `-1`
    sentinels in the MIDDLE of the row. A length-based mask cannot know
    which slots are invalid.

    With `use_derived_valid_lengths=False` this is a RED control: `PaddingMask`
    sees every column as live, so the gathered-zero invalid slot corrupts
    softmax. With `use_derived_valid_lengths=True` (production after the fix)
    the sentinel-aware mask hides the slot.

    `topk_lengths_list` may stop short of the first `-1`; the gather then
    zero-fills the slots in between. `mask_ignores_topk=True` feeds the
    valid-length kernel the full stride instead, which re-admits those zero
    rows (RED control for the `min(first -1, topk_lengths)` cap).

    `no_topk_lengths=True` runs both kernels without a lengths array, as the
    decode arm does when the graph omits `topk_lengths`; `topk_lengths_list`
    must then be the full stride.
    """
    comptime dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.bfloat16
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )

    var q_len = 1
    var num_keys = cache_len + q_len
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
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

    # ---- sparse indices: full-length (`topk_lengths == indices_stride`)
    # with a `-1` suffix from `sentinel_positions[bi]` on, which is the shape
    # the indexer's bounded top-k produces. Everything from that column on is
    # an invalid slot the gather zero-fills.
    var total_indices = batch_size * indices_stride
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    var mult = _coprime_multiplier(cache_len)
    for bi in range(batch_size):
        var sentinel = sentinel_positions[bi]
        for i in range(indices_stride):
            if i >= sentinel:
                h_indices[bi * indices_stride + i] = Int32(-1)
            else:
                var t = (i * mult + 1 + bi * 7) % cache_len
                var page_idx = t // PAGE_SIZE
                var tok_in_page = t % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[bi * indices_stride + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    # Production passes the constant indices_stride; the short-topk cases
    # pass less.
    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    var full_topk_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_lengths_list[bi])
        full_topk_host[bi] = Int32(indices_stride)
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)
    var full_topk_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(full_topk_device, full_topk_host)

    # ---- populate cache: real values at the non-sentinel rows, poison
    # everywhere else.
    for bi in range(batch_size):
        for t in range(max_pages_per_batch * PAGE_SIZE):
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + (t // PAGE_SIZE)]
            )
            var page_tok = t % PAGE_SIZE
            var slot = (
                block_id * NUM_LAYERS * PAGE_SIZE + page_tok
            ) * kv_params.num_heads
            var is_real = False
            for i in range(indices_stride):
                if i >= sentinel_positions[bi]:
                    continue
                var real_t = (i * mult + 1 + bi * 7) % cache_len
                if real_t == t:
                    is_real = True
                    break
            for d in range(K_CACHE_DIM):
                var idx = slot * K_CACHE_DIM + d
                if is_real and t < cache_len:
                    blocks_host[idx] = cast_saturating[.float8_e4m3fn](
                        _real_k_pattern(bi, t, d)
                    )
                else:
                    blocks_host[idx] = cast_saturating[.float8_e4m3fn](
                        _real_k_pattern(bi, t, d) * Float32(1000)
                    )

    ctx.enqueue_copy(blocks_device, blocks_host)

    var q_device = ctx.enqueue_create_buffer[dtype](
        batch_size * num_heads * K_CACHE_DIM
    )
    var q_host = ctx.enqueue_create_host_buffer[dtype](
        batch_size * num_heads * K_CACHE_DIM
    )
    for bi in range(batch_size):
        for h in range(num_heads):
            for d in range(K_CACHE_DIM):
                q_host[
                    (bi * num_heads + h) * K_CACHE_DIM + d
                ] = cast_saturating[.float8_e4m3fn](
                    _q_pattern(bi, h, d, num_heads)
                )
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[out_dtype](
        batch_size * num_heads * KV_LATENT_DIM
    )
    ctx.enqueue_memset(out_device, 0)

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

    var q_padded = TileTensor(
        q_device.unsafe_ptr(),
        TileLayout(
            (batch_size, Idx[1], Idx[num_heads], Idx[K_CACHE_DIM]),
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
            (batch_size, Idx[1], Idx[num_heads], Idx[KV_LATENT_DIM]),
            (
                Idx[num_heads * KV_LATENT_DIM],
                Idx[num_heads * KV_LATENT_DIM],
                Idx[KV_LATENT_DIM],
                Idx[1],
            ),
        ),
    )
    var scalar_args_buf_tt = TileTensor(
        scalar_args_dev.unsafe_ptr(), row_major[3]()
    )

    var scratch_buf = ctx.enqueue_create_buffer[dtype](
        batch_size * indices_stride * K_CACHE_DIM
    )
    var k_cache_blocks_base = k_cache.blocks_raw_ptr()
    # Without a lengths array the kernels must not read the pointer, so a
    # dangling one faults if they do.
    var gather_topk = topk_lengths_device.unsafe_ptr().as_unsafe_any_origin()
    comptime if no_topk_lengths:
        gather_topk = MutPointer[Int32, MutAnyOrigin].unsafe_dangling()
    ctx.enqueue_function[
        _amd_sparse_mla_gather_kv_rows[
            type_of(k_cache),
            K_CACHE_DIM,
            GATHER_ROWS_PER_BLOCK,
            has_topk_lengths=not no_topk_lengths,
        ]
    ](
        k_cache,
        k_cache_blocks_base,
        d_indices_device.unsafe_ptr(),
        gather_topk,
        scratch_buf.unsafe_ptr(),
        Int32(indices_stride),
        grid_dim=(
            ceildiv(indices_stride, GATHER_ROWS_PER_BLOCK),
            batch_size,
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
        row_major(batch_size, indices_stride, Idx[1], Idx[K_CACHE_DIM]),
    )

    # Both arms mask with `PaddingMask`; only the lengths differ. RED feeds it
    # the caller's constant `indices_stride` (what production passed), GREEN
    # feeds it the production kernel's derived per-row length.
    var derived_lengths = ctx.enqueue_create_buffer[.uint32](batch_size)
    comptime if use_derived_valid_lengths:
        # The kernel also clears the zero row the in-place reader uses; start
        # it non-zero so the check below sees the clear.
        var zero_row = ctx.enqueue_create_buffer[.uint8](K_CACHE_DIM)
        zero_row.enqueue_fill(255)
        # The scan kernel always reads a lengths array; with none, the cap is
        # the full stride.
        var scan_full_stride = mask_ignores_topk or no_topk_lengths
        ctx.enqueue_function[
            _amd_sparse_mla_row_valid_lengths[
                _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS
            ]
        ](
            d_indices_device.unsafe_ptr().as_unsafe_any_origin(),
            (
                full_topk_device if scan_full_stride else topk_lengths_device
            ).unsafe_ptr(),
            derived_lengths.unsafe_ptr(),
            Int32(indices_stride),
            zero_row.unsafe_ptr(),
            Int32(K_CACHE_DIM),
            grid_dim=(batch_size, 1, 1),
            block_dim=(_AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS, 1, 1),
        )
        var zero_row_host = ctx.enqueue_create_host_buffer[.uint8](K_CACHE_DIM)
        ctx.enqueue_copy(zero_row_host, zero_row)
        ctx.synchronize()
        for i in range(K_CACHE_DIM):
            assert_true(zero_row_host[i] == 0, "zero row not cleared")
        var topk_lengths_tt = TileTensor(
            derived_lengths.unsafe_ptr(), row_major(batch_size)
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
    else:
        var topk_lengths_tt = TileTensor(
            topk_lengths_device.unsafe_ptr().unsafe_bitcast[UInt32](),
            row_major(batch_size),
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

    ctx.synchronize()

    var out_host_buf = ctx.enqueue_create_host_buffer[out_dtype](
        batch_size * num_heads * KV_LATENT_DIM
    )
    ctx.enqueue_copy(out_host_buf, out_device)
    ctx.synchronize()
    for i in range(batch_size * num_heads * KV_LATENT_DIM):
        out_host[i] = out_host_buf[i]

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = full_topk_device
    _ = scalar_args_dev
    _ = scratch_buf


def _oracle_sentinel[
    num_heads: Int,
](
    batch_size: Int,
    cache_len: Int,
    indices_stride: Int,
    sentinel_positions: List[Int],
    topk_lengths_list: List[Int],
    out_ref: MutPointer[Float64, MutAnyOrigin],
) raises:
    """Restricted-softmax f64 oracle over the same valid rows used by
    `_run_sentinel_case`: slots before both the first `-1` and the row's
    `topk_lengths`."""
    comptime scale_f64 = Float64(0.125)
    var mult = _coprime_multiplier(cache_len)

    for bi in range(batch_size):
        var valid = min(sentinel_positions[bi], topk_lengths_list[bi])
        for h in range(num_heads):
            var logits = List[Float64]()
            var toks = List[Int]()
            for i in range(indices_stride):
                if i >= valid:
                    continue
                var t = (i * mult + 1 + bi * 7) % cache_len
                var dot = Float64(0)
                for d in range(K_CACHE_DIM):
                    dot += Float64(
                        cast_saturating[.float8_e4m3fn](
                            _q_pattern(bi, h, d, num_heads)
                        )
                    ) * Float64(
                        cast_saturating[.float8_e4m3fn](
                            _real_k_pattern(bi, t, d)
                        )
                    )
                logits.append(dot * scale_f64)
                toks.append(t)

            assert_true(len(logits) > 0, "sentinel oracle: empty softmax")
            var mx = Float64(-1e300)
            for v in logits:
                mx = max(mx, v)
            var sum_exp = Float64(0)
            for v in logits:
                sum_exp += exp(v - mx)

            var o_base = (bi * num_heads + h) * KV_LATENT_DIM
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


def _check_sentinel[
    num_heads: Int,
    use_derived_valid_lengths: Bool = True,
    mask_ignores_topk: Bool = False,
    no_topk_lengths: Bool = False,
](
    name: String,
    batch_size: Int,
    cache_len: Int,
    indices_stride: Int,
    sentinel_positions: List[Int],
    topk_lengths_list: List[Int],
    ctx: DeviceContext,
    *,
    expect_pass: Bool,
) raises:
    var out_size = batch_size * num_heads * KV_LATENT_DIM

    var got = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.synchronize()
    _run_sentinel_case[
        num_heads=num_heads,
        use_derived_valid_lengths=use_derived_valid_lengths,
        mask_ignores_topk=mask_ignores_topk,
        no_topk_lengths=no_topk_lengths,
    ](
        batch_size,
        cache_len,
        indices_stride,
        sentinel_positions,
        topk_lengths_list,
        got.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )

    var ref_vals = List[Float64](length=out_size, fill=0.0)
    _oracle_sentinel[num_heads=num_heads](
        batch_size,
        cache_len,
        indices_stride,
        sentinel_positions,
        topk_lengths_list,
        ref_vals.unsafe_ptr().as_unsafe_any_origin(),
    )

    var max_abs = Float64(0)
    var max_rel = Float64(0)
    var num_bad = 0
    for bi in range(batch_size):
        var b_max_abs = Float64(0)
        var b_max_rel = Float64(0)
        for h in range(num_heads):
            for d in range(KV_LATENT_DIM):
                var idx = (bi * num_heads + h) * KV_LATENT_DIM + d
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
            "] batch=",
            bi,
            " sentinel=",
            sentinel_positions[bi],
            " max_abs=",
            b_max_abs,
            " max_rel=",
            b_max_rel,
        )

    print(
        "[",
        name,
        "] use_derived_valid_lengths=",
        use_derived_valid_lengths,
        " overall max_abs=",
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

    if expect_pass:
        assert_true(
            num_bad == 0,
            "["
            + name
            + "] derived per-row valid lengths: output outside"
            " atol+rtol*|ref| of the oracle -- mask regression",
        )
        print("  PASSED")
    else:
        assert_true(
            num_bad > 0,
            "["
            + name
            + "] RED CONTROL DID NOT FIRE: length-based PaddingMask"
            " matched the sentinel oracle -- the bug is not reproduced",
        )
        print("  RED CONTROL FIRED AS EXPECTED (PaddingMask diverges)")


def _check_fused_matches_gather[
    num_heads: Int
](
    batch_size: Int,
    cache_len: Int,
    topk_lengths_list: List[Int],
    indices_stride: Int,
    ctx: DeviceContext,
) raises:
    """Reading the selected rows in place must reproduce the gather path bit
    for bit: both fill the same LDS image from the same cache rows (zeros for
    the slots past a batch's top-k length) and run the same decode kernel."""
    var out_size = batch_size * num_heads * KV_LATENT_DIM
    var gathered = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var fused = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.synchronize()
    _run_case[num_heads=num_heads, sparse_mla=True](
        batch_size,
        cache_len,
        topk_lengths_list,
        indices_stride,
        gathered.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )
    _run_case[num_heads=num_heads, sparse_mla=True, fused=True](
        batch_size,
        cache_len,
        topk_lengths_list,
        indices_stride,
        fused.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )
    var mismatches = 0
    for i in range(out_size):
        if bitcast[DType.uint16](gathered[i]) != bitcast[DType.uint16](
            fused[i]
        ):
            mismatches += 1
    print("fused_matches_gather mismatches:", mismatches, "of", out_size)
    assert_true(mismatches == 0, "fused sparse rows differ from the gather")


def _check_closed_form_lengths[
    has_topk_lengths: Bool
](
    cache_lens: List[Int],
    selection_lag: Int,
    indices_stride: Int,
    ctx: DeviceContext,
) raises:
    """The gather's closed-form valid length matches the row's first `-1`.

    Each row is packed the way the indexer's bounded top-k packs it for a
    token that saw `cache_len + 1 - selection_lag` keys: that many valid slots
    (capped at the stride), then `-1`. The kernel must write the same count
    the `-1` scan would find.
    """
    comptime dtype = DType.float8_e4m3fn
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=K_CACHE_DIM, is_mla=True
    )
    var batch_size = len(cache_lens)
    var max_keys = 0
    for b in range(batch_size):
        max_keys = max(max_keys, cache_lens[b] + 1)
    var max_pages_per_batch = ceildiv(max_keys, PAGE_SIZE)
    var total_pages = batch_size * max_pages_per_batch
    var block_elems = total_pages * NUM_LAYERS * PAGE_SIZE * K_CACHE_DIM
    var blocks_device = ctx.enqueue_create_buffer[dtype](block_elems)
    ctx.enqueue_memset(blocks_device, 0)

    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(i)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var expected = List[Int](length=batch_size, fill=0)
    var h_indices = ctx.enqueue_create_host_buffer[.int32](
        batch_size * indices_stride
    )
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(cache_lens[b])
        var selected_for = max(cache_lens[b] + 1 - selection_lag, 0)
        expected[b] = min(indices_stride, selected_for)
        for i in range(indices_stride):
            var value = Int32(-1)
            if i < expected[b]:
                # Distinct, in-range positions in a scrambled order.
                var t = (i * 7 + 3) % selected_for
                value = Int32(
                    (b * max_pages_per_batch + t // PAGE_SIZE) * PAGE_SIZE
                    + t % PAGE_SIZE
                )
            h_indices[b * indices_stride + i] = value
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](
        batch_size * indices_stride
    )
    ctx.enqueue_copy(d_indices_device, h_indices)
    var topk_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for b in range(batch_size):
        topk_host[b] = Int32(indices_stride)
    var topk_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_device, topk_host)

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
        UInt32(max_keys),
    )
    var k_cache = kv_collection.get_key_cache(0)

    var scratch_buf = ctx.enqueue_create_buffer[dtype](
        batch_size * indices_stride * K_CACHE_DIM
    )
    var valid_lengths = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_memset(valid_lengths, 0xFF)
    var topk_ptr = topk_device.unsafe_ptr().as_unsafe_any_origin()
    comptime if not has_topk_lengths:
        topk_ptr = MutPointer[Int32, MutAnyOrigin].unsafe_dangling()
    ctx.enqueue_function[
        _amd_sparse_mla_gather_kv_rows_and_lengths[
            type_of(k_cache),
            K_CACHE_DIM,
            GATHER_ROWS_PER_BLOCK,
            has_topk_lengths=has_topk_lengths,
        ]
    ](
        k_cache,
        k_cache.blocks_raw_ptr(),
        d_indices_device.unsafe_ptr(),
        topk_ptr,
        scratch_buf.unsafe_ptr(),
        valid_lengths.unsafe_ptr(),
        Int32(indices_stride),
        Int32(selection_lag),
        grid_dim=(
            ceildiv(indices_stride, GATHER_ROWS_PER_BLOCK),
            batch_size,
            1,
        ),
        block_dim=(
            K_CACHE_DIM // _AMD_SPARSE_MLA_GATHER_VEC,
            GATHER_ROWS_PER_BLOCK,
            1,
        ),
    )
    var got = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    ctx.enqueue_copy(got, valid_lengths)
    ctx.synchronize()
    for b in range(batch_size):
        var first_negative = indices_stride
        for i in range(indices_stride):
            if h_indices[b * indices_stride + i] < 0:
                first_negative = i
                break
        print(
            "closed_form_lengths lag=",
            selection_lag,
            " cache_len=",
            cache_lens[b],
            " got=",
            got[b],
            " expected=",
            expected[b],
        )
        assert_true(
            Int(got[b]) == expected[b] and expected[b] == first_negative,
            "closed-form valid length differs from the row's first -1",
        )

    _ = blocks_device
    _ = lookup_table_device
    _ = cache_lengths_device
    _ = d_indices_device
    _ = topk_device
    _ = scratch_buf


def main() raises:
    # `PaddingMask` is AMD-only by design (mirrors mla_decode_branch_fp8's
    # own `default_accelerator().is_amd_gpu() and sparse_mla` gate): the SM100 native
    # decode kernel this file's `flare_mla_decoding` calls into on NVIDIA
    # only accepts NullMask/CausalMask/SlidingWindowCausalMask. A bare
    # `comptime if not default_accelerator().is_amd_gpu(): return` followed by more
    # code does NOT block elaboration of that code on the other arch (only
    # a real if/else does), so the whole body must be nested inside the
    # `comptime if` below. The native SM100 sparse path is already covered
    # by test_mla_decode_sparse_kv_fp8.mojo and friends.
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            # GLM-5.3 shape: 8 heads, kv_lora_rank 512, qk_rope 64
            # (depth 576, v 512), page 128, indices_stride 2048, batch >= 2
            # with different topk_lengths, at least one batch below
            # indices_stride.
            var topk_lengths: List[Int] = [2048, 1200, 530]
            _check[num_heads=8, sparse_mla=True](
                "sparse_gather",
                batch_size=3,
                cache_len=4096,
                topk_lengths_list=topk_lengths,
                indices_stride=2048,
                ctx=ctx,
                expect_pass=True,
            )
            _check_fused_matches_gather[num_heads=8](
                batch_size=3,
                cache_len=4096,
                topk_lengths_list=topk_lengths,
                indices_stride=2048,
                ctx=ctx,
            )
            # The production split, `amd_sparse_mla_decode_num_partitions`:
            # 16 splits of 128 rows, so the 1200- and 530-row tokens leave 6
            # and 11 splits wholly past their valid length.
            var sm_count = ctx.get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
            var topk_split = amd_sparse_mla_decode_num_partitions(
                3, 2048, 8, sm_count
            )
            _check[num_heads=8, sparse_mla=True](
                "sparse_gather_topk_split",
                batch_size=3,
                cache_len=4096,
                topk_lengths_list=topk_lengths,
                indices_stride=2048,
                ctx=ctx,
                expect_pass=True,
                num_partitions=topk_split,
            )
            # Contexts of 1, 64 and 130 tokens: all but the first one or two
            # splits are empty, so an empty split must neither produce a NaN
            # nor reach the reduce. 256 splits leave up to 255 empty.
            var short_contexts: List[Int] = [1, 64, 130]
            for num_partitions in [topk_split, 256]:
                _check[num_heads=8, sparse_mla=True](
                    String(t"sparse_gather_short_context_np{num_partitions}"),
                    batch_size=3,
                    cache_len=4096,
                    topk_lengths_list=short_contexts,
                    indices_stride=2048,
                    ctx=ctx,
                    expect_pass=True,
                    num_partitions=num_partitions,
                )
            # RED control: identical (poisoned) cache, dense NullMask
            # decode over the whole cache -- must fail.
            _check[num_heads=8, sparse_mla=False](
                "dense_red_control",
                batch_size=3,
                cache_len=4096,
                topk_lengths_list=topk_lengths,
                indices_stride=2048,
                ctx=ctx,
                expect_pass=False,
            )

            # Sentinel-aware mask regression case. Production passes
            # `topk_lengths == indices_stride`; a `-1` sentinel may land
            # anywhere in the row.
            #
            # `indices_stride` MUST be a multiple of the decode kernel's BN
            # (128): the kernel always loads a full BN-row tile, so a shorter
            # scratch row is read past its end into uninitialised device
            # memory and the comparison stops being reproducible. Production's
            # 2048 is 16*BN, so this constraint is a property of the harness,
            # not of the kernel under test.
            # The RED control runs the same gather and a length-based
            # `PaddingMask`, which cannot hide the middle slot. The GREEN
            # control uses `SparseIndexMask`.
            # Production has ~700 invalid slots of 2048 at a 1300-token
            # prefix. One sentinel moves the softmax denominator by ~0.3%,
            # which is an order of magnitude under ATOL and would leave the
            # red control silent; keep the ratio instead of the count.
            # First invalid column per batch: 40/128 and 60/128 valid, so
            # 69% and 53% of each row is a zero-filled invalid slot. That is
            # production's ratio (~700 invalid of 2048 at a 1300-token
            # prefix). One sentinel would move the denominator ~0.3%, an
            # order of magnitude under ATOL, and the red control would sit
            # silent while the bug was fully present.
            var sentinel_positions = List[Int](length=2, fill=0)
            sentinel_positions[0] = 40
            sentinel_positions[1] = 60
            var full_topk: List[Int] = [128, 128]
            _check_sentinel[num_heads=8, use_derived_valid_lengths=False](
                "sentinel_red_control_padding_mask",
                batch_size=2,
                cache_len=256,
                indices_stride=128,
                sentinel_positions=sentinel_positions,
                topk_lengths_list=full_topk,
                ctx=ctx,
                expect_pass=False,
            )
            _check_sentinel[num_heads=8, use_derived_valid_lengths=True](
                "sentinel_green_control_derived_lengths",
                batch_size=2,
                cache_len=256,
                indices_stride=128,
                sentinel_positions=sentinel_positions,
                topk_lengths_list=full_topk,
                ctx=ctx,
                expect_pass=True,
            )
            # The decode arm when the graph omits `topk_lengths`: no lengths
            # array at all, so both kernels cap at the stride.
            _check_sentinel[num_heads=8, no_topk_lengths=True](
                "sentinel_green_no_topk_lengths",
                batch_size=2,
                cache_len=256,
                indices_stride=128,
                sentinel_positions=sentinel_positions,
                topk_lengths_list=full_topk,
                ctx=ctx,
                expect_pass=True,
            )

            # `topk_lengths` shorter than the first `-1`, with live indices
            # in between: the gather zero-fills 24 of 40 and 36 of 60 slots,
            # so the mask must stop at `topk_lengths`. RED masks at the first
            # `-1` alone and must fail.
            var short_topk: List[Int] = [16, 24]
            _check_sentinel[num_heads=8, use_derived_valid_lengths=True](
                "short_topk_green_capped_lengths",
                batch_size=2,
                cache_len=256,
                indices_stride=128,
                sentinel_positions=sentinel_positions,
                topk_lengths_list=short_topk,
                ctx=ctx,
                expect_pass=True,
            )
            _check_sentinel[
                num_heads=8,
                use_derived_valid_lengths=True,
                mask_ignores_topk=True,
            ](
                "short_topk_red_mask_ignores_topk",
                batch_size=2,
                cache_len=256,
                indices_stride=128,
                sentinel_positions=sentinel_positions,
                topk_lengths_list=short_topk,
                ctx=ctx,
                expect_pass=False,
            )

            # Closed-form lengths: fewer, exactly, and more visible keys
            # than slots, read on time and late.
            var cache_lens: List[Int] = [0, 5, 126, 127, 300]
            for lag in range(3):
                _check_closed_form_lengths[has_topk_lengths=False](
                    cache_lens, lag, 128, ctx
                )
                _check_closed_form_lengths[has_topk_lengths=True](
                    cache_lens, lag, 128, ctx
                )

            _check_s_gt_1_guard(ctx)
            _check_selection_lag_end_to_end(ctx)

            print("test_mla_decode_sparse_kv_fp8_gather: ALL PASSED")
