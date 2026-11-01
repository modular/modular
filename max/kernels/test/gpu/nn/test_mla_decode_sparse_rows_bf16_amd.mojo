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
"""Check BF16 AMD sparse MLA against a dense oracle and the gather path.

The production arm compacts physical row indices, then decodes them in place
with block-M 16 partitions. The gather path here is a parity reference; its
chunk boundary is exercised separately from the unchunked production path.

The test uses a paged BF16 cache with 512 latent values and 64 padded values
per row. Index rows select 512 pools of four positions plus a three-position
tail. Interior invalid slots exercise compaction without truncating the row.

`-D NUM_HEADS` sets the query heads per device (default 32, GLM-5.3-Flash at
TP2). At 16 or fewer heads the split heuristic takes its other branch.
"""

from std.collections import List
from std.math import ceildiv, exp, nan
from std.sys import get_defined_int, size_of
from std.testing import assert_equal, assert_true

from max.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    HostBuffer,
    default_accelerator,
)
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_sparse_mla_bf16_num_partitions,
    amd_sparse_mla_decode_num_partitions,
)
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_GATHER_VEC,
    _AMD_SPARSE_MLA_MAX_LAUNCH_TOKENS,
    _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS,
    _amd_sparse_mla_bf16_rows,
    _amd_sparse_mla_compact_indices,
    _amd_sparse_mla_gather_kv_rows,
)
from nn.attention.mha_mask import PaddingMask
from nn.attention.mha_utils import MHAConfig

comptime dtype = DType.bfloat16
comptime num_heads = get_defined_int["NUM_HEADS", 32]()
comptime k_cache_dim = 576
comptime kv_latent_dim = 512
comptime page_size = 128
comptime kpool = 4
comptime pools_selected = 512
comptime stride = pools_selected * kpool + kpool - 1
comptime ctx_len = 80000
comptime num_pages = ceildiv(ctx_len, page_size)
# Physical pages: more than the context needs, and the page table permutes
# them, so physical slot ids are not token positions.
comptime num_phys_pages = 640
comptime page_perm_mul = 389
comptime gather_rows_per_block = 7
# Caps the gather reference's scratch: 256 tokens x 2051 BF16 rows is ~600 MB.
comptime gather_chunk_tokens = 256
comptime scale = Float32(0.0625)


def _mix(x: UInt64) -> UInt64:
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _unit(x: UInt64) -> Float32:
    """Maps a hash to [-1, 1)."""
    return Float32(Int(_mix(x) >> 40)) / Float32(1 << 23) - 1.0


def _phys_page(logical: Int) -> Int:
    return (logical * page_perm_mul + 17) % num_phys_pages


def _phys_slot(pos: Int) -> Int32:
    return Int32(_phys_page(pos // page_size) * page_size + pos % page_size)


def _mp[d: DType](buf: DeviceBuffer[d]) -> MutPointer[Scalar[d], MutAnyOrigin]:
    """Device pointer of a buffer the caller holds by shared reference."""
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


struct Bench:
    """Holds the paged cache and scalar arguments shared by the test cases."""

    var blocks_host: HostBuffer[dtype]
    var blocks: DeviceBuffer[dtype]
    var lut: DeviceBuffer[.uint32]
    var cache_lengths: DeviceBuffer[.uint32]
    var scalar_args: DeviceBuffer[.int64]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime block_elems = num_phys_pages * page_size * k_cache_dim
        var bh = ctx.enqueue_create_host_buffer[dtype](block_elems)
        ctx.synchronize()
        var block_rows = TileTensor(
            bh, row_major[num_phys_pages * page_size, k_cache_dim]()
        )
        comptime assert block_rows.flat_rank == 2
        for row in range(num_phys_pages * page_size):
            for d in range(k_cache_dim):
                var v = Float32(0)
                if d < kv_latent_dim:
                    v = _unit(UInt64(row * k_cache_dim + d))
                block_rows[row, d] = v.cast[dtype]()
        self.blocks = ctx.enqueue_create_buffer[dtype](block_elems)
        ctx.enqueue_copy(self.blocks, bh)
        self.blocks_host = bh^

        var lut_h = ctx.enqueue_create_host_buffer[.uint32](num_pages)
        ctx.synchronize()
        for p in range(num_pages):
            lut_h[p] = UInt32(_phys_page(p))
        self.lut = ctx.enqueue_create_buffer[.uint32](num_pages)
        ctx.enqueue_copy(self.lut, lut_h)
        var cl_h = ctx.enqueue_create_host_buffer[.uint32](1)
        ctx.synchronize()
        cl_h[0] = UInt32(ctx_len)
        self.cache_lengths = ctx.enqueue_create_buffer[.uint32](1)
        ctx.enqueue_copy(self.cache_lengths, cl_h)
        self.scalar_args = ctx.enqueue_create_buffer[.int64](3)
        ctx.enqueue_memset(self.scalar_args, 0)
        ctx.synchronize()
        _ = lut_h^
        _ = cl_h^


def _fill_indices(
    host: HostBuffer[.int32], seq_len: Int, hole_period: Int
) -> Int:
    """Writes k-pooled physical index rows; returns the total valid slots."""
    var total_valid = 0
    for t in range(seq_len):
        var pos = ctx_len - seq_len + t
        # Pools strictly before the query's own (partial) pool.
        var npools = pos // kpool
        var step = npools // pools_selected
        var base = t * stride
        for j in range(pools_selected):
            var h = _mix(UInt64(t) * 1000003 + UInt64(j))
            var pool = j * step + Int(h % UInt64(step))
            var hole = (
                hole_period > 0 and Int((h >> 32) % UInt64(hole_period)) == 0
            )
            for k in range(kpool):
                var slot = Int32(-1) if hole else _phys_slot(pool * kpool + k)
                host[base + j * kpool + k] = slot
                if not hole:
                    total_valid += 1
        for k in range(kpool - 1):
            host[base + pools_selected * kpool + k] = _phys_slot(
                pos - (kpool - 2) + k
            )
            total_valid += 1
    return total_valid


@fieldwise_init
struct ArmBuffers(Movable):
    var seq_len: Int
    var q: DeviceBuffer[dtype]
    var out: DeviceBuffer[dtype]
    var indices: DeviceBuffer[.int32]
    var topk_lengths: DeviceBuffer[.int32]
    var compact: DeviceBuffer[.int32]
    var valid_lengths: DeviceBuffer[.uint32]
    var zero_row: DeviceBuffer[.uint8]
    var scratch: DeviceBuffer[dtype]


def _make_case(
    ctx: DeviceContext,
    seq_len: Int,
    hole_period: Int,
    q_host_out: HostBuffer[dtype],
    idx_host_out: HostBuffer[.int32],
) raises -> ArmBuffers:
    var q_size = seq_len * num_heads * k_cache_dim
    for i in range(q_size):
        var v = Float32(0)
        if i % k_cache_dim < kv_latent_dim:
            v = _unit(UInt64(0xABCDEF) * 7919 + UInt64(i))
        q_host_out[i] = v.cast[dtype]()
    var valid = _fill_indices(idx_host_out, seq_len, hole_period)
    print(
        t"# case"
        t" T={seq_len} valid_slots_per_row={Float64(valid) / Float64(seq_len)} of"
        t" {stride}"
    )
    var q = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q, q_host_out)
    var indices = ctx.enqueue_create_buffer[.int32](seq_len * stride)
    ctx.enqueue_copy(indices, idx_host_out)
    var tl_h = ctx.enqueue_create_host_buffer[.int32](seq_len)
    ctx.synchronize()
    for t in range(seq_len):
        tl_h[t] = Int32(stride)
    var topk_lengths = ctx.enqueue_create_buffer[.int32](seq_len)
    ctx.enqueue_copy(topk_lengths, tl_h)
    var out = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    ctx.enqueue_memset(out, 0)
    var chunk_tokens = min(seq_len, gather_chunk_tokens)
    var bufs = ArmBuffers(
        seq_len,
        q^,
        out^,
        indices^,
        topk_lengths^,
        ctx.enqueue_create_buffer[.int32](seq_len * stride),
        ctx.enqueue_create_buffer[.uint32](seq_len),
        ctx.enqueue_create_buffer[.uint8](k_cache_dim * size_of[dtype]()),
        ctx.enqueue_create_buffer[dtype](chunk_tokens * stride * k_cache_dim),
    )
    ctx.synchronize()
    _ = tl_h^
    return bufs^


comptime Collection = PagedKVCacheCollection[
    dtype,
    KVCacheStaticParams(num_heads=1, head_size=k_cache_dim, is_mla=True),
    page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
]


def _collection(b: Bench) -> Collection:
    var blocks_tt = TileTensor(
        _mp(b.blocks),
        row_major(
            Int64(num_phys_pages),
            Idx[1],
            Int64(1),
            Idx[page_size],
            Idx[1],
            Idx[k_cache_dim],
        ),
    )
    var cache_lengths_tt = TileTensor(_mp(b.cache_lengths), row_major(Int64(1)))
    var lut_tt = TileTensor(_mp(b.lut), row_major(Int64(1), Int64(num_pages)))
    return Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lut_tt.as_imm().as_unsafe_any_origin(),
        UInt32(0),
        UInt32(ctx_len),
    )


def stage_compact(ctx: DeviceContext, a: ArmBuffers) raises:
    ctx.enqueue_function[
        _amd_sparse_mla_compact_indices[
            _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS
        ]
    ](
        _mp(a.indices),
        _mp(a.topk_lengths),
        _mp(a.compact),
        _mp(a.valid_lengths),
        Int32(stride),
        _mp(a.zero_row),
        Int32(k_cache_dim * size_of[dtype]()),
        grid_dim=(a.seq_len, 1, 1),
        block_dim=(_AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS, 1, 1),
    )


def stage_chunks(
    ctx: DeviceContext,
    b: Bench,
    a: ArmBuffers,
    do_gather: Bool,
    do_decode: Bool,
) raises:
    var kv_collection = _collection(b)
    var k_cache = kv_collection.get_key_cache(0)
    var k_cache_blocks_base = k_cache.blocks_raw_ptr()
    var seq_len = a.seq_len
    var indices_stride = stride
    var scalar_args_tt = TileTensor(_mp(b.scalar_args), row_major[3]())
    var chunk_tokens = min(seq_len, gather_chunk_tokens)
    for token0 in range(0, seq_len, chunk_tokens):
        var tokens = min(chunk_tokens, seq_len - token0)
        if do_gather:
            ctx.enqueue_function[
                _amd_sparse_mla_gather_kv_rows[
                    type_of(k_cache),
                    k_cache_dim,
                    gather_rows_per_block,
                ]
            ](
                k_cache,
                k_cache_blocks_base,
                _mp(a.compact).unsafe_offset(token0 * indices_stride),
                _mp(a.topk_lengths).unsafe_offset(token0),
                _mp(a.scratch),
                Int32(indices_stride),
                grid_dim=(
                    ceildiv(indices_stride, gather_rows_per_block),
                    tokens,
                    1,
                ),
                block_dim=(
                    k_cache_dim // _AMD_SPARSE_MLA_GATHER_VEC,
                    gather_rows_per_block,
                    1,
                ),
            )
        if not do_decode:
            continue
        var k_scratch = TileTensor(
            _mp(a.scratch),
            row_major(tokens, indices_stride, Idx[1], Idx[k_cache_dim]),
        )
        var q_tokens = TileTensor(
            _mp(a.q).unsafe_offset(token0 * num_heads * k_cache_dim),
            TileLayout(
                (tokens, Idx[1], Idx[num_heads], Idx[k_cache_dim]),
                (
                    Idx[num_heads * k_cache_dim],
                    Idx[num_heads * k_cache_dim],
                    Idx[k_cache_dim],
                    Idx[1],
                ),
            ),
        )
        var out_tokens = TileTensor(
            _mp(a.out).unsafe_offset(token0 * num_heads * kv_latent_dim),
            TileLayout(
                (tokens, Idx[1], Idx[num_heads], Idx[kv_latent_dim]),
                (
                    Idx[num_heads * kv_latent_dim],
                    Idx[num_heads * kv_latent_dim],
                    Idx[kv_latent_dim],
                    Idx[1],
                ),
            ),
        )
        var lengths_tt = TileTensor(
            _mp(a.valid_lengths).unsafe_offset(token0),
            row_major(tokens),
        )
        flare_mla_decoding[
            config=MHAConfig[dtype](num_heads, k_cache_dim),
            decoding_warp_split_k=False,
        ](
            out_tokens,
            q_tokens,
            k_scratch,
            PaddingMask(lengths_tt),
            scale,
            ctx,
            scalar_args_tt,
            num_partitions=amd_sparse_mla_decode_num_partitions(
                tokens,
                indices_stride,
                num_heads,
                ctx.device().get_attribute(
                    DeviceAttribute.MULTIPROCESSOR_COUNT
                ),
            ),
        )


def stage_inplace[
    max_launch_tokens: Int = _AMD_SPARSE_MLA_MAX_LAUNCH_TOKENS
](ctx: DeviceContext, b: Bench, a: ArmBuffers) raises:
    var kv_collection = _collection(b)
    _amd_sparse_mla_bf16_rows[
        num_heads,
        k_cache_dim,
        kv_latent_dim,
        max_launch_tokens=max_launch_tokens,
    ](
        TileTensor(
            _mp(a.out), row_major(a.seq_len, Idx[num_heads], Idx[kv_latent_dim])
        ),
        TileTensor(
            _mp(a.q), row_major(a.seq_len, Idx[num_heads], Idx[k_cache_dim])
        ),
        kv_collection.get_key_cache(0),
        _mp(a.indices),
        _mp(a.topk_lengths),
        stride,
        _mp(a.compact),
        _mp(a.valid_lengths),
        _mp(a.zero_row).unsafe_bitcast[Scalar[dtype]](),
        scale,
        ctx,
        TileTensor(_mp(b.scalar_args), row_major[3]()),
    )


def run_arm(ctx: DeviceContext, b: Bench, a: ArmBuffers) raises:
    stage_compact(ctx, a)
    stage_chunks(ctx, b, a, True, True)


def _reference_row(
    b: Bench,
    q_host: HostBuffer[dtype],
    idx_host: HostBuffer[.int32],
    t: Int,
    h: Int,
    truncate_at_first_hole: Bool,
    mut out: List[Float64],
):
    """Dense fp32-accumulated attention of one (token, head) over its valid
    selected rows, read straight from the paged cache (no compaction)."""
    var q_tensor = TileTensor(
        q_host,
        row_major(
            len(q_host) // (num_heads * k_cache_dim), num_heads, k_cache_dim
        ),
    )
    var k_tensor = TileTensor(
        b.blocks_host, row_major[num_phys_pages * page_size, k_cache_dim]()
    )
    var index_tensor = TileTensor(
        idx_host, row_major(len(idx_host) // stride, stride)
    )
    comptime assert q_tensor.flat_rank == 3
    comptime assert k_tensor.flat_rank == 2
    comptime assert index_tensor.flat_rank == 2
    var scores = List[Float32]()
    var rows = List[Int]()
    var q = List[Float32](capacity=k_cache_dim)
    for d in range(k_cache_dim):
        q.append(q_tensor[t, h, d].cast[.float32]())
    var m = Float32(-1e30)
    for s in range(stride):
        var e = Int(index_tensor[t, s])
        if e < 0:
            if truncate_at_first_hole:
                break
            continue
        # A physical slot ID addresses its cache row directly.
        var acc = Float32(0)
        for d in range(k_cache_dim):
            acc += q[d] * k_tensor[e, d].cast[.float32]()
        acc *= scale
        scores.append(acc)
        rows.append(e)
        m = max(m, acc)
    var denom = Float32(0)
    for i in range(len(scores)):
        scores[i] = exp(scores[i] - m)
        denom += scores[i]
    for d in range(kv_latent_dim):
        out[d] = 0
    for i in range(len(scores)):
        var p = scores[i] / denom
        for d in range(kv_latent_dim):
            out[d] += Float64(p * k_tensor[rows[i], d].cast[.float32]())


def check_correctness[
    max_launch_tokens: Int = _AMD_SPARSE_MLA_MAX_LAUNCH_TOKENS
](
    ctx: DeviceContext,
    b: Bench,
    seq_len: Int,
    hole_period: Int,
    tokens: List[Int],
    label: String,
    expect_np: Int = 0,
) raises:
    var np = amd_sparse_mla_bf16_num_partitions(
        seq_len,
        stride,
        num_heads,
        ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
    )
    if expect_np > 0:
        assert_equal(np, expect_np, "case does not reach the intended split")
    var q_host = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var idx_host = ctx.enqueue_create_host_buffer[.int32](seq_len * stride)
    ctx.synchronize()
    var a = _make_case(ctx, seq_len, hole_period, q_host, idx_host)
    ctx.enqueue_memset(a.zero_row, UInt8(255))
    run_arm(ctx, b, a)
    var gathered = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var zero_row = ctx.enqueue_create_host_buffer[.uint8](
        k_cache_dim * size_of[dtype]()
    )
    var valid = ctx.enqueue_create_host_buffer[.uint32](seq_len)
    ctx.enqueue_copy(gathered, a.out)
    ctx.enqueue_copy(zero_row, a.zero_row)
    ctx.enqueue_copy(valid, a.valid_lengths)
    ctx.synchronize()
    for i in range(k_cache_dim * size_of[dtype]()):
        assert_true(zero_row[i] == 0, "compaction did not clear zero row")
    # One extra key per row moves the output by about 1/stride, which the
    # value oracle cannot resolve, so check each count exactly. Every
    # topk_length here is `stride`.
    for t in range(seq_len):
        var want = 0
        for s in range(stride):
            if idx_host[t * stride + s] >= 0:
                want += 1
        assert_equal(
            Int(valid[t]), want, "compact length differs from valid-index count"
        )
    ctx.enqueue_memset(a.out, nan[dtype]())
    stage_inplace[max_launch_tokens](ctx, b, a)
    var out_host = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    ctx.enqueue_copy(out_host, a.out)
    ctx.synchronize()
    var ref_row = List[Float64](length=kv_latent_dim, fill=0)
    var max_abs = Float64(0)
    var max_ref = Float64(0)
    var max_rel = Float64(0)
    var neg_max_abs = Float64(0)
    var parity_max_abs = Float64(0)
    var bad = 0
    var out_tensor = TileTensor(
        out_host, row_major(seq_len, num_heads, kv_latent_dim)
    )
    var gathered_tensor = TileTensor(
        gathered, row_major(seq_len, num_heads, kv_latent_dim)
    )
    comptime assert out_tensor.flat_rank == 3
    comptime assert gathered_tensor.flat_rank == 3
    for t in tokens:
        for h in range(num_heads):
            _reference_row(b, q_host, idx_host, t, h, False, ref_row)
            var row_ref_max = Float64(0)
            for d in range(kv_latent_dim):
                row_ref_max = max(row_ref_max, abs(ref_row[d]))
            for d in range(kv_latent_dim):
                var got = Float64(out_tensor[t, h, d].cast[.float64]())
                if got != got:
                    bad += 1
                var err = abs(got - ref_row[d])
                var gather_value = Float64(
                    gathered_tensor[t, h, d].cast[.float64]()
                )
                parity_max_abs = max(parity_max_abs, abs(got - gather_value))
                max_abs = max(max_abs, err)
                max_ref = max(max_ref, abs(ref_row[d]))
                max_rel = max(max_rel, err / max(row_ref_max, 1e-6))
            # Negative control: truncating each row at its first hole (the
            # suffix-only length rule) must disagree when holes exist.
            if hole_period > 0 and h == 0:
                _reference_row(b, q_host, idx_host, t, h, True, ref_row)
                for d in range(kv_latent_dim):
                    var got = Float64(out_tensor[t, h, d].cast[.float64]())
                    neg_max_abs = max(neg_max_abs, abs(got - ref_row[d]))
    print(
        t"correctness,{label},T={seq_len},tokens_checked={len(tokens)},heads={num_heads},np={np},max_abs_err={max_abs},max_abs_ref={max_ref},max_rel_err_vs_row_max={max_rel},nan={bad},parity_max_abs={parity_max_abs},negctl_truncate_max_abs={neg_max_abs}"
    )
    assert_true(bad == 0, "nonfinite sparse MLA output")
    assert_true(max_abs < 2e-3, "sparse MLA differs from valid-row oracle")
    assert_true(parity_max_abs < 0.01, "in-place MLA differs from gather")
    assert_true(neg_max_abs > 0.1, "hole negative control is insensitive")
    _ = a^
    _ = q_host^
    _ = idx_host^
    _ = out_host^
    _ = gathered^
    _ = zero_row^
    _ = valid^


def check_empty(
    ctx: DeviceContext, b: Bench, seq_len: Int, all_invalid: Bool
) raises:
    var np = amd_sparse_mla_bf16_num_partitions(
        seq_len,
        stride,
        num_heads,
        ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
    )
    if seq_len == 257:
        assert_true(np == 1, "empty-row unsplit case must use one partition")
    else:
        assert_true(np > 1, "empty-row split case must use a reducer")
    var q_host = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var idx_host = ctx.enqueue_create_host_buffer[.int32](seq_len * stride)
    ctx.synchronize()
    var a = _make_case(ctx, seq_len, 32, q_host, idx_host)
    if all_invalid:
        for i in range(seq_len * stride):
            idx_host[i] = -1
        ctx.enqueue_copy(a.indices, idx_host)
    else:
        var lengths = ctx.enqueue_create_host_buffer[.int32](seq_len)
        for i in range(seq_len):
            lengths[i] = 0
        ctx.enqueue_copy(a.topk_lengths, lengths)
        _ = lengths^
    ctx.enqueue_memset(a.out, nan[dtype]())
    stage_compact(ctx, a)
    stage_inplace(ctx, b, a)
    var out = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var valid = ctx.enqueue_create_host_buffer[.uint32](seq_len)
    ctx.enqueue_copy(out, a.out)
    ctx.enqueue_copy(valid, a.valid_lengths)
    ctx.synchronize()
    for i in range(seq_len):
        assert_true(valid[i] == 0, "empty row has a nonzero compact length")
    for i in range(seq_len * num_heads * kv_latent_dim):
        var value = out[i].cast[.float32]()
        assert_true(value == 0, "empty row output must be finite zero")
    _ = a^
    _ = q_host^
    _ = idx_host^
    _ = out^
    _ = valid^


def check_attention_replay(ctx: DeviceContext, b: Bench, seq_len: Int) raises:
    var output_size = seq_len * num_heads * kv_latent_dim
    var q_host = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var idx_host = ctx.enqueue_create_host_buffer[.int32](seq_len * stride)
    ctx.synchronize()
    var a = _make_case(ctx, seq_len, 32, q_host, idx_host)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            stage_compact(rec, a)
            stage_inplace(rec, b, a)

    var graph = DeviceGraph.create(ctx, build)
    var topk_host = ctx.enqueue_create_host_buffer[.int32](seq_len)
    var graph_out = ctx.enqueue_create_host_buffer[dtype](output_size)
    var eager_out = ctx.enqueue_create_host_buffer[dtype](output_size)
    for replay in range(3):
        if replay == 1:
            for i in range(seq_len * stride):
                idx_host[i] = -1
        elif replay == 2:
            _ = _fill_indices(idx_host, seq_len, 32)
        for i in range(seq_len):
            topk_host[i] = Int32(0 if replay == 2 else stride)
        ctx.enqueue_copy(a.indices, idx_host)
        ctx.enqueue_copy(a.topk_lengths, topk_host)
        a.zero_row.enqueue_fill(UInt8(255))
        graph.replay()
        ctx.enqueue_copy(graph_out, a.out)
        ctx.synchronize()
        stage_compact(ctx, a)
        stage_inplace(ctx, b, a)
        ctx.enqueue_copy(eager_out, a.out)
        ctx.synchronize()
        for i in range(output_size):
            assert_true(
                graph_out[i] == eager_out[i],
                "captured sparse MLA differs from eager replay input",
            )
            if replay > 0:
                assert_true(graph_out[i] == 0, "empty replay must return zero")
    _ = a^
    _ = q_host^
    _ = idx_host^
    _ = topk_host^
    _ = graph_out^
    _ = eager_out^


def main() raises:
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            # The arm puts one query token per grid.z slice in each launch.
            assert_true(
                _AMD_SPARSE_MLA_MAX_LAUNCH_TOKENS
                <= ctx.device().get_attribute(DeviceAttribute.MAX_GRID_DIM_Z),
                "the arm's tokens per launch exceed the device grid.z limit",
            )
            var b = Bench(ctx)
            check_correctness(ctx, b, 1, 32, [0], "decode")
            var small: List[Int] = [0, 1, 7, 100, 254, 255, 256]
            check_correctness(ctx, b, 257, 32, small, "chunk_boundary")
            check_correctness(ctx, b, 257, 2, small, "mid_row_holes")
            # Three launches of 100, 100 and 57 tokens, as the arm splits a
            # batch past the device's `grid.z` limit.
            check_correctness[max_launch_tokens=100](
                ctx, b, 257, 2, small, "sliced_launch"
            )
            # Batched split-K, the regime of the batch 8 and 32 decode steps;
            # at 32 heads each entry also spans two 16-head blocks. The
            # expected splits are MI355X's (256 CUs).
            comptime assert (
                num_heads == 32 or num_heads == 16
            ), "expected splits are pinned for 16 and 32 heads"
            comptime split_np8 = 16 if num_heads == 32 else 32
            comptime split_np32 = 4 if num_heads == 32 else 16
            check_correctness(
                ctx, b, 8, 2, [0, 3, 7], "split_batch8", split_np8
            )
            check_correctness(
                ctx, b, 32, 2, [0, 13, 31], "split_batch32", split_np32
            )
            var big: List[Int] = [
                0,
                255,
                256,
                4097,
                8191,
                8192,
                12345,
                16127,
                16128,
                16383,
            ]
            check_correctness(ctx, b, 16384, 32, big, "prefill_sample")
            check_empty(ctx, b, 1, True)
            check_empty(ctx, b, 1, False)
            check_empty(ctx, b, 257, True)
            check_empty(ctx, b, 257, False)
            check_attention_replay(ctx, b, 1)
            check_attention_replay(ctx, b, 257)
            _ = b^
