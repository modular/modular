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
"""Benchmarks the BF16 AMD sparse MLA arm, which reads rows in place.

The arm compacts each token's index row, then decodes directly from the
paged cache. `test_mla_decode_sparse_rows_bf16_amd.mojo` owns correctness;
`--trace` launches the arm once for a profiler.

The per-GPU shape has 32 heads (`-D NUM_HEADS=N` to change), 2051 selected
slots per row, and an 80k-token paged BF16 cache. Index rows contain 512 four-position pools and a three-position
tail. `--hole-period` inserts four invalid slots per selected pool to exercise
compaction across holes.

Buffers are allocated outside the timed region. Before each timed call a 1 GiB
buffer is overwritten to flush caches; `--warm` skips the flush.

Usage: `--case prefill|decode|all`, `--iters N`, `--hole-period P`
(0 = no holes), `--warm`, `--t N`, `--trace`.
"""

from std.collections import List
from std.math import ceildiv
from std.sys import argv, get_defined_int, size_of

from max.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    HostBuffer,
    default_accelerator,
)
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_sparse_mla_bf16_num_partitions,
)
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_ROW_VALID_LENGTHS_THREADS,
    _amd_sparse_mla_bf16_rows,
    _amd_sparse_mla_compact_indices,
)

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
comptime scale = Float32(0.0625)
comptime flush_bytes = 1024 * 1024 * 1024
comptime scalar_arg_count = 3


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


def _sorted(var xs: List[Float64]) -> List[Float64]:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            xs.swap_elements(j - 1, j)
            j -= 1
    return xs^


def _pct(xs: List[Float64], p: Int) -> Float64:
    var last_index = len(xs) - 1
    return xs[min(last_index, last_index * p // 100)]


@fieldwise_init
struct Opts(Copyable, Movable):
    var which_case: String
    var iters: Int
    var hole_period: Int
    var warm: Bool
    var t: Int
    var trace: Bool


struct Bench:
    """Device state shared by every case: the paged cache and flush buffer."""

    var blocks: DeviceBuffer[dtype]
    var lut: DeviceBuffer[.uint32]
    var cache_lengths: DeviceBuffer[.uint32]
    var scalar_args: DeviceBuffer[.int64]
    var flush: DeviceBuffer[.uint8]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime block_rows = num_phys_pages * page_size
        comptime block_elems = block_rows * k_cache_dim
        var bh = ctx.enqueue_create_host_buffer[dtype](block_elems)
        ctx.synchronize()
        var bh_rows = TileTensor(bh, row_major(block_rows, Idx[k_cache_dim]))
        for row in range(block_rows):
            for d in range(k_cache_dim):
                var v = Float32(0)
                if d < kv_latent_dim:
                    v = _unit(UInt64(row * k_cache_dim + d))
                bh_rows[row, d] = v.cast[dtype]()
        self.blocks = ctx.enqueue_create_buffer[dtype](block_elems)
        ctx.enqueue_copy(self.blocks, bh)

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
        self.scalar_args = ctx.enqueue_create_buffer[.int64](scalar_arg_count)
        ctx.enqueue_memset(self.scalar_args, 0)
        self.flush = ctx.enqueue_create_buffer[.uint8](flush_bytes)
        ctx.synchronize()
        _ = bh^
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
    var bufs = ArmBuffers(
        seq_len,
        q^,
        out^,
        indices^,
        topk_lengths^,
        ctx.enqueue_create_buffer[.int32](seq_len * stride),
        ctx.enqueue_create_buffer[.uint32](seq_len),
        ctx.enqueue_create_buffer[.uint8](k_cache_dim * size_of[dtype]()),
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


def stage_arm(ctx: DeviceContext, b: Bench, a: ArmBuffers) raises:
    var kv_collection = _collection(b)
    _amd_sparse_mla_bf16_rows[num_heads, k_cache_dim, kv_latent_dim](
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
        TileTensor(_mp(b.scalar_args), row_major[scalar_arg_count]()),
    )


def _flush(ctx: DeviceContext, b: Bench, warm: Bool) raises:
    if not warm:
        ctx.enqueue_memset(b.flush, UInt8(1))


def _time(
    ctx: DeviceContext,
    b: Bench,
    a: ArmBuffers,
    compact_only: Bool,
    iters: Int,
    warm: Bool,
) raises -> List[Float64]:
    @always_inline
    def body(c: DeviceContext) raises {imm}:
        if compact_only:
            stage_compact(c, a)
        else:
            stage_arm(c, b, a)

    var us = List[Float64]()
    for i in range(iters + 2):
        _flush(ctx, b, warm)
        var ns = ctx.execution_time(body, 1)
        if i >= 2:
            us.append(Float64(ns) / 1000.0)
    return _sorted(us^)


def bench_case(ctx: DeviceContext, b: Bench, seq_len: Int, opts: Opts) raises:
    var q_host = ctx.enqueue_create_host_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var idx_host = ctx.enqueue_create_host_buffer[.int32](seq_len * stride)
    ctx.synchronize()
    var a = _make_case(ctx, seq_len, opts.hole_period, q_host, idx_host)
    stage_arm(ctx, b, a)
    ctx.synchronize()
    if opts.trace:
        print("# traced arm T=", seq_len, sep="")
        return
    var iters = opts.iters
    if iters == 0:
        comptime prefill_min_tokens = 4096
        comptime prefill_iters = 20
        comptime decode_iters = 200
        iters = prefill_iters if seq_len >= prefill_min_tokens else decode_iters
    var np = amd_sparse_mla_bf16_num_partitions(
        seq_len,
        stride,
        num_heads,
        ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
    )
    for compact_only in [False, True]:
        var us = _time(ctx, b, a, compact_only, iters, opts.warm)
        print(
            t"timing,T={seq_len},{"compact" if compact_only else "arm_total"},np={np},iters={iters},l2={"warm" if opts.warm else "flushed"},median_us={_pct(us, 50)},p10_us={_pct(us, 10)},p90_us={_pct(us, 90)}"
        )
    _ = a^
    _ = q_host^
    _ = idx_host^


def main() raises:
    var opts = Opts("all", 0, 32, False, 0, False)
    var args = argv()
    var i = 1
    while i < len(args):
        if args[i] == "--case":
            opts.which_case = String(args[i + 1])
            i += 1
        elif args[i] == "--iters":
            opts.iters = Int(args[i + 1])
            i += 1
        elif args[i] == "--hole-period":
            opts.hole_period = Int(args[i + 1])
            i += 1
        elif args[i] == "--t":
            opts.t = Int(args[i + 1])
            i += 1
        elif args[i] == "--warm":
            opts.warm = True
        elif args[i] == "--trace":
            opts.trace = True
        else:
            raise Error(String(t"unknown argument {args[i]}"))
        i += 1

    if (
        opts.which_case != "prefill"
        and opts.which_case != "decode"
        and opts.which_case != "all"
    ):
        raise Error(String(t"unknown case {opts.which_case}"))

    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            var b = Bench(ctx)
            var num_cus = ctx.device().get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
            print(
                t"# CUs={num_cus} heads={num_heads} stride={stride} ctx_len={ctx_len} hole_period={opts.hole_period}"
            )
            var ts = List[Int]()
            if opts.t != 0:
                ts.append(opts.t)
            else:
                if opts.which_case == "prefill" or opts.which_case == "all":
                    ts.append(16384)
                if opts.which_case == "decode" or opts.which_case == "all":
                    for t in [1, 8, 32, 64, 128]:
                        ts.append(t)
            for t in ts:
                bench_case(ctx, b, t, opts)
            _ = b^
