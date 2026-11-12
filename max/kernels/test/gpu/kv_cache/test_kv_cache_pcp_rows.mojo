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
"""Prefill context parallelism's prefix round trip, against a hand-built image.

Every rank of a `dcp`-way group holds the prefix positions it owns at their
local slots, as decode context parallelism stores them. Each rank packs its
rows with `kv_cache_pack_owned_rows`, the packed rows are concatenated in rank
order as the all-gather does, and `kv_cache_unpack_prefix_rows` writes them
into a whole-context cache with its own lookup table. The packed rows and the
whole-context cache are checked byte for byte: every prefix position's row
and scales at the right place, and the initial bytes everywhere else,
including the packed rows' other byte columns and the rows past each rank's
share. The packed rows start out as a second marker, not the caches' poison,
so a pack that read a slot past a rank's share, or an unpack that wrote a
position past a request's cache length, changes a checked byte. Covers a
bf16 latent row with no scales, an FP8 indexer key with an FP32 scale, and a
packed NVFP4 latent with its E4M3 scales.
"""

from std.collections import Set
from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext
from kv_cache.context_parallel import ContextParallelLayout
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import RowMajorLayout, TileTensor, row_major
from nn.kv_cache_gather import (
    kv_cache_pack_owned_rows,
    kv_cache_unpack_prefix_rows,
)
from std.testing import assert_equal

comptime page_size = 128
comptime num_layers = 2
comptime layer_idx = 1
comptime poison = UInt8(0x7F)
# The packed rows' initial byte: distinct from the caches' poison, so the
# slots the pack must skip read differently from the cache rows it must not
# copy, and the unpack's guard keeps them out of the whole-context cache.
comptime unwritten = UInt8(0xA5)
# A packed row lands past one leading word and leaves two trailing ones, as
# a second cache's row would sit beside it.
comptime col_offset = 4
comptime trailing = 8
comptime extra_rows = 3


def blocks_tt[
    dt: DType
](mut dev: DeviceBuffer[dt], num_blocks: Int, width: Int) -> TileTensor[
    dt, RowMajorLayout[Int, Int, Int, Int, Int, Int], origin_of(dev)
]:
    return TileTensor(
        dev, row_major(num_blocks, 1, num_layers, page_size, 1, width)
    )


def row_index(block: Int, row: Int) -> Int:
    """Index of cache row `(block, layer_idx, row)` among all rows."""
    return (block * num_layers + layer_idx) * page_size + row


struct Prefixes(Movable):
    """Each request's cached prefix and two page tables over one pool: the
    DCP one (global positions' local slots) and the whole-context one."""

    var cache_lens: List[Int]
    var lut_cols: Int
    var num_blocks: Int
    var dcp_lut: List[Int]
    var whole_lut: List[Int]

    def __init__(out self):
        seed(7)
        # An empty prefix, one token, page edges, and runs past several pages.
        self.cache_lens = [0, 1, 127, 128, 300, 1025, 7]
        var max_len = 0
        for n in self.cache_lens:
            max_len = max(max_len, n)
        self.lut_cols = ceildiv(max_len, page_size) + 1
        var bsz = len(self.cache_lens)
        self.num_blocks = 2 * bsz * self.lut_cols + 4
        self.dcp_lut = List[Int](
            length=bsz * self.lut_cols, fill=self.num_blocks
        )
        self.whole_lut = List[Int](
            length=bsz * self.lut_cols, fill=self.num_blocks
        )
        var used = Set[Int]()
        for i in range(bsz):
            for c in range(self.lut_cols):
                for which in range(2):
                    var b = Int(random_ui64(0, UInt64(self.num_blocks - 1)))
                    while b in used:
                        b = Int(random_ui64(0, UInt64(self.num_blocks - 1)))
                    used.add(b)
                    if which == 0:
                        self.dcp_lut[i * self.lut_cols + c] = b
                    else:
                        self.whole_lut[i * self.lut_cols + c] = b

    def batch_size(self) -> Int:
        return len(self.cache_lens)

    def max_cache(self) -> Int:
        var m = 0
        for n in self.cache_lens:
            m = max(m, n)
        return m

    def dcp_block(self, bs: Int, slot: Int) -> Int:
        return self.dcp_lut[bs * self.lut_cols + slot // page_size]

    def whole_block(self, bs: Int, pos: Int) -> Int:
        return self.whole_lut[bs * self.lut_cols + pos // page_size]


def row_byte(bs: Int, pos: Int, i: Int) -> UInt8:
    """Byte `i` of request `bs`'s row at position `pos`: never the poison."""
    var v = UInt8((bs * 131 + pos * 17 + i * 7 + 3) % 251)
    return v + 1 if v >= poison else v


def scale_byte(bs: Int, pos: Int, i: Int) -> UInt8:
    return row_byte(bs, pos + 7919, i + 1)


def device_bytes[
    dt: DType
](ctx: DeviceContext, bytes: List[UInt8]) raises -> DeviceBuffer[dt]:
    var n = len(bytes) // size_of[Scalar[dt]]()
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    var p = host.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(len(bytes)):
        p[i] = bytes[i]
    var dev = ctx.enqueue_create_buffer[dt](n)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    return dev


def host_bytes[
    dt: DType
](ctx: DeviceContext, dev: DeviceBuffer[dt], n: Int) raises -> List[UInt8]:
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    var p = host.unsafe_ptr().unsafe_bitcast[UInt8]()
    var out = List[UInt8](capacity=n * size_of[Scalar[dt]]())
    for i in range(n * size_of[Scalar[dt]]()):
        out.append(p[i])
    return out^


def u32_buffer(
    ctx: DeviceContext, values: List[Int]
) raises -> DeviceBuffer[DType.uint32]:
    var host = ctx.enqueue_create_host_buffer[DType.uint32](len(values))
    ctx.synchronize()
    for i in range(len(values)):
        host[i] = UInt32(values[i])
    var dev = ctx.enqueue_create_buffer[DType.uint32](len(values))
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    return dev


def count_mismatches(a: List[UInt8], b: List[UInt8]) -> Int:
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
    return bad


def run_case[
    cache_dtype: DType,
    head_size: Int,
    scale_dtype: DType,
    scale_cols: Int,
    granularity: Int,
    tail: Int,
    dcp_size: Int,
    dcp_interleave: Int,
](ctx: DeviceContext) raises:
    print(
        "== pcp rows",
        cache_dtype,
        head_size,
        "scales",
        scale_cols,
        "dcp",
        dcp_size,
        "interleave",
        dcp_interleave,
    )
    # The scale pool is `ceil((head_size - tail) / granularity)` columns
    # wide, as the collection lays it out.
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=head_size, is_mla=True, fp8_tail_size=tail
    )
    comptime value_bytes = head_size * size_of[Scalar[cache_dtype]]()
    comptime scale_bytes = scale_cols * size_of[Scalar[scale_dtype]]()
    comptime row_bytes = value_bytes + scale_bytes
    comptime width = col_offset + row_bytes + trailing
    comptime scaled = scale_cols > 0
    comptime span = dcp_size * dcp_interleave
    var p = Prefixes()
    var bsz = p.batch_size()
    var rows = p.num_blocks * num_layers * page_size
    # An unscaled cache still gets a (never read) scale pool of one column.
    comptime pool_scale_bytes = max(scale_cols, 1) * size_of[
        Scalar[scale_dtype]
    ]()

    # Each rank's DCP cache: its owned positions at their local slots.
    var rank_values = List[List[UInt8]]()
    var rank_scales = List[List[UInt8]]()
    for rank in range(dcp_size):
        var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
        var values = List[UInt8](length=rows * value_bytes, fill=poison)
        var scales = List[UInt8](length=rows * pool_scale_bytes, fill=poison)
        for bs in range(bsz):
            for pos in range(p.cache_lens[bs]):
                if not layout.owns(pos):
                    continue
                var slot = layout.local_slot(pos)
                var r = row_index(p.dcp_block(bs, slot), slot % page_size)
                for i in range(value_bytes):
                    values[r * value_bytes + i] = row_byte(bs, pos, i)
                comptime if scaled:
                    for i in range(scale_bytes):
                        scales[r * scale_bytes + i] = scale_byte(bs, pos, i)
        rank_values.append(values^)
        rank_scales.append(scales^)

    # Each request's packed rows: the most local slots any rank holds.
    var row_offsets: List[Int] = [0]
    for bs in range(bsz):
        row_offsets.append(
            row_offsets[bs] + ceildiv(p.cache_lens[bs], span) * dcp_interleave
        )
    var packed_rows = row_offsets[bsz] + extra_rows

    var cache_lengths_dev = u32_buffer(ctx, p.cache_lens)
    var dcp_lut_dev = u32_buffer(ctx, p.dcp_lut)
    var whole_lut_dev = u32_buffer(ctx, p.whole_lut)
    var row_offsets_dev = u32_buffer(ctx, row_offsets)
    var dcp_lut_tt = TileTensor[mut=False](
        dcp_lut_dev, row_major((bsz, p.lut_cols))
    )
    var whole_lut_tt = TileTensor[mut=False](
        whole_lut_dev, row_major((bsz, p.lut_cols))
    )
    var cache_lengths_tt = TileTensor[mut=False](
        cache_lengths_dev, row_major(bsz)
    )
    var row_offsets_tt = TileTensor[mut=False](
        row_offsets_dev, row_major(bsz + 1)
    )

    var gathered = List[UInt8](capacity=dcp_size * packed_rows * width)
    for rank in range(dcp_size):
        var values_dev = device_bytes[cache_dtype](ctx, rank_values[rank])
        var scales_dev = device_bytes[scale_dtype](ctx, rank_scales[rank])
        var out_dev = device_bytes[DType.uint8](
            ctx, List[UInt8](length=packed_rows * width, fill=unwritten)
        )
        var out_tt = TileTensor(out_dev, row_major((packed_rows, width)))
        var coll = PagedKVCacheCollection[
            cache_dtype,
            kv_params,
            page_size,
            scale_dtype_=scale_dtype,
            quantization_granularity_=granularity,
        ](
            blocks_tt(values_dev, p.num_blocks, head_size),
            cache_lengths_tt,
            dcp_lut_tt,
            UInt32(1),
            UInt32(p.max_cache()),
            scales=blocks_tt(scales_dev, p.num_blocks, max(scale_cols, 1)),
            scales_lookup_table=dcp_lut_tt,
        )
        kv_cache_pack_owned_rows[
            scale_row_elems=scale_cols,
            col_offset=col_offset,
            dcp_size=dcp_size,
            dcp_interleave=dcp_interleave,
        ](out_tt, row_offsets_tt, coll.get_key_cache(layer_idx), rank, ctx)
        ctx.synchronize()
        var out = host_bytes(ctx, out_dev, packed_rows * width)

        # The packed rows: each local slot's row at the request's offset.
        var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
        var expected = List[UInt8](length=packed_rows * width, fill=unwritten)
        for bs in range(bsz):
            var local = layout.local_len(p.cache_lens[bs])
            for slot in range(local):
                var pos = layout.global_pos(slot)
                var base = (row_offsets[bs] + slot) * width + col_offset
                for i in range(value_bytes):
                    expected[base + i] = row_byte(bs, pos, i)
                comptime if scaled:
                    for i in range(scale_bytes):
                        expected[base + value_bytes + i] = scale_byte(
                            bs, pos, i
                        )
        assert_equal(
            count_mismatches(out, expected),
            0,
            msg="packed rows, rank " + String(rank),
        )
        for b in out:
            gathered.append(b)

    # The whole-context cache: every prefix position at its own row.
    var whole_values_dev = device_bytes[cache_dtype](
        ctx, List[UInt8](length=rows * value_bytes, fill=poison)
    )
    var whole_scales_dev = device_bytes[scale_dtype](
        ctx, List[UInt8](length=rows * pool_scale_bytes, fill=poison)
    )
    var gathered_dev = device_bytes[DType.uint8](ctx, gathered)
    var gathered_tt = TileTensor[mut=False](
        gathered_dev, row_major((dcp_size * packed_rows, width))
    )
    var whole = PagedKVCacheCollection[
        cache_dtype,
        kv_params,
        page_size,
        scale_dtype_=scale_dtype,
        quantization_granularity_=granularity,
    ](
        blocks_tt(whole_values_dev, p.num_blocks, head_size),
        cache_lengths_tt,
        whole_lut_tt,
        UInt32(1),
        UInt32(p.max_cache()),
        scales=blocks_tt(whole_scales_dev, p.num_blocks, max(scale_cols, 1)),
        scales_lookup_table=whole_lut_tt,
    )
    kv_cache_unpack_prefix_rows[
        scale_row_elems=scale_cols,
        col_offset=col_offset,
        dcp_size=dcp_size,
        dcp_interleave=dcp_interleave,
    ](gathered_tt, row_offsets_tt, whole.get_key_cache(layer_idx), ctx)
    ctx.synchronize()

    var expected_values = List[UInt8](length=rows * value_bytes, fill=poison)
    var expected_scales = List[UInt8](
        length=rows * pool_scale_bytes, fill=poison
    )
    for bs in range(bsz):
        for pos in range(p.cache_lens[bs]):
            var r = row_index(p.whole_block(bs, pos), pos % page_size)
            for i in range(value_bytes):
                expected_values[r * value_bytes + i] = row_byte(bs, pos, i)
            comptime if scaled:
                for i in range(scale_bytes):
                    expected_scales[r * scale_bytes + i] = scale_byte(
                        bs, pos, i
                    )
    assert_equal(
        count_mismatches(
            host_bytes(ctx, whole_values_dev, rows * head_size),
            expected_values,
        ),
        0,
        msg="whole-context values",
    )
    comptime if scaled:
        assert_equal(
            count_mismatches(
                host_bytes(ctx, whole_scales_dev, rows * scale_cols),
                expected_scales,
            ),
            0,
            msg="whole-context scales",
        )


def run_layouts[
    cache_dtype: DType,
    head_size: Int,
    scale_dtype: DType,
    scale_cols: Int,
    granularity: Int,
    tail: Int,
](ctx: DeviceContext) raises:
    run_case[
        cache_dtype, head_size, scale_dtype, scale_cols, granularity, tail, 1, 1
    ](ctx)
    run_case[
        cache_dtype, head_size, scale_dtype, scale_cols, granularity, tail, 2, 1
    ](ctx)
    run_case[
        cache_dtype, head_size, scale_dtype, scale_cols, granularity, tail, 4, 1
    ](ctx)
    run_case[
        cache_dtype, head_size, scale_dtype, scale_cols, granularity, tail, 8, 1
    ](ctx)
    run_case[
        cache_dtype,
        head_size,
        scale_dtype,
        scale_cols,
        granularity,
        tail,
        4,
        page_size,
    ](ctx)


def main() raises:
    with DeviceContext() as ctx:
        # The bf16 latent: 576 values, no scales (a one-column pool, never
        # read).
        run_layouts[DType.bfloat16, 576, DType.float32, 0, 576, 0](ctx)
        # The FP8 indexer key: 128 values and one FP32 scale.
        run_layouts[DType.float8_e4m3fn, 128, DType.float32, 1, 128, 0](ctx)
        # The packed NVFP4 latent: 256 E2M1 pairs and 64 FP8 rope bytes, and
        # one E4M3 scale per 16 latent elements.
        run_layouts[DType.uint8, 320, DType.float8_e4m3fn, 32, 8, 64](ctx)
