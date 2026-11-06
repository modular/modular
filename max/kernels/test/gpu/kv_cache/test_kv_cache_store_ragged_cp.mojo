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
"""Decode context-parallel ragged KV stores, against the plain store.

Every rank of a `dcp`-way group gets the same lookup table and the same global
`cache_lengths` and runs `kv_cache_store_ragged` and
`kv_cache_store_scales_ragged` with its own rank. The rank caches are checked
byte for byte against an image built from one non-CP store of the same
inputs: each token's row (values and scale) on its owner, at the owner's local
slot, and the poison everywhere else. The cache is shaped like the DSA index
cache (MLA layout, one FP8 head of 128 with one FP32 scale).
"""

from std.collections import Set
from std.math import ceildiv
from std.random import random_float64, random_ui64, seed
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext
from kv_cache.context_parallel import ContextParallelLayout
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, RowMajorLayout, TileTensor, row_major
from nn.kv_cache_ragged import (
    kv_cache_store_ragged,
    kv_cache_store_scales_ragged,
)
from std.testing import assert_equal

comptime cache_dtype = DType.float8_e4m3fn
comptime scale_dtype = DType.float32
comptime head_size = 128
comptime granularity = 128
comptime scale_cols = head_size // granularity
comptime page_size = 128
comptime num_layers = 2
comptime layer_idx = 1
comptime poison = UInt8(0x7F)
comptime kv_params = KVCacheStaticParams(
    num_heads=1, head_size=head_size, is_mla=True
)


def blocks_tt[
    dt: DType
](mut dev: DeviceBuffer[dt], num_blocks: Int, width: Int) -> TileTensor[
    dt, RowMajorLayout[Int, Int, Int, Int, Int, Int], origin_of(dev)
]:
    return TileTensor(
        dev, row_major(num_blocks, 1, num_layers, page_size, 1, width)
    )


def row_offset(block: Int, row: Int, width: Int) -> Int:
    """Element offset of cache row `(block, layer_idx, row)`."""
    return (block * num_layers + layer_idx) * page_size * width + row * width


struct Batch(Movable):
    """A ragged batch: prefill rows, decode rows and logical-page crossings."""

    var prompt_lens: List[Int]
    var cache_lens: List[Int]
    var lut_cols: Int
    var num_blocks: Int
    var lut: List[Int]

    def __init__(out self):
        seed(11)
        # Rows 1, 6 and 7 decode one token; 1 and 6 end a page. Row 5 crosses
        # global position 1024, which is a logical-page boundary for every
        # group size here, and row 3 crosses 256 and 384.
        self.prompt_lens = [17, 1, 31, 130, 3, 40, 1, 1]
        self.cache_lens = [0, 127, 13, 250, 6, 1000, 1023, 2047]
        var max_context = 0
        self.num_blocks = 4
        for i in range(len(self.prompt_lens)):
            var context = self.cache_lens[i] + self.prompt_lens[i]
            max_context = max(max_context, context)
            self.num_blocks += ceildiv(context, page_size)
        self.lut_cols = ceildiv(max_context, page_size)
        # Unused columns point past the pool; used ones get distinct random
        # blocks.
        self.lut = List[Int](
            length=len(self.prompt_lens) * self.lut_cols, fill=self.num_blocks
        )
        var used = Set[Int]()
        for i in range(len(self.prompt_lens)):
            var pages = ceildiv(
                self.cache_lens[i] + self.prompt_lens[i], page_size
            )
            for c in range(pages):
                var b = Int(random_ui64(0, UInt64(self.num_blocks - 1)))
                while b in used:
                    b = Int(random_ui64(0, UInt64(self.num_blocks - 1)))
                used.add(b)
                self.lut[i * self.lut_cols + c] = b

    def batch_size(self) -> Int:
        return len(self.prompt_lens)

    def total_length(self) -> Int:
        var total = 0
        for n in self.prompt_lens:
            total += n
        return total

    def block(self, bs: Int, slot: Int) -> Int:
        return self.lut[bs * self.lut_cols + slot // page_size]


def store[
    dcp_size: Int, dcp_interleave: Int
](
    ctx: DeviceContext,
    batch: Batch,
    mut values_dev: DeviceBuffer[cache_dtype],
    mut scales_dev: DeviceBuffer[scale_dtype],
    x_dev: DeviceBuffer[cache_dtype],
    x_scales_dev: DeviceBuffer[scale_dtype],
    row_offsets_dev: DeviceBuffer[DType.uint32],
    cache_lengths_dev: DeviceBuffer[DType.uint32],
    lut_dev: DeviceBuffer[DType.uint32],
    dcp_rank: Int,
) raises:
    """Runs both stores for one rank into one cache."""
    var total = batch.total_length()
    var max_prompt = 0
    var max_cache = 0
    for i in range(batch.batch_size()):
        max_prompt = max(max_prompt, batch.prompt_lens[i])
        max_cache = max(max_cache, batch.cache_lens[i])
    var lut_tt = TileTensor[mut=False](
        lut_dev, row_major((batch.batch_size(), batch.lut_cols))
    )
    var coll = PagedKVCacheCollection[
        cache_dtype,
        kv_params,
        page_size,
        scale_dtype_=scale_dtype,
        quantization_granularity_=granularity,
    ](
        blocks_tt(values_dev, batch.num_blocks, head_size),
        TileTensor[mut=False](cache_lengths_dev, row_major(batch.batch_size())),
        lut_tt,
        UInt32(max_prompt),
        UInt32(max_cache),
        scales=blocks_tt(scales_dev, batch.num_blocks, scale_cols),
        scales_lookup_table=lut_tt,
    )
    var cache = coll.get_key_cache(layer_idx)
    var row_offsets_tt = TileTensor[mut=False](
        row_offsets_dev, row_major(batch.batch_size() + 1)
    )
    var x_tt = TileTensor[mut=False](
        x_dev, row_major((total, Idx[1], Idx[head_size]))
    )
    var x_scales_tt = TileTensor[mut=False](
        x_scales_dev, row_major((total, Idx[1], Idx[scale_cols]))
    )

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var x_tt} -> SIMD[cache_dtype, width]:
        return x_tt.load[width=width](idx)

    @inline(.always)
    def scales_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var x_scales_tt} -> SIMD[scale_dtype, width]:
        return x_scales_tt.load[width=width](idx)

    kv_cache_store_ragged[dcp_size=dcp_size, dcp_interleave=dcp_interleave](
        cache,
        (total, 1, head_size),
        row_offsets_tt,
        input_fn,
        ctx,
        dcp_rank,
    )
    kv_cache_store_scales_ragged[
        dcp_size=dcp_size, dcp_interleave=dcp_interleave
    ](
        cache,
        (total, 1, scale_cols),
        row_offsets_tt,
        scales_fn,
        ctx,
        dcp_rank,
    )


def poisoned[dt: DType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dt]:
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    var bytes = host.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(n * size_of[Scalar[dt]]()):
        bytes[i] = poison
    var dev = ctx.enqueue_create_buffer[dt](n)
    ctx.enqueue_copy(dev, host)
    return dev


def to_bytes[
    dt: DType
](ctx: DeviceContext, dev: DeviceBuffer[dt], n: Int) raises -> List[UInt8]:
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    var bytes = host.unsafe_ptr().unsafe_bitcast[UInt8]()
    var out = List[UInt8](capacity=n * size_of[Scalar[dt]]())
    for i in range(n * size_of[Scalar[dt]]()):
        out.append(bytes[i])
    return out^


def expected_rank_image(
    batch: Batch,
    ref_bytes: List[UInt8],
    row_bytes: Int,
    owner_of: List[Int],
    slot_of: List[Int],
    rank: Int,
) -> List[UInt8]:
    """Poison, plus the reference row of every token `rank` owns at its slot."""
    var image = List[UInt8](length=len(ref_bytes), fill=poison)
    var tok = 0
    for bs in range(batch.batch_size()):
        for t in range(batch.prompt_lens[bs]):
            var pos = batch.cache_lens[bs] + t
            if owner_of[tok] == rank:
                var slot = slot_of[tok]
                var src = row_offset(batch.block(bs, pos), pos % page_size, 1)
                var dst = row_offset(batch.block(bs, slot), slot % page_size, 1)
                for i in range(row_bytes):
                    image[dst * row_bytes + i] = ref_bytes[src * row_bytes + i]
            tok += 1
    return image^


def count_mismatches(a: List[UInt8], b: List[UInt8]) -> Int:
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
    return bad


def run_case[dcp_size: Int, dcp_interleave: Int](ctx: DeviceContext) raises:
    print("== store_ragged_cp dcp", dcp_size, "interleave", dcp_interleave)
    var batch = Batch()
    var total = batch.total_length()
    var bsz = batch.batch_size()

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](bsz + 1)
    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](bsz)
    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        bsz * batch.lut_cols
    )
    var x_host = ctx.enqueue_create_host_buffer[cache_dtype](total * head_size)
    var x_scales_host = ctx.enqueue_create_host_buffer[scale_dtype](
        total * scale_cols
    )
    ctx.synchronize()
    var offset = 0
    for i in range(bsz):
        row_offsets_host[i] = UInt32(offset)
        cache_lengths_host[i] = UInt32(batch.cache_lens[i])
        offset += batch.prompt_lens[i]
    row_offsets_host[bsz] = UInt32(offset)
    for i in range(bsz * batch.lut_cols):
        lut_host[i] = UInt32(batch.lut[i])
    # Random E4M3 bytes, never the poison byte or the other NaN.
    var x_bytes = x_host.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(total * head_size):
        var b = UInt8(random_ui64(0, 253))
        x_bytes[i] = b + 1 if b >= 0x7F else b
    for i in range(total * scale_cols):
        x_scales_host[i] = Float32(random_float64(0.5, 2.0))

    var row_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](bsz + 1)
    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](bsz)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](bsz * batch.lut_cols)
    var x_dev = ctx.enqueue_create_buffer[cache_dtype](total * head_size)
    var x_scales_dev = ctx.enqueue_create_buffer[scale_dtype](
        total * scale_cols
    )
    ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    ctx.enqueue_copy(lut_dev, lut_host)
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(x_scales_dev, x_scales_host)

    var values_n = batch.num_blocks * num_layers * page_size * head_size
    var scales_n = batch.num_blocks * num_layers * page_size * scale_cols

    # The non-CP store is the reference.
    var ref_values = poisoned[cache_dtype](ctx, values_n)
    var ref_scales = poisoned[scale_dtype](ctx, scales_n)
    store[1, 1](
        ctx,
        batch,
        ref_values,
        ref_scales,
        x_dev,
        x_scales_dev,
        row_offsets_dev,
        cache_lengths_dev,
        lut_dev,
        0,
    )
    var ref_value_bytes = to_bytes(ctx, ref_values, values_n)
    var ref_scale_bytes = to_bytes(ctx, ref_scales, scales_n)

    var owner_of = List[Int]()
    var slot_of = List[Int]()
    for bs in range(bsz):
        for t in range(batch.prompt_lens[bs]):
            var pos = batch.cache_lens[bs] + t
            var layout = ContextParallelLayout[dcp_size, dcp_interleave](0)
            owner_of.append(layout.owner(pos))
            slot_of.append(layout.local_slot(pos))

    var stored = 0
    for rank in range(dcp_size):
        var values = poisoned[cache_dtype](ctx, values_n)
        var scales = poisoned[scale_dtype](ctx, scales_n)
        store[dcp_size, dcp_interleave](
            ctx,
            batch,
            values,
            scales,
            x_dev,
            x_scales_dev,
            row_offsets_dev,
            cache_lengths_dev,
            lut_dev,
            rank,
        )
        var value_bad = count_mismatches(
            to_bytes(ctx, values, values_n),
            expected_rank_image(
                batch,
                ref_value_bytes,
                head_size * size_of[Scalar[cache_dtype]](),
                owner_of,
                slot_of,
                rank,
            ),
        )
        var scale_bad = count_mismatches(
            to_bytes(ctx, scales, scales_n),
            expected_rank_image(
                batch,
                ref_scale_bytes,
                scale_cols * size_of[Scalar[scale_dtype]](),
                owner_of,
                slot_of,
                rank,
            ),
        )
        assert_equal(value_bad, 0, msg="value bytes, rank " + String(rank))
        assert_equal(scale_bad, 0, msg="scale bytes, rank " + String(rank))
        for o in owner_of:
            if o == rank:
                stored += 1
    # Every token is owned by exactly one rank.
    assert_equal(stored, total)

    _ = row_offsets_dev
    _ = cache_lengths_dev
    _ = lut_dev
    _ = x_dev
    _ = x_scales_dev


def main() raises:
    with DeviceContext() as ctx:
        # A group of one is the plain store.
        run_case[1, 1](ctx)
        run_case[2, 1](ctx)
        run_case[2, page_size](ctx)
        run_case[4, 1](ctx)
        run_case[4, page_size](ctx)
        run_case[8, 1](ctx)
        run_case[8, page_size](ctx)
