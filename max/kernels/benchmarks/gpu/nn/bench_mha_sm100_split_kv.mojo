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
"""Times SM100 attention over a split 192/128 KV cache against the same
attention over the cache padded to 256, at MiMo-V2.6-Flash's full-attention
group (16 query heads per KV head, page 128) or, with `--group=swa`, its
sliding-window group (8 query heads per KV head, a 128-key window and a sink
logit per query head).

Each shape runs both arms back to back, `--reps` times. The split arm gets
the partition count an engine puts in the dispatch metadata
(`mha_split_kv_decode_num_partitions`, or each of `--p` with `--mode=sweep`);
the padded arm runs its production route, which sizes its own count.

`--mode=decode` covers decode (`--q=1`) and speculative verify (`--q=8`) at
every `--batches` x `--caches`; `--mode=prefill` runs one `--chunks` prompt
over each `--prefixes` cached prefix. `--kvh` picks 1, 2 or 4 KV heads per
device (TP4, TP2, or all four under DP8), or 0 for all three; the
sliding-window group runs 2 or 4 (TP4 or TP2).
"""

from std.collections import OptionalReg
from std.math import align_up, ceildiv, rsqrt

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.gpu.mha import MHADecodeDispatchMetadata, flash_attention
from nn.attention.gpu.nvidia.sm100.dispatch import (
    mha_split_kv_decode_num_partitions,
)
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from nn.attention.mha_mask import CausalMask, SlidingWindowCausalMask

comptime dtype = DType.bfloat16
comptime page_size = 128
comptime full_group = 16
comptime swa_group = 8
comptime window = 128
comptime sm_count = DeviceContext.default_device_info.sm_count


def _ints(text: String) raises -> List[Int]:
    var values = List[Int]()
    for part in text.split(","):
        values.append(Int(part))
    return values^


def run[
    num_kv_heads: Int,
    k_dim: Int,
    v_dim: Int,
    row: Int,
    planes: Int,
    swa: Bool,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch: Int,
    seq_len: Int,
    cache_len: Int,
    meta_p: Int,
    name: String,
) raises:
    """Times one attention launch; `meta_p > 0` hands it dispatch metadata
    with that partition count, as the graph op does for decode and verify."""
    comptime group = swa_group if swa else full_group
    comptime num_q_heads = group * num_kv_heads
    comptime params = KVCacheStaticParams(
        num_kv_heads, row, v_head_size=v_dim
    ) if planes == 1 else KVCacheStaticParams(num_kv_heads, row)
    comptime Collection = PagedKVCacheCollection[
        dtype,
        params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var num_keys = seq_len + cache_len
    var pages_per_seq = ceildiv(num_keys, page_size)
    var total_pages = batch * pages_per_seq
    var lut_cols = align_up(pages_per_seq, 8) + 16
    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch * lut_cols
    )
    for b in range(batch):
        for i in range(lut_cols):
            lut_host[b * lut_cols + i] = UInt32(
                b * pages_per_seq + i if i < pages_per_seq else total_pages
            )
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](batch * lut_cols)
    ctx.enqueue_copy(lut_dev, lut_host)
    var cl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch)
    var off_host = ctx.enqueue_create_host_buffer[DType.uint32](batch + 1)
    off_host[0] = 0
    for b in range(batch):
        cl_host[b] = UInt32(cache_len)
        off_host[b + 1] = UInt32((b + 1) * seq_len)
    var cl_dev = ctx.enqueue_create_buffer[DType.uint32](batch)
    ctx.enqueue_copy(cl_dev, cl_host)
    var off_dev = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    ctx.enqueue_copy(off_dev, off_host)

    var page_elems = planes * page_size * num_kv_heads * row
    var blocks_dev = ctx.enqueue_create_buffer[dtype](
        (total_pages + 1) * page_elems
    )
    blocks_dev.enqueue_fill(0.01)
    comptime BlocksLayout = Collection.blocks_tt_layout
    var shape = Coord[*BlocksLayout.shape_types]()
    shape[0] = Int64(total_pages + 1)
    shape[2] = Int64(1)
    var strides = Coord[*BlocksLayout.stride_types]()
    strides[0] = Int64(page_elems)
    strides[1] = Int64(page_size * num_kv_heads * row)
    var blocks = TileTensor(blocks_dev, BlocksLayout(shape, strides))
    var collection = Collection(
        blocks.as_unsafe_any_origin(),
        TileTensor(cl_dev, row_major(Int64(batch)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(Int64(batch), Int64(lut_cols)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(num_keys),
    )
    var rows = batch * seq_len
    var q_dev = ctx.enqueue_create_buffer[dtype](rows * num_q_heads * k_dim)
    q_dev.enqueue_fill(0.01)
    var o_dev = ctx.enqueue_create_buffer[dtype](rows * num_q_heads * v_dim)
    var q = TileTensor(
        q_dev, row_major(Int64(rows), Idx[num_q_heads], Idx[k_dim])
    ).as_imm()
    var o = TileTensor(
        o_dev, row_major(Int64(rows), Idx[num_q_heads], Idx[v_dim])
    )
    var offsets = TileTensor(off_dev, row_major(Int64(batch + 1))).as_imm()
    var k_cache = collection.get_key_cache(0)
    var v_cache = collection.get_value_cache(0)
    var scale = rsqrt(Float32(192))
    var meta: OptionalReg[MHADecodeDispatchMetadata] = None
    if meta_p > 0:
        meta = MHADecodeDispatchMetadata(batch, seq_len, meta_p, num_keys)
    var sinks_dev = ctx.enqueue_create_buffer[dtype](num_q_heads)
    sinks_dev.enqueue_fill(0.5)
    var sinks: OptionalReg[ImmutTileTensor1D[dtype]] = None
    comptime if swa:
        sinks = (
            TileTensor(sinks_dev, row_major(Int64(num_q_heads)))
            .as_imm()
            .as_unsafe_any_origin()
        )

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var q,
        var o,
        var k_cache,
        var v_cache,
        var offsets,
        var meta,
        var sinks,
        imm,
    }:
        @inline(.always)
        def launch(ctx: DeviceContext) raises {imm}:
            comptime if swa:
                flash_attention[ragged=True, sink=True](
                    o,
                    q,
                    k_cache,
                    v_cache,
                    SlidingWindowCausalMask[window](),
                    offsets,
                    scale,
                    ctx,
                    sink_weights=sinks,
                    decode_dispatch_metadata=meta,
                )
            else:
                flash_attention[ragged=True](
                    o,
                    q,
                    k_cache,
                    v_cache,
                    CausalMask(),
                    offsets,
                    scale,
                    ctx,
                    decode_dispatch_metadata=meta,
                )

        bencher_iter_custom(b, launch, ctx)

    m.bench_function(
        bench_fn,
        BenchId(
            String(
                name,
                " swa" if swa else "",
                " kvh=",
                num_kv_heads,
                " b=",
                batch,
                " q=",
                seq_len,
                " cache=",
                cache_len,
                " P=",
                meta_p,
            )
        ),
    )
    ctx.synchronize()
    _ = blocks_dev^
    _ = q_dev^
    _ = o_dev^
    _ = lut_dev^
    _ = cl_dev^
    _ = off_dev^
    _ = sinks_dev^


def native[
    kvh: Int, swa: Bool
](
    ctx: DeviceContext,
    mut m: Bench,
    batch: Int,
    q: Int,
    cache: Int,
    meta_p: Int,
    name: String,
) raises:
    run[kvh, 192, 128, 320, 1, swa](ctx, m, batch, q, cache, meta_p, name)


def padded[
    kvh: Int, swa: Bool
](ctx: DeviceContext, mut m: Bench, batch: Int, q: Int, cache: Int) raises:
    # The padded route ignores the metadata's count; it is handed metadata
    # because the graph op always passes it.
    comptime group = swa_group if swa else full_group
    var meta_p = 1 if q * group <= 128 else 0
    run[kvh, 256, 256, 256, 2, swa](ctx, m, batch, q, cache, meta_p, "padded")


def shape[
    kvh: Int, swa: Bool
](
    ctx: DeviceContext,
    mut m: Bench,
    mode: String,
    batch: Int,
    q: Int,
    cache: Int,
    sweep: List[Int],
) raises:
    if mode == "sweep":
        for p in sweep:
            native[kvh, swa](ctx, m, batch, q, cache, p, "native")
        return
    var p = 0
    if mode == "decode":
        p = mha_split_kv_decode_num_partitions[sm_count](
            batch, cache + q, kvh, window if swa else 0
        )
    native[kvh, swa](ctx, m, batch, q, cache, p, "native")
    padded[kvh, swa](ctx, m, batch, q, cache)


def grid[
    kvh: Int, swa: Bool
](ctx: DeviceContext, mut m: Bench, mode: String) raises:
    var sweep = _ints(arg_parse("p", "1,2,4,8"))
    if mode == "prefill":
        for chunk in _ints(arg_parse("chunks", "2048,8192")):
            for prefix in _ints(arg_parse("prefixes", "0,32768,131072")):
                shape[kvh, swa](ctx, m, mode, 1, chunk, prefix, sweep)
        return
    for q in _ints(arg_parse("q", "1,8")):
        for batch in _ints(arg_parse("batches", "1,8,16,32,64")):
            for cache in _ints(arg_parse("caches", "4096,32768,131072,262144")):
                shape[kvh, swa](ctx, m, mode, batch, q, cache, sweep)


def main() raises:
    var mode = arg_parse("mode", "decode")
    var kvh = arg_parse("kvh", 0)
    var reps = arg_parse("reps", 3)
    var swa = arg_parse("group", "full") == "swa"
    var m = Bench()
    with DeviceContext() as ctx:
        for _ in range(reps):
            if swa:
                comptime for heads in [2, 4]:
                    if kvh == 0 or kvh == heads:
                        grid[heads, True](ctx, m, mode)
                continue
            comptime for heads in [1, 2, 4]:
                if kvh == 0 or kvh == heads:
                    grid[heads, False](ctx, m, mode)
    m.dump_report()
