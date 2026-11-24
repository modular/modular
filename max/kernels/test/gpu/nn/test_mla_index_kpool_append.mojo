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
"""Checks `kpool_append_ragged_paged` against the unfused k-pool append.

The reference pools with `kpool_ring_close_kernel` and `kpool_compress_kernel`,
then quantizes each pooled key on the host as one FP8 group with a float32
scale and writes it where the pooled cache store does: pool `p` of a request
covers token rows `kpool * p` through `kpool * p + kpool - 1`, `head_dim //
kpool` values each, with the pool's scale on every one of those rows. The
cache starts from random bytes in every layer and page, so a stray or missing
write shows up as a byte difference anywhere in the cache.
"""

from std.math import ceildiv
from std.random import rand, seed
from std.testing import assert_equal, assert_true

from internal_utils.fp8_utils import (
    cast_saturating,
    compute_dynamic_fp8_scale,
    fp8_quantize,
)
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.attention.gpu.mla_index_kpool import (
    kpool_append_ragged_paged,
    kpool_compress_kernel,
    kpool_ring_close_kernel,
)

comptime HEAD_DIM = 128
comptime KPOOL = 4
comptime POOLED_DIM = HEAD_DIM // KPOOL
comptime PAGE_SIZE = 128
comptime NUM_SLOTS = 16
comptime NUM_LAYERS = 2
comptime LAYER = 1
comptime SCALE_UB = Float32(1200)
comptime FP8 = DType.float8_e4m3fn


def run_case(
    seq_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext
) raises:
    """Runs one append and compares every cache and scale byte.

    Args:
        seq_lens: New tokens per request in this call.
        cache_lens: Cached-prefix length per request.
        ctx: Device context.
    """
    var batch = len(seq_lens)
    var total_tokens = 0
    var max_tokens = 0
    for r in range(batch):
        total_tokens += seq_lens[r]
        max_tokens = max(max_tokens, cache_lens[r] + seq_lens[r])

    var key_elems = total_tokens * HEAD_DIM
    comptime tail_elems = NUM_SLOTS * 2 * KPOOL * HEAD_DIM
    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](key_elems)
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](key_elems)
    var ape_host = ctx.enqueue_create_host_buffer[.float32](KPOOL * HEAD_DIM)
    var tail_host = ctx.enqueue_create_host_buffer[.bfloat16](tail_elems)
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch + 1)
    var pro_host = ctx.enqueue_create_host_buffer[.uint32](batch + 1)
    var clen_host = ctx.enqueue_create_host_buffer[.uint32](batch)
    var slot_host = ctx.enqueue_create_host_buffer[.uint32](batch)
    var closed_host = ctx.enqueue_create_host_buffer[.int32](batch)
    ctx.synchronize()

    # Signed keys of a few units, so pools differ in amax and the scales do.
    rand(k_host.unsafe_ptr(), key_elems)
    rand(gate_host.unsafe_ptr(), key_elems)
    rand(ape_host.unsafe_ptr(), KPOOL * HEAD_DIM)
    rand(tail_host.unsafe_ptr(), tail_elems)
    for i in range(key_elems):
        k_host[i] = k_host[i] * 6 - 3
        gate_host[i] = gate_host[i] * 4 - 2
    for i in range(tail_elems):
        tail_host[i] = tail_host[i] * 6 - 3

    iro_host[0] = 0
    pro_host[0] = 0
    for r in range(batch):
        var len_r = cache_lens[r]
        var n_r = seq_lens[r]
        iro_host[r + 1] = iro_host[r] + UInt32(n_r)
        var full = max((len_r + n_r) // KPOOL - ceildiv(len_r, KPOOL), 0)
        pro_host[r + 1] = pro_host[r] + UInt32(full)
        clen_host[r] = UInt32(len_r)
        # Slots out of row order, so reading the ring by row is caught.
        slot_host[r] = UInt32((NUM_SLOTS - 1 - 3 * r) % NUM_SLOTS)
        closed_host[r] = -1
    var total_pools = Int(pro_host[batch])

    # The paged cache: random pages, a reversed page table, and a second
    # layer, so only one layer of the right pages may change.
    var pages_per_seq = ceildiv(max_tokens, PAGE_SIZE)
    var num_blocks = batch * pages_per_seq + 2
    comptime page_stride = NUM_LAYERS * PAGE_SIZE * POOLED_DIM
    comptime scale_page_stride = NUM_LAYERS * PAGE_SIZE
    var kc_len = num_blocks * page_stride
    var ks_len = num_blocks * scale_page_stride
    var lut_len = batch * pages_per_seq
    var kc_init = ctx.enqueue_create_host_buffer[FP8](kc_len)
    var ks_init = ctx.enqueue_create_host_buffer[.float32](ks_len)
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_len)
    ctx.synchronize()
    rand(kc_init.unsafe_ptr(), kc_len)
    rand(ks_init.unsafe_ptr(), ks_len)
    for i in range(lut_len):
        lut_host[i] = UInt32(num_blocks - 1 - i)

    var pooled_rows = max(total_pools, 1)
    var k_dev = ctx.enqueue_create_buffer[.bfloat16](key_elems)
    var gate_dev = ctx.enqueue_create_buffer[.bfloat16](key_elems)
    var ape_dev = ctx.enqueue_create_buffer[.float32](KPOOL * HEAD_DIM)
    var tail_dev = ctx.enqueue_create_buffer[.bfloat16](tail_elems)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var pro_dev = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var clen_dev = ctx.enqueue_create_buffer[.uint32](batch)
    var slot_dev = ctx.enqueue_create_buffer[.uint32](batch)
    var closed_dev = ctx.enqueue_create_buffer[.int32](batch)
    var ring_dev = ctx.enqueue_create_buffer[.bfloat16](batch * HEAD_DIM)
    var pooled_dev = ctx.enqueue_create_buffer[.bfloat16](
        pooled_rows * HEAD_DIM
    )
    var kc_dev = ctx.enqueue_create_buffer[FP8](kc_len)
    var ks_dev = ctx.enqueue_create_buffer[.float32](ks_len)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](lut_len)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(gate_dev, gate_host)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.enqueue_copy(tail_dev, tail_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(pro_dev, pro_host)
    ctx.enqueue_copy(clen_dev, clen_host)
    ctx.enqueue_copy(slot_dev, slot_host)
    ctx.enqueue_copy(closed_dev, closed_host)
    ctx.enqueue_copy(kc_dev, kc_init)
    ctx.enqueue_copy(ks_dev, ks_init)
    ctx.enqueue_copy(lut_dev, lut_host)

    var k_t = TileTensor(k_dev, row_major(total_tokens, HEAD_DIM))
    var gate_t = TileTensor(gate_dev, row_major(total_tokens, HEAD_DIM))
    var ape_t = TileTensor(ape_dev, row_major(KPOOL, HEAD_DIM))
    var tail_t = TileTensor(tail_dev, row_major(NUM_SLOTS, 2, KPOOL, HEAD_DIM))
    var iro_t = TileTensor(iro_dev, row_major(batch + 1))
    var pro_t = TileTensor(pro_dev, row_major(batch + 1))
    var clen_t = TileTensor(clen_dev, row_major(batch))
    var slot_t = TileTensor(slot_dev, row_major(batch))
    var closed_t = TileTensor(closed_dev, row_major(batch))
    var ring_t = TileTensor(ring_dev, row_major(batch, HEAD_DIM))
    var pooled_t = TileTensor(pooled_dev, row_major(pooled_rows, HEAD_DIM))

    # The reference pooling.
    comptime ring_kernel = kpool_ring_close_kernel[
        .bfloat16,
        ring_t.LayoutType,
        ring_t.origin,
        closed_t.LayoutType,
        closed_t.origin,
        type_of(tail_t.as_imm()).LayoutType,
        ImmOrigin(tail_t.origin),
        type_of(k_t.as_imm()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(gate_t.as_imm()).LayoutType,
        ImmOrigin(gate_t.origin),
        type_of(ape_t.as_imm()).LayoutType,
        ImmOrigin(ape_t.origin),
        type_of(iro_t.as_imm()).LayoutType,
        ImmOrigin(iro_t.origin),
        type_of(clen_t.as_imm()).LayoutType,
        type_of(slot_t.as_imm()).LayoutType,
        ImmOrigin(slot_t.origin),
        ring_t.Engine,
        closed_t.Engine,
        type_of(tail_t.as_imm()).Engine,
        type_of(k_t.as_imm()).Engine,
        type_of(gate_t.as_imm()).Engine,
        type_of(ape_t.as_imm()).Engine,
        type_of(iro_t.as_imm()).Engine,
        type_of(clen_t.as_imm()).Engine,
        type_of(slot_t.as_imm()).Engine,
        HEAD_DIM,
        KPOOL,
    ]
    ctx.enqueue_function[ring_kernel](
        ring_t,
        closed_t,
        tail_t.as_imm(),
        k_t.as_imm(),
        gate_t.as_imm(),
        ape_t.as_imm(),
        iro_t.as_imm(),
        clen_t.as_imm(),
        slot_t.as_imm(),
        Int32(batch),
        grid_dim=(batch, 1, 1),
        block_dim=(HEAD_DIM, 1, 1),
    )
    if total_pools > 0:
        comptime compress_kernel = kpool_compress_kernel[
            .bfloat16,
            type_of(k_t.as_imm()).LayoutType,
            ImmOrigin(k_t.origin),
            type_of(gate_t.as_imm()).LayoutType,
            ImmOrigin(gate_t.origin),
            type_of(ape_t.as_imm()).LayoutType,
            ImmOrigin(ape_t.origin),
            type_of(iro_t.as_imm()).LayoutType,
            ImmOrigin(iro_t.origin),
            type_of(pro_t.as_imm()).LayoutType,
            ImmOrigin(pro_t.origin),
            type_of(clen_t.as_imm()).LayoutType,
            pooled_t.LayoutType,
            pooled_t.origin,
            type_of(k_t.as_imm()).Engine,
            type_of(gate_t.as_imm()).Engine,
            type_of(ape_t.as_imm()).Engine,
            type_of(iro_t.as_imm()).Engine,
            type_of(pro_t.as_imm()).Engine,
            type_of(clen_t.as_imm()).Engine,
            pooled_t.Engine,
            HEAD_DIM,
            KPOOL,
        ]
        ctx.enqueue_function[compress_kernel](
            pooled_t,
            k_t.as_imm(),
            gate_t.as_imm(),
            ape_t.as_imm(),
            iro_t.as_imm(),
            pro_t.as_imm(),
            clen_t.as_imm(),
            grid_dim=(total_pools, 1, 1),
            block_dim=(HEAD_DIM, 1, 1),
        )

    # The fused append, on the cache the graph op builds.
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=POOLED_DIM, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        FP8,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=HEAD_DIM,
    ]
    var collection = Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                kc_dev,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(NUM_LAYERS),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[POOLED_DIM],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(clen_dev, row_major(Int64(batch)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(Int64(batch), Int64(pages_per_seq)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(max_tokens),
        UInt32(max_tokens),
        rebind[Collection.scales_tt_type](
            TileTensor(
                ks_dev,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(NUM_LAYERS),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[1],
                ),
            ).as_unsafe_any_origin()
        ),
        page_stride=page_stride,
        scales_page_stride=scale_page_stride,
    )
    kpool_append_ragged_paged[head_dim=HEAD_DIM, kpool=KPOOL](
        tail_t.as_imm(),
        k_t.as_imm(),
        gate_t.as_imm(),
        ape_t.as_imm(),
        iro_t.as_imm(),
        clen_t.as_imm(),
        slot_t.as_imm(),
        collection.get_key_cache(LAYER),
        SCALE_UB,
        ctx,
    )

    var ring_host = ctx.enqueue_create_host_buffer[.bfloat16](batch * HEAD_DIM)
    var pooled_host = ctx.enqueue_create_host_buffer[.bfloat16](
        pooled_rows * HEAD_DIM
    )
    var kc_got = ctx.enqueue_create_host_buffer[FP8](kc_len)
    var ks_got = ctx.enqueue_create_host_buffer[.float32](ks_len)
    ctx.enqueue_copy(ring_host, ring_dev)
    ctx.enqueue_copy(pooled_host, pooled_dev)
    ctx.enqueue_copy(closed_host, closed_dev)
    ctx.enqueue_copy(kc_got, kc_dev)
    ctx.enqueue_copy(ks_got, ks_dev)
    ctx.synchronize()

    # Expected cache: the initial bytes with every closed and compressed pool
    # quantized and stored.
    var kc_want = List[Scalar[FP8]](capacity=kc_len)
    for i in range(kc_len):
        kc_want.append(kc_init[i])
    var ks_want = List[Float32](capacity=ks_len)
    for i in range(ks_len):
        ks_want.append(ks_init[i])

    var stored = 0
    for r in range(batch):
        var num_pools = Int(pro_host[r + 1]) - Int(pro_host[r])
        var closed = Int(closed_host[r])
        for j in range(-1, num_pools):
            var pool_id: Int
            var src: Int
            var from_ring = j < 0
            if from_ring:
                if closed < 0:
                    continue
                pool_id = closed
                src = r * HEAD_DIM
            else:
                pool_id = ceildiv(cache_lens[r], KPOOL) + j
                src = (Int(pro_host[r]) + j) * HEAD_DIM
            var amax = Float32(0)
            for c in range(HEAD_DIM):
                var v = (
                    ring_host[src + c] if from_ring else pooled_host[src + c]
                ).cast[.float32]()
                amax = max(amax, abs(v))
            var scale, scale_recip = compute_dynamic_fp8_scale[FP8](
                amax, SCALE_UB
            )
            for c in range(HEAD_DIM):
                var v = (
                    ring_host[src + c] if from_ring else pooled_host[src + c]
                ).cast[.float32]()
                var q = cast_saturating[FP8](fp8_quantize[FP8](v, scale_recip))
                var pos = pool_id * KPOOL + c // POOLED_DIM
                var page = pos // PAGE_SIZE
                var row = pos % PAGE_SIZE
                var blk = Int(lut_host[r * pages_per_seq + page])
                var row_base = (blk * NUM_LAYERS + LAYER) * PAGE_SIZE + row
                kc_want[row_base * POOLED_DIM + c % POOLED_DIM] = q
                ks_want[row_base] = scale
            stored += 1

    assert_true(stored > 0, "the case closes or builds no pool")
    for i in range(kc_len):
        assert_equal(
            kc_got[i].cast[.float32](),
            kc_want[i].cast[.float32](),
            String(t"cache element {i}"),
        )
    for i in range(ks_len):
        assert_equal(ks_got[i], ks_want[i], String(t"scale element {i}"))
    print(t"PASS {stored} pools, batch {batch}, {total_tokens} tokens")


def main() raises:
    seed(1)
    with DeviceContext() as ctx:
        # Decode at every ring phase: only the requests at `cache_len % 4 ==
        # 3` close a pool, and none builds one from new tokens alone.
        run_case(
            [1, 1, 1, 1, 1, 1, 1, 1],
            [100, 101, 102, 103, 4000, 4001, 4002, 4003],
            ctx,
        )
        # MTP verify, 6 tokens per request: a pending ring pool closes, and
        # requests whose new tokens complete another pool also build it.
        run_case(
            [6, 6, 6, 6, 6, 6, 6, 6],
            [100, 101, 102, 103, 7001, 7002, 7003, 7004],
            ctx,
        )
        # Ragged verify widths, including requests too short to close.
        run_case([6, 6, 1, 6, 3, 6, 2, 6], [0, 5, 6, 7, 9, 10, 31, 12000], ctx)
        # Prefill chunks spanning many pages, one starting mid-pool.
        run_case([1500, 548], [0, 37], ctx)
        # A chunked prefill sharing the batch with decodes.
        run_case([2045, 1, 1, 1], [3, 50, 51, 52], ctx)
