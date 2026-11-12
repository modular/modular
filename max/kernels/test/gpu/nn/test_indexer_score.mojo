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
"""Checks `indexer_score_ragged_paged`'s GPU scores against fp64 at
DeepSeek-V4-Flash's indexer shapes (32 heads per device, head_dim 128, 2048
candidates, a shuffled page table of 32 slots a page) and hashes them, so two
builds of the kernel can be compared bit for bit. Up to 64 rows each row is its
own request (decode); 2048 rows are one request's prefill chunk. Head counts
that do not split into eighths or whole staged query loads take the fallback
paths; a launch capped to 3 column blocks folds the rest onto them, as grid y's
limit does at a 1M context."""

from std.math import abs, ceildiv, max
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.attention.indexer_score import (
    MAX_GRID_Y,
    _indexer_score_gpu,
    indexer_score_grid,
    indexer_score_ragged_paged,
)
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal, assert_true

comptime N_CAND = 2048
comptime CAP = N_CAND // 2
comptime PAGE = 32


def _fnv1a(ptr: Pointer[UInt8, _], n: Int) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        h = (h ^ ptr[i].cast[DType.uint64]()) * 0x100000001B3
    return h


def run[
    kv_type: DType, H: Int = 32, D: Int = 128, max_col_blocks: Int = 0
](ctx: DeviceContext, m: Int) raises -> UInt64:
    """Returns the scores' hash; ``max_col_blocks`` caps grid y when set."""
    seed(m * 31 + 7)
    var batch = m if m <= 64 else 1
    var rows_per = m // batch
    var pages_per = ceildiv(CAP + rows_per + 2, PAGE)
    var total_pages = batch * pages_per

    var offsets_h = unsafe_alloc[UInt32](batch + 1)
    var base_h = unsafe_alloc[Int32](batch)
    var cutoff_h = unsafe_alloc[Int32](m)
    var lens_h = unsafe_alloc[UInt32](batch)
    var lut_h = unsafe_alloc[UInt32](total_pages)
    for b in range(batch + 1):
        offsets_h[b] = UInt32(b * rows_per)
    for b in range(batch):
        # Closed entries anywhere from none to every candidate column.
        var closed = Int(random_ui64(0, CAP))
        base_h[b] = Int32(closed)
        lens_h[b] = UInt32(closed * 4)
        for r in range(rows_per):
            cutoff_h[b * rows_per + r] = Int32(
                closed + Int(random_ui64(0, 2)) + r // 4
            )
        for p in range(pages_per):
            lut_h[b * pages_per + p] = UInt32(
                ((b * pages_per + p) * 7919) % total_pages
            )

    var q_h = unsafe_alloc[BFloat16](m * H * D)
    var w_h = unsafe_alloc[Float32](m * H)
    var kv_elems = total_pages * PAGE * D
    var kv_h = unsafe_alloc[Scalar[kv_type]](kv_elems)
    random(
        TileTensor(q_h, row_major(Int64(m * H * D))), BFloat16(-1), BFloat16(1)
    )
    random(TileTensor(w_h, row_major(Int64(m * H))), Float32(0), Float32(1))
    random(
        TileTensor(kv_h, row_major(Int64(kv_elems))),
        Scalar[kv_type](-1),
        Scalar[kv_type](1),
    )

    var q_d = ctx.enqueue_create_buffer[DType.bfloat16](m * H * D)
    var w_d = ctx.enqueue_create_buffer[DType.float32](m * H)
    var kv_d = ctx.enqueue_create_buffer[kv_type](kv_elems)
    var offsets_d = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var base_d = ctx.enqueue_create_buffer[DType.int32](batch)
    var cutoff_d = ctx.enqueue_create_buffer[DType.int32](m)
    var lens_d = ctx.enqueue_create_buffer[DType.uint32](batch)
    var lut_d = ctx.enqueue_create_buffer[DType.uint32](total_pages)
    var out_d = ctx.enqueue_create_buffer[DType.float32](m * N_CAND)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(w_d, w_h)
    ctx.enqueue_copy(kv_d, kv_h)
    ctx.enqueue_copy(offsets_d, offsets_h)
    ctx.enqueue_copy(base_d, base_h)
    ctx.enqueue_copy(cutoff_d, cutoff_h)
    ctx.enqueue_copy(lens_d, lens_h)
    ctx.enqueue_copy(lut_d, lut_h)

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=D, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_tt = TileTensor(
        kv_d,
        row_major(
            Int64(total_pages), Idx[1], Int64(1), Idx[PAGE], Idx[1], Idx[D]
        ),
    )
    var lens_tt = TileTensor(lens_d, row_major(Int64(batch)))
    var lut_tt = TileTensor(lut_d, row_major(Int64(batch), Int64(pages_per)))
    var coll = Collection(
        rebind[Collection.blocks_tt_type](kv_tt.as_unsafe_any_origin()),
        lens_tt.as_imm().as_unsafe_any_origin(),
        lut_tt.as_imm().as_unsafe_any_origin(),
        UInt32(rows_per),
        UInt32(CAP * 4),
    )

    var out_tt = TileTensor(out_d, row_major(Int64(m), Idx[N_CAND]))
    var q_tt = TileTensor(q_d, row_major(Int64(m), Idx[H], Idx[D])).as_imm()
    var w_tt = TileTensor(w_d, row_major(Int64(m), Idx[H])).as_imm()
    var offsets_tt = TileTensor(offsets_d, row_major(Int64(batch + 1))).as_imm()
    var base_tt = TileTensor(base_d, row_major(Int64(batch))).as_imm()
    var cutoff_tt = TileTensor(cutoff_d, row_major(Int64(m))).as_imm()
    comptime if max_col_blocks == 0:
        indexer_score_ragged_paged[num_heads=H](
            out_tt.address_space_cast[.GENERIC](),
            q_tt,
            w_tt,
            offsets_tt,
            base_tt,
            cutoff_tt,
            coll.get_key_cache(0),
            ctx,
        )
    else:
        comptime assert H % 8 == 0, "a capped launch runs both GPU paths"
        if m <= 4:
            _indexer_score_gpu[
                num_heads=H,
                head_dim=D,
                chunk=8,
                block_size=64,
                lanes=8,
                max_col_blocks=max_col_blocks,
            ](
                out_tt.address_space_cast[.GENERIC](),
                q_tt,
                w_tt,
                offsets_tt,
                base_tt,
                cutoff_tt,
                coll.get_key_cache(0),
                m,
                batch,
                N_CAND,
                ctx,
            )
        else:
            _indexer_score_gpu[
                num_heads=H,
                head_dim=D,
                chunk=8,
                block_size=128,
                lanes=1,
                max_col_blocks=max_col_blocks,
            ](
                out_tt.address_space_cast[.GENERIC](),
                q_tt,
                w_tt,
                offsets_tt,
                base_tt,
                cutoff_tt,
                coll.get_key_cache(0),
                m,
                batch,
                N_CAND,
                ctx,
            )
    var out_h = unsafe_alloc[Float32](m * N_CAND)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # Every column of the checked rows against fp64: dead columns are 0 and
    # live ones sum each head's clamped dot product with its weight.
    var bad = 0
    var checked = 0
    var row_step = 1 if m <= 64 else 97
    for t in range(0, m, row_step):
        var b = t // rows_per
        var base_b = Int(base_h[b])
        var cutoff_t = Int(cutoff_h[t])
        for c in range(N_CAND):
            var e = c if c < CAP else base_b + c - CAP
            var live = c < base_b if c < CAP else e < cutoff_t
            var ref_v = Float64(0)
            var mag = Float64(0)
            if live:
                var page = Int(lut_h[b * pages_per + e // PAGE])
                var off = (page * PAGE + e % PAGE) * D
                for h in range(H):
                    var acc = Float64(0)
                    var acc_mag = Float64(0)
                    for d in range(D):
                        var term = Float64(q_h[(t * H + h) * D + d]) * Float64(
                            kv_h[off + d]
                        )
                        acc += term
                        acc_mag += abs(term)
                    var wh = Float64(w_h[t * H + h])
                    ref_v += max(acc, 0) * wh
                    mag += acc_mag * wh
            var got = Float64(out_h[t * N_CAND + c])
            checked += 1
            if not (abs(got - ref_v) <= mag * 1e-5 + 1e-6):
                bad += 1
    print(
        t"indexer_score kv= {kv_type} H= {H} D= {D} max_col_blocks="
        t" {max_col_blocks} M= {m} checked= {checked} bad= {bad}"
    )
    assert_true(bad == 0, "indexer_score differs from the fp64 reference")

    var digest = _fnv1a(out_h.unsafe_bitcast[UInt8](), m * N_CAND * 4)
    if H == 32 and D == 128 and max_col_blocks == 0:
        print(
            "indexer_score kv=",
            kv_type,
            " M=",
            m,
            " scores=",
            hex(digest),
            sep="",
        )
    return digest


def check_grid_at_1m_context() raises:
    """At max_seq_len 1048576 (ratio 4) neither path's grid y passes the cap."""
    var num_cand = 2 * ceildiv(1048576, 4)
    for cols in [8, 128]:
        var grid = indexer_score_grid(4, num_cand, cols)
        assert_equal(grid[0], 4)
        assert_true(grid[1] <= MAX_GRID_Y, "grid y passes the launch limit")
        assert_equal(grid[1], min(ceildiv(num_cand, cols), MAX_GRID_Y))
    # The 8-lane path's 65536 tiles fold one onto the first block.
    assert_equal(ceildiv(num_cand, 8), MAX_GRID_Y + 1)


def main() raises:
    check_grid_at_1m_context()
    with DeviceContext() as ctx:
        for m in [1, 2, 4, 8, 16, 32, 64, 2048]:
            _ = run[DType.bfloat16](ctx, m)
            _ = run[DType.float8_e4m3fn](ctx, m)
        # 4 heads take neither the spread nor the staged load; 8 heads of 64
        # take the spread's staged load but the 1-lane path's strided one.
        for m in [1, 4, 8]:
            _ = run[DType.bfloat16, H=4](ctx, m)
            _ = run[DType.bfloat16, H=8, D=64](ctx, m)
        # Folded tiles score exactly as unfolded ones.
        for m in [1, 4, 8]:
            assert_equal(
                run[DType.bfloat16, max_col_blocks=3](ctx, m),
                run[DType.bfloat16](ctx, m),
            )
