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
"""The SM100 FP8 indexer's K-streaming kernel must score each key tile with
that tile's own k-scales.

The consumers read a tile's scales out of its K ring slot with a plain shared
load, then release the slot, and the load warp refills it by TMA. The refill
is an async-proxy write, so unless the load warp fences after its wait it can
land the slot's NEXT tile before a consumer's read returns: 128 keys are then
scored with the scales of the tile one ring depth later. It takes many
refills on a loaded SM, which is why short shapes and top-k comparisons never
see it.

The oracle needs no reference kernel. Scored with every k-scale 1.0, a slot's
value is the unscaled head sum no matter which ring slot the scale came from.
The kernel stores `k_scale * sum`, so scored with real scales every live slot
must equal `ks[key] * ones[slot]` bit for bit. The scales make a stale read
loud: a key in tile `t` of its entry gets `(-1)^t * (1 + (t % 16) / 4 + u/5)`,
so tiles up to 15 apart differ in magnitude and an odd ring depth also flips
the score's sign. A failing slot's scale is matched to the same lane of a
nearby tile, and the report histograms that tile offset.
"""

from std.math import align_up
from std.random import rand, random_float64, seed

from max.gpu.host import DeviceContext
from kv_cache.types import create_flat_kv_tma_tile
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_tensor import ImmTileTensor, MutTileTensor
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    _INDEX_SWIZZLE,
    KTMATileT,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.mha_operand import MHAOperand, RaggedMHAOperand
from std.sys.info import _has_blackwell_tcgen05
from std.testing import assert_equal

comptime NUM_HEADS = 32
comptime DEPTH = 128
comptime FILL = Float32(-7.0)


def _score[
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    vl_layout: TensorLayout,
    KOp: MHAOperand,
    KSOp: MHAOperand,
](
    output: MutTileTensor[.float32, output_layout, _],
    q: ImmTileTensor[.float8_e4m3fn, q_layout, _],
    q_s: ImmTileTensor[.float32, qs_layout, _],
    k_op: KOp,
    ks_op: KSOp,
    k_tma: KTMATileT[DType.float8_e4m3fn, _BM_KEY, DEPTH],
    valid_length: ImmTileTensor[.uint32, vl_layout, _],
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
) raises:
    fp8_index_score_sm100[
        DType.float8_e4m3fn,
        KOp,
        KSOp,
        NUM_HEADS,
        DEPTH,
        _is_cache_length_accurate=True,
        N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
    ](
        output,
        q,
        q_s,
        k_op,
        ks_op,
        k_tma,
        valid_length,
        batch_size,
        seq_len,
        num_keys,
        True,
        ctx,
    )


def _tile_scale(tile: Int) -> Float32:
    var sign = Float32(1) if tile % 2 == 0 else Float32(-1)
    return sign * (
        1.0 + Float32(tile % 16) * 0.25 + Float32(random_float64()) * 0.2
    )


def test_kscale_war(
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    runs: Int,
    ctx: DeviceContext,
) raises:
    var num_keys = cache_len + seq_len
    var total_q = batch_size * seq_len
    var total_k = batch_size * num_keys
    var n_out = total_q * num_keys
    print(
        "test_kscale_war batch",
        batch_size,
        "seq_len",
        seq_len,
        "cache_len",
        cache_len,
        "runs",
        runs,
    )

    var q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        total_q * NUM_HEADS * DEPTH
    )
    var qs_h = ctx.enqueue_create_host_buffer[.float32](total_q * NUM_HEADS)
    var k_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](total_k * DEPTH)
    var ks_h = ctx.enqueue_create_host_buffer[.float32](align_up(total_k, 4))
    var one_h = ctx.enqueue_create_host_buffer[.float32](align_up(total_k, 4))
    var vl_h = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var cro_h = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var unscaled_h = ctx.enqueue_create_host_buffer[.float32](n_out)
    var got_h = ctx.enqueue_create_host_buffer[.float32](n_out)
    ctx.synchronize()
    rand(q_h.as_span())
    rand(qs_h.as_span())
    rand(k_h.as_span())
    for i in range(align_up(total_k, 4)):
        ks_h[i] = 1.0
        one_h[i] = 1.0
    for b in range(batch_size):
        for k in range(num_keys):
            ks_h[b * num_keys + k] = _tile_scale(k // _BM_KEY)
    for b in range(batch_size + 1):
        vl_h[b] = UInt32(b * seq_len)
        cro_h[b] = UInt32(b * num_keys)

    var q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
        total_q * NUM_HEADS * DEPTH
    )
    var qs_d = ctx.enqueue_create_buffer[.float32](total_q * NUM_HEADS)
    var k_d = ctx.enqueue_create_buffer[.float8_e4m3fn](total_k * DEPTH)
    var ks_d = ctx.enqueue_create_buffer[.float32](align_up(total_k, 4))
    var one_d = ctx.enqueue_create_buffer[.float32](align_up(total_k, 4))
    var vl_d = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cro_d = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var o_d = ctx.enqueue_create_buffer[.float32](n_out)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(qs_d, qs_h)
    ctx.enqueue_copy(k_d, k_h)
    ctx.enqueue_copy(ks_d, ks_h)
    ctx.enqueue_copy(one_d, one_h)
    ctx.enqueue_copy(vl_d, vl_h)
    ctx.enqueue_copy(cro_d, cro_h)

    var q_t = TileTensor[mut=False](
        q_d.unsafe_ptr(), row_major((total_q, Idx[NUM_HEADS], Idx[DEPTH]))
    )
    var qs_t = TileTensor[mut=False](
        qs_d.unsafe_ptr(), row_major((total_q, Idx[NUM_HEADS]))
    )
    var vl_t = TileTensor[mut=False](
        vl_d.unsafe_ptr(), row_major(batch_size + 1)
    )
    var o_t = TileTensor(o_d.unsafe_ptr(), row_major((total_q, num_keys)))
    var cro_t = TileTensor[mut=False](
        cro_d.unsafe_ptr(), row_major(Coord(batch_size + 1))
    )
    var k_op = RaggedMHAOperand(
        TileTensor[mut=False](
            k_d.unsafe_ptr(), row_major(Coord(total_k, Idx[1], Idx[DEPTH]))
        ),
        cro_t,
    )
    var ks_op = RaggedMHAOperand(
        TileTensor[mut=False](
            ks_d.unsafe_ptr(), row_major(Coord(total_k, Idx[1], Idx[1]))
        ),
        cro_t,
    )
    var one_op = RaggedMHAOperand(
        TileTensor[mut=False](
            one_d.unsafe_ptr(), row_major(Coord(total_k, Idx[1], Idx[1]))
        ),
        cro_t,
    )
    var k_tma = rebind[KTMATileT[DType.float8_e4m3fn, _BM_KEY, DEPTH]](
        create_flat_kv_tma_tile[
            BN=_BM_KEY, BK=DEPTH, swizzle_mode=_INDEX_SWIZZLE
        ](
            ctx,
            rebind[Pointer[Float8_e4m3fn, ImmutAnyOrigin]](k_d.unsafe_ptr()),
            total_k,
            1,
            DEPTH,
        )
    )

    o_d.enqueue_fill(FILL)
    _score(
        o_t,
        q_t,
        qs_t,
        k_op,
        one_op,
        k_tma,
        vl_t,
        batch_size,
        seq_len,
        num_keys,
        ctx,
    )
    ctx.enqueue_copy(unscaled_h, o_d)
    ctx.synchronize()

    var bad_total = 0
    for run in range(runs):
        o_d.enqueue_fill(FILL)
        _score(
            o_t,
            q_t,
            qs_t,
            k_op,
            ks_op,
            k_tma,
            vl_t,
            batch_size,
            seq_len,
            num_keys,
            ctx,
        )
        ctx.enqueue_copy(got_h, o_d)
        ctx.synchronize()
        var bad_live = 0
        var bad_dead = 0
        # tile offsets -8..8 at 0..16; 17 = no same-lane match within 8 tiles
        var by_offset = List[Int](length=18, fill=0)
        var shown = 0
        for b in range(batch_size):
            for t in range(seq_len):
                var bound = num_keys - (seq_len - 1 - t)
                var row = (b * seq_len + t) * num_keys
                for k in range(num_keys):
                    var g = got_h[row + k]
                    var u = unscaled_h[row + k]
                    if k >= bound:
                        if g != FILL or u != FILL:
                            bad_dead += 1
                        continue
                    var own = ks_h[b * num_keys + k]
                    var want = own * u
                    if abs(g - want) <= 1e-6 * abs(want):
                        continue
                    bad_live += 1
                    var used = g / u
                    var slot = 17
                    for d in range(-8, 9):
                        var kk = k + d * _BM_KEY
                        if d == 0 or kk < 0 or kk >= num_keys:
                            continue
                        var s = ks_h[b * num_keys + kk]
                        if abs(s - used) <= 1e-4 * abs(s):
                            slot = d + 8
                            break
                    by_offset[slot] += 1
                    if shown < 4:
                        shown += 1
                        print(
                            "    b",
                            b,
                            "tok",
                            t,
                            "key",
                            k,
                            "got",
                            g,
                            "want",
                            want,
                            "| scale used",
                            used,
                            "own",
                            own,
                        )
        var hist = String()
        for i in range(17):
            if by_offset[i] > 0:
                hist += String(" tile", i - 8, ":", by_offset[i])
        if by_offset[17] > 0:
            hist += String(" unmatched:", by_offset[17])
        print(
            "  run",
            run,
            "live slots off",
            bad_live,
            "dead slots written",
            bad_dead,
            "| scale taken from" if bad_live > 0 else "",
            hist,
        )
        bad_total += bad_live + bad_dead
    assert_equal(bad_total, 0, "slots scored with another tile's k-scales")

    _ = q_d^
    _ = qs_d^
    _ = k_d^
    _ = ks_d^
    _ = one_d^
    _ = vl_d^
    _ = cro_d^
    _ = o_d^


def main() raises:
    comptime if not _has_blackwell_tcgen05():
        return
    seed(7)
    with DeviceContext() as ctx:
        test_kscale_war(4, 1024, 4096, 4, ctx)
        test_kscale_war(1, 8192, 0, 4, ctx)
