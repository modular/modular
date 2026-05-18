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
"""HKMhaPrefill correctness test that ACTUALLY exercises rescaling.

The first v2 test used Q=K=1 and a softly-varying V — uniform att,
running max never grows, so the rescale path through `mul_col` /
`exp2(max_prev - max)` was effectively a no-op.

This test makes the per-tile max GROW so the rescaling path fires:

  Q[i, d] = 1
  K[t][r, d] = 1 in tile 0; 2 in tile t>0    (BF16 exact)
  V[t][r, m] = (t*KV + r) / 32                (small monotone)

  att score[k, j] = sum_d K[k, d] * Q[j, d] = DEPTH * K_val_for_k
    -> tile 0 scores = DEPTH = 128
    -> tile 1 scores = 2 * DEPTH = 256

The strict online-softmax ground truth would be `mean(V[KV..2*KV-1, m])`
because exp2(scale*(max_prev - max_new)) ≈ 0 zeroes out tile 0.

HK's C2/C6 cluster interleaves PV strip 0 with the lazy rescale of
o_reg, then runs strips 1..3 AFTER the rescale while att_bf16 is
still at the OLD max scale. Tile 0's strips 1..3 thus survive the
rescale at their old scale, contributing a bounded artifact
`mean(V[KV/4..KV-1, m])` to the output. HKMhaPrefill reproduces this
approximation faithfully. At RESCALE_THRESHOLD=8 the magnitude is
bounded; in this extreme test (∆max ≈ 128) the artifact is large but
consistent.
"""

from std.gpu.host import DeviceContext
from std.testing import assert_almost_equal

from layout import TileTensor
from layout.coord import Coord, Idx, RuntimeInt
from layout.tile_layout import row_major

from nn.attention.gpu.amd_structured.hk_mha_prefill import (
    HKMhaConfig,
    hk_mha_prefill,
)


comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE
comptime KV_BLOCK = 64
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime NUM_TILES = 8  # HK kernel.cpp prologue requires >= 4
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK
comptime BATCH = 1


def test_v2_rescale[depth: Int](ctx: DeviceContext) raises:
    comptime SIZE_Q = BM * depth
    comptime SIZE_KV = NUM_TILES * KV_BLOCK * depth
    comptime SIZE_OUT = BM * depth

    comptime CONFIG = HKMhaConfig(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
        causal=False,
    )

    print(
        "--- HKMhaPrefill rescale path (tile-1 max > tile-0 max, D=",
        depth,
        ") ---",
    )

    var dev_q = ctx.enqueue_create_buffer[DType.bfloat16](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[DType.bfloat16](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[DType.bfloat16](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[DType.float32](SIZE_OUT)
    var dev_lvec = ctx.enqueue_create_buffer[DType.float32](BM)

    with dev_q.map_to_host() as host_q, dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for i in range(SIZE_Q):
            host_q[i] = BFloat16(1)
        # K[t=0] = 1, K[t>=1] = 2 (uniform within each tile).
        for t in range(NUM_TILES):
            var k_val = BFloat16(1) if t == 0 else BFloat16(2)
            for r in range(KV_BLOCK):
                for d in range(depth):
                    host_k[(t * KV_BLOCK + r) * depth + d] = k_val
        # V[t][r, m] = (t * KV + r) / 32.
        for t in range(NUM_TILES):
            for r in range(KV_BLOCK):
                var v_val = Float32(t * KV_BLOCK + r) / Float32(32)
                for m in range(depth):
                    host_v[(t * KV_BLOCK + r) * depth + m] = BFloat16(v_val)

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(
                RuntimeInt[DType.int32](Int32(BATCH)),
                RuntimeInt[DType.int32](Int32(SEQ_LEN)),
                Idx[NUM_HEADS](),
                Idx[depth](),
            )
        ),
    )
    var k_tt = TileTensor(
        dev_k,
        row_major(
            Coord(
                RuntimeInt[DType.int32](Int32(BATCH)),
                RuntimeInt[DType.int32](Int32(NUM_KEYS)),
                Idx[NUM_KV_HEADS](),
                Idx[depth](),
            )
        ),
    )
    var v_tt = TileTensor(
        dev_v,
        row_major(
            Coord(
                RuntimeInt[DType.int32](Int32(BATCH)),
                RuntimeInt[DType.int32](Int32(NUM_KEYS)),
                Idx[NUM_KV_HEADS](),
                Idx[depth](),
            )
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(
                RuntimeInt[DType.int32](Int32(BATCH)),
                RuntimeInt[DType.int32](Int32(SEQ_LEN)),
                Idx[NUM_HEADS](),
                Idx[depth](),
            )
        ),
    )
    var l_vec_tt = TileTensor(
        dev_lvec,
        row_major(
            Coord(
                RuntimeInt[DType.int32](Int32(BATCH)),
                Idx[NUM_HEADS](),
                RuntimeInt[DType.int32](Int32(SEQ_LEN)),
            )
        ),
    )

    hk_mha_prefill[CONFIG](q_tt, k_tt, v_tt, o_tt, l_vec_tt, Float32(1.0), ctx)

    # Expected per-lane (HK approximation + epilogue norm dynamics).
    #
    # Numerator (o_reg before div_col): tile 0's PV strips 1..3 survive
    # the rescale-mid-PV at the OLD max scale = V[KV/4..KV-1].
    # Tiles 1..NUM_TILES-1 add at the new max scale = V[KV..NUM_TILES*KV-1].
    #
    # Denominator (norm_vec): the epilogue's UNCONDITIONAL `mul(norm_vec,
    # norm_vec, scale_vec)` at every tail step zeroes the accumulated
    # tiles-1..(N-5) norm against the stale post-rescale `scale_vec`.
    # Only the final 4 tiles (N-4..N-1) contribute — denominator =
    # 4 * KV_BLOCK.
    var sum_strips_1_3: Float32 = 0
    for r in range(KV_BLOCK // 4, KV_BLOCK):
        sum_strips_1_3 += Float32(r) / Float32(32)
    var sum_tiles_1_n: Float32 = 0
    for r in range(KV_BLOCK, NUM_TILES * KV_BLOCK):
        sum_tiles_1_n += Float32(r) / Float32(32)
    var expected: Float32 = (sum_strips_1_3 + sum_tiles_1_n) / Float32(
        4 * KV_BLOCK
    )
    print("  expected per-lane (HK approx) =", expected)

    var mismatches = 0
    var max_diff: Float32 = 0
    # Tolerance: BF16 noise + bounded HK rescale-mid-PV approximation
    # noise. Scaled to ~3% of expected magnitude.
    var TOL: Float32 = 0.15
    with dev_out.map_to_host() as host_out:
        for q in range(BM):
            for d in range(depth):
                var got = host_out[q * depth + d]
                var diff = abs(got - expected)
                if diff > max_diff:
                    max_diff = diff
                if diff > TOL:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            "MISMATCH q=",
                            q,
                            " d=",
                            d,
                            " got=",
                            got,
                            " expected=",
                            expected,
                        )

    print("  mismatches=", mismatches, " max_diff=", max_diff)
    assert_almost_equal(Float32(mismatches), Float32(0))
    print("  PASSED")


def main() raises:
    print("=" * 60)
    print("HKMhaPrefill rescale-path GPU test")
    print("=" * 60)

    with DeviceContext() as ctx:
        test_v2_rescale[128](ctx)
        test_v2_rescale[64](ctx)

    print("=" * 60)
    print("ALL HKMhaPrefill RESCALE TESTS PASSED!")
    print("=" * 60)
