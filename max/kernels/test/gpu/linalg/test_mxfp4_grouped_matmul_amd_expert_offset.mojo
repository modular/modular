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
"""Regression test for the grouped MXFP4 matmul's 64-bit expert offsets.

The grouped kernels locate expert `e`'s packed weights at `e * N * K_BYTES`
bytes, which wraps in 32 bits once the stack passes 2 GiB; the wrapped pointer
reads zeros or another allocation, or faults, but never errors. This covers
`block_scaled_grouped_matmul_amd` and, on MI355X, the persistent and direct
`PreShuffledBGroupedGEMM` kernels.

The shape is Kimi K3's routed gate/up shard at four devices (`N=1536,
K=3584`), where expert 781 is the first past the bound. Experts 781 and 799
run next to in-bounds expert 100, the control that keeps a pass from being
vacuous. Outputs are compared by value against a per-expert reference that
reads compact copies of the routed weights, never the offset under test.

Usage:
  ./bazelw test //max/kernels/test/gpu/linalg:test_mxfp4_grouped_matmul_amd_expert_offset.mojo.test
"""

from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from std.math import align_up
from std.memory import bitcast
from std.random import random_ui64, seed

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import (
    PreShuffledBGroupedGEMM,
    Shuffler,
    block_scaled_matmul_amd,
    block_scaled_grouped_matmul_amd,
)

# K3's routed expert geometry at a tensor-parallel degree of four.
comptime N = 1536
comptime K = 3584
comptime PACKED_K = K // 2
comptime SCALE_K = K // MXFP4_SF_VECTOR_SIZE
comptime BYTES_PER_EXPERT = N * PACKED_K
comptime SCALES_PER_EXPERT = N * SCALE_K

# Enough experts to put the last few slices past the 32-bit bound.
comptime NUM_EXPERTS = 800

# The E8M0 byte range every other MXFP4 test uses: magnitudes 0.25 to 4, which
# keeps the f32 accumulator in range while still exercising scale dequant. 255
# is deliberately out of reach -- it encodes E8M0 NaN.
comptime SCALE_LO = 125
comptime SCALE_HI = 129


def test_expert_offset_beyond_int32[
    tokens_per_expert: Int,
    preshuffled: Bool = False,
    persistent: Bool = False,
](expert_ids_list: List[Int], ctx: DeviceContext) raises:
    """Routes one in-bounds and two past-2^31 experts through one grouped call.

    Parameters:
        tokens_per_expert: Rows each routed expert receives, which also sizes
            the M grid and so selects the tile config.
        preshuffled: Runs `PreShuffledBGroupedGEMM` on a preshuffled stack
            instead of `block_scaled_grouped_matmul_amd` on a row-major one.
        persistent: Selects the persistent preshuffled kernel over the direct
            one. Only meaningful with `preshuffled`.

    Args:
        expert_ids_list: Expert ids to route, at least one below and one at or
            above the first id whose byte offset exceeds `Int32.MAX`.
        ctx: Device context.
    """
    comptime assert (
        preshuffled or not persistent
    ), "persistent selects a preshuffled-B kernel"

    var num_active = len(expert_ids_list)
    var total_tokens = num_active * tokens_per_expert

    comptime path = (
        "preshuffled persistent" if persistent else (
            "preshuffled direct" if preshuffled else "row-major"
        )
    )
    print(
        "  expert-offset regression (",
        path,
        "): experts=",
        NUM_EXPERTS,
        " N=",
        N,
        " K=",
        K,
        " tokens/expert=",
        tokens_per_expert,
        " routed=",
        num_active,
    )

    # --- The expert stack: memset, then only the routed slices written ---
    var b_dev = ctx.enqueue_create_buffer[.uint8](
        NUM_EXPERTS * BYTES_PER_EXPERT
    )
    # Scales are held as raw bytes so the preshuffled path can write shuffled
    # ones; every view handed to a kernel bitcasts them back to E8M0.
    var b_scales_dev = ctx.enqueue_create_buffer[.uint8](
        NUM_EXPERTS * SCALES_PER_EXPERT
    )
    # Zeroed weights make an unrouted expert's contribution exactly zero, so a
    # read of the wrong slice shows up as a mismatch rather than as noise. A
    # uniform scale fill is its own preshuffle, so it serves both layouts.
    ctx.enqueue_memset(b_dev, UInt8(0))
    ctx.enqueue_memset(b_scales_dev, UInt8(SCALE_LO + 2))

    var b_routed_dev = ctx.enqueue_create_buffer[.uint8](
        num_active * BYTES_PER_EXPERT
    )
    var b_scales_routed_dev = ctx.enqueue_create_buffer[.uint8](
        num_active * SCALES_PER_EXPERT
    )

    var b_slice_host = ctx.enqueue_create_host_buffer[.uint8](BYTES_PER_EXPERT)
    var b_scale_slice_host = ctx.enqueue_create_host_buffer[.uint8](
        SCALES_PER_EXPERT
    )
    var b_scale_pre_host = ctx.enqueue_create_host_buffer[.uint8](
        SCALES_PER_EXPERT
    )

    for slot in range(num_active):
        var expert_id = expert_ids_list[slot]
        # Independently random per slice, so reading the wrong expert cannot
        # coincide with reading the right one.
        for i in range(BYTES_PER_EXPERT):
            b_slice_host[i] = UInt8(random_ui64(0, 255))
        for i in range(SCALES_PER_EXPERT):
            b_scale_slice_host[i] = UInt8(random_ui64(SCALE_LO, SCALE_HI))

        var b_routed_sub = b_routed_dev.create_sub_buffer[.uint8](
            slot * BYTES_PER_EXPERT, BYTES_PER_EXPERT
        )
        var b_scale_routed_sub = b_scales_routed_dev.create_sub_buffer[.uint8](
            slot * SCALES_PER_EXPERT, SCALES_PER_EXPERT
        )
        ctx.enqueue_copy(b_routed_sub, b_slice_host)
        ctx.enqueue_copy(b_scale_routed_sub, b_scale_slice_host)

        # Placed with 64-bit host math, where a correct kernel reads it.
        var b_sub = b_dev.create_sub_buffer[.uint8](
            expert_id * BYTES_PER_EXPERT, BYTES_PER_EXPERT
        )
        var b_scale_sub = b_scales_dev.create_sub_buffer[.uint8](
            expert_id * SCALES_PER_EXPERT, SCALES_PER_EXPERT
        )
        comptime if preshuffled:
            Shuffler[1].preshuffle_b_5d[N=N, K_BYTES=PACKED_K](
                TileTensor(b_routed_sub, row_major[1, N, PACKED_K]()).as_imm(),
                TileTensor(
                    b_sub,
                    Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=PACKED_K],
                ),
                ctx,
            )
            Shuffler[1].preshuffle_scale_4d[MN=N, K_SCALES=SCALE_K](
                TileTensor(
                    b_scale_slice_host.unsafe_ptr(),
                    row_major(Coord(Idx[1], Idx[N], Idx[SCALE_K])),
                ),
                b_scale_pre_host,
            )
            ctx.enqueue_copy(b_scale_sub, b_scale_pre_host)
        else:
            ctx.enqueue_copy(b_sub, b_routed_sub)
            ctx.enqueue_copy(b_scale_sub, b_scale_slice_host)
        ctx.synchronize()

    # --- Activations and routing ---
    var a_host = ctx.enqueue_create_host_buffer[.uint8](total_tokens * PACKED_K)
    var a_scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        total_tokens * SCALE_K
    )
    for i in range(total_tokens * PACKED_K):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(total_tokens * SCALE_K):
        a_scales_host[i] = bitcast[.float8_e8m0fnu](
            UInt8(random_ui64(SCALE_LO, SCALE_HI))
        )

    var a_offsets_host = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var expert_ids_host = ctx.enqueue_create_host_buffer[.int32](num_active)
    a_offsets_host[0] = UInt32(0)
    for slot in range(num_active):
        a_offsets_host[slot + 1] = a_offsets_host[slot] + UInt32(
            tokens_per_expert
        )
        expert_ids_host[slot] = Int32(expert_ids_list[slot])

    var a_dev = ctx.enqueue_create_buffer[.uint8](total_tokens * PACKED_K)
    var a_scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * SCALE_K
    )
    var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](num_active)
    var c_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_ref_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    ctx.enqueue_copy(expert_ids_dev, expert_ids_host)

    # The preshuffled kernels read A-scales from one fixed-stride slot per
    # routed expert rather than from the token rows.
    var max_padded_M = align_up(tokens_per_expert, 32)
    var a_scales_pre_dev = ctx.enqueue_create_buffer[.uint8](
        num_active * max_padded_M * SCALE_K
    )
    comptime if preshuffled:
        Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=SCALE_K](
            TileTensor(
                a_scales_dev.unsafe_ptr().bitcast[UInt8](),
                row_major(Coord(total_tokens, Idx[SCALE_K])),
            ).as_imm(),
            TileTensor(
                a_scales_pre_dev,
                row_major(Coord(num_active * max_padded_M, Idx[SCALE_K])),
            ),
            TileTensor(
                a_offsets_dev, row_major(Coord(num_active + 1))
            ).as_imm(),
            num_active,
            tokens_per_expert,
            ctx.default_device_info.sm_count * 2,
            ctx,
        )

    # --- Reference: one ungrouped matmul per routed expert ---
    for slot in range(num_active):
        var token_start = slot * tokens_per_expert

        var a_expert_tt = TileTensor(
            a_dev.unsafe_ptr() + token_start * PACKED_K,
            row_major(Coord(tokens_per_expert, Idx[PACKED_K])),
        ).as_imm()
        var b_expert_tt = TileTensor(
            b_routed_dev.unsafe_ptr() + slot * BYTES_PER_EXPERT,
            row_major[N, PACKED_K](),
        ).as_imm()
        var sfa_expert_tt = TileTensor(
            a_scales_dev.unsafe_ptr() + token_start * SCALE_K,
            row_major(Coord(tokens_per_expert, Idx[SCALE_K])),
        ).as_imm()
        var sfb_expert_tt = TileTensor(
            b_scales_routed_dev.unsafe_ptr().bitcast[Float8_e8m0fnu]()
            + slot * SCALES_PER_EXPERT,
            row_major[N, SCALE_K](),
        ).as_imm()
        var c_expert_tt = TileTensor(
            c_ref_dev.unsafe_ptr() + token_start * N,
            row_major(Coord(tokens_per_expert, Idx[N])),
        )

        block_scaled_matmul_amd(
            c_expert_tt,
            a_expert_tt,
            b_expert_tt,
            sfa_expert_tt,
            sfb_expert_tt,
            ctx,
        )
    ctx.synchronize()

    # --- The grouped kernel under test ---
    var a_tt = TileTensor(
        a_dev, row_major(Coord(total_tokens, Idx[PACKED_K]))
    ).as_imm()
    var b_scales_tt = TileTensor(
        b_scales_dev.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[NUM_EXPERTS, N, SCALE_K](),
    ).as_imm()
    var a_offsets_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active + 1))
    )
    var expert_ids_tt = TileTensor(expert_ids_dev, row_major(Coord(num_active)))
    var c_tt = TileTensor(c_dev, row_major(Coord(total_tokens, Idx[N])))

    comptime if preshuffled:
        var b_pre_tt = TileTensor(
            b_dev, row_major[NUM_EXPERTS, N * PACKED_K]()
        ).as_imm()
        var a_scales_pre_tt = TileTensor(
            a_scales_pre_dev.unsafe_ptr().bitcast[Float8_e8m0fnu](),
            row_major(Coord(num_active * max_padded_M, Idx[SCALE_K])),
        ).as_imm()
        # The dispatcher's fallback tile for this shape, launched directly so
        # that a tuned band added for it later cannot move coverage off either
        # kernel.
        comptime GEMM = PreShuffledBGroupedGEMM[
            cu_count=ctx.default_device_info.sm_count
        ]
        GEMM.launch[BM=64, BN=128, BK_ELEMS=512, WN=64, persistent=persistent](
            c_tt,
            a_tt,
            b_pre_tt,
            a_scales_pre_tt,
            b_scales_tt,
            a_offsets_tt,
            expert_ids_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
    else:
        var b_tt = TileTensor(
            b_dev, row_major[NUM_EXPERTS, N, PACKED_K]()
        ).as_imm()
        var a_scales_tt = TileTensor(
            a_scales_dev, row_major(Coord(total_tokens, Idx[SCALE_K]))
        ).as_imm()
        block_scaled_grouped_matmul_amd(
            c_tt,
            a_tt,
            b_tt,
            a_scales_tt,
            b_scales_tt,
            a_offsets_tt,
            expert_ids_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        total_tokens * N,
        atol=0.05,
        rtol=0.05,
    )

    print("    PASS")

    _ = a_dev^
    _ = b_dev^
    _ = b_routed_dev^
    _ = a_scales_dev^
    _ = a_scales_pre_dev^
    _ = b_scales_dev^
    _ = b_scales_routed_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^
    _ = c_dev^
    _ = c_ref_dev^


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        print("===> MXFP4 grouped matmul expert-offset width")

        # 781 is the first expert past Int32; 799 is the last in the stack.
        # 100 is the in-bounds control that makes a pass non-vacuous.
        var routed: List[Int] = [100, 781, 799]

        # 128 tokens per expert selects the BM=128 / BK_ELEMS=128 tile, the one
        # serving runs. 64 selects the BK_ELEMS=512 wide tile, since K_BYTES
        # here is a multiple of 256.
        print("-- prefill tile (BM=128, BK_ELEMS=128) --")
        test_expert_offset_beyond_int32[128](routed, ctx)

        print("-- decode tile (BM=64, BK_ELEMS=512) --")
        test_expert_offset_beyond_int32[64](routed, ctx)

        comptime if ctx.default_device_info == MI355X:
            print("-- preshuffled B, persistent kernel --")
            test_expert_offset_beyond_int32[
                128, preshuffled=True, persistent=True
            ](routed, ctx)

            print("-- preshuffled B, direct kernel --")
            test_expert_offset_beyond_int32[128, preshuffled=True](routed, ctx)
        else:
            print("-- preshuffled B: skipped, MI355X only --")

        print("==== expert-offset regression passed ====")
