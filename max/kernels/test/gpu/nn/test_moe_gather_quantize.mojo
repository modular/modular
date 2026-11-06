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
"""Byte-identity tests for the fused MoE routing-glue kernels.

`moe_gather_quantize_mx_amd` replaces the `F.gather` + `quantize_mx_amd`
pair the K3 `amd_tp_w4a8` path runs after `moe_create_indices`.
`moe_sort_gather_quantize_mx_amd` folds the `moe_create_indices` sort in
too, covering the whole glue chain in one launch on decode-sized batches.
Both must reproduce the chain's output bytes exactly: same source row per
sorted slot, same 32-element MX block partition, same E8M0 scale (rounding
toward +infinity), same fp8 cast. Anything else changes the grouped GEMM's
inputs and so the model's outputs.

The chain (gather then `quantize_mx_amd`, both through their production
entry points) is the reference; the fused kernel runs the same
`moe_create_indices` order. The order itself is checked only for what
downstream depends on -- per-group membership, bijection, `restore` as its
inverse -- since the intra-expert slot order follows a relaxed-atomic
cursor and is not load-bearing. The sort-fused kernel's
`expert_start_indices`/`expert_ids`/`expert_usage_stats` must byte-equal
`moe_create_indices`'s; its order is another valid grouping, so its
payload bytes are compared per source row.

The source-major scatter+quantize kernel (the op's large-P arm) runs the
same `moe_create_indices` outputs, so its rows land at identical sorted
positions and `run_source_major_case` checks its bytes row by row against
the gather+quantize kernel's, at the prefill chunk sizes the arm serves.
The `run_*_teeth` functions prove each check fails on the bug classes it
guards, and the `run_*_support_checks` functions pin the dispatch
envelopes.
"""

from std.atomic import Atomic, Ordering
from std.memory import bitcast, unsafe_stack_allocation
from std.random import random_float64, random_ui64, seed
from std.utils.numerics import inf, nan
from std.sys import align_of
from std.testing import assert_equal, assert_raises, assert_true

import max.gpu.primitives.block as block
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import PDL
from max.gpu.sync import barrier
from layout import Coord, ImmTileTensor, MutTileTensor, TileTensor, row_major
from layout.coord import DynamicCoord
from std.utils import StaticTuple
import max.gpu.primitives.warp as warp

from linalg.block_scaled_quantization import (
    moe_gather_quantize_mx_amd,
    moe_sort_gather_quantize_fused_supported,
    moe_sort_gather_quantize_mx_amd,
    moe_source_scatter_quantize_mx_amd,
    moe_source_scatter_quantize_supported,
    quantize_mxfp8_lane_group,
    quantize_mx_amd,
)
from nn.gather_scatter import Axis, gather
from nn.moe import moe_create_indices

# K3 production constants: top-16 of 896 experts, latent width 3584, bf16
# activations quantized to FP8-MX with one E8M0 scale per 32 elements.
comptime NUM_EXPERTS = 896
comptime TOPK = 16
comptime LATENT = 3584
comptime SF = 32
comptime SCALE_COLS = LATENT // SF

# The sort-fused kernel's launch geometry, mirrored from the production
# kernel: one CTA per source pair, one 8-element vector per thread.
comptime SORT_BLOCK_THREADS = LATENT // 8
comptime MAX_QUERIES = 64
comptime MAX_WARPS = 16


def gen_topk_ids[
    num_experts: Int = NUM_EXPERTS, topk: Int = TOPK
](spread: String, num_tokens: Int, rng_seed: Int) raises -> List[Int32]:
    """Seeded expert ids: `topk` distinct experts per token.

    "u" draws uniformly; "s" gives every token the same experts (max
    cursor contention); "p" draws from a 64-expert pool, leaving the rest
    empty.
    """
    seed(rng_seed)
    var ids = List[Int32](length=num_tokens * topk, fill=Int32(0))
    var pool_size = num_experts if spread == "u" else 64

    def pick_experts() {imm pool_size} -> List[Int32]:
        var picked = List[Int32]()
        while len(picked) < topk:
            var e = Int32(random_ui64(0, UInt64(pool_size - 1)))
            if e not in picked:
                picked.append(e)
        return picked^

    var shared = List[Int32]()
    if spread == "s":
        shared = pick_experts()
    for t in range(num_tokens):
        var picked = shared.copy() if spread == "s" else pick_experts()
        for k in range(topk):
            ids[t * topk + k] = picked[k]
    return ids^


def gen_latent[
    num_cols: Int = LATENT
](num_tokens: Int, rng_seed: Int) raises -> List[Scalar[.bfloat16]]:
    """Deterministic bf16 rows: a per-block magnitude ladder, with the first
    row holding the values the scale path must handle identically on both
    sides: NaN, +Inf, -0.0, a denormal, and 0.0 blocks."""
    seed(rng_seed + 1)
    var bf16 = List[Scalar[.bfloat16]]()
    for row in range(num_tokens):
        for col in range(num_cols):
            var block = col // SF
            var val: Float64
            if row == 0:
                if block == 0:
                    val = nan[.float64]()
                elif block == 1:
                    val = inf[.float64]()
                elif block == 2:
                    val = -0.0
                elif block == 3:
                    val = 1e-40  # bf16 subnormal
                elif block == 4:
                    val = 0.0
                else:
                    val = (
                        random_float64(-4.0, 4.0)
                        * Float64(1 << ((block - 5) % 9))
                        / 256.0
                    )
            else:
                val = (
                    random_float64(-4.0, 4.0)
                    * Float64(1 << (block % 9))
                    / 256.0
                )
            bf16.append(val.cast[.bfloat16]())
    return bf16^


def build_gather_indices(
    order: DeviceBuffer[.uint32],
    divisor: Int,
    num_slots: Int,
    ctx: DeviceContext,
) raises -> DeviceBuffer[.uint32]:
    """Materializes ``order // divisor`` as the gather's index buffer."""
    var order_h = ctx.enqueue_create_host_buffer[.uint32](num_slots)
    order.enqueue_copy_to(order_h)
    ctx.synchronize()
    var indices_h = ctx.enqueue_create_host_buffer[.uint32](num_slots)
    for p in range(num_slots):
        indices_h[p] = order_h[p] // UInt32(divisor)
    var indices_d = ctx.enqueue_create_buffer[.uint32](num_slots)
    ctx.enqueue_copy(indices_d, indices_h)
    return indices_d


def launch_gather(
    latent: ImmTileTensor[.bfloat16, ...],
    indices: ImmTileTensor[.uint32, ...],
    gathered: MutTileTensor[.bfloat16, ...],
    ctx: DeviceContext,
) raises:
    """The `mo.gather` op's GPU path over pre-divided row indices."""
    gather[axis=0](
        gathered.make_dynamic[.int64](),
        latent.make_dynamic[.int64](),
        indices.make_dynamic[.int64](),
        context=ctx,
    )


def check_order_contract(
    topk: List[Int32],
    order: HostBuffer[.uint32],
    start: HostBuffer[.uint32],
    restore: HostBuffer[.uint32],
    num_slots: Int,
) raises:
    """The index contract downstream relies on: `order` is a bijection, each
    slot sits inside its expert's group, and `restore` inverts `order`."""
    var counts = List[Int](length=NUM_EXPERTS, fill=0)
    for i in range(num_slots):
        counts[Int(topk[i])] += 1
    var ref_start = List[Int](length=NUM_EXPERTS + 1, fill=0)
    for g in range(NUM_EXPERTS):
        ref_start[g + 1] = ref_start[g] + counts[g]
    for g in range(NUM_EXPERTS + 1):
        assert_equal(
            Int(start[g]), ref_start[g], "expert_start_indices mismatch"
        )

    var seen = List[Bool](length=num_slots, fill=False)
    for p in range(num_slots):
        var i = Int(order[p])
        assert_true(0 <= i and i < num_slots, "order value out of range")
        assert_true(not seen[i], "order visits a slot twice")
        seen[i] = True
        var g = Int(topk[i])
        assert_true(
            ref_start[g] <= p and p < ref_start[g + 1],
            "slot sorted outside its expert's range",
        )
    for i in range(num_slots):
        assert_equal(
            Int(order[Int(restore[i])]),
            i,
            "restore is not order's inverse",
        )


def check_bytes_equal[
    num_cols: Int = LATENT, scale_cols: Int = SCALE_COLS
](
    chain_quantized: HostBuffer[.float8_e4m3fn],
    chain_scales: HostBuffer[.float8_e8m0fnu],
    fused_quantized: HostBuffer[.float8_e4m3fn],
    fused_scales: HostBuffer[.float8_e8m0fnu],
    num_rows: Int,
) raises:
    """Fused bytes == chain bytes, per sorted row: payload and scales."""
    var fused_q = TileTensor(fused_quantized, row_major(num_rows, num_cols))
    var chain_q = TileTensor(chain_quantized, row_major(num_rows, num_cols))
    var fused_s = TileTensor(fused_scales, row_major(num_rows, scale_cols))
    var chain_s = TileTensor(chain_scales, row_major(num_rows, scale_cols))
    for p in range(num_rows):
        for c in range(num_cols):
            assert_equal(
                bitcast[.uint8](fused_q[p, c]),
                bitcast[.uint8](chain_q[p, c]),
                "fused payload row differs from the chain's",
            )
        for c in range(scale_cols):
            assert_equal(
                bitcast[.uint8](fused_s[p, c]),
                bitcast[.uint8](chain_s[p, c]),
                "fused scale row differs from the chain's",
            )


def run_case(num_tokens: Int, spread: String, ctx: DeviceContext) raises:
    """One shape x spread: chain vs fused over every sorted row."""
    var slots = num_tokens * TOPK
    var topk_h = gen_topk_ids(spread, num_tokens, 7)
    var latent_h = gen_latent(num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * LATENT)
    var gathered_d = ctx.enqueue_create_buffer[.bfloat16](slots * LATENT)
    var chain_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var chain_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    var fused_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var fused_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(NUM_EXPERTS + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(NUM_EXPERTS))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, LATENT))
    var gathered_tt = TileTensor(gathered_d, row_major(slots, LATENT))
    var chain_q_tt = TileTensor(chain_q_d, row_major(slots, LATENT))
    var chain_s_tt = TileTensor(chain_s_d, row_major(slots, SCALE_COLS))
    var fused_q_tt = TileTensor(fused_q_d, row_major(slots, LATENT))
    var fused_s_tt = TileTensor(fused_s_d, row_major(slots, SCALE_COLS))

    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    launch_gather(
        latent_tt,
        TileTensor(
            build_gather_indices(order_d, TOPK, slots, ctx), row_major(slots)
        ),
        gathered_tt,
        ctx,
    )
    quantize_mx_amd(ctx, chain_q_tt, chain_s_tt, gathered_tt)
    moe_gather_quantize_mx_amd(
        ctx, fused_q_tt, fused_s_tt, latent_tt, order_tt, TOPK
    )

    var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var start_h = ctx.enqueue_create_host_buffer[.uint32](NUM_EXPERTS + 1)
    var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var chain_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * LATENT
    )
    var chain_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    var fused_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * LATENT
    )
    var fused_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    ctx.enqueue_copy(order_h, order_d)
    ctx.enqueue_copy(start_h, start_d)
    ctx.enqueue_copy(restore_h, restore_d)
    ctx.enqueue_copy(chain_q_h, chain_q_d)
    ctx.enqueue_copy(chain_s_h, chain_s_d)
    ctx.enqueue_copy(fused_q_h, fused_q_d)
    ctx.enqueue_copy(fused_s_h, fused_s_d)
    ctx.synchronize()

    check_order_contract(topk_h, order_h, start_h, restore_h, slots)
    check_bytes_equal(chain_q_h, chain_s_h, fused_q_h, fused_s_h, slots)
    print("ok M=", num_tokens, " spread=", spread, sep="")

    # Keep every device buffer alive past the last kernel use.
    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = gathered_d
    _ = chain_q_d
    _ = chain_s_d
    _ = fused_q_d
    _ = fused_s_d
    _ = order_h
    _ = start_h
    _ = restore_h
    _ = chain_q_h
    _ = chain_s_h
    _ = fused_q_h
    _ = fused_s_h


def run_teeth[broken: Int](ctx: DeviceContext) raises:
    """Proves `check_bytes_equal` fails on the bug classes it guards against.

    1: a wrong gather divisor (the chain maps slots to the wrong token
    rows; always in bounds since `order // 32 < M`).
    2: a dropped scale block (one live chain scale byte zeroed).
    3: an off-by-one order read (the fused kernel fed a rotated order).
    """
    comptime num_tokens = 8
    var slots = num_tokens * TOPK
    var topk_h = gen_topk_ids("u", num_tokens, 7)
    var latent_h = gen_latent(num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * LATENT)
    var gathered_d = ctx.enqueue_create_buffer[.bfloat16](slots * LATENT)
    var chain_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var chain_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    var fused_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var fused_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(NUM_EXPERTS + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(NUM_EXPERTS))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, LATENT))
    var gathered_tt = TileTensor(gathered_d, row_major(slots, LATENT))
    var chain_q_tt = TileTensor(chain_q_d, row_major(slots, LATENT))
    var chain_s_tt = TileTensor(chain_s_d, row_major(slots, SCALE_COLS))
    var fused_q_tt = TileTensor(fused_q_d, row_major(slots, LATENT))
    var fused_s_tt = TileTensor(fused_s_d, row_major(slots, SCALE_COLS))

    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    comptime if broken == 1:
        launch_gather(
            latent_tt,
            TileTensor(
                build_gather_indices(order_d, 2 * TOPK, slots, ctx),
                row_major(slots),
            ),
            gathered_tt,
            ctx,
        )
    else:
        launch_gather(
            latent_tt,
            TileTensor(
                build_gather_indices(order_d, TOPK, slots, ctx),
                row_major(slots),
            ),
            gathered_tt,
            ctx,
        )
    quantize_mx_amd(ctx, chain_q_tt, chain_s_tt, gathered_tt)

    var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    ctx.enqueue_copy(order_h, order_d)
    ctx.synchronize()

    comptime if broken == 3:
        # A rotated order: reading `order[p + 1]` everywhere.
        var rotated = List[UInt32](length=slots, fill=UInt32(0))
        for p in range(slots):
            rotated[p] = order_h[(p + 1) % slots]
        ctx.enqueue_copy(order_d, rotated)
        ctx.synchronize()

    moe_gather_quantize_mx_amd(
        ctx, fused_q_tt, fused_s_tt, latent_tt, order_tt, TOPK
    )

    var chain_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * LATENT
    )
    var chain_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    var fused_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * LATENT
    )
    var fused_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    ctx.enqueue_copy(chain_q_h, chain_q_d)
    ctx.enqueue_copy(chain_s_h, chain_s_d)
    ctx.enqueue_copy(fused_q_h, fused_q_d)
    ctx.enqueue_copy(fused_s_h, fused_s_d)
    ctx.synchronize()

    comptime if broken == 2:
        # Drop one LIVE scale block from the chain reference.
        var chain_s = TileTensor(chain_s_h, row_major(slots, SCALE_COLS))
        var corrupted = False
        for p in range(slots):
            for c in range(SCALE_COLS):
                if bitcast[.uint8](chain_s[p, c]) != UInt8(0):
                    chain_s[p, c] = bitcast[.float8_e8m0fnu](UInt8(0))
                    corrupted = True
                    break
            if corrupted:
                break
        assert_true(corrupted, "no live scale block to drop")

    var failed = False
    try:
        check_bytes_equal(chain_q_h, chain_s_h, fused_q_h, fused_s_h, slots)
    except e:
        failed = True
        print("teeth ", broken, " failed as expected: ", String(e), sep="")
    assert_true(
        failed,
        "broken variant " + String(broken) + " passed: the check has no teeth",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = gathered_d
    _ = chain_q_d
    _ = chain_s_d
    _ = fused_q_d
    _ = fused_s_d
    _ = order_h
    _ = chain_q_h
    _ = chain_s_h
    _ = fused_q_h
    _ = fused_s_h


# ===----------------------------------------------------------------------=== #
# The sort+gather+quantize kernel
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(SORT_BLOCK_THREADS)
    )
)
@__name(t"test_moe_sort_broken_rank_kernel_{BUG}")
def _moe_sort_broken_rank_kernel[
    BUG: Int
](
    order: MutPointer[UInt32, MutUntrackedOrigin],
    start: MutPointer[UInt32, MutUntrackedOrigin],
    restore: MutPointer[UInt32, MutUntrackedOrigin],
    ids: MutPointer[Int32, MutUntrackedOrigin],
    stats: MutPointer[UInt32, MutUntrackedOrigin],
    topk_ids: ImmPointer[Int32, ImmUntrackedOrigin],
    num_slots: Int32,
):
    """The frozen negative control for the index checks: the production
    sort phase with one deliberate bug per `BUG` (1: `<=` on the pair-key
    rank; 2: idle threads hold key 0 instead of the max-uint32 sentinel).
    Deliberately independent of the production kernel so the control cannot
    drift into sharing the bug it guards against."""
    var n = Int(num_slots)
    var i = block_idx.x
    var tid = thread_idx.x
    var lane = lane_id()
    var wid = warp_id()
    var nw = block_dim.x // WARP_SIZE

    var hist = unsafe_stack_allocation[
        NUM_EXPERTS, UInt32, address_space=.SHARED
    ]()
    var part = unsafe_stack_allocation[
        MAX_QUERIES * MAX_WARPS, UInt32, address_space=.SHARED
    ]()
    var result = unsafe_stack_allocation[
        MAX_QUERIES, UInt32, address_space=.SHARED
    ]()

    with PDL[overlap_at_beginning=True]():
        var e_i = UInt32(topk_ids.unsafe_offset(i).unsafe_load())
        var key_i = (e_i << 16) | UInt32(i)
        var lo = (i * NUM_EXPERTS) // n
        var hi = ((i + 1) * NUM_EXPERTS) // n
        var num_queries = 1 + (hi - lo) + 1

        if i == 0:
            for x in range(tid, NUM_EXPERTS, SORT_BLOCK_THREADS):
                hist.unsafe_offset(x).unsafe_store(UInt32(0))
            barrier()

        var j0 = 4 * tid
        comptime idle_key = UInt32(0) if BUG == 2 else UInt32(0xFFFFFFFF)
        var keys = SIMD[.uint32, 4](idle_key)
        if j0 < n:
            var chunk = topk_ids.unsafe_offset(j0).unsafe_load[
                width=4, alignment=16
            ]()
            keys = (chunk.cast[.uint32]() << SIMD[.uint32, 4](16)) | SIMD[
                .uint32, 4
            ](
                UInt32(j0),
                UInt32(j0 + 1),
                UInt32(j0 + 2),
                UInt32(j0 + 3),
            )

        for q in range(num_queries):
            var bound = key_i
            if q > 0:
                bound = UInt32(lo + q - 1) << 16
            comptime if BUG == 1:
                if q == 0:
                    bound = bound + 1
            var cnt = (
                keys.lt(SIMD[.uint32, 4](bound)).cast[.uint32]().reduce_add()
            )
            cnt = warp.sum(cnt)
            if lane == 0:
                part.unsafe_offset(q * MAX_WARPS + wid).unsafe_store(cnt)
        barrier()
        if wid == 0:
            for q in range(num_queries):
                var v = UInt32(0)
                if lane < nw:
                    v = part.unsafe_offset(q * MAX_WARPS + lane).unsafe_load()
                v = warp.sum(v)
                if lane == 0:
                    result.unsafe_offset(q).unsafe_store(v)
        barrier()

        var pos = result.unsafe_offset(0).unsafe_load()
        # Keep the bug's inflated ranks inside the buffer; the honest kernel
        # is in range by construction.
        pos = min(pos, UInt32(n - 1))

        # The honest histogram and usage stats, so the tooth fails on the
        # rank bug it guards, not trivially on the stats.
        if i == 0:
            if j0 < n:
                comptime for l in range(4):
                    _ = Atomic[UInt32, scope="workgroup"].fetch_add[
                        ordering=Ordering.RELAXED
                    ](hist.unsafe_offset(Int(keys[l]) >> 16), UInt32(1))
            var local_max = UInt32(0)
            for x in range(tid, NUM_EXPERTS, SORT_BLOCK_THREADS):
                local_max = max(local_max, hist.unsafe_offset(x).unsafe_load())
            var global_max = block.max[
                block_size=SORT_BLOCK_THREADS, broadcast=False
            ](local_max)
            if tid == 0:
                stats.unsafe_offset(0).unsafe_store(global_max)
                stats.unsafe_offset(1).unsafe_store(UInt32(NUM_EXPERTS))

        if tid == 0:
            order.unsafe_offset(Int(pos)).unsafe_store(UInt32(i))
            restore.unsafe_offset(i).unsafe_store(pos)
            for x in range(lo, hi):
                start.unsafe_offset(x).unsafe_store(
                    result.unsafe_offset(1 + x - lo).unsafe_load()
                )
                ids.unsafe_offset(x).unsafe_store(Int32(x))
            if i == n - 1:
                start.unsafe_offset(NUM_EXPERTS).unsafe_store(UInt32(n))


def launch_broken_rank[
    bug: Int
](
    order: MutTileTensor[.uint32, ...],
    start: MutTileTensor[.uint32, ...],
    restore: MutTileTensor[.uint32, ...],
    ids: MutTileTensor[.int32, ...],
    stats: MutTileTensor[.uint32, ...],
    topk_ids: ImmTileTensor[.int32, ...],
    ctx: DeviceContext,
) raises:
    comptime kernel = _moe_sort_broken_rank_kernel[bug]
    var slots = Int(topk_ids.dim[0]())
    ctx.enqueue_function[kernel](
        order.unsafe_ptr(),
        start.unsafe_ptr(),
        restore.unsafe_ptr(),
        ids.unsafe_ptr(),
        stats.unsafe_ptr(),
        topk_ids.unsafe_ptr(),
        Int32(slots),
        grid_dim=(slots, 1, 1),
        block_dim=(SORT_BLOCK_THREADS, 1, 1),
    )


def check_sort_indices_contract[
    num_experts: Int = NUM_EXPERTS
](
    topk: List[Int32],
    ref_start: HostBuffer[.uint32],
    ref_ids: HostBuffer[.int32],
    ref_stats: HostBuffer[.uint32],
    order: HostBuffer[.uint32],
    start: HostBuffer[.uint32],
    restore: HostBuffer[.uint32],
    ids: HostBuffer[.int32],
    stats: HostBuffer[.uint32],
    num_slots: Int,
) raises:
    """The byte-identity scope for the sort-fused kernel's index outputs.

    `expert_start_indices`, `expert_ids` and `expert_usage_stats` must be
    byte-equal to `moe_create_indices`'s on the same ids. The order pair
    follows a relaxed-atomic cursor in the reference, so the order check is
    per-group membership, bijection and `restore == order^-1` -- never exact
    intra-expert positions.
    """
    for g in range(num_experts + 1):
        assert_equal(
            Int(start[g]),
            Int(ref_start[g]),
            "sortfused expert_start_indices differs from moe_create_indices'",
        )
    for g in range(num_experts):
        assert_equal(
            Int(ids[g]), Int(ref_ids[g]), "sortfused expert_ids mismatch"
        )
    assert_equal(
        Int(stats[0]),
        Int(ref_stats[0]),
        "sortfused usage max count mismatch",
    )
    assert_equal(
        Int(stats[1]),
        Int(ref_stats[1]),
        "sortfused usage expert count mismatch",
    )
    var seen = List[Bool](length=num_slots, fill=False)
    for p in range(num_slots):
        var i = Int(order[p])
        assert_true(
            0 <= i and i < num_slots, "sortfused order value out of range"
        )
        assert_true(not seen[i], "sortfused order visits a slot twice")
        seen[i] = True
        var g = Int(topk[i])
        assert_true(
            Int(ref_start[g]) <= p and p < Int(ref_start[g + 1]),
            "sortfused slot sorted outside its expert's range",
        )
        assert_equal(
            Int(restore[i]), p, "sortfused restore is not order's inverse"
        )
    for i in range(num_slots):
        assert_true(seen[i], "sortfused order misses a slot")


def check_sort_bytes_per_source[
    topk: Int = TOPK, num_cols: Int = LATENT
](
    ref_order: HostBuffer[.uint32],
    ref_quantized: HostBuffer[.float8_e4m3fn],
    ref_scales: HostBuffer[.float8_e8m0fnu],
    order: HostBuffer[.uint32],
    quantized: HostBuffer[.float8_e4m3fn],
    scales: HostBuffer[.float8_e8m0fnu],
    num_slots: Int,
) raises:
    """The sort-fused payload bytes against the gather+quantize kernel's,
    per source row.

    The two kernels may land a source row at different intra-expert
    positions (both orders are valid groupings), so each candidate row is
    compared against the reference row holding the same source token: the
    quantize is deterministic, so every reference row with the same source
    is byte-identical."""
    comptime scale_cols = num_cols // SF
    var cand_q = TileTensor(quantized, row_major(num_slots, num_cols))
    var ref_q = TileTensor(ref_quantized, row_major(num_slots, num_cols))
    var cand_s = TileTensor(scales, row_major(num_slots, scale_cols))
    var ref_s = TileTensor(ref_scales, row_major(num_slots, scale_cols))
    var rep = List[Int](length=num_slots // topk, fill=-1)
    for p in range(num_slots):
        var src = Int(ref_order[p]) // topk
        if rep[src] < 0:
            rep[src] = p
    for p in range(num_slots):
        var src = Int(order[p]) // topk
        var ref_p = rep[src]
        assert_true(ref_p >= 0, "sortfused row maps to an unknown source")
        for c in range(num_cols):
            assert_equal(
                bitcast[.uint8](cand_q[p, c]),
                bitcast[.uint8](ref_q[ref_p, c]),
                (
                    "sortfused payload row differs from the gather+quantize"
                    " kernel's"
                ),
            )
        for c in range(scale_cols):
            assert_equal(
                bitcast[.uint8](cand_s[p, c]),
                bitcast[.uint8](ref_s[ref_p, c]),
                "sortfused scale row differs from the gather+quantize kernel's",
            )


def run_sort_case[
    num_experts: Int = NUM_EXPERTS, num_cols: Int = LATENT, topk: Int = TOPK
](num_tokens: Int, spread: String, ctx: DeviceContext) raises:
    """One shape x spread: the sort-fused kernel against the
    `moe_create_indices` + gather+quantize reference, per the byte-identity
    scope."""
    comptime scale_cols = num_cols // SF
    var slots = num_tokens * topk
    var topk_h = gen_topk_ids[num_experts, topk](spread, num_tokens, 7)
    var latent_h = gen_latent[num_cols](num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * num_cols)
    var ref_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var ref_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * scale_cols)
    var cand_order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var cand_start_d = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var cand_restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var cand_ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var cand_stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var cand_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var cand_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(num_experts + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(num_experts))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, num_cols))
    var ref_q_tt = TileTensor(ref_q_d, row_major(slots, num_cols))
    var ref_s_tt = TileTensor(ref_s_d, row_major(slots, scale_cols))
    var cand_order_tt = TileTensor(cand_order_d, row_major(slots))
    var cand_start_tt = TileTensor(cand_start_d, row_major(num_experts + 1))
    var cand_restore_tt = TileTensor(cand_restore_d, row_major(slots))
    var cand_ids_tt = TileTensor(cand_ids_d, row_major(num_experts))
    var cand_stats_tt = TileTensor(cand_stats_d, row_major(2))
    var cand_q_tt = TileTensor(cand_q_d, row_major(slots, num_cols))
    var cand_s_tt = TileTensor(cand_s_d, row_major(slots, scale_cols))

    # The reference: moe_create_indices, then the gather+quantize kernel
    # over its order.
    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    moe_gather_quantize_mx_amd(
        ctx, ref_q_tt, ref_s_tt, latent_tt, order_tt, topk
    )
    # The candidate: everything in one launch.
    moe_sort_gather_quantize_mx_amd[num_experts=num_experts, num_cols=num_cols](
        ctx,
        cand_q_tt,
        cand_s_tt,
        latent_tt,
        cand_order_tt,
        cand_start_tt,
        cand_restore_tt,
        cand_ids_tt,
        cand_stats_tt,
        topk_tt,
        topk,
    )

    var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var start_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var ids_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var ref_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * num_cols
    )
    var ref_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    var cand_order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var cand_start_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var cand_restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var cand_ids_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var cand_stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var cand_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * num_cols
    )
    var cand_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    ctx.enqueue_copy(order_h, order_d)
    ctx.enqueue_copy(start_h, start_d)
    ctx.enqueue_copy(restore_h, restore_d)
    ctx.enqueue_copy(ids_h, ids_d)
    ctx.enqueue_copy(stats_h, stats_d)
    ctx.enqueue_copy(ref_q_h, ref_q_d)
    ctx.enqueue_copy(ref_s_h, ref_s_d)
    ctx.enqueue_copy(cand_order_h, cand_order_d)
    ctx.enqueue_copy(cand_start_h, cand_start_d)
    ctx.enqueue_copy(cand_restore_h, cand_restore_d)
    ctx.enqueue_copy(cand_ids_h, cand_ids_d)
    ctx.enqueue_copy(cand_stats_h, cand_stats_d)
    ctx.enqueue_copy(cand_q_h, cand_q_d)
    ctx.enqueue_copy(cand_s_h, cand_s_d)
    ctx.synchronize()

    check_sort_indices_contract[num_experts](
        topk_h,
        start_h,
        ids_h,
        stats_h,
        cand_order_h,
        cand_start_h,
        cand_restore_h,
        cand_ids_h,
        cand_stats_h,
        slots,
    )
    check_sort_bytes_per_source[topk, num_cols](
        order_h, ref_q_h, ref_s_h, cand_order_h, cand_q_h, cand_s_h, slots
    )
    print(
        "ok sortfused M=",
        num_tokens,
        " E=",
        num_experts,
        " K=",
        num_cols,
        " spread=",
        spread,
        sep="",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = ref_q_d
    _ = ref_s_d
    _ = cand_order_d
    _ = cand_start_d
    _ = cand_restore_d
    _ = cand_ids_d
    _ = cand_stats_d
    _ = cand_q_d
    _ = cand_s_d
    _ = order_h
    _ = start_h
    _ = restore_h
    _ = ids_h
    _ = stats_h
    _ = ref_q_h
    _ = ref_s_h
    _ = cand_order_h
    _ = cand_start_h
    _ = cand_restore_h
    _ = cand_ids_h
    _ = cand_stats_h
    _ = cand_q_h
    _ = cand_s_h


def run_sort_out_of_range_case[
    num_experts: Int = NUM_EXPERTS, num_cols: Int = LATENT, topk: Int = TOPK
](num_tokens: Int, ctx: DeviceContext) raises:
    """Out-of-range expert ids through the sort-fused kernel.

    `moe_create_indices` files an id outside `[0, num_experts)` in no
    expert's CSR range: the pair gets no `token_expert_order` entry and no
    payload row, and its `restore_token_order` entry holds the
    `num_slots` sentinel. The sort-fused kernel must do the same, and must
    not index its LDS histogram with such an id -- the `e << 16` rank key
    truncates an id past 2^16 into a valid expert's key space, and a
    negative id into a histogram index past the array.

    The injected ids cover every class: `num_experts` itself, a negative
    id, and one past the 16-bit key packing.
    """
    comptime scale_cols = num_cols // SF
    var slots = num_tokens * topk
    var topk_h = gen_topk_ids[num_experts, topk]("u", num_tokens, 7)
    topk_h[1] = Int32(num_experts)
    topk_h[slots // 2] = Int32(-1)
    topk_h[slots - 3] = Int32(65536 + num_experts // 8)
    var latent_h = gen_latent[num_cols](num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * num_cols)
    var cand_order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var cand_start_d = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var cand_restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var cand_ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var cand_stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var cand_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var cand_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    var ref_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var ref_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * scale_cols)

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)
    # The reference payload runs over the reference order with its
    # unwritten tail pinned to slot 0, so the gather never reads an entry
    # no kernel writes.
    var poison_q = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * num_cols
    )
    var poison_s = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    for p in range(slots * num_cols):
        poison_q[p] = bitcast[.float8_e4m3fn](UInt8(0xAA))
    for p in range(slots * scale_cols):
        poison_s[p] = bitcast[.float8_e8m0fnu](UInt8(0xAA))
    ctx.enqueue_copy(cand_q_d, poison_q)
    ctx.enqueue_copy(cand_s_d, poison_s)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(num_experts + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(num_experts))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, num_cols))
    var cand_order_tt = TileTensor(cand_order_d, row_major(slots))
    var cand_start_tt = TileTensor(cand_start_d, row_major(num_experts + 1))
    var cand_restore_tt = TileTensor(cand_restore_d, row_major(slots))
    var cand_ids_tt = TileTensor(cand_ids_d, row_major(num_experts))
    var cand_stats_tt = TileTensor(cand_stats_d, row_major(2))
    var cand_q_tt = TileTensor(cand_q_d, row_major(slots, num_cols))
    var cand_s_tt = TileTensor(cand_s_d, row_major(slots, scale_cols))
    var ref_q_tt = TileTensor(ref_q_d, row_major(slots, num_cols))
    var ref_s_tt = TileTensor(ref_s_d, row_major(slots, scale_cols))

    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    moe_sort_gather_quantize_mx_amd[num_experts=num_experts, num_cols=num_cols](
        ctx,
        cand_q_tt,
        cand_s_tt,
        latent_tt,
        cand_order_tt,
        cand_start_tt,
        cand_restore_tt,
        cand_ids_tt,
        cand_stats_tt,
        topk_tt,
        topk,
    )

    var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var start_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var ids_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var cand_order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var cand_start_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var cand_restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var cand_ids_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var cand_stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var cand_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * num_cols
    )
    var cand_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    ctx.enqueue_copy(order_h, order_d)
    ctx.enqueue_copy(start_h, start_d)
    ctx.enqueue_copy(restore_h, restore_d)
    ctx.enqueue_copy(ids_h, ids_d)
    ctx.enqueue_copy(stats_h, stats_d)
    ctx.enqueue_copy(cand_order_h, cand_order_d)
    ctx.enqueue_copy(cand_start_h, cand_start_d)
    ctx.enqueue_copy(cand_restore_h, cand_restore_d)
    ctx.enqueue_copy(cand_ids_h, cand_ids_d)
    ctx.enqueue_copy(cand_stats_h, cand_stats_d)
    ctx.enqueue_copy(cand_q_h, cand_q_d)
    ctx.enqueue_copy(cand_s_h, cand_s_d)
    ctx.synchronize()

    # The index outputs must byte-equal `moe_create_indices`'s.
    for g in range(num_experts + 1):
        assert_equal(
            Int(cand_start_h[g]),
            Int(start_h[g]),
            (
                "sortfused out-of-range: expert_start_indices differs from"
                " moe_create_indices'"
            ),
        )
    for g in range(num_experts):
        assert_equal(
            Int(cand_ids_h[g]), Int(ids_h[g]), "sortfused out-of-range ids"
        )
    for s in range(2):
        assert_equal(
            Int(cand_stats_h[s]),
            Int(stats_h[s]),
            "sortfused out-of-range usage stats",
        )

    var valid = 0
    for i in range(slots):
        var e = Int(topk_h[i])
        if e >= 0 and e < num_experts:
            valid += 1
    assert_equal(
        Int(cand_start_h[num_experts]),
        valid,
        "sortfused out-of-range: start[E] is not the valid-pair count",
    )

    var seen = List[Bool](length=valid, fill=False)
    for i in range(slots):
        var e = Int(topk_h[i])
        if e >= 0 and e < num_experts:
            var pos = Int(cand_restore_h[i])
            assert_true(
                pos >= 0 and pos < valid,
                (
                    "sortfused out-of-range: a valid pair's restore entry is"
                    " outside the valid prefix"
                ),
            )
            assert_true(
                not seen[pos], "sortfused: two valid pairs share a sorted slot"
            )
            seen[pos] = True
            assert_equal(
                Int(cand_order_h[pos]),
                i,
                "sortfused out-of-range: restore is not order's inverse",
            )
            assert_true(
                Int(cand_start_h[e]) <= pos and pos < Int(cand_start_h[e + 1]),
                (
                    "sortfused out-of-range: a valid pair sorted outside its"
                    " expert's range"
                ),
            )
        else:
            assert_equal(
                Int(cand_restore_h[i]),
                slots,
                (
                    "sortfused out-of-range: an out-of-range id must hold the"
                    " num_slots sentinel"
                ),
            )

    # The payload: each valid pair's row matches the gather+quantize
    # kernel's bytes for the same source token; every row past the valid
    # prefix keeps the poison, so an out-of-range pair wrote nothing.
    var patched = ctx.enqueue_create_host_buffer[.uint32](slots)
    for p in range(slots):
        patched[p] = order_h[p] if p < valid else UInt32(0)
    ctx.enqueue_copy(order_d, patched)
    moe_gather_quantize_mx_amd(
        ctx, ref_q_tt, ref_s_tt, latent_tt, order_tt, topk
    )
    var ref_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * num_cols
    )
    var ref_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    ctx.enqueue_copy(ref_q_h, ref_q_d)
    ctx.enqueue_copy(ref_s_h, ref_s_d)
    ctx.synchronize()

    var cand_q = TileTensor(cand_q_h, row_major(slots, num_cols))
    var ref_q = TileTensor(ref_q_h, row_major(slots, num_cols))
    var cand_s = TileTensor(cand_s_h, row_major(slots, scale_cols))
    var ref_s = TileTensor(ref_s_h, row_major(slots, scale_cols))
    # A source token may appear at several sorted rows; the quantize is
    # deterministic, so any reference row with that source carries the
    # bytes.
    var rep = List[Int](length=slots, fill=-1)
    for p in range(valid):
        var src = Int(order_h[p]) // topk
        if rep[src] < 0:
            rep[src] = p
    for i in range(slots):
        var e = Int(topk_h[i])
        if e >= 0 and e < num_experts:
            var p = Int(cand_restore_h[i])
            var ref_p = rep[i // topk]
            assert_true(
                ref_p >= 0, "sortfused out-of-range: unknown reference source"
            )
            for c in range(num_cols):
                assert_equal(
                    bitcast[.uint8](cand_q[p, c]),
                    bitcast[.uint8](ref_q[ref_p, c]),
                    (
                        "sortfused out-of-range: a valid pair's payload row"
                        " differs from the gather+quantize kernel's"
                    ),
                )
            for c in range(scale_cols):
                assert_equal(
                    bitcast[.uint8](cand_s[p, c]),
                    bitcast[.uint8](ref_s[ref_p, c]),
                    "sortfused out-of-range: a valid pair's scale row differs",
                )
    for p in range(valid, slots):
        for c in range(num_cols):
            assert_equal(
                bitcast[.uint8](cand_q[p, c]),
                UInt8(0xAA),
                (
                    "sortfused out-of-range: an out-of-range pair wrote a"
                    " payload row"
                ),
            )
        for c in range(scale_cols):
            assert_equal(
                bitcast[.uint8](cand_s[p, c]),
                UInt8(0xAA),
                "sortfused out-of-range: an out-of-range pair wrote a scale",
            )
    print(
        "ok sortfused out-of-range M=",
        num_tokens,
        " E=",
        num_experts,
        " K=",
        num_cols,
        sep="",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = cand_order_d
    _ = cand_start_d
    _ = cand_restore_d
    _ = cand_ids_d
    _ = cand_stats_d
    _ = cand_q_d
    _ = cand_s_d
    _ = ref_q_d
    _ = ref_s_d
    _ = order_h
    _ = start_h
    _ = restore_h
    _ = ids_h
    _ = stats_h
    _ = cand_order_h
    _ = cand_start_h
    _ = cand_restore_h
    _ = cand_ids_h
    _ = cand_stats_h
    _ = cand_q_h
    _ = cand_s_h
    _ = ref_q_h
    _ = ref_s_h
    _ = poison_q
    _ = poison_s
    _ = patched


def run_gather_order_guard_case(ctx: DeviceContext) raises:
    """An order entry naming no slot reads no `input` row.

    `token_expert_order` is u32 and `moe_create_indices` leaves the slot of
    an out-of-range expert id unwritten, so an entry can name a token past
    the last. The kernel must skip that row rather than read past the end of
    `input`, and must leave the row it would have written untouched.
    """
    comptime num_tokens = 8
    comptime slots = num_tokens * TOPK
    var latent_h = gen_latent[LATENT](num_tokens, 11)
    var order_h = List[UInt32](length=slots, fill=UInt32(0))
    for p in range(slots):
        order_h[p] = UInt32(p)
    order_h[7] = UInt32(0xFFFFFFFF)

    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * LATENT)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * SCALE_COLS)
    var ref_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var ref_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * SCALE_COLS)
    ctx.enqueue_copy(latent_d, latent_h)
    ctx.enqueue_copy(order_d, order_h)
    var poison_q = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * LATENT
    )
    var poison_s = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    for p in range(slots * LATENT):
        poison_q[p] = bitcast[.float8_e4m3fn](UInt8(0xAA))
    for p in range(slots * SCALE_COLS):
        poison_s[p] = bitcast[.float8_e8m0fnu](UInt8(0xAA))
    ctx.enqueue_copy(q_d, poison_q)
    ctx.enqueue_copy(s_d, poison_s)

    var latent_tt = TileTensor(latent_d, row_major(num_tokens, LATENT))
    moe_gather_quantize_mx_amd(
        ctx,
        TileTensor(q_d, row_major(slots, LATENT)),
        TileTensor(s_d, row_major(slots, SCALE_COLS)),
        latent_tt,
        TileTensor(order_d, row_major(slots)),
        TOPK,
    )
    # The reference: the same order with the bad entry pointed at slot 0.
    var patched = order_h.copy()
    patched[7] = UInt32(0)
    var patched_d = ctx.enqueue_create_buffer[.uint32](slots)
    ctx.enqueue_copy(patched_d, patched)
    moe_gather_quantize_mx_amd(
        ctx,
        TileTensor(ref_q_d, row_major(slots, LATENT)),
        TileTensor(ref_s_d, row_major(slots, SCALE_COLS)),
        latent_tt,
        TileTensor(patched_d, row_major(slots)),
        TOPK,
    )

    var q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](slots * LATENT)
    var s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    var ref_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](slots * LATENT)
    var ref_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    ctx.enqueue_copy(q_h, q_d)
    ctx.enqueue_copy(s_h, s_d)
    ctx.enqueue_copy(ref_q_h, ref_q_d)
    ctx.enqueue_copy(ref_s_h, ref_s_d)
    ctx.synchronize()

    var q = TileTensor(q_h, row_major(slots, LATENT))
    var s = TileTensor(s_h, row_major(slots, SCALE_COLS))
    var ref_q = TileTensor(ref_q_h, row_major(slots, LATENT))
    var ref_s = TileTensor(ref_s_h, row_major(slots, SCALE_COLS))
    for p in range(slots):
        for c in range(LATENT):
            var got = bitcast[.uint8](q[p, c])
            var want = UInt8(0xAA) if p == 7 else bitcast[.uint8](ref_q[p, c])
            assert_equal(
                got,
                want,
                "gather+quantize: an order entry naming no slot read a row",
            )
        for c in range(SCALE_COLS):
            var got = bitcast[.uint8](s[p, c])
            var want = UInt8(0xAA) if p == 7 else bitcast[.uint8](ref_s[p, c])
            assert_equal(
                got,
                want,
                "gather+quantize: an order entry naming no slot wrote a scale",
            )
    print("ok gather+quantize order guard")

    _ = latent_d
    _ = order_d
    _ = q_d
    _ = s_d
    _ = ref_q_d
    _ = ref_s_d
    _ = patched_d
    _ = latent_h
    _ = order_h
    _ = poison_q
    _ = poison_s
    _ = q_h
    _ = s_h
    _ = ref_q_h
    _ = ref_s_h
    _ = q
    _ = s
    _ = ref_q
    _ = ref_s
    _ = patched


def run_sort_teeth[bug: Int](ctx: DeviceContext) raises:
    """Proves the index-contract checks fail on the bug classes they guard
    against: 1, a `<=` pair-key rank; 2, idle threads holding key 0 (the
    sentinel bug); 3, a wrong num_experts reaching the launcher."""
    comptime num_tokens = 32
    var slots = num_tokens * TOPK
    var topk_h = gen_topk_ids("u", num_tokens, 7)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    ctx.enqueue_copy(topk_d, topk_h)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(NUM_EXPERTS + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(NUM_EXPERTS))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))

    comptime if bug == 3:
        # A wrong num_experts: the launcher must refuse an ids tensor that
        # disagrees with the comptime expert count.
        var ids_wrong_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS // 2)
        var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * LATENT)
        var cand_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
        var cand_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            slots * SCALE_COLS
        )
        var ids_wrong_tt = TileTensor(ids_wrong_d, row_major(NUM_EXPERTS // 2))
        var latent_tt = TileTensor(latent_d, row_major(num_tokens, LATENT))
        var cand_q_tt = TileTensor(cand_q_d, row_major(slots, LATENT))
        var cand_s_tt = TileTensor(cand_s_d, row_major(slots, SCALE_COLS))
        with assert_raises(contains="disagrees"):
            moe_sort_gather_quantize_mx_amd[
                num_experts=NUM_EXPERTS, num_cols=LATENT
            ](
                ctx,
                cand_q_tt,
                cand_s_tt,
                latent_tt,
                order_tt,
                start_tt,
                restore_tt,
                ids_wrong_tt,
                stats_tt,
                topk_tt,
                TOPK,
            )
        _ = ids_wrong_d
        _ = latent_d
        _ = cand_q_d
        _ = cand_s_d
        return

    # Poison the outputs so an unwritten slot is unmistakably invalid.
    ctx.enqueue_memset[.uint32](order_d, UInt32(0xFFFFFFFF))
    ctx.enqueue_memset[.uint32](restore_d, UInt32(0xFFFFFFFF))
    launch_broken_rank[bug](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )

    # The reference index outputs on the same ids.
    var ref_order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ref_start_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
    var ref_restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ref_ids_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
    var ref_stats_d = ctx.enqueue_create_buffer[.uint32](2)
    moe_create_indices[input_type=DType.int32](
        TileTensor(ref_order_d, row_major(slots)),
        TileTensor(ref_start_d, row_major(NUM_EXPERTS + 1)),
        TileTensor(ref_restore_d, row_major(slots)),
        TileTensor(ref_ids_d, row_major(NUM_EXPERTS)),
        TileTensor(ref_stats_d, row_major(2)),
        topk_tt,
        ctx,
    )

    var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var start_h = ctx.enqueue_create_host_buffer[.uint32](NUM_EXPERTS + 1)
    var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var ids_h = ctx.enqueue_create_host_buffer[.int32](NUM_EXPERTS)
    var stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var ref_start_h = ctx.enqueue_create_host_buffer[.uint32](NUM_EXPERTS + 1)
    var ref_ids_h = ctx.enqueue_create_host_buffer[.int32](NUM_EXPERTS)
    var ref_stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    ctx.enqueue_copy(order_h, order_d)
    ctx.enqueue_copy(start_h, start_d)
    ctx.enqueue_copy(restore_h, restore_d)
    ctx.enqueue_copy(ids_h, ids_d)
    ctx.enqueue_copy(stats_h, stats_d)
    ctx.enqueue_copy(ref_start_h, ref_start_d)
    ctx.enqueue_copy(ref_ids_h, ref_ids_d)
    ctx.enqueue_copy(ref_stats_h, ref_stats_d)
    ctx.synchronize()

    var failed = False
    try:
        check_sort_indices_contract(
            topk_h,
            ref_start_h,
            ref_ids_h,
            ref_stats_h,
            order_h,
            start_h,
            restore_h,
            ids_h,
            stats_h,
            slots,
        )
    except e:
        failed = True
        print(
            "sortfused tooth ", bug, " failed as expected: ", String(e), sep=""
        )
    assert_true(
        failed,
        "broken sortfused variant "
        + String(bug)
        + " passed: the check has no teeth",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = ref_order_d
    _ = ref_start_d
    _ = ref_restore_d
    _ = ref_ids_d
    _ = ref_stats_d
    _ = order_h
    _ = start_h
    _ = restore_h
    _ = ids_h
    _ = stats_h
    _ = ref_start_h
    _ = ref_ids_h
    _ = ref_stats_h


# ===----------------------------------------------------------------------=== #
# The source-major scatter+quantize kernel
# ===----------------------------------------------------------------------=== #


def run_source_major_case[
    num_experts: Int = NUM_EXPERTS, num_cols: Int = LATENT, topk: Int = TOPK
](num_tokens: Int, spread: String, check_rows: Int, ctx: DeviceContext) raises:
    """One shape x spread: the source-major kernel against the
    `moe_create_indices` + gather+quantize reference, per the byte-identity
    scope.

    Both kernels run over the same `moe_create_indices` outputs, so their
    rows land at identical sorted positions and the byte check is exact,
    row by row. The index outputs must also survive the source-major pass
    unchanged: the kernel never touches them, and the op's large-P arm
    returns them straight from `moe_create_indices`.
    """
    comptime scale_cols = num_cols // SF
    var slots = num_tokens * topk
    var rows = min(check_rows, slots)
    var topk_h = gen_topk_ids[num_experts, topk](spread, num_tokens, 7)
    var latent_h = gen_latent[num_cols](num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * num_cols)
    var ref_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var ref_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * scale_cols)
    var cand_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var cand_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(num_experts + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(num_experts))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, num_cols))
    var ref_q_tt = TileTensor(ref_q_d, row_major(slots, num_cols))
    var ref_s_tt = TileTensor(ref_s_d, row_major(slots, scale_cols))
    var cand_q_tt = TileTensor(cand_q_d, row_major(slots, num_cols))
    var cand_s_tt = TileTensor(cand_s_d, row_major(slots, scale_cols))

    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    # The index outputs as `moe_create_indices` wrote them.
    var start0_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var ids0_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var stats0_h = ctx.enqueue_create_host_buffer[.uint32](2)
    ctx.enqueue_copy(start0_h, start_d)
    ctx.enqueue_copy(ids0_h, ids_d)
    ctx.enqueue_copy(stats0_h, stats_d)

    # The reference: the gather+quantize kernel over the same order.
    moe_gather_quantize_mx_amd(
        ctx, ref_q_tt, ref_s_tt, latent_tt, order_tt, topk
    )
    # The candidate: the source-major kernel over the same restore.
    moe_source_scatter_quantize_mx_amd[num_cols=num_cols](
        ctx, cand_q_tt, cand_s_tt, latent_tt, restore_tt, topk
    )

    var start_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var ids_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var ref_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        rows * num_cols
    )
    var ref_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        rows * scale_cols
    )
    var cand_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        rows * num_cols
    )
    var cand_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        rows * scale_cols
    )
    ctx.enqueue_copy(start_h, start_d)
    ctx.enqueue_copy(ids_h, ids_d)
    ctx.enqueue_copy(stats_h, stats_d)
    ctx.enqueue_copy(ref_q_h, ref_q_d)
    ctx.enqueue_copy(ref_s_h, ref_s_d)
    ctx.enqueue_copy(cand_q_h, cand_q_d)
    ctx.enqueue_copy(cand_s_h, cand_s_d)
    ctx.synchronize()

    for g in range(num_experts + 1):
        assert_equal(
            Int(start_h[g]),
            Int(start0_h[g]),
            "the source-major pass changed expert_start_indices",
        )
    for g in range(num_experts):
        assert_equal(
            Int(ids_h[g]),
            Int(ids0_h[g]),
            "the source-major pass changed expert_ids",
        )
    for e in range(2):
        assert_equal(
            Int(stats_h[e]),
            Int(stats0_h[e]),
            "the source-major pass changed expert_usage_stats",
        )
    check_bytes_equal[num_cols, scale_cols](
        ref_q_h, ref_s_h, cand_q_h, cand_s_h, rows
    )
    print(
        "ok srcmajor M=",
        num_tokens,
        " E=",
        num_experts,
        " K=",
        num_cols,
        " spread=",
        spread,
        " rows=",
        rows,
        sep="",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = ref_q_d
    _ = ref_s_d
    _ = cand_q_d
    _ = cand_s_d
    _ = start0_h
    _ = ids0_h
    _ = stats0_h
    _ = start_h
    _ = ids_h
    _ = stats_h
    _ = ref_q_h
    _ = ref_s_h
    _ = cand_q_h
    _ = cand_s_h


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(SORT_BLOCK_THREADS)
    )
)
@__name(t"test_moe_source_broken_kernel_{BUG}")
def _moe_source_broken_kernel[
    BUG: Int
](
    output: MutPointer[Scalar[.float8_e4m3fn], MutUntrackedOrigin],
    scales: MutPointer[Scalar[.float8_e8m0fnu], MutUntrackedOrigin],
    input: ImmPointer[Scalar[.bfloat16], ImmUntrackedOrigin],
    restore: ImmPointer[UInt32, ImmUntrackedOrigin],
    topk: Int32,
    num_slots: Int32,
):
    """The frozen negative control for the source-major checks: the
    production kernel with one deliberate bug per `BUG` (1: the restore
    entries read from the next token; 2: the first pair's first scale store
    dropped; 3: the out-of-range sentinel guard removed). Deliberately
    independent of the production kernel so the control cannot drift into
    sharing the bug it guards against."""
    var t = block_idx.x
    var tid = thread_idx.x
    var col = 8 * tid
    comptime in_align = align_of[SIMD[.bfloat16, 8]]()
    comptime out_align = align_of[SIMD[.float8_e4m3fn, 8]]()

    with PDL[overlap_at_beginning=True]():
        var vec = input.unsafe_offset(t * LATENT + col).unsafe_load[
            width=8, alignment=in_align
        ]()
        var quantized: SIMD[.float8_e4m3fn, 8]
        var e8m0_scale: Float8_e8m0fnu
        quantized, e8m0_scale = quantize_mxfp8_lane_group[
            width=8,
            out_dtype=.float8_e4m3fn,
            scales_dtype=.float8_e8m0fnu,
            SF_VECTOR_SIZE=SF,
        ](vec)
        var pair0 = t * Int(topk)
        for k in range(Int(topk)):
            var pos = restore.unsafe_offset(pair0 + k).unsafe_load()
            comptime if BUG == 1:
                # The next token's restore entries: every sorted position
                # still gets written exactly once, but by the wrong token's
                # row. (A rotation inside the token's own pair set would
                # pass: every pair of a token carries the same row.)
                var nt = Int(num_slots) // Int(topk)
                pos = restore.unsafe_offset(
                    ((t + 1) % nt) * Int(topk) + k
                ).unsafe_load()
            comptime if BUG == 3:
                # The sentinel guard removed: the out-of-range pair writes
                # through pos == num_slots into the canary row.
                output.unsafe_offset(Int(pos) * LATENT + col).unsafe_store[
                    alignment=out_align
                ](quantized)
                if tid % 4 == 0:
                    scales.unsafe_offset(
                        Int(pos) * SCALE_COLS + tid // 4
                    ).unsafe_store(e8m0_scale)
            else:
                if Int(pos) < Int(num_slots):
                    output.unsafe_offset(Int(pos) * LATENT + col).unsafe_store[
                        alignment=out_align
                    ](quantized)
                    comptime if BUG == 2:
                        # The first pair's first scale block dropped.
                        if not (k == 0 and tid == 0) and tid % 4 == 0:
                            scales.unsafe_offset(
                                Int(pos) * SCALE_COLS + tid // 4
                            ).unsafe_store(e8m0_scale)
                    else:
                        if tid % 4 == 0:
                            scales.unsafe_offset(
                                Int(pos) * SCALE_COLS + tid // 4
                            ).unsafe_store(e8m0_scale)


def run_source_major_out_of_range_case[
    num_experts: Int = NUM_EXPERTS, num_cols: Int = LATENT, topk: Int = TOPK
](num_tokens: Int, ctx: DeviceContext) raises:
    """Out-of-range expert ids through the production source-scatter kernel.

    `moe_create_indices` pins an out-of-range pair's restore entry to the
    `num_slots` sentinel, and the source-scatter kernel must skip that
    pair: no payload row, no scale row, and no write past the output
    buffers -- the sentinel is one past the last row, so a missing guard
    writes off the end. The candidate buffers carry one canary row past
    `slots` to catch that write, and every row no valid pair owns keeps
    the canary pattern.
    """
    comptime scale_cols = num_cols // SF
    var slots = num_tokens * topk
    var topk_h = gen_topk_ids[num_experts, topk]("u", num_tokens, 7)
    topk_h[1] = Int32(num_experts)
    topk_h[slots // 2] = Int32(-1)
    topk_h[slots - 3] = Int32(65536 + num_experts // 8)
    var latent_h = gen_latent[num_cols](num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * num_cols)
    # One canary row past `slots`: inside the allocation, outside the
    # tensor, exactly where an unguarded sentinel write lands.
    var cand_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
        (slots + 1) * num_cols
    )
    var cand_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        (slots + 1) * scale_cols
    )
    var ref_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * num_cols)
    var ref_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * scale_cols)

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)
    var canary_q = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        (slots + 1) * num_cols
    )
    var canary_s = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        (slots + 1) * scale_cols
    )
    for p in range((slots + 1) * num_cols):
        canary_q[p] = bitcast[.float8_e4m3fn](UInt8(0xAA))
    for p in range((slots + 1) * scale_cols):
        canary_s[p] = bitcast[.float8_e8m0fnu](UInt8(0xAA))
    ctx.enqueue_copy(cand_q_d, canary_q)
    ctx.enqueue_copy(cand_s_d, canary_s)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(num_experts + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(num_experts))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, num_cols))
    var cand_q_tt = TileTensor(cand_q_d, row_major(slots, num_cols))
    var cand_s_tt = TileTensor(cand_s_d, row_major(slots, scale_cols))
    var ref_q_tt = TileTensor(ref_q_d, row_major(slots, num_cols))
    var ref_s_tt = TileTensor(ref_s_d, row_major(slots, scale_cols))

    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    moe_source_scatter_quantize_mx_amd[num_cols=num_cols](
        ctx, cand_q_tt, cand_s_tt, latent_tt, restore_tt, topk
    )

    var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
    var start_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    ctx.enqueue_copy(order_h, order_d)
    ctx.enqueue_copy(restore_h, restore_d)
    ctx.enqueue_copy(start_h, start_d)
    ctx.synchronize()

    var valid = 0
    for i in range(slots):
        var e = Int(topk_h[i])
        if e >= 0 and e < num_experts:
            valid += 1
    assert_equal(
        Int(start_h[num_experts]),
        valid,
        "srcmajor out-of-range: start[E] is not the valid-pair count",
    )
    for i in range(slots):
        var e = Int(topk_h[i])
        if e >= 0 and e < num_experts:
            assert_true(
                Int(restore_h[i]) < valid,
                (
                    "srcmajor out-of-range: a valid pair's restore entry is"
                    " outside the valid prefix"
                ),
            )
        else:
            assert_equal(
                Int(restore_h[i]),
                slots,
                (
                    "srcmajor out-of-range: an out-of-range id must hold the"
                    " num_slots sentinel"
                ),
            )

    # The reference payload runs over the reference order with its
    # unwritten tail pinned to slot 0, so the gather never reads an entry
    # no kernel writes.
    var patched = ctx.enqueue_create_host_buffer[.uint32](slots)
    for p in range(slots):
        patched[p] = order_h[p] if p < valid else UInt32(0)
    ctx.enqueue_copy(order_d, patched)
    moe_gather_quantize_mx_amd(
        ctx, ref_q_tt, ref_s_tt, latent_tt, order_tt, topk
    )
    var ref_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * num_cols
    )
    var ref_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * scale_cols
    )
    var cand_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        (slots + 1) * num_cols
    )
    var cand_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        (slots + 1) * scale_cols
    )
    ctx.enqueue_copy(ref_q_h, ref_q_d)
    ctx.enqueue_copy(ref_s_h, ref_s_d)
    ctx.enqueue_copy(cand_q_h, cand_q_d)
    ctx.enqueue_copy(cand_s_h, cand_s_d)
    ctx.synchronize()

    var cand_q = TileTensor(cand_q_h, row_major(slots + 1, num_cols))
    var cand_s = TileTensor(cand_s_h, row_major(slots + 1, scale_cols))
    var ref_q = TileTensor(ref_q_h, row_major(slots, num_cols))
    var ref_s = TileTensor(ref_s_h, row_major(slots, scale_cols))
    # A source token may appear at several sorted rows; the quantize is
    # deterministic, so any reference row with that source carries the
    # bytes.
    var rep = List[Int](length=slots, fill=-1)
    for p in range(valid):
        var src = Int(order_h[p]) // topk
        if rep[src] < 0:
            rep[src] = p
    for i in range(slots):
        var e = Int(topk_h[i])
        if e >= 0 and e < num_experts:
            var p = Int(restore_h[i])
            var ref_p = rep[i // topk]
            assert_true(
                ref_p >= 0, "srcmajor out-of-range: unknown reference source"
            )
            for c in range(num_cols):
                assert_equal(
                    bitcast[.uint8](cand_q[p, c]),
                    bitcast[.uint8](ref_q[ref_p, c]),
                    (
                        "srcmajor out-of-range: a valid pair's payload row"
                        " differs from the gather+quantize kernel's"
                    ),
                )
            for c in range(scale_cols):
                assert_equal(
                    bitcast[.uint8](cand_s[p, c]),
                    bitcast[.uint8](ref_s[ref_p, c]),
                    "srcmajor out-of-range: a valid pair's scale row differs",
                )
    for p in range(valid, slots + 1):
        for c in range(num_cols):
            assert_equal(
                bitcast[.uint8](cand_q[p, c]),
                UInt8(0xAA),
                (
                    "srcmajor out-of-range: an out-of-range pair wrote a"
                    " payload row"
                ),
            )
        for c in range(scale_cols):
            assert_equal(
                bitcast[.uint8](cand_s[p, c]),
                UInt8(0xAA),
                "srcmajor out-of-range: an out-of-range pair wrote a scale",
            )
    print(
        "ok srcmajor out-of-range M=",
        num_tokens,
        " E=",
        num_experts,
        " K=",
        num_cols,
        sep="",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = cand_q_d
    _ = cand_s_d
    _ = ref_q_d
    _ = ref_s_d
    _ = topk_h
    _ = latent_h
    _ = canary_q
    _ = canary_s


def run_source_major_teeth[bug: Int](ctx: DeviceContext) raises:
    """Proves the source-major checks fail on the bug classes they guard
    against: 1, a restore entry read from the wrong token; 2, a dropped
    scale block; 3, a missing out-of-range sentinel guard, which must
    write past the output when the sentinel is not skipped."""
    comptime num_tokens = 8
    var slots = num_tokens * TOPK
    var topk_h = gen_topk_ids("u", num_tokens, 7)
    var latent_h = gen_latent(num_tokens, 11)

    var topk_d = ctx.enqueue_create_buffer[.int32](slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
    var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
    var ids_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
    var stats_d = ctx.enqueue_create_buffer[.uint32](2)
    var latent_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * LATENT)
    # One row past the output end: the canary a missing sentinel guard
    # would write through `restore[i] == num_slots`.
    var cand_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
        (slots + 1) * LATENT
    )
    var cand_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        (slots + 1) * SCALE_COLS
    )
    var ref_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
    var ref_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slots * SCALE_COLS)

    ctx.enqueue_copy(topk_d, topk_h)
    ctx.enqueue_copy(latent_d, latent_h)

    var order_tt = TileTensor(order_d, row_major(slots))
    var start_tt = TileTensor(start_d, row_major(NUM_EXPERTS + 1))
    var restore_tt = TileTensor(restore_d, row_major(slots))
    var ids_tt = TileTensor(ids_d, row_major(NUM_EXPERTS))
    var stats_tt = TileTensor(stats_d, row_major(2))
    var topk_tt = TileTensor(topk_d, row_major(slots))
    var latent_tt = TileTensor(latent_d, row_major(num_tokens, LATENT))
    var ref_q_tt = TileTensor(ref_q_d, row_major(slots, LATENT))
    var ref_s_tt = TileTensor(ref_s_d, row_major(slots, SCALE_COLS))

    comptime if bug == 3:
        # One out-of-range expert id: the pair never lands in the CSR and
        # its restore entry holds the num_slots sentinel.
        topk_h[3] = Int32(-1)
        ctx.enqueue_copy(topk_d, topk_h)

    moe_create_indices[input_type=DType.int32](
        order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
    )
    moe_gather_quantize_mx_amd(
        ctx, ref_q_tt, ref_s_tt, latent_tt, order_tt, TOPK
    )

    comptime if bug == 3:
        var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
        ctx.enqueue_copy(restore_h, restore_d)
        ctx.synchronize()
        assert_equal(
            Int(restore_h[3]),
            slots,
            "the out-of-range pair must hold the num_slots sentinel",
        )
        # Poison the output, canary row included.
        ctx.enqueue_memset[.float8_e4m3fn](
            cand_q_d, bitcast[.float8_e4m3fn](UInt8(0xAA))
        )
        # Direct enqueue: the launcher's shape checks would refuse the
        # over-allocated output the canary needs.
        ctx.enqueue_function[_moe_source_broken_kernel[3]](
            cand_q_d.unsafe_ptr(),
            cand_s_d.unsafe_ptr(),
            latent_d.unsafe_ptr(),
            restore_d.unsafe_ptr(),
            Int32(TOPK),
            Int32(slots),
            grid_dim=(num_tokens, 1, 1),
            block_dim=(SORT_BLOCK_THREADS, 1, 1),
        )
        var out_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
            (slots + 1) * LATENT
        )
        ctx.enqueue_copy(out_h, cand_q_d)
        ctx.synchronize()
        var out = TileTensor(out_h, row_major(slots + 1, LATENT))
        var canary_intact = True
        for c in range(LATENT):
            if bitcast[.uint8](out[slots, c]) != UInt8(0xAA):
                canary_intact = False
        assert_true(
            not canary_intact,
            "the missing-sentinel tooth passed: the guard is load-bearing",
        )
        print("srcmajor tooth 3 failed as expected: canary row corrupted")
        _ = topk_d
        _ = order_d
        _ = start_d
        _ = restore_d
        _ = ids_d
        _ = stats_d
        _ = latent_d
        _ = cand_q_d
        _ = cand_s_d
        _ = ref_q_d
        _ = ref_s_d
        _ = restore_h
        _ = out_h
        return

    # Poison the candidate scales so a dropped store is unmistakable: the
    # scale path never emits a negative E8M0, so 0xAA cannot match.
    ctx.enqueue_memset[.float8_e8m0fnu](
        cand_s_d, bitcast[.float8_e8m0fnu](UInt8(0xAA))
    )
    ctx.enqueue_function[_moe_source_broken_kernel[bug]](
        cand_q_d.unsafe_ptr(),
        cand_s_d.unsafe_ptr(),
        latent_d.unsafe_ptr(),
        restore_d.unsafe_ptr(),
        Int32(TOPK),
        Int32(slots),
        grid_dim=(num_tokens, 1, 1),
        block_dim=(SORT_BLOCK_THREADS, 1, 1),
    )

    var ref_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](slots * LATENT)
    var ref_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    var cand_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        slots * LATENT
    )
    var cand_s_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slots * SCALE_COLS
    )
    ctx.enqueue_copy(ref_q_h, ref_q_d)
    ctx.enqueue_copy(ref_s_h, ref_s_d)
    ctx.enqueue_copy(cand_q_h, cand_q_d)
    ctx.enqueue_copy(cand_s_h, cand_s_d)
    ctx.synchronize()

    var failed = False
    try:
        check_bytes_equal(ref_q_h, ref_s_h, cand_q_h, cand_s_h, slots)
    except e:
        failed = True
        print(
            "srcmajor tooth ", bug, " failed as expected: ", String(e), sep=""
        )
    assert_true(
        failed,
        "broken srcmajor variant "
        + String(bug)
        + " passed: the check has no teeth",
    )

    _ = topk_d
    _ = order_d
    _ = start_d
    _ = restore_d
    _ = ids_d
    _ = stats_d
    _ = latent_d
    _ = cand_q_d
    _ = cand_s_d
    _ = ref_q_d
    _ = ref_s_d
    _ = ref_q_h
    _ = ref_s_h
    _ = cand_q_h
    _ = cand_s_h


def run_source_major_support_checks() raises:
    """The dispatch envelope of the op's large-P arm: the source-major
    kernel covers the production row widths, and the rows that would
    overflow its CTA keep the sorted-row kernel."""
    # The K3 production width, and the arm the op picks at the prefill
    # chunk: the sort-fused kernel is out (its slot cap) and the
    # source-major kernel is in.
    assert_true(
        moe_source_scatter_quantize_supported(LATENT),
        "the production latent width must take the source-major kernel",
    )
    assert_true(
        not moe_sort_gather_quantize_fused_supported(
            8192 * TOPK, NUM_EXPERTS, LATENT
        )
        and moe_source_scatter_quantize_supported(LATENT),
        "the prefill chunk must take the source-major arm",
    )
    # The minimum row: one 4-lane scale group.
    assert_true(
        moe_source_scatter_quantize_supported(32),
        "a 32-column row is the minimum the kernel covers",
    )
    # Not a whole MX block; and below one 4-lane scale group.
    assert_true(
        not moe_source_scatter_quantize_supported(24),
        "a 24-column row is not a whole number of MX blocks",
    )
    assert_true(
        not moe_source_scatter_quantize_supported(16),
        "a 16-column row cannot form one 4-lane scale group",
    )
    # The 1024-thread CTA cap: at and past it.
    assert_true(
        moe_source_scatter_quantize_supported(8192),
        "an 8192-column row is the widest CTA the kernel covers",
    )
    assert_true(
        not moe_source_scatter_quantize_supported(8192 + 32),
        "past 1024 threads the sorted-row kernel takes over",
    )
    print("ok srcmajor support envelope")


def run_sort_support_checks() raises:
    """The dispatch envelope: what the one-launch kernel covers, and what
    falls back to the two-launch chain."""
    comptime K = LATENT
    # Every decode shape the K3 path runs: M in 1..64 at top-16.
    for m in [1, 8, 16, 32, 64]:
        assert_true(
            moe_sort_gather_quantize_fused_supported(m * TOPK, NUM_EXPERTS, K),
            "decode shape M=" + String(m) + " must take the fused kernel",
        )
    # The prefill chunk: the two-launch chain.
    assert_true(
        not moe_sort_gather_quantize_fused_supported(
            8192 * TOPK, NUM_EXPERTS, K
        ),
        "the prefill chunk must keep the two-launch chain",
    )
    # Past the dispatch bound.
    assert_true(
        not moe_sort_gather_quantize_fused_supported(1028, NUM_EXPERTS, K),
        "past the dispatch bound the chain takes over",
    )
    # The 16-bit key packing.
    assert_true(
        not moe_sort_gather_quantize_fused_supported(512, 65536, K),
        "65536 experts do not fit the 16-bit key packing",
    )
    # The fused-reduce query budget: 2 + ceil(E/P) <= 64.
    assert_true(
        not moe_sort_gather_quantize_fused_supported(8, NUM_EXPERTS, K),
        "8 slots exceed the fused-reduce query budget at 896 experts",
    )
    # The register-held keys: 4 per thread over a K // 8-thread block.
    assert_true(
        not moe_sort_gather_quantize_fused_supported(512, NUM_EXPERTS, 64),
        "a 64-column row cannot hold 512 keys in registers",
    )
    # The reduce's partials: one LDS column per warp, so the block may hold
    # at most MAX_WARPS warps. The widest supported row is exactly that
    # block.
    var widest_cols = 8 * MAX_WARPS * WARP_SIZE
    assert_true(
        moe_sort_gather_quantize_fused_supported(
            1024, NUM_EXPERTS, widest_cols
        ),
        "a row as wide as the reduce's partial columns must be supported",
    )
    assert_true(
        not moe_sort_gather_quantize_fused_supported(
            1024, NUM_EXPERTS, widest_cols + 8 * WARP_SIZE
        ),
        "one warp past the reduce's partial columns must fall back",
    )
    # The LDS budget: the histogram shares it with the reduce's 4352 bytes,
    # which leaves 15296 experts of a 64 KiB workgroup.
    assert_true(
        moe_sort_gather_quantize_fused_supported(512, 15296, K),
        "the widest histogram one workgroup's LDS holds must be supported",
    )
    assert_true(
        not moe_sort_gather_quantize_fused_supported(512, 15297, K),
        "one expert past the LDS histogram budget must fall back",
    )
    print("ok sortfused support envelope")


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        for m in [1, 8, 16, 32, 64]:
            for spread in ["u", "s", "p"]:
                run_case(m, spread, ctx)

        # Fail fast: the launcher rejects an order that does not hold one
        # slot per (token, expert) pair.
        var slots = 8 * TOPK
        var latent_d = ctx.enqueue_create_buffer[.bfloat16](8 * LATENT)
        var order_d = ctx.enqueue_create_buffer[.uint32](slots - 1)
        var fused_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
            slots * LATENT
        )
        var fused_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            slots * SCALE_COLS
        )
        var latent_tt = TileTensor(latent_d, row_major(8, LATENT))
        var order_tt = TileTensor(order_d, row_major(slots - 1))
        var fused_q_tt = TileTensor(fused_q_d, row_major(slots, LATENT))
        var fused_s_tt = TileTensor(fused_s_d, row_major(slots, SCALE_COLS))
        with assert_raises(contains="slots"):
            moe_gather_quantize_mx_amd(
                ctx, fused_q_tt, fused_s_tt, latent_tt, order_tt, TOPK
            )
        # ... and one whose scales row is not `K // 32` wide.
        var good_order_d = ctx.enqueue_create_buffer[.uint32](slots)
        var short_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            slots * (SCALE_COLS - 1)
        )
        var short_s_tt = TileTensor(short_s_d, row_major(slots, SCALE_COLS - 1))
        with assert_raises(contains="shape"):
            moe_gather_quantize_mx_amd(
                ctx,
                fused_q_tt,
                short_s_tt,
                latent_tt,
                TileTensor(good_order_d, row_major(slots)),
                TOPK,
            )
        _ = good_order_d
        _ = short_s_d
        _ = latent_d
        _ = order_d
        _ = fused_q_d
        _ = fused_s_d

        run_teeth[1](ctx)
        run_teeth[2](ctx)
        run_teeth[3](ctx)
        run_gather_order_guard_case(ctx)

        for m in [1, 8, 16, 32, 64]:
            for spread in ["u", "s", "p"]:
                run_sort_case(m, spread, ctx)

        # A small latent width shrinks the CTA below one warp, taking the
        # kernel's LDS-atomic fold instead of the warp-shuffle reduce.
        for spread in ["u", "s", "p"]:
            run_sort_case[64, 128, 4](16, spread, ctx)

        # A non-power-of-two fan-out exercises the source-row division
        # instead of a shift.
        for spread in ["u", "s", "p"]:
            run_sort_case[64, 256, 3](8, spread, ctx)

        # Out-of-range ids must file no slot, exactly as the reference
        # does: at production width (warp-shuffle reduce) and at a width
        # below one warp (the LDS-atomic fold).
        run_sort_out_of_range_case(8, ctx)
        run_sort_out_of_range_case[64, 256, 4](8, ctx)

        run_sort_teeth[1](ctx)
        run_sort_teeth[2](ctx)
        run_sort_teeth[3](ctx)
        run_sort_support_checks()

        # The source-major kernel: the op's large-P arm, at the prefill
        # chunk sizes it serves. Full-row checks at M=2048; at M=8192 a
        # 256-row prefix keeps the D2H copy small.
        for spread in ["u", "s", "p"]:
            run_source_major_case(2048, spread, 2048 * TOPK, ctx)
        for spread in ["u", "s", "p"]:
            run_source_major_case(8192, spread, 256, ctx)

        # Out-of-range ids through the production source-scatter kernel: the
        # sentinel pairs write nothing, and nothing lands past the buffers.
        run_source_major_out_of_range_case(8, ctx)
        run_source_major_out_of_range_case[64, 256, 4](8, ctx)

        # Fail fast: the launcher rejects a restore that does not hold one
        # slot per (token, expert) pair.
        var sm_slots = 8 * TOPK
        var sm_latent_d = ctx.enqueue_create_buffer[.bfloat16](8 * LATENT)
        var sm_restore_d = ctx.enqueue_create_buffer[.uint32](sm_slots - 1)
        var sm_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
            sm_slots * LATENT
        )
        var sm_s_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            sm_slots * SCALE_COLS
        )
        var sm_latent_tt = TileTensor(sm_latent_d, row_major(8, LATENT))
        var sm_restore_tt = TileTensor(sm_restore_d, row_major(sm_slots - 1))
        var sm_q_tt = TileTensor(sm_q_d, row_major(sm_slots, LATENT))
        var sm_s_tt = TileTensor(sm_s_d, row_major(sm_slots, SCALE_COLS))
        with assert_raises(contains="slots"):
            moe_source_scatter_quantize_mx_amd[num_cols=LATENT](
                ctx, sm_q_tt, sm_s_tt, sm_latent_tt, sm_restore_tt, TOPK
            )
        _ = sm_latent_d
        _ = sm_restore_d
        _ = sm_q_d
        _ = sm_s_d

        run_source_major_teeth[1](ctx)
        run_source_major_teeth[2](ctx)
        run_source_major_teeth[3](ctx)
        run_source_major_support_checks()
        print("all fused MoE routing-glue checks passed")
