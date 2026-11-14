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
"""Microbench of the K3 MoE routing glue chain, MI355X decode.

Times, per token count M and expert-spread mode, the three kernels every K3
MoE layer runs between the router and the first grouped GEMM, through the
same entry points the graph ops use:

1. `moe_create_indices` with int32 topk ids over 896 experts (`nn/moe.mojo`).
2. The `mo.gather` op's GPU entry (`gather_scatter.mojo`'s closure `gather`,
   no index prefetch): latent [M, 3584] bf16 rows fanned out to [M*16, 3584]
   by `token_expert_order // 16`, with the division applied inside the index
   load the way the graph fuses the cast/div producer into the gather.
3. `quantize_mx_amd`, the MI355X arm of `mo.quantize.dynamic.block.scaled`:
   bf16 [M*16, 3584] in, e4m3fn out plus [M*16, 112] E8M0 scales.

plus the two alternatives that replace launches 2+3, each timed alone
and chained behind `moe_create_indices`:

4. `moe_gather_quantize_mx_amd`: the fused gather+quantize, one
   launch, sorted row p reading source row `order[p] // 16` directly.
5. The quantize-unique variant (`opta`): `quantize_mx_amd` over the M
   unique token rows plus a bench-local byte gather that fans the fp8 rows
   and scale bytes out to the sorted rows (two launches).

6. `moe_sort_gather_quantize_mx_amd`: the expert sort, the fan-out gather
   and the MXFP8 quantize in one launch -- one CTA per source pair ranks
   its pair among all P `(expert, slot)` keys with a fused block count,
   writes its share of the CSR indices, then quants its source row at the
   sorted position. Timed here through the production launcher; the graph
   op dispatches on its support envelope, falling back to
   `moe_create_indices` + the source-major kernel outside it.

7. `moe_source_scatter_quantize_mx_amd`: one CTA per source token row
   quants the row once and fans the fp8 bytes out to the up-to-topk sorted
   positions `restore_token_order` gives its pairs, reading each bf16 row
   once instead of topk times. Timed alone (`srcmajor`) and
   chained behind `moe_create_indices` (`srcmajorchain`), plus the `abc`
   mode timing `fusedchain` vs `optachain` vs `srcmajorchain` in
   interleaved triplets; the graph op's large-P arm runs it after
   `moe_create_indices`.

plus chain modes running the pieces back-to-back per iteration, which is
what one layer pays per step, and interleaved A/B modes (`ab` timing
`fusedchain` against the sort-fused kernel, `abk` the two kernels alone,
so `sortfused - fused` isolates the sort phase's cost).

Correctness cross-checks against host references run once per shape before
any timing. `--broken=1` swaps in deliberately wrong references (one tap
dropped from each 32-element block max; divisor 15 for the gather source row)
that must fail, proving the checks have teeth. The sort-fused kernel's own
failure-mode checks live in the committed GPU test
(`test_moe_gather_quantize.mojo`), next to its byte-identity checks.

Timing instruments, per kernel:

- Batch mean: `reps` batches of exactly `iters` back-to-back launches through
  the `Bench`/`bencher_iter_custom` framework (matches
  the production regime of back-to-back launches in a captured graph).
- Tail probe: `--tail-iters` individually synchronized launches, host
  monotonic around launch+sync, giving a per-launch distribution (p50/p99/
  p999/max). Includes the launch+sync round trip, so its level sits above
  the batch mean.
- Batch probe: `--batch-tail` batches of `--batch-size` launches, per-launch
  mean per batch, for tail visibility in the back-to-back regime.

Output: the framework's human report plus machine-readable `RESULT,...`
lines.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.builtin.sort import sort
from std.math import ceil
from std.memory import bitcast
from std.random import random_float64, random_ui64, seed
from std.sys import align_of
from std.time import monotonic

from max.benchmark import bencher_iter_custom
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import (
    Coord,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.coord import DynamicCoord
from nn.gather_scatter import Axis, gather
from nn.moe import moe_create_indices
from linalg.block_scaled_quantization import (
    moe_gather_quantize_mx_amd,
    moe_sort_gather_quantize_fused_supported,
    moe_sort_gather_quantize_mx_amd,
    moe_source_scatter_quantize_mx_amd,
    quantize_mx_amd,
)
from linalg.block_scaled_utils import compute_mxfp8_block_scale
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from internal_utils import arg_parse
from std.testing import assert_equal, assert_true

# K3 production constants: top-16 of 896 experts, latent width 3584, bf16
# activations quantized to FP8-MX with one E8M0 scale per 32 elements.
comptime NUM_EXPERTS = 896
comptime TOPK = 16
comptime LATENT = 3584
comptime SF = MXFP8_SF_VECTOR_SIZE
comptime SCALE_COLS = LATENT // SF

# Above this many fanned rows the host-side checks sample a prefix of the
# output rows instead of reading every row back.
comptime FULL_CHECK_MAX_ROWS = 4096


# ===----------------------------------------------------------------------=== #
# Launchers (the production entry points)
# ===----------------------------------------------------------------------=== #


def launch_create_indices(
    order: MutTileTensor[.uint32, ...],
    start: MutTileTensor[.uint32, ...],
    restore: MutTileTensor[.uint32, ...],
    ids: MutTileTensor[.int32, ...],
    stats: MutTileTensor[.uint32, ...],
    topk: ImmTileTensor[.int32, ...],
    ctx: DeviceContext,
) raises:
    moe_create_indices[input_type=DType.int32](
        order, start, restore, ids, stats, topk, ctx
    )


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


def launch_quant(
    quantized: MutTileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    scales: MutTileTensor[.float8_e8m0fnu, address_space=.GENERIC, ...],
    gathered: ImmTileTensor[.bfloat16, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    # The graph op dispatches through `quantize_dynamic_block_scaled`, whose
    # MI355X arm is exactly this call with the default thread cap.
    quantize_mx_amd(ctx, quantized, scales, gathered)


def launch_fused(
    quantized: MutTileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    scales: MutTileTensor[.float8_e8m0fnu, address_space=.GENERIC, ...],
    latent: ImmTileTensor[.bfloat16, address_space=.GENERIC, ...],
    order: ImmTileTensor[.uint32, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """The fused gather+quantize: sorted row p reads latent[order[p] // 16]."""
    moe_gather_quantize_mx_amd(ctx, quantized, scales, latent, order, TOPK)


# The quantize-unique variant's second launch: fan the unique rows' fp8
# bytes and scale bytes out to the sorted rows. Bench-local, timing only
# the byte movement.
@__name("bench_gather_quantized_rows_kernel")
def _gather_quantized_rows_kernel[
    bytes_per_thread: Int,
    out_data_layout: TensorLayout,
    out_scales_layout: TensorLayout,
    in_data_layout: TensorLayout,
    in_scales_layout: TensorLayout,
    order_layout: TensorLayout,
](
    out_data: TileTensor[.float8_e4m3fn, out_data_layout, MutUntrackedOrigin],
    out_scales: TileTensor[
        .float8_e8m0fnu, out_scales_layout, MutUntrackedOrigin
    ],
    in_data: TileTensor[.float8_e4m3fn, in_data_layout, ImmUntrackedOrigin],
    in_scales: TileTensor[
        .float8_e8m0fnu, in_scales_layout, ImmUntrackedOrigin
    ],
    order: TileTensor[.uint32, order_layout, ImmUntrackedOrigin],
    topk: Int32,
):
    var row = block_idx.x
    var src = Int(order[row]) // Int(topk)
    var t = thread_idx.x
    var base = t * bytes_per_thread
    if base < LATENT:
        out_data.store((row, base), in_data.load[bytes_per_thread]((src, base)))
    if t < SCALE_COLS:
        out_scales.store((row, t), in_scales.load((src, t)))


def launch_gather_bytes(
    quantized: MutTileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    scales: MutTileTensor[.float8_e8m0fnu, address_space=.GENERIC, ...],
    quantized_unique: ImmTileTensor[
        .float8_e4m3fn, address_space=.GENERIC, ...
    ],
    scales_unique: ImmTileTensor[.float8_e8m0fnu, address_space=.GENERIC, ...],
    order: ImmTileTensor[.uint32, address_space=.GENERIC, ...],
    num_slots: Int,
    ctx: DeviceContext,
) raises:
    comptime bytes_per_thread = 8
    comptime kernel = _gather_quantized_rows_kernel[
        bytes_per_thread,
        type_of(quantized).LayoutType,
        type_of(scales).LayoutType,
        type_of(quantized_unique).LayoutType,
        type_of(scales_unique).LayoutType,
        type_of(order).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        quantized,
        scales,
        quantized_unique,
        scales_unique,
        order,
        Int32(TOPK),
        grid_dim=(num_slots, 1, 1),
        block_dim=(LATENT // bytes_per_thread, 1, 1),
    )


def launch_source_major(
    quantized: MutTileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    scales: MutTileTensor[.float8_e8m0fnu, address_space=.GENERIC, ...],
    latent: ImmTileTensor[.bfloat16, address_space=.GENERIC, ...],
    restore: ImmTileTensor[.uint32, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """The source-major kernel: one CTA per source row, restore-driven."""
    moe_source_scatter_quantize_mx_amd[num_cols=LATENT](
        ctx, quantized, scales, latent, restore, TOPK
    )


# ===----------------------------------------------------------------------=== #
# The one-launch sort+gather+quantize, timed through its production launcher.
# ===----------------------------------------------------------------------=== #


def launch_sort_fused(
    quantized: MutTileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    scales: MutTileTensor[.float8_e8m0fnu, address_space=.GENERIC, ...],
    latent: ImmTileTensor[.bfloat16, address_space=.GENERIC, ...],
    order: MutTileTensor[.uint32, address_space=.GENERIC, ...],
    start: MutTileTensor[.uint32, address_space=.GENERIC, ...],
    restore: MutTileTensor[.uint32, address_space=.GENERIC, ...],
    ids: MutTileTensor[.int32, address_space=.GENERIC, ...],
    stats: MutTileTensor[.uint32, address_space=.GENERIC, ...],
    topk: ImmTileTensor[.int32, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """The production one-launch sort+gather+quantize over the flat ids."""
    moe_sort_gather_quantize_mx_amd[num_experts=NUM_EXPERTS, num_cols=LATENT](
        ctx,
        quantized,
        scales,
        latent,
        order,
        start,
        restore,
        ids,
        stats,
        topk,
        TOPK,
    )


# ===----------------------------------------------------------------------=== #
# Host references and correctness checks
# ===----------------------------------------------------------------------=== #


def check_create_indices(
    topk: List[Int32],
    order: HostBuffer[.uint32],
    start: HostBuffer[.uint32],
    restore: HostBuffer[.uint32],
    ids: HostBuffer[.int32],
    stats: HostBuffer[.uint32],
) raises:
    """Checks the device output against a histogram + CSR host reference.

    Within an expert group the kernel's slot order follows an atomic cursor
    and is explicitly not load-bearing, so the order check is per-group
    membership plus `restore` as its inverse, not exact positions.
    """
    var slots = len(topk)
    var counts = List[Int](length=NUM_EXPERTS, fill=0)
    for i in range(slots):
        var e = Int(topk[i])
        assert_true(0 <= e and e < NUM_EXPERTS, "topk id out of expert range")
        counts[e] += 1
    var ref_start = List[Int](length=NUM_EXPERTS + 1, fill=0)
    var max_count = 0
    for g in range(NUM_EXPERTS):
        ref_start[g + 1] = ref_start[g] + counts[g]
        max_count = max(max_count, counts[g])
    for g in range(NUM_EXPERTS + 1):
        assert_equal(
            Int(start[g]), ref_start[g], "expert_start_indices mismatch"
        )
    for g in range(NUM_EXPERTS):
        assert_equal(Int(ids[g]), g, "expert_ids mismatch")
    assert_equal(Int(stats[0]), max_count, "usage max count mismatch")
    assert_equal(Int(stats[1]), NUM_EXPERTS, "usage expert count mismatch")

    var seen = List[Bool](length=slots, fill=False)
    for p in range(slots):
        var i = Int(order[p])
        assert_true(0 <= i and i < slots, "order value out of range")
        assert_true(not seen[i], "order visits a slot twice")
        seen[i] = True
        var g = Int(topk[i])
        assert_true(
            ref_start[g] <= p and p < ref_start[g + 1],
            "slot sorted outside its expert's range",
        )
        assert_equal(Int(restore[i]), p, "restore is not order's inverse")


def check_gather(
    latent: List[Scalar[.bfloat16]],
    order: HostBuffer[.uint32],
    gathered: HostBuffer[.bfloat16],
    num_rows: Int,
    num_tokens: Int,
    broken: Bool,
) raises:
    """Checks `gathered[p, :] == latent[order[p] // 16, :]` bitwise.

    `broken` uses divisor 15 in the reference, which must fail.
    """
    var divisor = 15 if broken else TOPK
    var gathered_rows = TileTensor(gathered, row_major(num_rows, LATENT))
    for p in range(num_rows):
        var src = Int(order[p]) // divisor
        if src >= num_tokens:
            raise Error(
                "gather source row out of range: the reference disagrees"
            )
        for c in range(LATENT):
            var want = latent[src * LATENT + c]
            var got = gathered_rows[p, c]
            assert_equal(
                got.to_bits(),
                want.to_bits(),
                "gathered row does not match its source latent row",
            )


def check_quant(
    gathered: HostBuffer[.bfloat16],
    quantized: HostBuffer[.float8_e4m3fn],
    scales: HostBuffer[.float8_e8m0fnu],
    num_rows: Int,
    broken: Bool,
) raises:
    """Checks the MXFP8 output against a host reference.

    The reference recomputes each 32-element block max from the input and
    derives the E8M0 scale and data bytes with `compute_mxfp8_block_scale`,
    the same scale derivation the kernel uses (the f32 -> E8M0 cast rounds
    toward +infinity, per CUTLASS, not the MX spec's floor). `broken` drops
    one tap from each block max, which must fail.
    """
    var taps = SF - 1 if broken else SF
    comptime min_distinct_scales = 8
    var distinct_scales = List[UInt8]()
    var nonzero = 0
    var total = 0
    var gathered_rows = TileTensor(gathered, row_major(num_rows, LATENT))
    var quantized_rows = TileTensor(quantized, row_major(num_rows, LATENT))
    var scale_rows = TileTensor(scales, row_major(num_rows, SCALE_COLS))
    for row in range(num_rows):
        for block in range(SCALE_COLS):
            var col = block * SF
            var group_max = Float32(0)
            for e in range(taps):
                group_max = max(
                    group_max, abs(gathered_rows[row, col + e].cast[.float32]())
                )
            var want_scale: Float8_e8m0fnu
            var multiplier: Float32
            var dead: Bool
            want_scale, multiplier, dead = compute_mxfp8_block_scale[
                .float8_e8m0fnu
            ](group_max)
            assert_true(not dead, "bench input produced a dead MX block")
            var want_bits = bitcast[.uint8](want_scale)
            assert_equal(
                bitcast[.uint8](scale_rows[row, block]),
                want_bits,
                "E8M0 scale mismatch",
            )
            if want_bits not in distinct_scales:
                distinct_scales.append(want_bits)
            for e in range(SF):
                var want = (
                    gathered_rows[row, col + e].cast[.float32]() * multiplier
                ).cast[.float8_e4m3fn]()
                var got = bitcast[.uint8](quantized_rows[row, col + e])
                assert_equal(
                    got, bitcast[.uint8](want), "quantized data mismatch"
                )
                if got != UInt8(0):
                    nonzero += 1
                total += 1
    # Non-vacuity: a single scale for the whole tensor would satisfy the
    # comparisons without exercising the block grouping.
    assert_true(
        len(distinct_scales) >= min(min_distinct_scales, SCALE_COLS),
        "input exercised too few distinct scales",
    )
    assert_true(nonzero > total // 2, "most output bytes are zero")


def check_fused_vs_chain(
    chain_quantized: HostBuffer[.float8_e4m3fn],
    chain_scales: HostBuffer[.float8_e8m0fnu],
    alt_quantized: HostBuffer[.float8_e4m3fn],
    alt_scales: HostBuffer[.float8_e8m0fnu],
    num_rows: Int,
    broken: Bool,
) raises:
    """Checks a candidate kernel's bytes against the chain's, per sorted row.

    The chain (gather then `quantize_mx_amd`) is the reference; a candidate
    kernel must reproduce its payload and scale bytes exactly. `broken`
    compares each row against the chain row shifted by one, the output a
    kernel reading `order[p + 1]` would produce, which must fail.
    """
    var alt_q = TileTensor(alt_quantized, row_major(num_rows, LATENT))
    var chain_q = TileTensor(chain_quantized, row_major(num_rows, LATENT))
    var alt_s = TileTensor(alt_scales, row_major(num_rows, SCALE_COLS))
    var chain_s = TileTensor(chain_scales, row_major(num_rows, SCALE_COLS))
    var shift = 1 if broken else 0
    for p in range(num_rows):
        var ref_row = (p + shift) % num_rows
        for c in range(LATENT):
            assert_equal(
                bitcast[.uint8](alt_q[p, c]),
                bitcast[.uint8](chain_q[ref_row, c]),
                "fusion candidate payload row differs from the chain's",
            )
        for c in range(SCALE_COLS):
            assert_equal(
                bitcast[.uint8](alt_s[p, c]),
                bitcast[.uint8](chain_s[ref_row, c]),
                "fusion candidate scale row differs from the chain's",
            )


def check_sort_indices(
    topk: List[Int32],
    ref_start: HostBuffer[.uint32],
    ref_ids: HostBuffer[.int32],
    ref_stats: HostBuffer[.uint32],
    order: HostBuffer[.uint32],
    start: HostBuffer[.uint32],
    restore: HostBuffer[.uint32],
    ids: HostBuffer[.int32],
    stats: HostBuffer[.uint32],
) raises:
    """Checks the sort-fused kernel's index outputs against the reference's.

    `expert_start_indices`, `expert_ids` and `expert_usage_stats` must be
    byte-equal to `moe_create_indices`'s outputs on the same ids.
    `token_expert_order`/`restore_token_order` follow a relaxed-atomic
    cursor in the reference, so the order check is per-group membership,
    bijection and `restore == order^-1` -- never exact intra-expert
    positions.
    """
    var slots = len(topk)
    for g in range(NUM_EXPERTS + 1):
        assert_equal(
            Int(start[g]),
            Int(ref_start[g]),
            "sortfused expert_start_indices differs from moe_create_indices'",
        )
    for g in range(NUM_EXPERTS):
        assert_equal(
            Int(ids[g]), Int(ref_ids[g]), "sortfused expert_ids mismatch"
        )
    assert_equal(
        Int(stats[0]), Int(ref_stats[0]), "sortfused usage max count mismatch"
    )
    assert_equal(
        Int(stats[1]),
        Int(ref_stats[1]),
        "sortfused usage expert count mismatch",
    )
    var seen = List[Bool](length=slots, fill=False)
    for p in range(slots):
        var i = Int(order[p])
        assert_true(0 <= i and i < slots, "sortfused order value out of range")
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
    for i in range(slots):
        assert_true(seen[i], "sortfused order misses a slot")


def check_sort_bytes(
    num_tokens: Int,
    ref_order: HostBuffer[.uint32],
    ref_quantized: HostBuffer[.float8_e4m3fn],
    ref_scales: HostBuffer[.float8_e8m0fnu],
    order: HostBuffer[.uint32],
    quantized: HostBuffer[.float8_e4m3fn],
    scales: HostBuffer[.float8_e8m0fnu],
    num_rows: Int,
) raises:
    """Checks the sort-fused kernel's payload bytes against the fused
    kernel's, per source row.

    The two kernels may land a source row at different intra-expert
    positions (both orders are valid groupings), so each sort-fused row is
    compared against the fused row holding the same source token: the
    quantize is deterministic, so every fused row with the same source is
    byte-identical. Requires the full row range (`num_rows == slots`), which
    the decode sizes always take.
    """
    var slots = num_tokens * TOPK
    assert_true(
        num_rows == slots,
        "the sortfused byte check reads full decode sizes",
    )
    var cand_q = TileTensor(quantized, row_major(num_rows, LATENT))
    var ref_q = TileTensor(ref_quantized, row_major(num_rows, LATENT))
    var cand_s = TileTensor(scales, row_major(num_rows, SCALE_COLS))
    var ref_s = TileTensor(ref_scales, row_major(num_rows, SCALE_COLS))
    var rep = List[Int](length=num_tokens, fill=-1)
    for p in range(slots):
        var src = Int(ref_order[p]) // TOPK
        if rep[src] < 0:
            rep[src] = p
    for p in range(num_rows):
        var src = Int(order[p]) // TOPK
        var ref_p = rep[src]
        assert_true(ref_p >= 0, "sortfused row maps to an unknown source")
        for c in range(LATENT):
            assert_equal(
                bitcast[.uint8](cand_q[p, c]),
                bitcast[.uint8](ref_q[ref_p, c]),
                "sortfused payload row differs from the fused kernel's",
            )
        for c in range(SCALE_COLS):
            assert_equal(
                bitcast[.uint8](cand_s[p, c]),
                bitcast[.uint8](ref_s[ref_p, c]),
                "sortfused scale row differs from the fused kernel's",
            )


def run_broken_check[
    CheckFn: def() raises -> None
](check: CheckFn, label: String) raises -> Bool:
    """Runs a deliberately wrong reference check, reporting an expected
    failure."""
    try:
        check()
    except e:
        print("BROKEN-CHECK " + label + " failed as expected: ", String(e))
        return True
    return False


# ===----------------------------------------------------------------------=== #
# Timing
# ===----------------------------------------------------------------------=== #


def per_launch_us(total_ns: Int, launches: Int) -> Float64:
    """Mean microseconds per launch from a nanosecond total."""
    return Float64(total_ns) / Float64(launches) / 1000.0


def bench_launches[
    LaunchType: ImplicitlyCopyable & def(DeviceContext[]) raises -> None,
](
    mut b: Bench, launch: LaunchType, bench_id: BenchId, ctx: DeviceContext[]
) raises:
    @inline(.always)
    def bench_fn(mut b: Bencher) raises {var launch, imm}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext[]) raises {imm}:
            launch(ctx)

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(bench_fn, bench_id)


def tail_probe[
    LaunchType: def(DeviceContext[]) raises -> None,
](
    launch: LaunchType,
    ctx: DeviceContext[],
    samples: Int,
    warmup: Int,
    csv_path: String,
) raises -> List[Float64]:
    """Per-launch distribution: launch, sync, host-monotonic around each."""
    for _ in range(warmup):
        launch(ctx)
    ctx.synchronize()
    var times = List[Float64](capacity=samples)
    for _ in range(samples):
        var t0 = monotonic()
        launch(ctx)
        ctx.synchronize()
        times.append(per_launch_us(monotonic() - t0, 1))
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("sample_us\n")
            for sample in times:
                f.write(String(sample) + "\n")
    return times^


def batch_probe[
    LaunchType: def(DeviceContext[]) raises -> None,
](
    launch: LaunchType,
    ctx: DeviceContext[],
    batches: Int,
    batch_size: Int,
    warmup: Int,
    csv_path: String,
) raises -> List[Float64]:
    """Per-launch mean over `batch_size` back-to-back launches, per batch."""
    for _ in range(warmup):
        launch(ctx)
    ctx.synchronize()
    var means = List[Float64](capacity=batches)
    for _ in range(batches):
        var ns = ctx.execution_time(launch, batch_size)
        means.append(per_launch_us(ns, batch_size))
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("per_launch_us\n")
            for mean in means:
                f.write(String(mean) + "\n")
    return means^


def percentile_of(samples: List[Float64], q: Float64) raises -> Float64:
    """Nearest-rank percentile of an ascending-sorted sample list."""
    var n = len(samples)
    var idx = Int(ceil(q * Float64(n))) - 1
    if idx < 0:
        idx = 0
    if idx > n - 1:
        idx = n - 1
    return samples[idx]


def report_distribution(
    name: String,
    m_tokens: Int,
    spread: String,
    samples: List[Float64],
    extra: String,
) raises:
    var ordered = samples.copy()
    sort(ordered)
    print(
        "RESULT,",
        name,
        ",M=",
        m_tokens,
        ",spread=",
        spread,
        ",p50_us=",
        percentile_of(ordered, 0.50),
        ",p90_us=",
        percentile_of(ordered, 0.90),
        ",p99_us=",
        percentile_of(ordered, 0.99),
        ",p999_us=",
        percentile_of(ordered, 0.999),
        ",max_us=",
        ordered[len(ordered) - 1],
        ",n=",
        len(ordered),
        extra,
        sep="",
    )


def selected(kernels: String, name: String) -> Bool:
    """Exact comma-delimited token match, so "fused" never picks "fusedchain".
    """
    return ("," + name + ",") in ("," + kernels + ",")


def csv_path_for(
    prefix: String, name: String, m_tokens: Int, spread: String, kind: String
) -> String:
    if not prefix:
        return ""
    return (
        prefix
        + "_"
        + name
        + "_M"
        + String(m_tokens)
        + "_"
        + spread
        + "_"
        + kind
        + ".csv"
    )


def emit_batch_result(
    mut b: Bench,
    base: Int,
    name: String,
    m_tokens: Int,
    spread: String,
    reps: Int,
    iters: Int,
) raises:
    """Prints the batch-mean RESULT line for the reports added since `base`."""
    var total_ns = 0
    var total_iters = 0
    for r in range(base, len(b.info_vec)):
        for run in b.info_vec[r].result.runs:
            total_ns += run.duration
            total_iters += run.iterations
    assert_true(
        total_iters == reps * iters,
        "delivered iterations differ from requested",
    )
    var mean_us = per_launch_us(total_ns, total_iters)
    print(
        "RESULT,",
        name,
        ",M=",
        m_tokens,
        ",spread=",
        spread,
        ",mean_us=",
        mean_us,
        ",iters=",
        total_iters,
        sep="",
    )


def run_probes[
    LaunchType: def(DeviceContext[]) raises -> None,
](
    launch: LaunchType,
    ctx: DeviceContext[],
    name: String,
    m_tokens: Int,
    spread: String,
    tail_iters: Int,
    batch_tail: Int,
    batch_size: Int,
    warmup: Int,
    csv_prefix: String,
) raises:
    if tail_iters > 0:
        var samples = tail_probe(
            launch,
            ctx,
            tail_iters,
            warmup,
            csv_path_for(csv_prefix, name, m_tokens, spread, "tail"),
        )
        report_distribution(name, m_tokens, spread, samples, ",probe=sync")
    if batch_tail > 0:
        var means = batch_probe(
            launch,
            ctx,
            batch_tail,
            batch_size,
            warmup,
            csv_path_for(csv_prefix, name, m_tokens, spread, "batch"),
        )
        report_distribution(
            name, m_tokens, spread, means, ",probe=batch" + String(batch_size)
        )


def bench_and_probe[
    LaunchType: ImplicitlyCopyable & def(DeviceContext[]) raises -> None,
](
    mut b: Bench,
    launch: LaunchType,
    name: String,
    m_tokens: Int,
    spread: String,
    reps: Int,
    iters: Int,
    tail_iters: Int,
    batch_tail: Int,
    batch_size: Int,
    warmup: Int,
    csv_prefix: String,
    ctx: DeviceContext[],
) raises:
    """Benches one launch, prints its batch-mean RESULT line, then probes it."""
    var base = len(b.info_vec)
    bench_launches(
        b,
        launch,
        BenchId(name, input_id=String(m_tokens, "tok/", spread, "spread")),
        ctx,
    )
    emit_batch_result(b, base, name, m_tokens, spread, reps, iters)
    run_probes(
        launch,
        ctx,
        name,
        m_tokens,
        spread,
        tail_iters,
        batch_tail,
        batch_size,
        warmup,
        csv_prefix,
    )


# ===----------------------------------------------------------------------=== #
# Input generation
# ===----------------------------------------------------------------------=== #


def pick_distinct_experts(pool: Int) raises -> List[Int32]:
    """Draws TOPK distinct expert ids uniformly from [0, pool)."""
    var picked = List[Int32]()
    while len(picked) < TOPK:
        var e = Int32(random_ui64(0, UInt64(pool - 1)))
        if e not in picked:
            picked.append(e)
    return picked^


def gen_topk_ids(
    spread: String, num_tokens: Int, rng_seed: Int
) raises -> List[Int32]:
    """Seeded expert ids: 16 distinct experts per token, uniform, skewed, or
    drawn from a 96-expert hot pool so most experts stay empty while a few
    hold several tokens."""
    seed(rng_seed)
    var ids = List[Int32](length=num_tokens * TOPK, fill=Int32(0))
    if spread == "s":
        # Max cursor contention: every token picks the same 16 experts.
        var picked = pick_distinct_experts(NUM_EXPERTS)
        for i in range(num_tokens * TOPK):
            ids[i] = picked[i % TOPK]
    else:
        var pool = NUM_EXPERTS
        if spread == "e":
            pool = 96
        for t in range(num_tokens):
            var picked = pick_distinct_experts(pool)
            for k in range(TOPK):
                ids[t * TOPK + k] = picked[k]
    return ids^


def gen_latent(
    num_tokens: Int, rng_seed: Int
) raises -> List[Scalar[.bfloat16]]:
    """Deterministic bf16 latent rows on a per-block magnitude ladder.

    The ladder cycles the 32-element block magnitude over 2^-8..2^0 so the
    quant check sees many distinct E8M0 scales and no dead blocks.
    """
    seed(rng_seed + 1)
    var bf16 = List[Scalar[.bfloat16]]()
    for _ in range(num_tokens):
        for col in range(LATENT):
            var mag = Float64(1 << ((col // SF) % 9)) / 256.0
            bf16.append((random_float64(-4.0, 4.0) * mag).cast[.bfloat16]())
    return bf16^


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def main() raises:
    var m_tokens = arg_parse("M", 32)
    var spread_arg = arg_parse("spread", "u")
    var spread = "s" if spread_arg == "s" or spread_arg == "skew" else "u"
    var iters = arg_parse("iters", 1000)
    var reps = arg_parse("reps", 5)
    var warmup = max(arg_parse("warmup", 200), 1)
    var tail_iters = arg_parse("tail-iters", 2000)
    var batch_tail = arg_parse("batch-tail", 0)
    var batch_size = arg_parse("batch-size", 64)
    var do_check = arg_parse("check", True)
    var broken = arg_parse("broken", False)
    var check_rows = arg_parse("check-rows", 256)
    var csv_prefix = arg_parse("csv", "")
    var rng_seed = arg_parse("seed", 20260214)
    var ab_pairs = arg_parse("ab-pairs", 3)
    var kernels = arg_parse(
        "kernels",
        (
            "idx,gather,quant,chain,fused,fusedchain,opta,optachain,"
            "sortfusedchain,ab,abk,srcmajor,srcmajorchain,abc"
        ),
    )

    var slots = m_tokens * TOPK
    var num_check_rows = slots if slots <= FULL_CHECK_MAX_ROWS else check_rows
    if broken:
        # `--broken` checks every row, not a prefix.
        num_check_rows = slots

    var topk_h = gen_topk_ids(spread, m_tokens, rng_seed)
    var latent_bf16_h = gen_latent(m_tokens, rng_seed)

    print(
        "bench_moe_glue_chain: M=",
        m_tokens,
        " slots=",
        slots,
        " spread=",
        spread,
        " iters=",
        iters,
        "x",
        reps,
        " tail_iters=",
        tail_iters,
        " check_rows=",
        num_check_rows,
        " broken=",
        broken,
    )

    with DeviceContext() as ctx:
        var topk_d = ctx.enqueue_create_buffer[.int32](slots)
        var order_d = ctx.enqueue_create_buffer[.uint32](slots)
        var start_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
        var restore_d = ctx.enqueue_create_buffer[.uint32](slots)
        var ids_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
        var stats_d = ctx.enqueue_create_buffer[.uint32](2)
        var latent_d = ctx.enqueue_create_buffer[.bfloat16](m_tokens * LATENT)
        var gathered_d = ctx.enqueue_create_buffer[.bfloat16](slots * LATENT)
        var quantized_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
            slots * LATENT
        )
        var scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            slots * SCALE_COLS
        )
        # The quantize-unique variant's targets.
        var quantized_unique_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
            m_tokens * LATENT
        )
        var scales_unique_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            m_tokens * SCALE_COLS
        )
        # The sort-fused kernel's targets.
        var order2_d = ctx.enqueue_create_buffer[.uint32](slots)
        var start2_d = ctx.enqueue_create_buffer[.uint32](NUM_EXPERTS + 1)
        var restore2_d = ctx.enqueue_create_buffer[.uint32](slots)
        var ids2_d = ctx.enqueue_create_buffer[.int32](NUM_EXPERTS)
        var stats2_d = ctx.enqueue_create_buffer[.uint32](2)
        var quant2_d = ctx.enqueue_create_buffer[.float8_e4m3fn](slots * LATENT)
        var scales2_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
            slots * SCALE_COLS
        )

        ctx.enqueue_copy(topk_d, topk_h)
        ctx.enqueue_copy(latent_d, latent_bf16_h)

        var order_tt = TileTensor(order_d, row_major(slots))
        var start_tt = TileTensor(start_d, row_major(NUM_EXPERTS + 1))
        var restore_tt = TileTensor(restore_d, row_major(slots))
        var ids_tt = TileTensor(ids_d, row_major(NUM_EXPERTS))
        var stats_tt = TileTensor(stats_d, row_major(2))
        var topk_tt = TileTensor(topk_d, row_major(slots))
        var latent_tt = TileTensor(latent_d, row_major(m_tokens, LATENT))
        var gathered_tt = TileTensor(gathered_d, row_major(slots, LATENT))
        var quantized_tt = TileTensor(quantized_d, row_major(slots, LATENT))
        var scales_tt = TileTensor(scales_d, row_major(slots, SCALE_COLS))
        var quantized_unique_tt = TileTensor(
            quantized_unique_d, row_major(m_tokens, LATENT)
        )
        var scales_unique_tt = TileTensor(
            scales_unique_d, row_major(m_tokens, SCALE_COLS)
        )
        var order2_tt = TileTensor(order2_d, row_major(slots))
        var start2_tt = TileTensor(start2_d, row_major(NUM_EXPERTS + 1))
        var restore2_tt = TileTensor(restore2_d, row_major(slots))
        var ids2_tt = TileTensor(ids2_d, row_major(NUM_EXPERTS))
        var stats2_tt = TileTensor(stats2_d, row_major(2))
        var quant2_tt = TileTensor(quant2_d, row_major(slots, LATENT))
        var scales2_tt = TileTensor(scales2_d, row_major(slots, SCALE_COLS))

        # One round of every kernel, then the correctness checks.
        launch_create_indices(
            order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
        )
        launch_gather(
            latent_tt,
            TileTensor(
                build_gather_indices(order_d, TOPK, slots, ctx),
                row_major(slots),
            ),
            gathered_tt,
            ctx,
        )
        launch_quant(quantized_tt, scales_tt, gathered_tt, ctx)

        var order_h = ctx.enqueue_create_host_buffer[.uint32](slots)
        var start_h = ctx.enqueue_create_host_buffer[.uint32](NUM_EXPERTS + 1)
        var restore_h = ctx.enqueue_create_host_buffer[.uint32](slots)
        var ids_h = ctx.enqueue_create_host_buffer[.int32](NUM_EXPERTS)
        var stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
        var gathered_h = ctx.enqueue_create_host_buffer[.bfloat16](
            num_check_rows * LATENT
        )
        var quantized_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
            num_check_rows * LATENT
        )
        var scales_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
            num_check_rows * SCALE_COLS
        )
        ctx.enqueue_copy(order_h, order_d)
        ctx.enqueue_copy(start_h, start_d)
        ctx.enqueue_copy(restore_h, restore_d)
        ctx.enqueue_copy(ids_h, ids_d)
        ctx.enqueue_copy(stats_h, stats_d)
        # D2H copies transfer the destination's size, so these read the
        # prefix of the device buffers.
        ctx.enqueue_copy(gathered_h, gathered_d)
        ctx.enqueue_copy(quantized_h, quantized_d)
        ctx.enqueue_copy(scales_h, scales_d)

        # The fused kernel overwrites the chain's output buffers, so its
        # bytes land in their own host copy for the comparison.
        launch_fused(quantized_tt, scales_tt, latent_tt, order_tt, ctx)
        var alt_quantized_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
            num_check_rows * LATENT
        )
        var alt_scales_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
            num_check_rows * SCALE_COLS
        )
        ctx.enqueue_copy(alt_quantized_h, quantized_d)
        ctx.enqueue_copy(alt_scales_h, scales_d)
        ctx.synchronize()

        # The sort-fused kernel's host copies.
        var order2_h = ctx.enqueue_create_host_buffer[.uint32](slots)
        var start2_h = ctx.enqueue_create_host_buffer[.uint32](NUM_EXPERTS + 1)
        var restore2_h = ctx.enqueue_create_host_buffer[.uint32](slots)
        var ids2_h = ctx.enqueue_create_host_buffer[.int32](NUM_EXPERTS)
        var stats2_h = ctx.enqueue_create_host_buffer[.uint32](2)
        var sort_quantized_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
            num_check_rows * LATENT
        )
        var sort_scales_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
            num_check_rows * SCALE_COLS
        )
        # The production kernel covers decode slot counts only; outside its
        # support envelope the graph op falls back to `moe_create_indices`
        # plus a separate quantize kernel, and the sortfused checks and
        # timing modes self-skip.
        var sort_probe_live = moe_sort_gather_quantize_fused_supported(
            slots, NUM_EXPERTS, LATENT
        )

        if do_check or broken:
            check_create_indices(
                topk_h, order_h, start_h, restore_h, ids_h, stats_h
            )
            print("CHECK create_indices ok (M=", m_tokens, ")")
            if broken:

                def broken_gather() raises {imm}:
                    check_gather(
                        latent_bf16_h,
                        order_h,
                        gathered_h,
                        num_check_rows,
                        m_tokens,
                        True,
                    )

                var gather_failed = run_broken_check(broken_gather, "gather")

                def broken_quant() raises {imm}:
                    check_quant(
                        gathered_h, quantized_h, scales_h, num_check_rows, True
                    )

                var quant_failed = run_broken_check(broken_quant, "quant")
                if not gather_failed:
                    raise Error(
                        "broken gather reference (divisor 15) passed: the"
                        " check has no teeth"
                    )
                if not quant_failed:
                    raise Error(
                        "broken quant reference (dropped tap) passed: the"
                        " check has no teeth"
                    )

                def broken_fused() raises {imm}:
                    check_fused_vs_chain(
                        quantized_h,
                        scales_h,
                        alt_quantized_h,
                        alt_scales_h,
                        num_check_rows,
                        True,
                    )

                var fused_failed = run_broken_check(broken_fused, "fused")
                if not fused_failed:
                    raise Error(
                        "broken fused reference (shifted row) passed: the"
                        " check has no teeth"
                    )
                print(
                    "BROKEN-CHECK all three broken references failed as"
                    " required"
                )
                print(
                    "BROKEN-CHECK the sortfused kernel's own failure-mode"
                    " checks live in test_moe_gather_quantize.mojo"
                )
                return
            check_gather(
                latent_bf16_h,
                order_h,
                gathered_h,
                num_check_rows,
                m_tokens,
                False,
            )
            print("CHECK gather ok (rows=", num_check_rows, " of ", slots, ")")
            check_quant(
                gathered_h, quantized_h, scales_h, num_check_rows, False
            )
            print("CHECK quant ok (rows=", num_check_rows, " of ", slots, ")")
            check_fused_vs_chain(
                quantized_h,
                scales_h,
                alt_quantized_h,
                alt_scales_h,
                num_check_rows,
                False,
            )
            print("CHECK fused ok (rows=", num_check_rows, " of ", slots, ")")
            # The quantize-unique variant: quantize the unique rows, then
            # fan the bytes out.
            launch_quant(quantized_unique_tt, scales_unique_tt, latent_tt, ctx)
            launch_gather_bytes(
                quantized_tt,
                scales_tt,
                quantized_unique_tt,
                scales_unique_tt,
                order_tt,
                slots,
                ctx,
            )
            ctx.enqueue_copy(alt_quantized_h, quantized_d)
            ctx.enqueue_copy(alt_scales_h, scales_d)
            ctx.synchronize()
            check_fused_vs_chain(
                quantized_h,
                scales_h,
                alt_quantized_h,
                alt_scales_h,
                num_check_rows,
                False,
            )
            print("CHECK opta ok (rows=", num_check_rows, " of ", slots, ")")
            # The source-major kernel: same create_indices order, so its
            # rows land at exactly the chain's sorted positions.
            launch_source_major(
                quantized_tt, scales_tt, latent_tt, restore_tt, ctx
            )
            ctx.enqueue_copy(alt_quantized_h, quantized_d)
            ctx.enqueue_copy(alt_scales_h, scales_d)
            ctx.synchronize()
            check_fused_vs_chain(
                quantized_h,
                scales_h,
                alt_quantized_h,
                alt_scales_h,
                num_check_rows,
                False,
            )
            print(
                "CHECK srcmajor ok (rows=", num_check_rows, " of ", slots, ")"
            )

            if sort_probe_live:
                # Refresh the fused bytes as the reference (the opta check
                # reused the buffer), then run the sort-fused kernel.
                launch_fused(quantized_tt, scales_tt, latent_tt, order_tt, ctx)
                ctx.enqueue_copy(alt_quantized_h, quantized_d)
                ctx.enqueue_copy(alt_scales_h, scales_d)
                launch_sort_fused(
                    quant2_tt,
                    scales2_tt,
                    latent_tt,
                    order2_tt,
                    start2_tt,
                    restore2_tt,
                    ids2_tt,
                    stats2_tt,
                    topk_tt,
                    ctx,
                )
                ctx.enqueue_copy(order2_h, order2_d)
                ctx.enqueue_copy(start2_h, start2_d)
                ctx.enqueue_copy(restore2_h, restore2_d)
                ctx.enqueue_copy(ids2_h, ids2_d)
                ctx.enqueue_copy(stats2_h, stats2_d)
                ctx.enqueue_copy(sort_quantized_h, quant2_d)
                ctx.enqueue_copy(sort_scales_h, scales2_d)
                ctx.synchronize()
                check_sort_indices(
                    topk_h,
                    start_h,
                    ids_h,
                    stats_h,
                    order2_h,
                    start2_h,
                    restore2_h,
                    ids2_h,
                    stats2_h,
                )
                print("CHECK sortfused indices ok (M=", m_tokens, ")")
                check_sort_bytes(
                    m_tokens,
                    order_h,
                    alt_quantized_h,
                    alt_scales_h,
                    order2_h,
                    sort_quantized_h,
                    sort_scales_h,
                    num_check_rows,
                )
                print("CHECK sortfused bytes ok (rows=", num_check_rows, ")")

        # Timing. One Bench collects every kernel's human report.
        var bench = Bench(
            BenchConfig(
                num_repetitions=reps,
                max_iters=iters,
                num_warmup_iters=warmup,
                max_batch_size=iters,
                min_runtime_secs=0.0,
                max_runtime_secs=0.001,
            )
        )

        @inline(.always)
        def launch_idx(ctx: DeviceContext[]) raises {imm}:
            launch_create_indices(
                order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
            )

        var gather_indices_tt = TileTensor(
            build_gather_indices(order_d, TOPK, slots, ctx), row_major(slots)
        )

        @inline(.always)
        def launch_gather_fn(ctx: DeviceContext[]) raises {imm}:
            launch_gather(latent_tt, gather_indices_tt, gathered_tt, ctx)

        @inline(.always)
        def launch_quant_fn(ctx: DeviceContext[]) raises {imm}:
            launch_quant(quantized_tt, scales_tt, gathered_tt, ctx)

        @inline(.always)
        def launch_chain(ctx: DeviceContext[]) raises {imm}:
            launch_create_indices(
                order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
            )
            launch_gather(latent_tt, gather_indices_tt, gathered_tt, ctx)
            launch_quant(quantized_tt, scales_tt, gathered_tt, ctx)

        @inline(.always)
        def launch_fused_fn(ctx: DeviceContext[]) raises {imm}:
            launch_fused(quantized_tt, scales_tt, latent_tt, order_tt, ctx)

        @inline(.always)
        def launch_fused_chain(ctx: DeviceContext[]) raises {imm}:
            launch_create_indices(
                order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
            )
            launch_fused(quantized_tt, scales_tt, latent_tt, order_tt, ctx)

        @inline(.always)
        def launch_opta(ctx: DeviceContext[]) raises {imm}:
            launch_quant(quantized_unique_tt, scales_unique_tt, latent_tt, ctx)
            launch_gather_bytes(
                quantized_tt,
                scales_tt,
                quantized_unique_tt,
                scales_unique_tt,
                order_tt,
                slots,
                ctx,
            )

        @inline(.always)
        def launch_opta_chain(ctx: DeviceContext[]) raises {imm}:
            launch_create_indices(
                order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
            )
            launch_quant(quantized_unique_tt, scales_unique_tt, latent_tt, ctx)
            launch_gather_bytes(
                quantized_tt,
                scales_tt,
                quantized_unique_tt,
                scales_unique_tt,
                order_tt,
                slots,
                ctx,
            )

        @inline(.always)
        def launch_sort_fused_fn(ctx: DeviceContext[]) raises {imm}:
            launch_sort_fused(
                quant2_tt,
                scales2_tt,
                latent_tt,
                order2_tt,
                start2_tt,
                restore2_tt,
                ids2_tt,
                stats2_tt,
                topk_tt,
                ctx,
            )

        @inline(.always)
        def launch_source_major_fn(ctx: DeviceContext[]) raises {imm}:
            launch_source_major(
                quantized_tt, scales_tt, latent_tt, restore_tt, ctx
            )

        @inline(.always)
        def launch_source_major_chain(ctx: DeviceContext[]) raises {imm}:
            launch_create_indices(
                order_tt, start_tt, restore_tt, ids_tt, stats_tt, topk_tt, ctx
            )
            launch_source_major(
                quantized_tt, scales_tt, latent_tt, restore_tt, ctx
            )

        if selected(kernels, "idx"):
            bench_and_probe(
                bench,
                launch_idx,
                "moe_create_indices",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "gather"):
            bench_and_probe(
                bench,
                launch_gather_fn,
                "gather",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "quant"):
            bench_and_probe(
                bench,
                launch_quant_fn,
                "quantize_mx",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "chain"):
            bench_and_probe(
                bench,
                launch_chain,
                "chain",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "fused"):
            bench_and_probe(
                bench,
                launch_fused_fn,
                "moe_gather_quantize",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "fusedchain"):
            bench_and_probe(
                bench,
                launch_fused_chain,
                "fusedchain",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "opta"):
            bench_and_probe(
                bench,
                launch_opta,
                "opta",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "optachain"):
            bench_and_probe(
                bench,
                launch_opta_chain,
                "optachain",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "sortfusedchain"):
            if sort_probe_live:
                bench_and_probe(
                    bench,
                    launch_sort_fused_fn,
                    "sortfusedchain",
                    m_tokens,
                    spread,
                    reps,
                    iters,
                    tail_iters,
                    batch_tail,
                    batch_size,
                    warmup,
                    csv_prefix,
                    ctx,
                )
            else:
                print(
                    "SKIP sortfusedchain: slots=",
                    slots,
                    " outside the sort-fused kernel's decode range",
                )

        if selected(kernels, "ab"):
            if sort_probe_live:
                for _ in range(warmup):
                    launch_fused_chain(ctx)
                    launch_sort_fused_fn(ctx)
                ctx.synchronize()
                var total_fused = 0
                var total_sort = 0
                for pair in range(ab_pairs):
                    var t_fused = ctx.execution_time(launch_fused_chain, iters)
                    var t_sort = ctx.execution_time(launch_sort_fused_fn, iters)
                    total_fused += t_fused
                    total_sort += t_sort
                    print(
                        "RESULT,ab_pair,",
                        pair,
                        ",M=",
                        m_tokens,
                        ",spread=",
                        spread,
                        ",fusedchain_us=",
                        per_launch_us(t_fused, iters),
                        ",sortfusedchain_us=",
                        per_launch_us(t_sort, iters),
                        sep="",
                    )
                print(
                    "RESULT,ab_mean,M=",
                    m_tokens,
                    ",spread=",
                    spread,
                    ",fusedchain_us=",
                    per_launch_us(total_fused, ab_pairs * iters),
                    ",sortfusedchain_us=",
                    per_launch_us(total_sort, ab_pairs * iters),
                    ",pairs=",
                    ab_pairs,
                    sep="",
                )
            else:
                print(
                    "SKIP ab: slots=",
                    slots,
                    " outside the sort-fused kernel's decode range",
                )

        if selected(kernels, "abk"):
            if sort_probe_live:
                for _ in range(warmup):
                    launch_fused_fn(ctx)
                    launch_sort_fused_fn(ctx)
                ctx.synchronize()
                var total_fused_k = 0
                var total_sort_k = 0
                for pair in range(ab_pairs):
                    var t_fused_k = ctx.execution_time(launch_fused_fn, iters)
                    var t_sort_k = ctx.execution_time(
                        launch_sort_fused_fn, iters
                    )
                    total_fused_k += t_fused_k
                    total_sort_k += t_sort_k
                    print(
                        "RESULT,abk_pair,",
                        pair,
                        ",M=",
                        m_tokens,
                        ",spread=",
                        spread,
                        ",fused_us=",
                        per_launch_us(t_fused_k, iters),
                        ",sortfused_us=",
                        per_launch_us(t_sort_k, iters),
                        ",sort_phase_us=",
                        per_launch_us(t_sort_k - t_fused_k, iters),
                        sep="",
                    )
                print(
                    "RESULT,abk_mean,M=",
                    m_tokens,
                    ",spread=",
                    spread,
                    ",fused_us=",
                    per_launch_us(total_fused_k, ab_pairs * iters),
                    ",sortfused_us=",
                    per_launch_us(total_sort_k, ab_pairs * iters),
                    ",sort_phase_us=",
                    per_launch_us(
                        total_sort_k - total_fused_k, ab_pairs * iters
                    ),
                    ",pairs=",
                    ab_pairs,
                    sep="",
                )
            else:
                print(
                    "SKIP abk: slots=",
                    slots,
                    " outside the sort-fused kernel's decode range",
                )

        if selected(kernels, "srcmajor"):
            bench_and_probe(
                bench,
                launch_source_major_fn,
                "moe_source_scatter_quantize",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "srcmajorchain"):
            bench_and_probe(
                bench,
                launch_source_major_chain,
                "srcmajorchain",
                m_tokens,
                spread,
                reps,
                iters,
                tail_iters,
                batch_tail,
                batch_size,
                warmup,
                csv_prefix,
                ctx,
            )

        if selected(kernels, "abc"):
            # The large-P arm comparison: fusedchain vs optachain vs
            # srcmajorchain, interleaved.
            for _ in range(warmup):
                launch_fused_chain(ctx)
                launch_opta_chain(ctx)
                launch_source_major_chain(ctx)
            ctx.synchronize()
            var total_fused_c = 0
            var total_opta_c = 0
            var total_src_c = 0
            for pair in range(ab_pairs):
                var t_fused_c = ctx.execution_time(launch_fused_chain, iters)
                var t_opta_c = ctx.execution_time(launch_opta_chain, iters)
                var t_src_c = ctx.execution_time(
                    launch_source_major_chain, iters
                )
                total_fused_c += t_fused_c
                total_opta_c += t_opta_c
                total_src_c += t_src_c
                print(
                    "RESULT,abc_pair,",
                    pair,
                    ",M=",
                    m_tokens,
                    ",spread=",
                    spread,
                    ",fusedchain_us=",
                    per_launch_us(t_fused_c, iters),
                    ",optachain_us=",
                    per_launch_us(t_opta_c, iters),
                    ",srcmajorchain_us=",
                    per_launch_us(t_src_c, iters),
                    sep="",
                )
            print(
                "RESULT,abc_mean,M=",
                m_tokens,
                ",spread=",
                spread,
                ",fusedchain_us=",
                per_launch_us(total_fused_c, ab_pairs * iters),
                ",optachain_us=",
                per_launch_us(total_opta_c, ab_pairs * iters),
                ",srcmajorchain_us=",
                per_launch_us(total_src_c, ab_pairs * iters),
                ",pairs=",
                ab_pairs,
                sep="",
            )

        bench.dump_report()

        # Keep every device buffer alive past the last kernel use.
        _ = topk_d
        _ = order_d
        _ = start_d
        _ = restore_d
        _ = ids_d
        _ = stats_d
        _ = latent_d
        _ = gathered_d
        _ = quantized_d
        _ = scales_d
        _ = quantized_unique_d
        _ = scales_unique_d
        _ = order2_d
        _ = start2_d
        _ = restore2_d
        _ = ids2_d
        _ = stats2_d
        _ = quant2_d
        _ = scales2_d
        _ = alt_quantized_h
        _ = alt_scales_h
        _ = order_h
        _ = start_h
        _ = restore_h
        _ = ids_h
        _ = stats_h
        _ = gathered_h
        _ = quantized_h
        _ = scales_h
        _ = order2_h
        _ = start2_h
        _ = restore2_h
        _ = ids2_h
        _ = stats2_h
        _ = sort_quantized_h
        _ = sort_scales_h
    _ = topk_h
    _ = latent_bf16_h
