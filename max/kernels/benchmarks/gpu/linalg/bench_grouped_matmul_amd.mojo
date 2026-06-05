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
"""AMD MXFP4 grouped matmul bench with workload-driven routing.

Inputs describe the workload (total experts, active experts, batch M,
topk, expert skew, optional shared experts) instead of a hand-built
per-slot token list. The bench builds the routing tables (a_offsets,
expert_ids, sti, ei) so we can sweep one knob at a time.

algo selects the kernel (all MXFP4):
  - "dense"      -> mxfp4_grouped_matmul_amd (per-expert offsets, LDS-staged B)
  - "dense_preb" -> mxfp4_grouped_matmul_amd_preb (preshuffled B, direct VGPR loads)
  - "routed"     -> mxfp4_moe_matmul_amd_routed (scattered sort blocks)
"""

from std.math import align_up, ceildiv
from std.memory import bitcast
from std.os import abort
from std.random import seed, shuffle
from std.sys import (
    get_defined_int,
    get_defined_string,
)

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from internal_utils import arg_parse
from internal_utils._utils import InitializationType, init_vector_launch
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.amd import (
    Shuffler,
    mxfp4_grouped_matmul_amd,
    mxfp4_grouped_matmul_amd_preb,
    mxfp4_moe_matmul_amd_routed,
)


# ===----------------------------------------------------------------------=== #
# Skew parsing
# ===----------------------------------------------------------------------=== #


def _parse_csv_ints(s: String) raises -> List[Int]:
    # Accept "," or ";" as delimiter. kbench evals every yaml value as Python,
    # so a comma-separated string becomes a tuple and expands into a sweep
    # dimension that explodes the build matrix. Use ";" in yaml to keep the
    # value as one literal string.
    var stripped = s.strip("[]\"' ")
    var out = List[Int]()
    if stripped.byte_length() == 0:
        return out^
    var normalized = stripped.replace(";", ",")
    for tok in normalized.split(","):
        var t = String(tok).strip()
        if t.byte_length() == 0:
            continue
        out.append(Int(t))
    return out^


# ===----------------------------------------------------------------------=== #
# Bench name
# ===----------------------------------------------------------------------=== #


def _run_name(
    tag: String,
    num_experts: Int,
    num_active_experts: Int,
    M: Int,
    topk: Int,
    N: Int,
    K: Int,
    skew_spec: String,
) -> String:
    return String(
        tag,
        " : E=",
        num_experts,
        " A=",
        num_active_experts,
        " M=",
        M,
        " topk=",
        topk,
        " N=",
        N,
        " K=",
        K,
        " skew=",
        skew_spec,
    )


# ===----------------------------------------------------------------------=== #
# Per-expert route construction
# ===----------------------------------------------------------------------=== #


def _build_per_expert_routes(
    M: Int,
    topk: Int,
    num_active_experts: Int,
    num_shared_experts: Int,
    target_counts: List[Int],
    seed_val: UInt64,
) raises -> List[List[Tuple[Int, Int]]]:
    """For each active expert, list of (token_id, slot_id) routes.

    Shared slots [0, num_shared) are pinned per token to shared experts.
    Remaining slots are randomly shuffled across routed experts so global
    counts match target_counts (no per-token distinctness enforced).
    """
    var per_expert = List[List[Tuple[Int, Int]]]()
    for _ in range(num_active_experts):
        per_expert.append(List[Tuple[Int, Int]]())

    for s in range(num_shared_experts):
        for t in range(M):
            per_expert[s].append((t, s))

    var pool = List[Tuple[Int, Int]]()
    for t in range(M):
        for s in range(num_shared_experts, topk):
            pool.append((t, s))

    seed(Int(seed_val))
    shuffle(pool)

    var idx = 0
    for e in range(num_shared_experts, num_active_experts):
        for _ in range(target_counts[e]):
            per_expert[e].append(pool[idx])
            idx += 1
    return per_expert^


# ===----------------------------------------------------------------------=== #
# Dense path (mxfp4_grouped_matmul_amd)
# ===----------------------------------------------------------------------=== #


def bench_dense[
    num_experts: Int,
    N: Int,
    K: Int,
    num_shared_experts: Int,
    topk: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    M: Int,
    num_active_experts: Int,
    target_counts: List[Int],
    expert_id_pool: List[Int],
    skew_spec: String,
    init_type: InitializationType,
    max_tokens_capacity: Int = 0,
) raises:
    comptime packed_K = K // 2
    comptime scale_K = K // 32

    var total_routes = M * topk
    var total_flops = 2 * total_routes * N * K

    var a_dev = ctx.enqueue_create_buffer[DType.uint8](total_routes * packed_K)
    var b_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var a_scales_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        total_routes * scale_K
    )
    var b_scales_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var c_dev = ctx.enqueue_create_buffer[DType.float32](total_routes * N)
    var a_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )

    init_vector_launch[DType.uint8](
        a_dev, total_routes * packed_K, init_type, ctx
    )
    init_vector_launch[DType.uint8](
        b_dev, num_experts * N * packed_K, init_type, ctx
    )

    # float8_e8m0fnu cannot represent zero, so init_vector_launch crashes
    # the compiler with "format does not support Zero". memset preserves the
    # neutral "1.0" scale (0x7F) until we wire a proper random initializer.
    ctx.enqueue_memset(a_scales_dev, bitcast[DType.float8_e8m0fnu](UInt8(127)))
    ctx.enqueue_memset(b_scales_dev, bitcast[DType.float8_e8m0fnu](UInt8(127)))

    var a_off_h = ctx.enqueue_create_host_buffer[DType.uint32](
        num_active_experts + 1
    )
    var ei_h = ctx.enqueue_create_host_buffer[DType.int32](num_active_experts)
    a_off_h[0] = 0
    var max_count = 0
    for e in range(num_active_experts):
        var c = target_counts[e]
        a_off_h[e + 1] = a_off_h[e] + UInt32(c)
        ei_h[e] = Int32(expert_id_pool[e])
        if c > max_count:
            max_count = c
    # Production carries a capacity-bound `max_num_tokens_per_expert`
    # (default 8192 = `max_batch_input_tokens`), not the actual per-call
    # max. When `max_tokens_capacity > 0` simulate that.
    var max_tokens_for_kernel = (
        max_tokens_capacity if max_tokens_capacity > 0 else max_count
    )
    ctx.enqueue_copy(a_offsets_dev, a_off_h)
    ctx.enqueue_copy(expert_ids_dev, ei_h)
    ctx.synchronize()

    var a_tt = TileTensor[mut=False](
        a_dev, row_major(Coord(total_routes, Idx[packed_K]))
    )
    var b_tt = TileTensor[mut=False](
        b_dev, row_major[num_experts, N, packed_K]()
    )
    var sfa_tt = TileTensor[mut=False](
        a_scales_dev, row_major(Coord(total_routes, Idx[scale_K]))
    )
    var sfb_tt = TileTensor[mut=False](
        b_scales_dev, row_major[num_experts, N, scale_K]()
    )
    var aoff_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active_experts + 1))
    )
    var ei_tt = TileTensor(expert_ids_dev, row_major(Coord(num_active_experts)))
    var c_tt = TileTensor[mut=True](
        c_dev, row_major(Coord(total_routes, Idx[N]))
    )

    @parameter
    @__copy_capture(c_tt, a_tt, b_tt, sfa_tt, sfb_tt, aoff_tt, ei_tt)
    @always_inline
    def bench_func(mut bencher: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            mxfp4_grouped_matmul_amd(
                c_tt,
                a_tt,
                b_tt,
                sfa_tt,
                sfb_tt,
                aoff_tt,
                ei_tt,
                max_tokens_for_kernel,
                num_active_experts,
                ctx,
            )

        bencher.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId(
            _run_name(
                "gmm_amd (uint8 -> float32)",
                num_experts,
                num_active_experts,
                M,
                topk,
                N,
                K,
                skew_spec,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    _ = a_dev^
    _ = b_dev^
    _ = a_scales_dev^
    _ = b_scales_dev^
    _ = c_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^


# ===----------------------------------------------------------------------=== #
# Dense + preshuffled-B path (mxfp4_grouped_matmul_amd_preb)
# ===----------------------------------------------------------------------=== #


def bench_dense_preb[
    num_experts: Int,
    N: Int,
    K: Int,
    num_shared_experts: Int,
    topk: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    M: Int,
    num_active_experts: Int,
    target_counts: List[Int],
    expert_id_pool: List[Int],
    skew_spec: String,
    init_type: InitializationType,
    max_tokens_capacity: Int = 0,
    estimated_total_m: Int = 0,
) raises:
    comptime packed_K = K // 2
    comptime scale_K = K // 32

    var total_routes = M * topk
    var total_flops = 2 * total_routes * N * K

    var a_dev = ctx.enqueue_create_buffer[DType.uint8](total_routes * packed_K)
    # B buffer is the preshuffled layout, same total bytes as raw B. We don't
    # actually preshuffle here (no correctness check in the bench); the bytes
    # are random and the kernel times the same regardless of content.
    var b_pre_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var c_dev = ctx.enqueue_create_buffer[DType.float32](total_routes * N)
    var a_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )

    init_vector_launch[DType.uint8](
        a_dev, total_routes * packed_K, init_type, ctx
    )
    init_vector_launch[DType.uint8](
        b_pre_dev, num_experts * N * packed_K, init_type, ctx
    )

    var a_off_h = ctx.enqueue_create_host_buffer[DType.uint32](
        num_active_experts + 1
    )
    var ei_h = ctx.enqueue_create_host_buffer[DType.int32](num_active_experts)
    a_off_h[0] = 0
    var max_count = 0
    for e in range(num_active_experts):
        var c = target_counts[e]
        a_off_h[e + 1] = a_off_h[e] + UInt32(c)
        ei_h[e] = Int32(expert_id_pool[e])
        if c > max_count:
            max_count = c
    var max_tokens_for_kernel = (
        max_tokens_capacity if max_tokens_capacity > 0 else max_count
    )
    var max_padded_M = align_up(max_tokens_for_kernel, 32)
    ctx.enqueue_copy(a_offsets_dev, a_off_h)
    ctx.enqueue_copy(expert_ids_dev, ei_h)
    ctx.synchronize()

    # Preshuffled scale buffers (uint8 — the dispatcher reads these in
    # scale-4d byte order via PreshuffledScaleLoader). The bench skips
    # the actual preshuffle and just fills both with a valid E8M0 byte
    # (0x7F = magnitude 1) — kernel timing is content-independent, same
    # as the b_pre random-bytes approach above.
    var a_sc_pre_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * max_padded_M * scale_K
    )
    var b_sc_pre_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * scale_K
    )
    ctx.enqueue_memset(a_sc_pre_dev, UInt8(127))
    ctx.enqueue_memset(b_sc_pre_dev, UInt8(127))

    var a_tt = TileTensor[mut=False](
        a_dev, row_major(Coord(total_routes, Idx[packed_K]))
    )
    var b_pre_tt = TileTensor[mut=False](
        b_pre_dev, row_major[num_experts, N * packed_K]()
    )
    # Bitcast uint8 → float8_e8m0fnu at the TileTensor wrap to match the
    # dispatcher signature. The kernel internally re-bitcasts to uint8
    # for V# construction; the dtype is a wrapping convention.
    var sfa_tt = TileTensor[mut=False](
        a_sc_pre_dev.unsafe_ptr().bitcast[Scalar[DType.float8_e8m0fnu]](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    )
    var sfb_tt = TileTensor[mut=False](
        b_sc_pre_dev.unsafe_ptr().bitcast[Scalar[DType.float8_e8m0fnu]](),
        row_major[num_experts, N, scale_K](),
    )
    var aoff_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active_experts + 1))
    )
    var ei_tt = TileTensor(expert_ids_dev, row_major(Coord(num_active_experts)))
    var c_tt = TileTensor[mut=True](
        c_dev, row_major(Coord(total_routes, Idx[N]))
    )

    @parameter
    @__copy_capture(c_tt, a_tt, b_pre_tt, sfa_tt, sfb_tt, aoff_tt, ei_tt)
    @always_inline
    def bench_func(mut bencher: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            mxfp4_grouped_matmul_amd_preb(
                c_tt,
                a_tt,
                b_pre_tt,
                sfa_tt,
                sfb_tt,
                aoff_tt,
                ei_tt,
                max_tokens_for_kernel,
                num_active_experts,
                ctx,
                estimated_total_m,
            )

        bencher.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId(
            _run_name(
                "gmm_amd_preb (uint8 -> float32)",
                num_experts,
                num_active_experts,
                M,
                topk,
                N,
                K,
                skew_spec,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    _ = a_dev^
    _ = b_pre_dev^
    _ = a_sc_pre_dev^
    _ = b_sc_pre_dev^
    _ = c_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^


# ===----------------------------------------------------------------------=== #
# Routed path (mxfp4_moe_matmul_amd_routed)
# ===----------------------------------------------------------------------=== #


def bench_routed[
    num_experts: Int,
    N: Int,
    K: Int,
    num_shared_experts: Int,
    topk: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    M: Int,
    num_active_experts: Int,
    target_counts: List[Int],
    expert_id_pool: List[Int],
    per_expert_routes: List[List[Tuple[Int, Int]]],
    skew_spec: String,
    init_type: InitializationType,
) raises:
    comptime packed_K = K // 2
    comptime scale_K = K // 32
    comptime sort_block_m = 64
    comptime N_padded_scale = Shuffler[1].scale_padded_mn(N)
    comptime sfa_per_block_bytes = sort_block_m * scale_K
    comptime sfb_per_expert_bytes = N_padded_scale * scale_K

    var total_routes = M * topk
    var total_flops = 2 * total_routes * N * K

    var num_blocks = 0
    for e in range(num_active_experts):
        num_blocks += ceildiv(target_counts[e], sort_block_m)
    var size_expert_ids = num_blocks

    var a_dev = ctx.enqueue_create_buffer[DType.uint8](M * packed_K)
    var b_pre_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var sfa_pre_dev = ctx.enqueue_create_buffer[DType.uint8](
        size_expert_ids * sfa_per_block_bytes
    )
    var sfb_pre_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * sfb_per_expert_bytes
    )
    var sti_dev = ctx.enqueue_create_buffer[DType.uint32](
        size_expert_ids * sort_block_m
    )
    var ei_dev = ctx.enqueue_create_buffer[DType.int32](size_expert_ids)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](total_routes * N)

    init_vector_launch[DType.uint8](a_dev, M * packed_K, init_type, ctx)
    init_vector_launch[DType.uint8](
        b_pre_dev, num_experts * N * packed_K, init_type, ctx
    )
    init_vector_launch[DType.uint8](
        sfa_pre_dev, size_expert_ids * sfa_per_block_bytes, init_type, ctx
    )
    init_vector_launch[DType.uint8](
        sfb_pre_dev, num_experts * sfb_per_expert_bytes, init_type, ctx
    )

    var sti_h = ctx.enqueue_create_host_buffer[DType.uint32](
        size_expert_ids * sort_block_m
    )
    var ei_h = ctx.enqueue_create_host_buffer[DType.int32](size_expert_ids)
    var pad = UInt32(0xFFFFFF)

    var blk_idx = 0
    for e in range(num_active_experts):
        ref routes = per_expert_routes[e]
        var n = len(routes)
        var blocks_for_expert = ceildiv(n, sort_block_m)
        for blk in range(blocks_for_expert):
            ei_h[blk_idx + blk] = Int32(expert_id_pool[e])
            for r in range(sort_block_m):
                var route_idx = blk * sort_block_m + r
                var sti_pos = (blk_idx + blk) * sort_block_m + r
                if route_idx < n:
                    var pair = routes[route_idx]
                    var t = UInt32(pair[0])
                    var s = UInt32(pair[1])
                    sti_h[sti_pos] = t | (s << UInt32(24))
                else:
                    sti_h[sti_pos] = pad
        blk_idx += blocks_for_expert
    ctx.enqueue_copy(sti_dev, sti_h)
    ctx.enqueue_copy(ei_dev, ei_h)
    ctx.synchronize()

    var a_tt = TileTensor[mut=False](
        a_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(M, Idx[packed_K])),
    )
    var b_pre_tt = TileTensor[mut=False](
        b_pre_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx[1], Idx[num_experts * N * packed_K])),
    )
    var sfa_pre_tt = TileTensor[mut=False](
        sfa_pre_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx[1], size_expert_ids * sfa_per_block_bytes)),
    )
    var sfb_pre_tt = TileTensor[mut=False](
        sfb_pre_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx[1], Idx[num_experts * sfb_per_expert_bytes])),
    )
    var sti_tt = TileTensor[mut=False](
        sti_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(size_expert_ids * sort_block_m)),
    )
    var ei_tt = TileTensor[mut=False](
        ei_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(size_expert_ids)),
    )
    var c_tt = TileTensor[mut=True](
        c_dev, row_major(Coord(total_routes, Idx[N]))
    )

    @parameter
    @__copy_capture(c_tt, a_tt, b_pre_tt, sfa_pre_tt, sfb_pre_tt, sti_tt, ei_tt)
    @always_inline
    def bench_func(mut bencher: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            mxfp4_moe_matmul_amd_routed[topk=topk](
                c_tt,
                a_tt,
                b_pre_tt,
                sfa_pre_tt,
                sfb_pre_tt,
                sti_tt,
                ei_tt,
                M,
                size_expert_ids,
                ctx,
            )

        bencher.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId(
            _run_name(
                "gmm_amd_routed (uint8 -> float32)",
                num_experts,
                num_active_experts,
                M,
                topk,
                N,
                K,
                skew_spec,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    _ = a_dev^
    _ = b_pre_dev^
    _ = sfa_pre_dev^
    _ = sfb_pre_dev^
    _ = sti_dev^
    _ = ei_dev^
    _ = c_dev^


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def main() raises:
    comptime num_experts = get_defined_int["num_experts", 8]()
    comptime N = get_defined_int["N", 4096]()
    comptime K = get_defined_int["K", 7168]()
    comptime topk = get_defined_int["topk", 1]()
    comptime num_shared_experts = get_defined_int["num_shared_experts", 0]()
    comptime algo = get_defined_string["algo", "dense"]()

    comptime assert K % 128 == 0, "K must be a multiple of 128 (MFMA K dim)"
    comptime assert (
        num_shared_experts <= topk
    ), "num_shared_experts cannot exceed topk"
    comptime assert (
        algo == "dense" or algo == "dense_preb" or algo == "routed"
    ), "algo must be dense, dense_preb, or routed"

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var M = Int(arg_parse("M", 256))
    var seed_val = UInt64(Int(arg_parse("seed", 0)))
    var skew_spec = String(arg_parse("expert_skew", "uniform"))
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    # Production simulates a capacity-bound `max_num_tokens_per_expert`
    # (e.g. 8192 = `max_batch_input_tokens` default) regardless of actual
    # routing. When >0 this override is forwarded to the kernel instead of
    # the routing-derived max. Default 0 = use actual.
    var max_tokens_capacity = Int(arg_parse("max_tokens_capacity", 0))
    # Replay a literal serve routing: pass comma-separated per-slot token
    # counts and expert IDs. Both must be set together and both must have
    # length == num_active_experts. Bypasses skew synthesis.
    var target_counts_csv = String(arg_parse("target_counts", ""))
    var expert_ids_csv = String(arg_parse("expert_ids", ""))
    # EP topology: total GPUs the inputs are split across (DP). Used to
    # compute `estimated_total_m = M * topk // n_gpus_per_node` for the
    # preb dispatcher's persistent-vs-direct path switch.
    var n_gpus_per_node = Int(arg_parse("n_gpus_per_node", 8))

    if num_active_experts < num_shared_experts:
        abort(
            "num_active_experts="
            + String(num_active_experts)
            + " < num_shared_experts="
            + String(num_shared_experts)
        )
    if num_active_experts > num_experts:
        abort(
            "num_active_experts="
            + String(num_active_experts)
            + " > num_experts="
            + String(num_experts)
        )
    if topk > num_active_experts:
        abort(
            "topk="
            + String(topk)
            + " > num_active_experts="
            + String(num_active_experts)
        )

    if (
        target_counts_csv.byte_length() == 0
        or expert_ids_csv.byte_length() == 0
    ):
        abort(
            "target_counts and expert_ids are required (pass both as"
            ' ";"-separated lists of length num_active_experts)'
        )
    var target_counts = _parse_csv_ints(target_counts_csv)
    var expert_id_pool = _parse_csv_ints(expert_ids_csv)
    if len(target_counts) != num_active_experts:
        abort(
            "target_counts length="
            + String(len(target_counts))
            + " must equal num_active_experts="
            + String(num_active_experts)
        )
    if len(expert_id_pool) != num_active_experts:
        abort(
            "expert_ids length="
            + String(len(expert_id_pool))
            + " must equal num_active_experts="
            + String(num_active_experts)
        )
    for e in expert_id_pool:
        if e < 0 or e >= num_experts:
            abort(
                "expert_ids contains out-of-range value="
                + String(e)
                + " (num_experts="
                + String(num_experts)
                + ")"
            )

    print(
        "Config: algo=",
        algo,
        " num_experts=",
        num_experts,
        " num_active_experts=",
        num_active_experts,
        " num_shared=",
        num_shared_experts,
        " M=",
        M,
        " topk=",
        topk,
        " N=",
        N,
        " K=",
        K,
        " skew=",
        skew_spec,
    )
    print("  target_counts(len=", len(target_counts), "):", end=" ")
    for c in target_counts:
        print(c, end=" ")
    print()
    print("  expert_id_pool(len=", len(expert_id_pool), "):", end=" ")
    for e in expert_id_pool:
        print(e, end=" ")
    print()

    with DeviceContext() as ctx:
        var bench = Bench()

        comptime if algo == "dense":
            bench_dense[
                num_experts=num_experts,
                N=N,
                K=K,
                num_shared_experts=num_shared_experts,
                topk=topk,
            ](
                ctx,
                bench,
                M,
                num_active_experts,
                target_counts,
                expert_id_pool,
                skew_spec,
                init_type,
                max_tokens_capacity,
            )
        elif algo == "dense_preb":
            # Mirror production formula in expert_parallel.py:
            #   estimated_total_m = total_tokens * topk // n_gpus_per_node
            # In the bench `total_tokens` is the per-GPU input batch (=M),
            # so this collapses to `M * topk // n_gpus_per_node`. For Kimi
            # K2.5 (topk=8, EP=8) the multiplier is 1 → estimated == M.
            var estimated_total_m = M * topk // n_gpus_per_node
            bench_dense_preb[
                num_experts=num_experts,
                N=N,
                K=K,
                num_shared_experts=num_shared_experts,
                topk=topk,
            ](
                ctx,
                bench,
                M,
                num_active_experts,
                target_counts,
                expert_id_pool,
                skew_spec,
                init_type,
                max_tokens_capacity,
                estimated_total_m,
            )
        else:
            var per_expert_routes = _build_per_expert_routes(
                M,
                topk,
                num_active_experts,
                num_shared_experts,
                target_counts,
                seed_val,
            )
            bench_routed[
                num_experts=num_experts,
                N=N,
                K=K,
                num_shared_experts=num_shared_experts,
                topk=topk,
            ](
                ctx,
                bench,
                M,
                num_active_experts,
                target_counts,
                expert_id_pool,
                per_expert_routes,
                skew_spec,
                init_type,
            )

        bench.dump_report()
