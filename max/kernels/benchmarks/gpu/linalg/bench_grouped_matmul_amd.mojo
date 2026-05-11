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

algo selects the kernel (both are MXFP4):
  - "dense"  -> mxfp4_grouped_matmul_amd (per-expert offsets)
  - "routed" -> mxfp4_moe_matmul_amd_routed (scattered sort blocks)
"""

from std.math import ceildiv
from std.memory import bitcast
from std.os import abort
from std.random import Random, seed, shuffle
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
    mxfp4_moe_matmul_amd_routed,
)


# ===----------------------------------------------------------------------=== #
# Skew parsing
# ===----------------------------------------------------------------------=== #


def _parse_csv_floats(s: String) raises -> List[Float64]:
    var stripped = s.strip("[]\"' ")
    var out = List[Float64]()
    if stripped.byte_length() == 0:
        return out^
    for tok in stripped.split(","):
        var t = String(tok).strip()
        if t.byte_length() == 0:
            continue
        out.append(Float64(t))
    return out^


def _expand_skew(spec: String, n: Int) raises -> List[Float64]:
    """Returns length-n percentages summing to 100 over the routed set."""
    var s = spec.strip()
    if s.byte_length() == 0 or s == "uniform":
        var w = Float64(100) / Float64(n)
        var out = List[Float64]()
        for _ in range(n):
            out.append(w)
        return out^

    if s.startswith("top_heavy:"):
        var frac = Float64(s.removeprefix("top_heavy:"))
        if frac < 0.0 or frac >= 1.0:
            abort("top_heavy fraction must be in [0, 1)")
        var rest = (1.0 - frac) * 100.0 / Float64(n - 1) if n > 1 else 0.0
        var out = List[Float64]()
        out.append(frac * 100.0)
        for _ in range(n - 1):
            out.append(rest)
        return out^

    var parsed = _parse_csv_floats(String(s))
    if len(parsed) != n:
        abort(
            "expert_skew length="
            + String(len(parsed))
            + " but routed expert count="
            + String(n)
        )
    return parsed^


def _largest_remainder(weights: List[Float64], total: Int) -> List[Int]:
    """Rounds real-valued weights (summing to ~100) to ints summing to total."""
    var n = len(weights)
    var floors = List[Int]()
    var fracs = List[Float64]()
    var assigned = 0
    for i in range(n):
        var raw = Float64(total) * weights[i] / 100.0
        var f = Int(raw)
        floors.append(f)
        fracs.append(raw - Float64(f))
        assigned += f
    var remaining = total - assigned
    # Distribute remaining +1s to entries with the largest fractional part.
    while remaining > 0:
        var best = -1
        var best_frac = -1.0
        for i in range(n):
            if fracs[i] > best_frac:
                best_frac = fracs[i]
                best = i
        floors[best] += 1
        fracs[best] = -1.0
        remaining -= 1
    return floors^


# ===----------------------------------------------------------------------=== #
# Sampling
# ===----------------------------------------------------------------------=== #


def _sample_distinct(
    mut rng: Random, universe: Int, k: Int
) raises -> List[Int]:
    """Sample k distinct values from [0, universe). k must be <= universe."""
    if k > universe:
        abort(
            "sample_distinct: k="
            + String(k)
            + " > universe="
            + String(universe)
        )
    var pool = List[Int]()
    for i in range(universe):
        pool.append(i)
    # Partial Fisher-Yates with our seeded rng.
    for i in range(k):
        var u = rng.step()
        var j = i + Int(UInt32(u[0]) % UInt32(universe - i))
        var tmp = pool[i]
        pool[i] = pool[j]
        pool[j] = tmp
    var out = List[Int]()
    for i in range(k):
        out.append(pool[i])
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


def _build_target_counts(
    M: Int,
    topk: Int,
    num_active_experts: Int,
    num_shared_experts: Int,
    skew_spec: String,
) raises -> List[Int]:
    """Returns length-num_active_experts route counts summing to M*topk.

    Shared experts (first num_shared_experts entries) get exactly M each.
    Routed experts share M*(topk - num_shared_experts) by skew_spec.
    """
    var n_routed = num_active_experts - num_shared_experts
    var routed_total = M * (topk - num_shared_experts)
    var counts = List[Int]()
    for _ in range(num_shared_experts):
        counts.append(M)
    if n_routed == 0:
        return counts^
    var routed_skew = _expand_skew(skew_spec, n_routed)
    var routed_counts = _largest_remainder(routed_skew, routed_total)
    for c in routed_counts:
        counts.append(c)
    return counts^


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
    ctx.enqueue_copy(a_offsets_dev, a_off_h)
    ctx.enqueue_copy(expert_ids_dev, ei_h)
    ctx.synchronize()

    var a_tt = TileTensor[mut=False](
        a_dev, row_major(Coord(Idx(total_routes), Idx[packed_K]()))
    )
    var b_tt = TileTensor[mut=False](
        b_dev, row_major[num_experts, N, packed_K]()
    )
    var sfa_tt = TileTensor[mut=False](
        a_scales_dev, row_major(Coord(Idx(total_routes), Idx[scale_K]()))
    )
    var sfb_tt = TileTensor[mut=False](
        b_scales_dev, row_major[num_experts, N, scale_K]()
    )
    var aoff_tt = TileTensor(
        a_offsets_dev, row_major(Coord(Idx(num_active_experts + 1)))
    )
    var ei_tt = TileTensor(
        expert_ids_dev, row_major(Coord(Idx(num_active_experts)))
    )
    var c_tt = TileTensor[mut=True](
        c_dev, row_major(Coord(Idx(total_routes), Idx[N]()))
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
                max_count,
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
    ctx.enqueue_memset(sfa_pre_dev, UInt8(127))
    ctx.enqueue_memset(sfb_pre_dev, UInt8(127))

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
        row_major(Coord(Idx(M), Idx[packed_K]())),
    )
    var b_pre_tt = TileTensor[mut=False](
        b_pre_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx[1](), Idx[num_experts * N * packed_K]())),
    )
    var sfa_pre_tt = TileTensor[mut=False](
        sfa_pre_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx[1](), Idx(size_expert_ids * sfa_per_block_bytes))),
    )
    var sfb_pre_tt = TileTensor[mut=False](
        sfb_pre_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx[1](), Idx[num_experts * sfb_per_expert_bytes]())),
    )
    var sti_tt = TileTensor[mut=False](
        sti_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx(size_expert_ids * sort_block_m))),
    )
    var ei_tt = TileTensor[mut=False](
        ei_dev.unsafe_ptr().as_immutable(),
        row_major(Coord(Idx(size_expert_ids))),
    )
    var c_tt = TileTensor[mut=True](
        c_dev, row_major(Coord(Idx(total_routes), Idx[N]()))
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
        algo == "dense" or algo == "routed"
    ), "algo must be dense or routed"

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var M = Int(arg_parse("M", 256))
    var seed_val = UInt64(Int(arg_parse("seed", 0)))
    var skew_spec = String(arg_parse("expert_skew", "uniform"))
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )

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

    var target_counts = _build_target_counts(
        M, topk, num_active_experts, num_shared_experts, skew_spec
    )

    var rng = Random(seed=seed_val)
    var expert_id_pool = _sample_distinct(rng, num_experts, num_active_experts)

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
