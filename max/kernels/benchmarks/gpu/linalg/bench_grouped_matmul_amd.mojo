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
expert_ids) so we can sweep one knob at a time.

Runs the MXFP4 preshuffled-B grouped matmul (mxfp4_grouped_matmul_amd_preb:
preshuffled B, direct VGPR loads).
"""

from std.math import align_up, ceildiv
from std.os import abort
from std.random import random_ui64, seed
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
from internal_utils import arg_parse, CacheBustingBuffer, CACHE_BUST_BYTES
from internal_utils._utils import InitializationType
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.amd import mxfp4_grouped_matmul_amd_preb


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
# Preshuffled-B path (mxfp4_grouped_matmul_amd_preb)
# ===----------------------------------------------------------------------=== #


def bench_preb[
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
    cache_bust: Bool = True,
    cache_bust_gb: Float64 = 0.0,
    max_tokens_capacity: Int = 0,
    estimated_total_m: Int = 0,
) raises:
    comptime packed_K = K // 2
    comptime scale_K = K // 32

    # Workload is driven by the explicit per-expert token counts (matches the
    # Blackwell `num_tokens_by_expert` bench), not an `M*topk` fan-out.
    var total_routes = 0
    for i in range(num_active_experts):
        total_routes += target_counts[i]
    var total_flops = 2 * total_routes * N * K

    # Routing tables are tiny metadata read once per launch; keep them as plain
    # single buffers (not cache-busted).
    var a_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
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

    comptime simd_size = 4
    # B is the dominant operand. A single full copy is num_experts*N*packed_K
    # bytes; the kernel only reads the active experts' slices, so at decode
    # (few active experts) that working set stays resident unless we rotate
    # through many full-B copies. CACHE_BUST_BYTES is 2x the GPU cache, so
    # CACHE_BUST_BYTES * num_experts gives enough windows to evict even a
    # single-expert (M=1) read with a 2x margin (= 24 GiB at 48 experts). The
    # max() keeps >=2 windows if one full copy is itself larger than that.
    # Override with cache_bust_gb to set an explicit footprint.
    var b_full_bytes = num_experts * N * packed_K
    var b_budget = Int(
        cache_bust_gb * (1024.0 * 1024.0 * 1024.0)
    ) if cache_bust_gb > 0.0 else max(
        CACHE_BUST_BYTES * num_experts, 2 * b_full_bytes
    )
    var b_windows = ceildiv(b_budget, align_up(b_full_bytes, simd_size))
    print(
        "  cache_bust=",
        cache_bust,
        " B full=",
        b_full_bytes // (1024 * 1024),
        "MiB budget=",
        b_budget // (1024 * 1024),
        "MiB windows=",
        b_windows,
    )
    var cb_a = CacheBustingBuffer[DType.uint8](
        total_routes * packed_K, simd_size, ctx, cache_bust
    )
    var cb_b = CacheBustingBuffer[DType.uint8](
        num_experts * N * packed_K,
        simd_size,
        ctx,
        cache_bust,
        budget_bytes=b_budget,
    )
    var cb_c = CacheBustingBuffer[DType.float32](
        total_routes * N, simd_size, ctx, cache_bust
    )
    # Scale buffers (uint8 — the dispatcher reads these in scale-4d byte order
    # via PreshuffledScaleLoader). Fill the whole buffer with a valid E8M0 byte
    # (0x7F = magnitude 1).
    var cb_a_sc = CacheBustingBuffer[DType.uint8](
        num_experts * max_padded_M * scale_K, simd_size, ctx, cache_bust
    )
    var cb_b_sc = CacheBustingBuffer[DType.uint8](
        num_experts * N * scale_K, simd_size, ctx, cache_bust
    )

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    ctx.enqueue_memset(cb_a_sc.device_buffer(), UInt8(127))
    ctx.enqueue_memset(cb_b_sc.device_buffer(), UInt8(127))

    var aoff_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active_experts + 1))
    )
    var ei_tt = TileTensor(expert_ids_dev, row_major(Coord(num_active_experts)))

    @parameter
    @always_inline
    def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
        var a_tt = TileTensor[mut=False](
            cb_a.offset_ptr(iteration),
            row_major(Coord(total_routes, Idx[packed_K])),
        )
        var b_pre_tt = TileTensor[mut=False](
            cb_b.offset_ptr(iteration), row_major[num_experts, N * packed_K]()
        )
        # Bitcast uint8 → float8_e8m0fnu at the TileTensor wrap to match the
        # dispatcher signature. The kernel internally re-bitcasts to uint8 for
        # V# construction; the dtype is a wrapping convention.
        var sfa_tt = TileTensor[mut=False](
            cb_a_sc.offset_ptr(iteration).bitcast[
                Scalar[DType.float8_e8m0fnu]
            ](),
            row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
        )
        var sfb_tt = TileTensor[mut=False](
            cb_b_sc.offset_ptr(iteration).bitcast[
                Scalar[DType.float8_e8m0fnu]
            ](),
            row_major[num_experts, N, scale_K](),
        )
        var c_tt = TileTensor[mut=True](
            cb_c.offset_ptr(iteration), row_major(Coord(total_routes, Idx[N]))
        )
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

    @parameter
    @always_inline
    def bench_func(mut bencher: Bencher):
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

    _ = cb_a^
    _ = cb_b^
    _ = cb_a_sc^
    _ = cb_b_sc^
    _ = cb_c^
    _ = a_offsets_dev^
    _ = expert_ids_dev^


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def main() raises:
    comptime num_experts = get_defined_int["num_experts", 8]()
    comptime N = get_defined_int["N", 4096]()
    comptime K = get_defined_int["K", 7168]()
    comptime topk = get_defined_int["topk", 1]()
    comptime num_shared_experts = get_defined_int["num_shared_experts", 0]()
    comptime algo = get_defined_string["algo", "preb"]()

    comptime assert K % 128 == 0, "K must be a multiple of 128 (MFMA K dim)"
    comptime assert (
        num_shared_experts <= topk
    ), "num_shared_experts cannot exceed topk"
    comptime assert (
        algo == "preb"
    ), "algo must be 'preb' (the 'dense' and 'routed' paths were removed)"

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var M = Int(arg_parse("M", 256))
    # Expert-parallel degree: experts are sharded across this many GPUs, so a
    # token's `topk` global picks land on this rank's local shard only ~1/N of
    # the time. Per-rank routed-M = M * topk // n_gpus_per_node, matching
    # `nn/moe/expert_parallel.py` (total_tokens * topk // n_gpus_per_node).
    var n_gpus_per_node = Int(arg_parse("n_gpus_per_node", 4))
    var skew_spec = String(arg_parse("expert_skew", "uniform"))
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    # Production simulates a capacity-bound `max_num_tokens_per_expert`
    # (e.g. 8192 = `max_batch_input_tokens` default) regardless of actual
    # routing. When >0 this override is forwarded to the kernel instead of
    # the routing-derived max. Default 0 = use actual.
    var max_tokens_capacity = Int(arg_parse("max_tokens_capacity", 0))
    var cache_bust = Bool(arg_parse("cache_bust", True))
    # Buffer footprint (GiB) for the B-weights cache-busting buffer. 0 = auto:
    # 512 MiB * num_experts (enough windows to evict even a single-expert M=1
    # read; = 24 GiB at 48 experts). Set explicitly to cap or raise the budget.
    var cache_bust_gb = Float64(arg_parse("cache_bust_gb", 0.0))
    # Replay a literal serve routing: pass comma-separated per-slot token
    # counts and expert IDs. Both must be set together and both must have
    # length == num_active_experts. Bypasses skew synthesis.
    var target_counts_csv = String(arg_parse("target_counts", ""))
    var expert_ids_csv = String(arg_parse("expert_ids", ""))

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
    var target_counts = List[Int]()
    var expert_id_pool = List[Int]()
    var have_counts = target_counts_csv.byte_length() != 0
    var have_ids = expert_ids_csv.byte_length() != 0
    if not have_counts and not have_ids:
        # Synthesize realistic EP-sharded MoE routing. Each of `M` tokens routes
        # to `topk` distinct experts out of the GLOBAL pool
        # (num_experts * n_gpus_per_node); only picks landing on THIS rank's
        # local shard ([0, num_experts)) are processed here. So per-rank routed
        # rows ~= M * topk // n_gpus_per_node, matching
        # `nn/moe/expert_parallel.py` (total_tokens * topk // n_gpus_per_node),
        # and active experts follow coupon-collector over the local shard.
        # n_gpus_per_node=1 reduces to a single-GPU run (all picks local).
        # `seed(0)` keeps the routing reproducible across runs.
        seed(0)
        var num_tokens = M if M > 0 else 1
        var gpus = max(n_gpus_per_node, 1)
        var global_experts = num_experts * gpus
        var k = min(topk, global_experts)
        var counts = List[Int]()
        for _ in range(num_experts):
            counts.append(0)
        for _ in range(num_tokens):
            var picks = List[Int]()
            while len(picks) < k:
                var e = Int(random_ui64(0, UInt64(global_experts - 1)))
                var dup = False
                for i in range(len(picks)):
                    if picks[i] == e:
                        dup = True
                        break
                if not dup:
                    picks.append(e)
            # Count only picks that land on this rank's local shard.
            for i in range(len(picks)):
                if picks[i] < num_experts:
                    counts[picks[i]] += 1
        for e in range(num_experts):
            if counts[e] > 0:
                target_counts.append(counts[e])
                expert_id_pool.append(e)
        num_active_experts = len(target_counts)
    elif not have_counts or not have_ids:
        abort(
            "target_counts and expert_ids must be set together, or both"
            " omitted to synthesize uniform routing from M"
        )
    else:
        target_counts = _parse_csv_ints(target_counts_csv)
        expert_id_pool = _parse_csv_ints(expert_ids_csv)
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
        " n_gpus_per_node=",
        n_gpus_per_node,
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

        # `estimated_total_m` drives the preb dispatcher's band +
        # persistent-vs-direct switch. Production computes it from the routing
        # FORMULA (an estimate available at dispatch time), not the exact
        # per-expert counts — see nn/moe/expert_parallel.py:
        #   estimated_total_m = total_tokens * topk // n_gpus_per_node
        var estimated_total_m = M * topk // max(n_gpus_per_node, 1)
        bench_preb[
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
            cache_bust,
            cache_bust_gb,
            max_tokens_capacity,
            estimated_total_m,
        )

        bench.dump_report()
