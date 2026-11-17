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
"""Dense SM100 bf16 GEMM with an explicit `MatmulConfig`, on DiT shapes.

`bench_matmul` goes through `_matmul_gpu`'s dispatch, so it only ever measures
whatever config the heuristic or the tuning table picks. This benchmark builds
the `MatmulConfig` from its own defines and launches
`blackwell_matmul_tma_umma_warp_specialized` directly - the same kernel
`select_and_launch_sm100_config` launches for a tuning-table row - so a number
here is the number that row would get.

The three `shape` values are the MiniMax-H3 DiT linears, measured as
`x[M, K] @ W[N, K]^T` with `M` from the host plan's packed row count:

  0  qkv   N=21504  K=5376
  1  out   N=5376   K=7168
  2  down  N=5376   K=14336

Each arm runs once against `vendor_blas` at VERIFY_M rows (same comptime config,
small footprint) and is then timed at the requested `M`. Data is cache-cold by
construction at these sizes - A is 1.2 GB and C is 4.7 GB at M=109632 against a
126 MB L2 - so no cache-busting windows are needed.

Usage (kbench expands the define grid; see the sibling .yaml):

    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
      $KERNEL_BENCHMARKS_ROOT/gpu/linalg/bench_matmul_sm100_bf16_config.yaml
"""

from std.sys import get_defined_bool, get_defined_int

import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
)
from internal_utils._utils import InitializationType
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.tile_scheduler import RasterOrder

comptime dtype = DType.bfloat16
comptime MMA_K = 16
comptime VERIFY_M = 512
# Relative-difference bound `bench_matmul`'s verifier uses for bf16.
comptime VERIFY_REL_DIFF = 0.001

# --- shape under test -------------------------------------------------------
comptime SHAPE = get_defined_int["shape", 0]()
comptime N = 21504 if SHAPE == 0 else (28672 if SHAPE == 3 else 5376)
comptime K = (
    5376 if SHAPE
    == 0 else (7168 if SHAPE == 1 else (5376 if SHAPE == 3 else 14336))
)
comptime SHAPE_NAME = (
    "qkv" if SHAPE
    == 0 else ("out" if SHAPE == 1 else ("gate_up" if SHAPE == 3 else "down"))
)

# --- config under test ------------------------------------------------------
comptime MMA_M = get_defined_int["mma_m", 256]()
comptime MMA_N = get_defined_int["mma_n", 256]()
comptime CTA_GROUP = get_defined_int["cta_group", 2]()
comptime CLUSTER_X = get_defined_int["cluster_x", 2]()
comptime CLUSTER_Y = get_defined_int["cluster_y", 1]()
comptime BLOCK_SWIZZLE_SIZE = get_defined_int["block_swizzle_size", 8]()
comptime RASTERIZE_ORDER = get_defined_int["rasterize_order", 1]()
comptime K_GROUP_SIZE = get_defined_int["k_group_size", 1]()
comptime NUM_CLC_PIPELINE_STAGES = get_defined_int[
    "num_clc_pipeline_stages", 2
]()
comptime NUM_ACCUM_PIPELINE_STAGES = get_defined_int[
    "num_accum_pipeline_stages", 2
]()
# 0 keeps `MatmulConfig`'s own `_maximize_pipeline_stages`, as a 0 in a
# tuning-table row does.
comptime NUM_PIPELINE_STAGES = get_defined_int["num_pipeline_stages", 0]()
comptime SWAP_AB = get_defined_bool["swap_ab", False]()

comptime config = MatmulConfig[dtype, dtype, dtype, True](
    mma_shape=(MMA_M, MMA_N, MMA_K),
    cta_group=CTA_GROUP,
    cluster_shape=(CLUSTER_X, CLUSTER_Y, 1),
    block_swizzle_size=BLOCK_SWIZZLE_SIZE,
    raster_order=RasterOrder(Int32(RASTERIZE_ORDER)),
    AB_swapped=SWAP_AB,
    k_group_size=K_GROUP_SIZE,
    num_clc_pipeline_stages=NUM_CLC_PIPELINE_STAGES,
    num_accum_pipeline_stages=NUM_ACCUM_PIPELINE_STAGES,
    num_pipeline_stages=Optional[Int](
        NUM_PIPELINE_STAGES
    ) if NUM_PIPELINE_STAGES
    > 0 else None,
)


def arm_label() -> String:
    """Bench-id fragment naming every knob that shaped this binary."""
    return (
        "mma"
        + String(MMA_M)
        + "x"
        + String(MMA_N)
        + "cta"
        + String(CTA_GROUP)
        + "cl"
        + String(CLUSTER_X)
        + "x"
        + String(CLUSTER_Y)
        + "bz"
        + String(BLOCK_SWIZZLE_SIZE)
        + "rst"
        + String(RASTERIZE_ORDER)
        + "kg"
        + String(K_GROUP_SIZE)
        + "clc"
        + String(NUM_CLC_PIPELINE_STAGES)
        + "acc"
        + String(NUM_ACCUM_PIPELINE_STAGES)
        + "st"
        + String(NUM_PIPELINE_STAGES)
        + ("swap" if SWAP_AB else "noswap")
    )


def check_against_vendor(ctx: DeviceContext) raises:
    """Runs this arm and vendor BLAS over the same operands at VERIFY_M rows."""
    var m = VERIFY_M
    var cb_a = CacheBustingBuffer[dtype](m * K, ctx, False)
    var cb_b = CacheBustingBuffer[dtype](N * K, ctx, False)
    var cb_c = CacheBustingBuffer[dtype](m * N, ctx, False)
    var cb_ref = CacheBustingBuffer[dtype](m * N, ctx, False)
    cb_a.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_b.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_c.init_on_device(InitializationType.zero, ctx)
    cb_ref.init_on_device(InitializationType.zero, ctx)

    var a_tt = TileTensor(cb_a.offset_ptr(0), row_major(m, Idx[K])).as_imm()
    var b_tt = TileTensor(
        cb_b.offset_ptr(0), row_major(Idx[N], Idx[K])
    ).as_imm()
    blackwell_matmul_tma_umma_warp_specialized[transpose_b=True, config=config](
        TileTensor(cb_c.offset_ptr(0), row_major(m, Idx[N])),
        a_tt,
        b_tt,
        ctx,
    )
    vendor_blas.matmul[use_tf32=True](
        ctx,
        TileTensor(cb_ref.offset_ptr(0), row_major(m, Idx[N])),
        a_tt,
        b_tt,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    var got_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    var ref_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    ctx.enqueue_copy(got_host, cb_c.device_buffer())
    ctx.enqueue_copy(ref_host, cb_ref.device_buffer())
    ctx.synchronize()

    var bad = 0
    var max_err: Float32 = 0
    var total_abs_diff: Float32 = 0
    var total_abs_ref: Float32 = 0
    var rtol64: Float64
    var atol64: Float64
    rtol64, atol64 = pytorch_like_tolerances_for[.bfloat16]()
    var rtol = Float32(rtol64)
    var atol = Float32(atol64)
    for i in range(m * N):
        var got = got_host[i].cast[DType.float32]()
        var want = ref_host[i].cast[DType.float32]()
        var err = abs(got - want)
        total_abs_diff += err
        total_abs_ref += abs(want)
        max_err = max(max_err, err)
        if err > atol + rtol * abs(want):
            bad += 1
    var rel_diff = total_abs_diff / total_abs_ref if total_abs_ref > 0 else 0
    var verdict = "PASS" if bad == 0 and rel_diff <= VERIFY_REL_DIFF else "FAIL"
    print(
        t"CHECK {SHAPE_NAME} N={N} K={K} m={m} {arm_label()}"
        t" {verdict} bad={bad} max_abs_err={max_err}"
        t" rel_diff={rel_diff} (bound {VERIFY_REL_DIFF})"
    )

    _ = cb_a^
    _ = cb_b^
    _ = cb_c^
    _ = cb_ref^
    _ = got_host^
    _ = ref_host^


def main() raises:
    var m = Int(arg_parse("M", 109632))
    var run_benchmark = arg_parse("run_benchmark", True)
    var verify = arg_parse("verify", True)

    var bench = Bench()
    with DeviceContext() as ctx:
        if verify:
            check_against_vendor(ctx)

        var cb_a = CacheBustingBuffer[dtype](m * K, ctx, False)
        var cb_b = CacheBustingBuffer[dtype](N * K, ctx, False)
        var cb_c = CacheBustingBuffer[dtype](m * N, ctx, False)
        cb_a.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_b.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_c.init_on_device(InitializationType.zero, ctx)
        ctx.synchronize()

        @inline(.always)
        def kernel_launch(
            cx: DeviceContext[], iteration: Int
        ) raises {var cb_a, var cb_b, var cb_c, imm}:
            # `offset` is always 0 for a single-window buffer; the offset form
            # is the one `bench_matmul` uses, so enabling cache busting later is
            # a one-line change if these shapes ever shrink.
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=True, config=config
            ](
                TileTensor(cb_c.offset_ptr(iteration), row_major(m, Idx[N])),
                TileTensor(
                    cb_a.offset_ptr(iteration), row_major(m, Idx[K])
                ).as_imm(),
                TileTensor(
                    cb_b.offset_ptr(iteration), row_major(Idx[N], Idx[K])
                ).as_imm(),
                cx,
            )

        @inline(.always)
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, kernel_launch, ctx)

        var flops = ThroughputMeasure(BenchMetric.flops, 2 * m * N * K)
        if run_benchmark:
            bench.bench_function(
                bench_func,
                BenchId(
                    String("sm100_bf16_dit/")
                    + SHAPE_NAME
                    + "/N"
                    + String(N)
                    + "K"
                    + String(K)
                    + "/M"
                    + String(m)
                    + "/"
                    + arm_label()
                ),
                [flops],
            )
        else:
            kernel_launch(ctx, 0)

        _ = cb_a^
        _ = cb_b^
        _ = cb_c^

    if run_benchmark:
        bench.dump_report()
