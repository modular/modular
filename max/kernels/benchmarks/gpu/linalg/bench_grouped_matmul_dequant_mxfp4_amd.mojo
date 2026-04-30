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
"""Benchmark for MXFP4 dequant-then-FP8 grouped matmul on AMD CDNA GPUs.

Benchmarks the full mxfp4_dequant_grouped_matmul_amd pipeline:
  1. Dequant MXFP4 packed uint8 expert weights + E8M0 scales to FP8
  2. Cast BF16 activations to FP8
  3. FP8 grouped GEMM via grouped_matmul

Usage:
  br //max/kernels/benchmarks:gpu/linalg/bench_grouped_matmul_dequant_mxfp4_amd.mojo.test

Compile-time parameters (set via --define):
  N: Expert weight rows / output columns (default: 2048)
  K: Inner dimension, unpacked (default: 2048)
  num_experts: Total number of experts (default: 4)

Runtime arguments:
  --num_active_experts: Number of active experts (default: 2)
  --num_tokens_by_expert: Comma-separated token counts (default: "64,128")
  --expert_ids: Comma-separated expert indices (default: "0,2")
  --verify: Run verification against per-expert vendor BLAS (default: false)
  --run_benchmark: Run benchmark iterations (default: true)
"""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import get_defined_int, size_of
from std.random import random_float64, random_ui64
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import arg_parse
from internal_utils._utils import InitializationType, init_vector_launch
from layout import Coord, Idx, Layout, LayoutTensor, TileTensor, row_major

import linalg.matmul.vendor.blas as vendor_blas
from linalg.matmul.gpu.amd.mxfp4_dequant_grouped_matmul_amd import (
    mxfp4_dequant_grouped_matmul_amd,
)
from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import _cast_bf16_to_fp8
from linalg.mxfp4_dequant import dequant_mxfp4


def string_to_list(string: String) raises -> List[Int]:
    var s = string.strip("[]")
    var list = List[Int]()
    for i in s.split(","):
        try:
            list.append(Int(i))
        except:
            continue
    return list^


def bench_mxfp4_grouped_matmul[
    num_experts: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    verify: Bool,
    run_benchmark: Bool,
) raises:
    """Benchmark the full mxfp4_dequant_grouped_matmul_amd pipeline."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    var total_tokens = 0
    var max_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_tokens += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])

    var total_flops = 2 * total_tokens * N * K

    # Allocate device buffers
    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](total_tokens * K)
    var b_packed_device = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var b_scales_device = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_device = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )
    var c_device = ctx.enqueue_create_buffer[DType.bfloat16](total_tokens * N)

    # Initialize activations and packed weights
    init_vector_launch[DType.bfloat16](
        a_device,
        total_tokens * K,
        InitializationType.uniform_distribution,
        ctx,
    )
    init_vector_launch[DType.uint8](
        b_packed_device,
        num_experts * N * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )

    # Fill scales with exponent 127 (scale=1.0)
    var bs_hbuf = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    for i in range(num_experts * N * scale_K):
        bs_hbuf[i] = bitcast[DType.float8_e8m0fnu](UInt8(127))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)

    # Setup offsets and expert ids on host, copy to device
    var a_offsets_host = List(
        length=num_active_experts + 1, fill=Scalar[DType.uint32](0)
    )
    var expert_ids_host = List(
        length=num_active_experts, fill=Scalar[DType.int32](0)
    )
    a_offsets_host[0] = UInt32(0)
    for i in range(num_active_experts):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids[i])
    ctx.enqueue_copy(a_offsets_device, a_offsets_host)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host)
    ctx.synchronize()

    # Build TileTensors
    var a_tt = TileTensor(a_device, row_major((Idx(total_tokens), Idx[K]())))
    var b_packed_tt = TileTensor(
        b_packed_device, row_major[num_experts, N, packed_K]()
    )
    var b_scales_tt = TileTensor(
        b_scales_device, row_major[num_experts, N, scale_K]()
    )
    var a_offsets_tt = TileTensor(
        a_offsets_device,
        row_major(Coord(Idx(num_active_experts + 1))),
    )
    var expert_ids_tt = TileTensor(
        expert_ids_device,
        row_major(Coord(Idx(num_active_experts))),
    )
    var c_tt = TileTensor(c_device, row_major((Idx(total_tokens), Idx[N]())))

    @__copy_capture(
        c_tt,
        a_tt,
        b_packed_tt,
        b_scales_tt,
        a_offsets_tt,
        expert_ids_tt,
        max_tokens,
        total_tokens,
    )
    @parameter
    @always_inline
    def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
        mxfp4_dequant_grouped_matmul_amd(
            c_tt,
            a_tt,
            b_packed_tt,
            b_scales_tt,
            a_offsets_tt,
            expert_ids_tt,
            max_tokens,
            num_active_experts,
            total_tokens,
            ctx,
        )

    if run_benchmark:

        @parameter
        @always_inline
        def bench_func(mut bencher: Bencher) raises:
            bencher.iter_custom[kernel_launch](ctx)

        var flops = ThroughputMeasure(BenchMetric.flops, total_flops)
        var tok_str = "[" + ", ".join(num_tokens_by_expert) + "]"
        var eid_str = "[" + ", ".join(expert_ids) + "]"

        b.bench_function[bench_func](
            BenchId(
                String(
                    "mxfp4_grouped_matmul(",
                    "experts=",
                    num_experts,
                    ",active=",
                    num_active_experts,
                    ",N=",
                    N,
                    ",K=",
                    K,
                    ",tokens=",
                    tok_str,
                    ",ids=",
                    eid_str,
                    ")",
                )
            ),
            [flops],
        )
    else:
        kernel_launch(ctx, 0)

    if verify:
        # Verify against per-expert dequant + vendor BLAS
        var c_ref_device = ctx.enqueue_create_buffer[DType.bfloat16](
            total_tokens * N
        )
        ctx.enqueue_memset(c_ref_device, 0)
        ctx.synchronize()

        for i in range(num_active_experts):
            var token_start = Int(a_offsets_host[i])
            var token_end = Int(a_offsets_host[i + 1])
            var num_tokens = token_end - token_start
            var expert_id = Int(expert_ids_host[i])

            if num_tokens <= 0 or expert_id < 0:
                continue

            var b_fp8 = ctx.enqueue_create_buffer[fp8_type](N * K)
            var bp_tt = TileTensor(
                b_packed_device.unsafe_ptr() + expert_id * N * packed_K,
                row_major[N, packed_K](),
            )
            var bs_tt = TileTensor(
                b_scales_device.unsafe_ptr() + expert_id * N * scale_K,
                row_major[N, scale_K](),
            )
            var b_fp8_tt = TileTensor(b_fp8, row_major((Idx[N](), Idx[K]())))
            dequant_mxfp4(ctx, b_fp8_tt, bp_tt, bs_tt, num_rows=N, num_cols=K)

            var a_fp8 = ctx.enqueue_create_buffer[fp8_type](num_tokens * K)
            var a_slice_tt = TileTensor(
                a_device.unsafe_ptr() + token_start * K,
                row_major((Idx(num_tokens), Idx[K]())),
            )
            var a_fp8_tt = TileTensor(
                a_fp8, row_major((Idx(num_tokens), Idx[K]()))
            )
            _cast_bf16_to_fp8(ctx, a_fp8_tt, a_slice_tt, num_tokens, K)
            ctx.synchronize()

            var c_expert = ctx.enqueue_create_buffer[DType.bfloat16](
                num_tokens * N
            )
            var c_expert_tt = TileTensor(
                c_expert, row_major((Idx(num_tokens), Idx[N]()))
            )
            vendor_blas.matmul(
                ctx,
                c_expert_tt,
                a_fp8_tt,
                b_fp8_tt,
                c_row_major=True,
                transpose_b=True,
            )
            ctx.synchronize()

            # Copy expert result into the correct slice of the reference buffer
            var c_expert_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
                num_tokens * N
            )
            ctx.enqueue_copy(c_expert_host, c_expert)
            ctx.synchronize()

            var c_ref_host_slice = ctx.enqueue_create_host_buffer[
                DType.bfloat16
            ](total_tokens * N)
            ctx.enqueue_copy(c_ref_host_slice, c_ref_device)
            ctx.synchronize()

            for j in range(num_tokens * N):
                c_ref_host_slice[token_start * N + j] = c_expert_host[j]

            ctx.enqueue_copy(c_ref_device, c_ref_host_slice)
            ctx.synchronize()

            _ = b_fp8^
            _ = a_fp8^
            _ = c_expert^

        var c_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
            total_tokens * N
        )
        var c_ref_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
            total_tokens * N
        )
        ctx.enqueue_copy(c_host, c_device)
        ctx.enqueue_copy(c_ref_host, c_ref_device)
        ctx.synchronize()

        var sum_abs_diff = Float64(0.0)
        var sum_abs_ref = Float64(0.0)
        for i in range(total_tokens * N):
            var got = c_host[i].cast[DType.float64]()
            var exp = c_ref_host[i].cast[DType.float64]()
            sum_abs_diff += abs(got - exp)
            sum_abs_ref += abs(exp)

        var rel_diff = sum_abs_diff / max(sum_abs_ref, Float64(1e-12))
        print("  Verification: relative_difference =", rel_diff)
        if rel_diff > 0.01:
            raise String(
                "MXFP4 grouped matmul verification failed: rel_diff=",
                rel_diff,
            )
        print("  PASSED")

        _ = c_ref_device^

    _ = a_device^
    _ = b_packed_device^
    _ = b_scales_device^
    _ = a_offsets_device^
    _ = expert_ids_device^
    _ = c_device^
    _ = expert_ids_host^
    _ = a_offsets_host^


def bench_dequant_all_experts[
    num_experts: Int, N: Int, K: Int
](ctx: DeviceContext, mut b: Bench) raises:
    """Benchmark dequanting all expert weights from MXFP4 to FP8."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    var b_packed_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var b_scales_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var b_fp8_dev = ctx.enqueue_create_buffer[fp8_type](num_experts * N * K)

    init_vector_launch[DType.uint8](
        b_packed_dev,
        num_experts * N * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )
    var bs_hbuf = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    for i in range(num_experts * N * scale_K):
        bs_hbuf[i] = bitcast[DType.float8_e8m0fnu](UInt8(127))
    ctx.enqueue_copy(b_scales_dev, bs_hbuf)
    ctx.synchronize()

    @__copy_capture(b_packed_dev, b_scales_dev, b_fp8_dev)
    @parameter
    @always_inline
    def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
        for e in range(num_experts):
            var bp_tt = TileTensor(
                b_packed_dev.unsafe_ptr() + e * N * packed_K,
                row_major[N, packed_K](),
            )
            var bs_tt = TileTensor(
                b_scales_dev.unsafe_ptr() + e * N * scale_K,
                row_major[N, scale_K](),
            )
            var fp8_tt = TileTensor(
                b_fp8_dev.unsafe_ptr() + e * N * K,
                row_major((Idx[N](), Idx[K]())),
            )
            dequant_mxfp4(ctx, fp8_tt, bp_tt, bs_tt, num_rows=N, num_cols=K)

    @parameter
    @always_inline
    def bench_func(mut bencher: Bencher) raises:
        bencher.iter_custom[kernel_launch](ctx)

    comptime total_bytes = num_experts * (
        N * packed_K + N * scale_K + N * K * size_of[fp8_type]()
    )
    var bandwidth = ThroughputMeasure(BenchMetric.bytes, total_bytes)

    b.bench_function[bench_func](
        BenchId(
            String(
                "dequant_all_experts(",
                num_experts,
                "x",
                N,
                "x",
                K,
                ")",
            )
        ),
        [bandwidth],
    )

    _ = b_packed_dev^
    _ = b_scales_dev^
    _ = b_fp8_dev^


def main() raises:
    comptime N = get_defined_int["N", 2048]()
    comptime K = get_defined_int["K", 2048]()
    comptime num_experts = get_defined_int["num_experts", 4]()

    var num_active_experts = Int(arg_parse("num_active_experts", 2))
    var num_tokens_by_expert_str = String(
        arg_parse("num_tokens_by_expert", "64,128")
    )
    var expert_ids_str = String(arg_parse("expert_ids", "0,2"))
    var verify = arg_parse("verify", False)
    var run_benchmark = arg_parse("run_benchmark", True)
    var bench_all = arg_parse("bench_all", True)

    var num_tokens_by_expert = string_to_list(num_tokens_by_expert_str)
    var expert_ids = string_to_list(expert_ids_str)

    var b = Bench()
    with DeviceContext() as ctx:
        if run_benchmark:
            if bench_all:
                bench_dequant_all_experts[num_experts, N, K](ctx, b)

            bench_mxfp4_grouped_matmul[num_experts, N, K](
                ctx,
                b,
                num_active_experts,
                num_tokens_by_expert,
                expert_ids,
                verify,
                run_benchmark,
            )

    if run_benchmark:
        b.dump_report()
