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
"""Benchmark for the rowwise FP8 grouped matmul, through its public launcher.

Llama-4-Scout FP8-dynamic MoE shapes: 16 experts, `gate_up` weights transposed
to `[E, N=16384, K=5120]` (one fused gate+up projection) and `down` to
`[E, N=5120, K=8192]`, at decode (1, 2, 4 and 16 tokens on each expert) and at
prefill (128 tokens on each, i.e. 2048 tokens: the scalar kernel the baseline
arm runs needs about 330 ms per call for the gate_up prefill at 128, and kbench
stops a binary group that runs past its per-item timeout, which it does at 256
tokens per expert).

`out_type` selects which kernel of the op runs: on SM100 a bf16 output takes
the persistent tensor-core grouped GEMM, and every other case (fp32 or fp16
output, and every non-SM100 GPU) takes the tiled CUDA-core kernel. The shapes
are comptime so both paths get their static shapes (the tensor-core path is
only chosen for static shapes).
"""

from std.math import ceildiv
from std.sys import get_defined_dtype, get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from internal_utils._utils import InitializationType, init_vector_launch
from linalg.grouped_matmul import grouped_matmul_rowwise_dynamic_scaled_fp8
from layout import Idx, TileTensor, row_major

from std.utils.index import IndexList


def string_to_list(string: String) raises -> List[Int]:
    var s = string.strip("[]")
    var list = List[Int]()
    # Let a malformed entry raise: skipping it would silently shorten the
    # per-expert token list and benchmark a different routing than asked for.
    for i in s.split(","):
        list.append(Int(i))
    return list^


def bench_grouped_matmul_rowwise_fp8[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],  # (N, K)
](
    ctx: DeviceContext[],
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_input: List[Int],
    init_type: InitializationType,
) raises:
    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    if (
        len(num_tokens_by_expert) != num_active_experts
        or len(expert_ids_input) != num_active_experts
    ):
        raise Error(
            String(
                "num_tokens_by_expert and expert_ids need ",
                num_active_experts,
                " entries each (num_active_experts)",
            )
        )

    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for num_tokens in num_tokens_by_expert:
        # A negative count would shrink every buffer below the rows its offsets
        # claim, and its UInt32 offset would wrap to a huge row start, so the
        # kernel would read and write outside the activations.
        if num_tokens < 0:
            raise Error(
                String(
                    "num_tokens_by_expert must be non-negative, got ",
                    num_tokens,
                )
            )
        total_num_tokens += num_tokens
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, num_tokens)

    print(
        t"Config: num_active_experts={num_active_experts} N={N} K={K}"
        t" num_experts={num_experts} total_tokens={total_num_tokens}"
    )

    # Ragged routing, exactly as the MoE layer hands it to the op: prefix-sum
    # offsets, the real expert id per group, and the two host scalars.
    var a_offsets_host = List[UInt32](
        length=num_active_experts + 1, fill=UInt32(0)
    )
    var expert_ids_host = List[Int32](length=num_active_experts, fill=Int32(0))
    # Tokens routed to an inactive (-1) group keep their rows in the buffers but
    # have no matmul behind them (the kernel stores the zero reduction), so they
    # must not count toward the reported FLOPs.
    var matmul_tokens = 0
    for i in range(num_active_experts):
        if expert_ids_input[i] < -1 or expert_ids_input[i] >= num_experts:
            raise Error(
                String(
                    "expert id ",
                    expert_ids_input[i],
                    " is outside [-1, ",
                    num_experts,
                    ")",
                )
            )
        if expert_ids_input[i] != -1:
            matmul_tokens += num_tokens_by_expert[i]
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_input[i])

    var total_flops = 2 * matmul_tokens * N * K

    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N
    var a_scales_size = total_num_tokens
    var b_scales_size = num_experts * N

    var a_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scales_buffer = ctx.enqueue_create_buffer[.float32](a_scales_size)
    var b_scales_buffer = ctx.enqueue_create_buffer[.float32](b_scales_size)
    var a_offsets_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )

    var a_dev = TileTensor(a_buffer, row_major(total_num_tokens, Idx[K]))
    var b_dev = TileTensor(
        b_buffer, row_major(Idx[num_experts], Idx[N], Idx[K])
    )
    var c_dev = TileTensor(c_buffer, row_major(total_num_tokens, Idx[N]))
    var a_scales_dev = TileTensor(
        a_scales_buffer, row_major(total_num_tokens, Idx[1])
    )
    var b_scales_dev = TileTensor(
        b_scales_buffer, row_major(Idx[num_experts], Idx[N], Idx[1])
    )
    var a_offsets_dev = TileTensor(
        a_offsets_buffer, row_major(num_active_experts + 1)
    )
    var expert_ids_dev = TileTensor(
        expert_ids_buffer, row_major(num_active_experts)
    )

    init_vector_launch[a_type](a_buffer, a_size, init_type, ctx)
    init_vector_launch[b_type](b_buffer, b_size, init_type, ctx)
    init_vector_launch[.float32](a_scales_buffer, a_scales_size, init_type, ctx)
    init_vector_launch[.float32](b_scales_buffer, b_scales_size, init_type, ctx)
    ctx.enqueue_copy(a_offsets_buffer, a_offsets_host)
    ctx.enqueue_copy(expert_ids_buffer, expert_ids_host)

    @inline(.always)
    def bench_func(
        mut bench: Bencher,
    ) {
        var a_dev,
        var b_dev,
        var c_dev,
        var a_scales_dev,
        var b_scales_dev,
        var a_offsets_dev,
        var expert_ids_dev,
        imm,
        var max_num_tokens_by_expert,
        var num_active_experts,
    }:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext[], iteration: Int) raises {imm}:
            grouped_matmul_rowwise_dynamic_scaled_fp8[transpose_b=True,](
                c_dev,
                a_dev,
                b_dev,
                a_scales_dev,
                b_scales_dev,
                a_offsets_dev,
                expert_ids_dev,
                max_num_tokens_by_expert,
                num_active_experts,
                ctx,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)

    # Keep the id short: kplot prints it as the x label, and the dtype names in
    # full push the token counts off the axis.
    comptime out_tag = (
        "bf16" if c_type
        == DType.bfloat16 else ("f32" if c_type == DType.float32 else "f16")
    )
    var name = String(
        "fp8->",
        out_tag,
        " E",
        num_active_experts,
        " T",
        total_num_tokens,
        " N",
        N,
        " K",
        K,
    )
    bench.bench_function(
        bench_func,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    _ = a_buffer^
    _ = b_buffer^
    _ = c_buffer^
    _ = a_scales_buffer^
    _ = b_scales_buffer^
    _ = a_offsets_buffer^
    _ = expert_ids_buffer^
    _ = a_offsets_host^
    _ = expert_ids_host^


def main() raises:
    comptime in_type = get_defined_dtype["in_type", .float8_e4m3fn]()
    comptime out_type = get_defined_dtype["out_type", .bfloat16]()
    comptime N = get_defined_int["N", 16384]()
    comptime K = get_defined_int["K", 5120]()
    comptime num_experts = get_defined_int["num_experts", 16]()

    var num_active_experts = Int(arg_parse("num_active_experts", 16))
    var num_tokens_by_expert = string_to_list(
        String(
            arg_parse(
                "num_tokens_by_expert",
                "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            )
        )
    )
    var expert_ids = string_to_list(
        String(arg_parse("expert_ids", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"))
    )
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )

    var b = Bench()
    comptime expert_shape = (N, K)

    with DeviceContext() as ctx:
        bench_grouped_matmul_rowwise_fp8[
            in_type, out_type, num_experts, expert_shape
        ](
            ctx,
            b,
            num_active_experts,
            num_tokens_by_expert,
            expert_ids,
            init_type,
        )

    b.dump_report()
