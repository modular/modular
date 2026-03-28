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

from std.random import rand
from std.sys import get_defined_int

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from nn.moe import moe_create_indices, router_group_limited


def bench_moe_create_indices[
    num_tokens: Int, num_experts: Int
](ctx: DeviceContext, mut b: Bench) raises:
    var topk_h = ctx.enqueue_create_host_buffer[DType.uint32](num_tokens)
    for i in range(num_tokens):
        topk_h[i] = UInt32(i % num_experts)

    var topk_d = ctx.enqueue_create_buffer[DType.uint32](num_tokens)
    ctx.enqueue_copy[DType.uint32](topk_d.unsafe_ptr(), topk_h)

    var token_expert_order_d = ctx.enqueue_create_buffer[DType.uint32](
        num_tokens
    )
    var expert_start_indices_d = ctx.enqueue_create_buffer[DType.uint32](
        num_experts + 1
    )
    var restore_token_order_d = ctx.enqueue_create_buffer[DType.uint32](
        num_tokens
    )
    var expert_ids_d = ctx.enqueue_create_buffer[DType.int32](num_experts)
    var expert_usage_stats_d = ctx.enqueue_create_buffer[DType.uint32](2)

    var token_expert_order = TileTensor(
        token_expert_order_d, row_major[num_tokens]()
    )
    var expert_start_indices = TileTensor(
        expert_start_indices_d, row_major[num_experts + 1]()
    )
    var restore_token_order = TileTensor(
        restore_token_order_d, row_major[num_tokens]()
    )
    var expert_ids = TileTensor(expert_ids_d, row_major[num_experts]())
    var expert_usage_stats = TileTensor(expert_usage_stats_d, row_major[2]())
    var topk_ids = TileTensor(topk_d, row_major[num_tokens]())

    var context = DeviceContextPtr(ctx)

    @always_inline
    @__copy_capture(
        token_expert_order,
        expert_start_indices,
        restore_token_order,
        expert_ids,
        expert_usage_stats,
        topk_ids,
        context,
    )
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            moe_create_indices[input_type=DType.uint32, target="gpu"](
                token_expert_order,
                expert_start_indices,
                restore_token_order,
                expert_ids,
                expert_usage_stats,
                topk_ids,
                context,
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "moe_create_indices",
            input_id=String(num_tokens, "tok/", num_experts, "exp"),
        )
    )

    ctx.synchronize()

    _ = topk_d
    _ = topk_h
    _ = token_expert_order_d
    _ = expert_start_indices_d
    _ = restore_token_order_d
    _ = expert_ids_d
    _ = expert_usage_stats_d


def bench_router_group_limited[
    num_tokens: Int,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
](ctx: DeviceContext, mut b: Bench) raises:
    comptime dtype = DType.float32

    var scores_h = alloc[Scalar[dtype]](num_tokens * n_routed_experts)
    var bias_h = alloc[Scalar[dtype]](n_routed_experts)
    rand[dtype](scores_h, num_tokens * n_routed_experts)
    rand[dtype](bias_h, n_routed_experts)

    var scores_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_routed_experts
    )
    var bias_d = ctx.enqueue_create_buffer[dtype](n_routed_experts)
    var indices_d = ctx.enqueue_create_buffer[DType.int32](
        num_tokens * n_experts_per_tok
    )
    var weights_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_experts_per_tok
    )

    ctx.enqueue_copy(scores_d, scores_h)
    ctx.enqueue_copy(bias_d, bias_h)

    var expert_indices = TileTensor(
        indices_d,
        row_major[num_tokens, n_experts_per_tok](),
    )
    var expert_weights = TileTensor(
        weights_d,
        row_major[num_tokens, n_experts_per_tok](),
    )
    var expert_scores = TileTensor(
        scores_d,
        row_major[num_tokens, n_routed_experts](),
    )
    var expert_bias = TileTensor(bias_d, row_major[n_routed_experts]())
    var routed_scaling_factor = Float32(1.0)

    var context = DeviceContextPtr(ctx)

    @always_inline
    @__copy_capture(
        expert_indices,
        expert_weights,
        expert_scores,
        expert_bias,
        routed_scaling_factor,
        context,
    )
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            router_group_limited[
                scores_type=dtype,
                bias_type=dtype,
                n_routed_experts=n_routed_experts,
                n_experts_per_tok=n_experts_per_tok,
                n_groups=n_groups,
                topk_group=topk_group,
                norm_weights=True,
                target="gpu",
            ](
                expert_indices,
                expert_weights,
                expert_scores.as_immut(),
                expert_bias.as_immut(),
                routed_scaling_factor,
                context,
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "router_group_limited",
            input_id=String(
                num_tokens,
                "tok/",
                n_routed_experts,
                "exp/",
                n_experts_per_tok,
                "per_tok",
            ),
        )
    )

    ctx.synchronize()

    _ = scores_d
    _ = bias_d
    _ = indices_d
    _ = weights_d

    scores_h.free()
    bias_h.free()


def main() raises:
    comptime num_tokens = get_defined_int["num_tokens", 4096]()
    comptime num_experts = get_defined_int["num_experts", 256]()
    comptime n_experts_per_tok = get_defined_int["n_experts_per_tok", 8]()
    comptime n_groups = get_defined_int["n_groups", 8]()
    comptime topk_group = get_defined_int["topk_group", 4]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_moe_create_indices[num_tokens, num_experts](ctx, m)
        bench_router_group_limited[
            num_tokens, num_experts, n_experts_per_tok, n_groups, topk_group
        ](ctx, m)

    m.dump_report()
