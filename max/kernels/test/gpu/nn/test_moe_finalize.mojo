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

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout._fillers import random
from nn.moe import moe_finalize
from std.testing import assert_almost_equal


def test_moe_finalize[
    down_type: DType,
    weight_type: DType,
    out_type: DType,
    num_experts_per_token: Int,
](num_tokens: Int, hidden: Int, ctx: DeviceContext) raises:
    """Checks `moe_finalize` against a host-computed reference.

    Reads each token's `num_experts_per_token` rows through a reversed
    permutation and reproduces the kernel's fp32-accumulated weighted sum on
    the host.
    """
    var total_m = num_tokens * num_experts_per_token

    var down_host = ctx.enqueue_create_host_buffer[down_type](total_m * hidden)
    var restore_order_host = ctx.enqueue_create_host_buffer[.uint32](total_m)
    var router_weight_host = ctx.enqueue_create_host_buffer[weight_type](
        total_m
    )
    var output_host = ctx.enqueue_create_host_buffer[out_type](
        num_tokens * hidden
    )
    ctx.synchronize()

    var down_host_2d = TileTensor(down_host, row_major(total_m, hidden))
    var router_weight_host_2d = TileTensor(
        router_weight_host, row_major(num_tokens, num_experts_per_token)
    )

    random(down_host_2d, min=-1.0, max=1.0)
    random(router_weight_host_2d, min=-1.0, max=1.0)

    for i in range(total_m):
        restore_order_host[i] = UInt32(total_m - 1 - i)

    var down_dev = ctx.enqueue_create_buffer[down_type](total_m * hidden)
    var restore_order_dev = ctx.enqueue_create_buffer[.uint32](total_m)
    var router_weight_dev = ctx.enqueue_create_buffer[weight_type](total_m)
    var output_dev = ctx.enqueue_create_buffer[out_type](num_tokens * hidden)

    ctx.enqueue_copy(down_dev, down_host)
    ctx.enqueue_copy(restore_order_dev, restore_order_host)
    ctx.enqueue_copy(router_weight_dev, router_weight_host)

    var down = TileTensor(down_dev, row_major(total_m, hidden))
    var restore_order = TileTensor(restore_order_dev, row_major(total_m))
    var router_weight = TileTensor(
        router_weight_dev, row_major(num_tokens, num_experts_per_token)
    )
    var output = TileTensor(output_dev, row_major(num_tokens, hidden))

    moe_finalize["gpu"](output, down, restore_order, router_weight, ctx)

    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var output_2d = TileTensor(output_host, row_major(num_tokens, hidden))
    for t in range(num_tokens):
        for col in range(hidden):
            var acc = Float32(0)
            for k in range(num_experts_per_token):
                var row = Int(restore_order_host[t * num_experts_per_token + k])
                var w = Float32(router_weight_host_2d[Coord(t, k)])
                acc += Float32(down_host_2d[Coord(row, col)]) * w
            assert_almost_equal(
                output_2d[Coord(t, col)],
                acc.cast[out_type](),
                atol=2e-2,
                rtol=2e-2,
            )


def main() raises:
    with DeviceContext() as ctx:
        # Decode-shaped: 64 tokens, top-6, hidden 4096 (a multiple of the
        # bf16 SIMD width, so the vectorized path runs).
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 6](64, 4096, ctx)

        # A hidden width that is not a multiple of the bf16 SIMD width, so the
        # scalar path runs.
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 6](37, 4093, ctx)

        # Single-token decode.
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 6](1, 4096, ctx)

        # A different top-k and an all-bfloat16 router weight.
        test_moe_finalize[.bfloat16, .bfloat16, .bfloat16, 8](5, 1536, ctx)
