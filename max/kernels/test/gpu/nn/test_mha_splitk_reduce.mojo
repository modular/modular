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

from std.math import exp, exp2, isnan, nan
from std.testing import assert_almost_equal, assert_true

from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.attention.gpu.mha import mha_splitk_reduce


def test_reduce[
    depth: Int, use_exp2: Bool, partitions: Int = 3
](ctx: DeviceContext) raises:
    comptime batches = 2
    comptime heads = 3
    comptime output_size = batches * heads * depth
    comptime guard_size = 13
    var partials_host = ctx.enqueue_create_host_buffer[.float32](
        partitions * output_size
    )
    var sums_host = ctx.enqueue_create_host_buffer[.float32](
        partitions * batches * heads
    )
    var maxima_host = ctx.enqueue_create_host_buffer[.float32](
        partitions * batches * heads
    )
    var partials = TileTensor(
        partials_host, row_major[partitions, batches, heads, depth]()
    )
    var sums = TileTensor(sums_host, row_major[partitions, batches, heads]())
    var maxima = TileTensor(
        maxima_host, row_major[partitions, batches, heads]()
    )
    for p in range(partitions):
        for b in range(batches):
            for h in range(heads):
                # An empty partition must not leak its NaN partial outputs.
                sums[p, b, h] = Float32(p + 1) if p != 1 else 0.0
                maxima[p, b, h] = Float32(p) * 0.5
                for d in range(depth):
                    partials[p, b, h, d] = (
                        Float32(p * 100 + b * 10 + h)
                        + Float32(d) / Float32(depth) if p
                        != 1 else nan[.float32]()
                    )

    var partials_dev = ctx.enqueue_create_buffer[.float32](len(partials_host))
    var sums_dev = ctx.enqueue_create_buffer[.float32](len(sums_host))
    var maxima_dev = ctx.enqueue_create_buffer[.float32](len(maxima_host))
    var output_dev = ctx.enqueue_create_buffer[.float32](
        output_size + guard_size
    )
    ctx.enqueue_copy(partials_dev, partials_host)
    ctx.enqueue_copy(sums_dev, sums_host)
    ctx.enqueue_copy(maxima_dev, maxima_host)
    ctx.enqueue_memset(output_dev, nan[.float32]())
    comptime kernel = mha_splitk_reduce[
        .float32, .float32, depth, heads, WARP_SIZE, use_exp2
    ]
    ctx.enqueue_function[kernel](
        partials_dev,
        output_dev,
        sums_dev,
        maxima_dev,
        Int32(batches),
        Int32(partitions),
        grid_dim=(1, heads, batches),
        block_dim=WARP_SIZE,
    )
    var output_host = ctx.enqueue_create_host_buffer[.float32](len(output_dev))
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()
    var output = TileTensor(output_host, row_major[batches, heads, depth]())
    for b in range(batches):
        for h in range(heads):
            for d in range(depth):
                var numerator = Float32(0)
                var denominator = Float32(0)
                for p in range(partitions):
                    if p == 1:
                        continue
                    var delta = Float32(p - (partitions - 1)) * 0.5
                    var weight = (
                        exp2(delta) if use_exp2 else exp(delta)
                    ) * Float32(p + 1)
                    numerator += partials[p, b, h, d] * weight
                    denominator += weight
                var expected = numerator / denominator
                assert_almost_equal(
                    output[b, h, d], expected, atol=1e-4, rtol=1e-5
                )
    for i in range(output_size, output_size + guard_size):
        assert_true(isnan(output_host[i]))


def main() raises:
    with DeviceContext() as ctx:
        test_reduce[32, False](ctx)
        test_reduce[128, True](ctx)
        test_reduce[32, True, 9](ctx)
        test_reduce[128, False, 9](ctx)
