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

from std.math import fma
from std.testing import assert_almost_equal, assert_equal
from std.utils.numerics import nan

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from layout.coord import Coord
from layout.layout import Layout
from layout._fillers import random
from nn.moe import moe_finalize, moe_finalize_bounded


def test_moe_finalize[
    down_type: DType,
    weight_type: DType,
    out_type: DType,
    num_experts_per_token: Int,
    row_pad: Int = 0,
](num_tokens: Int, hidden: Int, ctx: DeviceContext) raises:
    """Checks `moe_finalize` bit for bit against a host-computed reference.

    Reads each token's `num_experts_per_token` rows through a reversed
    permutation and reproduces the kernel's sum on the host: one fused
    multiply-add per (row, weight) in ascending k, in fp32, which is what the
    device's `v_fma_f32` chain computes. Any reordering, a dropped fusion or a
    different accumulation width shows up as a mismatched bit.

    `row_pad` adds elements to each `down` row, so the view's row stride is
    wider than its width. The kernel must fall back to the generic body there,
    which reads rows through the layout instead of assuming they are dense.
    """
    var total_m = num_tokens * num_experts_per_token
    var down_stride = hidden + row_pad

    var down_host = ctx.enqueue_create_host_buffer[down_type](
        total_m * down_stride
    )
    var restore_order_host = ctx.enqueue_create_host_buffer[.uint32](total_m)
    var router_weight_host = ctx.enqueue_create_host_buffer[weight_type](
        total_m
    )
    var output_host = ctx.enqueue_create_host_buffer[out_type](
        num_tokens * hidden
    )
    ctx.synchronize()

    # `random` fills a view densely and ignores its stride, so fill the whole
    # buffer through a dense view of it. Otherwise the padded case below would
    # compare never-written bytes, and a stale NaN would read as a mismatch.
    var down_host_dense = TileTensor(down_host, row_major(total_m, down_stride))
    random(down_host_dense, min=-1.0, max=1.0)

    # The unit stride stays a compile-time `Idx[1]`, so `down` is statically
    # row-major and the only thing deciding between the gather launch and the
    # generic body is the runtime `dense` check in `moe_finalize`: a zero pad
    # takes the gather, a padded row stride takes the generic body.
    var down_host_2d = TileTensor(
        down_host, Layout(Coord(total_m, hidden), Coord(down_stride, Idx[1]))
    )
    var router_weight_host_2d = TileTensor(
        router_weight_host, row_major(num_tokens, num_experts_per_token)
    )

    random(router_weight_host_2d, min=-1.0, max=1.0)

    for i in range(total_m):
        restore_order_host[i] = UInt32(total_m - 1 - i)

    var down_dev = ctx.enqueue_create_buffer[down_type](total_m * down_stride)
    var restore_order_dev = ctx.enqueue_create_buffer[.uint32](total_m)
    var router_weight_dev = ctx.enqueue_create_buffer[weight_type](total_m)
    var output_dev = ctx.enqueue_create_buffer[out_type](num_tokens * hidden)

    ctx.enqueue_copy(down_dev, down_host)
    ctx.enqueue_copy(restore_order_dev, restore_order_host)
    ctx.enqueue_copy(router_weight_dev, router_weight_host)

    var down = TileTensor(
        down_dev, Layout(Coord(total_m, hidden), Coord(down_stride, Idx[1]))
    )
    var restore_order = TileTensor(restore_order_dev, row_major(total_m))
    var router_weight = TileTensor(
        router_weight_dev, row_major(num_tokens, num_experts_per_token)
    )
    var output = TileTensor(output_dev, row_major(num_tokens, hidden))

    moe_finalize(output, down, restore_order, router_weight, ctx)

    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var output_2d = TileTensor(output_host, row_major(num_tokens, hidden))
    var mismatches = 0
    for t in range(num_tokens):
        for col in range(hidden):
            var acc = Float32(0)
            for k in range(num_experts_per_token):
                var row = Int(restore_order_host[t * num_experts_per_token + k])
                var x = Float32(down_host_2d[row, col])
                var w = Float32(router_weight_host_2d[t, k])
                acc = fma(x, w, acc)
            var want = acc.cast[out_type]()[0]
            var got = output_2d[t, col]
            if got != want:
                if mismatches == 0:
                    print(
                        "first mismatch at",
                        t,
                        col,
                        "got",
                        got,
                        "want",
                        want,
                    )
                mismatches += 1
    assert_equal(mismatches, 0)


def test_moe_finalize_bounded[
    num_experts_per_token: Int
](num_tokens: Int, hidden: Int, valid: Int, ctx: DeviceContext) raises:
    """Checks that `moe_finalize_bounded` ignores the rows past its bound.

    The rows from `valid` on hold NaN, so reading even one of them, weighted
    or not, poisons the output; the reference adds only the rows below it.
    """
    var total_m = num_tokens * num_experts_per_token

    var down_host = ctx.enqueue_create_host_buffer[.bfloat16](total_m * hidden)
    var restore_order_host = ctx.enqueue_create_host_buffer[.uint32](total_m)
    var router_weight_host = ctx.enqueue_create_host_buffer[.float32](total_m)
    var valid_host = ctx.enqueue_create_host_buffer[.uint32](1)
    var output_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden
    )
    ctx.synchronize()

    var down_host_2d = TileTensor(down_host, row_major(total_m, hidden))
    var router_weight_host_2d = TileTensor(
        router_weight_host, row_major(num_tokens, num_experts_per_token)
    )
    random(down_host_2d, min=-1.0, max=1.0)
    random(router_weight_host_2d, min=-1.0, max=1.0)
    for row in range(valid, total_m):
        for col in range(hidden):
            down_host_2d[row, col] = nan[.bfloat16]()
    for i in range(total_m):
        restore_order_host[i] = UInt32(total_m - 1 - i)
    valid_host[0] = UInt32(valid)

    var down_dev = ctx.enqueue_create_buffer[.bfloat16](total_m * hidden)
    var restore_order_dev = ctx.enqueue_create_buffer[.uint32](total_m)
    var router_weight_dev = ctx.enqueue_create_buffer[.float32](total_m)
    var valid_dev = ctx.enqueue_create_buffer[.uint32](1)
    var output_dev = ctx.enqueue_create_buffer[.bfloat16](num_tokens * hidden)
    ctx.enqueue_copy(down_dev, down_host)
    ctx.enqueue_copy(restore_order_dev, restore_order_host)
    ctx.enqueue_copy(router_weight_dev, router_weight_host)
    ctx.enqueue_copy(valid_dev, valid_host)

    moe_finalize_bounded(
        TileTensor(output_dev, row_major(num_tokens, hidden)),
        TileTensor(down_dev, row_major(total_m, hidden)),
        TileTensor(restore_order_dev, row_major(total_m)),
        TileTensor(
            router_weight_dev, row_major(num_tokens, num_experts_per_token)
        ),
        TileTensor(valid_dev, row_major(1)),
        ctx,
    )
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var output_2d = TileTensor(output_host, row_major(num_tokens, hidden))
    for t in range(num_tokens):
        for col in range(hidden):
            var acc = Float32(0)
            for k in range(num_experts_per_token):
                var row = Int(restore_order_host[t * num_experts_per_token + k])
                if row >= valid:
                    continue
                var w = Float32(router_weight_host_2d[t, k])
                acc += Float32(down_host_2d[row, col]) * w
            assert_almost_equal(
                output_2d[t, col],
                acc.cast[.bfloat16](),
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

        # A top-k the gather kernel has no instantiation for, so the generic
        # elementwise body runs and has to agree with it too.
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 3](33, 3584, ctx)

        # Kimi K3 decode: hidden 3584, top 16 of 896 experts, bf16 rows, fp32
        # router weights, bf16 output. 32 tokens is one decode step on a TP8
        # rank; 33 puts a partial workgroup on the tail of the launch.
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 16](32, 3584, ctx)
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 16](33, 3584, ctx)

        # The same share at a token count past the small-block threshold (146
        # tokens at this width), so the 256-thread launch is checked too. The
        # reference is a host loop over rows x columns x top-k, so this stays
        # well below the 8192-token prefill shape the bench covers.
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 16](256, 3584, ctx)

        # A `down` view whose rows are wider than its width, as a column slice
        # of a padded tensor would be. The gather launch addresses rows as
        # `row * hidden`, so it must decline this view and let the generic body
        # read through the real strides.
        test_moe_finalize[.bfloat16, .float32, .bfloat16, 16, 64](32, 3584, ctx)

        # Bounded, through every launch: the gather kernel's narrow-vector,
        # small-block and 256-thread launches (top-8, hidden 5120), the generic
        # vector body (a top-k with no gather instantiation) and the generic
        # scalar body (a ragged width). The bounds cover no valid row, a
        # partial bound and every row valid.
        test_moe_finalize_bounded[8](3, 5120, 0, ctx)
        test_moe_finalize_bounded[8](32, 5120, 37, ctx)
        test_moe_finalize_bounded[8](32, 5120, 256, ctx)
        test_moe_finalize_bounded[8](128, 5120, 300, ctx)
        test_moe_finalize_bounded[3](33, 3584, 50, ctx)
        test_moe_finalize_bounded[6](7, 4093, 20, ctx)
