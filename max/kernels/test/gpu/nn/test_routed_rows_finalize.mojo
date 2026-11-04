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
# The routed-rows finalize against a model of its roundings, at prime, ragged
# and persistent token counts, routed counts from none to all, top_k 2 and 8,
# float32 and bf16 output; then each planted fault, which must fail.
#
# Every row past the routed count holds NaN, and the output starts as a
# sentinel, so a dropped slot that is read, or a token that is skipped, shows.
# The model rounds where the kernel does and adds in float64; each output is
# held to one bf16 ulp of every term it sums. A bf16 output must also equal,
# bit for bit, the float32 launch's output rounded to nearest even.

from std.math import isfinite, sqrt
from std.random import rand, seed
from std.testing import assert_true
from std.utils.numerics import nan

from max.gpu.host import DeviceBuffer, DeviceContext

from nn.routed_rows_finalize import (
    HIDDEN,
    POISONS,
    In,
    RoutedRowsFinalizeArgs,
    routed_rows_finalize,
)

comptime EPS = Float32(1e-6)
comptime GROUPS = 5
comptime SENTINEL = Float32(-12288.0)


def _in[T: DType](b: DeviceBuffer[T]) -> In[T]:
    return b.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()


def launch[
    out_dtype: DType, //, top_k: Int, poison: Int
](
    tokens: Int,
    down: DeviceBuffer[DType.bfloat16],
    gamma: DeviceBuffer[DType.bfloat16],
    offsets: DeviceBuffer[DType.uint32],
    restore: DeviceBuffer[DType.uint32],
    weight: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[out_dtype],
    ctx: DeviceContext,
) raises:
    routed_rows_finalize[top_k, poison](
        RoutedRowsFinalizeArgs(
            _in(down),
            _in(gamma),
            _in(offsets),
            _in(restore),
            _in(weight),
            dst.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(tokens),
            Int64(GROUPS - 1),
            EPS,
        ),
        ctx,
    )


def check[
    top_k: Int, poison: Int, out_dtype: DType
](tokens: Int, routed: Int, ctx: DeviceContext) raises -> Int:
    """Returns the channels out of bound or not finite."""
    var rows = tokens * top_k
    var n = max(rows, 1)
    var down_h = ctx.enqueue_create_host_buffer[.bfloat16](n * HIDDEN)
    var gamma_h = ctx.enqueue_create_host_buffer[.bfloat16](HIDDEN)
    var offs_h = ctx.enqueue_create_host_buffer[.uint32](GROUPS)
    var rest_h = ctx.enqueue_create_host_buffer[.uint32](n)
    var w_h = ctx.enqueue_create_host_buffer[.float32](n)
    var out_h = ctx.enqueue_create_host_buffer[out_dtype](
        max(tokens, 1) * HIDDEN
    )
    var wide_h = ctx.enqueue_create_host_buffer[.float32](
        max(tokens, 1) * HIDDEN
    )
    ctx.synchronize()
    seed(tokens * 31 + routed)
    rand(down_h.unsafe_ptr(), rows * HIDDEN)
    rand(gamma_h.unsafe_ptr(), HIDDEN)
    rand(w_h.unsafe_ptr(), rows)
    for i in range(rows * HIDDEN):
        down_h[i] = (down_h[i] - 0.5) * 6.0
    for c in range(HIDDEN):
        gamma_h[c] = gamma_h[c] - 0.5
    for i in range(routed * HIDDEN, rows * HIDDEN):
        down_h[i] = nan[DType.bfloat16]()
    for g in range(GROUPS):
        offs_h[g] = UInt32(routed * g // (GROUPS - 1))
    for i in range(rows):
        rest_h[i] = UInt32((i * 7919 + tokens) % rows)
    for i in range(max(tokens, 1) * HIDDEN):
        out_h[i] = SENTINEL.cast[out_dtype]()
        wide_h[i] = SENTINEL

    var down = ctx.enqueue_create_buffer[DType.bfloat16](n * HIDDEN)
    var gamma = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var offsets = ctx.enqueue_create_buffer[DType.uint32](GROUPS)
    var restore = ctx.enqueue_create_buffer[DType.uint32](n)
    var weight = ctx.enqueue_create_buffer[DType.float32](n)
    var out_d = ctx.enqueue_create_buffer[out_dtype](max(tokens, 1) * HIDDEN)
    var wide_d = ctx.enqueue_create_buffer[DType.float32](
        max(tokens, 1) * HIDDEN
    )
    ctx.enqueue_copy(down, down_h)
    ctx.enqueue_copy(gamma, gamma_h)
    ctx.enqueue_copy(offsets, offs_h)
    ctx.enqueue_copy(restore, rest_h)
    ctx.enqueue_copy(weight, w_h)
    ctx.enqueue_copy(out_d, out_h)
    ctx.enqueue_copy(wide_d, wide_h)
    launch[top_k, poison](
        tokens, down, gamma, offsets, restore, weight, out_d, ctx
    )
    # The bf16 output's reference: the float32 launch, unpoisoned past the
    # faults both launches share.
    comptime if out_dtype != DType.float32:
        launch[top_k, poison if poison < 4 else 0](
            tokens, down, gamma, offsets, restore, weight, wide_d, ctx
        )
    ctx.enqueue_copy(out_h, out_d)
    ctx.enqueue_copy(wide_h, wide_d)
    ctx.synchronize()

    var bad = 0
    var normed = List[Float64](length=top_k * HIDDEN, fill=0)
    for t in range(tokens):
        for k in range(top_k):
            var row = Int(rest_h[t * top_k + k])
            if row >= routed:
                continue
            var acc = Float64(0)
            for c in range(HIDDEN):
                var x = Float64(
                    (down_h[row * HIDDEN + c] * (1 + gamma_h[c])).cast[
                        .float32
                    ]()
                )
                acc += x * x
            var inv = 1.0 / sqrt(acc / Float64(HIDDEN) + Float64(EPS))
            for c in range(HIDDEN):
                var x = Float64(
                    (down_h[row * HIDDEN + c] * (1 + gamma_h[c])).cast[
                        .float32
                    ]()
                )
                normed[k * HIDDEN + c] = Float64(
                    Scalar[.bfloat16](Float32(x * inv)).cast[.float32]()
                )
        for c in range(HIDDEN):
            var want = Float64(0)
            var bound = Float64(0)
            for k in range(top_k):
                if Int(rest_h[t * top_k + k]) >= routed:
                    continue
                var term = normed[k * HIDDEN + c] * Float64(w_h[t * top_k + k])
                want += term
                bound += abs(term) * 0.0078125
            var got = out_h[t * HIDDEN + c].cast[.float32]()
            comptime if out_dtype != DType.float32:
                bound += abs(want) * 0.00390625
                if (
                    out_h[t * HIDDEN + c]
                    != wide_h[t * HIDDEN + c].cast[out_dtype]()
                ):
                    bad += 1
                    continue
            if not isfinite(got) or abs(Float64(got) - want) > bound + 1e-30:
                bad += 1
    print(
        "  top_k ",
        top_k,
        " ",
        out_dtype,
        " poison ",
        poison,
        " tokens ",
        tokens,
        " routed ",
        routed,
        ": ",
        bad,
        " of ",
        tokens * HIDDEN,
        " channels out of bound",
        sep="",
    )
    _ = down
    _ = gamma
    _ = offsets
    _ = restore
    _ = weight
    _ = out_d
    _ = wide_d
    return bad


def run[
    top_k: Int, poison: Int, out_dtype: DType = DType.float32
](ctx: DeviceContext) raises -> Int:
    """Cases that fail."""
    var failed = 0
    for tokens in [1, 3, 8, 37, 149]:
        var rows = tokens * top_k
        for routed in [rows, 0, (rows * 2) // 5 + 1]:
            failed += Int(
                check[top_k, poison, out_dtype](tokens, routed, ctx) != 0
            )
    _ = check[top_k, poison, out_dtype](0, 0, ctx)
    return failed


def run_all[poison: Int](ctx: DeviceContext) raises -> Int:
    """Cases that fail over both launch variants and both output dtypes."""
    return (
        run[2, poison](ctx)
        + run[8, poison](ctx)
        + run[2, poison, DType.bfloat16](ctx)
        + run[8, poison, DType.bfloat16](ctx)
    )


def main() raises:
    with DeviceContext() as ctx:
        var failed = run_all[0](ctx)
        assert_true(failed == 0, String(failed, " cases failed"))
        comptime for poison in range(1, POISONS + 1):
            var caught = run_all[poison](ctx)
            assert_true(
                caught > 0, String("poison ", poison, ": no case failed")
            )
            print("poison", poison, "caught by", caught, "cases")
        print("All tests passed!")
