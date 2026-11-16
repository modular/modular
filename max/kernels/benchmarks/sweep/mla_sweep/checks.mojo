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
"""Agreement between two arms' outputs, and the capture check."""

from std.math import ceildiv, sqrt
from std.memory import bitcast
from std.utils.numerics import isfinite

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)

comptime _BLOCK = 256
comptime _GRID = 1024
comptime _SUMS = 6


def _agreement_kernel(
    reference: Pointer[BFloat16, ImmutAnyOrigin],
    candidate: Pointer[BFloat16, ImmutAnyOrigin],
    n: Int64,
    partials: Pointer[Float64, MutAnyOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var dot = Float64(0)
    var norm_ref = Float64(0)
    var norm_got = Float64(0)
    var sum_abs_ref = Float64(0)
    var sum_abs_error = Float64(0)
    var nonfinite = Float64(0)
    var i = tid
    while i < Int(n):
        var r = reference[unsafe_offset=i].cast[DType.float64]()
        var g = candidate[unsafe_offset=i].cast[DType.float64]()
        if isfinite(g):
            dot += r * g
            norm_ref += r * r
            norm_got += g * g
            sum_abs_ref += abs(r)
            sum_abs_error += abs(g - r)
        else:
            nonfinite += 1
        i += stride
    var base = tid * _SUMS
    partials[unsafe_offset=base] = dot
    partials[unsafe_offset=base + 1] = norm_ref
    partials[unsafe_offset=base + 2] = norm_got
    partials[unsafe_offset=base + 3] = sum_abs_ref
    partials[unsafe_offset=base + 4] = sum_abs_error
    partials[unsafe_offset=base + 5] = nonfinite


@fieldwise_init
struct Agreement(ImplicitlyCopyable, Writable):
    """How closely a candidate output reproduces a reference output."""

    var cosine: Float64
    var error_fraction: Float64
    var nonfinite: Int

    def ok(self, min_cosine: Float64, max_error_fraction: Float64) -> Bool:
        return (
            self.nonfinite == 0
            and self.cosine >= min_cosine
            and self.error_fraction <= max_error_fraction
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "cosine=",
            self.cosine,
            " error_fraction=",
            self.error_fraction,
            " nonfinite=",
            self.nonfinite,
        )


def agreement(
    ctx: DeviceContext,
    reference: DeviceBuffer[DType.bfloat16, ...],
    candidate: DeviceBuffer[DType.bfloat16, ...],
    count: Int,
) raises -> Agreement:
    """Returns the cosine and mean error fraction of the first `count`
    elements of `candidate` against `reference`.

    Reduces on the device, so a gigabyte output costs one pass rather than a
    host copy. A zero reference yields a cosine of 0, so an output that is
    identically zero cannot pass.
    """
    var grid = min(_GRID, max(1, ceildiv(count, _BLOCK)))
    var partials = ctx.enqueue_create_buffer[DType.float64](
        grid * _BLOCK * _SUMS
    )
    ctx.enqueue_function[_agreement_kernel](
        reference.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        candidate.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        Int64(count),
        partials.unsafe_ptr().as_unsafe_any_origin(),
        grid_dim=grid,
        block_dim=_BLOCK,
    )
    var sums: Array[Float64, _SUMS] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with partials.map_to_host() as host:
        for t in range(grid * _BLOCK):
            for k in range(_SUMS):
                sums[k] += host[t * _SUMS + k]
    var denom = sqrt(sums[1] * sums[2])
    return Agreement(
        sums[0] / denom if denom > 0 else 0.0,
        sums[4] / sums[3] if sums[3] > 0 else Float64.MAX,
        Int(sums[5]),
    )


def check_agreement(
    ctx: DeviceContext,
    reference: DeviceBuffer[DType.bfloat16, ...],
    candidate: DeviceBuffer[DType.bfloat16, ...],
    count: Int,
    name: String,
    raise_on_mismatch: Bool = True,
    min_cosine: Float64 = 0.99,
    max_error_fraction: Float64 = 0.05,
) raises -> Bool:
    """Prints a `CHECK` line for `candidate` against `reference`, and raises
    unless they agree or `raise_on_mismatch` is off.

    Returns whether they agree.
    """
    var a = agreement(ctx, reference, candidate, count)
    var ok = a.ok(min_cosine, max_error_fraction)
    print("CHECK ", "ok" if ok else "MISMATCH", " ", name, ": ", a, sep="")
    if not ok and raise_on_mismatch:
        raise Error(name, " disagrees with its reference: ", a)
    return ok


def verify_capture[
    LaunchFn: def(DeviceContext[]) raises -> None
](
    ctx: DeviceContext[],
    launch: LaunchFn,
    output: DeviceBuffer[DType.bfloat16, ...],
    count: Int,
    repeats: Int,
) raises:
    """Raises unless graph replays of `launch` reproduce an eager launch's
    first `count` outputs bit for bit, also after unrelated allocations.

    A launch that bakes a host value or a transient buffer into the graph
    passes eagerly and fails here.
    """
    launch(ctx)
    ctx.synchronize()
    var view = DeviceBuffer[DType.bfloat16](
        ctx, output.unsafe_ptr(), count, owning=False
    )
    var expected = List[UInt16](capacity=count)
    with view.map_to_host() as host:
        for i in range(count):
            expected.append(bitcast[DType.uint16, 1](host[i])[0])

    def build(
        mut builder: DeviceGraphBuilder,
    ) raises {imm launch, imm repeats}:
        with builder.recording_context() as rec:
            for _ in range(max(repeats, 1)):
                launch(rec)

    var graph = DeviceGraph.create(ctx, build)
    for _ in range(3):
        var pressure = ctx.enqueue_create_buffer[DType.bfloat16](count)
        pressure.enqueue_fill(7)
        view.enqueue_fill(-1)
        graph.replay()
        ctx.synchronize()
        with view.map_to_host() as host:
            for i in range(count):
                if bitcast[DType.uint16, 1](host[i])[0] != expected[i]:
                    raise Error("graph replay output differs at ", i)
        _ = pressure^
    print("CAPTURE_VERIFIED elements=", count, " replays=3", sep="")
