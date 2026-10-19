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
"""Benchmarks the public fused residual RMSNorm launcher with dropout disabled."""

from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.math import sqrt
from std.testing import assert_almost_equal, assert_equal, assert_true

from internal_utils import arg_parse
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.rms_norm_fused_residual import rms_norm_fused_residual_gpu

comptime WARMUP = 5
comptime SAMPLES = 30


def _median(times: List[Float64]) -> Float64:
    var ordered = times.copy()
    sort(Span(ordered))
    var middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def _bench[dtype: DType](ctx: DeviceContext, rows: Int, cols: Int) raises:
    var elements = rows * cols
    var input_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var residual_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var output_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var residual_output_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    for i in range(elements):
        input_h[i] = Scalar[dtype](Float32(i % 37 - 18) * 0.015625)
        residual_h[i] = Scalar[dtype](0.5 + Float32(i % 19) * 0.0078125)
        output_h[i] = 0
        residual_output_h[i] = 0
    for i in range(cols):
        gamma_h[i] = Scalar[dtype](1.0 + Float32(i % 13) * 0.015625)

    var input_d = ctx.enqueue_create_buffer[dtype](elements)
    var residual_d = ctx.enqueue_create_buffer[dtype](elements)
    var output_d = ctx.enqueue_create_buffer[dtype](elements)
    var residual_output_d = ctx.enqueue_create_buffer[dtype](elements)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(residual_d, residual_h)
    ctx.enqueue_copy(output_d, output_h)
    ctx.enqueue_copy(residual_output_d, residual_output_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    var shape = Coord(rows, cols)
    var input_tt = TileTensor(input_d, row_major(shape))
    var residual_tt = TileTensor(residual_d, row_major(shape))
    var output_tt = TileTensor(output_d, row_major(shape))
    var residual_output_tt = TileTensor(residual_output_d, row_major(shape))
    var gamma_tt = TileTensor(gamma_d, row_major(cols))

    @inline(.always)
    def input_fn[
        width: Int
    ](coords: Coord) {var input_tt} -> SIMD[dtype, width]:
        return input_tt.load[width=width](coords)

    @inline(.always)
    def residual_fn[
        width: Int
    ](coords: Coord) {var residual_tt} -> SIMD[dtype, width]:
        return residual_tt.load[width=width](coords)

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var output_tt}:
        output_tt.store[width=width, alignment=alignment](coords, value)

    @inline(.always)
    def residual_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var residual_output_tt}:
        residual_output_tt.store[width=width, alignment=alignment](
            coords, value
        )

    def launch(
        lctx: DeviceContext,
    ) raises {
        imm input_fn,
        imm residual_fn,
        imm output_fn,
        imm residual_output_fn,
        imm shape,
        imm gamma_tt,
    }:
        rms_norm_fused_residual_gpu[multiply_before_cast=True](
            input_fn,
            residual_fn,
            residual_output_fn,
            output_fn,
            shape,
            gamma_tt,
            Float32(1e-5),
            Scalar[dtype](0),
            lctx,
        )

    ctx.synchronize()
    for _ in range(WARMUP):
        launch(ctx)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(SAMPLES):
        times.append(Float64(ctx.execution_time(launch, 1)))

    ctx.enqueue_copy(output_h, output_d)
    ctx.enqueue_copy(residual_output_h, residual_output_d)
    ctx.synchronize()
    comptime tolerance = 0.02 if dtype == DType.bfloat16 else 1e-4
    for row in range(rows):
        var sum_squares = Float64(0)
        for col in range(cols):
            var index = row * cols + col
            var added = input_h[index] + residual_h[index]
            assert_equal(residual_output_h[index], added)
            sum_squares += Float64(added) * Float64(added)
        var rms = sqrt(sum_squares / Float64(cols) + 1e-5)
        for col in range(cols):
            var index = row * cols + col
            var expected = Scalar[dtype](
                Float64(input_h[index] + residual_h[index])
                / rms
                * Float64(gamma_h[col])
            )
            assert_almost_equal(output_h[index], expected, rtol=tolerance)

    var description = String(
        t"kernel=rms_norm_fused_residual dtype={dtype} rows={rows} "
    ) + String(t"cols={cols} dropout=0 multiply_before_cast=true")
    for sample in range(SAMPLES):
        print(t"sample {description} index={sample} ns={times[sample]}")
    print(t"summary {description} samples={SAMPLES} median_ns={_median(times)}")
    var benchmark = Bench()
    var report = Report()
    for elapsed in times:
        report.runs.append(Batch(Int(elapsed), 1, True))
    benchmark.info_vec.append(BenchmarkInfo(description, report^))
    benchmark.dump_report()
    _ = input_d^
    _ = residual_d^
    _ = output_d^
    _ = residual_output_d^
    _ = gamma_d^


def main() raises:
    var dtype = arg_parse("dtype", String(""))
    if dtype != "bfloat16" and dtype != "float32":
        raise Error("dtype must be bfloat16 or float32")
    var rows = arg_parse("rows", 0)
    var cols = arg_parse("cols", 0)
    assert_true(rows > 0 and cols > 0)
    with DeviceContext() as ctx:
        assert_true(ctx.is_compatible(), "A compatible GPU is required")
        print(
            t"config warmup={WARMUP} samples={SAMPLES} "
            t"launches_per_sample=1 l2_flush=false correctness=host_reference"
            t" csv_statistic=mean raw_statistic=median"
        )
        if dtype == "bfloat16":
            _bench[.bfloat16](ctx, rows, cols)
        else:
            _bench[.float32](ctx, rows, cols)
