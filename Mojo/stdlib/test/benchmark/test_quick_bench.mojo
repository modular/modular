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

import std.math
from std.memory import Alignment, alloc, Layout
from std.random import randint
from std.time import sleep

from std.benchmark import BenchId, BenchMetric, QuickBench, ThroughputMeasure
from std.testing import TestSuite


def vec_reduce[
    N: Int, dtype: DType
](x: Pointer[Scalar[dtype], ImmutAnyOrigin]) -> Scalar[dtype]:
    var total: Scalar[dtype] = 0
    for i in range(N):
        total += x[unsafe_offset=i]
    return total


def vec_add[
    N: Int, dtype: DType
](
    x: Pointer[Scalar[dtype], MutAnyOrigin],
    y: Pointer[Scalar[dtype], ImmutAnyOrigin],
) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    for i in range(N):
        x[unsafe_offset=i] += y[unsafe_offset=i]
    return x


def dummy() -> None:
    sleep(0.5)


def dummy(x0: Int, x1: Int) -> Float32:
    return Float32(x0 + x1)


def test_captures() raises:
    var qb = QuickBench()

    qb.run(
        lambda () -> None: dummy(),
        bench_id=BenchId("dummy_none"),
        measures=[ThroughputMeasure(BenchMetric.flops, 1)],
    )

    var x0 = 1
    var x1 = 2
    qb.run(
        lambda () {x0, x1} -> Float32: dummy(x0, x1),
        bench_id=BenchId("dummy_2"),
        measures=[ThroughputMeasure(BenchMetric.flops, 1)],
    )

    qb.dump_report()


@inline(.always)
def exp(x: SIMD[.float32, 4]) -> type_of(x):
    return std.math.exp(x)


@inline(.always)
def tanh(x: SIMD[.float32, 4]) -> type_of(x):
    return std.math.tanh(x)


def test_mojo_math() raises:
    var qb = QuickBench()

    var x: SIMD[.float32, 4] = 1.0
    qb.run(
        lambda () {x} -> SIMD[.float32, 4]: exp(x),
        bench_id=BenchId("exp"),
        measures=[ThroughputMeasure(BenchMetric.bytes, 4)],  # 4 bytes per call
    )

    qb.run(
        lambda () {x} -> SIMD[.float32, 4]: tanh(x),
        bench_id=BenchId("tanh"),
        measures=[ThroughputMeasure(BenchMetric.bytes, 4)],  # 4 bytes per call
    )
    qb.dump_report()


def test_custom() raises:
    comptime N = 1024
    comptime alignment = Alignment.of_bytes[64]()
    comptime dtype = DType.int32
    var xy_layout = Layout[Scalar[dtype], alignment=alignment](length=N)
    var x_alloc = alloc(xy_layout).into_managed()
    var y_alloc = alloc(xy_layout).into_managed()
    var x = x_alloc.unsafe_ptr()
    var y = y_alloc.unsafe_ptr()
    randint[dtype](x, N, 0, 255)
    randint[dtype](y, N, 0, 255)

    var qb = QuickBench()

    qb.run(
        lambda () {x} -> Scalar[dtype]: vec_reduce[N, dtype](
            x.as_unsafe_any_origin()
        ),
        bench_id=BenchId("vec_reduce"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, N)  # N additions per call
        ],
    )

    qb.run(
        lambda () {x, y} -> Pointer[Scalar[dtype], MutAnyOrigin]: vec_add[
            N, dtype
        ](x.as_unsafe_any_origin(), y.as_unsafe_any_origin()),
        bench_id=BenchId("vec_add"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, N)  # N additions per call
        ],
    )

    qb.dump_report()


def test_all() raises:
    # Width of columns is dynamic based on the longest value as a string, so
    # only test the first column.

    # CHECK: name,
    # CHECK: exp ,
    # CHECK: tanh,
    test_mojo_math()

    # CHECK: name      ,
    # CHECK: vec_reduce,
    # CHECK: vec_add   ,
    test_custom()

    # CHECK: name      ,
    # CHECK: dummy_none,
    # CHECK: dummy_2   ,
    test_captures()


def main() raises:
    # NOTE: we pass an empty list since the benchmark infra also tries to parse
    # the arguments for its own purposes.
    TestSuite.discover_tests[__functions_in_module()](
        cli_args=List[StaticString]()
    ).run()
