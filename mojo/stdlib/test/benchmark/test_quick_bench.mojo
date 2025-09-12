# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import math
from random import randint
from time import sleep

from benchmark import BenchId, BenchMetric, QuickBench, ThroughputMeasure


fn vec_reduce[
    N: Int, dtype: DType
](x: UnsafePointer[Scalar[dtype]]) -> Scalar[dtype]:
    var total: Scalar[dtype] = 0
    for i in range(N):
        total += x[i]
    return total


fn vec_add[
    N: Int, dtype: DType
](
    x: UnsafePointer[Scalar[dtype]], y: UnsafePointer[Scalar[dtype]]
) -> UnsafePointer[Scalar[dtype]]:
    for i in range(N):
        x[i] += y[i]
    return x


fn dummy() -> None:
    sleep(0.5)


fn dummy(x0: Int) -> Float32:
    return x0


fn dummy(x0: Int, x1: Int) -> Float32:
    return x0 + x1


fn dummy(x0: Int, x1: Int, x2: Int) -> Float32:
    return x0 + x1 + x2


fn dummy(x0: Int, x1: Int, x2: Int, x3: Int) -> Float32:
    return x0 + x1 + x2 + x3


fn dummy(x0: Int, x1: Int, x2: Int, x3: Int, x4: Int) -> Float32:
    return x0 + x1 + x2 + x3 + x4


fn dummy(x0: Int, x1: Int, x2: Int, x3: Int, x4: Int, x5: Int) -> Float32:
    return x0 + x1 + x2 + x3 + x4 + x5


fn dummy(
    x0: Int, x1: Int, x2: Int, x3: Int, x4: Int, x5: Int, x6: Int
) -> Float32:
    return x0 + x1 + x2 + x3 + x4 + x5 + x6


fn dummy(
    x0: Int, x1: Int, x2: Int, x3: Int, x4: Int, x5: Int, x6: Int, x7: Int
) -> Float32:
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7


fn dummy(
    x0: Int,
    x1: Int,
    x2: Int,
    x3: Int,
    x4: Int,
    x5: Int,
    x6: Int,
    x7: Int,
    x8: Int,
) -> Float32:
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8


fn dummy(
    x0: Int,
    x1: Int,
    x2: Int,
    x3: Int,
    x4: Int,
    x5: Int,
    x6: Int,
    x7: Int,
    x8: Int,
    x9: Int,
) -> Float32:
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9


fn test_overloaded() raises:
    var qb = QuickBench()

    qb.run[T_out = NoneType._mlir_type](
        dummy,
        bench_id=BenchId("dummy_none"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )
    qb.run[Int, T_out=Float32](
        dummy,
        1,
        bench_id=BenchId("dummy_1"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        bench_id=BenchId("dummy_2"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        bench_id=BenchId("dummy_3"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        bench_id=BenchId("dummy_4"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        bench_id=BenchId("dummy_5"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        bench_id=BenchId("dummy_6"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        bench_id=BenchId("dummy_7"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        bench_id=BenchId("dummy_8"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        bench_id=BenchId("dummy_9"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        bench_id=BenchId("dummy_10"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ),
    )

    qb.dump_report()


@always_inline
fn exp(x: SIMD[DType.float32, 4]) -> __type_of(x):
    return math.exp(x)


@always_inline
fn tanh(x: SIMD[DType.float32, 4]) -> __type_of(x):
    return math.tanh(x)


fn test_mojo_math() raises:
    var qb = QuickBench()

    qb.run(
        exp,
        1.0,
        bench_id=BenchId("exp"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.bytes, 4)  # 4 bytes per call
        ),
    )

    qb.run(
        tanh,
        1.0,
        bench_id=BenchId("tanh"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.bytes, 4)  # 4 bytes per call
        ),
    )
    qb.dump_report()


fn test_custom() raises:
    alias N = 1024
    alias alignment = 64
    alias dtype = DType.int32
    var x = UnsafePointer[Scalar[dtype],].alloc[alignment=alignment](N)
    var y = UnsafePointer[Scalar[dtype],].alloc[alignment=alignment](N)
    randint[dtype](x, N, 0, 255)
    randint[dtype](y, N, 0, 255)

    var qb = QuickBench()

    qb.run(
        vec_reduce[N, dtype],
        x,
        bench_id=BenchId("vec_reduce"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, N)  # N additions per call
        ),
    )

    qb.run(
        vec_add[N, dtype],
        x,
        y,
        bench_id=BenchId("vec_add"),
        measures=List[ThroughputMeasure](
            ThroughputMeasure(BenchMetric.flops, N)  # N additions per call
        ),
    )

    qb.dump_report()
    x.free()
    y.free()


fn main() raises:
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
    # CHECK: dummy_1   ,
    # CHECK: dummy_2   ,
    # CHECK: dummy_3   ,
    # CHECK: dummy_4   ,
    # CHECK: dummy_5   ,
    # CHECK: dummy_6   ,
    # CHECK: dummy_7   ,
    # CHECK: dummy_8   ,
    # CHECK: dummy_9   ,
    # CHECK: dummy_10  ,
    test_overloaded()
