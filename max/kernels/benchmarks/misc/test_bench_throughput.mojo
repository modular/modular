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

from random import randint, seed

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)


fn test[N: Int = 1024 * 1024]() -> UInt32:
    # seed(0)
    alias alignment = 64
    alias type = DType.uint32
    var x = UnsafePointer[Scalar[type]].alloc(N, alignment=alignment)
    randint[type](x, N, 0, 255)
    var s: UInt32 = 0
    for i in range(N):
        s += 123 * x[i].cast[DType.uint32]()
    return s


fn bench_func[
    func: fn[size: Int] () -> UInt32, size: Int
](mut m: Bench, op_name: String) raises:
    alias num_elements = size * 1024 * 1024

    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher):
        @parameter
        @always_inline
        fn call_fn():
            var x = func[num_elements]()
            keep(x)

        b.iter[call_fn]()

    alias apple_metric = BenchMetric(10, "apple", "Apples/s")
    alias orange_metric = BenchMetric(11, "orange", "Oranges/s")

    var measures = List[ThroughputMeasure](
        ThroughputMeasure(BenchMetric.flops, num_elements * 2),  # FMA's
        ThroughputMeasure("DataMovement", num_elements * 4),  # uint32 = 4 bytes
        ThroughputMeasure(BenchMetric.elements, num_elements),
        # custom metrics
        ThroughputMeasure(apple_metric, num_elements),
        ThroughputMeasure(orange_metric, num_elements * 2),
    )
    # NOTE: only one set of metrics can be used in one bench object (ie, one bench report).
    m.bench_function[bench_iter](BenchId(op_name), measures=measures)

    """
    Note: We can also use the following variadic signature:
    m.bench_function[bench_iter](BenchId(op_name),
        ThroughputMeasure(BenchMetric.flops, num_elements * 2),  # FMA's
        ThroughputMeasure("bytes", num_elements * 4),  # uint32 = 4 bytes
        ThroughputMeasure(BenchMetric.elements, num_elements),
        # custom metrics
        ThroughputMeasure(apple_metric, num_elements),
        ThroughputMeasure(orange_metric, num_elements * 2),
    )
    """


def main():
    var m = Bench()
    bench_func[test, 8](m, "test8")
    bench_func[test, 16](m, "test16")
    bench_func[test, 32](m, "test32")

    m.dump_report()
