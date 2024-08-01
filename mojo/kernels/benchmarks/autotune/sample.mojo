# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo build %s

from sys import env_get_string, env_get_int
from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from time import sleep


fn bench_func[dtype: DType, M: Int, N: Int, stage: Int](inout m: Bench) raises:
    @parameter
    @always_inline
    fn bench_iter(inout b: Bencher):
        @parameter
        @always_inline
        fn call_fn():
            sleep((M + N + stage) * 0.0001)
            pass

        b.iter[call_fn]()

    alias size = 32
    alias num_elements = size

    var measures = List[ThroughputMeasure](
        ThroughputMeasure(BenchMetric.flops, num_elements * 2),  # FMA's
        ThroughputMeasure(
            BenchMetric.bytes, num_elements * 4
        ),  # uint32 = 4 bytes
        ThroughputMeasure(BenchMetric.elements, num_elements),
    )

    var name = "gemm/m=" + str(M) + "/n=" + str(N) + "/stage=" + str(stage)
    m.bench_function[bench_iter](BenchId(name), measures=measures)


fn main() raises:
    alias dtype_str = env_get_string["DTYPE", "DType.float16"]()
    alias M = env_get_int["M", 0]()
    alias N = env_get_int["N", 0]()
    alias stages = env_get_int["STAGES", 0]()
    alias dtype = DType.float16

    var m = Bench()

    bench_func[dtype, M, N, stages](m)

    m.dump_report()
