# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build %s

from sys import env_get_string, env_get_int
from benchmark import (
    BenchConfig,
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
            pass

        b.iter[call_fn]()

    var name = "gemm/m=" + str(M) + "/n=" + str(N) + "/stage=" + str(stage)
    m.bench_function[bench_iter](BenchId(name))


fn main() raises:
    alias dtype_str = env_get_string["DTYPE", "DType.float16"]()
    alias M = env_get_int["M", 0]()
    alias N = env_get_int["N", 0]()
    alias stages = env_get_int["STAGES", 0]()
    alias dtype = DType.float16

    var m = Bench(BenchConfig(max_iters=1, max_batch_size=1, warmup_iters=0))

    bench_func[dtype, M, N, stages](m)

    m.dump_report()
