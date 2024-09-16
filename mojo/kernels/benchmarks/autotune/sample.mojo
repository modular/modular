# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s

from sys import env_get_string, env_get_int
from internal_utils import env_get_dtype, env_get_shape, int_list_to_tuple
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


fn bench_func[
    dtype: DType, M: Int, N: Int, K: Int, stages: Int
](inout m: Bench) raises:
    @parameter
    @always_inline
    fn bench_iter(inout b: Bencher):
        @parameter
        @always_inline
        fn call_fn():
            pass

        b.iter[call_fn]()

    var name = "gemm/dtype=" + str(dtype) + "/m=" + str(M) + "/n=" + str(
        N
    ) + "/k=" + str(N) + "/stages=" + str(stages)
    m.bench_function[bench_iter](BenchId(name))


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.float16]()
    alias shape_int_list = env_get_shape["shape", "1024x1024x1024"]()
    alias shape = int_list_to_tuple[shape_int_list]()
    alias stages = env_get_int["stages", 0]()

    var m = Bench(
        BenchConfig(max_iters=1, max_batch_size=1, min_warmuptime_secs=0)
    )

    bench_func[dtype, shape[0], shape[1], shape[2], stages](m)

    m.dump_report()
