# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug-no-assert %s

from sys import env_get_string, env_get_int, env_get_bool, env_get_dtype
from internal_utils import (
    env_get_shape,
    int_list_to_tuple,
    arg_parse,
)
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
](mut m: Bench, verify: Bool) raises:
    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher):
        @parameter
        @always_inline
        fn call_fn():
            pass

        b.iter[call_fn]()

    var name = String(
        "gemm/dtype=", dtype, "/m=", M, "/n=", N, "/k=", N, "/stages=", stages
    )
    m.bench_function[bench_iter](BenchId(name))
    if verify:
        print("verifying dummy results...PASS")


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.float16]()
    alias shape_int_list = env_get_shape["shape", "1024x1024x1024"]()
    alias shape = int_list_to_tuple[shape_int_list]()
    alias stages = env_get_int["stages", 0]()
    alias verify = env_get_bool["verify", 0]()

    var runtime_x = arg_parse("x", 0)

    var m = Bench(
        BenchConfig(max_iters=1, max_batch_size=1, min_warmuptime_secs=0)
    )

    bench_func[dtype, shape[0], shape[1], shape[2], stages](m, verify)

    m.dump_report()
