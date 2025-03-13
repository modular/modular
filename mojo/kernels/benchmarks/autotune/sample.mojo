# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug-no-assert %s

from sys import env_get_bool, env_get_dtype, env_get_int, env_get_string
from time import sleep

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from internal_utils import arg_parse, env_get_shape, int_list_to_tuple, Mode


fn bench_func[
    dtype: DType, M: Int, N: Int, K: Int, stages: Int
](mut m: Bench, mode: Mode) raises:
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

    if mode == Mode.BENCHMARK:
        m.bench_function[bench_iter](BenchId(name))
    if mode == Mode.VERIFY:
        print("verifying dummy results...PASS")
    if mode == Mode.RUN:
        print("pretending to run the kernel...PASS")


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.float16]()
    alias shape_int_list = env_get_shape["shape", "1024x1024x1024"]()
    alias shape = int_list_to_tuple[shape_int_list]()
    alias stages = env_get_int["stages", 0]()

    var runtime_x = arg_parse("x", 0)

    # define benchmark mode: [run, benchmark, verify] or a combo (run+benchmark)
    var mode = Mode(arg_parse("mode", "benchmark"))

    print("mode=" + String(mode))
    if mode == Mode.RUN:
        print("-- mode: run kernel once")
    if mode == Mode.BENCHMARK:
        print("-- mode: run kernel benchmark")
    if mode == Mode.VERIFY:
        print("-- mode: verify kernel")

    var m = Bench(
        BenchConfig(max_iters=1, max_batch_size=1, min_warmuptime_secs=0)
    )

    bench_func[dtype, shape[0], shape[1], shape[2], stages](m, mode)

    m.dump_report()
