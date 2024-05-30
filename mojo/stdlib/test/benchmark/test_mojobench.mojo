# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: mojo %s -r 2 -o %t.csv | FileCheck %s
# RUN: cat %t.csv | FileCheck %s --check-prefix=CHECK-OUT
# RUN: mojo %s -t | FileCheck %s --check-prefix=CHECK-TEST

from benchmark import Bench, BenchConfig, Bencher, BenchId, Mode


@parameter
fn bench1(inout b: Bencher):
    @parameter
    fn to_bench():
        print("hello")

    b.iter[to_bench]()


@parameter
fn bench2(inout b: Bencher, mystr: String):
    @parameter
    fn to_bench():
        print(mystr)

    b.iter[to_bench]()


def main():
    var m = Bench(BenchConfig(max_iters=10_000))
    m.bench_function[bench1](BenchId("bench1"))

    var inputs = List[String]()
    inputs.append("input1")
    inputs.append("input2")
    for i in range(len(inputs)):
        m.bench_with_input[String, bench2](
            BenchId("bench2", str(i)),
            inputs[i],
            throughput_elems=len(inputs[i]),
        )

    # CHECK: Benchmark results
    # CHECK: bench1
    # CHECK-NEXT: bench1
    # CHECK-NEXT: bench2/0
    # CHECK-NEXT: bench2/0
    # CHECK-NEXT: bench2/1
    # CHECK-NEXT: bench2/1
    # CHECK-OUT: bench1
    m.dump_report()

    # CHECK-TEST-COUNT-1: hello
