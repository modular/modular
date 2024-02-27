# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s -r 2 -o %t.csv | FileCheck %s
# RUN: cat %t.csv | FileCheck %s --check-prefix=CHECK-OUT
# RUN: mojo %s -t | FileCheck %s --check-prefix=CHECK-TEST

from mojobench import Bencher, MojoBench, BenchId, MojoBenchConfig, Mode


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
    var m = MojoBench(MojoBenchConfig(max_iters=10_000))
    m.bench_function[bench1](BenchId("bench1"))

    var inputs = DynamicVector[String]()
    inputs.push_back("input1")
    inputs.push_back("input2")
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
