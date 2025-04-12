# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s -r 2 -o %t.csv | FileCheck %s
# RUN: cat %t.csv | FileCheck %s --check-prefix=CHECK-OUT
# RUN: mojo %s -t | FileCheck %s --check-prefix=CHECK-TEST

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    Format,
    Mode,
    ThroughputMeasure,
)


@parameter
fn bench1(mut b: Bencher):
    @parameter
    fn to_bench():
        print("hello")

    b.iter[to_bench]()


@parameter
fn bench2(mut b: Bencher, mystr: String) raises:
    @parameter
    fn to_bench():
        print(mystr)

    b.iter[to_bench]()


def main():
    var m = Bench(BenchConfig(max_iters=10_000))
    m.bench_function[bench1](
        BenchId("bench1"),
        ThroughputMeasure(BenchMetric.elements, 0),
        ThroughputMeasure(BenchMetric.flops, 0),
    )

    var inputs = List[String]()
    inputs.append("input1")
    inputs.append("input2")
    for i in range(len(inputs)):
        m.bench_with_input[String, bench2](
            BenchId("bench2", String(i)),
            inputs[i],
            ThroughputMeasure(BenchMetric.elements, len(inputs[i])),
            ThroughputMeasure(BenchMetric.flops, len(inputs[i])),
        )

    m.config.verbose_timing = True

    # Check default print format
    # CHECK: -----------------------
    # CHECK: | name     | met (ms)
    # CHECK: -----------------------
    # CHECK: | bench1   |
    # CHECK: | bench2/0 |
    # CHECK: | bench2/1 |
    # CHECK: -----------------------
    print(m)

    # CHECK: name,met (ms),iters,throughput (GElems/s),Arithmetic (GFLOPS/s),min (ms),mean (ms),max (ms),duration (ms)
    # CHECK: "bench1",
    # CHECK: "bench2/0",
    # CHECK: "bench2/1",
    m.config.format = Format.csv
    print(m)

    # CHECK: bench1
    # CHECK-NEXT: bench1
    # CHECK-NEXT: bench2/0
    # CHECK-NEXT: bench2/0
    # CHECK-NEXT: bench2/1
    # CHECK-NEXT: bench2/1
    # CHECK-OUT: bench1
    m.config.format = Format.tabular
    m.dump_report()

    # CHECK-TEST-COUNT-1: hello
