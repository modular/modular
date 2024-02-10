# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from mojobench import Bencher, MojoBench, BenchId, MojoBenchConfig


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
    var m = MojoBench(MojoBenchConfig(out_file=Path("./tmp_bench_results")))
    m.bench_function[bench1](BenchId("bench1"))

    var inputs = DynamicVector[String]()
    inputs.push_back("input1")
    inputs.push_back("input2")
    for i in range(len(inputs)):
        m.bench_with_input[String, bench2](BenchId("bench2", str(i)), inputs[i])

    m.dump_report()
