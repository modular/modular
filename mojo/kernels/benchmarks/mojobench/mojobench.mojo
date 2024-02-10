# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from collections import Optional
from stdlib.builtin.file import FileHandle


@value
struct MojoBenchConfig:
    var out_file: Optional[Path]
    var min_runtime: Int

    fn __init__(
        inout self, out_file: Optional[Path] = None, min_runtime: Int = 10_000
    ):
        self.out_file = out_file  # TODO: check for env var as well
        self.min_runtime = min_runtime


@value
struct BenchId:
    var func_name: String
    var input_id: Optional[String]

    fn __init__(inout self, func_name: String, input_id: String):
        self.func_name = func_name
        self.input_id = input_id

    fn __init__(inout self, func_name: String):
        self.func_name = func_name
        self.input_id = None


@value
struct MojoBench:
    var config: MojoBenchConfig
    var names: DynamicVector[String]
    var mets: DynamicVector[Float64]

    fn __init__(inout self, config: MojoBenchConfig = MojoBenchConfig()):
        self.config = config
        self.names = DynamicVector[String](0)
        self.mets = DynamicVector[Float64](0)

    fn bench_with_input[
        T: AnyType,
        bench_fn: fn (inout Bencher, T) capturing -> None,
    ](inout self, bench_id: BenchId, input: T) raises:
        @parameter
        fn input_closure(inout b: Bencher):
            bench_fn(b, input)

        self.bench_function[input_closure](bench_id)

    fn bench_function[
        bench_fn: fn (inout Bencher) capturing -> None
    ](inout self, bench_id: BenchId) raises:
        let start = time.now()
        var iters_vec = DynamicVector[Int]()
        var times_vec = DynamicVector[Int]()
        var elapsed = 0
        var total_iters = 0
        while elapsed < self.config.min_runtime:
            # TODO: copy logic from Benchmark to boostrap the num_iters
            var b = Bencher(num_iters=100)
            bench_fn(b)
            iters_vec.push_back(b.num_iters)
            times_vec.push_back(b.elapsed)
            elapsed += b.elapsed
            total_iters += b.num_iters

        let full_name = bench_id.func_name + "/" + bench_id.input_id.value() if bench_id.input_id else bench_id.func_name
        self.names.push_back(full_name)
        self.mets.push_back(Float64(elapsed) / total_iters)

    fn dump_report(self) raises:
        var report = String("name, mean execution time (ns)\n")
        for i in range(len(self.mets)):
            let sep = "\n" if i < len(self.mets) - 1 else ""
            report += self.names[i] + "," + self.mets[i] + sep
        print("------------------------------------------")
        print("Benchmark results")
        print("------------------------------------------")
        print(report)

        if self.config.out_file:
            with open(self.config.out_file.value(), "w") as f:
                f.write(report)


@value
@register_passable
struct Bencher:
    var num_iters: Int
    var elapsed: Int

    fn __init__(num_iters: Int) -> Self:
        return Self {num_iters: num_iters, elapsed: 0}

    fn iter[iter_fn: fn () capturing -> None](inout self):
        let start = time.now()
        for _ in range(self.num_iters):
            iter_fn()
        let stop = time.now()
        self.elapsed = stop - start
