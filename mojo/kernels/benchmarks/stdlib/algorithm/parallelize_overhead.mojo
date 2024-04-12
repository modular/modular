# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK: Benchmark results

from algorithm import (
    sync_parallelize,
    parallelize,
)
from sys.info import num_physical_cores
from benchmark import Bencher, BenchId, Bench


@parameter
fn bench_empty_sync_parallelize(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn parallel_fn(thread_id: Int):
        pass

    sync_parallelize[parallel_fn](num_physical_cores())


@parameter
fn bench_empty_parallelize(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn parallel_fn(thread_id: Int):
        pass

    parallelize[parallel_fn](num_physical_cores())


def main():
    var m = Bench()
    m.bench_function[bench_empty_sync_parallelize](BenchId("sync_parallelize"))
    m.bench_function[bench_empty_parallelize](BenchId("parallelize"))
    m.dump_report()
