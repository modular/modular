# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: asan

# RUN: %mojo-no-debug-no-assert %s | FileCheck %s
from sys.info import num_physical_cores

from algorithm import parallelize, sync_parallelize
from benchmark import Bench, Bencher, BenchId, keep, BenchConfig


@parameter
fn bench_empty_sync_parallelize(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn parallel_fn(thread_id: Int):
        keep(thread_id)

    sync_parallelize[parallel_fn](num_physical_cores())


@parameter
fn bench_empty_parallelize(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn parallel_fn(thread_id: Int):
        keep(thread_id)

    parallelize[parallel_fn](num_physical_cores())


# CHECK: sync_parallelize
# CHECK: parallelize
def main():
    var m = Bench()
    m.bench_function[bench_empty_sync_parallelize](BenchId("sync_parallelize"))
    m.bench_function[bench_empty_parallelize](BenchId("parallelize"))
    m.dump_report()
