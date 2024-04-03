# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s -t | FileCheck %s
# CHECK: Benchmark results

from math import add
from random import random_si64

from buffer import NDBuffer
from benchmark import Bencher, BenchId, Bench
from tensor import Tensor
from nn.gather_scatter import gather, gather_reduce
from runtime.llcl import Runtime

from utils.index import Index


@parameter
fn bench_gather_reduce(inout b: Bencher):
    alias type = DType.float32
    var num_rows = 500000
    var embedding_dim = 32
    var multi_hot_dim = 100
    alias l3_size = 32  # mb
    alias clear_size = l3_size * 2 * 1_000_000
    var num_indices = clear_size // (
        sizeof[type]() * embedding_dim * multi_hot_dim
    )
    var input = Tensor[type](num_rows, embedding_dim)
    var output = Tensor[type](num_indices, embedding_dim)
    var indices = Tensor[DType.int32](num_indices, multi_hot_dim)
    input._to_ndbuffer[2]().fill(1)
    output._to_ndbuffer[2]().zero()
    for i in range(indices.shape()[0]):
        for j in range(indices.shape()[1]):
            indices[i][j] = random_si64(0, num_rows).cast[DType.int32]()

    @parameter
    fn to_bench():
        with Runtime(threads=1) as rt:
            gather_reduce[type, 0, 1, simdwidthof[type](), add,](
                output._to_ndbuffer[2](),
                input._to_ndbuffer[2](),
                indices._to_ndbuffer[2](),
                0,
            )

    b.iter[to_bench]()

    print(output[0, 0])
    _ = output
    _ = indices
    _ = input


def main():
    var m = Bench()
    m.bench_function[bench_gather_reduce](
        BenchId("gather_reduce_dlrm1_multihot")
    )
    m.dump_report()
