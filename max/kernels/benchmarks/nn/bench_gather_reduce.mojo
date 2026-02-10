# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from random import random_si64
from sys import simd_width_of, size_of

from benchmark import Bench, Bencher, BenchId
from layout._layout import row_major
from layout._coord import Coord, Idx
from layout._tile_tensor import TileTensor
from nn.gather_scatter import gather_reduce

from utils import IndexList


@always_inline
fn add(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


@parameter
fn bench_gather_reduce(mut b: Bencher):
    comptime type = DType.float32
    var num_rows = 500000
    var embedding_dim = 32
    var multi_hot_dim = 100
    comptime l3_size = 32  # mb
    comptime clear_size = l3_size * 2 * 1_000_000
    var num_indices = clear_size // (
        size_of[type]() * embedding_dim * multi_hot_dim
    )
    var input_shape = IndexList[2](num_rows, embedding_dim)
    var output_shape = IndexList[2](num_indices, embedding_dim)
    var indices_shape = IndexList[2](num_indices, multi_hot_dim)
    var input_storage = UnsafePointer[Scalar[type]].alloc(
        input_shape.flattened_length()
    )
    var output_storage = UnsafePointer[Scalar[type]].alloc(
        output_shape.flattened_length()
    )
    var indices_storage = UnsafePointer[Int32].alloc(
        indices_shape.flattened_length()
    )
    var input = TileTensor(input_storage, row_major(Coord(input_shape))).fill(1)
    var output = TileTensor(
        output_storage, row_major(Coord(output_shape))
    ).fill(0)
    var indices = TileTensor(indices_storage, row_major(Coord(indices_shape)))
    for i in range(Int(indices.dim[0]())):
        for j in range(Int(indices.dim[1]())):
            indices[i, j] = random_si64(0, num_rows).cast[DType.int32]()

    @parameter
    fn to_bench():
        gather_reduce[type, 0, 1, simd_width_of[type](), add](
            output,
            input,
            indices,
            0,
        )

    b.iter[to_bench]()

    print(output[0, 0])
    input.ptr.free()
    output.ptr.free()
    indices.ptr.free()


def main():
    var m = Bench()
    m.bench_function[bench_gather_reduce](
        BenchId("gather_reduce_dlrm1_multihot")
    )
    m.dump_report()
