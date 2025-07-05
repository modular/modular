# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from random import rand, randint

from benchmark import *
from buffer.dimlist import Dim
from nn.gather_scatter import scatter_elements
from tensor_internal import DynamicTensor

from utils.index import Index


fn bench_scatter(mut m: Bench, spec: ScatterSpec) raises:
    @parameter
    @always_inline
    fn bench_scatter_wrapper(mut b: Bencher, concrete_spec: ScatterSpec) raises:
        bench_scatter(b, concrete_spec)

    m.bench_with_input[ScatterSpec, bench_scatter_wrapper](
        BenchId("scatter", String(spec)), spec
    )


@parameter
fn bench_scatter(mut bencher: Bencher, spec: ScatterSpec):
    var index_rand_min = 0
    var index_rand_max = spec.m1 - 1

    var input_shape = Index(spec.m1, spec.m2)
    var indices_shape = Index(spec.n1, spec.n2)

    var data_ptr = UnsafePointer[Float32].alloc(input_shape.flattened_length())
    rand(data_ptr, input_shape.flattened_length())
    var data_tensor = DynamicTensor[DType.float32, 2](data_ptr, input_shape)

    var indices_ptr = UnsafePointer[Int32].alloc(
        indices_shape.flattened_length()
    )
    randint(
        indices_ptr,
        indices_shape.flattened_length(),
        index_rand_min,
        index_rand_max,
    )
    var indices_tensor = DynamicTensor[DType.int32, 2](
        indices_ptr, indices_shape
    )

    var updates_ptr = UnsafePointer[Float32].alloc(
        indices_shape.flattened_length()
    )
    rand(updates_ptr, indices_shape.flattened_length())
    var updates_tensor = DynamicTensor[DType.float32, 2](
        updates_ptr, indices_shape
    )

    var output_ptr = UnsafePointer[Float32].alloc(
        input_shape.flattened_length()
    )
    var output_tensor = DynamicTensor[DType.float32, 2](output_ptr, input_shape)

    @always_inline
    @parameter
    fn bench_fn():
        @always_inline
        @parameter
        fn reduce_fn[
            _dtype: DType, width: Int
        ](
            input_val: SIMD[_dtype, width], update_val: SIMD[_dtype, width]
        ) -> SIMD[_dtype, width]:
            return input_val + update_val

        try:
            scatter_elements[reduce_fn](
                data_tensor,
                indices_tensor,
                updates_tensor,
                spec.axis,
                output_tensor,
            )
        except e:
            print("Err => ", e)

    bencher.iter[bench_fn]()

    _ = data_tensor
    _ = indices_tensor
    _ = updates_tensor
    _ = output_tensor


@fieldwise_init
struct ScatterSpec(Copyable, Movable, Stringable):
    var axis: Int
    var m1: Int
    var m2: Int
    var n1: Int
    var n2: Int

    @no_inline
    fn __str__(self) -> String:
        return String(
            "axis=",
            self.axis,
            ";Dim=(",
            self.m1,
            ",",
            self.m2,
            ")(",
            self.n1,
            ",",
            self.n2,
            ")",
        )


def main():
    var m = Bench(BenchConfig(num_repetitions=2))
    bench_scatter(m, ScatterSpec(axis=1, m1=400, m2=400, n1=200, n2=200))
    bench_scatter(m, ScatterSpec(axis=1, m1=1000, m2=1000, n1=200, n2=200))
    m.dump_report()
