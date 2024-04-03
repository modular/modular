# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s -t | FileCheck %s
# CHECK: Benchmark results

from benchmark import *
from nn.gather_scatter import scatter_elements
from tensor import Tensor, TensorShape
from collections.vector import InlinedFixedVector
from random import random_si64


fn linear_fill[
    type: DType,
](inout t: Tensor[type], elems: InlinedFixedVector[Scalar[type]]):
    var buf = t._to_buffer()
    for i in range(t.num_elements()):
        buf[i] = elems[i]


fn bench_scatter(inout m: Bench, spec: ScatterSpec) raises:
    @parameter
    @always_inline
    fn bench_scatter_wrapper(inout b: Bencher, concrete_spec: ScatterSpec):
        bench_scatter(b, concrete_spec)

    m.bench_with_input[ScatterSpec, bench_scatter_wrapper](
        BenchId("scatter", str(spec)), spec
    )


fn bench_scatter(inout bencher: Bencher, spec: ScatterSpec) capturing:
    var data = InlinedFixedVector[Float32](spec.m1 * spec.m2)
    var indices = InlinedFixedVector[Int32](spec.n1 * spec.n2)
    var updates = InlinedFixedVector[Float32](spec.n1 * spec.n2)

    var rand_min = -1000
    var rand_max = 1000
    var index_rand_min = 0
    var index_rand_max = spec.n1 * spec.n2 - 1

    for x in range(spec.m1 * spec.m2):
        var val = random_si64(rand_min, rand_max)
        data[x] = val.cast[DType.float32]()

    for x in range(spec.n1 * spec.n2):
        var val = random_si64(rand_min, rand_max)
        updates[x] = val.cast[DType.float32]()

    for x in range(spec.n1 * spec.n2):
        var val = random_si64(index_rand_min, index_rand_max)
        indices[x] = val.cast[DType.int32]()

    var input_shape = TensorShape(spec.m1, spec.m2)
    var indices_shape = TensorShape(spec.n1, spec.n2)
    var data_tensor = Tensor[DType.float32](input_shape)
    var indices_tensor = Tensor[DType.int32](indices_shape)
    var updates_tensor = Tensor[DType.float32](indices_shape)
    var output_tensor = Tensor[DType.float32](input_shape)

    linear_fill(data_tensor, data)
    linear_fill(indices_tensor, indices)
    linear_fill(updates_tensor, updates)

    @always_inline
    @parameter
    fn bench_fn():
        @always_inline
        @parameter
        fn reduce_fn[
            _type: DType, width: Int
        ](
            input_val: SIMD[_type, width], update_val: SIMD[_type, width]
        ) -> SIMD[_type, width]:
            return input_val + update_val

        try:
            scatter_elements[reduce_fn](
                data_tensor._to_ndbuffer[2](),
                indices_tensor._to_ndbuffer[2](),
                updates_tensor._to_ndbuffer[2](),
                spec.axis,
                output_tensor._to_ndbuffer[2](),
            )
        except e:
            print("Err => ", e)

    bencher.iter[bench_fn]()

    _ = data_tensor
    _ = indices_tensor
    _ = updates_tensor
    _ = output_tensor


@value
struct ScatterSpec(Stringable):
    var axis: Int
    var m1: Int
    var m2: Int
    var n1: Int
    var n2: Int

    fn __str__(self) -> String:
        return (
            "axis="
            + str(self.axis)
            + ";Dim=("
            + str(self.m1)
            + ","
            + str(self.m2)
            + ")("
            + str(self.n1)
            + ","
            + str(self.n2)
            + ")"
        )


def main():
    var m = Bench(BenchConfig(num_repetitions=2))
    bench_scatter(m, ScatterSpec(axis=1, m1=400, m2=400, n1=200, n2=200))
    bench_scatter(m, ScatterSpec(axis=1, m1=1000, m2=1000, n1=200, n2=200))
    m.dump_report()
