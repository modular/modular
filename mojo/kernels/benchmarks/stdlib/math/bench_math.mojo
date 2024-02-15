# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from mojobench import MojoBench, Bencher, BenchId
from benchmark import keep
import math
from algorithm.functional import vectorize


fn apply[
    func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, width],
    type: DType,
](input: Buffer[type], output: Buffer[type]):
    @parameter
    fn _func[width: Int](idx: Int):
        output.simd_store(idx, input.simd_load[width](idx))

    vectorize[_func, simdwidthof[type]()](len(input))


def bench_unary[
    func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, width],
    type: DType,
](inout m: MojoBench, size: Int, op_name: String):
    alias alignment = 64
    var input_ptr = DTypePointer[type].aligned_alloc(alignment, size)
    var output_ptr = DTypePointer[type].aligned_alloc(alignment, size)

    @parameter
    fn bench(inout b: Bencher, size: Int):
        @parameter
        fn iter_fn():
            apply[func](
                Buffer[type](input_ptr, size),
                Buffer[type](output_ptr, size),
            )
            keep(output_ptr)

        b.iter[iter_fn]()

    m.bench_with_input[Int, bench](
        BenchId(op_name, str(size)),
        size,
        throughput_elems=size * sizeof[type](),
    )

    DTypePointer[type].free(input_ptr)
    DTypePointer[type].free(output_ptr)


def main():
    var m = MojoBench()
    for i in range(4):
        bench_unary[math.exp, DType.float32](m, 1 << (10 + i), "exp")
        bench_unary[math.erf, DType.float32](m, 1 << (10 + i), "erf")
    m.dump_report()
