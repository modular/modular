# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This uses mandelbrot as an example to test how the entire stdlib works
# together.
#
# ===----------------------------------------------------------------------=== #
from algorithm.functional import vectorize, vectorize_unroll
from random import rand
import benchmark

alias type = DType.uint8
alias width = simdwidthof[type]()
alias unit = benchmark.Unit.ns
# increasing will reduce the benefit of passing the size as a paramater
alias multiplier = 1
# Add .5 of the elements that fit into a simd register
alias size: Int = (width * (multiplier + 0.5)).to_int()
alias param_loops = multiplier + 1
alias arg_loops = size % width + multiplier
alias unroll_factor = 2

alias p1 = DTypePointer[type].alloc(size)
alias p2 = DTypePointer[type].alloc(size)


fn arg_size():
    @parameter
    fn sum_all[width: Int](i: Int):
        p2.simd_store(i, p1.simd_load[width](i) * i)

    vectorize[width, sum_all](size)


fn arg_size_unroll():
    @parameter
    fn sum_all[width: Int](i: Int):
        p2.simd_store(i, p1.simd_load[width](i) * i)

    vectorize_unroll[width, unroll_factor, sum_all](size)


fn param_size():
    @parameter
    fn sum_all[width: Int](i: Int):
        p2.simd_store(i, p1.simd_load[width](i) * i)

    vectorize[width, size, sum_all]()


fn param_size_unroll():
    @parameter
    fn sum_all[width: Int](i: Int):
        p2.simd_store(i, p1.simd_load[width](i) * i)

    vectorize_unroll[width, size, unroll_factor, sum_all]()


fn main():
    rand(p1, size)
    let arg = benchmark.run[arg_size](max_runtime_secs=0.5).mean(unit)
    let param = benchmark.run[param_size](max_runtime_secs=0.5).mean(unit)
    let arg_unroll = benchmark.run[arg_size_unroll](max_runtime_secs=0.5).mean(
        unit
    )
    let param_unroll = benchmark.run[param_size_unroll](
        max_runtime_secs=0.5
    ).mean(unit)

    print(
        "calculating",
        size,
        "elements,",
        width,
        "elements fit into the SIMD register\n",
    )

    print("size as argument", arg_loops, "loops at:  ", arg, unit)
    print("     unroll_factor:", unroll_factor, "        ", arg_unroll, unit)
    print("\nsize as parameter", param_loops, "loops at: ", param, unit)
    print("     unroll_factor:", unroll_factor, "        ", param_unroll, unit)
    print(
        "\nPassing size as a parameter is",
        arg_unroll / param_unroll,
        "times faster than passing size as an argument due to less loops",
    )
