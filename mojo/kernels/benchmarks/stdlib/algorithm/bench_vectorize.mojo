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

# XFAIL: *
# RUN: %mojo %s -t | FileCheck %s
# CHECK: Benchmark results

from random import rand

from algorithm.functional import vectorize
from benchmark import Unit, run
from memory.unsafe import DTypePointer

alias type = DType.uint8
alias width = simdwidthof[type]()
alias unit = Unit.ns
# increasing will reduce the benefit of passing the size as a paramater
alias multiplier = 2
# Add .5 of the elements that fit into a simd register
alias size: Int = (multiplier * width + (width * 0.5)).to_int()
alias unroll_factor = 2
alias its = 1000


fn main():
    var p1 = DTypePointer[type].alloc(size)
    var p2 = DTypePointer[type].alloc(size)

    rand(p1, size)

    @parameter
    fn arg_size():
        @parameter
        fn closure[width: Int](i: Int):
            p2.store(i, p1.load[width=width](i) + p2.load[width=width](i))

        for i in range(its):
            vectorize[closure, width](size)

    @parameter
    fn param_size():
        @parameter
        fn closure[width: Int](i: Int):
            p2.store(i, p1.load[width=width](i) + p2.load[width=width](i))

        for i in range(its):
            vectorize[closure, width, size=size]()

    @parameter
    fn arg_size_unroll():
        @parameter
        fn closure[width: Int](i: Int):
            p2.store(i, p1.load[width=width](i) + p2.load[width=width](i))

        for i in range(its):
            vectorize[closure, width, unroll_factor=unroll_factor](size)

    @parameter
    fn param_size_unroll():
        @parameter
        fn closure[width: Int](i: Int):
            p2.store(i, p1.load[width=width](i) + p2.load[width=width](i))

        for i in range(its):
            vectorize[closure, width, size=size, unroll_factor=unroll_factor]()

    var arg = run[arg_size](max_runtime_secs=0.5).mean(unit)
    print(p2.load[size]())
    memset_zero(p2, size)

    var param = run[param_size](max_runtime_secs=0.5).mean(unit)
    print(p2.load[size]())
    memset_zero(p2, size)

    var arg_unroll = run[arg_size_unroll](max_runtime_secs=0.5).mean(unit)
    print(p2.load[size]())
    memset_zero(p2, size)

    var param_unroll = run[param_size_unroll](max_runtime_secs=0.5).mean(unit)
    print(p2.load[size]())

    print(
        "calculating",
        size,
        "elements,",
        width,
        "elements fit into the SIMD register\n",
    )

    print(" size as argument:", arg, unit)
    print("         unrolled:", arg_unroll, unit)
    print()
    print("size as parameter:", param, unit)
    print("         unrolled:", param_unroll, unit)
    print(
        "\nPassing size as a parameter and unrolling is",
        arg_unroll / param_unroll,
        "x faster",
    )
    p1.free()
    p2.free()
