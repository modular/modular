# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import isqrt, sqrt
from sys import has_neon, simdwidthof

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu import *
from linalg.fast_div import FastDiv
from gpu.host import DeviceContext
from gpu.host._compile import _get_nvptx_target
from testing import *
from utils.index import Index


def run_elementwise[type: DType](ctx: DeviceContext):
    alias length = 256

    var divisors = NDBuffer[type, 1, DimList(length)].stack_allocation()
    var remainders = NDBuffer[type, 1, DimList(length)].stack_allocation()

    var out_divisors = ctx.create_buffer[type](length)
    var out_remainders = ctx.create_buffer[type](length)

    var out_divisors_buffer = NDBuffer[type, 1](out_divisors.ptr, (length))
    var out_remainders_buffer = NDBuffer[type, 1](out_remainders.ptr, (length))

    @always_inline
    @__copy_capture(out_divisors_buffer, out_remainders_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        alias fast_div = FastDiv[DType.uint32](4)
        var idx = idx0[0]

        out_divisors_buffer[idx] = (idx / fast_div).cast[type]()
        out_remainders_buffer[idx] = (idx % fast_div).cast[type]()

    elementwise[func, simd_width=1, target="cuda"](Index(length), ctx)

    ctx.synchronize()

    ctx.enqueue_copy_from_device(divisors.data, out_divisors)
    ctx.enqueue_copy_from_device(remainders.data, out_remainders)

    for i in range(length):
        print(divisors[i], remainders[i])
        assert_equal(divisors[i], i // 4, msg="the divisor is not correct")
        assert_equal(remainders[i], i % 4, msg="the remainder is not correct")

    _ = out_divisors
    _ = out_remainders
    _ = out_divisors_buffer
    _ = out_remainders_buffer
    _ = divisors
    _ = remainders


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.uint32](ctx)
