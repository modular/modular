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

    var in_host = NDBuffer[type, 1, DimList(length)].stack_allocation()
    var out_host = NDBuffer[type, 1, DimList(length)].stack_allocation()

    var flattened_length = in_host.num_elements()

    var out_device = ctx.create_buffer[type](flattened_length)

    var out_buffer = NDBuffer[type, 1](out_device.ptr, (length))

    @always_inline
    @__copy_capture(out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        alias fast_div = FastDiv[DType.uint32](4)

        out_buffer[idx0[0]] = (fast_div / idx0[0]).cast[type]()

    elementwise[func, 1, target="cuda"](Index(length), ctx)

    ctx.synchronize()

    ctx.enqueue_copy_from_device(out_host.data, out_device)

    for i in range(length):
        assert_equal(out_host[i], i // 4)

    _ = out_device
    _ = out_host


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.uint32](ctx)
