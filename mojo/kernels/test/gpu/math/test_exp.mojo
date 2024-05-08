# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import exp, isclose
from sys.info import has_neon, triple_is_nvidia_cuda

from algorithm.functional import _elementwise_impl
from benchmark._cuda import run
from buffer import NDBuffer, DimList
from builtin.io import _printf
from gpu import *
from gpu.host import Context, Dim, Function, Stream
from gpu.host._compile import _get_nvptx_target
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from gpu.host.sync import synchronize
from testing import *

from utils.index import Index


def run_elementwise[type: DType]():
    alias length = 256

    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()

    var in_host = NDBuffer[type, 1, DimList(length)].stack_allocation()
    var out_host = NDBuffer[type, 1, DimList(length)].stack_allocation()

    var flattened_length = in_host.num_elements()
    for i in range(length):
        in_host[i] = 0.001 * (Scalar[type](i) - length // 2)

    var in_device = _malloc[type](flattened_length)
    var out_device = _malloc[type](flattened_length)

    _copy_host_to_device(in_device, in_host.data, flattened_length)

    var in_buffer = NDBuffer[type, 1](in_device, (length))
    var out_buffer = NDBuffer[type, 1](out_device, (length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[1]](idx0)

        out_buffer.store[width=simd_width](
            idx, exp(in_buffer.load[width=simd_width](idx))
        )

    _elementwise_impl[
        func, pack_size, 1, use_blocking_impl=True, target="cuda"
    ](
        StaticIntTuple[1](length),
    )
    synchronize()

    _copy_device_to_host(out_host.data, out_device, flattened_length)

    for i in range(length):
        assert_almost_equal(
            out_host[i],
            exp(in_host[i]),
            msg="values did not match at position "
            + str(i)
            + " for dtype="
            + str(type),
            atol=1e-08 if type == DType.float32 else 1e-04,
            rtol=1e-05 if type == DType.float32 else 1e-03,
        )

    _free(in_device)
    _free(out_device)


# CHECK-NOT: CUDA_ERROR
def main():
    with Context() as ctx:

        @parameter
        if not has_neon():
            run_elementwise[DType.float16]()

        run_elementwise[DType.float32]()
