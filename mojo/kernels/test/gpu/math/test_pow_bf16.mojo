# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from algorithm.functional import _elementwise_impl
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import Context
from gpu.host._compile import _get_nvptx_target
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.host.sync import synchronize
from testing import assert_almost_equal

alias type = DType.float32


def run_elementwise[do_bfloat_exp: Bool](exponent: Int):
    alias length = 256

    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()

    var in_host = NDBuffer[type, 1, DimList(length)].stack_allocation()
    var out_host = NDBuffer[type, 1, DimList(length)].stack_allocation()

    var flattened_length = in_host.num_elements()

    # Add a small constant to avoid 0^-pow.
    alias epsilon = 0.001
    for i in range(length):
        in_host[i] = (Scalar[type](i) - length // 2) + epsilon

    var in_device = _malloc[type](flattened_length)
    var out_device = _malloc[type](flattened_length)

    _copy_host_to_device(in_device, in_host.data, flattened_length)

    var in_buffer = NDBuffer[type, 1](in_device, (length))
    var out_buffer = NDBuffer[type, 1](out_device, (length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer, exponent)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[1]](idx0)

        var val = in_buffer.load[width=simd_width](idx).cast[DType.bfloat16]()
        var result: SIMD[DType.bfloat16, simd_width]

        @parameter
        if do_bfloat_exp:
            result = val ** SIMD[DType.bfloat16, simd_width](exponent)
        else:
            result = val**exponent
        out_buffer.store[width=simd_width](idx, result.cast[DType.float32]())

    _elementwise_impl[
        func, pack_size, 1, use_blocking_impl=True, target="cuda"
    ](
        StaticIntTuple[1](length),
    )
    synchronize()

    _copy_device_to_host(out_host.data, out_device, flattened_length)

    for i in range(length):
        var expected_value: SIMD[DType.float32, 1]

        @parameter
        if do_bfloat_exp:
            expected_value = in_host[i] ** SIMD[DType.float32, 1](exponent)
        else:
            expected_value = in_host[i] ** exponent

        assert_almost_equal[type, 1](
            out_host[i],
            expected_value,
            msg="values did not match at position " + str(i),
            atol=1e-04,
            rtol=2e-02,
        )

    _free(in_device)
    _free(out_device)


# CHECK-NOT: CUDA_ERROR
def main():
    with Context() as ctx:
        run_elementwise[False](-1)
        run_elementwise[False](2)
        run_elementwise[False](3)
        run_elementwise[False](5)
        run_elementwise[False](6)
        run_elementwise[True](2)
        run_elementwise[True](3)
        run_elementwise[True](5)
        run_elementwise[True](6)
