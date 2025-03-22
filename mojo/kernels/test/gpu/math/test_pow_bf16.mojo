# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from testing import assert_almost_equal

from utils import Index

alias type = DType.float32


def run_elementwise[do_bfloat_exp: Bool](exponent: Int, ctx: DeviceContext):
    alias length = 256
    alias pack_size = simdwidthof[type, target = _get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[type](length)
    var out_device = ctx.enqueue_create_buffer[type](length)

    # Add a small constant to avoid 0^-pow.
    alias epsilon = 0.001
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = (Scalar[type](i) - length // 2) + epsilon

    var in_buffer = NDBuffer[type, 1](in_device.unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[type, 1](out_device.unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer, exponent)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)

        var val = in_buffer.load[width=simd_width](idx).cast[DType.bfloat16]()
        var result: SIMD[DType.bfloat16, simd_width]

        @parameter
        if do_bfloat_exp:
            result = val ** SIMD[DType.bfloat16, simd_width](exponent)
        else:
            result = val**exponent
        out_buffer.store[width=simd_width](idx, result.cast[DType.float32]())

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value: SIMD[DType.float32, 1]

            @parameter
            if do_bfloat_exp:
                expected_value = in_host[i] ** SIMD[DType.float32, 1](exponent)
            else:
                expected_value = in_host[i] ** exponent

            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=1e-04,
                rtol=2e-02,
            )


def main():
    with DeviceContext() as ctx:
        run_elementwise[False](-1, ctx)
        run_elementwise[False](2, ctx)
        run_elementwise[False](3, ctx)
        run_elementwise[False](5, ctx)
        run_elementwise[False](6, ctx)
        run_elementwise[True](2, ctx)
        run_elementwise[True](3, ctx)
        run_elementwise[True](5, ctx)
        run_elementwise[True](6, ctx)
