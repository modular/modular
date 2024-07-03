# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import sqrt, rsqrt
from sys.info import has_neon

from algorithm.functional import _elementwise_impl_gpu
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host._compile import _get_nvptx_target
from testing import *


def run_elementwise[
    type: DType,
    kernel_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
](ctx: DeviceContext):
    alias length = 256

    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()

    var in_host = NDBuffer[type, 1, DimList(length)].stack_allocation()
    var out_host = NDBuffer[type, 1, DimList(length)].stack_allocation()

    var flattened_length = in_host.num_elements()
    for i in range(length):
        in_host[i] = 0.001 * abs(Scalar[type](i) - length // 2)

    var in_device = ctx.create_buffer[type](flattened_length)
    var out_device = ctx.create_buffer[type](flattened_length)

    ctx.enqueue_copy_to_device(in_device, in_host.data)

    var in_buffer = NDBuffer[type, 1](in_device.ptr, (length))
    var out_buffer = NDBuffer[type, 1](out_device.ptr, (length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[1]](idx0)

        out_buffer.store[width=simd_width](
            idx, rsqrt(in_buffer.load[width=simd_width](idx))
        )

    _elementwise_impl_gpu[func, pack_size](StaticIntTuple[1](length), ctx)

    ctx.synchronize()

    ctx.enqueue_copy_from_device(out_host.data, out_device)

    for i in range(length):
        assert_almost_equal(
            out_host[i],
            rsqrt(in_host[i]),
            msg="values did not match at position "
            + str(i)
            + " for dtype="
            + str(type),
            atol=1e-08 if type is DType.float32 else 1e-04,
            rtol=1e-05 if type is DType.float32 else 1e-03,
        )

    _ = in_device
    _ = out_device


# CHECK-NOT: CUDA_ERROR
def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.float16, sqrt](ctx)
        run_elementwise[DType.float32, sqrt](ctx)
        run_elementwise[DType.float64, sqrt](ctx)
        run_elementwise[DType.float16, rsqrt](ctx)
        run_elementwise[DType.float32, rsqrt](ctx)
        run_elementwise[DType.float64, rsqrt](ctx)
