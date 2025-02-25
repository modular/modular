# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from testing import assert_almost_equal

from utils import Index

alias type = DType.float32


def run_elementwise(exponent: BFloat16, ctx: DeviceContext):
    alias length = 256

    alias pack_size = simdwidthof[type, target = _get_gpu_target()]()

    var in_host = NDBuffer[type, 1, DimList(length)].stack_allocation()
    var out_host = NDBuffer[type, 1, DimList(length)].stack_allocation()

    var flattened_length = in_host.num_elements()

    # Add a small constant to avoid 0^-pow.
    alias epsilon = 0.001
    for i in range(length):
        in_host[i] = abs((Scalar[type](i) - length // 2) + epsilon)

    var in_device = ctx.enqueue_create_buffer[type](flattened_length)
    var out_device = ctx.enqueue_create_buffer[type](flattened_length)

    ctx.enqueue_copy(in_device, in_host.data)

    var in_buffer = NDBuffer[type, 1](in_device.unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[type, 1](out_device.unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer, exponent)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)

        var val = in_buffer.load[width=simd_width](idx).cast[DType.bfloat16]()
        var result = val ** SIMD[DType.bfloat16, simd_width](exponent)
        out_buffer.store[width=simd_width](idx, result.cast[DType.float32]())

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    ctx.enqueue_copy(out_host.data, out_device)
    ctx.synchronize()

    for i in range(length):
        var expected_value = in_host[i] ** exponent.cast[DType.float32]()
        assert_almost_equal(
            out_host[i],
            expected_value,
            msg=String("values did not match at position ", i),
            atol=1e-04,
            rtol=2e-02,
        )

    _ = in_device
    _ = out_device


def main():
    # NOTE: This is expected to fail. Keeping this around as a negative test
    # so we know when its fixed.
    with DeviceContext() as ctx:
        run_elementwise(0.375, ctx)
