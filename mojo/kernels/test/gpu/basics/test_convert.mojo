# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc_managed,
)
from gpu.intrinsics import convert
from memory.unsafe import DTypePointer
from testing import *
from gpu.host._compile import _compile_code, _get_nvptx_target


def test_convert_asm():
    @parameter
    fn my_cast[frm: DType, to: DType](x: Scalar[frm]) -> Scalar[to]:
        return x.cast[to]()

    assert_true(
        "cvt.rn.f16.f32"
        in str(
            _compile_code[
                my_cast[DType.float32, DType.float16], emission_kind="asm"
            ]().asm
        )
    )

    assert_true(
        "cvt.f32.f16"
        in str(
            _compile_code[
                my_cast[DType.float16, DType.float32], emission_kind="asm"
            ]().asm
        )
    )


fn convert_kernel[
    src_type: DType, dst_type: DType, size: Int
](dst_ptr: DTypePointer[dst_type]):
    @parameter
    for i in range(0, size, 2):
        var src_vec = SIMD[src_type, 2](i, i + 1)
        var dst_vec = convert[src_type, dst_type, 2](src_vec)
        SIMD.store(dst_ptr, i, dst_vec)


fn test_convert[src_type: DType, dst_type: DType]() raises:
    """Test the convertion ptx instruction.

    We can't verify this just by compilation. The instruction converts two values
     and swaps their reorder, which should be verified by checking runtime results.
    """

    alias size = 4
    var ptr = _malloc_managed[dst_type](size)

    for i in range(size):
        ptr[i] = 0

    var kernel = Function[convert_kernel[src_type, dst_type, size]]()
    kernel(ptr, grid_dim=(1), block_dim=(1))
    synchronize()

    for i in range(size):
        assert_equal(ptr[i], i)

    _free(ptr)


fn main() raises:
    try:
        with Context():
            test_convert_asm()
            # Only support 2xFP32 -> 2xBF16 convertion via ptx.
            test_convert[DType.float32, DType.bfloat16]()
    except e:
        print("CUDA_ERROR:", e)
