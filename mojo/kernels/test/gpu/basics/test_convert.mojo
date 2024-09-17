# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host import DeviceContext
from gpu.host._compile import _compile_code, _get_nvptx_target
from testing import *


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
](dst_ptr: UnsafePointer[Scalar[dst_type]]):
    @parameter
    for i in range(0, size, 2):
        var src_vec = SIMD[src_type, 2](i, i + 1)
        var dst_vec = src_vec.cast[dst_type]()
        dst_ptr.store(i, dst_vec)


fn test_convert[src_type: DType, dst_type: DType](ctx: DeviceContext) raises:
    """Test the convertion ptx instruction.

    We can't verify this just by compilation. The instruction converts two values
     and swaps their reorder, which should be verified by checking runtime results.
    """

    alias size = 4
    var host_ptr = UnsafePointer[Scalar[dst_type]].alloc(size)
    var device_buf = ctx.create_buffer[dst_type](size)

    for i in range(size):
        host_ptr[i] = 0

    ctx.enqueue_copy_to_device(device_buf, host_ptr)

    var kernel = ctx.compile_function[
        convert_kernel[src_type, dst_type, size]
    ]()
    ctx.enqueue_function(kernel, device_buf, grid_dim=(1), block_dim=(1))
    ctx.enqueue_copy_from_device(host_ptr, device_buf)

    for i in range(size):
        assert_equal(host_ptr[i], i)

    _ = device_buf
    host_ptr.free()


fn main() raises:
    with DeviceContext() as ctx:
        test_convert_asm()
        # Only support 2xFP32 -> 2xBF16 convertion via ptx.
        test_convert[DType.float32, DType.bfloat16](ctx)
