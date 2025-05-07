# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from memory import UnsafePointer
from testing import *


def test_convert_asm():
    @parameter
    fn my_cast[
        frm: DType, to: DType
    ](output: UnsafePointer[Scalar[to]], x: Scalar[frm]):
        output[] = x.cast[to]()

    assert_true(
        "cvt.rn.f16.f32"
        in _compile_code_asm[
            my_cast[DType.float32, DType.float16],
            emission_kind="asm",
            target = _get_gpu_target["sm_80"](),
        ]()
    )

    assert_true(
        "v_cvt_f16_f32_e32"
        in _compile_code_asm[
            my_cast[DType.float32, DType.float16],
            emission_kind="asm",
            target = _get_gpu_target["mi300x"](),
        ]()
    )

    assert_true(
        "cvt.f32.f16"
        in _compile_code_asm[
            my_cast[DType.float16, DType.float32],
            emission_kind="asm",
            target = _get_gpu_target["sm_80"](),
        ]()
    )

    assert_true(
        "v_cvt_f32_f16_e32"
        in _compile_code_asm[
            my_cast[DType.float16, DType.float32],
            emission_kind="asm",
            target = _get_gpu_target["mi300x"](),
        ]()
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
    var device_buf = ctx.enqueue_create_buffer[dst_type](size)
    _ = device_buf.enqueue_fill(0)

    ctx.enqueue_function[convert_kernel[src_type, dst_type, size]](
        device_buf, grid_dim=(1), block_dim=(1)
    )
    with device_buf.map_to_host() as host_buf:
        for i in range(size):
            assert_equal(host_buf[i], i)


fn main() raises:
    with DeviceContext() as ctx:
        test_convert_asm()
        # Only support 2xFP32 -> 2xBF16 convertion via ptx.
        test_convert[DType.float32, DType.bfloat16](ctx)
