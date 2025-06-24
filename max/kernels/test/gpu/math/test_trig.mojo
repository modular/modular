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

from math import cos, sin

from gpu.host import DeviceContext
from testing import assert_almost_equal


fn run_func[
    type: DType, kernel_fn: fn (SIMD[type, 1]) capturing -> SIMD[type, 1]
](
    out_prefix: String,
    val: Scalar[type],
    ref_: Scalar[type],
    ctx: DeviceContext,
) raises:
    print("test trigonometric functions on gpu")

    var out = ctx.enqueue_create_buffer[type](1)

    @parameter
    fn kernel(out_dev: UnsafePointer[Scalar[type]], lhs: SIMD[type, 1]):
        var result = kernel_fn(lhs)
        out_dev[0] = result

    ctx.enqueue_function[kernel](out, val, grid_dim=1, block_dim=1)
    with out.map_to_host() as out_host:
        assert_almost_equal(
            out_host[0],
            ref_,
            msg=String("while testing ", out_prefix, " for the dtype ", type),
            atol=1e-2 if type.is_half_float() else 1e-8,
        )


def main():
    @parameter
    fn cos_fn(val: SIMD[DType.float16, 1]) -> SIMD[DType.float16, 1]:
        return cos(val)

    @parameter
    fn cos_fn(val: SIMD[DType.float32, 1]) -> SIMD[DType.float32, 1]:
        return cos(val)

    @parameter
    fn sin_fn(val: SIMD[DType.float16, 1]) -> SIMD[DType.float16, 1]:
        return sin(val)

    @parameter
    fn sin_fn(val: SIMD[DType.float32, 1]) -> SIMD[DType.float32, 1]:
        return sin(val)

    with DeviceContext() as ctx:
        run_func[DType.float32, cos_fn]("cos", 10, -0.83907192945480347, ctx)
        run_func[DType.float16, cos_fn]("cos", 10, -0.8388671875, ctx)
        run_func[DType.float32, sin_fn]("sin", 10, -0.54402029514312744, ctx)
        run_func[DType.float16, sin_fn]("sin", 10, -0.5439453125, ctx)
