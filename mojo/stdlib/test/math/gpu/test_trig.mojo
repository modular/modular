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
from testing import assert_almost_equal, TestSuite


fn run_func[
    dtype: DType, kernel_fn: fn (Scalar[dtype]) capturing -> Scalar[dtype]
](
    out_prefix: String,
    val: Scalar[dtype],
    ref_: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    print("test trigonometric functions on gpu")

    var out = ctx.enqueue_create_buffer[dtype](1)

    @parameter
    fn kernel(out_dev: UnsafePointer[Scalar[dtype]], lhs: Scalar[dtype]):
        var result = kernel_fn(lhs)
        out_dev[0] = result

    ctx.enqueue_function_checked[kernel, kernel](
        out, val, grid_dim=1, block_dim=1
    )
    with out.map_to_host() as out_host:
        assert_almost_equal(
            out_host[0],
            ref_,
            msg=String("while testing ", out_prefix, " for the dtype ", dtype),
            atol=1e-2 if dtype.is_half_float() else 1e-8,
        )


def test_trig():
    @parameter
    fn cos_fn(val: Float16) -> Float16:
        return cos(val)

    @parameter
    fn cos_fn(val: Float32) -> Float32:
        return cos(val)

    @parameter
    fn sin_fn(val: Float16) -> Float16:
        return sin(val)

    @parameter
    fn sin_fn(val: Float32) -> Float32:
        return sin(val)

    with DeviceContext() as ctx:
        run_func[DType.float32, cos_fn]("cos", 10, -0.83907192945480347, ctx)
        run_func[DType.float16, cos_fn]("cos", 10, -0.8388671875, ctx)
        run_func[DType.float32, sin_fn]("sin", 10, -0.54402029514312744, ctx)
        run_func[DType.float16, sin_fn]("sin", 10, -0.5439453125, ctx)


def main():
    var suite = TestSuite()

    suite.test[test_trig]()

    suite^.run()
