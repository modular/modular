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

from math import recip

from gpu.host import DeviceContext
from testing import assert_almost_equal, TestSuite


fn run_func[
    dtype: DType
](val: Scalar[dtype], ref_: Scalar[dtype], ctx: DeviceContext) raises:
    var out = ctx.enqueue_create_buffer[dtype](1)

    @parameter
    fn kernel(out_dev: UnsafePointer[Scalar[dtype]], lhs: Scalar[dtype]):
        var result = recip(lhs)
        out_dev[0] = result

    ctx.enqueue_function_checked[kernel, kernel](
        out, val, grid_dim=1, block_dim=1
    )
    with out.map_to_host() as out_host:
        assert_almost_equal(
            out_host[0],
            ref_,
            msg=String("while testing recip for the dtype ", dtype),
            atol=1e-8,
        )


def test_recip():
    with DeviceContext() as ctx:
        run_func[DType.float64](8, 0.125, ctx)
        run_func[DType.float32](5, 0.2, ctx)
        run_func[DType.float16](-4, -0.25, ctx)
        run_func[DType.bfloat16](2, 0.5, ctx)


def main():
    var suite = TestSuite()

    suite.test[test_recip]()

    suite^.run()
