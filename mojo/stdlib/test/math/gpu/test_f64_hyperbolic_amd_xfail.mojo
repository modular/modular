# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
# REQUIRES: AMD-GPU
# XFAIL: *
# RUN: %bare-mojo --target-accelerator=mi250x %s

from test_utils import ulp_distance

from std.gpu.host import DeviceContext
from std.math import acosh, asinh, atanh, sinh, tanh
from std.testing import assert_true, TestSuite


fn check_unary[
    name: StaticString,
    kernel_fn: fn(Float64) capturing -> Float64,
](val: Float64, ctx: DeviceContext) raises:
    var out = ctx.enqueue_create_buffer[DType.float64](1)
    var expected = kernel_fn(val)

    @parameter
    fn kernel(
        out_dev: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
        x: Float64,
    ):
        out_dev[0] = kernel_fn(x)

    ctx.enqueue_function_experimental[kernel](out, val, grid_dim=1, block_dim=1)
    with out.map_to_host() as out_host:
        var ulp = ulp_distance(out_host[0], expected)
        assert_true(ulp <= 10, String(name, " ulp=", ulp))


def test_f64_hyperbolic_amd() raises:
    @parameter
    fn sinh_fn(x: Float64) -> Float64:
        return sinh(x)

    @parameter
    fn tanh_fn(x: Float64) -> Float64:
        return tanh(x)

    @parameter
    fn asinh_fn(x: Float64) -> Float64:
        return asinh(x)

    @parameter
    fn acosh_fn(x: Float64) -> Float64:
        return acosh(x)

    @parameter
    fn atanh_fn(x: Float64) -> Float64:
        return atanh(x)

    with DeviceContext() as ctx:
        check_unary["sinh", sinh_fn](0.5, ctx)
        check_unary["tanh", tanh_fn](0.5, ctx)
        check_unary["asinh", asinh_fn](0.5, ctx)
        check_unary["acosh", acosh_fn](2.0, ctx)
        check_unary["atanh", atanh_fn](0.5, ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
