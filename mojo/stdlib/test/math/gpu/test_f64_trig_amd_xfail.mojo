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
from std.math import acos, asin, atan, atan2, cos, sin, tan
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


fn check_binary[
    name: StaticString,
    kernel_fn: fn(Float64, Float64) capturing -> Float64,
](lhs: Float64, rhs: Float64, ctx: DeviceContext) raises:
    var out = ctx.enqueue_create_buffer[DType.float64](1)
    var expected = kernel_fn(lhs, rhs)

    @parameter
    fn kernel(
        out_dev: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
        x: Float64,
        y: Float64,
    ):
        out_dev[0] = kernel_fn(x, y)

    ctx.enqueue_function_experimental[kernel](
        out, lhs, rhs, grid_dim=1, block_dim=1
    )
    with out.map_to_host() as out_host:
        var ulp = ulp_distance(out_host[0], expected)
        assert_true(ulp <= 10, String(name, " ulp=", ulp))


def test_f64_trig_amd() raises:
    @parameter
    fn sin_fn(x: Float64) -> Float64:
        return sin(x)

    @parameter
    fn cos_fn(x: Float64) -> Float64:
        return cos(x)

    @parameter
    fn tan_fn(x: Float64) -> Float64:
        return tan(x)

    @parameter
    fn asin_fn(x: Float64) -> Float64:
        return asin(x)

    @parameter
    fn acos_fn(x: Float64) -> Float64:
        return acos(x)

    @parameter
    fn atan_fn(x: Float64) -> Float64:
        return atan(x)

    @parameter
    fn atan2_fn(x: Float64, y: Float64) -> Float64:
        return atan2(x, y)

    with DeviceContext() as ctx:
        check_unary["sin", sin_fn](0.5, ctx)
        check_unary["cos", cos_fn](0.5, ctx)
        check_unary["tan", tan_fn](0.5, ctx)
        check_unary["asin", asin_fn](0.5, ctx)
        check_unary["acos", acos_fn](0.5, ctx)
        check_unary["atan", atan_fn](0.5, ctx)
        check_binary["atan2", atan2_fn](0.5, 2.0, ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
