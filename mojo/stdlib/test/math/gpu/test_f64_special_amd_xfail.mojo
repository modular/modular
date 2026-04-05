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
from std.math import (
    cbrt,
    erf,
    erfc,
    expm1,
    gamma,
    hypot,
    j0,
    j1,
    lgamma,
    log10,
    logb,
    y0,
    y1,
)
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


def test_f64_special_amd() raises:
    @parameter
    fn expm1_fn(x: Float64) -> Float64:
        return expm1(x)

    @parameter
    fn log10_fn(x: Float64) -> Float64:
        return log10(x)

    @parameter
    fn erfc_fn(x: Float64) -> Float64:
        return erfc(x)

    @parameter
    fn cbrt_fn(x: Float64) -> Float64:
        return cbrt(x)

    @parameter
    fn gamma_fn(x: Float64) -> Float64:
        return gamma(x)

    @parameter
    fn lgamma_fn(x: Float64) -> Float64:
        return lgamma(x)

    @parameter
    fn logb_fn(x: Float64) -> Float64:
        return logb(x)

    @parameter
    fn j0_fn(x: Float64) -> Float64:
        return j0(x)

    @parameter
    fn j1_fn(x: Float64) -> Float64:
        return j1(x)

    @parameter
    fn y0_fn(x: Float64) -> Float64:
        return y0(x)

    @parameter
    fn y1_fn(x: Float64) -> Float64:
        return y1(x)

    @parameter
    fn hypot_fn(x: Float64, y: Float64) -> Float64:
        return hypot(x, y)

    with DeviceContext() as ctx:
        check_unary["expm1", expm1_fn](0.5, ctx)
        check_unary["log10", log10_fn](10.0, ctx)
        check_unary["erfc", erfc_fn](0.5, ctx)
        check_unary["cbrt", cbrt_fn](8.0, ctx)
        check_unary["gamma", gamma_fn](5.0, ctx)
        check_unary["lgamma", lgamma_fn](5.0, ctx)
        check_unary["logb", logb_fn](8.0, ctx)
        check_unary["j0", j0_fn](0.5, ctx)
        check_unary["j1", j1_fn](0.5, ctx)
        check_unary["y0", y0_fn](0.5, ctx)
        check_unary["y1", y1_fn](0.5, ctx)
        check_binary["hypot", hypot_fn](3.0, 4.0, ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
