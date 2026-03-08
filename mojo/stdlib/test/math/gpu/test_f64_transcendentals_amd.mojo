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

from test_utils import ulp_distance

from std.math import cosh, erf, exp, exp2, log, log1p, log2, pow
from std.gpu.host import DeviceContext
from std.sys import is_amd_gpu
from std.testing import assert_true, TestSuite


fn check_unary_ulp[
    name: StaticString,
    kernel_fn: fn(Float64) capturing -> Float64,
](val: Float64, ctx: DeviceContext,) raises:
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
        assert_true(
            ulp <= 10,
            String(
                name,
                "(",
                val,
                ") exceeded 10 ULP on AMD gfx90a: gpu=",
                out_host[0],
                " expected=",
                expected,
                " ulp=",
                ulp,
            ),
        )


fn check_binary_ulp[
    name: StaticString,
    kernel_fn: fn(Float64, Float64) capturing -> Float64,
](lhs: Float64, rhs: Float64, ctx: DeviceContext,) raises:
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
        assert_true(
            ulp <= 10,
            String(
                name,
                "(",
                lhs,
                ", ",
                rhs,
                ") exceeded 10 ULP on AMD gfx90a: gpu=",
                out_host[0],
                " expected=",
                expected,
                " ulp=",
                ulp,
            ),
        )


def test_f64_transcendentals_amd() raises:
    comptime if not is_amd_gpu["gfx90a"]():
        return

    @parameter
    fn exp_fn(x: Float64) -> Float64:
        return exp(x)

    @parameter
    fn exp2_fn(x: Float64) -> Float64:
        return exp2(x)

    @parameter
    fn log_fn(x: Float64) -> Float64:
        return log(x)

    @parameter
    fn log2_fn(x: Float64) -> Float64:
        return log2(x)

    @parameter
    fn log1p_fn(x: Float64) -> Float64:
        return log1p(x)

    @parameter
    fn cosh_fn(x: Float64) -> Float64:
        return cosh(x)

    @parameter
    fn erf_fn(x: Float64) -> Float64:
        return erf(x)

    @parameter
    fn pow_fn(x: Float64, y: Float64) -> Float64:
        return pow(x, y)

    with DeviceContext() as ctx:
        check_unary_ulp["exp", exp_fn](0.5, ctx)
        check_unary_ulp["exp", exp_fn](2.0, ctx)
        check_unary_ulp["exp2", exp2_fn](0.5, ctx)
        check_unary_ulp["exp2", exp2_fn](4.0, ctx)
        check_unary_ulp["log", log_fn](0.5, ctx)
        check_unary_ulp["log", log_fn](2.0, ctx)
        check_unary_ulp["log2", log2_fn](0.5, ctx)
        check_unary_ulp["log2", log2_fn](8.0, ctx)
        check_unary_ulp["log1p", log1p_fn](1e-6, ctx)
        check_unary_ulp["log1p", log1p_fn](0.5, ctx)
        check_unary_ulp["cosh", cosh_fn](0.5, ctx)
        check_unary_ulp["cosh", cosh_fn](2.0, ctx)
        check_unary_ulp["erf", erf_fn](0.5, ctx)
        check_unary_ulp["erf", erf_fn](1.5, ctx)
        check_binary_ulp["pow", pow_fn](2.0, 3.0, ctx)
        check_binary_ulp["pow", pow_fn](0.5, 2.0, ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
