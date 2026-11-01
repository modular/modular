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

from std.math import isnan, log10
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from test_utils.math import (
    f32_max_subnormal_bits,
    f32_min_subnormal_bits,
    f64_max_subnormal_bits,
    f64_min_subnormal_bits,
    log10_f32_inputs,
    log10_f64_inputs,
    num_log10_inputs,
    num_log10_specials,
    ulp_diff,
)

from std.utils.coord import Coord


def run_on_gpu[
    dtype: DType,
    math_fn: def[fn_dtype: DType, fn_width: SIMDLength](
        SIMD[fn_dtype, fn_width]
    ) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point(),
](ctx: DeviceContext, inputs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    assert_equal(len(inputs), num_log10_inputs)

    var in_device = ctx.enqueue_create_buffer[dtype](num_log10_inputs)
    var out_device = ctx.enqueue_create_buffer[dtype](num_log10_inputs)
    with in_device.map_to_host() as in_host:
        for i in range(num_log10_inputs):
            in_host[i] = inputs[i]

    var in_buffer = Span(
        unsafe_ptr=in_device.unsafe_ptr(), length=num_log10_inputs
    )
    var out_buffer = Span(
        unsafe_ptr=out_device.unsafe_ptr(), length=num_log10_inputs
    )

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(num_log10_inputs), ctx)

    var out = List[Scalar[dtype]](capacity=num_log10_inputs)
    with out_device.map_to_host() as out_host:
        for i in range(num_log10_inputs):
            out.append(out_host[i])
    return out^


# ===----------------------------------------------------------------------=== #
# log10 float64
# ===----------------------------------------------------------------------=== #


def test_log10_float64() raises:
    # glibc's log10 is not correctly rounded and is up to 2 ULP from the true
    # value near 1 (e.g. log10(0.9999999999999691)), so allow 2 ULP vs libm.
    comptime max_ulp_log10 = 2
    # 10^n is exactly representable in float64 for n <= 22.
    comptime max_exact_pow10 = 22
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            var inputs = log10_f64_inputs()
            var got = run_on_gpu[.float64, log10](ctx, inputs)
            var max_ulp = 0
            var max_ulp_host = 0
            for i in range(num_log10_inputs):
                var x = inputs[i]
                var expected = _call_libm["log10"](x)
                var d = ulp_diff(got[i], expected)
                max_ulp = max(max_ulp, d)
                assert_true(
                    d <= max_ulp_log10,
                    String(
                        t"gpu log10({x}) = {got[i]}, expected"
                        t" {expected} ({d} ulp)"
                    ),
                )
                var host = log10(x)
                var dh = ulp_diff(host, expected)
                max_ulp_host = max(max_ulp_host, dh)
                assert_true(
                    dh <= max_ulp_log10,
                    String(
                        t"host log10({x}) = {host}, expected"
                        t" {expected} ({dh} ulp)"
                    ),
                )
            for n in range(max_exact_pow10 + 1):
                assert_equal(got[num_log10_specials + n], Float64(n))
            print("log10 float64 max ulp: gpu", max_ulp, "host", max_ulp_host)


# ===----------------------------------------------------------------------=== #
# log10 float32
# ===----------------------------------------------------------------------=== #


def test_log10_float32() raises:
    # Against float64 libm rounded to float32; that reference is itself
    # within 1 ULP of the true value. Every target runs the same algorithm,
    # so GPU and host agree to within the rounding of fused multiply-adds.
    comptime max_ulp_f32 = 1
    with DeviceContext() as ctx:
        var inputs = log10_f32_inputs()
        var got = run_on_gpu[.float32, log10](ctx, inputs)
        var max_ulp = 0
        var max_ulp_host = 0
        for i in range(num_log10_inputs):
            var x = inputs[i]
            var expected = _call_libm["log10"](x.cast[.float64]()).cast[
                .float32
            ]()
            var d = ulp_diff(got[i], expected)
            max_ulp = max(max_ulp, d)
            assert_true(
                d <= max_ulp_f32,
                String(
                    t"gpu log10({x}) = {got[i]}, expected {expected} ({d} ulp)"
                ),
            )
            var host = log10(x)
            var dh = ulp_diff(host, expected)
            max_ulp_host = max(max_ulp_host, dh)
            assert_true(
                dh <= max_ulp_f32,
                String(
                    t"host log10({x}) = {host}, expected {expected} ({dh} ulp)"
                ),
            )
        for n in range(11):
            assert_equal(got[num_log10_specials + n], Float32(n))
        print("log10 float32 max ulp: gpu", max_ulp, "host", max_ulp_host)


# ===----------------------------------------------------------------------=== #
# log10 subnormal inputs
# ===----------------------------------------------------------------------=== #


def _check_subnormals_on_gpu[
    dtype: DType
](
    ctx: DeviceContext, subnormals: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    var inputs = List[Scalar[dtype]](capacity=num_log10_inputs)
    for x in subnormals:
        inputs.extend([x, -x])
    while len(inputs) < num_log10_inputs:
        inputs.append(1.0)
    var got = run_on_gpu[dtype, log10](ctx, inputs)
    for i in range(len(subnormals)):
        var x = subnormals[i]
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[dtype]()
        var d = ulp_diff(got[2 * i], expected)
        assert_true(
            d <= 1,
            String(
                t"gpu log10({x}) = {got[2 * i]}, expected {expected} ({d} ulp)"
            ),
        )
        # A negative subnormal is still negative, not a zero.
        assert_true(
            isnan(got[2 * i + 1]),
            String(t"gpu log10({-x}) = {got[2 * i + 1]}, expected nan"),
        )


def test_log10_subnormal() raises:
    """The log reduction reads a subnormal's exponent from its bits, so
    subnormal inputs survive targets that flush subnormal arithmetic operands.
    """
    with DeviceContext() as ctx:
        _check_subnormals_on_gpu[.float32](
            ctx,
            [
                bitcast[.float32](f32_min_subnormal_bits),
                bitcast[.float32](UInt32(2)),
                bitcast[.float32](UInt32(0x0000_0123)),
                bitcast[.float32](UInt32(0x0040_0000)),
                bitcast[.float32](UInt32(0x0055_5555)),
                bitcast[.float32](f32_max_subnormal_bits),
            ],
        )
        _check_subnormals_on_gpu[.bfloat16](
            ctx,
            [
                bitcast[.bfloat16](UInt16(1)),
                bitcast[.bfloat16](UInt16(0x0023)),
                bitcast[.bfloat16](UInt16(0x0040)),
                bitcast[.bfloat16](UInt16(0x007F)),
            ],
        )
        _check_subnormals_on_gpu[.float16](
            ctx,
            [
                bitcast[.float16](UInt16(1)),
                bitcast[.float16](UInt16(0x0123)),
                bitcast[.float16](UInt16(0x0200)),
                bitcast[.float16](UInt16(0x03FF)),
            ],
        )
        comptime if not default_accelerator().is_apple_gpu():
            _check_subnormals_on_gpu[.float64](
                ctx,
                [
                    bitcast[.float64](f64_min_subnormal_bits),
                    bitcast[.float64](f64_max_subnormal_bits),
                ],
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
