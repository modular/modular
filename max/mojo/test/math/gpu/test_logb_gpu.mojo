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

from std.math import inf, isnan, logb, nan
from std.math.math import _call_libm
from std.memory import bitcast
from std.builtin.dtype import _unsigned_integral_type_of
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from test_utils.math import f64_min_subnormal_bits, xorshift64
from std.utils.numerics import FPUtils

from std.utils.coord import Coord

comptime length = 16384
# 2^64 / golden ratio, the Fibonacci-hashing multiplier.
comptime golden_ratio_u64: UInt64 = 0x9E37_79B9_7F4A_7C15


def run_on_gpu[
    dtype: DType,
    math_fn: def[fn_dtype: DType, fn_width: SIMDLength](
        SIMD[fn_dtype, fn_width]
    ) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point(),
](ctx: DeviceContext, inputs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    assert_equal(len(inputs), length)

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = inputs[i]

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var out = List[Scalar[dtype]](capacity=length)
    with out_device.map_to_host() as out_host:
        for i in range(length):
            out.append(out_host[i])
    return out^


def _logb_inputs[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    comptime I = _unsigned_integral_type_of[dtype]()
    comptime nbits = bit_width_of[dtype]()
    var v: List[Scalar[dtype]] = [
        0.0,
        -0.0,
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
        -nan[dtype](),
    ]
    v.reserve(length)
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime mantissa_mask = (Scalar[I](1) << Scalar[I](mantissa_width)) - 1
    var sign_bit = Scalar[I](1) << Scalar[I](nbits - 1)
    # Every normal exponent field, both signs, with the smallest and largest
    # mantissa.
    for e in range(1, (1 << FPUtils[dtype].exponent_width()) - 1):
        var top = Scalar[I](e) << Scalar[I](mantissa_width)
        for bits in [top, top | mantissa_mask]:
            v.append(bitcast[dtype](bits))
            v.append(bitcast[dtype](bits | sign_bit))
    # Single-bit and all-ones-below patterns: subnormals at every bit position
    # and powers of two across the encoding.
    for i in range(nbits - 1):
        var b = Scalar[I](1) << Scalar[I](i)
        v.append(bitcast[dtype](b))
        v.append(bitcast[dtype](b | sign_bit))
        v.append(bitcast[dtype](b | (b - 1)))
    # Pseudo-random bit patterns over the whole encoding space.
    var state = golden_ratio_u64
    while len(v) < length:
        v.append(bitcast[dtype](xorshift64(state).cast[I]()))
    return v^


def _same(a: Scalar, b: type_of(a)) -> Bool:
    comptime I = _unsigned_integral_type_of[a.dtype]()
    return (isnan(a) and isnan(b)) or bitcast[I](a) == bitcast[I](b)


def _test_logb[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var inputs = _logb_inputs[dtype]()
    var got = run_on_gpu[dtype, logb](ctx, inputs)
    for i in range(length):
        var x = inputs[i]
        var expected = _call_libm["logb"](x)
        var host = logb(x)
        assert_true(
            _same(got[i], expected),
            String(t"gpu logb[{dtype}]({x}) = {got[i]}, expected {expected}"),
        )
        assert_true(
            _same(host, expected),
            String(t"host logb[{dtype}]({x}) = {host}, expected {expected}"),
        )


def test_logb() raises:
    with DeviceContext() as ctx:
        _test_logb[.float32](ctx)
        _test_logb[.float16](ctx)
        _test_logb[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _test_logb[.float64](ctx)


def _test_logb_subnormal[
    dtype: DType
](
    ctx: DeviceContext, tiny: Scalar[dtype], tiny_logb: Scalar[dtype]
) raises where dtype.is_floating_point():
    # logb reads subnormals from their bits, so a target that flushes
    # subnormal inputs (Metal fast math) must still return the true exponent.
    comptime I = _unsigned_integral_type_of[dtype]()
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    comptime sign_bit = Scalar[I](1) << Scalar[I](bit_width_of[dtype]() - 1)
    var inputs: List[Scalar[dtype]] = [tiny]
    var expected: List[Scalar[dtype]] = [tiny_logb]
    inputs.reserve(length)
    expected.reserve(length)
    var state = golden_ratio_u64
    while len(inputs) < length:
        var i = len(inputs) % mantissa_width
        var b = Scalar[I](1) << Scalar[I](i)
        var bits = b | (xorshift64(state).cast[I]() & (b - 1))
        if len(inputs) % 2 == 1:
            bits |= sign_bit
        inputs.append(bitcast[dtype](bits))
        expected.append(Scalar[dtype](emin - mantissa_width + i))
    var got = run_on_gpu[dtype, logb](ctx, inputs)
    for i in range(length):
        assert_equal(
            got[i],
            expected[i],
            String(t"gpu logb[{dtype}]({inputs[i]})"),
        )
        assert_equal(
            logb(inputs[i]),
            expected[i],
            String(t"host logb[{dtype}]({inputs[i]})"),
        )


def test_logb_subnormal() raises:
    with DeviceContext() as ctx:
        _test_logb_subnormal[.float32](ctx, 1.0e-40, -133.0)
        _test_logb_subnormal[.float32](
            ctx, bitcast[.float32](UInt32(1)), -149.0
        )
        comptime if not default_accelerator().is_apple_gpu():
            _test_logb_subnormal[.float64](ctx, 1.0e-310, -1030.0)
            _test_logb_subnormal[.float64](
                ctx, bitcast[.float64](f64_min_subnormal_bits), -1074.0
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
