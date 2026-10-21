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

from std.math import isnan, remainder
from std.sys import simd_width_of
from std.utils.numerics import FPUtils

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from std.testing import TestSuite
from test_utils.math import (
    PairMode,
    libm_call_binary,
    pair_inputs,
    remainder_specials,
)

from std.utils.coord import Coord

comptime length = 8192
comptime seed = 1234


def _check[
    dtype: DType, //
](
    got: Scalar[dtype],
    expected: Scalar[dtype],
    x: Scalar[dtype],
    y: Scalar[dtype],
    side: StaticString,
    i: Int,
) raises:
    # remainder is exact, so it must match libm bit for bit; comparing bits
    # also distinguishes +0 from -0.
    var ok: Bool
    if isnan(expected):
        ok = isnan(got)
    else:
        ok = got.to_bits() == expected.to_bits()
    if not ok:
        raise Error(
            t"{side} remainder {dtype} at {i}: ({x}, {y}) got {got} expected",
            t" {expected}",
        )


def run_remainder[
    dtype: DType, //
](
    ctx: DeviceContext,
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()

    var out_device = ctx.enqueue_create_buffer[dtype](length)
    var x_buffer = Span(unsafe_ptr=x_device.unsafe_ptr(), length=length)
    var y_buffer = Span(unsafe_ptr=y_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var x_buffer, var y_buffer}:
        var idx = Int(idx0[0])
        var a = x_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var b = y_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, remainder(a, b)
        )

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = x_host[i]
            var y = y_host[i]
            var expected = libm_call_binary["remainderf", "remainder"](x, y)
            _check(out_host[i], expected, x, y, "gpu", i)
            _check(remainder(x, y), expected, x, y, "host", i)


def _fill_pairs[
    dtype: DType
](
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
    mode: PairMode,
    seed: UInt64,
) raises where dtype.is_floating_point():
    """Fills x and y with the `pair_inputs` for `mode`."""
    var pairs = pair_inputs(remainder_specials[dtype](), mode, seed, length)
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host:
        for i in range(length):
            x_host[i] = pairs[0][i]
            y_host[i] = pairs[1][i]


def _run_all[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var x = ctx.enqueue_create_buffer[dtype](length)
    var y = ctx.enqueue_create_buffer[dtype](length)
    var modes: List[PairMode] = [
        .random_bits,
        .integer_y,
        .small_quotients,
        .near_multiples,
    ]
    for i, mode in enumerate(modes):
        _fill_pairs(x, y, mode, UInt64(seed + i))
        run_remainder(ctx, x, y)


def test_remainder() raises:
    with DeviceContext() as ctx:
        _run_all[.float32](ctx)
        _run_all[.float16](ctx)
        _run_all[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run_all[.float64](ctx)


comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074
comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))


def _run_unary[
    dtype: DType, //, math_fn: UnaryFn
](ctx: DeviceContext, xs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Applies `math_fn` to `xs` on the GPU."""
    var n = len(xs)
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    # Pad to a whole number of packs so every lane holds a real input.
    var padded = ((n + pack_size - 1) // pack_size) * pack_size
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = xs[i] if i < n else xs[0]
    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    var out = List[Scalar[dtype]]()
    with out_device.map_to_host() as out_host:
        for i in range(n):
            out.append(out_host[i])
    return out^


def _check_exact[
    dtype: DType, //
](
    xs: List[Scalar[dtype]],
    got: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
    name: String,
) raises:
    for i in range(len(xs)):
        if got[i].to_bits() != want[i].to_bits():
            raise Error(t"gpu {name}({xs[i]}) = {got[i]}, expected {want[i]}")


def _remainder_by_min[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime y = SIMD[dtype, width](
        from_bits=SIMD[FPUtils[dtype].integral_type, width](1)
    )
    return remainder(x, y)


def _remainder_by_min_normal[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime y = SIMD[dtype, width](
        from_bits=SIMD[FPUtils[dtype].integral_type, width](
            1 << FPUtils[dtype].mantissa_width()
        )
    )
    return remainder(x, y)


def _remainder_of_one[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return remainder(SIMD[dtype, width](1.0), x)


# Remainders by a subnormal divisor, and of a subnormal dividend by a small
# divisor, have subnormal results; they must survive targets that flush
# subnormals (Apple GPUs in float32).
def test_remainder_subnormal_f32() raises:
    with DeviceContext() as ctx:
        var xs: List[Float32] = [f32_min, f32_min * 4, -f32_max_sub, 8.0]
        # Every input is a multiple of 2^-149, so each remainder is a zero of
        # the input's sign.
        _check_exact(
            xs,
            _run_unary[_remainder_by_min](ctx, xs),
            [0.0, 0.0, -0.0, 0.0],
            "remainder by 2^-149",
        )
        var positive: List[Float32] = [f32_min, f32_min * 4, 8.0]
        _check_exact(
            positive,
            _run_unary[_remainder_of_one](ctx, positive),
            [0.0, 0.0, 1.0],
            "remainder of 1.0",
        )
        # 2^149 mod (2^23 - 1) = 2^11, a subnormal result.
        var sub: List[Float32] = [f32_max_sub]
        _check_exact(
            sub,
            _run_unary[_remainder_of_one](ctx, sub),
            [f32_min * 2048],
            "remainder of 1.0 by the largest subnormal",
        )
        # Subnormal dividend with the smallest normal divisor: the remainder
        # is x itself, or x - 2^-126 when x is above half of it.
        var tiny_y: List[Float32] = [
            f32_min * 3,
            -f32_min * 5,
            f32_min * Float32(3 << 21),
        ]
        _check_exact(
            tiny_y,
            _run_unary[_remainder_by_min_normal](ctx, tiny_y),
            [f32_min * 3, -f32_min * 5, -f32_min * Float32(1 << 21)],
            "remainder by 2^-126",
        )


def test_remainder_subnormal_f64() raises:
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            var xs: List[Float64] = [f64_min, -f64_max_sub, 8.0]
            _check_exact(
                xs,
                _run_unary[_remainder_by_min](ctx, xs),
                [0.0, -0.0, 0.0],
                "remainder by 2^-1074",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
