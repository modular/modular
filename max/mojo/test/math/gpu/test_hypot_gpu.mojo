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

from std.math import hypot, inf, isinf, isnan, nan
from std.sys import simd_width_of
from std.utils.numerics import max_finite

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from std.testing import TestSuite, assert_equal, assert_raises
from test_utils.math import (
    PairMode,
    hypot_specials,
    libm_call_binary,
    pair_inputs,
)

from std.utils.coord import Coord

comptime length = 8192
comptime seed = 1234


def _check[
    dtype: DType, //, max_ulp: Int
](
    got: Scalar[dtype],
    expected: Scalar[dtype],
    x: Scalar[dtype],
    y: Scalar[dtype],
    side: StaticString,
    i: Int,
) raises -> Int:
    var ok: Bool
    var d = 0
    var got_bits = Int(got._to_bits_signed())
    var expected_bits = Int(expected._to_bits_signed())
    if isnan(expected):
        ok = isnan(got)
    elif isinf(expected):
        ok = got_bits == expected_bits and got_bits > 0
    elif isnan(got) or isinf(got) or got_bits < 0 or expected_bits < 0:
        ok = False
    elif expected_bits == 0:
        ok = got_bits == 0
    else:
        # Nonnegative finite encodings cannot overflow this signed distance.
        d = abs(got_bits - expected_bits)
        ok = d <= max_ulp
    if not ok:
        raise Error(
            t"{side} hypot {dtype} at {i}: ({x}, {y}) got {got} expected",
            t" {expected} ({d} ulp)",
        )
    return d


def run_hypot[
    dtype: DType, //, max_ulp: Int
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
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, hypot(a, b))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var worst = 0
    var worst_host = 0
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = x_host[i]
            var y = y_host[i]
            var expected = libm_call_binary["hypotf", "hypot"](x, y)
            worst = max(
                worst,
                _check[max_ulp](
                    out_host[i],
                    expected,
                    x,
                    y,
                    "gpu",
                    i,
                ),
            )
            worst_host = max(
                worst_host,
                _check[max_ulp](hypot(x, y), expected, x, y, "host", i),
            )
    print(t"hypot {dtype} max ulp gpu: {worst} host: {worst_host}")


def _fill_pairs[
    dtype: DType
](
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
    mode: PairMode,
    seed: UInt64,
) raises where dtype.is_floating_point():
    """Fills x and y with the `pair_inputs` for `mode`."""
    var pairs = pair_inputs(hypot_specials[dtype](), mode, seed, length)
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host:
        for i in range(length):
            x_host[i] = pairs[0][i]
            y_host[i] = pairs[1][i]


def _run_all[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var x = ctx.enqueue_create_buffer[dtype](length)
    var y = ctx.enqueue_create_buffer[dtype](length)
    # Compare representable steps to the rounded libm oracle, not real-valued
    # ULP error. Narrow types can round the float32 paths to adjacent values.
    comptime hypot_ulp = 2 if dtype in (DType.float32, DType.float64) else 1
    var modes: List[PairMode] = [.random_bits, .integer_y, .small_quotients]
    for i, mode in enumerate(modes):
        _fill_pairs(x, y, mode, UInt64(seed + i))
        run_hypot[hypot_ulp](ctx, x, y)


def _test_oracle[dtype: DType]() raises:
    def check(got: Scalar[dtype], expected: Scalar[dtype]) raises -> Int:
        return _check[2](got, expected, 0, 0, "oracle", 0)

    var big = max_finite[dtype]()
    var infinity = inf[dtype]()
    var snan = Scalar[dtype](from_bits=infinity._to_bits_signed() | 1)
    with assert_raises():
        _ = check(infinity, big)
    with assert_raises():
        _ = check(big, infinity)
    with assert_raises():
        _ = check(snan, infinity)
    with assert_raises():
        _ = check(snan, big)
    with assert_raises():
        _ = check(Scalar[dtype](-0.0), Scalar[dtype](0.0))
    with assert_raises():
        _ = check(Scalar[dtype](-2.0), Scalar[dtype](2.0))
    var adjacent = Scalar[dtype](
        from_bits=Scalar[dtype](1)._to_bits_signed() + 1
    )
    assert_equal(check(adjacent, Scalar[dtype](1)), 1)
    assert_equal(check(infinity, infinity), 0)
    assert_equal(check(snan, nan[dtype]()), 0)


def test_oracle() raises:
    _test_oracle[.float32]()
    _test_oracle[.float64]()
    _test_oracle[.float16]()
    _test_oracle[.bfloat16]()


def test_hypot() raises:
    with DeviceContext() as ctx:
        _run_all[.float32](ctx)
        _run_all[.float16](ctx)
        _run_all[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run_all[.float64](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
