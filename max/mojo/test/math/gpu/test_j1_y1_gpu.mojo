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

from std.math import exp, frexp, inf, isinf, isnan, j1, ldexp, nan, sqrt, y1
from std.math.math import _call_libm
from std.sys import simd_width_of
from std.utils.numerics import FPUtils

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from std.utils.coord import Coord

comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()


def _input[dtype: DType](i: Int) -> Scalar[dtype]:
    """Builds a dense linear sweep, a log-spaced sweep of both signs up to
    the dtype's range, and special values."""
    comptime num_linear = 4000
    comptime num_log = 4000
    comptime ln10 = 2.302585092994046
    if i < num_linear:
        return Scalar[dtype](-20.0 + Float64(i) * (120.0 / num_linear))
    if i < num_linear + num_log:
        var k = i - num_linear
        var lo = -37.0
        var hi = 38.0
        comptime if dtype == .float64:
            lo = -300.0
            hi = 307.0
        elif dtype == .float16:
            lo = -7.0
            hi = 4.8
        var mag = exp((lo + (hi - lo) * Float64(k) / num_log) * ln10)
        return Scalar[dtype](mag if k % 2 == 0 else -mag)
    var specials: List[Float64] = [
        0.0,
        -0.0,
        inf[.float64](),
        -inf[.float64](),
        nan[.float64](),
        2.0,
        -2.0,
        1.9999999,
        2.0000001,
        2.1971413260310170,  # first zero of y1
        3.8317059702075123,  # first zero of j1
        5.4296810407941351,
        7.0155866698156188,
        1e-30,
        3e-8,
        8.0,
        4.5454,
        2.8571,
        1e4,
        65504.0,
    ]
    var j = (i - num_linear - num_log) % len(specials)
    return Scalar[dtype](specials[j])


def _ulp[dtype: DType](v: Float64) -> Float64:
    """Returns the spacing of `dtype` values in the binade of `v`, in float64;
    below the normal range it is the smallest subnormal."""
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    var e = Int(frexp(abs(v))[1]) - 1 if v != 0 else emin
    return ldexp(Float64(1), Int32(max(e, emin) - mant))


def _err_ulp[
    dtype: DType
](got: Scalar[dtype], want: Scalar[dtype], x: Scalar[dtype]) -> Float64:
    """Returns |got - want| in `dtype` ulps of max(|want|, envelope).

    For |x| >= 1 the scale is at least the oscillation envelope
    sqrt(2 / (pi |x|)), so values near the zeros of j1 and y1, where the
    functions are ill-conditioned, are judged by absolute error.
    """
    var g = got.cast[.float64]()
    var w = want.cast[.float64]()
    if g == w:
        return 0.0
    comptime sqrt_two_over_pi = 0.7978845608028654
    var scale = abs(w)
    var ax = abs(x.cast[.float64]())
    if ax >= 1.0:
        scale = max(scale, sqrt_two_over_pi / sqrt(ax))
    return abs(g - w) / _ulp[dtype](scale)


def _check[
    dtype: DType, math_fn: UnaryFn, libm_name: StaticString
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    comptime length = 8192
    comptime full_max_ulp = 4.0
    # Half-precision results are an f32 result rounded once, so a 1 ulp
    # difference only reflects which side of a rounding boundary f32 landed.
    comptime half_max_ulp = 1.0
    comptime max_ulp = full_max_ulp if dtype in (
        DType.float32,
        DType.float64,
    ) else half_max_ulp

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = _input[dtype](i)
    var out_device = ctx.enqueue_create_buffer[dtype](length)

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

    var worst_gpu = 0.0
    var worst_host = 0.0
    var worst_x = 0.0
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = in_host[i]
            # A float64 reference avoids needing the host's j1f/y1f.
            var want = _call_libm[libm_name](x.cast[.float64]()).cast[dtype]()
            var got_gpu = out_host[i]
            var got_host = math_fn(x)
            var msg = String(
                t"{libm_name} {dtype} x={x} libm={want} gpu={got_gpu}",
                t" host={got_host}",
            )
            if isnan(want):
                assert_true(isnan(got_gpu), msg=msg)
                assert_true(isnan(got_host), msg=msg)
                continue
            if isinf(want) or want == 0:
                assert_equal(got_gpu, want, msg=msg)
                assert_equal(got_host, want, msg=msg)
                continue
            var e_gpu = _err_ulp(got_gpu, want, x)
            var e_host = _err_ulp(got_host, want, x)
            if e_gpu > worst_gpu:
                worst_gpu = e_gpu
                worst_x = x.cast[.float64]()
            worst_host = max(worst_host, e_host)
            assert_true(e_gpu <= max_ulp, msg=String(t"{msg} ulp={e_gpu}"))
            assert_true(e_host <= max_ulp, msg=String(t"{msg} ulp={e_host}"))
    print(
        libm_name,
        dtype,
        "max ulp gpu:",
        worst_gpu,
        "at x =",
        worst_x,
        "host:",
        worst_host,
    )


def _check_all[
    math_fn: UnaryFn, libm_name: StaticString
](ctx: DeviceContext) raises:
    _check[.float32, math_fn, libm_name](ctx)
    _check[.float16, math_fn, libm_name](ctx)
    _check[.bfloat16, math_fn, libm_name](ctx)
    comptime if not default_accelerator().is_apple_gpu():
        _check[.float64, math_fn, libm_name](ctx)


def test_j1() raises:
    with DeviceContext() as ctx:
        _check_all[j1, "j1"](ctx)


def test_y1() raises:
    with DeviceContext() as ctx:
        _check_all[y1, "y1"](ctx)


def _check_infinities[dtype: DType]() raises where dtype.is_floating_point():
    # The limits at both infinities are +0 (the comparison against libm
    # above cannot see the sign of a zero); y1 is undefined below 0.
    var pos_zero = Scalar[dtype](0).to_bits()
    assert_equal(j1(inf[dtype]()).to_bits(), pos_zero)
    assert_equal(j1(-inf[dtype]()).to_bits(), pos_zero)
    assert_equal(y1(inf[dtype]()).to_bits(), pos_zero)
    assert_true(isnan(y1(-inf[dtype]())))


def test_j1_y1_infinities() raises:
    _check_infinities[.float32]()
    _check_infinities[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
