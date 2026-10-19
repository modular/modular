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

from std.math import frexp, inf, isinf, isnan, j0, ldexp, nan, sqrt, y0
from std.math.math import _call_libm
from std.sys import simd_width_of
from std.utils.numerics import FPUtils

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.testing import TestSuite

from std.utils.coord import Coord

comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()


def _inputs() -> List[Float64]:
    var xs = List[Float64]()
    # Dense sweep over the oscillating region, both signs.
    for i in range(-4000, 4001):
        xs.append(Float64(i) * 0.0125 + 0.0003)
    # Geometric sweep across the whole float64 range.
    var x = 1.0e-300
    while x < 1.0e300:
        xs.append(x)
        x *= 1.37
    # Zeros of j0 and y0, branch thresholds, hard reduction cases, and special
    # values.
    var specials: List[Float64] = [
        0.0,
        0.8935769662791675,
        2.404825557695773,
        3.957678419314858,
        5.520078110286311,
        7.086051060301773,
        8.653727912911013,
        1.0,
        2.0,
        1.9999999,
        2.857,
        4.5454,
        2.8571414947509766,
        4.545452117919922,
        8.0,
        131071.9,
        131072.0,
        131072.1,
        1099511627775.0,
        1099511627776.0,
        1.0e22,
        5.0e15,
        6.5536e4,
        1.0e10,
        3.4e38,
        1.0e308,
        1.0e-8,
        1.0e-40,
        4.9e-324,
        inf[.float64](),
        nan[.float64](),
    ]
    for s in specials:
        xs.append(s)
        xs.append(-s)
    return xs^


def _ulp[dtype: DType](v: Float64) -> Float64:
    """Returns the spacing of `dtype` values in the binade of `v`, in float64;
    below the normal range it is the smallest subnormal."""
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    var e = Int(frexp(abs(v))[1]) - 1 if v != 0 else emin
    return ldexp(Float64(1), Int32(max(e, emin) - mant))


def _envelope(x: Float64) -> Float64:
    """Returns the magnitude scale of j0/y0 around x."""
    comptime two_over_pi = 0.6366197723675814
    comptime envelope_cutoff = 2.0
    var ax = abs(x)
    if ax < envelope_cutoff:
        return 1.0
    return sqrt(two_over_pi / ax)


def _error[
    dtype: DType
](
    x: Float64, got: Float64, want_f64: Float64
) raises -> Float64 where dtype.is_floating_point():
    """Returns the error of `got` against libm's float64 result `want_f64`.

    The error is in ulps of the result, except near zeros of the function,
    where it is ill-conditioned and the bound is in ulps of the envelope
    `sqrt(2 / (pi |x|))` instead: the smaller of the two is returned. Both
    use the true ulp spacing of `dtype`. Special values must match exactly.
    """
    var want = Scalar[dtype](want_f64).cast[.float64]()
    if isnan(want) or isinf(want) or isnan(got) or isinf(got):
        if (isnan(want) and isnan(got)) or got == want:
            return 0.0
        raise Error(t"special mismatch at x={x} got={got} want={want_f64}")
    var err = abs(got - want_f64)
    var ulp_err = err / _ulp[dtype](want_f64)
    var env_err = err / _ulp[dtype](_envelope(x))
    return min(ulp_err, env_err)


def _run[
    dtype: DType, math_fn: UnaryFn, name: StaticString
](ctx: DeviceContext, xs: List[Float64]) raises where dtype.is_floating_point():
    comptime full_max_ulp = 4.0
    # Half-precision results round a float32 evaluation once.
    comptime half_max_ulp = 1.0
    comptime tol = full_max_ulp if dtype in (
        DType.float32,
        DType.float64,
    ) else half_max_ulp
    var length = len(xs)
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Scalar[dtype](xs[i])

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
            var xd = x.cast[.float64]()
            var want = _call_libm[name](xd)
            var host = math_fn(x).cast[.float64]()
            var got = out_host[i]
            var gpu_out = got.cast[.float64]()
            var gpu_err = _error[dtype](xd, gpu_out, want)
            var host_err = _error[dtype](xd, host, want)
            if gpu_err > worst_gpu:
                worst_gpu = gpu_err
                worst_x = xd
            worst_host = max(worst_host, host_err)
            # Tests the passing condition so that a NaN error fails.
            if not (gpu_err <= tol and host_err <= tol):
                raise Error(
                    t"{name}[{dtype}] x={xd} gpu={got} host={host}",
                    t" libm={want} gpu_err={gpu_err} host_err={host_err}",
                    t" tol={tol}",
                )
    print(
        name,
        dtype,
        "max err (ulp, or envelope ulp near zeros): gpu=",
        worst_gpu,
        "at x=",
        worst_x,
        "host=",
        worst_host,
    )


def _run_j0_y0[
    dtype: DType
](ctx: DeviceContext, xs: List[Float64]) raises where dtype.is_floating_point():
    _run[dtype, j0, "j0"](ctx, xs)
    _run[dtype, y0, "y0"](ctx, xs)


def test_j0_y0_gpu() raises:
    var xs = _inputs()
    with DeviceContext() as ctx:
        _run_j0_y0[.float32](ctx, xs)
        _run_j0_y0[.float16](ctx, xs)
        _run_j0_y0[.bfloat16](ctx, xs)
        comptime if not ctx.target.is_apple_gpu():
            _run_j0_y0[.float64](ctx, xs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
