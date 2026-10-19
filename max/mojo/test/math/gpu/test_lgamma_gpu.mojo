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

from std.math import frexp, inf, isinf, isnan, lgamma, max, nan
from std.math.math import _call_libm
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from std.testing import TestSuite
from test_utils.math import (
    lgamma_bit_refs,
    lgamma_zeros_f32,
    lgamma_zeros_f64,
    within_two_ulp,
)

from std.utils.coord import Coord

comptime length = 1 << 16
comptime golden_ratio_conjugate = 0.6180339887498949


def _ulp(v: Float64, mantissa_bits: Int) -> Float64:
    if v == 0:
        return 0.0
    var e = frexp(abs(v))[1]
    return Float64(2.0) ** (Float64(e) - Float64(mantissa_bits + 1))


def _fill_inputs[dtype: DType](mut host: DeviceBuffer[dtype]) raises:
    comptime n = length // 8
    with host.map_to_host() as h:
        for j in range(2 * n):
            # Step 40 / 2**14 is exact, so every pole in [-40, 0] is hit.
            h[j] = Scalar[dtype](-40.0 + 40.0 * Float64(j) / Float64(2 * n))
        for j in range(2 * n):
            h[2 * n + j] = Scalar[dtype](10.0 * Float64(j + 1) / Float64(2 * n))
        for j in range(n):
            # Dense around the zeros at 1 and 2 and the minimum near 1.46.
            h[4 * n + j] = Scalar[dtype](0.5 + 2.0 * Float64(j) / Float64(n))
        for j in range(n):
            h[5 * n + j] = Scalar[dtype](
                Float64(10.0) ** (-30.0 + 60.0 * Float64(j) / Float64(n))
            )
        for j in range(n):
            h[6 * n + j] = Scalar[dtype](
                -(Float64(10.0) ** (1.6 + 6.0 * Float64(j) / Float64(n)))
            )
        for j in range(n):
            var u = Float64(j) * golden_ratio_conjugate
            h[7 * n + j] = Scalar[dtype](-8.0 + 16.0 * (u - Float64(Int(u))))

        var specials: List[Float64] = [
            0.0,
            -0.0,
            1.0,
            2.0,
            -1.0,
            -2.0,
            3.0,
            0.5,
            -0.5,
            -2.5,
            1.4616321449683622,
            1e-8,
            -1e-8,
            1e-30,
            -1e-30,
            1e30,
            -1e30,
            -4503599627370495.5,
            -8388607.5,
            8.0,
            7.999999,
            0.9,
            1.7316,
            1.2316,
            0.7316,
            0.2316,
        ]
        var base = length - len(specials) - 4
        for i in range(len(specials)):
            h[base + i] = Scalar[dtype](specials[i])
        h[length - 4] = inf[dtype]()
        h[length - 3] = -inf[dtype]()
        h[length - 2] = nan[dtype]()
        h[length - 1] = Scalar[dtype].MAX_FINITE


# (x, lgamma(x)) next to negative integers and at half-integers.
def _near_f32() -> List[Tuple[Float64, Float64]]:
    return [
        (-0.5, 1.2655121088027954),
        (-2.5, -0.056243717670440674),
        (-100.5, -364.9009704589844),
        (-7.000202178955078, -0.019211480394005775),
        (-8.99999713897705, -0.037489715963602066),
        (-0.9999989867210388, 13.802319526672363),
        (-2.0000009536743164, 13.169795989990234),
        (-2.999000072479248, 5.117325782775879),
        (-6.999000072479248, -1.6153165102005005),
        (-7.000000953674316, 5.337780475616455),
        (-9.999999046325684, -1.241466760635376),
        (-14.00100040435791, -18.286544799804688),
    ]


def _near_f64() -> List[Tuple[Float64, Float64]]:
    return [
        (-0.5, 1.2655121234846454),
        (-2.5, -0.056243716497674054),
        (-100.5, -364.90096830942736),
        (-7.0002022, -0.019315607864957115),
        (-13.999999999988846, 0.028024653614358217),
        (-16.999999999999996, -0.23400878325950386),
        (-1.000000000001, 27.630932219297208),
        (-2.999999, 12.023752344715612),
        (-5.001, 2.1185589728099194),
        (-6.999999999999, 19.10577085823423),
        (-10.000001, -1.2889043661138306),
        (-13.999999999999, 2.439711036561623),
    ]


def _run_refs[
    input_dtype: DType, //, dtype: DType, bits_input: Bool = False
](
    ctx: DeviceContext, refs: List[Tuple[Scalar[input_dtype], Float64]]
) raises where dtype.is_floating_point():
    """Checks GPU lgamma against `refs` within 2 ulp."""
    comptime bits_dtype = DType.uint32 if dtype == .float32 else DType.uint64
    var n = len(refs)
    var in_device = ctx.enqueue_create_buffer[dtype](n)
    var out_device = ctx.enqueue_create_buffer[dtype](n)
    with in_device.map_to_host() as h:
        for i in range(n):
            comptime if bits_input:
                h[i] = Scalar[dtype](from_bits=refs[i][0].cast[bits_dtype]())
            else:
                h[i] = Scalar[dtype](refs[i][0])

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=n)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=n)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, lgamma(val))

    elementwise[1, target="gpu"](func, Coord(n), ctx)

    comptime mbits = DType.mantissa_width[dtype]()
    with out_device.map_to_host() as out_host:
        for i in range(n):
            var want = Scalar[dtype](refs[i][1])
            var got = out_host[i]
            if not within_two_ulp(got, want):
                raise Error(
                    (
                        t"gpu {dtype}: lgamma(input={refs[i][0]}) = {got},"
                        t" exact {want}"
                    ),
                )
            var err = abs(got.cast[.float64]() - refs[i][1])
            var tol = 2 * _ulp(refs[i][1], mbits)
            if not (err <= tol):
                raise Error(t"gpu {dtype}: error {err} exceeds {tol}")


@fieldwise_init
struct _ErrStats(Copyable, Writable):
    var pos_ulp: Float64
    var neg_ulp: Float64
    var near_zero_abs: Float64


def _check[
    dtype: DType
](
    x: Scalar[dtype],
    got: Scalar[dtype],
    expected: Scalar[dtype],
    what: String,
    mut stats: _ErrStats,
) raises:
    if isnan(expected) or isinf(expected):
        if not ((isnan(expected) and isnan(got)) or got == expected):
            raise Error(t"{what}: lgamma({x}) = {got}, libm {expected}")
        return

    if (
        isinf(got)
        or isnan(got)
        or (got._to_bits_signed() < 0) != (expected._to_bits_signed() < 0)
    ):
        raise Error(t"{what}: lgamma({x}) = {got}, libm {expected}")

    # Tolerances are in ulps of the libm value, which is itself off by a few
    # ulp on the negative axis. Positive x within 0.25 of the zeros at 1 and 2
    # adds an absolute floor. Everywhere else, and for negative x, there is
    # none: the reflection is computed in
    # double-word arithmetic, and the negative zeros are checked against
    # exact references in `_run_refs`.
    comptime mbits = DType.mantissa_width[dtype]()
    var neg = x < 0
    var near_pos_zero = abs(x - 1) < 0.25 or abs(x - 2) < 0.25
    var k: Float64
    var atol: Float64
    comptime if dtype == .float64:
        k = 8.0 if neg else 4.0
        atol = 4e-16 if near_pos_zero else 0.0
    elif dtype == .float32:
        k = 8.0 if neg else 4.0
        atol = 1.5e-7 if near_pos_zero else 0.0
    else:
        k = 1.0
        atol = 1e-6

    var e = expected.cast[.float64]()
    var err = abs(got.cast[.float64]() - e)
    var ulp = _ulp(e, mbits)
    var tol = k * ulp + atol
    if abs(e) < 0.5:
        stats.near_zero_abs = max(stats.near_zero_abs, err)
    elif neg:
        stats.neg_ulp = max(stats.neg_ulp, err / ulp)
    else:
        stats.pos_ulp = max(stats.pos_ulp, err / ulp)
    if not (err <= tol):
        raise Error(
            t"{what}: lgamma({x}) = {got}, libm {expected},",
            t" err {err} > tol {tol}",
        )


def _run[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    _fill_inputs(in_device)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, lgamma(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var gpu_stats = _ErrStats(0, 0, 0)
    var host_stats = _ErrStats(0, 0, 0)
    var gpu_what = String(t"gpu {dtype}")
    var host_what = String(t"host {dtype}")
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = in_host[i]
            var expected = _call_libm["lgamma"](x)
            _check(x, out_host[i], expected, gpu_what, gpu_stats)
            _check(x, lgamma(x), expected, host_what, host_stats)
    print(dtype, "gpu", gpu_stats, "host", host_stats)


def _run_half_subnormals[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime bits_dtype = DType.uint16
    comptime max_sub = UInt16(0x007F if dtype == .bfloat16 else 0x03FF)
    var tops: List[UInt16] = [1, 2, 16, 64, max_sub]
    var refs = List[Tuple[Scalar[dtype], Float64]]()
    for b in tops:
        for sign in range(2):
            var bits = b | (UInt16(0x8000) if sign == 1 else UInt16(0))
            var x = Scalar[dtype](from_bits=bits.cast[bits_dtype]())
            refs.append((x, _call_libm["lgamma"](x.cast[.float64]())))
    _run_refs[dtype](ctx, refs)


def test_lgamma_gpu() raises:
    with DeviceContext() as ctx:
        _run_half_subnormals[.float16](ctx)
        _run_half_subnormals[.bfloat16](ctx)
        _run[.float32](ctx)
        _run_refs[.float32](ctx, lgamma_zeros_f32())
        _run_refs[.float32](ctx, _near_f32())
        _run_refs[.float32, True](ctx, lgamma_bit_refs[.float32]())
        _run[.float16](ctx)
        _run[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run[.float64](ctx)
            _run_refs[.float64](ctx, lgamma_zeros_f64())
            _run_refs[.float64](ctx, _near_f64())
            _run_refs[.float64, True](ctx, lgamma_bit_refs[.float64]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
