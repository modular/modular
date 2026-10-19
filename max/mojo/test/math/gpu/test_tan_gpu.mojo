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

# Checks `tan` on GPU and on host against libm called directly. Huge inputs use
# the same ULP bound as small ones: both libm and `tan` reduce exactly
# (Payne-Hanek), so a looser bound there would only hide a reduction bug.

from std.math import align_up, exp2, inf, isinf, isnan, nan, pi, tan
from std.builtin.dtype import _integral_type_of
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from std.utils.coord import Coord
from std.utils.numerics import FPUtils

comptime max_ulp_wide = 2.0
comptime inv_2_pow_53 = 1.0 / 9007199254740992.0
# f16/bf16 `tan` and libm both round tanf, so they differ by at most a
# double rounding of a few-ulp float32 difference.
comptime max_ulp_half = 1.0
# |x| above which `tan` switches from Cody-Waite to Payne-Hanek reduction.
comptime f32_reduction_threshold = 24576.0
comptime f64_reduction_threshold = 1572864.0
comptime f32_max_exp = 127.99
comptime f64_max_exp = 1023.99


def _step[
    dtype: DType, //
](x: Scalar[dtype], n: Int) -> Scalar[dtype] where dtype.is_floating_point():
    """Moves a non-negative finite x by n representable values."""
    comptime int_dtype = _integral_type_of[dtype]()
    return bitcast[dtype](bitcast[int_dtype](x) + Scalar[int_dtype](n))


def _ulp_error[
    dtype: DType, //
](
    got: Scalar[dtype], expected: Scalar[dtype]
) -> Float64 where dtype.is_floating_point():
    if isnan(expected) or isnan(got):
        return 0.0 if isnan(expected) and isnan(got) else inf[.float64]()
    if got == expected:
        # Equal zeros must also agree in sign.
        if got == 0 and (1.0 / got.cast[.float64]()) != (
            1.0 / expected.cast[.float64]()
        ):
            return inf[.float64]()
        return 0.0
    if isinf(got) or isinf(expected):
        return inf[.float64]()
    var diff = abs(got.cast[.float64]() - expected.cast[.float64]())
    # `ulp` relies on nextafter, which has no half-precision overload.
    var mag = abs(expected)
    var spacing: Float64
    if mag == Scalar[dtype].MAX_FINITE:
        spacing = mag.cast[.float64]() - _step(mag, -1).cast[.float64]()
    else:
        spacing = _step(mag, 1).cast[.float64]() - mag.cast[.float64]()
    return diff / spacing


def _check_ulp_error[dtype: DType]() raises where dtype.is_floating_point():
    var ni = -inf[dtype]()
    assert_equal(_ulp_error(Scalar[dtype](0), ni), inf[.float64]())
    assert_equal(_ulp_error(inf[dtype](), ni), inf[.float64]())
    assert_equal(_ulp_error(Scalar[dtype].MAX_FINITE, ni), inf[.float64]())
    assert_equal(
        _ulp_error(inf[dtype](), Scalar[dtype].MAX_FINITE), inf[.float64]()
    )
    assert_equal(_ulp_error(ni, ni), Float64(0))
    assert_equal(_ulp_error(nan[dtype](), ni), inf[.float64]())
    assert_equal(
        _ulp_error(Scalar[dtype](0), -Scalar[dtype](0)), inf[.float64]()
    )
    assert_equal(
        _ulp_error(
            _step(Scalar[dtype].MAX_FINITE, -1), Scalar[dtype].MAX_FINITE
        ),
        Float64(1),
    )


def test_ulp_error() raises:
    _check_ulp_error[.float16]()
    _check_ulp_error[.bfloat16]()
    _check_ulp_error[.float32]()
    _check_ulp_error[.float64]()


def _check[
    dtype: DType, //
](
    ctx: DeviceContext,
    name: String,
    inputs: List[Scalar[dtype]],
    max_ulp: Float64,
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var n = len(inputs)
    var padded = align_up(n, pack_size)

    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = inputs[i] if i < n else Scalar[dtype](0)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, tan(val))

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    var worst_gpu = 0.0
    var worst_gpu_x = Scalar[dtype](0)
    var worst_host = 0.0
    with out_device.map_to_host() as out_host:
        for i in range(n):
            var x = inputs[i]
            var expected = _call_libm["tan"](x)
            var got = out_host[i]
            var host = tan(x)
            var e_gpu = _ulp_error(got, expected)
            var e_host = _ulp_error(host, expected)
            if e_gpu > worst_gpu:
                worst_gpu = e_gpu
                worst_gpu_x = x
            worst_host = max(worst_host, e_host)
            if e_gpu > max_ulp or e_host > max_ulp:
                print(
                    t"tan mismatch {dtype} {name} x = {x} gpu = {got}",
                    t"host = {host} libm = {expected}",
                )
    print(
        t"tan {dtype} {name} n = {n} max ulp gpu = {worst_gpu} at",
        t"{worst_gpu_x} host = {worst_host}",
    )
    assert_true(
        worst_gpu <= max_ulp and worst_host <= max_ulp,
        String(t"tan {dtype} {name} exceeded {max_ulp} ulp"),
    )


def _lcg(mut state: UInt64) -> Float64:
    """Returns a uniform value in [0, 1) from a Knuth MMIX LCG step."""
    state = state * 6364136223846793005 + 1442695040888963407
    return Float64(state >> 11) * inv_2_pow_53


def _specials[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    var tiny = _step(Scalar[dtype](0), 1)
    return [
        Scalar[dtype](0),
        -Scalar[dtype](0),
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
        Scalar[dtype].MAX_FINITE,
        -Scalar[dtype].MAX_FINITE,
        Scalar[dtype].MIN_FINITE,
        tiny,
        -tiny,
        Scalar[dtype](1e-30),
        Scalar[dtype](1e-3),
        Scalar[dtype](pi / 4),
        -Scalar[dtype](pi / 4),
        Scalar[dtype](pi / 2),
        -Scalar[dtype](pi / 2),
    ]


def _dense[
    dtype: DType
](lo: Float64, hi: Float64, count: Int) -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    var v = List[Scalar[dtype]]()
    for i in range(count):
        v.append(
            Scalar[dtype](lo + (hi - lo) * Float64(i) / Float64(count - 1))
        )
    return v^


def _near_pio2_multiples[
    dtype: DType
](count: Int, scale: Float64) -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    var v = List[Scalar[dtype]]()
    for i in range(1, count + 1):
        var c = Scalar[dtype](Float64(i) * scale * (pi / 2))
        v.append(c)
        v.append(_step(c, -1))
        v.append(_step(c, 1))
        v.append(_step(c, -2))
        v.append(_step(c, 2))
        v.append(-c)
    return v^


def _huge[
    dtype: DType
](min_exp: Float64, max_exp: Float64, count: Int) -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Returns log-uniform magnitudes with scrambled mantissas, both signs."""
    var v = List[Scalar[dtype]]()
    var state: UInt64 = 42
    for _ in range(count):
        var e = min_exp + (max_exp - min_exp) * _lcg(state)
        var mag = Scalar[dtype](exp2(e))
        v.append(mag if _lcg(state) < 0.5 else -mag)
    return v^


def _test_wide[
    dtype: DType
](
    ctx: DeviceContext, max_ulp: Float64, threshold: Float64, max_exp: Float64
) raises where dtype.is_floating_point():
    _check(ctx, "specials", _specials[dtype](), max_ulp)
    _check(ctx, "[-100, 100]", _dense[dtype](-100, 100, 40001), max_ulp)
    _check(ctx, "[-pi/2, pi/2]", _dense[dtype](-1.6, 1.6, 8193), max_ulp)
    _check(ctx, "near k*pi/2", _near_pio2_multiples[dtype](3000, 1.0), max_ulp)
    _check(
        ctx,
        "near k*pi/2 (large k)",
        _near_pio2_multiples[dtype](2000, 997.0),
        max_ulp,
    )
    _check(
        ctx,
        "around reduction switch",
        _dense[dtype](threshold * 0.5, threshold * 2.0, 8193),
        max_ulp,
    )
    _check(ctx, "huge", _huge[dtype](3.0, max_exp, 20000), max_ulp)


def test_tan_float32() raises:
    with DeviceContext() as ctx:
        _test_wide[.float32](
            ctx, max_ulp_wide, f32_reduction_threshold, f32_max_exp
        )


def test_tan_float64() raises:
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            _test_wide[.float64](
                ctx, max_ulp_wide, f64_reduction_threshold, f64_max_exp
            )


def _check_hard_cases[
    dtype: DType
](
    ctx: DeviceContext,
    inputs: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
) raises where dtype.is_floating_point():
    """Checks inputs that once had more than 1.5 ulp of error against their
    correctly rounded tan (mpmath), to within 1 ulp on GPU and host."""
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var n = len(inputs)
    var padded = align_up(n, pack_size)
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = inputs[i] if i < n else Scalar[dtype](0)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, tan(val))

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    with out_device.map_to_host() as out_host:
        for i in range(n):
            var e_gpu = _ulp_error(out_host[i], want[i])
            var e_host = _ulp_error(tan(inputs[i]), want[i])
            assert_true(
                e_gpu <= 1.0 and e_host <= 1.0,
                String(
                    t"tan {dtype} x = {inputs[i]} gpu = {out_host[i]}",
                    t" host = {tan(inputs[i])} want {want[i]}",
                ),
            )


def test_tan_hard_cases() raises:
    with DeviceContext() as ctx:
        _check_hard_cases[.float32](
            ctx,
            [-3992.179, -1541.7366943359375, -5.193172931671143],
            [0.999819, 0.99980086, 1.9171497],
        )
        comptime if not default_accelerator().is_apple_gpu():
            _check_hard_cases[.float64](
                ctx,
                [1.3745834964292172e291, 7.386753707202125],
                [1.9923528895765197, 1.9822256111149423],
            )


def _test_half[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    _check(ctx, "specials", _specials[dtype](), max_ulp_half)
    _check(ctx, "[-100, 100]", _dense[dtype](-100, 100, 8001), max_ulp_half)
    _check(
        ctx, "near k*pi/2", _near_pio2_multiples[dtype](200, 1.0), max_ulp_half
    )


def _subnormals[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    var top = (1 << FPUtils[dtype].mantissa_width()) - 1
    var v = List[Scalar[dtype]]()
    for k in [1, 2, 3, top // 7, top // 2, top // 2 + 1, top - 1, top]:
        v.append(_step(Scalar[dtype](0), k))
        v.append(-_step(Scalar[dtype](0), k))
    return v^


def test_tan_subnormals() raises:
    # tan(x) == x for subnormal x, so the result must keep the input's bits
    # even on targets whose float ops flush subnormals.
    with DeviceContext() as ctx:
        _check(ctx, "subnormals", _subnormals[.float16](), 0.0)
        _check(ctx, "subnormals", _subnormals[.bfloat16](), 0.0)
        _check(ctx, "subnormals", _subnormals[.float32](), 0.0)


def test_tan_float16() raises:
    with DeviceContext() as ctx:
        _test_half[.float16](ctx)
        _check(
            ctx, "signed half overflow", [Float16(177.5), Float16(-177.5)], 0.0
        )


def test_tan_bfloat16() raises:
    with DeviceContext() as ctx:
        _test_half[.bfloat16](ctx)


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    # Apple GPUs have no float64 support; the body is empty there, so report
    # the test as skipped rather than passed.
    comptime if default_accelerator().is_apple_gpu():
        suite.skip[test_tan_float64]()
    suite^.run()
