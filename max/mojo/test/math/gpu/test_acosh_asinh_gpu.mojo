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

from std.math import acosh, asinh, exp, inf, isinf, isnan, nan
from std.math.math import _call_libm, _log_accurate
from std.memory import bitcast
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, HostBuffer, default_accelerator
from std.testing import assert_true, TestSuite

from std.utils.coord import Coord
from std.utils.numerics import FPUtils, max_finite

comptime length = 16384
comptime num_specials = 32
comptime LN_10 = 2.302585092994045684
# f16/bf16 compute in f32 and round once more, so they get one ULP.
comptime max_ulp_wide = 2
comptime max_ulp_narrow = 1


def _is_tiny_identity[
    dtype: DType
](x: Scalar[dtype]) -> Bool where dtype.is_floating_point():
    comptime FP = FPUtils[dtype]
    comptime magnitude_mask = (
        UInt64(1) << UInt64(bit_width_of[dtype]() - 1)
    ) - 1
    var magnitude = FP.bitcast_to_uint(x).cast[.uint64]() & magnitude_mask
    comptime tiny = Scalar[dtype](
        3.725290298461914e-09 if dtype == .float64 else 2.44140625e-04
    )
    return (
        magnitude > 0 and magnitude < FP.bitcast_to_uint(tiny).cast[.uint64]()
    )


def _ulp_diff[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Int where dtype.is_floating_point():
    if isnan(a) or isnan(b):
        return 0 if isnan(a) and isnan(b) else Int.MAX
    if isinf(a) or isinf(b):
        return 0 if a == b else Int.MAX
    # A ULP budget must not hide a wrong sign on zeros or tiny values.
    if FPUtils[dtype].get_sign(a) != FPUtils[dtype].get_sign(b):
        return Int.MAX
    var a_bits = FPUtils[dtype].bitcast_to_uint(a).cast[.uint64]()
    var b_bits = FPUtils[dtype].bitcast_to_uint(b).cast[.uint64]()
    # Same-sign finite encodings differ by less than Int.MAX, even for f64.
    return Int(max(a_bits, b_bits) - min(a_bits, b_bits))


def _check_comparator[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    var infinity = inf[dtype]()
    var largest = max_finite[dtype]()
    assert_true(_ulp_diff(infinity, infinity) == 0)
    assert_true(_ulp_diff(-infinity, -infinity) == 0)
    assert_true(_ulp_diff(nan[dtype](), nan[dtype]()) == 0)
    assert_true(_ulp_diff(largest, infinity) == Int.MAX)
    assert_true(_ulp_diff(infinity, largest) == Int.MAX)
    assert_true(_ulp_diff(-largest, -infinity) == Int.MAX)
    assert_true(_ulp_diff(infinity, -infinity) == Int.MAX)
    assert_true(_ulp_diff(nan[dtype](), T(1)) == Int.MAX)
    assert_true(_ulp_diff(T(1), nan[dtype]()) == Int.MAX)
    assert_true(_ulp_diff(T(0), T(-0.0)) == Int.MAX)
    assert_true(_ulp_diff(largest, -largest) == Int.MAX)
    assert_true(_ulp_diff(T(1), T(1)) == 0)
    var one_bits = FPUtils[dtype].bitcast_to_integer(T(1))
    var next_one = FPUtils[dtype].bitcast_from_integer(one_bits + 1)
    assert_true(_ulp_diff(T(1), next_one) == 1)
    comptime FP = FPUtils[dtype]
    comptime minimum_normal_bits = 1 << FP.mantissa_width()
    var tiny_magnitudes: List[Int] = [
        1,
        minimum_normal_bits // 2,
        minimum_normal_bits - 1,
        minimum_normal_bits,
    ]
    for bits in tiny_magnitudes:
        var x = FP.bitcast_from_integer(bits)
        assert_true(_is_tiny_identity(x))
        assert_true(_is_tiny_identity(FP.set_sign(x, True)))
        assert_true(_ulp_diff(x, FP.set_sign(x, True)) == Int.MAX)
    assert_true(not _is_tiny_identity(T(0)))
    assert_true(not _is_tiny_identity(T(1)))


def test_comparator() raises:
    _check_comparator[.float32]()
    _check_comparator[.float64]()
    _check_comparator[.float16]()
    _check_comparator[.bfloat16]()


def _exp10(e: Float64) -> Float64:
    return exp(e * LN_10)


def _fill_inputs[
    dtype: DType, is_acosh: Bool
](in_host: HostBuffer[dtype]) raises where dtype.is_floating_point():
    """Fills specials, then a linear, a log-scale, and a near-domain sweep."""
    comptime T = Scalar[dtype]
    comptime is_wide = dtype in (DType.float64, DType.float32)
    var specials: List[T] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
        max_finite[dtype](),
        -max_finite[dtype](),
        T(1.0e-30) if is_wide else T(1.0e-4),
        T(-1.0e-30) if is_wide else T(-1.0e-4),
        4096.0,
        268435456.0,
        1.0000001,
        0.9999999,
        3.0,
        10.0,
        -10.0,
    ]
    comptime FP = FPUtils[dtype]
    comptime minimum_normal_bits = 1 << FP.mantissa_width()
    var tiny_magnitudes: List[Int] = [
        1,
        minimum_normal_bits // 2,
        minimum_normal_bits - 1,
        minimum_normal_bits,
    ]
    for bits in tiny_magnitudes:
        var x = FP.bitcast_from_integer(bits)
        specials.append(x)
        specials.append(FP.set_sign(x, True))
    var n_spec = len(specials)
    assert_true(n_spec <= num_specials)
    for i in range(num_specials):
        in_host[i] = specials[i] if i < n_spec else T(1.5)

    # Largest finite decimal exponent of the dtype, for the log sweep.
    var max_e: Float64
    comptime if dtype == .float64:
        max_e = 307.0
    elif dtype in (DType.float32, DType.bfloat16):
        max_e = 38.0
    else:
        max_e = 4.7
    var min_e = -30.0 if is_wide else -4.0

    var seg = (length - num_specials) // 3
    for j in range(seg):
        var s = Float64(j) / Float64(seg)
        var idx = num_specials + j
        comptime if is_acosh:
            in_host[idx] = T(1.0 + 15.0 * s)
        else:
            in_host[idx] = T(-8.0 + 16.0 * s)
        # Log-scale sweep across the whole finite range.
        var mag = _exp10(min_e + (max_e - min_e) * s)
        comptime if is_acosh:
            in_host[idx + seg] = T(1.0 + mag)
        else:
            in_host[idx + seg] = T(mag) if j % 2 == 0 else T(-mag)
        # Near the hard spot: x just above 1 for acosh, |x| near 2 for asinh.
        comptime if is_acosh:
            in_host[idx + 2 * seg] = T(1.0 + _exp10(-17.0 + 17.5 * s))
        else:
            in_host[idx + 2 * seg] = T(-2.5 + 5.0 * s)
    for idx in range(num_specials + 3 * seg, length):
        in_host[idx] = T(0.25)


def _run[
    dtype: DType,
    is_acosh: Bool,
    max_ulp: Int,
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime name = "acosh" if is_acosh else "asinh"
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        _fill_inputs[dtype, is_acosh](in_host)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var result: type_of(val)
        comptime if is_acosh:
            result = acosh(val)
        else:
            result = asinh(val)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, result)

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var worst = 0
    var worst_x = Scalar[dtype](0)
    var worst_host = 0
    var worst_host_x = Scalar[dtype](0)
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = in_host[i]
            # libm is the reference; host std.math runs the same Mojo code
            # as the GPU, so it is checked against libm too.
            # Evaluating libm in float64 and rounding once keeps the reference
            # tighter than float32 libm, which is itself near 2 ulp.
            var expected: Scalar[dtype]
            var host: Scalar[dtype]
            comptime if is_acosh:
                expected = _call_libm["acosh"](x.cast[.float64]()).cast[dtype]()
                host = acosh(x)
            else:
                expected = _call_libm["asinh"](x.cast[.float64]()).cast[dtype]()
                host = asinh(x)
            var got = out_host[i]
            comptime if not is_acosh:
                # Check bits, not floating comparisons that can flush inputs.
                comptime FP = FPUtils[dtype]
                if _is_tiny_identity(x):
                    assert_true(
                        FP.bitcast_to_uint(got) == FP.bitcast_to_uint(x)
                        and FP.bitcast_to_uint(host) == FP.bitcast_to_uint(x),
                        msg=String(
                            t"asinh[{dtype}] tiny identity failed at x={x}"
                        ),
                    )
            var d = _ulp_diff(got, expected)
            var d_host = _ulp_diff(host, expected)
            if d > worst:
                worst = d
                worst_x = x
            if d_host > worst_host:
                worst_host = d_host
                worst_host_x = x
            assert_true(
                d_host <= max_ulp,
                msg=String(
                    t"host {name}[{dtype}] x={x} mojo={host} libm={expected}"
                    t" ulp={d_host}"
                ),
            )
            # Signed zeros must match exactly, including sign.
            if expected == 0:
                var e_bits = bitcast[.uint64](expected.cast[.float64]())
                assert_true(
                    bitcast[.uint64](got.cast[.float64]()) == e_bits
                    and bitcast[.uint64](host.cast[.float64]()) == e_bits,
                    msg=String(
                        t"signed zero mismatch at x={x}: {got} vs {expected}"
                    ),
                )
            assert_true(
                d <= max_ulp,
                msg=String(
                    t"{name}[{dtype}] x={x} gpu={got} libm={expected} ulp={d}"
                ),
            )
    print(
        name,
        dtype,
        "max ulp vs libm: gpu",
        worst,
        "at x =",
        worst_x,
        "| host",
        worst_host,
        "at x =",
        worst_host_x,
    )


def _run_all_dtypes[is_acosh: Bool]() raises:
    with DeviceContext() as ctx:
        _run[.float32, is_acosh, max_ulp_wide](ctx)
        _run[.float16, is_acosh, max_ulp_narrow](ctx)
        _run[.bfloat16, is_acosh, max_ulp_narrow](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run[.float64, is_acosh, max_ulp_wide](ctx)


def _check_hard_cases[
    dtype: DType, is_acosh: Bool
](
    ctx: DeviceContext,
    inputs: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
) raises where dtype.is_floating_point():
    """Checks inputs that once had more than 1.5 ulp of error against their
    correctly rounded results (mpmath), to within 1 ulp on GPU and host."""
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var n = len(inputs)
    var padded = (n + pack_size - 1) // pack_size * pack_size
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = inputs[i] if i < n else Scalar[dtype](2)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var result: type_of(val)
        comptime if is_acosh:
            result = acosh(val)
        else:
            result = asinh(val)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, result)

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    with out_device.map_to_host() as out_host:
        for i in range(n):
            var x = inputs[i]
            var host = acosh(x) if is_acosh else asinh(x)
            assert_true(
                _ulp_diff(out_host[i], want[i]) <= 1
                and _ulp_diff(host, want[i]) <= 1,
                msg=String(
                    t"x={x} gpu={out_host[i]} host={host} want={want[i]}"
                ),
            )


def test_hard_cases() raises:
    with DeviceContext() as ctx:
        _check_hard_cases[.float32, True](
            ctx,
            [1.0070128440856934, 1.0001211166381836, 1.007747769355774],
            [0.118360996, 0.015563688, 0.12440085],
        )
        _check_hard_cases[.float32, False](
            ctx, [0.2525882124900818], [0.24997663]
        )
        comptime if not default_accelerator().is_apple_gpu():
            _check_hard_cases[.float64, True](
                ctx,
                [1.025717012207915, 1.0000303429856943, 1.0000001191542771],
                [
                    0.22630746557989662,
                    0.007790100416046396,
                    0.00048816856667639394,
                ],
            )


def _check_log_subnormals[
    dtype: DType
](
    ctx: DeviceContext,
    inputs: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
) raises where dtype.is_floating_point():
    comptime pack_size = 4
    var n = len(inputs)
    var padded = (n + pack_size - 1) // pack_size * pack_size
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = inputs[i] if i < n else Scalar[dtype](2)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, _log_accurate(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    with out_device.map_to_host() as out_host:
        for i in range(n):
            assert_true(
                _ulp_diff(out_host[i], want[i]) <= 1,
                msg=String(t"x={inputs[i]} gpu={out_host[i]} want={want[i]}"),
            )


def test_log_subnormals() raises:
    """The shared log reduction reads a subnormal's exponent from its bits, so
    the smallest and largest subnormal inputs survive flush-to-zero targets."""
    with DeviceContext() as ctx:
        _check_log_subnormals[.float32](
            ctx,
            [
                Float32(from_bits=Int32(1)),
                1,
                Float32(from_bits=Int32(0x007FFFFF)),
                2,
            ],
            [-103.278931, 0, -87.3365479, 0.693147182],
        )
        comptime if not default_accelerator().is_apple_gpu():
            _check_log_subnormals[.float64](
                ctx,
                [
                    Float64(from_bits=Int64(1)),
                    1,
                    Float64(from_bits=Int64(0x000FFFFFFFFFFFFF)),
                    2,
                ],
                [-744.4400719213812, 0, -708.3964185322641, 0.6931471805599453],
            )


def test_asinh() raises:
    _run_all_dtypes[False]()


def test_acosh() raises:
    _run_all_dtypes[True]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
