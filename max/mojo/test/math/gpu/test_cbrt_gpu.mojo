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

from std.math import cbrt, inf, isinf, isnan, ldexp, max, nan
from std.math import ulp as spacing
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import TestSuite, assert_equal, assert_true

from std.utils.coord import Coord

comptime num_sweep = 16384
comptime max_ulp_wide = 4
comptime max_ulp_narrow = 1
# The documented float32 maximum is 0.505 ulp from the true root.
comptime max_f32_err = 0.506


def _special_values[dtype: DType]() -> List[Scalar[dtype]]:
    return [
        Scalar[dtype](0.0),
        Scalar[dtype](-0.0),
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
        Scalar[dtype].MIN_FINITE,
        Scalar[dtype].MAX_FINITE,
        Scalar[dtype](1.0),
        Scalar[dtype](-1.0),
        Scalar[dtype](8.0),
        Scalar[dtype](-27.0),
        Scalar[dtype](0.001),
        Scalar[dtype](2.0),
        Scalar[dtype](3.0),
    ]


def _inputs[dtype: DType]() -> List[Scalar[dtype]]:
    """Returns special values, a log-uniform sweep, and a linear sweep."""
    var values = _special_values[dtype]()
    # The smallest subnormal and the largest finite exponent of each dtype.
    comptime e_min = -1074 if dtype == .float64 else (
        -149 if dtype == .float32 else (-133 if dtype == .bfloat16 else -24)
    )
    comptime e_max = 1023 if dtype == .float64 else (
        127 if dtype in (DType.float32, DType.bfloat16) else 15
    )
    for i in range(num_sweep):
        var e = e_min + (i * (e_max - e_min + 1)) // num_sweep
        comptime mantissa_scramble_prime = 7919
        var frac = Float64((i * mantissa_scramble_prime) % 1024) / 1024.0
        var v = ldexp(Float64(1.0) + frac, Int32(e))
        values.append((-v if i % 2 else v).cast[dtype]())
    for i in range(num_sweep):
        values.append((Float64(i) / num_sweep * 20.0 - 10.0).cast[dtype]())
    return values^


def _bits(v: Scalar) -> UInt64:
    return v.to_bits[.uint64]()


def _ulp_diff(got: Scalar, want: type_of(got)) -> UInt64:
    """Returns the ulp distance between two values of one dtype.

    Bit-identical values (NaN included) are 0 apart. A NaN, an infinity, or a
    sign mismatch on either side, other than +0 against -0, is `UInt64.MAX`,
    so no tolerance lets such a result through.
    """
    var a = _bits(got)
    var b = _bits(want)
    if a == b:
        return 0
    if isnan(got) or isnan(want) or isinf(got) or isinf(want):
        return UInt64.MAX
    comptime sign = UInt64(1) << UInt64(bit_width_of[got.dtype]() - 1)
    var mag_a = a & (sign - 1)
    var mag_b = b & (sign - 1)
    if (a & sign) != (b & sign):
        return 0 if (mag_a | mag_b) == 0 else UInt64.MAX
    return mag_a - mag_b if mag_a >= mag_b else mag_b - mag_a


def _check_cbrt[
    dtype: DType, max_ulp: UInt64
](ctx: DeviceContext) raises -> Int where dtype.is_floating_point():
    """Returns the number of inputs outside the tolerance."""
    var values = _inputs[dtype]()
    var length = len(values)
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = values[i]

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, cbrt(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var worst: UInt64 = 0
    var worst_x = Scalar[dtype](0)
    var worst_host: UInt64 = 0
    var worst_err = Float64(0)
    var failures = 0
    with out_device.map_to_host() as out_host:
        for i in range(length):
            var x = values[i]
            var got = out_host[i]
            # libm has no 16-bit entry points, so those dtypes are checked
            # against `cbrtf` rounded back to `dtype`.
            var want: Scalar[dtype]
            comptime if dtype in (DType.float32, DType.float64):
                want = _call_libm["cbrt"](x)
            else:
                want = _call_libm["cbrt"](x.cast[.float32]()).cast[dtype]()
            var host = cbrt(x)
            var ulp: UInt64 = 0
            var host_ulp: UInt64 = 0
            var bad: Bool
            if isnan(want):
                # NaN payloads differ between targets, so any NaN is right.
                bad = not (isnan(got) and isnan(host))
            else:
                # `_ulp_diff` rejects a NaN, an infinity, or a sign mismatch
                # outright; a zero result must also carry the input's sign.
                ulp = _ulp_diff(got, want)
                host_ulp = _ulp_diff(host, want)
                bad = not (ulp <= max_ulp and host_ulp <= max_ulp)
                if want == 0:
                    bad = bad or _bits(got) != _bits(want)
                    bad = bad or _bits(host) != _bits(want)
            comptime if dtype == .float32:
                # libm `cbrtf` is itself 1.3 ulp off, so also bound the error
                # against the float64 root, which is tight enough to see a
                # regression from 0.505 ulp.
                var root = _call_libm["cbrt"](x.cast[.float64]())
                if want != 0 and not isnan(want) and not isinf(want):
                    var err = abs(got.cast[.float64]() - root)
                    err /= spacing(got).cast[.float64]()
                    worst_err = max(worst_err, err)
                    bad = bad or not (err <= max_f32_err)
            if ulp > worst:
                worst = ulp
                worst_x = x
            worst_host = max(worst_host, host_ulp)
            if bad:
                if failures < 10:
                    print(
                        dtype,
                        t"cbrt({x}) = {got} (gpu), {host} (host),",
                        t"want {want} ({ulp}/{host_ulp} ulp)",
                    )
                failures += 1
    print(
        dtype,
        t"max ulp gpu {worst} at x = {worst_x}, host {worst_host},",
        t"max error vs true root {worst_err} ulp,",
        t"failures {failures}",
    )
    return failures


def test_ulp_diff() raises:
    """The comparator must not let a wrong sign, NaN, or infinity through."""
    comptime f32_inf = inf[.float32]()
    comptime f64_inf = inf[.float64]()
    assert_equal(_ulp_diff(Float32(1.0), Float32(1.0)), 0)
    assert_equal(_ulp_diff(nan[.float32](), nan[.float32]()), 0)
    assert_equal(_ulp_diff(Float32(0.0), Float32(-0.0)), 0)
    assert_equal(_ulp_diff(f32_inf, f32_inf), 0)
    assert_equal(_ulp_diff(Float32(-1.0), Float32(-1.0000001)), 1)
    assert_equal(_ulp_diff(Float32(1.0), Float32(0.99999994)), 1)
    assert_equal(_ulp_diff(Float32(-4.0), Float32(1.0)), UInt64.MAX)
    # A subnormal literal would flush to zero; build the value from bits.
    comptime neg_f32_min = Float32(from_bits=UInt32(0x80000001))
    assert_equal(_ulp_diff(neg_f32_min, Float32(0.0)), UInt64.MAX)
    assert_equal(_ulp_diff(f32_inf, Float32.MAX_FINITE), UInt64.MAX)
    assert_equal(_ulp_diff(nan[.float32](), Float32(1.0)), UInt64.MAX)
    assert_equal(_ulp_diff(Float32(1.0), nan[.float32]()), UInt64.MAX)
    assert_equal(_ulp_diff(f64_inf, Float64.MAX_FINITE), UInt64.MAX)
    assert_equal(
        _ulp_diff(bitcast[.float64](UInt64(0x7FF0000000000001)), f64_inf),
        UInt64.MAX,
    )
    assert_equal(_ulp_diff(Float64(-4.0), Float64(1.0)), UInt64.MAX)
    assert_equal(_ulp_diff(Float64.MAX_FINITE, -Float64.MAX_FINITE), UInt64.MAX)
    assert_equal(_ulp_diff(Float64(1.0), Float64(1.0000000000000002)), 1)
    assert_equal(_ulp_diff(Float16(1.0), Float16(1.0009765625)), 1)
    assert_equal(_ulp_diff(BFloat16(1.0), BFloat16(1.0078125)), 1)


def test_cbrt_gpu() raises:
    with DeviceContext() as ctx:
        var failures = _check_cbrt[.float32, max_ulp_narrow](ctx)
        failures += _check_cbrt[.float16, max_ulp_narrow](ctx)
        failures += _check_cbrt[.bfloat16, max_ulp_narrow](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            # glibc `cbrt` is itself up to ~2 ulp off (the musl algorithm used
            # here is within 1 ulp of the true root), so 4 ulp covers both.
            failures += _check_cbrt[.float64, max_ulp_wide](ctx)
        assert_equal(failures, 0)


def _from_bits[dtype: DType](bits: UInt64) -> Scalar[dtype]:
    comptime if dtype == .float64:
        return bitcast[.float64](bits)._refine[dtype]()
    return bitcast[.float32](UInt32(bits))._refine[dtype]()


def _check_mixed_lanes[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    """Checks special and subnormal lanes sharing one vector with normals."""
    comptime w = 8
    comptime is_f64 = dtype == .float64
    comptime mant = 52 if is_f64 else 23
    comptime bias = 1023 if is_f64 else 127
    comptime sign = UInt64(1) << UInt64(63 if is_f64 else 31)
    comptime exp_mask = UInt64((1 << (63 if is_f64 else 31)) - (1 << mant))
    comptime quiet_nan = exp_mask | (UInt64(1) << UInt64(mant - 1))
    # The subnormal lanes are perfect cubes: cbrt(2^-1074) = 2^-358 and
    # cbrt(2^-147) = 2^-49; the negative lane is three binades up.
    comptime tiny_bits = UInt64(1 if is_f64 else 4)
    comptime cube_e = (-1074 if is_f64 else -147) // 3
    # 0, -0, inf, -inf, nan, tiny, -8 * tiny, -27.
    var in_bits = SIMD[.uint64, w](
        0,
        sign,
        exp_mask,
        sign | exp_mask,
        quiet_nan,
        tiny_bits,
        sign | (tiny_bits * 8),
        0,
    )
    var want_bits = SIMD[.uint64, w](
        0,
        sign,
        exp_mask,
        sign | exp_mask,
        quiet_nan,
        UInt64(bias + cube_e) << UInt64(mant),
        sign | (UInt64(bias + cube_e + 1) << UInt64(mant)),
        0,
    )
    var x = SIMD[dtype, w]()
    var want = SIMD[dtype, w]()
    for i in range(w):
        x[i] = _from_bits[dtype](in_bits[i])
        want[i] = _from_bits[dtype](want_bits[i])
    x[7] = -27.0
    want[7] = -3.0

    var in_device = ctx.enqueue_create_buffer[dtype](w)
    var out_device = ctx.enqueue_create_buffer[dtype](w)
    with in_device.map_to_host() as in_host:
        for i in range(w):
            in_host[i] = x[i]

    def kernel(
        out_dev: Pointer[Scalar[dtype], MutAnyOrigin],
        in_dev: Pointer[Scalar[dtype], MutAnyOrigin],
    ):
        out_dev.unsafe_store[width=w](0, cbrt(in_dev.unsafe_load[width=w](0)))

    ctx.enqueue_function[kernel](out_device, in_device, grid_dim=1, block_dim=1)
    with out_device.map_to_host() as out_host:
        for i in range(w):
            if isnan(want[i]):
                assert_true(isnan(out_host[i]), msg=String(dtype, " lane ", i))
            else:
                assert_equal(
                    _bits(out_host[i]),
                    _bits(want[i]),
                    msg=String(dtype, " lane ", i),
                )


def test_cbrt_gpu_mixed_lanes() raises:
    with DeviceContext() as ctx:
        _check_mixed_lanes[.float32](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _check_mixed_lanes[.float64](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
