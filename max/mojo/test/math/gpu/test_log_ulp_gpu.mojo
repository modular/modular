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

from std.math import exp2, inf, isinf, isnan, log, nan
from std.math.math import _log_accurate
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from test_utils.math import libm_call_f64

from std.utils.coord import Coord

comptime num_linear = 1 << 15
comptime num_geometric = 1 << 14
comptime _flushes_float32_subnormals = default_accelerator().is_nvidia_gpu()


def _ulp_diff[
    dtype: DType
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Int where dtype.is_floating_point():
    if isnan(got) or isnan(want):
        return 0 if isnan(got) and isnan(want) else Int.MAX
    if got == want:
        return 0
    if isinf(got) or isinf(want):
        return Int.MAX
    comptime sign_shift = UInt64(bit_width_of[dtype]() - 1)
    var got_bits = got.to_bits().cast[.uint64]()
    var want_bits = want.to_bits().cast[.uint64]()
    if (
        got != 0
        and want != 0
        and ((got_bits >> sign_shift) != (want_bits >> sign_shift))
    ):
        return Int.MAX
    # Same-sign finite magnitudes differ by less than Int.MAX, even for f64.
    comptime magnitude_mask = (UInt64(1) << sign_shift) - 1
    var got_magnitude = got_bits & magnitude_mask
    var want_magnitude = want_bits & magnitude_mask
    return Int(
        max(got_magnitude, want_magnitude) - min(got_magnitude, want_magnitude)
    )


def _test_log_ulp_comparator[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    assert_equal(_ulp_diff(nan[dtype](), nan[dtype]()), 0)
    assert_equal(_ulp_diff(nan[dtype](), T(1)), Int.MAX)
    assert_equal(_ulp_diff(T(1), nan[dtype]()), Int.MAX)
    assert_equal(_ulp_diff(inf[dtype](), inf[dtype]()), 0)
    assert_equal(_ulp_diff(-inf[dtype](), -inf[dtype]()), 0)
    assert_equal(_ulp_diff(T.MAX_FINITE, inf[dtype]()), Int.MAX)
    assert_equal(_ulp_diff(inf[dtype](), T.MAX_FINITE), Int.MAX)
    assert_equal(_ulp_diff(-T.MAX_FINITE, -inf[dtype]()), Int.MAX)
    assert_equal(_ulp_diff(inf[dtype](), -inf[dtype]()), Int.MAX)
    assert_equal(_ulp_diff(T(-1), T(1)), Int.MAX)
    assert_equal(_ulp_diff(T(0), T(-0.0)), 0)
    assert_equal(_ulp_diff(T(1), T(1)), 0)
    assert_equal(_ulp_diff(T(-1), T(-1)), 0)


def test_log_ulp_comparator() raises:
    _test_log_ulp_comparator[.float64]()
    _test_log_ulp_comparator[.float32]()
    _test_log_ulp_comparator[.float16]()
    _test_log_ulp_comparator[.bfloat16]()
    var wrong_sign = Float64(from_bits=UInt64(0x3FFF5D4DC0C4548D))
    var reference = Float64(from_bits=UInt64(0xC000A2B23F3BAB73))
    assert_equal(_ulp_diff(wrong_sign, reference), Int.MAX)
    assert_equal(_ulp_diff(reference, wrong_sign), Int.MAX)
    var next_positive = Float64(from_bits=UInt64(0x3FF0000000000001))
    var next_negative = Float64(from_bits=UInt64(0xBFF0000000000001))
    assert_equal(_ulp_diff(next_positive, Float64(1)), 1)
    assert_equal(_ulp_diff(next_negative, Float64(-1)), 1)
    assert_equal(_ulp_diff(Float64(0), Float64.MAX_FINITE), 0x7FEFFFFFFFFFFFFF)
    var min_subnormal = Float64(from_bits=UInt64(1))
    assert_equal(_ulp_diff(min_subnormal, Float64(-0.0)), 1)
    assert_equal(_ulp_diff(-min_subnormal, Float64(0)), 1)


def _apply[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    comptime if name == "log_accurate":
        return _log_accurate(x)
    else:
        comptime assert name == "log"
        return log(x)


def _inputs[
    dtype: DType, name: StaticString
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    # Linear over the interesting range, then geometric over the whole
    # positive magnitude range.
    var lo = 0.0
    var hi = 4.0
    var e_lo = Float64(-1074 if dtype == .float64 else -149)
    var e_hi = Float64(1023 if dtype == .float64 else 127)
    comptime if dtype in (DType.float16, DType.bfloat16):
        e_lo = -24.0 if dtype == .float16 else -133.0
        e_hi = 15.0 if dtype == .float16 else 127.0
    # NVIDIA (`lg2.approx.ftz`) flushes float32 subnormal inputs to zero.
    # bfloat16 `log` goes through float32, and bfloat16 subnormals
    # (2^-133 .. 2^-127) are float32 subnormals.
    comptime if dtype == .float32 or (
        dtype == .bfloat16 and _flushes_float32_subnormals
    ):
        e_lo = -126.0

    var xs = List[T]()
    for i in range(num_linear):
        xs.append(T(lo + (hi - lo) * Float64(i) / Float64(num_linear)))
    for i in range(num_geometric):
        var t = Float64(i) / Float64(num_geometric)
        var m = exp2(e_lo + (e_hi - e_lo) * t)
        xs.append(T(m))
    xs.append(nan[dtype]())
    xs.append(inf[dtype]())
    xs.append(-inf[dtype]())
    xs.append(T(0))
    xs.append(T(-0.0))
    xs.append(T(-1))
    xs.append(T(-2))
    xs.append(T(1))
    xs.append(T.MAX_FINITE)
    xs.append(T.MIN_FINITE)
    comptime if dtype == .bfloat16 and not _flushes_float32_subnormals:
        xs.append(T(from_bits=UInt16(0x0001)))
        xs.append(T(from_bits=UInt16(0x007F)))
    return xs^


def _check[
    dtype: DType, name: StaticString
](
    ctx: DeviceContext,
    inputs: List[Scalar[dtype]],
    max_ulp: Int,
    abs_tol: Float64 = 0.0,
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    comptime libm_name: StaticString = (
        "log" if name == "log_accurate" else name
    )
    var length = len(inputs)

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = inputs[i]

    var in_ptr = in_device.unsafe_ptr()
    var out_ptr = out_device.unsafe_ptr()

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_ptr, var in_ptr}:
        var idx = Int(idx0[0])
        var val = in_ptr.unsafe_load[width=simd_width](idx)
        out_ptr.unsafe_store[width=simd_width](idx, _apply[name](val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var gpu_worst = 0
    var cpu_worst = 0
    var gpu_msg = String()
    var cpu_msg = String()
    with out_device.map_to_host() as out_host:
        for i in range(length):
            var x = inputs[i]
            var want = libm_call_f64[libm_name](x)
            var cpu_val = _apply[name](x)
            var gpu = _ulp_diff(out_host[i], want)
            var cpu = _ulp_diff(cpu_val, want)
            if (
                gpu != Int.MAX
                and abs(Float64(out_host[i]) - Float64(want)) <= abs_tol
            ):
                gpu = 0
            if (
                cpu != Int.MAX
                and abs(Float64(cpu_val) - Float64(want)) <= abs_tol
            ):
                cpu = 0
            if gpu > gpu_worst:
                gpu_worst = gpu
                gpu_msg = String(
                    t"GPU {name}({x}) = {out_host[i]}, reference {want}"
                )
            if cpu > cpu_worst:
                cpu_worst = cpu
                cpu_msg = String(
                    t"CPU {name}({x}) = {cpu_val}, reference {want}"
                )
    print(dtype, name, "max ulp: gpu", gpu_worst, gpu_msg)
    print(dtype, name, "max ulp: cpu", cpu_worst, cpu_msg)
    assert_true(gpu_worst <= max_ulp, gpu_msg)
    assert_true(cpu_worst <= max_ulp, cpu_msg)


def _run[
    dtype: DType, name: StaticString
](
    ctx: DeviceContext, max_ulp: Int, abs_tol: Float64 = 0.0
) raises where dtype.is_floating_point():
    _check[dtype, name](ctx, _inputs[dtype, name](), max_ulp, abs_tol)


def test_log_subnormal_gpu() raises:
    """`_log_reduce` reads a subnormal's exponent from its bits, so the
    smallest and largest subnormals, and the smallest normal, of each dtype
    stay within 1 ulp of libm. The float32 kernel is called directly because
    public float32 `log` does not use it yet."""
    with DeviceContext() as ctx:
        var f32: List[Float32] = [
            Float32(from_bits=UInt32(1)),
            Float32(from_bits=UInt32(0x007FFFFF)),
            Float32(from_bits=UInt32(0x00800000)),
        ]
        _check[.float32, "log_accurate"](ctx, f32, max_ulp=1)
        comptime if not _flushes_float32_subnormals:
            _check[.float32, "log"](
                ctx,
                f32 + [Float32(9.1835e-41), Float32(-9.1835e-41)],
                max_ulp=2,
                abs_tol=2.0**-21,
            )
        comptime if not default_accelerator().is_apple_gpu():
            var f64: List[Float64] = [
                Float64(from_bits=UInt64(1)),
                Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF)),
                Float64(from_bits=UInt64(0x0010000000000000)),
            ]
            _check[.float64, "log"](ctx, f64, max_ulp=1)


def test_log_gpu() raises:
    with DeviceContext() as ctx:
        # NVIDIA computes float32 log as ln2 * lg2.approx(x), whose absolute
        # error is about 2**-22; near x = 1 that is many ulp of the result.
        _run[.float32, "log"](ctx, max_ulp=2, abs_tol=2.0**-21)
        _run[.float16, "log"](ctx, max_ulp=1)
        _run[.bfloat16, "log"](ctx, max_ulp=1)
        comptime if not default_accelerator().is_apple_gpu():
            _run[.float64, "log"](ctx, max_ulp=1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
