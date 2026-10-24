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

from std.math import exp2, inf, isinf, isnan, log1p, nan
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from test_utils.math import libm_call_f64
from std.utils.numerics import FPUtils

from std.utils.coord import Coord

comptime num_linear = 1 << 15
comptime num_geometric = 1 << 14


def _ulp_diff[
    dtype: DType, //
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Int where dtype.is_floating_point():
    """Returns same-sign finite ULP distance, or rejects class/sign mismatches.
    """
    if isnan(want) or isnan(got):
        return 0 if isnan(want) and isnan(got) else Int.MAX
    var got_bits = got.to_bits().cast[.uint64]()
    var want_bits = want.to_bits().cast[.uint64]()
    if isinf(want) or isinf(got):
        return 0 if got_bits == want_bits else Int.MAX
    comptime sign_mask = UInt64(1) << UInt64(bit_width_of[dtype]() - 1)
    if (got_bits & sign_mask) != (want_bits & sign_mask):
        return Int.MAX
    # Same-sign finite magnitudes fit in Int, even for float64.
    var got_magnitude = got_bits & (sign_mask - 1)
    var want_magnitude = want_bits & (sign_mask - 1)
    return Int(
        max(got_magnitude, want_magnitude) - min(got_magnitude, want_magnitude)
    )


def _test_ulp_diff[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    comptime U = type_of(T(1).to_bits())
    var positive_inf = inf[dtype]()
    var quiet_nan = nan[dtype]()
    assert_equal(_ulp_diff(positive_inf, positive_inf), 0)
    assert_equal(_ulp_diff(-positive_inf, -positive_inf), 0)
    assert_equal(_ulp_diff(T.MAX_FINITE, positive_inf), Int.MAX)
    assert_equal(_ulp_diff(positive_inf, T.MAX_FINITE), Int.MAX)
    assert_equal(_ulp_diff(T.MIN_FINITE, -positive_inf), Int.MAX)
    assert_equal(_ulp_diff(-positive_inf, positive_inf), Int.MAX)
    assert_equal(_ulp_diff(quiet_nan, quiet_nan), 0)
    assert_equal(_ulp_diff(quiet_nan, T(1)), Int.MAX)
    assert_equal(_ulp_diff(T(1), quiet_nan), Int.MAX)
    assert_equal(_ulp_diff(T(0), T(0)), 0)
    assert_equal(_ulp_diff(T(-0.0), T(-0.0)), 0)
    assert_equal(_ulp_diff(T(-0.0), T(0)), Int.MAX)
    assert_equal(_ulp_diff(T(-1), T(1)), Int.MAX)
    var next_one = T(from_bits=T(1).to_bits() + 1)
    assert_equal(_ulp_diff(next_one, T(1)), 1)
    assert_equal(_ulp_diff(T(1), next_one), 1)
    assert_equal(_ulp_diff(-next_one, T(-1)), 1)
    assert_equal(_ulp_diff(T(-1), -next_one), 1)
    var tiny = T(from_bits=U(1))
    assert_equal(_ulp_diff(tiny, T(0)), 1)
    assert_equal(_ulp_diff(-tiny, T(-0.0)), 1)


def test_ulp_diff() raises:
    _test_ulp_diff[.float64]()
    _test_ulp_diff[.float32]()
    _test_ulp_diff[.float16]()
    _test_ulp_diff[.bfloat16]()
    # The old signed subtraction wrapped to Int.MIN for this pair.
    var want = Float64(from_bits=0xC000A2B23F3BAB73)
    var got = Float64(from_bits=0x3FFF5D4DC0C4548D)
    assert_equal(_ulp_diff(got, want), Int.MAX)


def _inputs[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    # Linear over the interesting range, then geometric over the whole
    # magnitude range on both sides of zero.
    var lo = -1.0
    var hi = 3.0
    var e_lo = Float64(-1074 if dtype == .float64 else -149)
    var e_hi = Float64(1023 if dtype == .float64 else 127)
    comptime if dtype in (DType.float16, DType.bfloat16):
        e_lo = -24.0 if dtype == .float16 else -133.0
        e_hi = 15.0 if dtype == .float16 else 127.0

    var xs = List[T]()
    for i in range(num_linear):
        xs.append(T(lo + (hi - lo) * Float64(i) / Float64(num_linear)))
    for i in range(num_geometric):
        var t = Float64(i) / Float64(num_geometric)
        var m = exp2(e_lo + (e_hi - e_lo) * t)
        xs.append(T(m))
        # Approaches -1 from above.
        xs.append(T(-m) if m >= 1.0 else T(m - 1.0))
    xs.append(nan[dtype]())
    xs.append(inf[dtype]())
    xs.append(-inf[dtype]())
    xs.append(T(0))
    xs.append(T(-0.0))
    # Subnormals: min, max and a few in between, both signs.
    comptime U = type_of(T(1).to_bits())
    comptime mant = (U(1) << U(FPUtils[dtype].mantissa_width())) - 1
    for bits in [U(1), U(2), mant >> 1, mant - 1, mant]:
        xs.append(T(from_bits=bits))
        xs.append(-T(from_bits=bits))
    xs.append(T(-1))
    xs.append(T(-2))
    xs.append(T(1))
    xs.append(T.MAX_FINITE)
    xs.append(T.MIN_FINITE)
    return xs^


def _run[
    dtype: DType
](ctx: DeviceContext, max_ulp: Int) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var inputs = _inputs[dtype]()
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
        out_ptr.unsafe_store[width=simd_width](idx, log1p(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var gpu_worst = 0
    var cpu_worst = 0
    var gpu_msg = String()
    var cpu_msg = String()
    with out_device.map_to_host() as out_host:
        for i in range(length):
            var x = inputs[i]
            var want = libm_call_f64["log1p"](x)
            var cpu_val = log1p(x)
            var gpu = _ulp_diff(out_host[i], want)
            var cpu = _ulp_diff(cpu_val, want)
            if gpu > gpu_worst:
                gpu_worst = gpu
                gpu_msg = String(
                    t"GPU log1p({x}) = {out_host[i]}, reference {want}"
                )
            if cpu > cpu_worst:
                cpu_worst = cpu
                cpu_msg = String(
                    t"CPU log1p({x}) = {cpu_val}, reference {want}"
                )
    print(dtype, "log1p", "max ulp: gpu", gpu_worst, gpu_msg)
    print(dtype, "log1p", "max ulp: cpu", cpu_worst, cpu_msg)
    assert_true(gpu_worst <= max_ulp, gpu_msg)
    assert_true(cpu_worst <= max_ulp, cpu_msg)


def test_log1p_gpu() raises:
    with DeviceContext() as ctx:
        _run[.float32](ctx, max_ulp=1)
        _run[.float16](ctx, max_ulp=1)
        _run[.bfloat16](ctx, max_ulp=1)
        comptime if not default_accelerator().is_apple_gpu():
            _run[.float64](ctx, max_ulp=1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
