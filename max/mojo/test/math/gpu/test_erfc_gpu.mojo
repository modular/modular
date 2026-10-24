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

from std.math import erfc, inf, isinf, isnan, nan
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from std.testing import assert_true, TestSuite
from test_utils.math import libm_call_f64

from std.utils.coord import Coord

comptime length = 1 << 16
comptime num_specials = 16
comptime max_ulp_wide = 2
# glibc's float64 erfc is itself off by over 2 ulp in places (at
# 1.2257173382173383 it gives 0.08301996487567713, correctly rounded is
# ...716), so the libm comparison stays loose and
# `test_erfc_f64_hard_cases` pins accuracy.
comptime max_ulp_f64 = 4
# The f32 result is rounded again to the narrow type, so allow one ULP of
# double rounding.
comptime max_ulp_narrow = 1


def _ulp_diff[
    dtype: DType
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Int where dtype.is_floating_point():
    """Counts the floats between `got` and `want`. Bit-identical values give
    0, as do two NaNs of any payload or sign; a NaN against a non-NaN, an
    infinity, or differing signs (other than +0 and -0) give `Int.MAX`."""
    comptime sign = UInt64(1) << UInt64(bit_width_of[dtype]() - 1)
    var a = got.to_bits().cast[.uint64]()
    var b = want.to_bits().cast[.uint64]()
    if a == b or (isnan(got) and isnan(want)):
        return 0
    if isnan(got) or isnan(want) or isinf(got) or isinf(want):
        return Int.MAX
    var ma = a & (sign - 1)
    var mb = b & (sign - 1)
    if ((a ^ b) & sign) != 0:
        return Int(ma + mb) if ma == 0 or mb == 0 else Int.MAX
    return Int(max(ma, mb) - min(ma, mb))


def _special_values[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    # An input whose erfc is subnormal in the type.
    var subnormal_erfc_x: Scalar[dtype]
    comptime if dtype == .float64:
        subnormal_erfc_x = 26.6
    else:
        subnormal_erfc_x = 9.5
    return [
        nan[dtype](),
        inf[dtype](),
        -inf[dtype](),
        0.0,
        -0.0,
        0.84375,
        -0.84375,
        1.25,
        -1.25,
        6.0,
        -6.0,
        28.0,
        -28.0,
        subnormal_erfc_x,
        Scalar[dtype].MAX_FINITE,
        Scalar[dtype].MIN_FINITE,
    ]


def _run_erfc[
    dtype: DType
](
    ctx: DeviceContext, lo: Float64, hi: Float64, max_ulp: Int
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    var specials = _special_values[dtype]()
    with in_device.map_to_host() as in_host:
        var dense = length - num_specials
        for i in range(dense):
            var t = Float64(i) / Float64(dense)
            in_host[i] = Scalar[dtype](lo + (hi - lo) * t)
        for i in range(num_specials):
            in_host[dense + i] = specials[i]

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, erfc(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var gpu_worst = 0
    var cpu_worst = 0
    var gpu_msg = String()
    var cpu_msg = String()
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = in_host[i]
            var want = libm_call_f64["erfc"](x)
            var gpu = _ulp_diff(out_host[i], want)
            var cpu = _ulp_diff(erfc(x), want)
            if gpu > gpu_worst:
                gpu_worst = gpu
                gpu_msg = String(t"GPU erfc({x}) = {out_host[i]}, libm {want}")
            if cpu > cpu_worst:
                cpu_worst = cpu
                cpu_msg = String(t"CPU erfc({x}) = {erfc(x)}, libm {want}")
    print(dtype, "max ulp vs libm: gpu", gpu_worst, gpu_msg)
    print(dtype, "max ulp vs libm: cpu", cpu_worst, cpu_msg)
    assert_true(gpu_worst <= max_ulp, gpu_msg)
    assert_true(cpu_worst <= max_ulp, cpu_msg)


def test_erfc_f64_hard_cases() raises:
    """Checks inputs that once had more than 1.5 ulp of error against their
    correctly rounded erfc (mpmath), to within 1 ulp on GPU and host."""
    comptime if default_accelerator().is_apple_gpu():
        return
    comptime dtype = DType.float64
    with DeviceContext() as ctx:
        comptime pack_size = simd_width_of[dtype, target=ctx.target]()
        var inputs: List[Float64] = [
            0.8291682292057292,
            0.8324116746883992,
            1.1728488886441404,
            2.834710867771694,
        ]
        var want: List[Float64] = [
            0.2409478422146139,
            0.23911254431355505,
            0.09718456490977953,
            6.10057131957997e-05,
        ]
        var n = len(inputs)
        var padded = (n + pack_size - 1) // pack_size * pack_size
        var in_device = ctx.enqueue_create_buffer[dtype](padded)
        var out_device = ctx.enqueue_create_buffer[dtype](padded)
        with in_device.map_to_host() as in_host:
            for i in range(padded):
                in_host[i] = inputs[i] if i < n else 0.0

        var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
        var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

        @inline(.always)
        def func[
            simd_width: Int, alignment: Int = 1
        ](idx0: Coord) {var out_buffer, var in_buffer}:
            var idx = Int(idx0[0])
            var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
            out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
                idx, erfc(val)
            )

        elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

        with out_device.map_to_host() as out_host:
            for i in range(n):
                var x = inputs[i]
                assert_true(
                    _ulp_diff(out_host[i], want[i]) <= 1
                    and _ulp_diff(erfc(x), want[i]) <= 1,
                    String(
                        t"erfc({x}) gpu {out_host[i]} host {erfc(x)} want"
                        t" {want[i]}"
                    ),
                )


def test_erfc_gpu() raises:
    with DeviceContext() as ctx:
        _run_erfc[.float32](ctx, lo=-12.0, hi=12.0, max_ulp=max_ulp_wide)
        _run_erfc[.float16](ctx, lo=-6.0, hi=6.0, max_ulp=max_ulp_narrow)
        _run_erfc[.bfloat16](ctx, lo=-6.0, hi=6.0, max_ulp=max_ulp_narrow)
        comptime if not default_accelerator().is_apple_gpu():
            _run_erfc[.float64](ctx, lo=-8.0, hi=29.0, max_ulp=max_ulp_f64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
