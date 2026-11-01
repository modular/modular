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

from std.math import erf, exp2, inf, isinf, isnan, nan
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_true, TestSuite
from test_utils.math import libm_call_f64

from std.utils.coord import Coord

comptime num_linear = 1 << 15
comptime num_geometric = 1 << 14


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


def _apply[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    comptime assert name == "erf"
    return erf(x)


def _inputs[
    dtype: DType, name: StaticString
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    # Linear over the interesting range, then geometric over the whole
    # positive and negative magnitude range.
    comptime lo = -7.0
    comptime hi = 7.0
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
        xs.append(T(-m))
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
    return xs^


def _run[
    dtype: DType, name: StaticString
](
    ctx: DeviceContext, max_ulp: Int, abs_tol: Float64 = 0.0
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var inputs = _inputs[dtype, name]()
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
            var want = libm_call_f64[name](x)
            var cpu_val = _apply[name](x)
            var gpu = _ulp_diff(out_host[i], want)
            var cpu = _ulp_diff(cpu_val, want)
            if abs(Float64(out_host[i]) - Float64(want)) <= abs_tol:
                gpu = 0
            if abs(Float64(cpu_val) - Float64(want)) <= abs_tol:
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


def test_erf_gpu() raises:
    with DeviceContext() as ctx:
        _run[.float32, "erf"](ctx, max_ulp=2)
        _run[.float16, "erf"](ctx, max_ulp=1)
        _run[.bfloat16, "erf"](ctx, max_ulp=1)
        comptime if not default_accelerator().is_apple_gpu():
            # The float64 reference is libm erf, itself up to about 1 ulp off.
            _run[.float64, "erf"](ctx, max_ulp=2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
