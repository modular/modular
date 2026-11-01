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

from std.math import atanh, frexp, inf, isinf, isnan, nan, nextafter
from std.math.math import _call_libm, _frexp_bits
from std.memory import bitcast
from std.sys import simd_width_of
from std.sys.info import bit_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.utils.numerics import FPUtils
from std.testing import assert_equal, assert_true, TestSuite

from std.utils.coord import Coord


def _ulp_distance[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Int where dtype.is_floating_point():
    """Distance between `a` and `b` in representable values of `dtype`.

    Bit-identical values (a matching zero or infinity included) and two NaNs
    of any payload are 0 apart. A NaN, an infinity or a sign mismatch on one
    side only is `Int.MAX`, so no ulp tolerance can accept a finite value
    where an infinity or NaN is required, or vice versa. Only +-0 count as
    the same sign.
    """
    comptime assert dtype in (DType.float32, DType.float64)
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    comptime sign = Scalar[uint_type](1 << (bit_width_of[dtype]() - 1))
    var ua = bitcast[uint_type](a)
    var ub = bitcast[uint_type](b)
    if ua == ub or (isnan(a) and isnan(b)):
        return 0
    if isnan(a) or isnan(b) or isinf(a) or isinf(b):
        return Int.MAX
    var ma = Int(ua & ~sign)
    var mb = Int(ub & ~sign)
    if (ua & sign) != (ub & sign) and ma != 0 and mb != 0:
        return Int.MAX
    return abs(ma - mb)


def _test_ulp_distance_impl[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    var one = T(1.0)
    var big = T.MAX_FINITE
    var pos_inf = inf[dtype]()
    assert_equal(_ulp_distance(one, one), 0)
    assert_equal(_ulp_distance(one, nextafter(one, pos_inf)), 1)
    assert_equal(_ulp_distance(nextafter(one, T(0)), one), 1)
    assert_equal(_ulp_distance(T(0.0), T(-0.0)), 0)
    assert_equal(_ulp_distance(T(-0.0), nextafter(T(0), one)), 1)
    assert_equal(_ulp_distance(pos_inf, pos_inf), 0)
    assert_equal(_ulp_distance(-pos_inf, -pos_inf), 0)
    assert_equal(_ulp_distance(nan[dtype](), nan[dtype]()), 0)
    # libm and the Mojo port return NaNs of different sign or payload for
    # |x| > 1; both are correct.
    assert_equal(_ulp_distance(nan[dtype](), -nan[dtype]()), 0)
    assert_equal(_ulp_distance(T(0) / T(0), nan[dtype]()), 0)
    # The largest finite value is one bit pattern from infinity, which a
    # 2 ulp tolerance must still reject.
    assert_equal(_ulp_distance(big, pos_inf), Int.MAX)
    assert_equal(_ulp_distance(-big, -pos_inf), Int.MAX)
    assert_equal(_ulp_distance(pos_inf, -pos_inf), Int.MAX)
    assert_equal(_ulp_distance(nan[dtype](), one), Int.MAX)
    assert_equal(_ulp_distance(one, nan[dtype]()), Int.MAX)
    assert_equal(_ulp_distance(-one, one), Int.MAX)
    assert_equal(_ulp_distance(T(-1e-30), T(1e-30)), Int.MAX)


def test_ulp_distance() raises:
    _test_ulp_distance_impl[.float32]()
    _test_ulp_distance_impl[.float64]()


def _check[
    dtype: DType
](
    ctx: DeviceContext, inputs: List[Scalar[dtype]], max_ulp: Int
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
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
        out_ptr.unsafe_store[width=simd_width](idx, atanh(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    comptime name = "atanh"
    var worst_gpu = 0
    var worst_host = 0
    var worst_gpu_host = 0
    with out_device.map_to_host() as out_host:
        for i in range(length):
            var x = inputs[i]
            var expected = _call_libm[name](x)
            var host = atanh(x)
            var gpu_ulp = _ulp_distance(out_host[i], expected)
            var host_ulp = _ulp_distance(host, expected)
            worst_gpu = max(worst_gpu, gpu_ulp)
            worst_host = max(worst_host, host_ulp)
            var gpu_host_ulp = _ulp_distance(out_host[i], host)
            worst_gpu_host = max(worst_gpu_host, gpu_host_ulp)
            assert_true(
                gpu_ulp <= max_ulp and host_ulp <= max_ulp,
                String(
                    t"{name}[{dtype}]({x}): gpu={out_host[i]} host={host}"
                    t" libm={expected}"
                ),
            )
            if expected == 0:
                assert_true(
                    bitcast[.uint64](out_host[i].cast[.float64]())
                    == bitcast[.uint64](expected.cast[.float64]()),
                    String(t"{name}({x}) lost the sign of zero"),
                )
    print(
        name,
        dtype,
        "max ulp vs libm gpu:",
        worst_gpu,
        "host:",
        worst_host,
        "gpu vs host:",
        worst_gpu_host,
    )


def _specials[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    return [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
        1e-30,
        -1e-30,
    ]


def _atanh_inputs[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    var xs = _specials[dtype]()
    comptime n = 32768
    for i in range(-n, n + 1):
        xs.append(Scalar[dtype](i) / n)
    # Approach +-1, where 1 - x loses all its bits.
    var t = Scalar[dtype](1.0)
    for _ in range(120):
        t *= 0.7
        xs.append(1.0 - t)
        xs.append(t - 1.0)
        xs.append(t)
        xs.append(-t)
    xs.append(1.0000001)
    xs.append(-1.0000001)
    xs.extend(_subnormals[dtype]())
    return xs^


def _subnormals[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    """The smallest and largest subnormal, the smallest normal, and an interior
    subnormal of `dtype`, for `_frexp_bits` and the `atanh` tiny path."""
    comptime T = Scalar[dtype]
    comptime min_normal = T(
        1.17549435e-38 if dtype == .float32 else 2.2250738585072014e-308
    )
    var min_sub = nextafter(T(0), T(1))
    return [
        min_sub,
        -min_sub,
        nextafter(min_normal, T(0)),
        min_normal,
        min_sub * 12345,
    ]


def _check_frexp_bits[
    dtype: DType
](
    ctx: DeviceContext, inputs: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    """Runs `_frexp_bits` on the GPU and compares each (fraction, exponent)
    pair to host `frexp`, which the GPU `frexp` must match without flushing
    subnormal inputs."""
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var length = len(inputs)

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var frac_device = ctx.enqueue_create_buffer[dtype](length)
    var exp_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = inputs[i]

    var in_ptr = in_device.unsafe_ptr()
    var frac_ptr = frac_device.unsafe_ptr()
    var exp_ptr = exp_device.unsafe_ptr()

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var frac_ptr, var exp_ptr, var in_ptr}:
        var idx = Int(idx0[0])
        var val = in_ptr.unsafe_load[width=simd_width](idx)
        var fe = _frexp_bits(val)
        frac_ptr.unsafe_store[width=simd_width](idx, fe[0])
        exp_ptr.unsafe_store[width=simd_width](idx, fe[1])

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    with frac_device.map_to_host() as frac_host:
        with exp_device.map_to_host() as exp_host:
            for i in range(length):
                var x = inputs[i]
                var want = frexp(x)
                assert_true(
                    _ulp_distance(frac_host[i], want[0]) == 0
                    and _ulp_distance(exp_host[i], want[1]) == 0,
                    String(
                        t"_frexp_bits[{dtype}]({x}): gpu=({frac_host[i]},"
                        t" {exp_host[i]}) host frexp=({want[0]}, {want[1]})"
                    ),
                )


def test_frexp_bits_f32() raises:
    with DeviceContext() as ctx:
        var xs = _subnormals[.float32]()
        xs.append(1.0)
        xs.append(0.75)
        xs.append(-123.45)
        xs.append(1e30)
        _check_frexp_bits[.float32](ctx, xs)


def test_frexp_bits_f64() raises:
    with DeviceContext() as ctx:
        comptime if not default_accelerator().is_apple_gpu():
            var xs = _subnormals[.float64]()
            xs.append(1.0)
            xs.append(0.75)
            xs.append(-123.45)
            xs.append(1e300)
            _check_frexp_bits[.float64](ctx, xs)


def test_atanh_f64() raises:
    with DeviceContext() as ctx:
        comptime if not default_accelerator().is_apple_gpu():
            comptime max_ulp = 3
            _check[.float64](ctx, _atanh_inputs[.float64](), max_ulp)


def test_atanh_f32() raises:
    comptime max_ulp = 2
    with DeviceContext() as ctx:
        _check[.float32](ctx, _atanh_inputs[.float32](), max_ulp)


def _check_subnormals[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    """Checks that `atanh` returns every subnormal input of `dtype` unchanged
    on the GPU, as host `atanh` does, even where the GPU flushes subnormals."""
    comptime uint_type = DType.uint16 if bit_width_of[dtype]() == 16 else (
        DType.uint32
    )
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime stride = 1 if dtype != .float32 else 4099
    var xs = List[Scalar[dtype]]()
    var m = 1
    while m < (1 << mantissa_width):
        var x = bitcast[dtype](Scalar[uint_type](m))
        xs.append(x)
        xs.append(-x)
        m += stride
    xs.append(bitcast[dtype](Scalar[uint_type]((1 << mantissa_width) - 1)))
    var length = len(xs)

    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = xs[i]
    var in_ptr = in_device.unsafe_ptr()
    var out_ptr = out_device.unsafe_ptr()

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_ptr, var in_ptr}:
        var idx = Int(idx0[0])
        var val = in_ptr.unsafe_load[width=simd_width](idx)
        out_ptr.unsafe_store[width=simd_width](idx, atanh(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    with out_device.map_to_host() as out_host:
        for i in range(length):
            assert_equal(
                bitcast[uint_type](out_host[i]),
                bitcast[uint_type](atanh(xs[i])),
                String(t"atanh[{dtype}]({xs[i]})"),
            )


def test_atanh_subnormals() raises:
    with DeviceContext() as ctx:
        _check_subnormals[.float32](ctx)
        _check_subnormals[.float16](ctx)
        _check_subnormals[.bfloat16](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
