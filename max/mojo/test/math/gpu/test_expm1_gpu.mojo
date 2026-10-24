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

from std.math import expm1, inf, isinf, isnan, nan
from std.memory import bitcast
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from test_utils.math import libm_call

from std.utils.coord import Coord

# Sweep ends just past the overflow threshold log(MAX_FINITE) of each dtype.
comptime expm1_f64_sweep_hi = 720.0
comptime expm1_f32_sweep_hi = 95.0
comptime expm1_libm = libm_call["expm1f", "expm1"]


def _ulp_distance[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Int where dtype.is_floating_point():
    comptime assert dtype in (DType.float32, DType.float64)
    if isnan(a) or isnan(b):
        return 0 if isnan(a) and isnan(b) else Int.MAX
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    comptime sign_bit = Scalar[uint_type](1) << Scalar[uint_type](
        31 if dtype == .float32 else 63
    )
    var ia = bitcast[uint_type](a)
    var ib = bitcast[uint_type](b)
    if ia == ib:
        return 0
    if isinf(a) or isinf(b) or ((ia ^ ib) & sign_bit) != 0:
        return Int.MAX
    # Same-sign finite encodings differ by at most Int.MAX, even for f64.
    return Int(max(ia, ib) - min(ia, ib))


def _check[
    dtype: DType
](
    ctx: DeviceContext, inputs: List[Scalar[dtype]], max_ulp: Int
) raises where dtype.is_floating_point():
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    comptime sign_bit = Scalar[uint_type](1) << Scalar[uint_type](
        31 if dtype == .float32 else 63
    )
    comptime min_normal_bits = Scalar[uint_type](1) << Scalar[uint_type](
        23 if dtype == .float32 else 52
    )
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
        out_ptr.unsafe_store[width=simd_width](idx, expm1(val))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var worst_gpu = 0
    var worst_host = 0
    var worst_gpu_host = 0
    with out_device.map_to_host() as out_host:
        for i in range(length):
            var x = inputs[i]
            var expected = expm1_libm(x)
            var host = expm1(x)
            var gpu_ulp = _ulp_distance(out_host[i], expected)
            var host_ulp = _ulp_distance(host, expected)
            worst_gpu = max(worst_gpu, gpu_ulp)
            worst_host = max(worst_host, host_ulp)
            worst_gpu_host = max(
                worst_gpu_host, _ulp_distance(out_host[i], host)
            )
            assert_true(
                gpu_ulp <= max_ulp and host_ulp <= max_ulp,
                String(
                    t"expm1[{dtype}]({x}): gpu={out_host[i]} host={host}"
                    t" libm={expected}"
                ),
            )
            var input_bits = bitcast[uint_type](x)
            if (input_bits & ~sign_bit) <= min_normal_bits:
                # ULP tolerances would accept zero for the minimum subnormal.
                assert_equal(
                    bitcast[uint_type](out_host[i]),
                    input_bits,
                    msg=String(t"expm1[{dtype}]({x}) lost tiny GPU bits"),
                )
                assert_equal(
                    bitcast[uint_type](host),
                    input_bits,
                    msg=String(t"expm1[{dtype}]({x}) lost tiny host bits"),
                )
    print(
        "expm1",
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


def _expm1_inputs[
    dtype: DType
](hi: Scalar[dtype]) -> List[Scalar[dtype]] where dtype.is_floating_point():
    var xs = _specials[dtype]()
    # Dense sweep over the whole finite range, past both saturation points.
    comptime n = 16384
    for i in range(n + 1):
        xs.append(-60.0 + (hi + 70.0) * Scalar[dtype](i) / n)
    # Dense sweep of the reduced-argument region and the k = -1, 0, 1 cases.
    for i in range(n + 1):
        xs.append(-1.5 + 3.0 * Scalar[dtype](i) / n)
    # Geometric sweep through small normal magnitudes.
    var t = Scalar[dtype](1.0)
    for _ in range(140):
        t *= 0.6
        xs.append(t)
        xs.append(-t)
    # Arithmetic generation can stall at the minimum subnormal or flush to zero.
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    comptime sign_bit = Scalar[uint_type](1) << Scalar[uint_type](
        31 if dtype == .float32 else 63
    )
    comptime mant_bits = Scalar[uint_type](23 if dtype == .float32 else 52)
    comptime min_normal_bits = Scalar[uint_type](1) << mant_bits
    for bits in [
        Scalar[uint_type](1),
        2,
        min_normal_bits >> 1,
        min_normal_bits - 1,
        min_normal_bits,
    ]:
        xs.append(bitcast[dtype](bits))
        xs.append(bitcast[dtype](bits | sign_bit))
    return xs^


def test_expm1_f64() raises:
    with DeviceContext() as ctx:
        # `main` skips this test on Apple GPU, which has no float64.
        comptime if not default_accelerator().is_apple_gpu():
            comptime max_ulp = 2
            _check[.float64](
                ctx, _expm1_inputs[.float64](expm1_f64_sweep_hi), max_ulp
            )


def test_expm1_f32() raises:
    with DeviceContext() as ctx:
        comptime max_ulp = 1
        _check[.float32](
            ctx, _expm1_inputs[.float32](expm1_f32_sweep_hi), max_ulp
        )


def _check_half_subnormals[
    dtype: DType, utype: DType
](
    ctx: DeviceContext, bits: List[Scalar[utype]]
) raises where dtype.is_floating_point():
    var n = len(bits)
    var in_device = ctx.enqueue_create_buffer[dtype](n)
    var out_device = ctx.enqueue_create_buffer[dtype](n)
    with in_device.map_to_host() as in_host:
        for i in range(n):
            in_host[i] = bitcast[dtype](bits[i])
    var in_ptr = in_device.unsafe_ptr()
    var out_ptr = out_device.unsafe_ptr()

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_ptr, var in_ptr}:
        var idx = Int(idx0[0])
        out_ptr.unsafe_store(idx, expm1(in_ptr.unsafe_load(idx)))

    elementwise[1, target="gpu"](func, Coord(n), ctx)
    with out_device.map_to_host() as out_host:
        for i in range(n):
            var x = bitcast[dtype](bits[i])
            var expected = expm1_libm(x.cast[.float64]()).cast[dtype]()
            assert_equal(
                bitcast[utype](out_host[i]),
                bitcast[utype](expected),
                msg=String(t"expm1[{dtype}]({x}) = {out_host[i]}"),
            )


def test_expm1_bfloat16_subnormals() raises:
    with DeviceContext() as ctx:
        # Zeros, min/max subnormal, min normal, +-inf, 1.0, -1.0 (both signs
        # for the subnormals).
        var bits: List[UInt16] = [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x0002,
            0x0040,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x7F80,
            0xFF80,
            0x3F80,
            0xBF80,
        ]
        _check_half_subnormals[.bfloat16](ctx, bits)


def test_expm1_float16_subnormals() raises:
    with DeviceContext() as ctx:
        var bits: List[UInt16] = [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x0002,
            0x0200,
            0x03FF,
            0x83FF,
            0x0400,
            0x8400,
            0x7C00,
            0xFC00,
            0x3C00,
            0xBC00,
        ]
        _check_half_subnormals[.float16](ctx, bits)


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    comptime if default_accelerator().is_apple_gpu():
        suite.skip[test_expm1_f64]()
    suite^.run()
