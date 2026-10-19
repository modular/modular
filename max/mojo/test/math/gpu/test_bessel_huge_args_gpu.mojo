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

# GPU counterpart of the stdlib `test_bessel_huge_args` and
# `test_subnormal_inputs` tests: Bessel functions where `x - pi/4` is within
# 2^-61 of a multiple of pi/2, and subnormal inputs and results, which are
# also correct on targets that flush subnormals (Apple GPUs). References are
# correctly rounded (MPFR via mpmath).

from std.math import inf, isfinite, isnan, j0, j1, nan, y0, y1
from std.sys import bit_width_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import TestSuite, assert_false, assert_true

from std.utils.coord import Coord
from std.utils.numerics import FPUtils

comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()


def _run[
    dtype: DType, //, math_fn: UnaryFn
](ctx: DeviceContext, xs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Applies `math_fn` to `xs` on the GPU."""
    var length = len(xs)
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    # Pad to a whole number of packs so every lane holds a real input.
    var padded = ((length + pack_size - 1) // pack_size) * pack_size
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = xs[i] if i < length else xs[0]
    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    var out = List[Scalar[dtype]]()
    with out_device.map_to_host() as out_host:
        for i in range(length):
            out.append(out_host[i])
    return out^


@inline(.always)
def _finite_distance[
    dtype: DType, //
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> UInt64 where dtype.is_floating_point():
    comptime I = Scalar[FPUtils[dtype].integral_type]
    comptime magnitude_mask = I((1 << (bit_width_of[dtype]() - 1)) - 1)
    var gb = got._to_bits_signed()
    var wb = want._to_bits_signed()
    var gm = (gb & magnitude_mask).cast[.uint64]()
    var wm = (wb & magnitude_mask).cast[.uint64]()
    if (gb < 0) != (wb < 0):
        # Finite magnitudes sum without overflowing UInt64, even for f64.
        return gm + wm
    return gm - wm if gm >= wm else wm - gm


def _reference_matches[
    dtype: DType, //, max_ulp: Int
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Bool where dtype.is_floating_point():
    if isnan(want):
        return isnan(got)
    if not isfinite(want) or want == 0.0:
        return got._to_bits_signed() == want._to_bits_signed()
    if not isfinite(got):
        return False
    return _finite_distance(got, want) <= UInt64(max_ulp)


def _check_reference_classes[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime U = Scalar[FPUtils[dtype].uint_type]
    var negative_inf = -inf[dtype]()
    var negative_max = Scalar[dtype](
        from_bits=negative_inf._to_bits_signed().cast[
            FPUtils[dtype].uint_type
        ]()
        - U(1)
    )
    assert_false(_reference_matches[1](negative_max, negative_inf))
    assert_false(_reference_matches[1](-negative_max, -negative_inf))
    assert_false(_reference_matches[1](-negative_inf, negative_inf))
    assert_false(_reference_matches[1](negative_inf, negative_max))
    assert_false(_reference_matches[1](nan[dtype](), Scalar[dtype](1.0)))
    assert_false(_reference_matches[1](Scalar[dtype](1.0), nan[dtype]()))
    assert_true(_reference_matches[1](negative_inf, negative_inf))
    assert_true(_reference_matches[1](nan[dtype](), nan[dtype]()))
    assert_false(_reference_matches[1](Scalar[dtype](-0.0), Scalar[dtype](0.0)))
    assert_false(_reference_matches[1](Scalar[dtype](-1.0), Scalar[dtype](1.0)))
    assert_true(_reference_matches[1](Scalar[dtype](1.0), Scalar[dtype](1.0)))


def test_reference_classes() raises:
    _check_reference_classes[.float32]()
    _check_reference_classes[.float64]()


def _check[
    dtype: DType, //, max_ulp: Int
](
    xs: List[Scalar[dtype]],
    got: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
    name: String,
) raises where dtype.is_floating_point():
    var worst = UInt64(0)
    for i in range(len(xs)):
        assert_true(
            _reference_matches[max_ulp](got[i], want[i]),
            String(t"gpu {name}({xs[i]}) = {got[i]}, expected {want[i]}"),
        )
        if isfinite(got[i]) and isfinite(want[i]):
            worst = max(worst, _finite_distance(got[i], want[i]))
    print(t"{name} {dtype} max ulp: {worst}")


def _check_bessel[
    dtype: DType, max_ulp: Int
](
    ctx: DeviceContext,
    xs: List[Scalar[dtype]],
    j0s: List[Scalar[dtype]],
    y0s: List[Scalar[dtype]],
    j1s: List[Scalar[dtype]],
    y1s: List[Scalar[dtype]],
) raises where dtype.is_floating_point():
    _check[max_ulp](xs, _run[j0](ctx, xs), j0s, "j0")
    _check[max_ulp](xs, _run[y0](ctx, xs), y0s, "y0")
    _check[max_ulp](xs, _run[j1](ctx, xs), j1s, "j1")
    _check[max_ulp](xs, _run[y1](ctx, xs), y1s, "y1")


def test_bessel_phase() raises:
    with DeviceContext() as ctx:
        comptime if not default_accelerator().is_apple_gpu():
            # The closest a float64 comes to an odd multiple of pi/4.
            _check_bessel[.float64, 1](
                ctx,
                [2.6596863241632707e255],
                [-1.5471230315595117e-128],
                [-3.62581117707763e-147],
                [-3.62581117707763e-147],
                [1.5471230315595117e-128],
            )
        # Float32 inputs closest to odd multiples of pi/4 in the Cody-Waite
        # and Payne-Hanek ranges, excluding 21390802.0, where j0 itself
        # cancels below float32 resolution.
        _check_bessel[.float32, 2](
            ctx,
            [
                126.44910430908203,
                2.356194496154785,
                7.0685834884643555,
                1119.1923828125,
                11.780972480773926,
                3.864589459726451e28,
                10999692288.0,
                2.319091691391733e25,
                761394495488.0,
                26258.216796875,
                1354837.75,
            ],
            [
                0.07095455378293991,
                0.025495409965515137,
                0.2996954321861267,
                0.023849960416555405,
                -0.002456285525113344,
                -4.058713373798328e-15,
                7.607637144246837e-06,
                1.6568419813206559e-13,
                -3.162302429125601e-15,
                0.004923879634588957,
                0.0006854825769551098,
            ],
            [
                -7.013900903984904e-05,
                0.5143632888793945,
                -0.005248181521892548,
                -2.6644340778148035e-06,
                -0.23234421014785767,
                -3.276943780290961e-24,
                7.569287163472927e-15,
                5.358108865942336e-22,
                -9.14397844553605e-07,
                -2.3399774562449238e-08,
                -5.669753208592532e-11,
            ],
            [
                0.0002104217273881659,
                0.5292403101921082,
                0.0158381387591362,
                7.990550329850521e-06,
                -0.23265525698661804,
                -3.276943780290961e-24,
                7.915098528584862e-15,
                5.358108865942336e-22,
                -9.14397844553605e-07,
                7.035906435248762e-08,
                1.9627835468050137e-10,
            ],
            [
                -0.07095538824796677,
                0.07969999313354492,
                -0.3007933497428894,
                -0.023849964141845703,
                -0.007385152857750654,
                4.058713373798328e-15,
                -7.607637144246837e-06,
                -1.6568419813206559e-13,
                3.1617018827659977e-15,
                -0.004923879634588957,
                -0.0006854825769551098,
            ],
        )


def test_subnormal_inputs_f32() raises:
    comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
    comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
    with DeviceContext() as ctx:
        var positive: List[Float32] = [f32_min, f32_min * 4, f32_max_sub, 8.0]
        # y1(x) = -2/(pi x) and y0(x) ~ (2/pi) log(x) stay finite for the
        # largest subnormal; y1 overflows for the two smallest inputs.
        _check[1](
            positive,
            _run[y1](ctx, positive),
            [
                -inf[.float32](),
                -inf[.float32](),
                -5.415762869885221e37,
                -0.1580604612827301,
            ],
            "y1",
        )
        _check[1](
            positive,
            _run[y0](ctx, positive),
            [
                -65.82321166992188,
                -64.940673828125,
                -55.67397689819336,
                0.22352148592472076,
            ],
            "y0",
        )


def test_subnormal_results() raises:
    # j1(x) = x/2 to the precision of the type for tiny x, so the result is
    # subnormal for |x| < 2^-125 and must be rounded to nearest even, not
    # flushed. j0 stays 1 and y0 finite, both signs and the range edges.
    with DeviceContext() as ctx:
        var xs: List[Float32] = [
            1.401298464324817e-45,
            4.203895392974451e-45,
            1.0448781999237998e-40,
            1.1754942106924411e-38,
            1.1754943508222875e-38,
            1.175494490952134e-38,
            1.4693679385278594e-38,
            -5.877471754111438e-39,
        ]
        _check[1](
            xs,
            _run[j1](ctx, xs),
            [
                0.0,
                2.802596928649634e-45,
                5.224320934695783e-41,
                5.877471754111438e-39,
                5.877471754111438e-39,
                5.877471754111438e-39,
                7.346839692639297e-39,
                -2.938735877055719e-39,
            ],
            "j1",
        )
        var ones = List[Float32]()
        for _ in range(len(xs)):
            ones.append(1.0)
        _check[1](xs, _run[j0](ctx, xs), ones, "j0")

        var bxs: List[BFloat16] = [
            9.183549615799121e-41,
            4.5917748078995606e-40,
            5.877471754111438e-39,
            1.1663108012064884e-38,
            1.1754943508222875e-38,
            1.1846779004380866e-38,
            2.350988701644575e-38,
        ]
        _check[1](
            bxs,
            _run[j1](ctx, bxs),
            [
                0.0,
                1.8367099231598242e-40,
                2.938735877055719e-39,
                5.877471754111438e-39,
                5.877471754111438e-39,
                5.877471754111438e-39,
                1.1754943508222875e-38,
            ],
            "j1",
        )


def test_subnormal_inputs_f64() raises:
    comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074
    comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            var positive: List[Float64] = [f64_min, f64_max_sub, 8.0]
            _check[1](
                positive,
                _run[y1](ctx, positive),
                [-inf[.float64](), -2.861117485757029e307, -0.1580604617312475],
                "y1",
            )
            _check[1](
                positive,
                _run[y0](ctx, positive),
                [-473.9990734230043, -451.0529710071285, 0.22352148938756622],
                "y0",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
