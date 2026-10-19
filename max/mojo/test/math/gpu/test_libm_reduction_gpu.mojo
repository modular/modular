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

# Checks the pi/2 argument reduction behind `tan` on GPU and host at the floats
# closest to multiples of pi/2, where the reduced
# argument cancels to a few bits and any error in the reduction is magnified.
# GPU backends contract a multiply into a dependent add, so a reduction that
# only works when every product is rounded separately fails here.
#
# The hard cases below were found by exhaustive search over k with exact
# integer arithmetic: for each k, the float nearest k*pi/2, ranked by
# |x - k*pi/2| / k (the reduction error grows with k) and by |x - k*pi/2|.
# They cover k < 2^21 for float32 and k < 2^23 for float64, which spans both
# the Cody-Waite and the Payne-Hanek paths, plus the known worst case over
# the whole format.

from std.builtin.dtype import _integral_type_of
from std.math import align_up, inf, isinf, isnan, nan, tan, ulp
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from std.utils.coord import Coord

comptime max_ulp = 2.0
comptime pio2_hi = 1.5707963267948966
comptime pio2_lo = 6.123233995736766e-17
comptime switch_sweep_radius = 1500

comptime _HARD_F32: List[Float64] = [
    1.5707963705062866,
    3.1415927410125732,
    4.71238899230957,
    9.42477798461914,
    14.137166976928711,
    18.84955596923828,
    23.56194496154785,
    28.274333953857422,
    37.69911193847656,
    47.1238899230957,
    215.1990966796875,
    252.89820861816406,
    290.5973205566406,
    505.7964172363281,
    796.3937377929688,
    1011.5928344726562,
    1807.986572265625,
    2023.1856689453125,
    2238.384765625,
    2668.782958984375,
    3615.97314453125,
    4046.371337890625,
    4476.76953125,
    5337.56591796875,
    5854.35791015625,
    6715.154296875,
    7231.9462890625,
    7575.95068359375,
    8092.74267578125,
    8953.5390625,
    9470.3310546875,
    10675.1318359375,
    11191.923828125,
    11708.7158203125,
    12225.5078125,
    12913.5166015625,
    13430.30859375,
    13947.1005859375,
    14463.892578125,
    15151.9013671875,
    15668.693359375,
    16185.4853515625,
    16702.27734375,
    17907.078125,
    18940.662109375,
    20145.462890625,
    21179.046875,
    22383.84765625,
    23417.431640625,
    23588.6484375,
    24451.015625,
    24622.232421875,
    25655.81640625,
    34609.35546875,
    39086.125,
    43562.89453125,
    48039.6640625,
    52516.43359375,
    56993.203125,
    61469.97265625,
    65946.7421875,
    69218.7109375,
    74900.28125,
    78172.25,
    83853.8203125,
    87125.7890625,
    92807.359375,
    96079.328125,
    101760.8984375,
    105032.8671875,
    108304.8359375,
    110714.4375,
    113986.40625,
    117258.375,
    119667.9765625,
    122939.9453125,
    126211.9140625,
    128621.515625,
    131893.484375,
    135165.453125,
    138437.421875,
    149800.5625,
    153072.53125,
    156344.5,
    167707.640625,
    170979.609375,
    174251.578125,
    177523.546875,
    185614.71875,
    188886.6875,
    192158.65625,
    195430.625,
    203521.796875,
    210065.734375,
    267058.9375,
    306145.0625,
    381045.34375,
    420131.46875,
    459217.59375,
    495031.75,
    534117.875,
    573204.0,
    612290.125,
    723004.5625,
    762090.6875,
    801176.8125,
    840262.9375,
    879349.0625,
    918435.1875,
    990063.5,
    1029149.625,
    1068235.75,
    1107321.875,
    1146408.0,
    1185494.125,
    1224580.25,
    1446009.125,
    1485095.25,
    1524181.375,
    1563267.5,
    1602353.625,
    1641439.75,
    1680525.875,
    1719612.0,
    1758698.125,
    1797784.25,
    1836870.375,
    1941040.875,
    1980127.0,
    2019213.125,
    2058299.25,
    2136471.5,
    2214643.75,
    2292816.0,
    2370988.25,
    2449160.5,
    2475158.75,
    2553331.0,
    2631503.25,
    2709675.5,
    2787847.75,
    2866020.0,
    2892018.25,
    2944192.25,
    2970190.5,
    3022364.5,
    3048362.75,
    3126535.0,
    3204707.25,
    3282879.5,
]

comptime _HARD_F64: List[Float64] = [
    1.5707963267948966,
    3.141592653589793,
    4.71238898038469,
    45.553093477052,
    51.83627878423159,
    91.106186954104,
    182.212373908208,
    364.424747816416,
    728.849495632832,
    1457.698991265664,
    2915.397982531328,
    5830.795965062656,
    11661.591930125313,
    22743.560015663308,
    46066.74387591393,
    91553.86390724055,
    92133.48775182787,
    138200.2316277418,
    183107.7278144811,
    184266.97550365573,
    229174.47169039503,
    230333.71937956967,
    276400.4632554836,
    321307.9594422229,
    366215.4556289622,
    368533.95100731147,
    413441.44719405076,
    458348.94338079006,
    460667.43875913933,
    505574.93494587863,
    552800.9265109672,
    597708.4226977065,
    642615.9188844458,
    687523.4150711851,
    732430.9112579244,
    737067.9020146229,
    781975.3982013622,
    826882.8943881015,
    871790.3905748408,
    916697.8867615801,
    961605.3829483194,
    966242.373705018,
    1011149.8698917573,
    1150509.3492086737,
    1195416.845395413,
    1240324.3415821523,
    1285231.8377688916,
    1330139.333955631,
    1375046.8301423702,
    1419954.3263291095,
    1464861.8225158488,
    1519043.3002159852,
    1563950.7964027245,
    1608858.2925894638,
    1698673.2849629424,
    1743580.7811496817,
    2525556.179351044,
    2570463.675537783,
    2615371.1717245225,
    3352439.0737391454,
    3397346.5699258847,
    3442254.066112624,
    4179321.968127247,
    4269136.960500726,
    5051112.358702088,
    5140927.351075566,
    5833087.75690345,
    5922902.749276929,
    6012717.741650407,
    6704878.147478291,
    6794693.139851769,
    6884508.132225248,
    7486853.545679653,
    7576668.538053132,
    7666483.53042661,
    7756298.522800089,
    8358643.936254494,
    8538273.921001451,
    9230434.326829335,
    9410064.311576292,
    10102224.717404176,
    10281854.702151133,
    10974015.107979016,
    11153645.092725974,
    11666175.5138069,
    11845805.498553857,
    12025435.483300814,
    12537965.90438174,
    12717595.889128698,
    12897225.873875655,
]


def _step[
    dtype: DType, //
](x: Scalar[dtype], n: Int) -> Scalar[dtype] where dtype.is_floating_point():
    """Moves a finite x by n representable values away from zero."""
    comptime int_dtype = _integral_type_of[dtype]()
    return bitcast[dtype](bitcast[int_dtype](x) + Scalar[int_dtype](n))


def _ulp_error[
    dtype: DType, //
](got: Scalar[dtype], want: Float64) -> Float64 where dtype.is_floating_point():
    """Measures error in output-dtype ULP against the float64 reference."""
    var got64 = got.cast[.float64]()
    if isnan(want) or isnan(got64):
        return 0.0 if isnan(want) and isnan(got64) else inf[.float64]()
    if got64 == want:
        if got64 == 0 and 1.0 / got64 != 1.0 / want:
            return inf[.float64]()
        return 0.0
    if isinf(got64) or isinf(want):
        return inf[.float64]()
    return abs(got64 - want) / ulp(want.cast[dtype]()).cast[.float64]()


def test_ulp_error() raises:
    var want = Float64(-2.133485385753704e18)
    var seven_ulp_error = Float64(-2.1334853857537021e18)
    assert_equal(_ulp_error(seven_ulp_error, want), Float64(7))
    assert_true(_ulp_error(seven_ulp_error, want) > max_ulp)
    assert_equal(_ulp_error(_step(want, 1), want), Float64(1))
    assert_equal(_ulp_error(Float64(0), -inf[.float64]()), inf[.float64]())
    assert_equal(_ulp_error(inf[.float64](), -inf[.float64]()), inf[.float64]())
    assert_equal(_ulp_error(nan[.float64](), want), inf[.float64]())
    assert_equal(_ulp_error(want, nan[.float64]()), inf[.float64]())
    assert_equal(_ulp_error(-inf[.float64](), -inf[.float64]()), Float64(0))
    assert_equal(_ulp_error(Float64(0), Float64(-0.0)), inf[.float64]())
    assert_equal(
        _ulp_error(Float32(1.0), 1.0 + 5.960464477539063e-8), Float64(0.5)
    )


def _check[
    dtype: DType, //
](
    ctx: DeviceContext,
    label: String,
    inputs: List[Scalar[dtype]],
    wants: List[Float64] = [],
) raises where dtype.is_floating_point():
    """Checks `tan` on GPU and host against `wants`, or float64 libm when
    `wants` is empty."""
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var n = len(inputs)
    var padded = align_up(n, pack_size)

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
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, tan(val))

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    var worst_gpu = 0.0
    var worst_host = 0.0
    var worst_x = Scalar[dtype](0)
    var failures = 0
    with out_device.map_to_host() as out_host:
        for i in range(n):
            var x = inputs[i]
            var xd = x.cast[.float64]()
            # float32 is judged against float64 libm, so its bound includes
            # the final rounding.
            var want = wants[i] if len(wants) > 0 else _call_libm["tan"](xd)
            var got = out_host[i]
            var host = tan(x)
            var e_gpu = _ulp_error(got, want)
            var e_host = _ulp_error(host, want)
            if e_gpu > worst_gpu:
                worst_gpu = e_gpu
                worst_x = x
            worst_host = max(worst_host, e_host)
            if (e_gpu > max_ulp or e_host > max_ulp) and failures < 20:
                failures += 1
                print(
                    t"tan {dtype} {label} x = {x} gpu = {got} ({e_gpu} ulp)",
                    t"host = {host} ({e_host} ulp) libm = {want}",
                )
    print(
        t"tan {dtype} {label} n = {n} max ulp gpu = {worst_gpu} at",
        t"{worst_x} host = {worst_host}",
    )
    assert_true(
        worst_gpu <= max_ulp and worst_host <= max_ulp,
        String(t"tan {dtype} {label} exceeded {max_ulp} ulp"),
    )


def _with_neighbors[
    dtype: DType
](xs: List[Float64], signed: Bool) -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Returns each x in `xs` rounded to `dtype`, its 2 neighbors on each side,
    and their negations when `signed`."""
    var v = List[Scalar[dtype]]()
    for x in xs:
        var c = Scalar[dtype](x)
        for d in range(-2, 3):
            var y = _step(c, d)
            v.append(y)
            if signed:
                v.append(-y)
    return v^


def _near_multiples(k_lo: Int, k_hi: Int, k_step: Int) -> List[Float64]:
    """Returns k*pi/2 for k in range(k_lo, k_hi, k_step), from
    k * (pio2_hi + pio2_lo) in double-word arithmetic, so that the nearest
    float64 (and, rounded once more, float32) is within an ulp."""
    var v = List[Float64]()
    for k in range(k_lo, k_hi, k_step):
        var kf = Float64(k)
        var hi = kf * pio2_hi
        var lo = kf.fma(pio2_hi, -hi) + kf * pio2_lo
        v.append(hi + lo)
    return v^


def _run_all[
    dtype: DType
](
    ctx: DeviceContext,
    hard: List[Float64],
    worst_x: Float64,
    worst_tan: Float64,
    small_limit_k: Int,
    sample_k: Int,
) raises where dtype.is_floating_point():
    var hard_tan = _with_neighbors[dtype](hard, True)
    _check(ctx, "hard", hard_tan)
    # The float closest to a multiple of pi/2 in the whole format. libm is
    # not reliable there (glibc's tan is off by 7.5 ulp in float64), so the
    # expected value is -cot(x - n*pi/2) rounded once, computed offline.
    var worst: List[Scalar[dtype]] = [
        Scalar[dtype](worst_x),
        -Scalar[dtype](worst_x),
    ]
    var worst_want: List[Float64] = [worst_tan, -worst_tan]
    _check(ctx, "worst case", worst, worst_want)

    # Sweeps of k*pi/2: small k, the reduction switch, and Payne-Hanek samples.
    var sweep = _near_multiples(1, 3000, 1)
    sweep.extend(
        _near_multiples(
            small_limit_k - switch_sweep_radius,
            small_limit_k + switch_sweep_radius,
            1,
        )
    )
    sweep.extend(
        _near_multiples(
            sample_k - switch_sweep_radius,
            sample_k + switch_sweep_radius,
            1,
        )
    )
    sweep.extend(_near_multiples(3001, 50 * sample_k, 50 * sample_k // 2000))
    _check(ctx, "k*pi/2 sweep", _with_neighbors[dtype](sweep, True))


# Cody-Waite ends at |x| = 24576 (k = 15646). The larger k samples the
# Payne-Hanek path at a historical switch location.
comptime f32_cody_waite_limit_k = 15646
comptime f32_payne_hanek_sample_k = 83443
comptime f32_worst_x = 7.729178919452902e28  # 16367173 * 2^72
comptime f32_worst_tan = -619283328.0

# Cody-Waite ends at |x| = 1572864 (k = 1001319). The larger k samples the
# Payne-Hanek path at |x| = 2^40, a historical switch location.
comptime f64_cody_waite_limit_k = 1001319
comptime f64_payne_hanek_sample_k = 699982214720
comptime f64_worst_x = 5.319372648326541e255  # 6381956970095103 * 2^797
comptime f64_worst_tan = -2.133485385753704e18


def test_reduction_float32() raises:
    with DeviceContext() as ctx:
        _run_all[.float32](
            ctx,
            materialize[_HARD_F32](),
            f32_worst_x,
            f32_worst_tan,
            f32_cody_waite_limit_k,
            f32_payne_hanek_sample_k,
        )


def test_reduction_float64() raises:
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            _run_all[.float64](
                ctx,
                materialize[_HARD_F64](),
                f64_worst_x,
                f64_worst_tan,
                f64_cody_waite_limit_k,
                f64_payne_hanek_sample_k,
            )


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    # Apple GPUs have no float64 support; the body is empty there, so report
    # the test as skipped rather than passed.
    comptime if default_accelerator().is_apple_gpu():
        suite.skip[test_reduction_float64]()
    suite^.run()
