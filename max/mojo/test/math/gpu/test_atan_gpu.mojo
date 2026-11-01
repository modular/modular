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

from std.math import (
    atan,
    atan2,
    ceildiv,
    cos,
    exp2,
    inf,
    isinf,
    isnan,
    nan,
    nextafter,
    pi,
    sin,
)
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import TestSuite, assert_equal
from test_utils.math import libm_call_binary

from std.utils.coord import Coord

comptime LOG2_10 = 3.321928094887362
comptime atan2_libm = libm_call_binary["atan2f", "atan2"]


def _int_dtype[dtype: DType]() -> DType:
    comptime if dtype == .float64:
        return .int64
    elif dtype == .float32:
        return .int32
    else:
        return .int16


def _ulp_diff[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Int where dtype.is_floating_point():
    """Returns the ULP distance between `a` and `b`, or -1 when exactly one is
    NaN or their signs differ (atan and atan2 never flip the sign, so this also
    catches signed-zero mismatches)."""
    if isnan(a) or isnan(b):
        return 0 if isnan(a) and isnan(b) else -1
    if isinf(a) or isinf(b):
        return 0 if a == b else -1
    comptime itype = _int_dtype[dtype]()
    var ia = Int(bitcast[itype, 1](a))
    var ib = Int(bitcast[itype, 1](b))
    if (ia < 0) != (ib < 0):
        return -1
    return abs(ia - ib)


def _max_ulp[dtype: DType](name: StaticString) -> Int:
    # Compare approximations with libm, which has its own rounding error.
    # f16/bf16 compute in f32 and allow one ULP of double rounding.
    comptime if dtype in (DType.float32, DType.float64):
        return 2 if name.startswith("atan2") else 1
    return 1


def _padded(n: Int) -> Int:
    comptime pad_multiple = 1024
    return ceildiv(n, pad_multiple) * pad_multiple


def _check[
    dtype: DType
](
    name: StaticString,
    got: Scalar[dtype],
    expected: Scalar[dtype],
    y: Float64,
    x: Float64,
    mut worst: Int,
) raises where dtype.is_floating_point():
    var d = _ulp_diff(got, expected)
    if d < 0 or d > _max_ulp[dtype](name):
        raise Error(
            t"{name}[{dtype}]({y}, {x}): got {got}, expected {expected},"
            t" ulp={d}"
        )
    worst = max(worst, d)


def _specials() -> List[Float64]:
    var v: List[Float64] = [
        0.0,
        1.0,
        0.5,
        2.0,
        0.4142135623730950,
        0.41421357,
        0.66,
        2.414213562373095,
        2.4142137,
        1e-5,
        1e-30,
        1e30,
        1e300,
        5e-324,
        1.401298464324817e-45,
        Float32(from_bits=UInt32(3)).cast[.float64](),
        Float32(from_bits=UInt32(4)).cast[.float64](),
        6.0e-8,
        # Subnormals of float16, bfloat16 and float32, whose atan is the input.
        3.0e-5,
        6.09e-5,
        9.2e-41,
        1.8367e-40,
        1.0e-39,
        5.0e-39,
        1.1754942e-38,
        65504.0,
        3.4028234663852886e38,
        1.7976931348623157e308,
        inf[DType.float64](),
    ]
    # The atan reduction breakpoints 7/16, 11/16, 19/16, 39/16, and the old
    # Cephes worst case near 0.432, with their float32 and float64 neighbors.
    var edges: List[Float64] = [
        0.4375,
        0.6875,
        1.1875,
        2.4375,
        0.43206868,
        0.44,
    ]
    for e in edges:
        var f = e.cast[.float32]()
        v.append(nextafter(f, 0.0).cast[.float64]())
        v.append(nextafter(f, 4.0).cast[.float64]())
        v.append(nextafter(e, 0.0))
        v.append(nextafter(e, 4.0))
        v.append(e)
    var out = List[Float64]()
    for x in v:
        out.append(x)
        out.append(-x)
    out.append(nan[DType.float64]())
    return out^


def _atan_inputs() -> List[Float64]:
    comptime num_sweep = 8192
    var out = _specials()
    # Linear sweep across the reduction splits.
    for i in range(num_sweep):
        out.append(-8.0 + 16.0 * Float64(i) / num_sweep)
    # Logarithmic sweep across magnitudes.
    for i in range(num_sweep):
        var e = -40.0 + 80.0 * Float64(i) / num_sweep
        var m = exp2(e * LOG2_10)
        out.append(m)
        out.append(-m)
    return out^


def _test_atan[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var vals = _atan_inputs()
    var n = len(vals)
    var length = _padded(n)
    var in_dev = ctx.enqueue_create_buffer[dtype](length)
    var out_dev = ctx.enqueue_create_buffer[dtype](length)
    with in_dev.map_to_host() as h:
        for i in range(length):
            h[i] = vals[i].cast[dtype]() if i < n else 0

    var in_span = Span(unsafe_ptr=in_dev.unsafe_ptr(), length=length)
    var out_span = Span(unsafe_ptr=out_dev.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_span, var in_span}:
        var idx = Int(idx0[0])
        var v = in_span.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_span.unsafe_ptr().unsafe_store[width=simd_width](idx, atan(v))

    comptime pack = simd_width_of[dtype, target=ctx.target]()
    elementwise[pack, target="gpu"](func, Coord(length), ctx)

    var worst = 0
    with in_dev.map_to_host() as ih, out_dev.map_to_host() as oh:
        for i in range(n):
            var x = ih[i]
            # _call_libm rounds f16/bf16 through atanf.
            var expected = _call_libm["atan"](x)
            var xf = x.cast[.float64]()
            _check("atan", oh[i], expected, 0.0, xf, worst)
    print("atan", dtype, "max ulp vs libm:", worst, "over", n, "inputs")


def _atan2_inputs() -> Tuple[List[Float64], List[Float64]]:
    var ys = List[Float64]()
    var xs = List[Float64]()
    var sp = _specials()
    for a in sp:
        for b in sp:
            ys.append(a)
            xs.append(b)
    # Points on circles of varied radius, densely in angle.
    var radii: List[Float64] = [1e-30, 1e-6, 0.3, 1.0, 7.5, 1e5, 1e30]
    for r in radii:
        for i in range(2048):
            var th = -pi + 2.0 * pi * (Float64(i) + 0.5) / 2048.0
            ys.append(r * sin(th))
            xs.append(r * cos(th))
    # Former worst cases, where the ratio lands near an atan breakpoint.
    var worst: List[Float64] = [
        3.0178835,
        6.7271781,
        203.92819,
        149.56111,
        -3.8479858657274626,
        3.695254715704384,
        1e-310,
        3e-310,
    ]
    for i in range(0, len(worst), 2):
        ys.append(worst[i])
        xs.append(worst[i + 1])
        ys.append(worst[i + 1])
        xs.append(-worst[i])
    # Extreme aspect ratios.
    for i in range(512):
        var e = -30.0 + 60.0 * Float64(i) / 512.0
        var s = exp2(e * LOG2_10)
        ys.append(s)
        xs.append(1.0)
        ys.append(-1.0)
        xs.append(-s)
    return (ys^, xs^)


def _test_atan2[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var inputs = _atan2_inputs()
    var ys = inputs[0].copy()
    var xs = inputs[1].copy()
    var n = len(ys)
    var length = _padded(n)
    var y_dev = ctx.enqueue_create_buffer[dtype](length)
    var x_dev = ctx.enqueue_create_buffer[dtype](length)
    var out_dev = ctx.enqueue_create_buffer[dtype](length)
    with y_dev.map_to_host() as yh, x_dev.map_to_host() as xh:
        for i in range(length):
            yh[i] = ys[i].cast[dtype]() if i < n else 0
            xh[i] = xs[i].cast[dtype]() if i < n else 1

    var y_span = Span(unsafe_ptr=y_dev.unsafe_ptr(), length=length)
    var x_span = Span(unsafe_ptr=x_dev.unsafe_ptr(), length=length)
    var out_span = Span(unsafe_ptr=out_dev.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_span, var y_span, var x_span}:
        var idx = Int(idx0[0])
        var y = y_span.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var x = x_span.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_span.unsafe_ptr().unsafe_store[width=simd_width](idx, atan2(y, x))

    comptime pack = simd_width_of[dtype, target=ctx.target]()
    elementwise[pack, target="gpu"](func, Coord(length), ctx)

    var worst = 0
    with y_dev.map_to_host() as yh, x_dev.map_to_host() as xh:
        with out_dev.map_to_host() as oh:
            for i in range(n):
                var y = yh[i]
                var x = xh[i]
                var expected = atan2_libm(y, x)
                var yf = y.cast[.float64]()
                var xf = x.cast[.float64]()
                _check("atan2", oh[i], expected, yf, xf, worst)
    print("atan2", dtype, "max ulp vs libm:", worst, "over", n, "inputs")


def _parse_hex(digits: StringSlice) -> UInt64:
    var v: UInt64 = 0
    for c in digits.codepoint_slices():
        var d = ord(c)
        v = v * 16 + UInt64(d - 48 if d <= 57 else d - 55)
    return v


def _test_atan2_reference[
    dtype: DType
](
    ctx: DeviceContext, table: StaticString
) raises where dtype.is_floating_point():
    comptime max_reference_ulp = 1
    comptime utype: DType = .uint64 if dtype == .float64 else .uint32
    var ys = List[Scalar[dtype]]()
    var xs = List[Scalar[dtype]]()
    var expected = List[Scalar[dtype]]()
    for line in table.splitlines():
        var cols = line.split()
        if len(cols) != 3:
            continue
        ys.append(bitcast[dtype, 1](Scalar[utype](_parse_hex(cols[0]))))
        xs.append(bitcast[dtype, 1](Scalar[utype](_parse_hex(cols[1]))))
        expected.append(bitcast[dtype, 1](Scalar[utype](_parse_hex(cols[2]))))
    var n = len(ys)
    var length = _padded(n)
    var y_dev = ctx.enqueue_create_buffer[dtype](length)
    var x_dev = ctx.enqueue_create_buffer[dtype](length)
    var out_dev = ctx.enqueue_create_buffer[dtype](length)
    with y_dev.map_to_host() as yh, x_dev.map_to_host() as xh:
        for i in range(length):
            yh[i] = ys[i] if i < n else 0
            xh[i] = xs[i] if i < n else 1

    var y_span = Span(unsafe_ptr=y_dev.unsafe_ptr(), length=length)
    var x_span = Span(unsafe_ptr=x_dev.unsafe_ptr(), length=length)
    var out_span = Span(unsafe_ptr=out_dev.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_span, var y_span, var x_span}:
        var idx = Int(idx0[0])
        var y = y_span.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var x = x_span.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_span.unsafe_ptr().unsafe_store[width=simd_width](idx, atan2(y, x))

    comptime pack = simd_width_of[dtype, target=ctx.target]()
    elementwise[pack, target="gpu"](func, Coord(length), ctx)

    var worst = 0
    with out_dev.map_to_host() as oh:
        for i in range(n):
            var d = _ulp_diff(oh[i], expected[i])
            if d < 0 or d > max_reference_ulp:
                raise Error(
                    t"atan2 reference[{dtype}]({ys[i]}, {xs[i]}): got {oh[i]},"
                    t" expected {expected[i]}, ulp={d}"
                )
            worst = max(worst, d)
    print("atan2", dtype, "max ulp vs MPFR:", worst, "over", n, "points")


def test_atan() raises:
    with DeviceContext() as ctx:
        _test_atan[.float32](ctx)
        _test_atan[.float16](ctx)
        _test_atan[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _test_atan[.float64](ctx)


def test_atan2() raises:
    with DeviceContext() as ctx:
        _test_atan2[.float32](ctx)
        _test_atan2[.float16](ctx)
        _test_atan2[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _test_atan2[.float64](ctx)


def test_atan2_reference() raises:
    # Correctly rounded atan2 values, computed with MPFR, for the points where the
    # implementation's measured error is largest (about 1.15 ulp) plus a random
    # sample. libm is itself off by up to an ulp, so the comparison above cannot
    # catch a drift from 1.25 to 1.9 ulp; these references can. Each row is
    # "y x expected" as hex bit patterns.
    comptime reference_f32: StaticString = """
        C0584530 4026856F BF6A25B0
        405E0250 402B6C18 3F69CC64
        411980FF 40D35060 3F77CC8B
        C0D49B31 4098E867 BF7281FC
        C0E5BCB7 40ABBC2D BF6DCB02
        40BC128D 408C6178 3F6DF9B4
        C084256E 40574D6F BF631F20
        40D0A8D4 409F6D03 3F6B19F9
        41125B6B 40D35B5B 3F7206F6
        40F4BED3 40E085D2 3F541652
        4064AA47 406064FC 3F4B7999
        40A06C8C 40E81D4B 3F1AD14D
        BEAE56C3 40A9F638 BD831DF5
        C0F843F6 40D69F08 BF5BA2E2
        404638D3 40C28876 3EF141A6
        40D7744A 40AEF374 3F6386C3
        388739E4 3D76852F 3A8C6D15
        C0C13EF2 40932050 BF6B8A15
        450622A6 471D1E66 3D5A585C
        4047BC96 411D8D9E 3E9D24AD
        C1140EF8 40D5C986 BF720AD6
        C0B38485 40A8E29B BF50DF3F
        40A5FB3F 411B7512 3EFB104A
        3EB16934 411FF6B1 3D0DE77B
        C0F2C02F 40E473BB BF50D3EF
        C03F8B66 408CC0DF BF18F3A7
        40703FD5 40201EB4 3F7BA11D
        40F660A0 40AD9E2C 3F74F9AC
        40BD02DB 4097D2CB 3F64E22D
        C113866F 411330CE BF495A3C
        411E508C 40DDA432 3F75C70C
        C0639EAB 4032712D BF67EA61
        C09BFC92 405DBB98 BF73F108
        4116C8AA 410ACABD 3F53A847
        40A7D91F 406E3D37 3F741F0E
        40DAD8A8 409E4E48 3F71CEB9
        4111760A 40C0C95C 3F7C4DDB
        4109CAA8 4105483D 3F4D51FA
        BFD4E38E 4050BF81 BEF16F17
        C0774F90 41171672 BEC6DFE8
        C115A5D8 40FE346E BF5DDEF9
        40D2813D 40AEB749 3F60C70A
        407437A8 404002B1 3F678DF2
        4093920A 40779449 3F5F7084
        4116417F 40CB5575 3F79D69C
        C08967AB 411D3945 BED2ED72
        3FDC5B0D 4053B7DD 3EF5ACA8
        3F821D3A 405AC200 3E940022
        3F98AF6A 40C3E647 3E450F08
        C022152C 40C33D3C BEC97199
        BFB8A4A1 4070F680 BEBB5533
        22C5A891 230CC361 3F1CB49F
        3F5188E5 434A4766 3B849738
        3A1A72A9 3E4968FE 3B444F01
        3FFCDE43 40B03624 3EB05C57
        C10B69BA 40BEC30F BF788504
        410DF65E 40CD5D0C 3F71D16B
        BEF83D22 40C2B019 BDA2DD11
        4528A4E1 492CAD20 3B7A0557
        C0B4A8B5 410209FB BF1B6BED
        C1010603 40EBEBC3 BF548569
        4026E774 401F7EEC 3F4EDF13
        E36EEDA1 66951302 BC4D240E
        401A71FB 40C8C0C9 3EBC03D7
        C3EBB591 DBFEA337 C0490FDB
        B9FAB241 C712C59C C0490FDB
        C021BDCA 40976D02 BEFB25F6
        388933EF 3AA8FEF6 3D4FA92B
        B6B826D4 AFC71280 BFC91204
        4734CE2D 456134C0 3FBF1E0F
        4114040B BF1C3919 3FD17E49
        B5D25937 91964219 BFC90FDB
        A7E896CD FE84B54D C0490FDB
        3A448E29 3CA5B4E3 3D17C276
        3F9B12CB 3202B9AC 3FC90FDB
        BB8B8EEC B50F49DE BFC913F6
        BF4F266E 4008CFC6 BEB94626
        CC6C2FA5 D6CE8E46 C0490FD8
        C10B3CAB C03C3F6A BFF2C85F
        B922F45C 9BEC8FA7 BFC90FDB
        372569C2 31126A6F 3FC908C6
        40F38A41 4046A409 3F977F22
        8395383B 46A8A5C6 80007141
        C0743BC7 AB8000CA BFC90FDB
        BC97D88C A3E2B3E7 BFC90FDB
        C053A986 4049DBED BF4F2130
        408DAEAF C1062535 4029F763
        52DDB038 C86CCA33 3FC90FDF
        89085A33 0F004C24 B9880947
        4ECFF5E4 81F11F2A 3FC90FDB
        49065B10 3C99794D 3FC90FDA
        C0B7DBEC 4015B75F BF9791F5
        C09C74B9 40B249A9 BF3863AE
        374DEDD8 225D6388 3FC90FDB
        BEF44C4F D1CDB9CD C0490FDB
        417E3778 24535158 3FC90FDB
"""

    comptime reference_f64: StaticString = """
        402C8F0FF1BCCF08 40254CA256AA269D 3FEDC26F7350C0DD
        4021E09699496D22 4019B86D3C61689A 3FEE4F6C08DA66B6
        C0105F21955FCD13 400A5FD68F6D891A BFEC912150F6D812
        C004E67D3A8F9C20 400037FEE7A212A4 BFED25C3B639986B
        C0071181A51368CE 40019C4863C54712 BFED6694EB39C810
        C02326E88BB5C69D 401D2C643E142BA4 BFED6FC626542E45
        4014835487416E30 400B4FAD9C1DDAD0 3FEF788005120C22
        4019BDB87261F90C 40110410899ACFDC 3FEF930B33DE8CCF
        C02362A488B0702A 4020AAAD3E267AE5 BFEB8A8EF6EFEE88
        40015510E22BD620 3FF6F21D5D83D848 3FEF8DEDF4A81792
        C00A8F21C9CDCC2E 40196028A3B55131 BFDEDB320A9C128E
        C0124CB518F067B3 4009E919F6E9AE77 BFEE8D561C82B26E
        C021C96091AEF983 401993E097CE8340 BFEE514AEE1C1D6A
        402286D20472292C 4018E179F9613C66 3FEF57C6154C1EEA
        4015A9391C3BA624 400DAEA293A629B0 3FEF0B1AB816601F
        3F2C70E5DC5EE188 3F23ECDCD1AF5783 3FEEB5B33646BF63
        4018E9685A02D774 40228541380FF578 3FE2F22945A118A3
        40FA5E4F8D79BF5F 40F524E2F1570192 3FECA32474E5EA8C
        BFB402D0E873AE00 402359B3F48BA170 BF808BBD7425E670
        BFFF9D25F1B07A00 3FF7320899B52D1E BFEE0278EE6D9D70
        4021AC88B92D29D6 4018D50BEF341E4E 3FEEAB1F11D0E714
        C016D1D8C129EBB3 40120895653565C3 BFECDD5D681986DF
        4009497E9B5E7618 40022B0A8CD78FBF 3FEE5433DA2D3C26
        C015B0BB881ACB8D 40124454D014A647 BFEBDE2478C5EFE6
        401B26840E377E40 40189500B775F054 3FEAB85D1FCAF9CF
        C0119C3C31F478A6 4017B302C406828B BFE47320232BFA67
        4017DF81288E2AEE 40137311A4E248BE 3FEC638F9FBFA9FB
        3FF611292B74EBE0 4021ACE38ECE1F77 3FC3D0CA9E4CC04E
        C020279408D2D859 401CA439E590D05C BFEB0E570886205E
        C02345499D30DDF0 40230F3714A718C3 BFE94F1FB24B0C66
        C0111FFCDC140DBD 40174A78E9A9789C BFE449C0374D50CF
        C010C5886B9FCB29 4021B90915DB9646 BFDC48E902E31F99
        3FFA3CD4DC98F773 401C3DE6E357CE53 3FCD3611DBEF7816
        401EBBB19C8AD6F4 4014AF2409B34C3C 3FEF4F19EE4D53DF
        84E76915EDD7034B 050B63639457FE70 BFCAF279A1ABECAF
        400D8D18C4B6B3A4 4015FAF2357DAD80 3FE2F0559B5DD603
        4021859F15D76800 40196DA7F827B970 3FEE2DC2C2F18411
        40077739B756637C 401B89744CFEDB62 3FD9C7318BF4EA05
        C01739F48162724A 40130BE1E296F662 BFEC4963A4FAF828
        3FFA6476F08877A0 4011293FA9006C84 3FD77DD7E863C4D9
        BFF5D0FD348DAC38 4023A4B5946F56CF BFC1A83604F768D4
        3FB9A4A1CFCFBC00 3FF0A3B31FE72062 3FB894F10479A57E
        C0066C4C5FE631DC 4005E87B6D6E7472 BFE98120ACEE4326
        3FDAF4827A124520 401EFFE832D5B09A 3FABCC30A11BFF23
        3FF44D3A5B2E20B0 40234DDFFE2457E2 3FC0BB15E949F211
        401023BCFA63A538 400720576DA40C0D 3FEE5F2B38FC7F42
        3FEF54B3134F29B2 40BF2661368EC19C 3F2017CAB9D920BA
        C023015C705A0B94 401A22BB267BE6EA BFEEFD88C1E1EE74
        3F6D92E1CBE95DD0 3F6475921879028C 3FEEE620E7B519FC
        40116ED648BB1682 4010F0DF5C868B2C 3FE9973D3927C0B0
        3FF0FEB509A5A7F8 401E366F84BE7356 3FC1E212268D5F20
        C019930D991CA482 4012F77CF86C3FD4 BFEDD876D6A4AD0B
        401411AA0F32B7FE 400B9DE4B1E551BF 3FEEFAFDF43EA0D0
        C013300A090DD419 4021FBC57E4E8789 BFDF5D5498C20E50
        4013EF87C7C80DF6 40237585CBE770C3 3FDE4C2048C71F99
        3FEA1339D5EBE2F0 4019B42F014351AE 3FC0251A4D8AD079
        C021F25680FBC58C 4017816D4A47EC74 BFEFB64A47E98993
        E7271EC9D177F0A0 674D50CA42500FCE BFC8EAF2D62BEB8D
        C021AEC0B62374E3 402095F1F68D7DAD BFEA281CB5DADA7A
        3FCADBEEEA7DCAC0 40221F6C7DF06C69 3F97B5755C0A0CBE
        C017734CD0678FF0 40167E82E5DD2C9E BFE9CC734C59B16D
        40113371ADBEEEA8 400E3BF9E22758E5 3FEB3174FD400AFF
        C01B52350D7857E5 4018C4D7D78929A7 BFEAB30947D547D8
        BFD4BF3C353B58A0 4022A26D679B88E2 BFA1CE82E96BF7BB
        4080BEA7A5FC4944 3EEAF984721CCA15 3FF921FB4DD29475
        BF6FCF55CEC22BA7 BC456BDF35A3A970 BFF921FB54442D1B
        377BEFEAB666E88E D3D40A5040ED1DB1 400921FB54442D18
        BF76888D41AC163D BBD7A0141A9CC1DE BFF921FB54442D18
        C0012CC9B6DC0D80 3FE67807F13CFA14 BFF41344027B1BDC
        4017E809BA68D570 40097B7CBD93D400 3FF14C4D915FA791
        D66718F1EEB21E60 6E954A2D70C8133E A7C15BC93C8C991C
        F75A7861A54756F0 FAC8F50180BC6F6C C00921FB54442D18
        4000788F99F08BBC C00246818EFBD273 400343BFC4806AC3
        3EE02673B5AE0067 3C303B58C88B6178 3FF921FB54442B16
        1785EB4DAFBC9B12 92D98D1CF7DE5D02 3FF921FB54442D18
        3FFB7AFF4BC5E310 C01DA31CAF6DEAB1 40074F7AB6EBDCCE
        839DC45160F32B52 6F7B0734CF9819DC 8000000000000000
        3FA1CA722788B93A 3DADAC88FE850BC8 3FF921FB54297D38
        C01F91B69EE28109 C016A28622EB1D81 C0018AED5AF3AD65
        BEB2450448588F7C BB39AC69C574DD01 BFF921FB54442D18
        C0800CBB78C0737F C1403B23CBC95879 C009217CC23E327D
        400105314F894844 401CD094085649A2 3FD26116A907B19C
        3FFE8D713E553090 BFF54D4ED13F72D0 40016FF05DC855BB
        3FB4D54EC5DC2D33 3E4946CC4679E1EC 3FF921FB2D70FB95
        C557F6DBB53EF402 772D7D90825F06EB 8E1A00E8879C9BA0
        40237DC77739B756 C023C0ECCB610F03 4002E72C743CBE53
        C009FC5B304FF10C C01F0D69553343D0 C005F6651FD931C9
        4036520303E502E1 3D5F757F5370BCBC 3FF921FB54442CBE
        4611860E50926060 2F685CAFD8D95974 3FF921FB54442D18
        BF4A3980542408AE BBF1D4BCED639616 BFF921FB54442D19
        BECF768362422E73 BF227FE4B8AE05F5 C008EB9229B60027
        3EB13BCB7747BC56 3E43C5DBB5CEC06C 3FF8FD4484414605
        412CCDF4E8BF07BB 4488EB56F023D39F 3C927E9DD309A1FB
        3FCBE42D7C20BD27 3F99C3C50364B0BD 3FF74B1ADD4D47C8
        B572E1459BE5A776 09CD8F8703AEDDC1 BFF921FB54442D18
        4002726CE54B6394 C021537A14F683FC 40070D360DEC9BBD
"""
    with DeviceContext() as ctx:
        _test_atan2_reference[.float32](ctx, reference_f32)
        comptime if not default_accelerator().is_apple_gpu():
            _test_atan2_reference[.float64](ctx, reference_f64)


def test_tiny_result_bits() raises:
    var y_bits: List[UInt32] = [
        0x00800000,
        0x80800000,
        0x007FFFFF,
        0x807FFFFF,
        1,
        3,
        5,
        7,
        1,
        1,
        1,
        0x80000001,
    ]
    var x_bits: List[UInt32] = [
        0x40000000,
        0x40000000,
        0x3F800000,
        0x3F800000,
        0x40000000,
        0x40000000,
        0x40000000,
        0x40000000,
        0x3FFFFFFF,
        0x40000000,
        0x40000001,
        0x40000000,
    ]
    var expected: List[UInt32] = [
        0x00400000,
        0x80400000,
        0x007FFFFF,
        0x807FFFFF,
        0,
        1,
        2,
        3,
        1,
        0,
        0,
        0x80000000,
    ]
    with DeviceContext() as ctx:
        var n = len(y_bits)
        var yd = ctx.enqueue_create_buffer[DType.float32](n)
        var xd = ctx.enqueue_create_buffer[DType.float32](n)
        var ad = ctx.enqueue_create_buffer[DType.float32](n)
        var bd = ctx.enqueue_create_buffer[DType.float32](n)
        with yd.map_to_host() as yh, xd.map_to_host() as xh:
            for i in range(n):
                yh[i] = Float32(from_bits=y_bits[i])
                xh[i] = Float32(from_bits=x_bits[i])
        var ys = Span(unsafe_ptr=yd.unsafe_ptr(), length=n)
        var xs = Span(unsafe_ptr=xd.unsafe_ptr(), length=n)
        var a = Span(unsafe_ptr=ad.unsafe_ptr(), length=n)
        var b = Span(unsafe_ptr=bd.unsafe_ptr(), length=n)

        @inline(.always)
        def func[
            width: Int, alignment: Int = 1
        ](idx0: Coord) {var ys, var xs, var a, var b}:
            var i = Int(idx0[0])
            var y = ys.unsafe_ptr().unsafe_load[width=width](i)
            var x = xs.unsafe_ptr().unsafe_load[width=width](i)
            a.unsafe_ptr().unsafe_store[width=width](i, atan(y))
            b.unsafe_ptr().unsafe_store[width=width](i, atan2(y, x))

        elementwise[4, target="gpu"](func, Coord(n), ctx)
        with ad.map_to_host() as ah, bd.map_to_host() as bh:
            for i in range(n):
                assert_equal(bitcast[.uint32, 1](ah[i]), y_bits[i])
                assert_equal(bitcast[.uint32, 1](bh[i]), expected[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
