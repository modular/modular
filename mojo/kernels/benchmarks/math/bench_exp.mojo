# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import *
from math.math import _exp_taylor, _simd_apply
from builtin.range import _StridedRange
from sys.arg import argv

from algorithm.functional import vectorize
from benchmark import keep
from mojobench import Bencher, BenchId, MojoBench


fn apply[
    func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, width],
    type: DType,
](input: Buffer[type], output: Buffer[type]):
    @parameter
    fn _func[width: Int](idx: Int):
        output.simd_store(idx, func(input.simd_load[width](idx)))

    vectorize[_func, simdwidthof[type]()](len(input))


def bench_unary[
    func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, width],
    type: DType,
](inout m: MojoBench, size_range: _StridedRange, op_name: String):
    for i in size_range:
        bench_unary[func, type](m, i, op_name)


def bench_unary[
    func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, width],
    type: DType,
](inout m: MojoBench, size: Int, op_name: String):
    alias alignment = 64
    var input_ptr = DTypePointer[type].alloc(size, alignment=alignment)
    var output_ptr = DTypePointer[type].alloc(size, alignment=alignment)

    @parameter
    fn bench(inout b: Bencher, size: Int):
        @parameter
        fn iter_fn():
            apply[func](
                Buffer[type](input_ptr, size),
                Buffer[type](output_ptr, size),
            )
            keep(output_ptr)

        b.iter[iter_fn]()

    m.bench_with_input[Int, bench](
        BenchId(op_name, str(size)),
        size,
        throughput_elems=size * sizeof[type](),
    )

    DTypePointer[type].free(input_ptr)
    DTypePointer[type].free(output_ptr)


fn ldexp2kf_opt[
    dtype: DType, simd_width: Int
](x_in: SIMD[dtype, simd_width], q_in: SIMD[DType.int32, simd_width]) -> SIMD[
    dtype, simd_width
]:
    var m = q_in >> 31
    m = (((m + q_in) >> 6) - m) << 4
    var q = q_in - (m << 2)
    m += 127
    if m < 0:
        m = 0

    # m = m <   0 ?   0 : m;
    # m = m > 255 ? 255 : m;
    if m > 255:
        m = 255

    #   u = intBitsToFloat(((int32_t)m) << 23);
    var u = bitcast[dtype, simd_width, DType.int32, simd_width](m << 23)
    var x = x_in * u * u * u * u
    #   u = intBitsToFloat(((int32_t)(q + 0x7f)) << 23);
    var xu = (
        ((q + SIMD[DType.int32, simd_width](0x7F)).cast[DType.int32]()) << 23
    )
    return x * xu.cast[dtype]()


fn pow2if[
    simd_width: Int
](q: SIMD[DType.int32, simd_width]) -> SIMD[DType.float32, simd_width]:
    var x = (
        ((q + SIMD[DType.int32, simd_width](0x7F)).cast[DType.int32]())
    ) << 23
    return bitcast[DType.float32, simd_width, DType.int32, simd_width](x)


fn ldexp2kf[
    dtype: DType, simd_width: Int
](d: SIMD[dtype, simd_width], e: SIMD[DType.int32, simd_width]) -> SIMD[
    dtype, simd_width
]:
    # return d * (pow2if[simd_width](e >> 1) * pow2if[simd_width](e - (e >> 1))).cast[dtype]();
    var ans = d * (pow2if[simd_width](e)).cast[dtype]()
    var y = bitcast[DType.int32, simd_width, dtype, simd_width](ans)
    var mask = (1 << (e - 23)) - 1

    var msb = y
    var idx = 0
    for i in range(32):
        if msb&0x1:
            idx = i
            break
        msb = msb >> 1

    # if e>=23:
    #     y=y-(y&mask)
    # if e>=23:
    #     y=y+1
    ans = bitcast[dtype, simd_width, DType.int32, simd_width](y)
    return ans


@always_inline
fn exp_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    var res = SIMD[type, simd_width]()

    @unroll
    for i in range(simd_width):
        res[i] = external_call["expf", Scalar[type]](arg[i])
    return res


@always_inline
fn ldexp_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width], e: SIMD[DType.int32, simd_width]) -> SIMD[
    type, simd_width
]:
    var res = SIMD[type, simd_width]()

    @unroll
    for i in range(simd_width):
        res[i] = external_call["ldexpf", Scalar[type]](arg)
    return res


fn exp_sleef[
    dtype: DType, simd_width: Int
](d: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    alias inv_lg2 = SIMD[dtype, simd_width](1.4426950408889634)
    alias lg2it = SIMD[dtype, simd_width](0.6931471805599453)

    var q = floor(d.fma(inv_lg2, 0.5))

    ## upper and lower parts of log(2)
    alias L2Uf = SIMD[dtype, simd_width](0.693145751953125)
    alias L2Lf = SIMD[dtype, simd_width](1.428606765330187045e-06)

    # var s = q.fma(-L2Uf, d)
    # s = q.fma(-L2Lf, s)
    var s = d - q * lg2it

    var u = SIMD[dtype, simd_width](0.000198527617612853646278381)
    u = u.fma(s, 0.00139304355252534151077271)
    u = u.fma(s, 0.00833336077630519866943359)
    u = u.fma(s, 0.0416664853692054748535156)
    u = u.fma(s, 0.166666671633720397949219)
    u = u.fma(s, 0.5)
    u = s * s * u + s + 1.0

    u = _exp_taylor(s)
    u = ldexp2kf(u, q.cast[DType.int32]())
    # u = ldexp_libm(u, q.cast[DType.int32]());
    return u


@always_inline
fn exp_mojo_opt[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Calculates elementwise `e^{X_i}`, where `X_i` is an element in the input
    SIMD vector at position `i`.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
       x: The input SIMD vector.

    Returns:
      A SIMD vector containing `e` raised to the power `Xi` where `Xi` is an
      element in the input SIMD vector.
    """
    constrained[type.is_floating_point(), "must be a floating point value"]()
    alias neg_ln2 = -0.69314718055966295651160180568695068359375
    alias inv_lg2 = 1.442695040888963407359924681001892137426646

    # upper and lower parts of log(2)=[L2Uf,L2Lf]
    alias L2Uf = 0.693145751953125
    alias L2Lf = 1.428606765330187045e-06

    alias min_val = SIMD[type, simd_width](-88.3762626647949)
    alias max_val = SIMD[type, simd_width](88.3762626647950)

    alias im_type = DType.float64
    var xc = clamp(x, min_val, max_val).cast[im_type]()
    var k = floor(xc.fma(inv_lg2, 0.5)).cast[im_type]()

    var r = k.fma(neg_ln2, xc)
    # var r = k.fma(-L2Lf, k.fma(-L2Uf, xc))
    var taylor_result = _exp_taylor(r.cast[im_type]()).cast[type]()
    var expr = ldexp(taylor_result, k.cast[DType.int32]())
    return expr
    # var val1 = (expr > min_val).select(expr, SIMD[type,simd_width](0))
    # return (val1 < max_val).select(val1, SIMD[type,simd_width](inf[type]()))


@always_inline
fn _exp_taylor_mlas[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return polynomial_evaluate[
        type,
        simd_width,
        VariadicList[SIMD[type, simd_width]](
            1.0,
            1.0,
            0.499999851,
            0.16666472,
            0.0416695364,
            0.00837312452,
            0.00137805939,
        ),
    ](x)


@always_inline
fn exp_mlas[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    constrained[type.is_floating_point(), "must be a floating point value"]()
    alias neg_ln2 = -0.69314718055966295651160180568695068359375
    alias inv_lg2 = 1.442695040888963407359924681001892137426646

    alias neg_ln2_hi = -6.93145752e-1
    alias neg_ln2_lo = -1.42860677e-6

    var min_val = -88.3762626647949
    var max_val = 88.3762626647950

    var xc = clamp(x, min_val, max_val)
    var k = floor(xc.fma(inv_lg2, 0.5))
    var r = k.fma(neg_ln2_hi, xc)
    var rr = k.fma(neg_ln2_lo, r)
    return max(ldexp(_exp_taylor_mlas(rr), k.cast[DType.int32]()), xc)


def accuracy_test():
    alias delta_min = -16
    alias delta_max = 15
    alias delta_range = delta_max - delta_min + 1

    var deltas = Buffer[DType.int32, delta_range].stack_allocation()
    deltas.zero()

    for i in range(0x3000_0000, 0x42B0_0000, 1):
        var f = bitcast[DType.float32, 1](SIMD[DType.uint32, 1](i))

        var r1 = exp_mlas(f)
        var r2 = exp_libm(f)

        var i1 = bitcast[DType.int32, 1](r1)
        var i2 = bitcast[DType.int32, 1](r2)

        var id = int(clamp(i1 - i2, delta_min, delta_max))
        deltas[id - delta_min] = deltas[id - delta_min] + 1

        if id == delta_max:
            print(f, r1, r2, i1 - i2)

    for i in range(delta_range):
        print("deltas", i + delta_min, deltas[i])


def main():
    var args = argv()
    for i in range(len(args)):
        if args[i] == "-t":
            accuracy_test()
            return

    var m = MojoBench()
    var problem_size = range(1 << 10, 1 << 12, 1 << 10)
    bench_unary[exp, DType.float32](m, problem_size, "mojo")
    bench_unary[exp_mojo_opt, DType.float32](m, problem_size, "mojo_opt")
    bench_unary[exp_libm, DType.float32](m, problem_size, "libm")
    bench_unary[exp_sleef, DType.float32](m, problem_size, "sleef")
    bench_unary[exp_mlas, DType.float32](m, problem_size, "mlas")
    m.dump_report()
