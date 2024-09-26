# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import inf, isnan, log, nan, sqrt
from sys import simdwidthof

from algorithm import elementwise, mean, sum
from buffer import Buffer
from memory import UnsafePointer

from utils import StaticIntTuple

# ===----------------------------------------------------------------------=== #
# kl_div
# ===----------------------------------------------------------------------=== #


fn kl_div(x: SIMD, y: __type_of(x)) -> __type_of(x):
    """Elementwise function for computing Kullback-Leibler divergence.

    $$
    \\mathrm{kl\\_div}(x, y) =
      \begin{cases}
        x \\log(x / y) - x + y & x > 0, y > 0 \\
        y & x = 0, y \\ge 0 \\
        \\infty & \\text{otherwise}
      \\end{cases}
    $$
    """
    return (isnan(x) or isnan(y)).select(
        __type_of(x)(nan[x.type]()),
        (x > 0 and y > 0).select(
            x * log(x / y) - x + y,
            (x == 0 and y >= 0).select(y, __type_of(x)(inf[x.type]())),
        ),
    )


fn kl_div[
    type: DType, //
](
    out: UnsafePointer[Scalar[type]],
    x: __type_of(out),
    y: __type_of(out),
    len: Int,
):
    @parameter
    fn kl_div_elementwise[
        simd_width: Int, rank: Int
    ](idx: StaticIntTuple[rank]):
        out[idx[0]] = rebind[Scalar[type]](
            kl_div(
                (x + idx[0]).load[width=simd_width](),
                (y + idx[0]).load[width=simd_width](),
            )
        )

    elementwise[kl_div_elementwise, simdwidthof[type]()](len)


# ===----------------------------------------------------------------------=== #
# correlation
# ===----------------------------------------------------------------------=== #


fn correlation[
    type: DType, //
](
    u: UnsafePointer[Scalar[type]],
    v: __type_of(u),
    len: Int,
    *,
    w: OptionalReg[__type_of(u)] = None,
    centered: Bool = True,
) -> Scalar[type]:
    """Compute the correlation distance between two 1-D arrays.

    The correlation distance between `u` and `v`, is
    defined as

    $$
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    $$

    where $`\\bar{u}`$ is the mean of the elements of `u`
    and $`x \\cdot y`$ is the dot product of $x$ and $y$.
    """
    var vw = __type_of(u)()
    var uw = __type_of(u)()
    var umu = Scalar[type]()
    var vmu = Scalar[type]()
    var w_val = __type_of(u)()
    if w:
        w_val = __type_of(u).alloc(len)
        _div(w_val, w.value(), _sum(w.value(), len), len)
    if centered:
        if w:
            umu = _dot(u, w_val, len)
            vmu = _dot(v, w_val, len)
        else:
            umu = _mean(u, len)
            vmu = _mean(v, len)
    if w:
        vw = __type_of(u).alloc(len)
        uw = __type_of(u).alloc(len)
        _mul(vw, v, w.value(), len)
        _mul(uw, u, w.value(), len)
    else:
        vw, uw = v, u
    var uv = _dot(u, vw, len)
    var uu = _dot(u, uw, len)
    var vv = _dot(v, vw, len)
    var dist = 1 - uv / sqrt(uu * vv)
    if w:
        vw.free()
        uw.free()
        w_val.free()
    return dist.clamp(0, 2)


# ===----------------------------------------------------------------------=== #
# cosine
# ===----------------------------------------------------------------------=== #


fn cosine[
    type: DType, //
](
    u: UnsafePointer[Scalar[type]],
    v: __type_of(u),
    len: Int,
    *,
    w: OptionalReg[__type_of(u)] = None,
) -> Scalar[type]:
    """Compute the Cosine distance between 1-D arrays.

    The Cosine distance between `u` and `v`, is defined as

    $$
    1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}.
    $$

    where $u \\cdot v$ is the dot product of $u$ and $v$.

    The cosine distance is also referred to as 'uncentered correlation',
    or 'reflective correlation'.
    """
    return correlation(u, v, len, w=w, centered=False)


# ===----------------------------------------------------------------------=== #
# utils
# ===----------------------------------------------------------------------=== #


fn _sqrt[
    type: DType, //
](out: UnsafePointer[Scalar[type]], x: __type_of(out), len: Int):
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        out[idx[0]] = rebind[Scalar[type]](
            sqrt((x + idx[0]).load[width=simd_width]())
        )

    elementwise[apply_fn, simdwidthof[type]()](len)


fn _mul[
    type: DType, //
](
    out: UnsafePointer[Scalar[type]],
    x: __type_of(out),
    y: __type_of(out),
    len: Int,
):
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        out[idx[0]] = rebind[Scalar[type]](
            (x + idx[0]).load[width=simd_width]()
            * (y + idx[0]).load[width=simd_width]()
        )

    elementwise[apply_fn, simdwidthof[type]()](len)


fn _div[
    type: DType, //
](
    out: UnsafePointer[Scalar[type]],
    x: __type_of(out),
    c: Scalar[type],
    len: Int,
):
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        out[idx[0]] = (
            rebind[Scalar[type]]((x + idx[0]).load[width=simd_width]()) / c
        )

    elementwise[apply_fn, simdwidthof[type]()](len)


fn _sum[
    type: DType, //
](src: UnsafePointer[Scalar[type]], len: Int) -> Scalar[type]:
    return sum(
        Buffer[type, address_space = src.address_space](
            UnsafePointer[_, _, False, *_](src), len
        )
    )


fn _mean[
    type: DType, //
](src: UnsafePointer[Scalar[type]], len: Int) -> Scalar[type]:
    return mean(
        Buffer[type, address_space = src.address_space](
            UnsafePointer[_, _, False, *_](src), len
        )
    )


fn _dot[
    type: DType, //
](x: UnsafePointer[Scalar[type]], y: __type_of(x), len: Int) -> Scalar[type]:
    alias simd_width = simdwidthof[type]()
    var accum_simd = SIMD[type, simd_width](0)
    var accum_scalar = Scalar[type](0)

    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        var xi = (x + idx[0]).load[width=simd_width]()
        var yi = (y + idx[0]).load[width=simd_width]()

        @parameter
        if simd_width == 1:
            accum_scalar += xi[0] * yi[0]
        else:
            accum_simd += rebind[__type_of(accum_simd)](xi * yi)

    elementwise[apply_fn, simd_width](len)

    return accum_simd.reduce_add() + accum_scalar
