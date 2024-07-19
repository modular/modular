# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import log, isnan, nan, inf, sqrt
from algorithm import elementwise, sum, mean
from memory.unsafe import DTypePointer
from buffer import Buffer


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


fn kl_div(out: DTypePointer, x: __type_of(out), y: __type_of(out), len: Int):
    @parameter
    fn kl_div_elementwise[
        simd_width: Int, rank: Int
    ](idx: StaticIntTuple[rank]):
        out[idx[0]] = rebind[Scalar[out.type]](
            kl_div(
                SIMD[size=simd_width].load(x + idx[0]),
                SIMD[size=simd_width].load(y + idx[0]),
            )
        )

    elementwise[kl_div_elementwise, simdwidthof[out.type]()](len)


# ===----------------------------------------------------------------------=== #
# correlation
# ===----------------------------------------------------------------------=== #


fn correlation(
    u: DTypePointer,
    v: __type_of(u),
    len: Int,
    *,
    w: Optional[__type_of(u)] = None,
    centered: Bool = True,
) -> Scalar[u.type]:
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
    var umu = Scalar[u.type]()
    var vmu = Scalar[u.type]()
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


fn cosine(
    u: DTypePointer,
    v: __type_of(u),
    len: Int,
    *,
    w: Optional[__type_of(u)] = None,
) -> Scalar[u.type]:
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


fn _sqrt(out: DTypePointer, x: __type_of(out), len: Int):
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        out[idx[0]] = rebind[Scalar[out.type]](
            sqrt(SIMD[size=simd_width].load(x + idx[0]))
        )

    elementwise[apply_fn, simdwidthof[out.type]()](len)


fn _mul(out: DTypePointer, x: __type_of(out), y: __type_of(out), len: Int):
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        out[idx[0]] = rebind[Scalar[out.type]](
            SIMD[size=simd_width].load(x + idx[0])
            * SIMD[size=simd_width].load(y + idx[0])
        )

    elementwise[apply_fn, simdwidthof[out.type]()](len)


fn _div(out: DTypePointer, x: __type_of(out), c: Scalar[out.type], len: Int):
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        out[idx[0]] = (
            rebind[Scalar[out.type]](SIMD[size=simd_width].load(x + idx[0])) / c
        )

    elementwise[apply_fn, simdwidthof[out.type]()](len)


fn _sum(src: DTypePointer, len: Int) -> Scalar[src.type]:
    return sum(
        Buffer[src.type, address_space = src.address_space](
            DTypePointer[_, _, False](src), len
        )
    )


fn _mean(src: DTypePointer, len: Int) -> Scalar[src.type]:
    return mean(
        Buffer[src.type, address_space = src.address_space](
            DTypePointer[_, _, False](src), len
        )
    )


fn _dot(x: DTypePointer, y: __type_of(x), len: Int) -> Scalar[x.type]:
    alias simd_width = simdwidthof[x.type]()
    var accum_simd = SIMD[x.type, simd_width](0)
    var accum_scalar = Scalar[x.type](0)

    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        var xi = SIMD[size=simd_width].load(x + idx[0])
        var yi = SIMD[size=simd_width].load(y + idx[0])

        @parameter
        if simd_width == 1:
            accum_scalar += xi[0] * yi[0]
        else:
            accum_simd += rebind[__type_of(accum_simd)](xi * yi)

    elementwise[apply_fn, simd_width](len)

    return accum_simd.reduce_add() + accum_scalar
