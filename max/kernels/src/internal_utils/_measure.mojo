# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg
from math import inf, isnan, log, nan, sqrt
from sys import simdwidthof

from algorithm import elementwise, mean, sum, vectorize
from algorithm.functional import unswitch
from buffer import NDBuffer

from utils import IndexList

# ===----------------------------------------------------------------------=== #
# kl_div
# ===----------------------------------------------------------------------=== #


fn kl_div(x: SIMD, y: __type_of(x)) -> __type_of(x):
    """Elementwise function for computing Kullback-Leibler divergence.

    $$
    \\mathrm{kl\\_div}(x, y) =
      \\begin{cases}
        x \\log(x / y) - x + y & x > 0, y > 0 \\\\
        y & x = 0, y \\ge 0 \\\\
        \\infty & \\text{otherwise}
      \\end{cases}
    $$
    """
    return (isnan(x) | isnan(y)).select(
        nan[x.dtype](),
        ((x > 0) & (y > 0)).select(
            x * log(x / y) - x + y,
            ((x == 0) & (y >= 0)).select(y, inf[x.dtype]()),
        ),
    )


fn kl_div[
    type: DType, //
](
    output: UnsafePointer[Scalar[type]],
    x: __type_of(output),
    y: __type_of(output),
    len: Int,
) raises:
    @parameter
    fn kl_div_elementwise[simd_width: Int, rank: Int](idx: IndexList[rank]):
        output.store(
            idx[0],
            kl_div(
                x.load[width=simd_width](idx[0]),
                y.load[width=simd_width](idx[0]),
            ),
        )

    elementwise[kl_div_elementwise, simdwidthof[type]()](len)


fn kl_div[
    type: DType, //, out_type: DType = DType.float64
](
    x: UnsafePointer[Scalar[type]],
    y: __type_of(x),
    len: Int,
) -> Scalar[
    out_type
]:
    alias simd_width = simdwidthof[type]()
    var accum_simd = SIMD[out_type, simd_width](0)
    var accum_scalar = Scalar[out_type](0)

    @parameter
    fn kl_div_elementwise[simd_width: Int](idx: Int):
        var xi = x.load[width=simd_width](idx).cast[out_type]()
        var yi = y.load[width=simd_width](idx).cast[out_type]()
        var kl = kl_div(xi, yi)

        # TODO: should use VDPBF16PS when applicable
        # (i.e., host has avx512_bf16, type = bf16, out_type = float32)
        @parameter
        if simd_width == 1:
            accum_scalar += kl[0]
        else:
            accum_simd += rebind[__type_of(accum_simd)](kl)

    vectorize[kl_div_elementwise, simd_width](len)

    return accum_simd.reduce_add() + accum_scalar


# ===----------------------------------------------------------------------=== #
# correlation
# ===----------------------------------------------------------------------=== #


fn correlation[
    type: DType, //, out_type: DType = type
](
    u: UnsafePointer[Scalar[type]],
    v: __type_of(u),
    len: Int,
    *,
    w: OptionalReg[__type_of(u)] = None,
    centered: Bool = True,
) raises -> Scalar[out_type]:
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
    var umu = Scalar[out_type]()
    var vmu = Scalar[out_type]()
    var w_val = __type_of(u)()
    if w:
        w_val = __type_of(u).alloc(len)
        _div(w_val, w.value(), _sum(w.value(), len), len)
    if centered:
        if w:
            umu = _dot[out_type=out_type](u, w_val, len)
            vmu = _dot[out_type=out_type](v, w_val, len)
        else:
            umu = _mean(u, len).cast[out_type]()
            vmu = _mean(v, len).cast[out_type]()

    var uv = Scalar[out_type]()
    var uu = Scalar[out_type]()
    var vv = Scalar[out_type]()

    alias simd_width = simdwidthof[type]()
    var uv_simd = SIMD[out_type, simd_width]()
    var uu_simd = SIMD[out_type, simd_width]()
    var vv_simd = SIMD[out_type, simd_width]()

    @parameter
    fn accumulate[weighted: Bool]():
        @parameter
        fn apply_wfn[simd_width: Int](idx: Int):
            var ui = u.load[width=simd_width](idx).cast[out_type]() - umu
            var vi = v.load[width=simd_width](idx).cast[out_type]() - vmu
            var uw = ui
            var vw = vi

            @parameter
            if weighted:
                var wi = w_val.load[width=simd_width](idx).cast[out_type]()
                uw *= wi
                vw *= wi

            var uvw = ui * vw
            var uuw = ui * uw
            var vvw = vi * vw

            @parameter
            if simd_width == 1:
                uv += uvw[0]
                uu += uuw[0]
                vv += vvw[0]
            else:
                uv_simd += rebind[__type_of(uv_simd)](uvw)
                uu_simd += rebind[__type_of(uu_simd)](uuw)
                vv_simd += rebind[__type_of(vv_simd)](vvw)

        vectorize[apply_wfn, simd_width](len)

    unswitch[accumulate](w.__bool__())

    uv += uv_simd.reduce_add()
    uu += uu_simd.reduce_add()
    vv += vv_simd.reduce_add()
    if w:
        w_val.free()

    return (uv / sqrt(uu * vv)).clamp(-1, 1)


fn uncentered_unweighted_correlation[
    type: DType, //, out_type: DType = type
](
    u: UnsafePointer[Scalar[type]],
    v: __type_of(u),
    len: Int,
) -> Scalar[
    out_type
]:
    """Compute the uncentered and unweighted correlation
    distance between two 1-D arrays.
    Unlike `correlation` with arguments set, this does not raise.

    The correlation distance between `u` and `v`, is
    defined as

    $$
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    $$

    where $`\\bar{u}`$ is the mean of the elements of `u`
    and $`x \\cdot y`$ is the dot product of $x$ and $y$.
    """
    var uv = _dot[out_type=out_type](u, v, len)
    var uu = _dot[out_type=out_type](u, u, len)
    var vv = _dot[out_type=out_type](v, v, len)
    alias eps = Scalar[out_type](1e-6)
    return uv / (sqrt(uu * vv) + eps)


# ===----------------------------------------------------------------------=== #
# cosine
# ===----------------------------------------------------------------------=== #


fn cosine[
    type: DType, //,
](u: UnsafePointer[Scalar[type]], v: __type_of(u), len: Int,) -> Float64:
    """Compute the Cosine distance between 1-D arrays.

    The Cosine distance between `u` and `v`, is defined as

    $$
    1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}.
    $$

    where $u \\cdot v$ is the dot product of $u$ and $v$.

    The cosine distance is also referred to as 'uncentered correlation',
    or 'reflective correlation'.
    """
    return 1 - uncentered_unweighted_correlation[out_type = DType.float64](
        u, v, len
    )


fn relative_difference[
    dtype: DType, //,
](
    output: UnsafePointer[Scalar[dtype]],
    ref_out: __type_of(output),
    len: Int,
) -> Float64:
    var sum_abs_diff: Float64 = 0.0
    var sum_abs_ref: Float64 = 0.0
    var size = len

    for idx in range(len):
        var ui = output[idx].cast[DType.float64]()
        var vi = ref_out[idx].cast[DType.float64]()

        sum_abs_diff += abs(ui - vi).cast[DType.float64]()

        sum_abs_ref += abs(vi).cast[DType.float64]()

    var mean_abs_diff = sum_abs_diff / Float64(size)
    var mean_abs_ref = sum_abs_ref / Float64(size)

    var rel_diff = mean_abs_diff / mean_abs_ref
    return rel_diff


# ===----------------------------------------------------------------------=== #
# utils
# ===----------------------------------------------------------------------=== #


fn _sqrt[
    type: DType, //
](output: UnsafePointer[Scalar[type]], x: __type_of(output), len: Int) raises:
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: IndexList[rank]):
        output.store(
            idx[0],
            rebind[SIMD[type, simd_width]](
                sqrt(x.load[width=simd_width](idx[0]))
            ),
        )

    elementwise[apply_fn, simdwidthof[type]()](len)


fn _mul[
    type: DType, //
](
    output: UnsafePointer[Scalar[type]],
    x: __type_of(output),
    y: __type_of(output),
    len: Int,
) raises:
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: IndexList[rank]):
        output.store(
            idx[0],
            rebind[SIMD[type, simd_width]](
                x.load[width=simd_width](idx[0])
                * y.load[width=simd_width](idx[0])
            ),
        )

    elementwise[apply_fn, simdwidthof[type]()](len)


fn _div[
    type: DType, //
](
    output: UnsafePointer[Scalar[type]],
    x: __type_of(output),
    c: Scalar[type],
    len: Int,
) raises:
    @parameter
    fn apply_fn[simd_width: Int, rank: Int](idx: IndexList[rank]):
        output.store(
            idx[0],
            rebind[SIMD[type, simd_width]](x.load[width=simd_width](idx[0]))
            / c,
        )

    elementwise[apply_fn, simdwidthof[type]()](len)


fn _sum[
    type: DType, //
](src: UnsafePointer[Scalar[type]], len: Int) raises -> Scalar[type]:
    return sum(NDBuffer[type, 1, address_space = src.address_space](src, len))


fn _mean[
    type: DType, //
](src: UnsafePointer[Scalar[type]], len: Int) raises -> Scalar[type]:
    return mean(NDBuffer[type, 1, address_space = src.address_space](src, len))


fn _dot[
    type: DType, //, out_type: DType = type
](x: UnsafePointer[Scalar[type]], y: __type_of(x), len: Int) -> Scalar[
    out_type
]:
    # loads are the expensive part, so we use the (probably) smaller
    # input type for determining simd width.
    alias simd_width = simdwidthof[type]()
    var accum_simd = SIMD[out_type, simd_width](0)
    var accum_scalar = Scalar[out_type](0)

    @parameter
    fn apply_fn[simd_width: Int](idx: Int):
        var xi = x.load[width=simd_width](idx).cast[out_type]()
        var yi = y.load[width=simd_width](idx).cast[out_type]()

        # TODO: should use VDPBF16PS when applicable
        # (i.e., host has avx512_bf16, type = bf16, out_type = float32)
        @parameter
        if simd_width == 1:
            accum_scalar += xi[0] * yi[0]
        else:
            accum_simd += rebind[__type_of(accum_simd)](xi * yi)

    vectorize[apply_fn, simd_width](len)

    return accum_simd.reduce_add() + accum_scalar
