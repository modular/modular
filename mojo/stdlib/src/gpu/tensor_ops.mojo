# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.mma import mma
from gpu.warp import shuffle_down


@always_inline
fn tc_reduce_gevm_8x[
    out_type: DType, in_type: DType, simd_width: Int
](val1: SIMD[in_type, simd_width], val2: SIMD[in_type, simd_width]) -> SIMD[
    out_type, simd_width
]:
    """Using Tensor Cores to do warp level reduction."""

    constrained[
        out_type == DType.float32 and in_type == DType.bfloat16,
        "unsupported input/output type",
    ]()

    var d_reg = SIMD[out_type, simd_width]()
    var a_reg = SIMD[in_type, simd_width * 2](1)
    mma(d_reg, a_reg, val1, d_reg)

    var c_reg = SIMD[out_type, simd_width]()
    mma(c_reg, a_reg, val2, d_reg)
    return c_reg


@always_inline
fn tc_reduce_gevm_4x[
    out_type: DType, in_type: DType, simd_width: Int
](val1: SIMD[in_type, simd_width]) -> SIMD[out_type, simd_width]:
    """Using Tensor Cores to do warp level reduction."""

    constrained[
        out_type == DType.float32 and in_type == DType.bfloat16,
        "unsupported input/output type",
    ]()

    var d_reg = SIMD[out_type, simd_width]()
    var a_reg = SIMD[in_type, simd_width * 2](1)
    mma(d_reg, a_reg, val1, d_reg)
    return d_reg


@always_inline
fn tc_reduce[
    in_type: DType, simd_width: Int, //, out_type: DType
](val: SIMD[in_type, simd_width]) -> Scalar[out_type]:
    @parameter
    if simd_width == 1:
        return _tc_reduce_scalar[out_type](rebind[Scalar[in_type]](val))
    else:
        return _tc_reduce_vector[out_type](val)


@always_inline
fn _tc_reduce_vector[
    in_type: DType, simd_width: Int, //, out_type: DType
](val: SIMD[in_type, simd_width]) -> Scalar[out_type]:
    """Using Tensor Cores to do warp level reduction."""

    @parameter
    if out_type == DType.float32 and in_type == DType.bfloat16:

        @parameter
        if simd_width == 1:
            return val[0].cast[out_type]()

        elif simd_width == 2:
            var tmp = val.cast[out_type]()
            return tmp[0] + tmp[1]

        elif simd_width == 4:
            # we do m16n8k8 tensor core matmul to get partial results in first row
            var d_reg = SIMD[out_type, 4]()
            var a_reg = SIMD[in_type, 4](1)
            var b_reg = SIMD[in_type, 2]()
            b_reg[0] = val[0]
            b_reg[1] = val[1]
            var c_reg = SIMD[out_type, 4]()

            # do another iteration for the next set of values where
            # result from previous used as C matrix for element wise
            # add by tensor cores
            mma(d_reg, a_reg, b_reg, c_reg)
            b_reg[0] = val[2]
            b_reg[1] = val[3]
            mma(c_reg, a_reg, b_reg, d_reg)

            # a third mma operation needed to sum the 8 elements
            var x_reg = SIMD[in_type, 4]()
            b_reg = SIMD[in_type, 2](1)
            x_reg = c_reg.cast[in_type]()
            d_reg = SIMD[out_type, 4]()
            mma(d_reg, x_reg, b_reg, d_reg)

            return d_reg[0]

        elif simd_width == 8:
            # matrix A is of tile shape 16x16 and can process partial
            # sums of 256 elements matrix B is all 1s so it can enable the summation
            var d_reg = SIMD[out_type, 4]()
            var b_reg = SIMD[in_type, 4](1)

            # perform a m16n8k16 tensor core matmul to get the first col
            # as resultant partial sums
            mma(d_reg, val, b_reg, d_reg)
            var res = d_reg[0] + d_reg[2]

            # the resultant matrix d from previous operation has its first column
            # containing the values and only threads T0, T4, T8, T16 ... T28
            # contain the results. We need to sum them to get final result with
            # 0x11111111 mask we can restrict the active shuffle threads to every 1 in 4

            res += shuffle_down(0x11111111, res, 16)
            res += shuffle_down(0x11111111, res, 8)
            res += shuffle_down(0x11111111, res, 4)

            return res
        elif simd_width > 8:
            var tmp = SIMD[out_type, simd_width // 8]()

            @parameter
            for i in range(0, simd_width, 8):
                tmp[i // 8] = _tc_reduce_vector[out_type](
                    val.slice[8, offset=i]()
                )

            return tmp.reduce_add()

        else:
            constrained[False, "unsupported simd_width for BF16"]()
            return val[0].cast[out_type]()
    else:
        constrained[False, "unsupported input/output type"]()
        return val[0].cast[out_type]()


@always_inline
fn _tc_reduce_scalar[
    in_type: DType, //, out_type: DType
](val: Scalar[in_type]) -> Scalar[out_type]:
    """Using Tensor Cores to do warp level reduction."""

    constrained[out_type is DType.float32]()

    @parameter
    if out_type is DType.float32 and in_type is DType.float16:
        var d_reg = SIMD[out_type, 2]()
        var a_reg = SIMD[in_type, 1](1)
        var b_reg = SIMD[in_type, 1](val)
        var c_reg = SIMD[out_type, 2]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = SIMD[in_type, 1]()
        d_reg[0] += d_reg[1]
        x_reg[0] = d_reg[0].cast[in_type]()
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]

    elif out_type is DType.float32 and in_type is DType.bfloat16:
        var d_reg = SIMD[out_type, 4]()
        var a_reg = SIMD[in_type, 4](1)
        var b_reg = SIMD[in_type, 2]()
        b_reg[0] = val
        var c_reg = SIMD[out_type, 4]()

        mma(d_reg, a_reg, b_reg, c_reg)
        b_reg = SIMD[in_type, 2](1)
        var x_reg = d_reg.cast[in_type]()
        mma(d_reg, x_reg, b_reg, c_reg)

        return d_reg[0]

    elif out_type is DType.float32 and in_type is DType.float32:
        var d_reg = SIMD[out_type, 4]()
        var a_reg = SIMD[in_type, 2](1)
        var b_reg = Scalar[in_type](val)
        var c_reg = SIMD[out_type, 4]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[out_type]()
        x_reg[0] = d_reg[0] + d_reg[1]
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]

    else:
        constrained[
            in_type is DType.float16 and out_type is DType.float16,
            "unsupported dtype",
        ]()
        var d_reg = SIMD[out_type, 2]()
        var a_reg = Scalar[in_type](1)
        var b_reg = Scalar[in_type](val)
        var c_reg = SIMD[out_type, 2]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[out_type]()
        x_reg[0] = d_reg[0] + d_reg[1]
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]
