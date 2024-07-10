# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.mma import mma
from gpu.shuffle import shuffle_down


@always_inline("nodebug")
fn tc_reduce[
    out_type: DType, in_type: DType
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
