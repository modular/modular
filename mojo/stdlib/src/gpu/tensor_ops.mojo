# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.mma import mma
from gpu.shuffle import shuffle_down


@always_inline("nodebug")
fn tc_reduce[type: DType](val: Scalar[type]) -> Scalar[type]:
    """Using Tensor Cores to do warp level reduction for types TF32 and FP16."""

    @parameter
    if type == DType.float32:
        # TF32 warp level reduction using Tensor Cores
        var d_reg = SIMD[type, 4]()
        var a_reg = SIMD[type, 2](1)
        var b_reg = Scalar[type](val)
        var c_reg = SIMD[type, 4]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[type]()
        x_reg[0] = d_reg[0] + d_reg[1]
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]
    else:
        constrained[type == DType.float16, "unsupported dtype"]()
        # FP16 warp level reduction using Tensor Cores
        var d_reg = SIMD[type, 2]()
        var a_reg = Scalar[type](1)
        var b_reg = Scalar[type](val)
        var c_reg = SIMD[type, 2]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[type]()
        x_reg[0] = d_reg[0] + d_reg[1]
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]


# FP32.FP16 warp level reduction using Tensor Cores
@always_inline("nodebug")
fn tc_reduce[
    out_type: DType, in_type: DType
](val: Scalar[in_type]) -> Scalar[out_type]:
    """Using Tensor Cores to do warp level reduction for mixed precision FP32.FP16
    and FP32.BF16.
    """

    constrained[out_type == DType.float32]()

    @parameter
    if in_type == DType.float16:
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
    else:
        constrained[in_type == DType.bfloat16, "unsupported dtype"]()
        var d_reg = SIMD[out_type, 4]()
        var a_reg = SIMD[in_type, 2](1)
        var b_reg = Scalar[in_type](val)
        var c_reg = SIMD[out_type, 4]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[in_type]()
        x_reg[0] = (d_reg[0] + d_reg[1]).cast[in_type]()
        mma(d_reg, a_reg, x_reg, c_reg)
        return d_reg[0]
