# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_msg
from Buffer import Buffer
from Functional import vectorize_unroll
from Int import Int
from Math import exp
from Numerics import neginf
from Range import range
from Reductions import max
from SIMD import SIMD
from Tuple import StaticTuple
from TypeUtilities import rebind


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@adaptive
fn reduce_add_simd[
    simd_width: __mlir_type.index,
    step_simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    scalar&: SIMD[1, type],
    vector&: SIMD[simd_width, type],
    val: SIMD[step_simd_width, type],
):
    """This functions adds val to either the scalar value or the vector value
    depending on the step_simd_width. This is useful when the simd_width varies
    between iterations as in vectorize.
    """
    assert_param_msg[step_simd_width == 1, "performing scalar reduction"]()

    # When the step_simd_width is 1, then we add to the scalar value.
    scalar += val[0]


@adaptive
fn reduce_add_simd[
    simd_width: __mlir_type.index,
    step_simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    scalar&: SIMD[1, type],
    vector&: SIMD[simd_width, type],
    val: SIMD[step_simd_width, type],
):
    """This functions adds val to either the scalar value or the vector value
    depending on the step_simd_width. This is useful when the simd_width varies
    between iterations as in vectorize.
    """
    assert_param_msg[step_simd_width > 1, "performing vector reduction"]()

    # When the step_simd_Width is the same as the simd_width, then we add to
    # the vector value.
    vector += rebind[SIMD[simd_width, type]](val)


# ===----------------------------------------------------------------------===#
# Softmax 2 Pass
# ===----------------------------------------------------------------------===#


fn _softmax_2_pass_step1[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](input: Buffer[buffer_size, type]) -> StaticTuple[
    2, __mlir_type[`!pop.scalar<`, type, `>`]
]:
    # STEP 1: find the runningMax and runningSum in each batch.
    #   runningMax = -∞
    #   runningSum = 0
    #   STAGE 1:
    #   for i = 0 to N do
    #     newMax = max(runningMax, Input[i])
    #     runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
    #     runningMax = newMax
    #   end for
    #   return runningMax, runningSum

    var running_max_vec = SIMD[simd_width, type].splat(neginf[type]())
    var running_sum_vec = SIMD[simd_width, type].splat(0)

    # TODO: Because vectorize cannot currently capture values from outside
    # scope, we therefore replicate the logic of Functional.vectorize here.
    # In the future (once we have non-isolated-from-above regions) we can
    # just reuse the Functional.vectorize code.
    let len = input.__len__()
    let vector_end = (len // simd_width) * simd_width

    for i in range(0, vector_end, simd_width):
        let simd_elem = input.simd_load[simd_width](i)
        let new_max_vec = SIMD[simd_width, type].splat(
            running_max_vec.max(simd_elem).reduce_max()
        )
        running_sum_vec = running_sum_vec * exp[simd_width, type](
            running_max_vec - new_max_vec
        ) + exp[simd_width, type](simd_elem - new_max_vec)
        running_max_vec = new_max_vec

    var running_max = running_max_vec.reduce_max()
    var running_sum = running_sum_vec.reduce_add()

    for ii in range(vector_end, len):  # TODO(#8365) use `i`
        let elem = input[ii]
        let new_max = running_max.max(elem)
        running_sum = running_sum * exp[1, type](running_max - new_max) + exp[
            1, type
        ](elem - new_max)
        running_max = new_max

    return StaticTuple[2, __mlir_type[`!pop.scalar<`, type, `>`]](
        running_max[0], running_sum[0]
    )


fn _softmax_2_pass_step2[
    simd_width: __mlir_type.index,
    unroll_factor: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    output: Buffer[buffer_size, type],
    input: Buffer[buffer_size, type],
    running_max: SIMD[1, type],
    running_sum: SIMD[1, type],
):
    # Step 2:
    #   for i = 0 to N do
    #     Output[i] = exp(Input[i] - runningMax) / runningSum
    #   end for

    @always_inline
    fn _step_2[simd_width: __mlir_type.index](idx: Int):
        let running_max_simd = SIMD[simd_width, type].splat(running_max)
        let running_sum_simd = SIMD[simd_width, type].splat(running_sum)
        let input_val = input.simd_load[simd_width](idx)
        output.simd_store[simd_width](
            idx,
            exp[simd_width, type](input_val - running_max_simd)
            / running_sum_simd,
        )

    vectorize_unroll[simd_width, unroll_factor, _step_2](output.__len__())


fn softmax_2_pass[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](output: Buffer[buffer_size, type], input: Buffer[buffer_size, type]):
    """Performs an unbatched softmax on an input tensor using the two-pass online
    algorithm. The unbatched two-pass online softmax is described in "Online
    normalizer calculation for softmax" (https://arxiv.org/abs/1805.02867) and
    "A full-stack search technique for domain optimized deep learning
    accelerators" (https://dl.acm.org/doi/abs/10.1145/3503222.3507767) and is
    defined as:

    procedure SoftmaxUnbatched(InputInput)
      runningMax = -∞
      runningSum = 0
      STAGE 1:
      for i = 0 to N do
        newMax = max(runningMax, Input[i])
        runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
        runningMax = newMax
      end for
      for i = 0 to N do
        Output[i] = exp(Input[i] - runningMax) / runningSum
      end for

    Args:
        simd_width (__mlir_type.index): The simd_width to use in vectorization.
        buffer_size (__mlir_type.index): The size of the input and output buffers.
        type (__mlir_type.`!kgen.dtype`): The type of the input and output buffers.
        output (Buffer[buffer_size, type]): The output buffer in which to store the softmax values.
        input (Buffer[buffer_size, type]): The input buffer used to compute the softmax.

    Returns:
        None
    """

    let running_info = _softmax_2_pass_step1[simd_width, buffer_size, type](
        input
    )

    let running_max = running_info[0]
    let running_sum = running_info[1]

    alias unroll_factor = 8  # TODO: search
    _softmax_2_pass_step2[simd_width, unroll_factor, buffer_size, type](
        output, input, running_max, running_sum
    )


# ===----------------------------------------------------------------------===#
# Softmax 3 Pass
# ===----------------------------------------------------------------------===#


fn _softmax_3_pass_step1[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](input: Buffer[buffer_size, type]) -> SIMD[1, type]:
    # STEP 1: find the max value in each batch
    # for i = 0 to N do
    #   maxVal = max(maxVal, Input[i])
    # end for
    return max[simd_width, buffer_size, type](input)


fn _softmax_3_pass_step2[
    simd_width: __mlir_type.index,
    unroll_factor: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    input: Buffer[buffer_size, type],
    output: Buffer[buffer_size, type],
    max_val: SIMD[1, type],
) -> SIMD[1, type]:
    # STEP 2: compute the exponential for each batch
    # for i = 0 to N do
    #   Output[i] = exp(Input[i] - maxVal)
    #   denom += Output[i]
    # end for

    alias outer_simd_width = simd_width

    var denom_scalar: SIMD[1, type] = 0
    var denom_simd: SIMD[outer_simd_width, type] = 0

    @always_inline
    fn step_2[simd_width: __mlir_type.index](idx: Int):
        let elem = exp[simd_width, type](
            input.simd_load[simd_width](idx)
            - SIMD[simd_width, type].splat(max_val)
        )
        output.simd_store[simd_width](idx, elem)
        reduce_add_simd[outer_simd_width, simd_width, type](
            denom_scalar, denom_simd, elem
        )

    vectorize_unroll[simd_width, unroll_factor, step_2](output.__len__())

    # Reduce the values from both the scalar and vector denom.
    return denom_scalar + denom_simd.reduce_add()


fn _softmax_3_pass_step3[
    simd_width: __mlir_type.index,
    unroll_factor: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](output: Buffer[buffer_size, type], denom: SIMD[1, type]):
    # STEP 3: normalize each batch
    # for i = 0 to N do
    #   Output[b, i] /= denom
    # end for
    let recip = 1 / denom

    @always_inline
    fn div[simd_width: __mlir_type.index](idx: Int):
        let simd_recip = SIMD[simd_width, type].splat(recip)
        output.simd_store[simd_width](
            idx, output.simd_load[simd_width](idx) * simd_recip
        )

    vectorize_unroll[simd_width, unroll_factor, div](output.__len__())


fn softmax_3_pass[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](output: Buffer[buffer_size, type], input: Buffer[buffer_size, type]):
    """Performs an unbatched softmax on an input tensor using the three-pass
    algorithm. The unbatched three-pass softmax is defined as:
    procedure SoftmaxUnbatched(InputInput)
      maxVal = -∞
      denom = 0
      STEP 1: find the max value in each batch
      for i = 0 to N do
        maxVal = max(maxVal, Input[b, i])
      end for
      STEP 2: compute the exponential for each batch
      for i = 0 to N do
        Output[b, i] = exp(Input[b, i] - maxVal)
        denom += Output[b, i]
      end for
      STEP 3: normalize each batch
      for i = 0 to N do
        Output[b, i] /= denom
      end for

    Args:
        simd_width (__mlir_type.index): The simd_width to use in vectorization.
        buffer_size (__mlir_type.index): The size of the input and output buffers.
        type (__mlir_type.`!kgen.dtype`): The type of the input and output buffers.
        output (Buffer[buffer_size, type]): The output buffer in which to store the softmax values.
        input (Buffer[buffer_size, type]): The input buffer used to compute the softmax.

    Returns:
        None
    """
    let max_val = _softmax_3_pass_step1[simd_width, buffer_size, type](input)
    alias unroll_factor = 8  # TODO: search
    let denom = _softmax_3_pass_step2[
        simd_width, unroll_factor, buffer_size, type
    ](input, output, max_val)
    _softmax_3_pass_step3[simd_width, unroll_factor, buffer_size, type](
        output, denom
    )
