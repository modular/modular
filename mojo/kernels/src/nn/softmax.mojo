# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Math import exp
from SIMD import SIMD
from Buffer import Buffer
from Int import Int
from Tuple import StaticTuple
from Numerics import neginf
from Reductions import max

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

    var i: Int = 0
    while i < vector_end:
        let simd_elem = input.simd_load[simd_width](i)
        let new_max_vec = SIMD[simd_width, type].splat(
            running_max_vec.max(simd_elem).reduce_max()
        )
        running_sum_vec = running_sum_vec * exp[simd_width, type](
            running_max_vec - new_max_vec
        ) + exp[simd_width, type](simd_elem - new_max_vec)
        running_max_vec = new_max_vec
        i += simd_width

    var running_max = running_max_vec.reduce_max()
    var running_sum = running_sum_vec.reduce_add()

    i = vector_end
    while i < len:
        let elem = input.__getitem__(i)
        let new_max = running_max.max(elem)
        running_sum = running_sum * exp[1, type](running_max - new_max) + exp[
            1, type
        ](elem - new_max)
        running_max = new_max
        i += 1

    return StaticTuple[2, __mlir_type[`!pop.scalar<`, type, `>`]].pair(
        running_max.__getitem__(0), running_sum.__getitem__(0)
    )


fn _softmax_2_pass_step2[
    simd_width: __mlir_type.index,
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

    # TODO: Because vectorize cannot currently capture values from outside
    # scope, we therefore replicate the logic of Functional.vectorize here.
    # In the future (once we have non-isolated-from-above regions) we can
    # just reuse the Functional.vectorize code.
    let len = output.__len__()
    let vector_end = (len // simd_width) * simd_width

    var i: Int = 0
    let running_max_vec = SIMD[simd_width, type].splat(running_max)
    let running_sum_vec = SIMD[simd_width, type].splat(running_sum)
    while i < vector_end:
        let simd_elem = input.simd_load[simd_width](i)
        output.simd_store[simd_width](
            i,
            exp[simd_width, type](simd_elem - running_max_vec)
            / running_sum_vec,
        )
        i += simd_width
    i = vector_end
    while i < len:
        let elem = input.__getitem__(i)
        output.__setitem__(
            i,
            exp[1, type](elem - running_max) / running_sum,
        )
        i += 1


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

    let running_max = running_info.__getitem__[0]()
    let running_sum = running_info.__getitem__[1]()

    _softmax_2_pass_step2[simd_width, buffer_size, type](
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
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    input: Buffer[buffer_size, type],
    output: Buffer[buffer_size, type],
    max: SIMD[1, type],
) -> SIMD[1, type]:
    # STEP 2: compute the exponential for each batch
    # for i = 0 to N do
    #   Output[i] = exp(Input[i] - maxVal)
    #   denom += Output[i]
    # end for
    var i: Int = 0
    var denom_simd = SIMD[simd_width, type].splat(0.0)
    var max_simd = SIMD[simd_width, type].splat(max)
    let len = input.__len__()
    # TODO: manually replicating vectorize logic. Replace with transform_reduce()
    # once closures are supported and transform_reduce implemented since max needs
    # to be captured.
    let vector_end = (len // simd_width) * simd_width
    while i < vector_end:
        let simd_elem = exp[simd_width, type](
            input.simd_load[simd_width](i) - max_simd
        )
        output.simd_store[simd_width](i, simd_elem)
        denom_simd += simd_elem
        i += simd_width
    i = vector_end
    var denom = denom_simd.reduce_add()
    while i < len:
        let elem = exp[1, type](input.__getitem__(i) - max)
        output.__setitem__(i, elem)
        denom += elem
        i += 1

    return denom


fn _softmax_3_pass_step3[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](output: Buffer[buffer_size, type], recip: SIMD[1, type]):
    # STEP 3: normalize each batch
    # for i = 0 to N do
    #   Output[b, i] /= denom
    # end for
    var i: Int = 0
    let len = output.__len__()
    let simd_recip = SIMD[simd_width, type].splat(recip)
    # TODO: manually replicating vectorize() logic. Replace with vectorize() once
    # closures are supported since recip needs to be captured.
    let vector_end = (len // simd_width) * simd_width
    while i < vector_end:
        let simd_elem = output.simd_load[simd_width](i) * simd_recip
        output.simd_store[simd_width](i, simd_elem)
        i += simd_width
    i = vector_end
    while i < len:
        output.__setitem__(i, output.__getitem__(i) * recip)
        i += 1


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
    let maxVal = _softmax_3_pass_step1[simd_width, buffer_size, type](input)
    let denom = _softmax_3_pass_step2[simd_width, buffer_size, type](
        input, output, maxVal
    )
    let recip = SIMD[1, type](1.0) / denom
    _softmax_3_pass_step3[simd_width, buffer_size, type](output, recip)
