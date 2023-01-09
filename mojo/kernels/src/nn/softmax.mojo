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
        running_sum = running_sum * exp[1, type](
            running_max - new_max
        ) + exp[1, type](elem - new_max)
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
    # Performs an unbatched softmax on an input tensor using the two-pass online
    # algorithm. The unbatched two-pass online softmax is described in "Online
    # normalizer calculation for softmax" (https://arxiv.org/abs/1805.02867) and
    # "A full-stack search technique for domain optimized deep learning
    # accelerators" (https://dl.acm.org/doi/abs/10.1145/3503222.3507767) and is
    # defined as:
    #
    # procedure SoftmaxUnbatched(InputInput)
    #   runningMax = -∞
    #   runningSum = 0
    #   STAGE 1:
    #   for i = 0 to N do
    #     newMax = max(runningMax, Input[i])
    #     runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
    #     runningMax = newMax
    #   end for
    #   for i = 0 to N do
    #     Output[i] = exp(Input[i] - runningMax) / runningSum
    #   end for

    let running_info = _softmax_2_pass_step1[simd_width, buffer_size, type](
        input
    )

    let running_max = running_info.__getitem__[0]()
    let running_sum = running_info.__getitem__[1]()

    _softmax_2_pass_step2[simd_width, buffer_size, type](
        output, input, running_max, running_sum
    )
