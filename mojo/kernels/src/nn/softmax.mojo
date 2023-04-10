# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import Buffer
from DType import DType
from Functional import vectorize_unroll
from Math import exp, identity, log, mul, reciprocal, sub
from List import Dim
from Numerics import neginf
from Range import range
from Reductions import max
from SIMD import SIMD
from StaticTuple import StaticTuple
from TypeUtilities import rebind


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


fn reduce_add_simd[
    simd_width: Int,
    step_simd_width: Int,
    type: DType,
](
    scalar&: SIMD[type, 1],
    vector&: SIMD[type, simd_width],
    val: SIMD[type, step_simd_width],
):
    """This functions adds val to either the scalar value or the vector value
    depending on the step_simd_width. This is useful when the simd_width varies
    between iterations as in vectorize.
    """

    @parameter
    if step_simd_width == 1:
        # When the step_simd_width is 1, then we add to the scalar value.
        scalar += val[0]
    else:
        # When the step_simd_Width is the same as the simd_width, then we add to
        # the vector value.
        vector += rebind[SIMD[type, simd_width]](val)


# ===----------------------------------------------------------------------===#
# Softmax 2 Pass
# ===----------------------------------------------------------------------===#


fn _softmax_2_pass_step1[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](input: Buffer[buffer_size, type]) -> StaticTuple[
    2, __mlir_type[`!pop.scalar<`, type.value, `>`]
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

    var running_max_vec = SIMD[type, simd_width].splat(neginf[type]())
    var running_sum_vec = SIMD[type, simd_width].splat(0)

    # TODO: Because vectorize cannot currently capture values from outside
    # scope, we therefore replicate the logic of Functional.vectorize here.
    # In the future (once we have non-isolated-from-above regions) we can
    # just reuse the Functional.vectorize code.
    let len = input.__len__()
    let vector_end = (len // simd_width) * simd_width

    for i in range(0, vector_end, simd_width):
        let simd_elem = input.simd_load[simd_width](i)
        let new_max_vec = SIMD[type, simd_width].splat(
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

    return StaticTuple[2, __mlir_type[`!pop.scalar<`, type.value, `>`]](
        running_max[0].value, running_sum[0].value
    )


fn _softmax_2_pass_step2[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
](
    output: Buffer[buffer_size, type],
    input: Buffer[buffer_size, type],
    running_max: SIMD[type, 1],
    running_sum: SIMD[type, 1],
):
    # Step 2:
    #   for i = 0 to N do
    #     Output[i] = exp(Input[i] - runningMax) / runningSum
    #   end for

    @always_inline
    fn _step_2[simd_width: Int](idx: Int):
        let running_max_simd = SIMD[type, simd_width].splat(running_max)
        let running_sum_simd = SIMD[type, simd_width].splat(running_sum)
        let input_val = input.simd_load[simd_width](idx)
        output.simd_store[simd_width](
            idx,
            exp[simd_width, type](input_val - running_max_simd)
            / running_sum_simd,
        )

    vectorize_unroll[simd_width, unroll_factor, _step_2](output.__len__())


fn softmax_2_pass[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
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
        simd_width (Int): The simd_width to use in vectorization.
        buffer_size (Dim): The size of the input and output buffers.
        type (DType): The type of the input and output buffers.
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


fn _softmax_3_pass_step_2[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
    pre_update_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    post_update_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
](
    output: Buffer[buffer_size, type],
    input: Buffer[buffer_size, type],
    max_val: SIMD[type, 1],
) -> SIMD[type, 1]:
    # STEP 2: compute for each batch
    # for i = 0 to N do
    #   Output[i] = pre_update_func(Input[i] - max_val)
    #   accum += post_update_func(Output[i])
    # end for
    alias outer_simd_width = simd_width

    var accum_scalar: SIMD[type, 1] = 0
    var accum_simd: SIMD[type, outer_simd_width] = 0

    @always_inline
    fn step_2[simd_width: Int](idx: Int):
        var elem = input.simd_load[simd_width](idx) - SIMD[
            type, simd_width
        ].splat(max_val)

        elem = pre_update_func[simd_width, type](elem)
        output.simd_store[simd_width](idx, elem)
        elem = post_update_func[simd_width, type](elem)
        reduce_add_simd[outer_simd_width, simd_width, type](
            accum_scalar, accum_simd, elem
        )

    vectorize_unroll[simd_width, unroll_factor, step_2](output.__len__())
    # Reduce the values from both the scalar and vector accum.
    return accum_scalar + accum_simd.reduce_add()


fn _softmax_3_pass_step_3[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
    accum_proc_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    accum_apply_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow, `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
](output: Buffer[buffer_size, type], accum: SIMD[type, 1],):
    # STEP 3: normalize each batch
    # accum = accum_proc_func(accum)
    # for i = 0 to N do
    #   accum_apply_func(Output[b, i], accum)
    # end for
    let accum_proc = accum_proc_func[1, type](accum)

    @always_inline
    fn step_3[simd_width: Int](idx: Int):
        let accum_simd = SIMD[type, simd_width].splat(accum_proc)
        var elem = output.simd_load[simd_width](idx)
        elem = accum_apply_func[simd_width, type](elem, accum_simd)
        output.simd_store[simd_width](idx, elem)

    vectorize_unroll[simd_width, unroll_factor, step_3](output.__len__())


fn _softmax_3_pass_base[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    step2_pre_update_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    step2_post_update_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    step3_accum_proc_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    step3_accum_apply_func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow, `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
](output: Buffer[buffer_size, type], input: Buffer[buffer_size, type]):
    """Performs an unbatched three-pass softmax. The actual behavior of each
    step can be different between the (regular) softmax and logsoftmax.

    Args:
        simd_width (Int): The simd_width to use in vectorization.
        buffer_size (Dim): The size of the input and output buffers.
        type (DType): The type of the input and output buffers.
        logsoftmax (Bool): Perform logsoftmax if True, regular softmax otherwise.
        output (Buffer[buffer_size, type]): The output buffer in which to store the softmax values.
        input (Buffer[buffer_size, type]): The input buffer used to compute the softmax.

    Returns:
        None
    """
    # STEP 1
    let max_val = max[simd_width, buffer_size, type](input)

    # STEP 2
    alias unroll_factor = 8  # TODO: search
    var accum = _softmax_3_pass_step_2[
        simd_width,
        unroll_factor,
        buffer_size,
        type,
        step2_pre_update_func,
        step2_post_update_func,
    ](output, input, max_val)

    # STEP 3
    _softmax_3_pass_step_3[
        simd_width,
        unroll_factor,
        buffer_size,
        type,
        step3_accum_proc_func,
        step3_accum_apply_func,
    ](output, accum)


fn softmax_3_pass[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
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
        simd_width (Int): The simd_width to use in vectorization.
        buffer_size (Dim): The size of the input and output buffers.
        type (DType): The type of the input and output buffers.
        output (Buffer[buffer_size, type]): The output buffer in which to store the softmax values.
        input (Buffer[buffer_size, type]): The input buffer used to compute the softmax.

    Returns:
        None
    """
    _softmax_3_pass_base[
        simd_width, buffer_size, type, exp, identity, reciprocal, mul
    ](output, input)


# ===----------------------------------------------------------------------===#
# LogSoftmax
# ===----------------------------------------------------------------------===#


fn logsoftmax[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](output: Buffer[buffer_size, type], input: Buffer[buffer_size, type]):
    """Performs an unbatched logsoftmax on an input tensor using the three-pass
    algorithm. The unbatched three-pass softmax is defined as:
    procedure SoftmaxUnbatched(InputInput)
      maxVal = -∞
      denom = 0
      STEP 1: find the max value in each batch
      for i = 0 to N do
        maxVal = max(maxVal, Input[b, i])
      end for
      STEP 2: compute the sum of exponential of each batch
      for i = 0 to N do
        Output[b, i] = Input[b, i] - maxVal
        accum += exp(Output[b, i])
      end for
      STEP 3: normalize each batch
      for i = 0 to N do
        Output[b, i] -= log(accum)
      end for

    Args:
        simd_width (Int): The simd_width to use in vectorization.
        buffer_size (Dim): The size of the input and output buffers.
        type (DType): The type of the input and output buffers.
        output (Buffer[buffer_size, type]): The output buffer in which to store the softmax values.
        input (Buffer[buffer_size, type]): The input buffer used to compute the softmax.

    Returns:
        None
    """
    _softmax_3_pass_base[
        simd_width, buffer_size, type, identity, exp, log, sub
    ](output, input)
