# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements SIMD reductions.

You can import these APIs from the `algorithm` package. For example:

```mojo
from std.algorithm import map_reduce
```
"""

from std.collections import OptionalReg
from std.math import align_down, ceildiv
from std.sys.info import align_of, simd_width_of, size_of

from std.algorithm import sync_parallelize, vectorize
from std.algorithm.functional import _get_num_workers
from std.bit import log2_floor
from std.math.math import max as _max, min as _min
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_valid_target
from std.runtime.asyncrt import DeviceContextPtr
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg

from std.utils.index import Index, IndexList, StaticTuple
from std.sys.info import has_apple_gpu_accelerator

from std._plugin import CurrentPlugin

# Import CPU implementations.
from .backend.cpu.reduction import _reduce_generator_cpu

# Import GPU implementations.
from .backend.gpu.reduction import _reduce_generator_gpu, reduce_launch

# ===-----------------------------------------------------------------------===#
# ND indexing helper
# ===-----------------------------------------------------------------------===#


@always_inline
def _get_nd_indices_from_flat_index(
    flat_index: Int, shape: IndexList, skip_dim: Int, out res: type_of(shape)
):
    """Converts a flat index into ND indices but skip over one of the dimensions.

    The ND indices will iterate from right to left. I.E

    shape = (20, 5, 2, N)
    _get_nd_indices_from_flat_index(1, shape, rank -1) = (0, 0, 1, 0)
    _get_nd_indices_from_flat_index(5, shape, rank -1) = (0, 2, 1, 0)
    _get_nd_indices_from_flat_index(50, shape, rank -1) = (5, 0, 0, 0)
    _get_nd_indices_from_flat_index(56, shape, rank -1) = (5, 1, 1, 0)

    We ignore the Nth dimension to allow that to be traversed in the elementwise
    function.

    Args:
        flat_index: The flat index to convert.
        shape: The shape of the ND space we are converting into.
        skip_dim: The dimension to skip over. This represents the dimension
                  which is being iterated across.
    Returns:
        Constructed ND-index.
    """

    # The inner dimensions ([outer, outer, inner]) are not traversed if
    # drop last is set.
    comptime if shape.size == 2:
        if skip_dim == 1:
            return {flat_index, 0}
        else:
            return {0, flat_index}

    comptime IntType = type_of(shape)._int_type

    res = {}
    var curr_index = IntType(flat_index)

    comptime for i in reversed(range(shape.size)):
        # There is one dimension we skip, this represents the inner loop that
        # is being traversed.
        if i == skip_dim:
            res[i] = 0
        else:
            curr_index, res.data[i] = divmod(
                curr_index, IntType(shape.get[i]())
            )


# ===-----------------------------------------------------------------------===#
# MOGG reduce functions.
# These take lambdas and don't assume contiguous inputs so can compose
# with mogg kernels / fusion.
# ===-----------------------------------------------------------------------===#


@always_inline
def _reduce_generator[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
    target: StaticString = "cpu",
](
    shape: IndexList[_, element_type=DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Reduce the given tensor using the given reduction function. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        target: The target to run on.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        context: The pointer to DeviceContext.
    """
    comptime assert is_valid_target[target](), "unsupported target"

    for i in range(len(shape)):
        if shape[i] == 0:
            return

    comptime if is_cpu[target]():
        _reduce_generator_cpu[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](shape, init, reduce_dim)
    elif CurrentPlugin.reduce_generator_fn[target]:
        return comptime (CurrentPlugin.reduce_generator_fn[target].value())[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](shape, init, reduce_dim)
    else:
        _reduce_generator_gpu[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](shape, init, reduce_dim, context.get_device_context())


@always_inline
def _reduce_generator_wrapper[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    reduce_function: def[width: SIMDSize](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    /,
    target: StaticString = "cpu",
](
    shape: IndexList[_, element_type=DType.int64],
    init: Scalar,
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    @always_inline
    @parameter
    def input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    def output_fn_wrapper[
        _dtype: DType,
        width: SIMDSize,
        rank: Int,
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    def reduce_fn[
        ty: DType, width: SIMDSize
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return reduce_function(
            v1._refine[dtype](),
            v2._refine[dtype](),
        )._refine[ty]()

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_fn,
        target=target,
    ](shape, init, reduce_dim, context)


@always_inline
def _reduce_generator[
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: SIMDSize](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
    target: StaticString = "cpu",
](
    shape: IndexList[_, element_type=DType.int64],
    init: Scalar,
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Reduce the given tensor using the given reduction function.

    Constraints:
        Target must be "cpu".

    Parameters:
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        target: The target to run on.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        context: The pointer to DeviceContext.
    """

    comptime num_reductions = 1

    @always_inline
    @parameter
    def output_fn_wrapper[
        dtype: DType, width: SIMDSize, rank: Int
    ](
        indices: IndexList[rank],
        val: StaticTuple[SIMD[dtype, width], num_reductions],
    ):
        output_0_fn[dtype, width, rank](indices, val[0])

    @always_inline
    @parameter
    def reduce_fn_wrapper[
        dtype: DType, width: SIMDSize, reduction_idx: Int
    ](val: SIMD[dtype, width], acc: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            reduction_idx < num_reductions
        ), "invalid reduction index"
        return reduce_function[dtype, width](val, acc)

    var init_wrapped = StaticTuple[Scalar[init.dtype], num_reductions](init)
    return _reduce_generator[
        num_reductions,
        init.dtype,
        input_0_fn,
        output_fn_wrapper,
        reduce_fn_wrapper,
        target,
    ](shape, init_wrapped, reduce_dim, context)


# ===-----------------------------------------------------------------------===#
# Dispatch overloads for max, min, sum, product, mean
# ===-----------------------------------------------------------------------===#


@always_inline
def max[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Computes the max across the input and output shape.

    This performs the max computation on the domain specified by `input_shape`,
    loading the inputs using the `input_fn`. The results are stored using
    the `output_fn`.

    Parameters:
        dtype: The dtype of the input and output.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the max on.
        context: The pointer to DeviceContext.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    def output_fn_wrapper[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    def reduce_impl[
        ty: DType, width: SIMDSize
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return _max(v1, v2)

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
    ](input_shape, Scalar[dtype].MIN, reduce_dim, context=context)


@always_inline
def min[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Computes the min across the input and output shape.

    This performs the min computation on the domain specified by `input_shape`,
    loading the inputs using the `input_fn`. The results are stored using
    the `output_fn`.

    Parameters:
        dtype: The dtype of the input and output.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the min on.
        context: The pointer to DeviceContext.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    def output_fn_wrapper[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    def reduce_impl[
        ty: DType, width: SIMDSize
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return _min(v1, v2)

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
    ](input_shape, Scalar[dtype].MAX, reduce_dim, context=context)


@always_inline
def sum[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Computes the sum across the input and output shape.

    This performs the sum computation on the domain specified by `input_shape`,
    loading the inputs using the `input_fn`. The results are stored using
    the `output_fn`.

    Parameters:
        dtype: The dtype of the input and output.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the sum on.
        context: The pointer to DeviceContext.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    def output_fn_wrapper[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    def reduce_impl[
        ty: DType, width: SIMDSize
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
    ](input_shape, Scalar[dtype](0), reduce_dim, context=context)


@always_inline
def product[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Computes the product across the input and output shape.
    This performs the product computation on the domain specified by `input_shape`,
    loading the inputs using the `input_fn`. The results are stored using
    the `output_fn`.

    Parameters:
        dtype: The dtype of the input and output.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the product on.
        context: The pointer to DeviceContext.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    def output_fn_wrapper[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    def reduce_impl[
        ty: DType, width: SIMDSize
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 * v2

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
    ](input_shape, Scalar[dtype](1), reduce_dim, context=context)


@always_inline
def mean[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    output_shape: type_of(input_shape),
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Computes the mean across the input and output shape.

    This performs the mean computation on the domain specified by `input_shape`,
    loading the inputs using the `input_fn`. The results' domain is
    `output_shape` which are stored using the `output_fn`.

    Parameters:
        dtype: The dtype of the input and output.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the mean on.
        output_shape: The output shape.
        context: The pointer to DeviceContext.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def description_fn() -> String:
        return ";".join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    trace_arg("output", output_shape, dtype),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mean",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):

        @always_inline
        @parameter
        def reduce_impl[
            ty: DType, width: SIMDSize
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        @always_inline
        @parameter
        def input_fn_wrapper[
            _dtype: DType, width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
            return input_fn[width, rank](idx)._refine[_dtype, width]()

        # For floats apply the reciprocal as a multiply.
        comptime if dtype.is_floating_point():
            # Apply mean division before storing to the output lambda.
            comptime float_type = DType.float32 if has_apple_gpu_accelerator() else DType.float64
            var reciprocal = Scalar[float_type](1.0) / Scalar[float_type](
                input_shape[reduce_dim]
            )

            @always_inline
            @__copy_capture(reciprocal)
            @parameter
            def wrapped_output_mul[
                _dtype: DType, width: SIMDSize, rank: Int
            ](indices: IndexList[rank], value: SIMD[_dtype, width]):
                var mean_val = value * reciprocal.cast[_dtype]()
                output_fn[width, rank](
                    indices, mean_val._refine[dtype, width]()
                )

            _reduce_generator[
                input_fn_wrapper,
                wrapped_output_mul,
                reduce_impl,
                target=target,
            ](
                input_shape,
                init=Scalar[dtype](0),
                reduce_dim=reduce_dim,
                context=context,
            )

        else:
            # For ints just a normal divide.
            var dim_size = input_shape[reduce_dim]

            @always_inline
            @__copy_capture(dim_size)
            @parameter
            def wrapped_output_div[
                _dtype: DType, width: SIMDSize, rank: Int
            ](indices: IndexList[rank], value: SIMD[_dtype, width]):
                var mean_val = value / SIMD[_dtype, width](dim_size)
                output_fn[width, rank](
                    indices, mean_val._refine[dtype, width]()
                )

            _reduce_generator[
                input_fn_wrapper,
                wrapped_output_div,
                reduce_impl,
                target=target,
            ](
                input_shape,
                init=Scalar[dtype](0),
                reduce_dim=reduce_dim,
                context=context,
            )


# ===-----------------------------------------------------------------------===#
# CPU-only public API functions
# ===-----------------------------------------------------------------------===#


@always_inline
@parameter
def map_reduce[
    simd_width: SIMDSize,
    dtype: DType,
    acc_type: DType,
    origins_gen: OriginSet,
    input_gen_fn: def[dtype: DType, width: Int](Int) capturing[
        origins_gen
    ] -> SIMD[dtype, width],
    origins_vec: OriginSet,
    reduce_vec_to_vec_fn: def[acc_type: DType, dtype: DType, width: SIMDSize](
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing[origins_vec] -> SIMD[acc_type, width],
    reduce_vec_to_scalar_fn: def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width]
    ) thin -> Scalar[dtype],
](dst: Span[mut=True, Scalar[dtype], _], init: Scalar[acc_type]) -> Scalar[
    acc_type
]:
    """Stores the result of calling input_gen_fn in dst and simultaneously
    reduce the result using a custom reduction function.

    Parameters:
        simd_width: The vector width for the computation.
        dtype: The buffer elements dtype.
        acc_type: The dtype of the reduction accumulator.
        origins_gen: The OriginSet of captured arguments by the input_gen_fn.
        input_gen_fn: A function that generates inputs to reduce.
        origins_vec: The OriginSet of captured arguments by the reduce_vec_to_vec_fn.
        reduce_vec_to_vec_fn: A mapping function. This function is used to
          combine (accumulate) two chunks of input data: e.g. we load two
          `8xfloat32` vectors of elements and need to reduce them into a single
          `8xfloat32` vector.
        reduce_vec_to_scalar_fn: A reduction function. This function is used to
          reduce a vector to a scalar. E.g. when we got `8xfloat32` vector and want
          to reduce it to an `float32` scalar.

    Args:
        dst: The output buffer.
        init: The initial value to use in accumulator.

    Returns:
        The computed reduction value.
    """

    @always_inline
    @parameter
    def output_fn[
        _dtype: DType, width: SIMDSize, rank: Int
    ](idx: Int, val: SIMD[_dtype, width]):
        dst.unsafe_ptr().store(idx, rebind[SIMD[dtype, width]](val))

    return map_reduce[
        simd_width,
        dtype,
        acc_type,
        origins_gen,
        input_gen_fn,
        origins_vec,
        reduce_vec_to_vec_fn,
        reduce_vec_to_scalar_fn,
        output_fn,
    ](len(dst), init)


@always_inline
@parameter
def map_reduce[
    simd_width: SIMDSize,
    dtype: DType,
    acc_type: DType,
    origins_gen: OriginSet,
    input_gen_fn: def[dtype: DType, width: Int](Int) capturing[
        origins_gen
    ] -> SIMD[dtype, width],
    origins_vec: OriginSet,
    reduce_vec_to_vec_fn: def[acc_type: DType, dtype: DType, width: SIMDSize](
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing[origins_vec] -> SIMD[acc_type, width],
    reduce_vec_to_scalar_fn: def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width]
    ) thin -> Scalar[dtype],
    output_fn: def[dtype_: DType, width: SIMDSize, alignment: Int](
        idx: Int, val: SIMD[dtype_, width]
    ) capturing -> None,
](length: Int, init: Scalar[acc_type]) -> Scalar[acc_type]:
    """Performs a vectorized map-reduce operation over a sequence.

    Parameters:
        simd_width: The SIMD vector width to use.
        dtype: The data type of the input elements.
        acc_type: The data type of the accumulator.
        origins_gen: Origin set for the input generation function.
        input_gen_fn: Function that generates input values at each index.
        origins_vec: Origin set for the reduction function.
        reduce_vec_to_vec_fn: Function that reduces a vector into the accumulator.
        reduce_vec_to_scalar_fn: Function that reduces a final vector to a scalar.
        output_fn: Function to output intermediate results.

    Args:
        length: The number of elements to process.
        init: The initial accumulator value.

    Returns:
        The final reduced scalar value.
    """
    comptime unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    comptime unrolled_simd_width = simd_width * unroll_factor
    var unrolled_vector_end = align_down(length, unrolled_simd_width)
    var vector_end = align_down(length, simd_width)

    var acc_unrolled_simd = SIMD[acc_type, unrolled_simd_width](init)
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        var val_simd = input_gen_fn[dtype, unrolled_simd_width](i)
        output_fn[dtype, unrolled_simd_width, align_of[dtype]()](i, val_simd)
        acc_unrolled_simd = reduce_vec_to_vec_fn(acc_unrolled_simd, val_simd)

    var acc_simd = SIMD[acc_type, simd_width](init)
    for i in range(unrolled_vector_end, vector_end, simd_width):
        var val_simd = input_gen_fn[dtype, simd_width](i)
        output_fn[dtype, simd_width, align_of[dtype]()](i, val_simd)
        acc_simd = reduce_vec_to_vec_fn(acc_simd, val_simd)

    var acc = reduce_vec_to_scalar_fn[acc_type, unrolled_simd_width](
        acc_unrolled_simd
    )
    acc = reduce_vec_to_vec_fn(acc, reduce_vec_to_scalar_fn(acc_simd))
    for i in range(vector_end, length):
        var val = input_gen_fn[dtype, 1](i)
        output_fn[dtype, 1, align_of[dtype]()](i, val)
        acc = reduce_vec_to_vec_fn(acc, val)
    return acc[0]


# ===-----------------------------------------------------------------------===#
# reduce
# ===-----------------------------------------------------------------------===#


@always_inline
@parameter
def reduce[
    reduce_fn: def[acc_type: DType, dtype: DType, width: SIMDSize](
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[acc_type, width],
    dtype: DType,
](src: Span[Scalar[dtype], _], init: Scalar[dtype]) raises -> Scalar[dtype]:
    """Computes a custom reduction of buffer elements.

    Parameters:
        reduce_fn: The lambda implementing the reduction.
        dtype: The dtype of the input.

    Args:
        src: The input buffer.
        init: The initial value to use in accumulator.

    Returns:
        The computed reduction value.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def input_fn[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return src.unsafe_ptr().load[width=width](idx[0])._refine[_dtype]()

    var out: Scalar[init.dtype] = 0

    @always_inline
    @parameter
    def output_fn[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        out = value._refine[init.dtype, 1]()

    @always_inline
    @parameter
    def reduce_fn_wrapper[
        _dtype: DType, width: SIMDSize
    ](acc: SIMD[_dtype, width], val: SIMD[_dtype, width]) -> SIMD[
        _dtype, width
    ]:
        return reduce_fn(acc, val)

    var shape = Index(len(src))

    _reduce_generator[
        input_fn,
        output_fn,
        reduce_fn_wrapper,
    ](shape, init=init, reduce_dim=0)

    return out


@always_inline
@parameter
def reduce_boolean[
    reduce_fn: def[dtype: DType, width: SIMDSize](SIMD[dtype, width]) capturing[
        _
    ] -> Bool,
    continue_fn: def(Bool) capturing[_] -> Bool,
    dtype: DType,
](src: Span[Scalar[dtype], _], init: Bool) -> Bool:
    """Computes a bool reduction of buffer elements. The reduction will early
    exit if the `continue_fn` returns False.

    Parameters:
        reduce_fn: A boolean reduction function. This function is used to reduce
          a vector to a scalar. E.g. when we got `8xfloat32` vector and want to
          reduce it to a `bool`.
        continue_fn: A function to indicate whether we want to continue
          processing the rest of the iterations. This takes the result of the
          reduce_fn and returns True to continue processing and False to early
          exit.
        dtype: The dtype of the input.

    Args:
        src: The input buffer.
        init: The initial value to use.

    Returns:
        The computed reduction value.
    """
    comptime simd_width = simd_width_of[dtype]()
    comptime unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    comptime unrolled_simd_width = simd_width * unroll_factor

    var length = len(src)
    var unrolled_vector_end = align_down(length, unrolled_simd_width)
    var vector_end = align_down(length, simd_width)
    var curr = init
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        curr = reduce_fn(src.unsafe_ptr().load[width=unrolled_simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(unrolled_vector_end, vector_end, simd_width):
        curr = reduce_fn(src.unsafe_ptr().load[width=simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(vector_end, length):
        curr = reduce_fn(src[i])
        if not continue_fn(curr):
            return curr
    return curr


# ===-----------------------------------------------------------------------===#
# max (Span overload)
# ===-----------------------------------------------------------------------===#


@always_inline
def _simd_max[
    dtype: DType,
    simd_width: SIMDSize,
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


@always_inline
@parameter
def _simd_max_elementwise[
    acc_type: DType,
    dtype: DType,
    simd_width: SIMDSize,
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise max of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return _max(x, y.cast[acc_type]())


def max[dtype: DType](src: Span[Scalar[dtype], _]) raises -> Scalar[dtype]:
    """Computes the max element in a buffer.

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer.

    Returns:
        The maximum of the buffer elements.

    Raises:
        If the operation fails.
    """
    return reduce[_simd_max_elementwise](src, Scalar[dtype].MIN)


# ===-----------------------------------------------------------------------===#
# min (Span overload)
# ===-----------------------------------------------------------------------===#


@always_inline
def _simd_min[
    dtype: DType, simd_width: SIMDSize
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the min element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_min()


@always_inline
@parameter
def _simd_min_elementwise[
    acc_type: DType, dtype: DType, simd_width: SIMDSize
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise min of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return _min(x, y.cast[acc_type]())


def min[dtype: DType](src: Span[Scalar[dtype], _]) raises -> Scalar[dtype]:
    """Computes the min element in a buffer.

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer.

    Returns:
        The minimum of the buffer elements.

    Raises:
        If the operation fails.
    """
    return reduce[_simd_min_elementwise](src, Scalar[dtype].MAX)


# ===-----------------------------------------------------------------------===#
# sum (Span and 1d overloads)
# ===-----------------------------------------------------------------------===#


@always_inline
def _simd_sum[
    dtype: DType, simd_width: SIMDSize
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


@always_inline
@parameter
def _simd_sum_elementwise[
    acc_type: DType, dtype: DType, simd_width: SIMDSize
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise sum of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x + y.cast[acc_type]()


def sum[dtype: DType](src: Span[Scalar[dtype], _]) raises -> Scalar[dtype]:
    """Computes the sum of buffer elements.

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer.

    Returns:
        The sum of the buffer elements.

    Raises:
        If the operation fails.
    """

    @parameter
    @always_inline
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().load[width=width](idx)
        )

    return sum[dtype, input_fn_1d](len(src))


def sum[
    dtype: DType,
    input_fn_1d: def[dtype_: DType, width: Int](idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int) raises -> Scalar[dtype]:
    """
    Computes the sum of a 1D array using a provided input function.

    This function performs a reduction (sum) over a 1-dimensional array of the specified length and data type.
    The input values are provided by the `input_fn_1d` function, which takes an index and returns a SIMD vector
    of the specified width and data type. The reduction is performed using a single thread for deterministic results.

    Parameters:
        dtype: The data type of the elements to sum.
        input_fn_1d: A function that takes a data type, SIMD width, and index, and returns a SIMD vector of input values.

    Args:
        length: The number of elements in the 1D array.

    Returns:
        The sum of all elements as a scalar of the specified data type.

    Raises:
        Any exception raised by the input function or reduction process.
    """

    @always_inline
    @parameter
    def input_fn_nd[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn_1d[dtype, width](idx[0])._refine[_dtype]()

    var out: Scalar[dtype] = 0

    @always_inline
    @parameter
    def output_fn[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        out = value._refine[dtype, 1]()

    @always_inline
    @parameter
    def reduce_fn_wrapper[
        dtype: DType, width: SIMDSize
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return acc + val

    var shape = IndexList[1](length)

    _reduce_generator[
        input_fn_nd,
        output_fn,
        reduce_fn_wrapper,
    ](
        shape,
        init=Scalar[dtype](0),
        reduce_dim=0,
    )

    return out


# ===-----------------------------------------------------------------------===#
# product (Span overload)
# ===-----------------------------------------------------------------------===#


@always_inline
def _simd_product[
    dtype: DType, simd_width: SIMDSize
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the product of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_mul()


@always_inline
@parameter
def _simd_product_elementwise[
    acc_type: DType, dtype: DType, simd_width: SIMDSize
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise product of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x * y.cast[acc_type]()


def product[dtype: DType](src: Span[Scalar[dtype], _]) raises -> Scalar[dtype]:
    """Computes the product of the buffer elements.

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer.

    Returns:
        The product of the buffer elements.

    Raises:
        If the operation fails.
    """
    return reduce[_simd_product_elementwise](src, Scalar[dtype](1))


# ===-----------------------------------------------------------------------===#
# mean (Span and 1d overloads)
# ===-----------------------------------------------------------------------===#


def mean[dtype: DType](src: Span[Scalar[dtype], _]) raises -> Scalar[dtype]:
    """Computes the mean value of the elements in a buffer.

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.

    Raises:
        If the operation fails.
    """

    assert len(src) != 0, "input must not be empty"

    @parameter
    @always_inline
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().load[width=width](idx)
        )

    return mean[dtype, input_fn_1d](len(src))


def mean[
    dtype: DType,
    input_fn_1d: def[dtype_: DType, width: Int](idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int) raises -> Scalar[dtype]:
    """Computes the arithmetic mean of values generated by a function.

    Parameters:
        dtype: The data type of the elements.
        input_fn_1d: A function that generates SIMD values at each index.

    Args:
        length: The number of elements to average.

    Returns:
        The mean value. For integral types, uses integer division.

    Raises:
        To comply with how generators are used in this module.
    """
    var total = sum[dtype, input_fn_1d](length)

    comptime if dtype.is_integral():
        return total // Scalar[dtype](length)
    else:
        return total / Scalar[dtype](length)


# ===-----------------------------------------------------------------------===#
# variance
# ===-----------------------------------------------------------------------===#


def variance[
    dtype: DType
](
    src: Span[Scalar[dtype], _], mean_value: Scalar[dtype], correction: Int = 1
) raises -> Scalar[dtype]:
    """Given a mean, computes the variance of elements in a buffer.

    The mean value is used to avoid a second pass over the data:

    ```text
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer.
        mean_value: The mean value of the buffer.
        correction: Normalize variance by size - correction.

    Returns:
        The variance value of the elements in a buffer.

    Raises:
        If the operation fails.
    """

    assert len(src) > 1, "input length must be greater than 1"

    @parameter
    @always_inline
    @__copy_capture(src)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().load[width=width](idx)
        )

    return variance[dtype, input_fn_1d](len(src), mean_value, correction)


def variance[
    dtype: DType,
    input_fn_1d: def[dtype_: DType, width: Int](idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int, mean_value: Scalar[dtype], correction: Int = 1) raises -> Scalar[
    dtype
]:
    """Computes the variance of values generated by a function.

    Variance is calculated as:

    $$
    \\operatorname{variance}(X) =  \\frac{ \\sum_{i=0}^{length-1} (X_i - \\operatorname{E}(X_i))^2}{size - correction}
    $$

    where `E` represents the deviation of a sample from the mean.

    This version takes the mean value as an argument to avoid a second pass
    over the data.

    Parameters:
        dtype: The data type of the elements.
        input_fn_1d: A function that generates SIMD values at each index.

    Args:
        length: The number of elements.
        mean_value: The pre-computed mean value.
        correction: Normalize variance by size - correction (default: 1 for sample variance).

    Returns:
        The variance value.

    Raises:
        If length is less than or equal to correction.
    """

    @always_inline
    @parameter
    def input_fn_nd[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        var mean_simd = SIMD[mean_value.dtype, width](mean_value).cast[_dtype]()
        var x = input_fn_1d[_dtype, width](idx[0])
        var diff = x.cast[_dtype]() - mean_simd
        return (diff * diff)._refine[_dtype]()

    var out: Scalar[dtype] = 0

    @always_inline
    @parameter
    def output_fn[
        _dtype: DType, width: SIMDSize, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        out = value._refine[dtype, 1]()

    @always_inline
    @parameter
    def reduce_fn_wrapper[
        dtype: DType, width: SIMDSize
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return acc + val

    var shape = IndexList[1](length)

    _reduce_generator[
        input_fn_nd,
        output_fn,
        reduce_fn_wrapper,
    ](
        shape,
        init=Scalar[mean_value.dtype](0),
        reduce_dim=0,
    )

    return out / Scalar[dtype](length - correction)


def variance[
    dtype: DType
](src: Span[Scalar[dtype], _], correction: Int = 1) raises -> Scalar[dtype]:
    """Computes the variance value of the elements in a buffer.

    ```text
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Parameters:
        dtype: The dtype of the input.

    Args:
        src: The buffer.
        correction: Normalize variance by size - correction (Default=1).

    Returns:
        The variance value of the elements in a buffer.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().load[width=width](idx)
        )

    return variance[dtype, input_fn_1d](len(src), correction)


def variance[
    dtype: DType,
    input_fn_1d: def[dtype_: DType, width: Int](idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int, correction: Int = 1) raises -> Scalar[dtype]:
    """Computes the variance of values generated by a function.

    This version computes the mean automatically in a first pass.

    Parameters:
        dtype: The data type of the elements.
        input_fn_1d: A function that generates SIMD values at each index.

    Args:
        length: The number of elements.
        correction: Normalize variance by size - correction (default: 1 for sample variance).

    Returns:
        The variance value.

    Raises:
        If length is less than or equal to correction.
    """
    var mean_value = mean[dtype, input_fn_1d](length)
    return variance[dtype, input_fn_1d](length, mean_value, correction)


# ===-----------------------------------------------------------------------===#
# cumsum function
# ===-----------------------------------------------------------------------===#


@always_inline
def _cumsum_small[
    dtype: DType
](dst: Span[mut=True, Scalar[dtype], _], src: Span[Scalar[dtype], _]):
    dst[0] = src[0]
    for i in range(1, len(dst)):
        dst[i] = src[i] + dst[i - 1]


def cumsum[
    dtype: DType
](dst: Span[mut=True, Scalar[dtype], _], src: Span[Scalar[dtype], _]):
    """Computes the cumulative sum of all elements in a buffer.
       dst[i] = src[i] + src[i-1] + ... + src[0].

    Parameters:
        dtype: The dtype of the input.

    Args:
        dst: The buffer that stores the result of cumulative sum operation.
        src: The buffer of elements for which the cumulative sum is computed.
    """

    assert len(src) != 0, "Input must not be empty"
    assert len(dst) != 0, "Output must not be empty"

    comptime simd_width = simd_width_of[dtype]()

    # For length less than simd_width do serial cumulative sum.
    # Similarly, for the case when simd_width == 2 serial should be faster.
    if len(dst) < simd_width or simd_width == 2:
        return _cumsum_small(dst, src)

    # Stores the offset (i.e., last value of previous simd_width-elements chunk,
    # replicated across all simd lanes, to be added to all elements of next
    # chunk.
    var offset = SIMD[dtype, simd_width](0)

    # Divide the buffer size to div_size chunks of simd_width elements,
    # to calculate using SIMD and do remaining (tail) serially.
    var div_size = align_down(len(dst), simd_width)

    # Number of inner-loop iterations (for shift previous result and add).
    comptime rep = log2_floor(simd_width)

    for i in range(0, div_size, simd_width):
        var x_simd = src.unsafe_ptr().load[width=simd_width](i)

        comptime for i in range(rep):
            x_simd += x_simd.shift_right[2**i]()

        dst.unsafe_ptr().store(i, x_simd)

    # e.g., Assuming input buffer 1, 2, 3, 4, 5, 6, 7, 8 and simd_width = 4
    # The first outer iteration of the above would be the following;
    # note log2(simd_width) = log2(4) = 2 inner iterations.
    #   1, 2, 3, 4
    # + 0, 1, 2, 3  (<-- this is the shift_right operation)
    # ------------
    #   1, 3, 5, 7
    # + 0, 0, 1, 3  (<-- this is the shift_right operation)
    # ------------
    #   1, 3, 6, 10

    # Accumulation phase: Loop over simd_width-element chunks,
    # and add the offset (where offset is a vector of simd_width
    # containing the last element of the previous chunk).
    # e.g.,
    # offset used in iteration 0: 0, 0, 0, 0
    # offset used in iteration 1: 10, 10, 10, 10
    for i in range(0, div_size, simd_width):
        var x_simd = dst.unsafe_ptr().load[width=simd_width](i) + offset
        dst.unsafe_ptr().store(i, x_simd)
        offset = SIMD[dtype, simd_width](x_simd[simd_width - 1])

    # Handles the tail, i.e., num of elements at the end that don't
    # fit within a simd_width-elements vector.
    for i in range(div_size, len(dst)):
        dst[i] = dst[i - 1] + src[i]
