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
from max.algorithm import map_reduce
```
"""

from std.math import align_down
from std.sys.info import align_of, simd_width_of

from std.bit import log2_floor
from std.math.math import max as _max, min as _min
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_valid_target

from std.utils.index import IndexList, StaticTuple
from std.utils.coord import Coord, CoordLike, DynamicCoord, coord_to_index_list

from std._plugin import CurrentPlugin

# Import CPU implementations.
from .backend.cpu.reduction import _reduce_generator_cpu

# Import GPU implementations.
from .backend.gpu.reduction import _reduce_generator_gpu


# ===-----------------------------------------------------------------------===#
# ND indexing helper
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _get_nd_indices_from_flat_index[
    element_types: TypeList[Trait=CoordLike, ...],
    //,
    out_dtype: DType = Coord[*element_types].DTYPE,
](
    flat_index: Int,
    shape: Coord[*element_types],
    skip_dim: Int,
) -> DynamicCoord[out_dtype, Coord[*element_types].rank]:
    """Converts a flat index into ND indices but skip over one of the dimensions.

    The ND indices will iterate from right to left. I.E

    shape = (20, 5, 2, N)
    _get_nd_indices_from_flat_index(1, shape, rank -1) = (0, 0, 1, 0)
    _get_nd_indices_from_flat_index(5, shape, rank -1) = (0, 2, 1, 0)
    _get_nd_indices_from_flat_index(50, shape, rank -1) = (5, 0, 0, 0)
    _get_nd_indices_from_flat_index(56, shape, rank -1) = (5, 1, 1, 0)

    We ignore the Nth dimension to allow that to be traversed in the elementwise
    function.

    Parameters:
        element_types: The element types of the shape `Coord`.
        out_dtype: The dtype of the returned coordinate values.

    Args:
        flat_index: The flat index to convert.
        shape: The shape of the ND space we are converting into.
        skip_dim: The dimension to skip over. This represents the dimension
                  which is being iterated across.

    Constraints:
        The shape `Coord` must be flat (no nested `Coord` elements).

    Returns:
        Constructed ND-index.
    """
    comptime assert Coord[
        *element_types
    ].is_flat, "`_get_nd_indices_from_flat_index` requires a flat `Coord` shape"

    comptime rank = Coord[*element_types].rank
    comptime Result = DynamicCoord[out_dtype, rank]
    var res = Result()

    # The inner dimensions ([outer, outer, inner]) are not traversed if
    # drop last is set.
    comptime if rank == 2:
        if skip_dim == 1:
            res[0] = rebind[Result.element_types[0]](
                Scalar[out_dtype](flat_index)
            )
            res[1] = rebind[Result.element_types[1]](Scalar[out_dtype](0))
        else:
            res[0] = rebind[Result.element_types[0]](Scalar[out_dtype](0))
            res[1] = rebind[Result.element_types[1]](
                Scalar[out_dtype](flat_index)
            )
        return res

    var curr_index = Scalar[out_dtype](flat_index)

    comptime for i in reversed(range(rank)):
        # There is one dimension we skip, this represents the inner loop that
        # is being traversed.
        if i == skip_dim:
            res[i] = rebind[Result.element_types[i]](Scalar[out_dtype](0))
        else:
            var quotient, remainder = divmod(
                curr_index, Scalar[out_dtype](shape[i].value())
            )
            curr_index = quotient
            res[i] = rebind[Result.element_types[i]](remainder)

    return res


@inline(.always)
def _get_nd_indices_from_flat_index(
    flat_index: Int, shape: IndexList, skip_dim: Int, out res: type_of(shape)
):
    """Converts a flat index into ND indices but skip over one of the dimensions.

    This overload forwards to the `Coord` implementation. See it for details.

    Args:
        flat_index: The flat index to convert.
        shape: The shape of the ND space we are converting into.
        skip_dim: The dimension to skip over. This represents the dimension
                  which is being iterated across.
    Returns:
        Constructed ND-index.
    """
    comptime dtype = type_of(shape).element_type
    var coords = _get_nd_indices_from_flat_index[out_dtype=dtype](
        flat_index, Coord(shape), skip_dim
    )

    res = {}
    comptime for i in range(type_of(shape).size):
        res.data[i] = rebind[type_of(shape)._int_type](coords[i].value())


# ===-----------------------------------------------------------------------===#
# MOGG reduce functions.
# These take lambdas and don't assume contiguous inputs so can compose
# with mogg kernels / fusion.
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _reduce_generator[
    num_reductions: Int,
    init_type: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: Int](Coord) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: SIMDLength](
        Coord, StaticTuple[SIMD[dtype, width], num_reductions]
    ) -> None,
    ReduceFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[ty: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) -> SIMD[ty, width],
    //,
    target: StaticString = "cpu",
    *,
    reduce_dim: Int,
](
    shape: Coord,
    init: StaticTuple[Scalar[init_type], num_reductions],
    input_0_fn: InputFnType,
    output_0_fn: OutputFnType,
    reduce_function: ReduceFnType,
    context: Optional[DeviceContext] = None,
) raises:
    """Reduce the given tensor using the given reduction function. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        InputFnType: The type of `input_0_fn` (inferred).
        OutputFnType: The type of `output_0_fn` (inferred).
        ReduceFnType: The type of `reduce_function` (inferred).
        target: The target to run on.
        reduce_dim: The dimension we are reducing.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        context: The pointer to DeviceContext.
    """
    comptime assert is_valid_target[target](), "unsupported target"

    # The plugin hook only accepts an `IndexList`, and the empty-dimension
    # check below needs runtime indexing that a `Coord` does not support, so
    # materialize an `IndexList` view of the shape once.
    var shape_index_list = coord_to_index_list(shape)

    for i in range(len(shape_index_list)):
        if shape_index_list[i] == 0:
            return

    comptime if is_cpu[target]():
        _reduce_generator_cpu[reduce_dim=reduce_dim](
            shape, init, input_0_fn, output_0_fn, reduce_function
        )
    elif case .Some(plugin_reduce) = CurrentPlugin.reduce_generator_fn:
        # The plugin hook takes `reduce_dim` as a runtime argument; feed it the
        # compile-time value.
        return plugin_reduce[num_reductions, init_type](
            shape_index_list,
            init,
            reduce_dim,
            input_0_fn,
            output_0_fn,
            reduce_function,
        )
    else:
        _reduce_generator_gpu[reduce_dim=reduce_dim](
            shape,
            init,
            input_0_fn,
            output_0_fn,
            reduce_function,
            context.value(),
        )


@inline(.always)
def _reduce_generator[
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: Int](Coord) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: SIMDLength](Coord, SIMD[dtype, width]) -> None,
    ReduceFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[ty: DType, width: SIMDLength](
        SIMD[ty, width], SIMD[ty, width]
    ) -> SIMD[ty, width],
    //,
    target: StaticString = "cpu",
    *,
    reduce_dim: Int,
](
    shape: Coord,
    init: Scalar,
    input_0_fn: InputFnType,
    output_0_fn: OutputFnType,
    reduce_function: ReduceFnType,
    context: Optional[DeviceContext] = None,
) raises:
    """Reduce the given tensor using the given reduction function.

    Parameters:
        InputFnType: The type of `input_0_fn` (inferred).
        OutputFnType: The type of `output_0_fn` (inferred).
        ReduceFnType: The type of `reduce_function` (inferred).
        target: The target to run on.
        reduce_dim: The dimension we are reducing.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        context: The pointer to DeviceContext.
    """

    comptime num_reductions = 1

    @inline(.always)
    def output_fn_wrapper[
        dtype: DType, width: SIMDLength
    ](indices: Coord, val: StaticTuple[SIMD[dtype, width], num_reductions]) {
        var output_0_fn
    }:
        output_0_fn[dtype, width](indices, val[0])

    @inline(.always)
    def reduce_fn_wrapper[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](val: SIMD[dtype, width], acc: SIMD[dtype, width]) {
        var reduce_function
    } -> SIMD[dtype, width]:
        comptime assert (
            reduction_idx < num_reductions
        ), "invalid reduction index"
        return reduce_function[dtype, width](val, acc)

    var init_wrapped = StaticTuple[Scalar[init.dtype], num_reductions](init)
    return _reduce_generator[target=target, reduce_dim=reduce_dim](
        shape,
        init_wrapped,
        input_0_fn,
        output_fn_wrapper,
        reduce_fn_wrapper,
        context,
    )


# ===-----------------------------------------------------------------------===#
# CPU-only public API functions
# ===-----------------------------------------------------------------------===#


# `input_gen_fn` and `output_fn` may read and write the same buffer, which
# is safe and the expected pattern here.
@inline(.always)
@__unsafe_nested_origins_read_only
def map_reduce[
    InputGenFnType: def[dtype: DType, width: Int](Int) -> SIMD[dtype, width],
    ReduceVecToVecFnType: def[acc_type: DType, dtype: DType, width: SIMDLength](
        SIMD[acc_type, width], SIMD[dtype, width]
    ) -> SIMD[acc_type, width],
    OutputFnType: def[dtype_: DType, width: SIMDLength, alignment: Int](
        Int, SIMD[dtype_, width]
    ) -> None,
    //,
    simd_width: SIMDLength,
    dtype: DType,
    acc_type: DType,
    reduce_vec_to_scalar_fn: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> Scalar[dtype],
](
    input_gen_fn: InputGenFnType,
    reduce_vec_to_vec_fn: ReduceVecToVecFnType,
    output_fn: OutputFnType,
    length: Int,
    init: Scalar[acc_type],
) -> Scalar[acc_type]:
    """Performs a vectorized map-reduce operation over a sequence.

    Parameters:
        InputGenFnType: The type of `input_gen_fn` (inferred).
        ReduceVecToVecFnType: The type of `reduce_vec_to_vec_fn` (inferred).
        OutputFnType: The type of `output_fn` (inferred).
        simd_width: The SIMD vector width to use.
        dtype: The data type of the input elements.
        acc_type: The data type of the accumulator.
        reduce_vec_to_scalar_fn: Function that reduces a final vector to a scalar.

    Args:
        input_gen_fn: Function that generates input values at each index.
        reduce_vec_to_vec_fn: Function that reduces a vector into the accumulator.
        output_fn: Function to output intermediate results.
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


@inline(.always)
def reduce[
    dtype: DType,
    ReduceFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[acc_type: DType, dtype: DType, width: SIMDLength](
        SIMD[acc_type, width], SIMD[dtype, width]
    ) -> SIMD[acc_type, width],
    //,
](
    src: Span[Scalar[dtype], _], init: Scalar[dtype], reduce_fn: ReduceFnType
) raises -> Scalar[dtype]:
    """Computes a custom reduction of buffer elements.

    Parameters:
        dtype: The dtype of the input.
        ReduceFnType: The type of `reduce_fn` (inferred).

    Args:
        src: The input buffer.
        init: The initial value to use in accumulator.
        reduce_fn: The lambda implementing the reduction.

    Returns:
        The computed reduction value.

    Raises:
        If the operation fails.
    """

    @inline(.always)
    def input_fn[
        _dtype: DType, width: Int
    ](idx: Coord) {imm} -> SIMD[_dtype, width]:
        return (
            src.unsafe_ptr()
            .unsafe_load[width=width](Int(idx[0]))
            ._refine[_dtype]()
        )

    var out: Scalar[init.dtype] = 0

    @inline(.always)
    def output_fn[
        _dtype: DType, width: SIMDLength
    ](indices: Coord, value: SIMD[_dtype, width]) {mut out}:
        out = value._refine[init.dtype, 1]()

    @inline(.always)
    def reduce_fn_wrapper[
        _dtype: DType, width: SIMDLength
    ](acc: SIMD[_dtype, width], val: SIMD[_dtype, width]) {
        var reduce_fn
    } -> SIMD[_dtype, width]:
        return reduce_fn(acc, val)

    var shape = Coord((len(src),))

    _reduce_generator[reduce_dim=0](
        shape, init, input_fn, output_fn, reduce_fn_wrapper
    )

    return out


# ===-----------------------------------------------------------------------===#
# max (Span overload)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _simd_max[
    dtype: DType,
    simd_width: SIMDLength,
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


@inline(.always)
def _simd_max_elementwise[
    acc_type: DType,
    dtype: DType,
    simd_width: SIMDLength,
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
    return reduce(src, Scalar[dtype].MIN, _simd_max_elementwise)


# ===-----------------------------------------------------------------------===#
# min (Span overload)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _simd_min_elementwise[
    acc_type: DType, dtype: DType, simd_width: SIMDLength
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
    return reduce(src, Scalar[dtype].MAX, _simd_min_elementwise)


# ===-----------------------------------------------------------------------===#
# sum (Span and 1d overloads)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _simd_sum[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


@inline(.always)
def _simd_sum_elementwise[
    acc_type: DType, dtype: DType, simd_width: SIMDLength
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

    @inline(.always)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) {imm} -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().unsafe_load[width=width](idx)
        )

    return sum[dtype](len(src), input_fn_1d)


def sum[
    dtype: DType,
    InputFn1DType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width],
](length: Int, input_fn_1d: InputFn1DType) raises -> Scalar[dtype]:
    """
    Computes the sum of a 1D array using a provided input function.

    This function performs a reduction (sum) over a 1-dimensional array of the specified length and data type.
    The input values are provided by the `input_fn_1d` function, which takes an index and returns a SIMD vector
    of the specified width and data type. The reduction is performed using a single thread for deterministic results.

    Parameters:
        dtype: The data type of the elements to sum.
        InputFn1DType: The type of `input_fn_1d` (inferred).

    Args:
        length: The number of elements in the 1D array.
        input_fn_1d: A function that takes a data type, SIMD width, and
          index, and returns a SIMD vector of input values.

    Returns:
        The sum of all elements as a scalar of the specified data type.

    Raises:
        Any exception raised by the input function or reduction process.
    """

    @inline(.always)
    def input_fn_nd[
        _dtype: DType, width: Int
    ](idx: Coord) {var input_fn_1d} -> SIMD[_dtype, width]:
        return input_fn_1d[dtype, width](Int(idx[0]))._refine[_dtype]()

    var out: Scalar[dtype] = 0

    @inline(.always)
    def output_fn[
        _dtype: DType, width: SIMDLength
    ](indices: Coord, value: SIMD[_dtype, width]) {mut out}:
        out = value._refine[dtype, 1]()

    @inline(.always)
    def reduce_fn_wrapper[
        dtype: DType, width: SIMDLength
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return acc + val

    var shape = Coord((length,))

    _reduce_generator[reduce_dim=0](
        shape, Scalar[dtype](0), input_fn_nd, output_fn, reduce_fn_wrapper
    )

    return out


# ===-----------------------------------------------------------------------===#
# product (Span overload)
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _simd_product_elementwise[
    acc_type: DType, dtype: DType, simd_width: SIMDLength
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
    return reduce(src, Scalar[dtype](1), _simd_product_elementwise)


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

    @inline(.always)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) {imm} -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().unsafe_load[width=width](idx)
        )

    return mean[dtype](len(src), input_fn_1d)


def mean[
    dtype: DType,
    InputFn1DType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width],
](length: Int, input_fn_1d: InputFn1DType) raises -> Scalar[dtype]:
    """Computes the arithmetic mean of values generated by a function.

    Parameters:
        dtype: The data type of the elements.
        InputFn1DType: The type of `input_fn_1d` (inferred).

    Args:
        length: The number of elements to average.
        input_fn_1d: A function that generates SIMD values at each index.

    Returns:
        The mean value. For integral types, uses integer division.

    Raises:
        To comply with how generators are used in this module.
    """
    var total = sum[dtype](length, input_fn_1d)

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

    @inline(.always)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) {var src} -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().unsafe_load[width=width](idx)
        )

    return variance[dtype](len(src), input_fn_1d, mean_value, correction)


def variance[
    dtype: DType,
    InputFn1DType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width],
](
    length: Int,
    input_fn_1d: InputFn1DType,
    mean_value: Scalar[dtype],
    correction: Int = 1,
) raises -> Scalar[dtype]:
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
        InputFn1DType: The type of `input_fn_1d` (inferred).

    Args:
        length: The number of elements.
        input_fn_1d: A function that generates SIMD values at each index.
        mean_value: The pre-computed mean value.
        correction: Normalize variance by size - correction (default: 1 for sample variance).

    Returns:
        The variance value.

    Raises:
        If length is less than or equal to correction.
    """

    @inline(.always)
    def input_fn_nd[
        _dtype: DType, width: Int
    ](idx: Coord) {var input_fn_1d, var mean_value} -> SIMD[_dtype, width]:
        var mean_simd = SIMD[mean_value.dtype, width](mean_value).cast[_dtype]()
        var x = input_fn_1d[_dtype, width](Int(idx[0]))
        var diff = x.cast[_dtype]() - mean_simd
        return (diff * diff)._refine[_dtype]()

    var out: Scalar[dtype] = 0

    @inline(.always)
    def output_fn[
        _dtype: DType, width: SIMDLength
    ](indices: Coord, value: SIMD[_dtype, width]) {mut out}:
        out = value._refine[dtype, 1]()

    @inline(.always)
    def reduce_fn_wrapper[
        dtype: DType, width: SIMDLength
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return acc + val

    var shape = Coord((length,))

    _reduce_generator[reduce_dim=0](
        shape,
        Scalar[mean_value.dtype](0),
        input_fn_nd,
        output_fn,
        reduce_fn_wrapper,
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

    @inline(.always)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) {imm} -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.unsafe_ptr().unsafe_load[width=width](idx)
        )

    return variance[dtype](len(src), input_fn_1d, correction)


def variance[
    dtype: DType,
    InputFn1DType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width],
](
    length: Int, input_fn_1d: InputFn1DType, correction: Int = 1
) raises -> Scalar[dtype]:
    """Computes the variance of values generated by a function.

    This version computes the mean automatically in a first pass.

    Parameters:
        dtype: The data type of the elements.
        InputFn1DType: The type of `input_fn_1d` (inferred).

    Args:
        length: The number of elements.
        input_fn_1d: A function that generates SIMD values at each index.
        correction: Normalize variance by size - correction (default: 1 for sample variance).

    Returns:
        The variance value.

    Raises:
        If length is less than or equal to correction.
    """
    var mean_value = mean[dtype](length, input_fn_1d)
    return variance[dtype](length, input_fn_1d, mean_value, correction)


# ===-----------------------------------------------------------------------===#
# cumsum function
# ===-----------------------------------------------------------------------===#


@inline(.always)
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
        var x_simd = src.unsafe_ptr().unsafe_load[width=simd_width](i)

        comptime for i in range(rep):
            x_simd += x_simd.shift_right[2**i]()

        dst.unsafe_ptr().unsafe_store(i, x_simd)

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
        var x_simd = dst.unsafe_ptr().unsafe_load[width=simd_width](i) + offset
        dst.unsafe_ptr().unsafe_store(i, x_simd)
        offset = SIMD[dtype, simd_width](x_simd[simd_width - 1])

    # Handles the tail, i.e., num of elements at the end that don't
    # fit within a simd_width-elements vector.
    for i in range(div_size, len(dst)):
        dst[i] = dst[i - 1] + src[i]
