# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements SIMD reductions.

You can import these APIs from the `algorithm` package. For example:

```mojo
from algorithm import map_reduce
```
"""

from math import all_true as _all_true
from math import any_true as _any_true
from math import div_ceil, iota, align_down
from math import min as _min
from math import none_true as _none_true
from math import align_down
from math.bit import cttz
from math.limit import max_or_inf, min_or_neginf
from sys.info import (
    is_little_endian,
    simdwidthof,
    sizeof,
    triple_is_nvidia_cuda,
)
from runtime.tracing import TraceLevel

from algorithm import (
    sync_parallelize,
    unroll,
    vectorize,
)
from algorithm.functional import _get_num_workers
from algorithm._gpu.reduction import reduce_launch
from memory.buffer import Buffer, NDBuffer, prod_dims
from memory.unsafe import Pointer
from runtime.llcl import OutputChainPtr

from utils.index import StaticIntTuple, Index
from utils.list import Dim, DimList

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _index_of_first_one[width: Int](val: SIMD[DType.bool, width]) -> Int:
    """Computes the index of the first one in the input value.

    The input is assumed to contain at least one non-zero element.

    Args:
      val: The mask containing ones and zeros.

    Returns:
      The index of the first one in the input mask.
    """

    constrained[is_little_endian(), "only correct on little endian systems"]()

    @parameter
    if width == 1:
        return 0
    elif width == 2:
        return 0 if val[0] else 1
    elif width == 4:
        if val[0]:
            return 0
        if val[1]:
            return 1
        if val[2]:
            return 2
        return 3
    elif width == 8:
        # Cast to int8 and count the number of trailing zeros.
        var local_val = val
        let i8_ptr = Pointer.address_of(local_val).bitcast[Int8]()
        return cttz(int(i8_ptr.load()))
    elif width == 16:
        # Cast to int16 and count the number of trailing zeros.
        var local_val = val
        let i16_ptr = Pointer.address_of(local_val).bitcast[Int16]()
        return cttz(int(i16_ptr.load()))
    elif width == 32:
        # Cast to int32 and count the number of trailing zeros.
        var local_val = val
        let i32_ptr = Pointer.address_of(local_val).bitcast[Int32]()
        return cttz(int(i32_ptr.load()))
    else:
        alias half_width: Int = width // 2
        let lhs = val.slice[half_width](0)
        if lhs != 0:
            return _index_of_first_one(lhs)

        let rhs = val.slice[half_width](half_width)
        return half_width + _index_of_first_one(rhs)


# ===----------------------------------------------------------------------===#
# ND indexing helper
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_nd_indices_from_flat_index[
    rank: Int,
](
    flat_index: Int, shape: StaticIntTuple[rank], skip_dim: Int
) -> StaticIntTuple[rank]:
    """Converts a flat index into ND indices but skip over one of the dimensons.

    The ND indices will iterate from right to left. I.E

    shape = (20, 5, 2, N)
    _get_nd_indices_from_flat_index(1, shape, rank -1) = (0, 0, 1, 0)
    _get_nd_indices_from_flat_index(5, shape, rank -1) = (0, 2, 1, 0)
    _get_nd_indices_from_flat_index(50, shape, rank -1) = (5, 0, 0, 0)
    _get_nd_indices_from_flat_index(56, shape, rank -1) = (5, 1, 1, 0)

    We ignore the Nth dimension to allow that to be traversed in the elementwise
    function.

    Parameters:
        rank: The rank of the ND index.
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
    @parameter
    if rank == 2:
        if skip_dim == 1:
            return StaticIntTuple[rank](flat_index, 0)
        else:
            return StaticIntTuple[rank](0, flat_index)

    var out = StaticIntTuple[rank]()
    var curr_index = flat_index

    @parameter
    if triple_is_nvidia_cuda():
        for i in range(rank - 1, -1, -1):
            # There is one dimension we skip, this represents the inner loop that
            # is being traversed.
            if i == skip_dim:
                out[i] = 0
            else:
                out[i] = curr_index._positive_rem(shape[i])
                curr_index = curr_index._positive_div(shape[i])
    else:

        @unroll
        for i in range(rank - 1, -1, -1):
            # There is one dimension we skip, this represents the inner loop that
            # is being traversed.
            if i == skip_dim:
                out[i] = 0
            else:
                out[i] = curr_index._positive_rem(shape[i])
                curr_index = curr_index._positive_div(shape[i])

    return out


# ===----------------------------------------------------------------------===#
# reduce
# ===----------------------------------------------------------------------===#


@always_inline
@parameter
fn map_reduce[
    simd_width: Int,
    size: Dim,
    type: DType,
    acc_type: DType,
    input_gen_fn: fn[type: DType, width: Int] (Int) capturing -> SIMD[
        type, width
    ],
    reduce_vec_to_vec_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
    reduce_vec_to_scalar_fn: fn[type: DType, width: Int] (
        SIMD[type, width]
    ) -> Scalar[type],
](dst: Buffer[size, type], init: Scalar[acc_type]) -> Scalar[acc_type]:
    """Stores the result of calling input_gen_fn in dst and simultaneously
    reduce the result using a custom reduction function.

    Parameters:
        simd_width: The vector width for the computation.
        size: The buffer size.
        type: The buffer elements dtype.
        acc_type: The dtype of the reduction accumulator.
        input_gen_fn: A function that generates inputs to reduce.
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
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor
    let length = len(dst)
    let unrolled_vector_end = align_down(length, unrolled_simd_width)
    let vector_end = align_down(length, simd_width)

    var acc_unrolled_simd = SIMD[acc_type, unrolled_simd_width].splat(init)
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        let val_simd = input_gen_fn[type, unrolled_simd_width](i)
        dst.simd_store(i, val_simd)
        acc_unrolled_simd = reduce_vec_to_vec_fn(acc_unrolled_simd, val_simd)

    var acc_simd = SIMD[acc_type, simd_width].splat(init)
    for i in range(unrolled_vector_end, vector_end, simd_width):
        let val_simd = input_gen_fn[type, simd_width](i)
        dst.simd_store(i, val_simd)
        acc_simd = reduce_vec_to_vec_fn(acc_simd, val_simd)

    var acc = reduce_vec_to_scalar_fn[acc_type, unrolled_simd_width](
        acc_unrolled_simd
    )
    acc = reduce_vec_to_vec_fn(acc, reduce_vec_to_scalar_fn(acc_simd))
    for i in range(vector_end, length):
        let val = input_gen_fn[type, 1](i)
        dst[i] = val
        acc = reduce_vec_to_vec_fn(acc, val)
    return acc[0].value


@always_inline
@parameter
fn reduce[
    reduce_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width]
](src: Buffer, init: Scalar) -> Scalar[init.element_type]:
    """Computes a custom reduction of buffer elements.

    Parameters:
        reduce_fn: The lambda implementing the reduction.
        : Ignore.

    Args:
        src: The input buffer.
        init: The initial value to use in accumulator.

    Returns:
        The computed reduction value.
    """

    @always_inline
    @parameter
    fn input_fn[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](src.simd_load[width](idx[0]))

    var out: Scalar[init.element_type] = 0

    @always_inline
    @parameter
    fn output_fn[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        out = rebind[Scalar[init.element_type]](value)

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        type: DType, width: Int
    ](acc: SIMD[type, width], val: SIMD[type, width]) -> SIMD[type, width]:
        return reduce_fn(acc, val)

    let shape = Index(len(src))

    try:
        _reduce_generator[
            input_fn,
            output_fn,
            reduce_fn_wrapper,
            single_thread_blocking_override=True,
        ](shape, init=init, reduce_dim=0, out_chain=OutputChainPtr())
    except e:
        trap(e)
    return out


@always_inline
@parameter
fn reduce_boolean[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width]
    ) capturing -> Bool,
    continue_fn: fn (Bool) capturing -> Bool,
](src: Buffer, init: Bool) -> Bool:
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
        : Ignore.

    Args:
        src: The input buffer.
        init: The initial value to use.

    Returns:
        The computed reduction value.
    """
    alias simd_width = simdwidthof[src.type]()
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor

    let length = len(src)
    let unrolled_vector_end = align_down(length, unrolled_simd_width)
    let vector_end = align_down(length, simd_width)
    var curr = init
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        curr = reduce_fn(src.simd_load[unrolled_simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(unrolled_vector_end, vector_end, simd_width):
        curr = reduce_fn(src.simd_load[simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(vector_end, length):
        curr = reduce_fn(src[i])
        if not continue_fn(curr):
            return curr
    return curr


@parameter
fn _reduce_3D[
    map_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
    reduce_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> Scalar[type],
](src: NDBuffer, dst: NDBuffer, init: Scalar[dst.type]):
    """Performs a reduction across axis 1 of a 3D input buffer."""

    alias simd_width = simdwidthof[dst.type]()

    let h = src.dim[0]()
    let w = src.dim[1]()
    let c = src.dim[2]()

    # If c is 1, we are reducing across the innermost axis, and we can launch H
    # reductions that each reduce W elements of a contiguous buffer.
    if c == 1:

        @parameter
        fn reduce_inner_axis():
            alias sz = src.shape.at[1]()
            # TODO: parallelize
            for i in range(h):
                let offset = src._offset(StaticIntTuple[src.rank](i, 0, 0))
                let input = Buffer[
                    sz, src.type, address_space = src.address_space
                ](offset, w)
                let val = reduce[map_fn](input, init)
                dst[StaticIntTuple[dst.rank](i, 0)] = val

        reduce_inner_axis()
        return

    # If c is not 1, the elements to reduce are not contiguous. If we reduce
    # cache_line_size rows at once, then we get better reuse of the loaded data.

    # The width of this should be a multiple of the cache line size in order to
    # reuse the full cache line when an element of C is loaded.
    @parameter
    fn get_unroll_factor[simd_width: Int, dtype_size: Int]() -> Int:
        alias cache_line_size = 64
        alias unroll_factor = cache_line_size // (simd_width * dtype_size)
        constrained[unroll_factor > 0, "unroll_factor must be > 0"]()
        return unroll_factor

    alias unroll_factor = get_unroll_factor[simd_width, sizeof[dst.type]()]()
    alias usimd_width = unroll_factor * simd_width
    for i in range(h):

        @always_inline
        @parameter
        fn reduce_w_chunked[simd_width: Int](idx: Int):
            var accum = SIMD[init.element_type, simd_width].splat(init)
            for j in range(w):
                let chunk = src.simd_load[simd_width](
                    StaticIntTuple[src.rank](i, j, idx)
                )
                accum = map_fn(accum, chunk)
            dst.simd_store(StaticIntTuple[dst.rank](i, idx), accum)

        vectorize[usimd_width, reduce_w_chunked](c)


@parameter
fn reduce[
    map_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
    reduce_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> Scalar[type],
    reduce_axis: Int,
](src: NDBuffer, dst: NDBuffer, init: Scalar[dst.type]):
    """Performs a reduction across reduce_axis of an NDBuffer (src) and stores
    the result in an NDBuffer (dst).

    First src is reshaped into a 3D tensor. Without loss of generality, the three
    axes will be referred to as [H,W,C], where the axis to reduce across is W,
    the axes before the reduce axis are packed into H, and the axes after the
    reduce axis are packed into C. i.e. a tensor with dims [D1, D2, ..., Di, ..., Dn]
    reducing across axis i gets packed into a 3D tensor with dims [H, W, C],
    where H=prod(D1,...,Di-1), W = Di, and C = prod(Di+1,...,Dn).

    Parameters:
        map_fn: A mapping function. This function is used when to combine
          (accumulate) two chunks of input data: e.g. we load two 8xfloat32 vectors
          of elements and need to reduce them to a single 8xfloat32 vector.
        reduce_fn: A reduction function. This function is used to reduce a
          vector to a scalar. E.g. when we got 8xfloat32 vector and want to reduce
          it to 1xfloat32.
        reduce_axis: The axis to reduce across.
        : Ignore.

    Args:
        src: The input buffer.
        dst: The output buffer.
        init: The initial value to use in accumulator.
    """

    let h_dynamic = prod_dims[0, reduce_axis](src)
    let w_dynamic = src.dim[reduce_axis]()
    let c_dynamic = prod_dims[reduce_axis + 1, src.rank](src)

    alias h_static = src.shape.product_range[0, reduce_axis]()
    alias w_static = src.shape.at[reduce_axis]()
    alias c_static = src.shape.product_range[reduce_axis + 1, src.rank]()

    alias input_3d_shape = DimList(h_static, w_static, c_static)
    alias output_2d_shape = DimList(h_static, c_static)

    let input_3d = NDBuffer[
        3, input_3d_shape, src.type, address_space = src.address_space
    ](src.data, Index(h_dynamic, w_dynamic, c_dynamic))
    let output_2d = NDBuffer[
        2, output_2d_shape, dst.type, address_space = dst.address_space
    ](
        dst.data,
        Index(h_dynamic, c_dynamic),
    )

    _reduce_3D[map_fn, reduce_fn](input_3d, output_2d, init)


# ===----------------------------------------------------------------------===#
# MOGG reduce functions.
# These take lambdas and don't assume contiguous inputs so can compose
# with mogg kernels / fusion.
# ===----------------------------------------------------------------------===#


@always_inline
@adaptive
fn _reduce_generator[
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple,
    init: Scalar,
    reduce_dim: Int,
    out_chain: OutputChainPtr,
) raises:
    """Reduce the given tensor using the given reduction function.

    Constraints:
        Target must be "cuda".

    Parameters:
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.
        target: The target to run on.
        : Ignore.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The our chain to attach results to.
    """
    constrained[target == "cuda", "only valid on GPUs"]()

    let reduce_dim_normalized = (
        len(shape) + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    if reduce_dim_normalized != len(shape) - 1:
        raise "GPU reduction currently limited to inner axis."

    let stream = out_chain.get_cuda_stream()
    reduce_launch[
        input_0_fn,
        output_0_fn,
        reduce_function,
    ](shape, reduce_dim_normalized, init, stream)


@always_inline
@adaptive
fn _reduce_generator[
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple,
    init: Scalar,
    reduce_dim: Int,
    out_chain: OutputChainPtr,
) raises:
    """Reduce the given tensor using the given reduction function.

    Constraints:
        Target must be "cpu".

    Parameters:
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.
        target: The target to run on.
        : Ignore.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The our chain to attach results to.
    """
    constrained[target == "cpu", "only valid on CPUs"]()

    alias rank = shape.size

    let reduce_dim_normalized = (
        rank + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    @parameter
    if shape.size == 1:
        _reduce_along_inner_dimension[
            input_0_fn,
            output_0_fn,
            reduce_function,
            single_thread_blocking_override=single_thread_blocking_override,
        ](shape, init, reduce_dim_normalized, out_chain)
    else:
        if rank - 1 == reduce_dim_normalized:
            _reduce_along_inner_dimension[
                input_0_fn,
                output_0_fn,
                reduce_function,
                single_thread_blocking_override=single_thread_blocking_override,
            ](shape, init, reduce_dim_normalized, out_chain)
        else:
            _reduce_along_outer_dimension[
                input_0_fn,
                output_0_fn,
                reduce_function,
                single_thread_blocking_override=single_thread_blocking_override,
            ](shape, init, reduce_dim_normalized, out_chain)


fn _reduce_along_inner_dimension[
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
](
    shape: StaticIntTuple,
    init_value: Scalar,
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.

    Parameters:
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.
        : Ignore.

    Args:
        shape: The shape of the tensor we are reducing
        init_value: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The chain to attach results to.
    """
    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    let total_size: Int = shape.flattened_length()
    if total_size == 0:
        return

    let reduce_dim_size = shape[reduce_dim]

    let parallelism_size: Int = total_size // reduce_dim_size

    let num_workers: Int

    @parameter
    if single_thread_blocking_override:
        num_workers = 1
    else:
        num_workers = _get_num_workers(total_size, out_chain.get_runtime())

    let chunk_size = div_ceil(parallelism_size, num_workers)

    alias unroll_factor = 8
    alias simd_width = simdwidthof[init_value.element_type]()
    alias unrolled_simd_width = simd_width * unroll_factor

    @always_inline
    @parameter
    fn reduce_rows_unrolled(start_row: Int, end_row: Int):
        # Manually hoist this out of the loops anyway. Not clear if we want to
        # hoist it all the way out of the async body.

        let unrolled_simd_compatible_size = align_down(
            reduce_dim_size, unrolled_simd_width
        )
        let simd_compatible_size = align_down(reduce_dim_size, simd_width)
        let simd_tail_size = (
            simd_compatible_size - unrolled_simd_compatible_size
        ) // simd_width
        let simd_reduce_size = unroll_factor if unrolled_simd_compatible_size > 0 else simd_tail_size
        let scalar_tail_size = reduce_dim_size - simd_compatible_size

        # Iterate over the non reduced dimensions.
        for flat_index in range(start_row, end_row):
            # In normal elementwise get_nd_indices skips the last dimension as it is the dimension being iterated over. In our case we don't know this yet so we do have to calculate the extra one.
            var indices = _get_nd_indices_from_flat_index(
                flat_index, shape, reduce_dim
            )

            # initialize our unrolled accumulator
            var acc_unrolled = StaticTuple[
                unroll_factor, SIMD[init_value.element_type, simd_width]
            ]()

            @unroll
            for i in range(unroll_factor):
                acc_unrolled[i] = SIMD[
                    init_value.element_type, simd_width
                ].splat(init_value)

            # Loop over unroll_factor*simd_width chunks.
            for idx in range(
                0, unrolled_simd_compatible_size, unrolled_simd_width
            ):

                @unroll
                for j in range(unroll_factor):
                    indices[reduce_dim] = idx + j * simd_width
                    let load_value = input_0_fn[
                        init_value.element_type, simd_width, shape.size
                    ](indices)
                    acc_unrolled[j] = reduce_function(
                        load_value, acc_unrolled[j]
                    )

            # Handle remaining simd_width chunks.
            @unroll
            for i in range(unroll_factor - 1):
                if i < simd_tail_size:
                    indices[reduce_dim] = (
                        i * simd_width + unrolled_simd_compatible_size
                    )
                    let load_value = input_0_fn[
                        init_value.element_type, simd_width, shape.size
                    ](indices)
                    acc_unrolled[i] = reduce_function(
                        load_value, acc_unrolled[i]
                    )

            # Reduce unrolled elements to a single SIMD value
            if simd_reduce_size == 2:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
            elif simd_reduce_size == 3:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[2], acc_unrolled[0]
                )
            elif simd_reduce_size == 4:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
                acc_unrolled[2] = reduce_function(
                    acc_unrolled[3], acc_unrolled[2]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[2], acc_unrolled[0]
                )
            elif simd_reduce_size == 5:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
                acc_unrolled[2] = reduce_function(
                    acc_unrolled[3], acc_unrolled[2]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[4], acc_unrolled[0]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[2], acc_unrolled[0]
                )
            elif simd_reduce_size == 6:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
                acc_unrolled[2] = reduce_function(
                    acc_unrolled[3], acc_unrolled[2]
                )
                acc_unrolled[4] = reduce_function(
                    acc_unrolled[5], acc_unrolled[4]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[2], acc_unrolled[0]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[4], acc_unrolled[0]
                )
            elif simd_reduce_size == 7:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
                acc_unrolled[2] = reduce_function(
                    acc_unrolled[3], acc_unrolled[2]
                )
                acc_unrolled[4] = reduce_function(
                    acc_unrolled[5], acc_unrolled[4]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[2], acc_unrolled[0]
                )
                acc_unrolled[4] = reduce_function(
                    acc_unrolled[6], acc_unrolled[4]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[4], acc_unrolled[0]
                )
            elif simd_reduce_size == 8:
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[1], acc_unrolled[0]
                )
                acc_unrolled[2] = reduce_function(
                    acc_unrolled[3], acc_unrolled[2]
                )
                acc_unrolled[4] = reduce_function(
                    acc_unrolled[5], acc_unrolled[4]
                )
                acc_unrolled[6] = reduce_function(
                    acc_unrolled[7], acc_unrolled[6]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[2], acc_unrolled[0]
                )
                acc_unrolled[4] = reduce_function(
                    acc_unrolled[6], acc_unrolled[4]
                )
                acc_unrolled[0] = reduce_function(
                    acc_unrolled[4], acc_unrolled[0]
                )

            # Reduce tail scalar values into our SIMD accumulator
            var acc_simd = acc_unrolled[0]

            @unroll
            for i in range(simd_width - 1):
                if i < scalar_tail_size:
                    indices[reduce_dim] = simd_compatible_size + i
                    let load_val = input_0_fn[
                        init_value.element_type, 1, shape.size
                    ](indices)
                    acc_simd[i] = reduce_function(load_val, acc_simd[i])

            # Reduce our SIMD accumulator to a scalar
            let acc_scalar = acc_simd.reduce[reduce_function]()

            # Store the result back to the output.
            indices[reduce_dim] = 0
            output_0_fn(indices, acc_scalar)

    @always_inline
    @parameter
    fn reduce_rows(i: Int):
        let start_parallel_offset = i * chunk_size
        let end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        let length = end_parallel_offset - start_parallel_offset
        if length <= 0:
            return

        reduce_rows_unrolled(start_parallel_offset, end_parallel_offset)

    @parameter
    if single_thread_blocking_override:
        reduce_rows_unrolled(0, parallelism_size)
    else:
        sync_parallelize[reduce_rows](out_chain, num_workers)


fn _reduce_along_outer_dimension[
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
](
    shape: StaticIntTuple,
    init: Scalar,
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.

    Parameters:
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.
        : Ignore.

    Args:
        shape: The shape of the tensor we are reducing
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The chain to attach results to.
    """
    alias rank = shape.size
    alias type = init.element_type

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    alias simd_width = simdwidthof[type]()

    let total_size: Int = shape.flattened_length()
    if total_size == 0:
        return

    let reduce_dim_size = shape[reduce_dim]
    let inner_dim = shape[rank - 1]

    # parallelize across slices of the input, where a slice is [reduce_dim, inner_dim]
    # the slice is composed of [reduce_dim, simd_width] chunks
    # these chunks are reduced simaltaneously across the reduce_dim using simd instructions
    # and accumulation
    let parallelism_size: Int = total_size // (reduce_dim_size * inner_dim)

    let num_workers: Int

    @parameter
    if single_thread_blocking_override:
        num_workers = 1
    else:
        num_workers = _get_num_workers(total_size, out_chain.get_runtime())

    let chunk_size = div_ceil(parallelism_size, num_workers)

    @parameter
    fn reduce_slices(i: Int):
        let start_parallel_offset = i * chunk_size
        let end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        let length = end_parallel_offset - start_parallel_offset

        if length <= 0:
            return

        for slice_idx in range(start_parallel_offset, end_parallel_offset):

            @always_inline
            @parameter
            fn reduce_chunk[simd_width: Int](inner_dim_idx: Int):
                var acc_simd = SIMD[init.element_type, simd_width].splat(init)
                let reduce_vector_idx = slice_idx * inner_dim + inner_dim_idx
                var indices = _get_nd_indices_from_flat_index(
                    reduce_vector_idx, shape, reduce_dim
                )
                for reduce_dim_idx in range(reduce_dim_size):
                    indices[reduce_dim] = reduce_dim_idx
                    let load_value = input_0_fn[
                        init.type, simd_width, shape.size
                    ](indices)
                    acc_simd = reduce_function(load_value, acc_simd)
                # Store the result back to the output.
                indices[reduce_dim] = 0
                output_0_fn(indices, acc_simd)

            vectorize[simd_width, reduce_chunk](inner_dim)

    @parameter
    if single_thread_blocking_override:
        reduce_slices(0)
    else:
        sync_parallelize[reduce_slices](out_chain, num_workers)


# ===----------------------------------------------------------------------===#
# max
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_max[
    type: DType,
    simd_width: Int,
](x: SIMD[type, simd_width]) -> Scalar[type]:
    """Computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


@always_inline
@closure
@parameter
fn _simd_max_elementwise[
    acc_type: DType,
    type: DType,
    simd_width: Int,
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise max of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x.max(y.cast[acc_type]())


fn max(src: Buffer) -> Scalar[src.type]:
    """Computes the max element in a buffer.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        The maximum of the buffer elements.
    """
    return reduce[_simd_max_elementwise](src, min_or_neginf[src.type]())


fn max[reduce_axis: Int](src: NDBuffer, dst: NDBuffer[src.rank, _, src.type]):
    """Computes the max across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.
        : Ignore.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_max_elementwise, _simd_max, reduce_axis](
        src, dst, min_or_neginf[src.type]()
    )


# ===----------------------------------------------------------------------===#
# min
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> Scalar[type]:
    """Computes the min element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_min()


@always_inline
@closure
@parameter
fn _simd_min_elementwise[
    acc_type: DType, type: DType, simd_width: Int
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise min of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x.min(y.cast[acc_type]())


fn min(src: Buffer) -> Scalar[src.type]:
    """Computes the min element in a buffer.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        The minimum of the buffer elements.
    """
    return reduce[_simd_min_elementwise](src, max_or_inf[src.type]())


fn min[reduce_axis: Int](src: NDBuffer, dst: NDBuffer[src.rank, _, src.type]):
    """Computes the min across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.
        : Ignore.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_min_elementwise, _simd_min, reduce_axis](
        src, dst, max_or_inf[src.type]()
    )


# ===----------------------------------------------------------------------===#
# sum
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_sum[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> Scalar[type]:
    """Computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


@always_inline
@closure
@parameter
fn _simd_sum_elementwise[
    acc_type: DType, type: DType, simd_width: Int
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise sum of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x + y.cast[acc_type]()


fn sum(src: Buffer) -> Scalar[src.type]:
    """Computes the sum of buffer elements.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        The sum of the buffer elements.
    """
    return reduce[_simd_sum_elementwise](src, Scalar[src.type](0))


fn sum[reduce_axis: Int](src: NDBuffer, dst: NDBuffer[src.rank, _, src.type]):
    """Computes the sum across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.
        : Ignore.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_sum_elementwise, _simd_sum, reduce_axis=reduce_axis](
        src, dst, Scalar[src.type](0)
    )


# ===----------------------------------------------------------------------===#
# product
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_product[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> Scalar[type]:
    """Computes the product of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_mul()


@always_inline
@closure
@parameter
fn _simd_product_elementwise[
    acc_type: DType, type: DType, simd_width: Int
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise product of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x * y.cast[acc_type]()


fn product(src: Buffer) -> Scalar[src.type]:
    """Computes the product of the buffer elements.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        The product of the buffer elements.
    """
    return reduce[_simd_product_elementwise](src, Scalar[src.type](1))


fn product[
    reduce_axis: Int
](src: NDBuffer, dst: NDBuffer[src.rank, _, src.type]):
    """Computes the product across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.
        : Product.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_product_elementwise, _simd_product, reduce_axis](
        src, dst, Scalar[src.type](1)
    )


# ===----------------------------------------------------------------------===#
# mean
# ===----------------------------------------------------------------------===#


@adaptive
fn mean(src: Buffer) -> Scalar[src.type]:
    """Computes the mean value of the elements in a buffer.

    Parameters:
        : Ignore.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.
    """

    debug_assert(len(src) != 0, "input must not be empty")

    let total = sum(src)
    let buffer_len = len(src)

    @parameter
    if src.type.is_integral():
        return total // buffer_len
    else:
        return total / buffer_len


fn mean[reduce_axis: Int](src: NDBuffer, dst: NDBuffer[src.rank, _, src.type]):
    """Computes the mean across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.
        : Ignore.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    alias simd_width = simdwidthof[dst.type]()
    sum[reduce_axis](src, dst)

    let n = src.dim[reduce_axis]()
    let dst_1d = dst.flatten()

    @parameter
    if dst.type.is_integral():

        @always_inline
        @parameter
        fn normalize_integral[simd_width: Int](idx: Int):
            let elem = dst_1d.simd_load[simd_width](idx)
            let to_store = elem // n
            dst_1d.simd_store(idx, to_store)

        vectorize[simd_width, normalize_integral](len(dst_1d))
    else:
        let n_recip = Scalar[dst.type](1) / n

        @always_inline
        @parameter
        fn normalize_floating[simd_width: Int](idx: Int):
            let elem = dst_1d.simd_load[simd_width](idx)
            let to_store = elem * n_recip
            dst_1d.simd_store(idx, to_store)

        vectorize[simd_width, normalize_floating](len(dst_1d))


@adaptive
fn mean[
    type: DType,
    input_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple,
    reduce_dim: Int,
    output_shape: StaticIntTuple[input_shape.size],
    out_chain: OutputChainPtr,
) raises:
    """Computes the mean across the input and output shape.

    This performs the mean computation on the domain specified by `input_shape`,
    storing the results using the`input_0_fn`. The results' domain is
    `output_shape` which are stored using the `output_0_fn`.

    Parameters:
        type: The type of the input and output.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.
        target: The target to run on.
        : Ignore.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the mean on.
        output_shape: The output shape.
        out_chain: The output chain to use.
    """

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    @always_inline
    fn input_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_fn[width, rank](idx))

    # For floats apply the reciprocal as a multiply.
    @parameter
    if type.is_floating_point():
        # Apply mean division before storing to the output lambda.
        let reciprocal = 1.0 / input_shape[reduce_dim]

        @always_inline
        @parameter
        fn wrapped_output_mul[
            _type: DType, width: Int, rank: Int
        ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
            let mean_val = value * reciprocal
            output_fn[width, rank](indices, rebind[SIMD[type, width]](mean_val))

        _reduce_generator[
            input_fn_wrapper,
            wrapped_output_mul,
            reduce_impl,
            single_thread_blocking_override=single_thread_blocking_override,
            target=target,
        ](
            input_shape,
            init=Scalar[type](0),
            reduce_dim=reduce_dim,
            out_chain=out_chain,
        )

    else:
        # For ints just a normal divide.
        let dim_size = input_shape[reduce_dim]

        @always_inline
        @parameter
        fn wrapped_output_div[
            _type: DType, width: Int, rank: Int
        ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
            let mean_val = value / dim_size
            output_fn[width, rank](indices, rebind[SIMD[type, width]](mean_val))

        _reduce_generator[
            input_fn_wrapper,
            wrapped_output_div,
            reduce_impl,
            single_thread_blocking_override=single_thread_blocking_override,
            target=target,
        ](
            input_shape,
            init=Scalar[type](0),
            reduce_dim=reduce_dim,
            out_chain=out_chain,
        )


# ===----------------------------------------------------------------------===#
# variance
# ===----------------------------------------------------------------------===#


fn variance(
    src: Buffer, mean_value: Scalar[src.type], correction: Int = 1
) -> Scalar[src.type]:
    """Given a mean, computes the variance of elements in a buffer.

    The mean value is used to avoid a second pass over the data:

    ```
    variance = sum((x - E(x))^2) / (size - correction)
    ```

    Parameters:
        : Ignore.

    Args:
        src: The buffer.
        mean_value: The mean value of the buffer.
        correction: Normalize variance by size - correction.

    Returns:
        The variance value of the elements in a buffer.
    """

    debug_assert(len(src) > 1, "input length must be greater than 1")

    @always_inline
    @parameter
    fn input_fn[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        let mean_simd = SIMD[mean_value.type, width].splat(mean_value).cast[
            _type
        ]()
        let x = src.simd_load[width](idx[0])
        let diff = x.cast[_type]() - mean_simd
        return rebind[SIMD[_type, width]](diff * diff)

    var out: Scalar[src.type] = 0

    @always_inline
    @parameter
    fn output_fn[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        out = rebind[Scalar[src.type]](value)

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        type: DType, width: Int
    ](acc: SIMD[type, width], val: SIMD[type, width]) -> SIMD[type, width]:
        return acc + val

    let shape = StaticIntTuple[1](len(src))

    try:
        _reduce_generator[
            input_fn,
            output_fn,
            reduce_fn_wrapper,
            single_thread_blocking_override=True,
        ](
            shape,
            init=Scalar[mean_value.type](0),
            reduce_dim=0,
            out_chain=OutputChainPtr(),
        )
    except e:
        trap(e)
    return out / (len(src) - correction)


fn variance(src: Buffer, correction: Int = 1) -> Scalar[src.type]:
    """Computes the variance value of the elements in a buffer.

    ```
    variance(src) = sum((x - E(x))^2) / (size - correction)
    ```

    Parameters:
        : Ignore.

    Args:
        src: The buffer.
        correction: Normalize variance by size - correction (Default=1).

    Returns:
        The variance value of the elements in a buffer.
    """

    let mean_value = mean(src)
    return variance(src, mean_value, correction)


# ===----------------------------------------------------------------------===#
# all_true
# ===----------------------------------------------------------------------===#


fn all_true(src: Buffer) -> Bool:
    """Returns True if all the elements in a buffer are True and False otherwise.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        True if all of the elements of the buffer are True and False otherwise.
    """

    @always_inline
    @parameter
    fn _reduce_fn[
        type: DType, simd_width: Int
    ](val: SIMD[type, simd_width]) -> Bool:
        @parameter
        if type == DType.bool:
            return _all_true(val.cast[DType.bool]())
        return _all_true(val != 0)

    @always_inline
    @parameter
    fn _continue_fn(val: Bool) -> Bool:
        return val

    return reduce_boolean[_reduce_fn, _continue_fn](src, False)


# ===----------------------------------------------------------------------===#
# any_true
# ===----------------------------------------------------------------------===#


fn any_true(src: Buffer) -> Bool:
    """Returns True if any the elements in a buffer are True and False otherwise.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        True if any of the elements of the buffer are True and False otherwise.
    """

    @always_inline
    @parameter
    fn _reduce_fn[
        type: DType, simd_width: Int
    ](val: SIMD[type, simd_width]) -> Bool:
        @parameter
        if type == DType.bool:
            return _any_true(val.cast[DType.bool]())
        return _any_true(val != 0)

    @always_inline
    @parameter
    fn _continue_fn(val: Bool) -> Bool:
        return not val

    return reduce_boolean[_reduce_fn, _continue_fn](src, False)


# ===----------------------------------------------------------------------===#
# none_true
# ===----------------------------------------------------------------------===#


fn none_true(src: Buffer) -> Bool:
    """Returns True if none of the elements in a buffer are True and False
    otherwise.

    Parameters:
        : Ignore.

    Args:
        src: The buffer.

    Returns:
        True if none of the elements of the buffer are True and False otherwise.
    """

    @always_inline
    @parameter
    fn _reduce_fn[
        type: DType, simd_width: Int
    ](val: SIMD[type, simd_width]) -> Bool:
        @parameter
        if type == DType.bool:
            return _none_true(val.cast[DType.bool]())
        return _none_true(val != 0)

    @always_inline
    @parameter
    fn _continue_fn(val: Bool) -> Bool:
        return val

    return reduce_boolean[_reduce_fn, _continue_fn](src, True)


# ===----------------------------------------------------------------------===#
# _argn
# ===----------------------------------------------------------------------===#


fn _argn[
    is_max: Bool
](
    input: NDBuffer,
    axis: Int,
    output: NDBuffer,
    out_chain: OutputChainPtr,
) raises:
    """
    Finds the indices of the maximum/minimum element along the specified axis.

    Parameters:
        is_max: If True compute then compute argmax, otherwise compute the
                argmin.
        : Ignore.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        out_chain: The chain to attach results to.
    """
    alias rank = input.rank
    alias simd_width = simdwidthof[input.type]()

    var canonical_axis = axis
    if canonical_axis < 0:
        canonical_axis += rank
    if not 0 <= canonical_axis < rank:
        raise Error("axis must be between [0, <input rank>)")

    # TODO: Generalize to mid axis.
    if canonical_axis != rank - 1:
        raise Error("axis other than innermost not supported yet")

    @unroll
    for subaxis in range(rank):
        if subaxis == canonical_axis:
            if output.dim(subaxis) != 1:
                raise Error("expected axis to have size 1 in output")
        elif input.dim(subaxis) != output.dim(subaxis):
            raise Error("input and output dims must match aside from 'axis'")

    let axis_size = input.dim(canonical_axis)
    let input_stride: Int
    let output_stride: Int
    let num_workers: Int
    let chunk_size: Int
    var parallel_size = 1

    @parameter
    if rank == 1:
        input_stride = input.num_elements()
        output_stride = output.num_elements()
        num_workers = 1
        chunk_size = 1
    else:
        input_stride = input.dynamic_stride[canonical_axis - 1]
        output_stride = output.dynamic_stride[canonical_axis - 1]

        for i in range(canonical_axis):
            parallel_size *= input.dim(i)

        # don't over-schedule if parallel_size < _get_num_workers output
        num_workers = _min(
            _get_num_workers(
                input.dynamic_shape.flattened_length(), out_chain.get_runtime()
            ),
            parallel_size,
        )
        chunk_size = div_ceil(parallel_size, num_workers)

    @parameter
    fn task_func(task_id: Int):
        @parameter
        @always_inline
        fn cmpeq[
            type: DType, simd_width: Int
        ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
            DType.bool, simd_width
        ]:
            @parameter
            if is_max:
                return a <= b
            else:
                return a >= b

        @parameter
        @always_inline
        fn cmp[
            type: DType, simd_width: Int
        ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
            DType.bool, simd_width
        ]:
            @parameter
            if is_max:
                return a < b
            else:
                return a > b

        # iterate over flattened axes
        let start = task_id * chunk_size
        let end = _min((task_id + 1) * chunk_size, parallel_size)
        for i in range(start, end):
            var input_offset = i * input_stride
            var output_offset = i * output_stride
            let input_dim_ptr = input.data.offset(input_offset)
            let output_dim_ptr = output.data.offset(output_offset)
            var global_val: Scalar[input.type]

            # initialize limits
            @parameter
            if is_max:
                global_val = min_or_neginf[input.type]()
            else:
                global_val = max_or_inf[input.type]()

            # initialize vector of maximal/minimal values
            var global_values: SIMD[input.type, simd_width]
            if axis_size < simd_width:
                global_values = global_val
            else:
                global_values = input_dim_ptr.simd_load[simd_width]()

            # iterate over values evenly divisible by simd_width
            var indices = iota[output.type, simd_width]()
            var global_indices = indices
            let last_simd_index = align_down(axis_size, simd_width)
            for j in range(simd_width, last_simd_index, simd_width):
                let curr_values = input_dim_ptr.simd_load[simd_width](j)
                indices += simd_width

                let mask = cmpeq(curr_values, global_values)
                global_indices = mask.select(global_indices, indices)
                global_values = mask.select(global_values, curr_values)

            @parameter
            if is_max:
                global_val = global_values.reduce_max()
            else:
                global_val = global_values.reduce_min()

            # Check trailing indices.
            var idx = Scalar[output.type](0)
            var found_min: Bool = False
            for j in range(last_simd_index, axis_size, 1):
                let elem = input_dim_ptr.load(j)
                if cmp(global_val, elem):
                    global_val = elem
                    idx = j
                    found_min = True

            # handle the case where min wasn't in trailing values
            if not found_min:
                var matching = global_values == global_val
                var min_indices = matching.select(
                    global_indices, max_or_inf[output.type]()
                )
                idx = min_indices.reduce_min()
            output_dim_ptr.store(idx)

    sync_parallelize[task_func](out_chain, parallel_size)


# ===----------------------------------------------------------------------===#
# argmax
# ===----------------------------------------------------------------------===#


fn argmax(
    input: NDBuffer,
    axis: Int,
    output: NDBuffer,
    out_chain: OutputChainPtr,
) raises:
    """
    Finds the indices of the maximum element along the specified axis.

    Parameters:
        : Ignore.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        out_chain: The chain to attach results to.
    """

    _argn[is_max=True](input, axis, output, out_chain)


fn argmax(
    input: NDBuffer,
    axis_buf: NDBuffer,
    output: NDBuffer,
    out_chain: OutputChainPtr,
) raises:
    """
    Finds the indices of the maximum element along the specified axis.

    Parameters:
        : Ignore.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
        out_chain: The chain to attach results to.
    """

    argmax(input, int(axis_buf[0]), output, out_chain)


# ===----------------------------------------------------------------------===#
# argmin
# ===----------------------------------------------------------------------===#


fn argmin(
    input: NDBuffer,
    axis: Int,
    output: NDBuffer,
    out_chain: OutputChainPtr,
) raises:
    """
    Finds the indices of the maximum element along the specified axis.

    Parameters:
        : Ignore.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        out_chain: The chain to attach results to.
    """

    _argn[is_max=False](input, axis, output, out_chain)


fn argmin(
    input: NDBuffer,
    axis_buf: NDBuffer,
    output: NDBuffer,
    out_chain: OutputChainPtr,
) raises:
    """
    Finds the indices of the minimum element along the specified axis.

    Parameters:
        : Ignore.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
        out_chain: The chain to attach results to.
    """

    argmin(input, int(axis_buf[0]), output, out_chain)


# ===----------------------------------------------------------------------===#
# shape function
# ===----------------------------------------------------------------------===#


fn reduce_shape[
    input_rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_buf: The input tensor.
        axis_buf: The axis tensor.

    Returns:
        The output shape.
    """

    # extract hyper parameter
    var axis = int(axis_buf[0])
    if axis < 0:
        axis += input_rank
    # TODO(#17512)
    debug_assert(
        0 <= axis and axis < input_rank,
        "normalized axis must be within range [0, input_rank)",
    )

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[axis] = 1
    return output_shape


# ===----------------------------------------------------------------------===#
# cumsum function
# ===----------------------------------------------------------------------===#


@always_inline
fn _floorlog2[n: Int]() -> Int:
    return 0 if n <= 1 else 1 + _floorlog2[n >> 1]()


@always_inline
fn _static_log2[n: Int]() -> Int:
    return 0 if n <= 1 else _floorlog2[n - 1]() + 1


@always_inline
fn _cumsum_small(dst: Buffer, src: Buffer[dst.size, dst.type]):
    dst[0] = src[0]
    for i in range(1, len(dst)):
        dst[i] = src[i] + dst[i - 1]


fn cumsum(dst: Buffer, src: Buffer[dst.size, dst.type]):
    """Computes the cumulative sum of all elements in a buffer.
       dst[i] = src[i] + src[i-1] + ... + src[0].

    Parameters:
        : Ignore.

    Args:
        dst: The buffer that stores the result of cumulative sum operation.
        src: The buffer of elements for which the cumulative sum is computed.
    """

    debug_assert(len(src) != 0, "Input must not be empty")
    debug_assert(len(dst) != 0, "Output must not be empty")

    alias simd_width = simdwidthof[dst.type]()

    # For length less than simd_width do serial cumulative sum.
    # Similarly, for the case when simd_width == 2 serial should be faster.
    if len(dst) < simd_width or simd_width == 2:
        return _cumsum_small(dst, src)

    # Stores the offset (i.e., last value of previous simd_width-elements chunk,
    # replicated across all simd lanes, to be added to all elements of next
    # chunk.
    var offset = SIMD[dst.type, simd_width](0)

    # Divide the buffer size to div_size chunks of simd_width elements,
    # to calculate using SIMD and do remaining (tail) serially.
    let div_size = align_down(len(dst), simd_width)

    # Number of inner-loop iterations (for shift previous result and add).
    alias rep = _static_log2[simd_width]()

    for i in range(0, div_size, simd_width):
        var x_simd = src.simd_load[simd_width](i)

        @parameter
        fn loop_body[idx: Int]():
            x_simd += x_simd.shift_right[2**idx]()

        unroll[rep, loop_body]()
        dst.simd_store(i, x_simd)

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
        let x_simd = dst.simd_load[simd_width](i) + offset
        dst.simd_store(i, x_simd)
        offset = offset.splat(x_simd[simd_width - 1])

    # Handles the tail, i.e., num of elements at the end that don't
    # fit within a simd_width-elements vector.
    for i in range(div_size, len(dst)):
        dst[i] = dst[i - 1] + src[i]
