# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements SIMD reductions."""

from Assert import assert_param, debug_assert
from Buffer import Buffer, NDBuffer, prod_dims
from DType import DType
from Functional import (
    vectorize,
    _get_num_workers,
    async_parallelize,
    unroll,
)
from Index import StaticIntTuple
from List import DimList, Dim
from LLCL import OutputChainPtr
from Math import (
    all_true as _all_true,
    any_true as _any_true,
    none_true as _none_true,
    div_ceil,
    min as _min,
)
from Limits import max_or_inf, min_or_neginf
from Range import range
from SIMD import SIMD
from TargetInfo import sizeof, simdwidthof

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
    if rank == 2 and skip_dim == 1:
        return StaticIntTuple[rank](flat_index, 0)
    elif rank == 2:
        return StaticIntTuple[rank](0, flat_index)

    var out = StaticIntTuple[rank]()
    var curr_index = flat_index

    @always_inline
    @parameter
    fn compute_shape[idx: Int]():
        alias i = rank - idx - 1
        # There is one dimension we skip, this represents the inner loop that
        # is being traversed.
        if i == skip_dim:
            out[i] = 0
        else:
            out[i] = curr_index % shape[i]
            curr_index //= shape[i]

    unroll[rank, compute_shape]()

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
    ) -> SIMD[type, 1],
](dst: Buffer[size, type], init: SIMD[acc_type, 1]) -> SIMD[acc_type, 1]:
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
    var acc_simd = SIMD[acc_type, unrolled_simd_width].splat(init)
    let len = dst.__len__()
    let vector_end = (len // unrolled_simd_width) * unrolled_simd_width
    for i in range(0, vector_end, unrolled_simd_width):
        let val_simd = input_gen_fn[type, unrolled_simd_width](i)
        dst.simd_store(i, val_simd)
        acc_simd = reduce_vec_to_vec_fn[acc_type, type, unrolled_simd_width](
            acc_simd, val_simd
        )

    var acc = reduce_vec_to_scalar_fn[acc_type, unrolled_simd_width](acc_simd)
    for i in range(vector_end, len):
        let val = input_gen_fn[type, 1](i)
        dst[i] = val
        acc = reduce_vec_to_vec_fn[acc_type, type, 1](acc, val)
    return acc[0].value


@always_inline
@parameter
fn reduce[
    simd_width: Int,
    size: Dim,
    type: DType,
    acc_type: DType,
    map_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
    reduce_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, 1],
](src: Buffer[size, type], init: SIMD[acc_type, 1]) -> SIMD[acc_type, 1]:
    """Computes a custom reduction of buffer elements.

    Parameters:
        simd_width: The vector width for the computation.
        size: The buffer size.
        type: The buffer elements dtype.
        acc_type: The dtype of the reduction accumulator.
        map_fn: A mapping function. This function is used when to combine
          (accumulate) two chunks of input data: e.g. we load two 8xfloat32 vectors
          of elements and need to reduce them to a single 8xfloat32 vector.
        reduce_fn: A reduction function. This function is used to reduce a
          vector to a scalar. E.g. when we got 8xfloat32 vector and want to reduce
          it to 1xfloat32.

    Args:
        src: The input buffer.
        init: The initial value to use in accumulator.

    Returns:
        The computed reduction value.
    """
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor
    var acc_simd = SIMD[acc_type, unrolled_simd_width].splat(init)
    let len = src.__len__()
    let vector_end = (len // unrolled_simd_width) * unrolled_simd_width
    for i in range(0, vector_end, unrolled_simd_width):
        acc_simd = map_fn[acc_type, type, unrolled_simd_width](
            acc_simd, src.simd_load[unrolled_simd_width](i)
        )

    var acc = reduce_fn[acc_type, unrolled_simd_width](acc_simd)
    for i in range(vector_end, len):
        acc = map_fn[acc_type, type, 1](acc, src[i])
    return acc[0].value


@always_inline
@parameter
fn reduce_boolean[
    simd_width: Int,
    size: Dim,
    type: DType,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width]
    ) capturing -> Bool,
    continue_fn: fn (Bool) capturing -> Bool,
](src: Buffer[size, type], init: Bool) -> Bool:
    """Computes a bool reduction of buffer elements. The reduction will early
    exit if the `continue_fn` returns False.

    Parameters:
        simd_width: The vector width for the computation.
        size: The buffer size.
        type: The buffer elements dtype.
        reduce_fn: A boolean reduction function. This function is used to reduce
          a vector to a scalar. E.g. when we got `8xfloat32` vector and want to
          reduce it to a `bool`.
        continue_fn: A function to indicate whether we want to continue
          processing the rest of the iterations. This takes the result of the
          reduce_fn and returns True to continue processing and False to early
          exit.

    Args:
        src: The input buffer.
        init: The initial value to use.

    Returns:
        The computed reduction value.
    """
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor
    let len = src.__len__()
    let vector_end = (len // unrolled_simd_width) * unrolled_simd_width
    var curr = init
    for i in range(0, vector_end, unrolled_simd_width):
        curr = reduce_fn[type, unrolled_simd_width](
            src.simd_load[unrolled_simd_width](i)
        )
        if not continue_fn(curr):
            return curr

    for ii in range(vector_end, len):  # TODO(#8365) use `i`
        curr = reduce_fn[type, 1](src[ii])
        if not continue_fn(curr):
            return curr
    return curr


@always_inline
@parameter
fn _reduce_3D[
    simd_width: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    acc_type: DType,
    map_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
    reduce_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, 1],
](
    src: NDBuffer[3, input_shape, type],
    dst: NDBuffer[
        2,
        output_shape,
        acc_type,
    ],
    init: SIMD[acc_type, 1],
):
    """Performs a reduction across axis 1 of a 3D input buffer."""

    let h = src.dim[0]()
    let w = src.dim[1]()
    let c = src.dim[2]()

    # If c is 1, we are reducing across the innermost axis, and we can launch H
    # reductions that each reduce W elements of a contiguous buffer.
    if c == 1:

        @always_inline
        @parameter
        fn reduce_inner_axis():
            alias sz = input_shape.at[1]()
            # TODO: parallelize
            for i in range(h):
                let offset = src._offset(StaticIntTuple[3](i, 0, 0))
                let input = Buffer[sz, type](offset, w)
                let val = reduce[
                    simd_width, sz, type, acc_type, map_fn, reduce_fn
                ](input, init)
                dst[StaticIntTuple[2](i, 0)] = val

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
        assert_param[unroll_factor > 0, "unroll_factor must be > 0"]()
        return unroll_factor

    alias unroll_factor = get_unroll_factor[simd_width, sizeof[type]()]()
    alias usimd_width = unroll_factor * simd_width
    for i in range(h):

        @always_inline
        @parameter
        fn reduce_w_chunked[simd_width: Int](idx: Int):
            var accum = SIMD[acc_type, simd_width].splat(init)
            for j in range(w):
                let chunk = src.simd_load[simd_width](
                    StaticIntTuple[3](i, j, idx)
                )
                accum = map_fn[acc_type, type, simd_width](accum, chunk)
            dst.simd_store[simd_width](StaticIntTuple[2](i, idx), accum)

        vectorize[usimd_width, reduce_w_chunked](c)


@always_inline
@parameter
fn reduce[
    simd_width: Int,
    rank: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    acc_type: DType,
    map_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
    reduce_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[type, 1],
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, acc_type],
    init: SIMD[acc_type, 1],
):
    """Performs a reduction across reduce_axis of an NDBuffer (src) and stores
    the result in an NDBuffer (dst).

    First src is reshaped into a 3D tensor. Without loss of generality, the three
    axes will be referred to as [H,W,C], where the axis to reduce across is W,
    the axes before the reduce axis are packed into H, and the axes after the
    reduce axis are packed into C. i.e. a tensor with dims [D1, D2, ..., Di, ..., Dn]
    reducing across axis i gets packed into a 3D tensor with dims [H, W, C],
    where H=prod(D1,...,Di-1), W = Di, and C = prod(Di+1,...,Dn).

    Parameters:
        simd_width: The vector width for the computation.
        rank: The rank of the input/output buffers.
        input_shape: The input buffer shape.
        output_shape: The output buffer shape.
        type: The buffer elements dtype.
        acc_type: The dtype of the reduction accumulator.
        map_fn: A mapping function. This function is used when to combine
          (accumulate) two chunks of input data: e.g. we load two 8xfloat32 vectors
          of elements and need to reduce them to a single 8xfloat32 vector.
        reduce_fn: A reduction function. This function is used to reduce a
          vector to a scalar. E.g. when we got 8xfloat32 vector and want to reduce
          it to 1xfloat32.
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
        init: The initial value to use in accumulator.
    """

    let h_dynamic = prod_dims[0, reduce_axis](src)
    let w_dynamic = src.dim[reduce_axis]()
    let c_dynamic = prod_dims[reduce_axis + 1, rank](src)

    alias h_static = input_shape.product_range[0, reduce_axis]()
    alias w_static = input_shape.at[reduce_axis]()
    alias c_static = input_shape.product_range[reduce_axis + 1, rank]()

    alias input_3d_shape = DimList(h_static, w_static, c_static)
    alias output_3d_shape = DimList(h_static, c_static)

    let input_3d = NDBuffer[3, input_3d_shape, type](
        src.data, StaticIntTuple[3](h_dynamic, w_dynamic, c_dynamic), type
    )
    let output_3d = NDBuffer[2, output_3d_shape, acc_type](
        dst.data, StaticIntTuple[2](h_dynamic, c_dynamic), acc_type
    )

    _reduce_3D[
        simd_width,
        input_3d_shape,
        output_3d_shape,
        type,
        acc_type,
        map_fn,
        reduce_fn,
    ](input_3d, output_3d, init)


# ===----------------------------------------------------------------------===#
# MOGG reduce functions.
# These take lambdas and don't assume contiguous inputs so can compose
# with mogg kernels / fusion.
# ===----------------------------------------------------------------------===#


@always_inline
@adaptive
fn _reduce_generator[
    type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    init_value: SIMD[type, 1],
    reduce_dim_maybe_neg: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.
    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        type: The element type we are reducing.
        rank: The rank of the tensor.
        simd_width: The SIMD vector width to use.
        single_thread_blocking_override: if set will run immediately
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.

    Args:
        input: The tensor we are reducing.
        init_value: The value to start the reduction from.
        reduce_dim_maybe_neg: The dimension we are reducing.
        out_chain: The our chain to attach results to.
    """
    assert_param[rank == 1, "Specialization for 1D"]()

    let shape = input.dynamic_shape
    let total_size: Int = shape[0]
    let simd_compatible_size = (total_size // simd_width) * simd_width

    if total_size == 0:

        @parameter
        if not single_thread_blocking_override:
            out_chain.mark_ready()
        return

    @always_inline
    @parameter
    fn reduce(ignored: Int):
        var acc_simd = SIMD[type, simd_width].splat(init_value)
        for idx in range(0, simd_compatible_size, simd_width):
            let indices = StaticIntTuple[rank](idx)
            let load_value = input_0_fn[type, simd_width, rank](indices)
            acc_simd = reduce_function[type, simd_width](load_value, acc_simd)

        # Final reduction. SIMD -> scalar.
        var acc_scalar = SIMD[type, 1].splat(init_value)
        for i in range(simd_width):
            let indices = StaticIntTuple[rank](i)
            acc_scalar = reduce_function[type, 1](acc_scalar, acc_simd[i])

        # The simds might not cover all the elements so we still need to scalar reduce those too.
        for i in range(simd_compatible_size, total_size, 1):
            let indices = StaticIntTuple[rank](i)
            let load_value = input_0_fn[type, 1, rank](indices)
            acc_scalar = reduce_function[type, 1](load_value, acc_scalar)

        # Store the result back to the output.
        let indices = StaticIntTuple[rank](0)
        output_0_fn[type, 1, rank](indices, acc_scalar)

    @parameter
    if single_thread_blocking_override:
        reduce(0)
    else:
        # Until the threading model allows partials we have to launch this on one
        # thread.
        async_parallelize[reduce](out_chain, 1)


@always_inline
@adaptive
fn _reduce_generator[
    type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    init_value: SIMD[type, 1],
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.
    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        type: The element type we are reducing.
        rank: The rank of the tensor.
        simd_width: The SIMD vector width to use.
        single_thread_blocking_override: if set will run immediately
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.

    Args:
        input: The tensor we are reducing.
        init_value: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The our chain to attach results to.
    """
    assert_param[rank > 1, "Specialization for ND where N > 1"]()

    let reduce_dim_normalized = (
        rank + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    # If the input is strided along the input dimension then we can simd
    # reduce over it directly.
    # TODO: Support more optimal case for reduce over non-strided.
    if input.stride(reduce_dim_normalized) == 1:
        alias unroll_factor = simd_width * 8
        _reduce_along_dimension[
            type,
            rank,
            simd_width,
            unroll_factor,
            single_thread_blocking_override,
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](input, init_value, reduce_dim_normalized, out_chain)
    else:
        # Scalar fallback.
        _reduce_along_dimension[
            type,
            rank,
            1,
            1,
            single_thread_blocking_override,
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](input, init_value, reduce_dim_normalized, out_chain)


@always_inline
fn _reduce_along_dimension[
    type: DType,
    rank: Int,
    simd_width: Int,
    unroll_factor: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    init_value: SIMD[type, 1],
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.
    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        type: The element type we are reducing.
        rank: The rank of the tensor.
        simd_width: The SIMD vector width to use.
        unroll_factor: The amount to unroll the inner loop by.
        single_thread_blocking_override: if set will run immediately
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
    Args:
        input: The tensor we are reducing
        init_value: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The chain to attach results to.
    """
    let shape = input.dynamic_shape

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    let total_size: Int = shape.flattened_length()
    if total_size == 0:

        @parameter
        if not single_thread_blocking_override:
            out_chain.mark_ready()
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

    @always_inline
    @parameter
    fn unrolled_inner_loop[simd_width: Int](start: Int, end: Int):
        # Manually hoist this out of the loops anyway. Not clear if we want to
        # hoist it all the way out of the async body.
        let simd_compatible_size = (reduce_dim_size // simd_width) * simd_width

        # Iterate over the non reduced dimensions.
        for flat_index in range(start, end):
            # In normal elementwise get_nd_indices skips the last dimension as it is the dimension being iterated over. In our case we don't know this yet so we do have to calculate the extra one.
            var indices = _get_nd_indices_from_flat_index[rank](
                flat_index, shape, reduce_dim
            )

            var acc_simd = SIMD[type, simd_width].splat(init_value)
            for idx in range(0, simd_compatible_size, simd_width):
                indices[reduce_dim] = idx
                let load_value = input_0_fn[type, simd_width, rank](indices)
                acc_simd = reduce_function[type, simd_width](
                    load_value, acc_simd
                )

            # Semi final reduction. SIMD -> scalar.
            var acc_scalar = acc_simd.reduce[reduce_function]()

            # The simds might not cover all the elements so we still need to scalar reduce those too.
            for i in range(simd_compatible_size, reduce_dim_size, 1):
                indices[reduce_dim] = i
                let load_value = input_0_fn[type, 1, rank](indices)
                acc_scalar = reduce_function[type, 1](load_value, acc_scalar)

            # Store the result back to the output.
            indices[reduce_dim] = 0
            output_0_fn[type, 1, rank](indices, acc_scalar)

    @always_inline
    @parameter
    fn vectorize_over_reduced_dim(i: Int):
        let start_parallel_offset = i * chunk_size
        let end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        let len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        # We will unroll only if it is safe
        # (i.e we won't unroll into other dimensions).
        if unroll_factor > reduce_dim_size:
            unrolled_inner_loop[simd_width](
                start_parallel_offset, end_parallel_offset
            )
        else:
            unrolled_inner_loop[unroll_factor](
                start_parallel_offset, end_parallel_offset
            )

    @parameter
    if single_thread_blocking_override:
        if unroll_factor > reduce_dim_size:
            unrolled_inner_loop[simd_width](0, parallelism_size)
        else:
            unrolled_inner_loop[unroll_factor](0, parallelism_size)
    else:
        async_parallelize[vectorize_over_reduced_dim](out_chain, num_workers)


# ===----------------------------------------------------------------------===#
# max
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_max[
    type: DType,
    simd_width: Int,
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
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


fn max[size: Dim, type: DType](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the max element in a buffer.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

    Args:
        src: The buffer.

    Returns:
        The maximum of the buffer elements.
    """
    return reduce[
        simdwidthof[type](), size, type, type, _simd_max_elementwise, _simd_max
    ](src, src[0])


fn max[
    rank: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the max across reduce_axis of an NDBuffer.

    Parameters:
        rank: The rank of the input/output buffers.
        input_shape: The input buffer shape.
        output_shape: The output buffer shape.
        type: The buffer elements dtype.
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[
        simdwidthof[type](),
        rank,
        input_shape,
        output_shape,
        type,
        type,
        _simd_max_elementwise,
        _simd_max,
        reduce_axis,
    ](src, dst, min_or_neginf[type]())


# ===----------------------------------------------------------------------===#
# min
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
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


fn min[size: Dim, type: DType](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the min element in a buffer.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

    Args:
        src: The buffer.

    Returns:
        The minimum of the buffer elements.
    """
    return reduce[
        simdwidthof[type](), size, type, type, _simd_min_elementwise, _simd_min
    ](src, src[0])


fn min[
    rank: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the min across reduce_axis of an NDBuffer.

    Parameters:
        rank: The rank of the input/output buffers.
        input_shape: The input buffer shape.
        output_shape: The output buffer shape.
        type: The buffer elements dtype.
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[
        simdwidthof[type](),
        rank,
        input_shape,
        output_shape,
        type,
        type,
        _simd_min_elementwise,
        _simd_min,
        reduce_axis,
    ](src, dst, max_or_inf[type]())


# ===----------------------------------------------------------------------===#
# sum
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_sum[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
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


fn sum[size: Dim, type: DType](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the sum of buffer elements.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

    Args:
        src: The buffer.

    Returns:
        The sum of the buffer elements.
    """
    return reduce[
        simdwidthof[type](), size, type, type, _simd_sum_elementwise, _simd_sum
    ](src, 0)


fn sum[
    rank: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the sum across reduce_axis of an NDBuffer.

    Parameters:
        rank: The rank of the input/output buffers.
        input_shape: The input buffer shape.
        output_shape: The output buffer shape.
        type: The buffer elements dtype.
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[
        simdwidthof[type](),
        rank,
        input_shape,
        output_shape,
        type,
        type,
        _simd_sum_elementwise,
        _simd_sum,
        reduce_axis,
    ](src, dst, 0)


# ===----------------------------------------------------------------------===#
# product
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_product[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
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


fn product[size: Dim, type: DType](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the product of the buffer elements.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

    Args:
        src: The buffer.

    Returns:
        The product of the buffer elements.
    """
    return reduce[
        simdwidthof[type](),
        size,
        type,
        type,
        _simd_product_elementwise,
        _simd_product,
    ](src, 1)


fn product[
    rank: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the product across reduce_axis of an NDBuffer.

    Parameters:
        rank: The rank of the input/output buffers.
        input_shape: The input buffer shape.
        output_shape: The output buffer shape.
        type: The buffer elements dtype.
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[
        simdwidthof[type](),
        rank,
        input_shape,
        output_shape,
        type,
        type,
        _simd_product_elementwise,
        _simd_product,
        reduce_axis,
    ](src, dst, 1)


# ===----------------------------------------------------------------------===#
# mean
# ===----------------------------------------------------------------------===#


fn mean[size: Dim, type: DType](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the mean value of the elements in a buffer.

    Parameters:
        size: The size of the input buffer..
        type: The type of the elements of the input buffer and output SIMD
              vector.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.
    """

    debug_assert(src.__len__() != 0, "input must not be empty")

    let total = sum(src)
    let buffer_len = src.__len__()

    @parameter
    if type.is_integral():
        return total // buffer_len
    else:
        return total / buffer_len


fn mean[
    rank: Int,
    input_shape: DimList,
    output_shape: DimList,
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the mean across reduce_axis of an NDBuffer.

    Parameters:
        rank: The rank of the input/output buffers.
        input_shape: The input buffer shape.
        output_shape: The output buffer shape.
        type: The buffer elements dtype.
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """

    alias simd_width = simdwidthof[type]()
    sum[rank, input_shape, output_shape, type, reduce_axis](src, dst)

    let n = src.dim[reduce_axis]()
    let dst_1d = dst.flatten()

    @parameter
    if type.is_integral():

        @always_inline
        @parameter
        fn normalize_integral[simd_width: Int](idx: Int):
            let elem = dst_1d.simd_load[simd_width](idx)
            let to_store = elem // n
            dst_1d.simd_store[simd_width](idx, to_store)

        vectorize[simd_width, normalize_integral](dst_1d.__len__())
    else:
        let n_recip = SIMD[type, 1](1) / n

        @always_inline
        @parameter
        fn normalize_floating[simd_width: Int](idx: Int):
            let elem = dst_1d.simd_load[simd_width](idx)
            let to_store = elem * n_recip
            dst_1d.simd_store[simd_width](idx, to_store)

        vectorize[simd_width, normalize_floating](dst_1d.__len__())


# ===----------------------------------------------------------------------===#
# variance
# ===----------------------------------------------------------------------===#


fn variance[
    size: Dim, type: DType
](
    src: Buffer[size, type],
    mean_value: SIMD[type, 1],
    correction: Int = 1,
) -> SIMD[type, 1]:
    """Given a mean, computes the variance of elements in a buffer.

    The mean value is used to avoid a second pass over the data:

    ```
    variance = sum((x - E(x))^2) / (size - correction)
    ```

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

    Args:
        src: The buffer.
        mean_value: The mean value of the buffer.
        correction: Normalize variance by size - correction.

    Returns:
        The variance value of the elements in a buffer.
    """
    debug_assert(src.__len__() > 1, "input length must be greater than 1")
    alias simd_width = simdwidthof[type]()

    @always_inline
    @parameter
    fn _simd_variance_elementwise[
        acc_type: DType,
        inner_type: DType,
        simd_width: Int,
    ](x: SIMD[acc_type, simd_width], y: SIMD[inner_type, simd_width]) -> SIMD[
        acc_type, simd_width
    ]:
        """Computes the equation $sum (x_i - u)^2 + y$"""
        let mean_simd = SIMD[type, simd_width].splat(mean_value).cast[
            acc_type
        ]()
        let diff = y.cast[acc_type]() - mean_simd
        return x + diff * diff

    let numerator: SIMD[type, 1] = reduce[
        simd_width, size, type, type, _simd_variance_elementwise, _simd_sum
    ](src, 0)
    return numerator / (src.__len__() - correction)


fn variance[
    size: Dim, type: DType
](src: Buffer[size, type], correction: Int = 1) -> SIMD[type, 1]:
    """Computes the variance value of the elements in a buffer.

    ```
    variance(src) = sum((x - E(x))^2) / (size - correction)
    ```

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

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


fn all_true[size: Dim, type: DType](src: Buffer[size, type]) -> Bool:
    """Returns True if all the elements in a buffer are True and False otherwise.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

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

    alias simd_width = simdwidthof[type]()
    return reduce_boolean[simd_width, size, type, _reduce_fn, _continue_fn](
        src, False
    )


# ===----------------------------------------------------------------------===#
# any_true
# ===----------------------------------------------------------------------===#


fn any_true[size: Dim, type: DType](src: Buffer[size, type]) -> Bool:
    """Returns True if any the elements in a buffer are True and False otherwise.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

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

    alias simd_width = simdwidthof[type]()
    return reduce_boolean[simd_width, size, type, _reduce_fn, _continue_fn](
        src, False
    )


# ===----------------------------------------------------------------------===#
# none_true
# ===----------------------------------------------------------------------===#


fn none_true[size: Dim, type: DType](src: Buffer[size, type]) -> Bool:
    """Returns True if none of the elements in a buffer are True and False
    otherwise.

    Parameters:
        size: The buffer size.
        type: The buffer elements dtype.

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

    alias simd_width = simdwidthof[type]()
    return reduce_boolean[simd_width, size, type, _reduce_fn, _continue_fn](
        src, True
    )


# ===----------------------------------------------------------------------===#
# _argn
# ===----------------------------------------------------------------------===#


@always_inline
fn _argn[
    type: DType,
    out_type: DType,
    axis_type: DType,
    rank: Int,
    is_max: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    """
    Finds the indices of the maximum/minimum element along the specified axis.

    Parameters:
        type: Type of the input tensor.
        out_type: Type of the output tensor.
        axis_type: Type of the axis tensor.
        rank: The rank of the input / output.
        is_max: If True compute then compute argmax, otherwise compute the
                argmin.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
        out_chain: The chain to attach results to.
    """
    alias simd_width = simdwidthof[type]()

    var axis = axis_buf[0].to_int()
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        out_chain.mark_error("axis must be between [0, <input rank>)")
        return

    # TODO: Generalize to mid axis.
    if axis != rank - 1:
        out_chain.mark_error("axis other than innermost not supported yet")
        return

    let d0 = input.dim(0)
    let d1 = input.dim(1)

    if output.dim(0) != d0:
        out_chain.mark_error("input and output dims[0] must match")
        return

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        for i in range(d0):
            # TODO: Parameterize the comparator.
            var global_val: SIMD[type, 1]

            @parameter
            if is_max:
                global_val = min_or_neginf[type]()
            else:
                global_val = max_or_inf[type]()
            var idx = 0

            @always_inline
            @parameter
            fn compute_row[simd_width: Int](j: Int):
                let vec = input.simd_load[simd_width](i, j)

                @parameter
                if is_max:
                    let curr_max = vec.reduce_max()
                    if global_val < curr_max:
                        global_val = curr_max

                        @unroll
                        for k in range(simd_width):
                            idx = j + k
                else:
                    let curr_min = vec.reduce_min()
                    if global_val < curr_min:
                        global_val = curr_min

                        @unroll
                        for k in range(simd_width):
                            idx = j + k

            vectorize[simd_width, compute_row](d1)

            let outIndices = StaticIntTuple[rank](i, 0)
            output[outIndices] = idx

    # TODO: Shard by dim 0.
    async_parallelize[task_func](out_chain, 1)


# ===----------------------------------------------------------------------===#
# argmax
# ===----------------------------------------------------------------------===#


@export
@always_inline
fn argmax[
    type: DType,
    out_type: DType,
    axis_type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    """
    Finds the indices of the maximum element along the specified axis.

    Parameters:
        type: Type of the input tensor.
        out_type: Type of the output tensor.
        axis_type: Type of the axis tensor.
        rank: The rank of the input / output.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
        out_chain: The chain to attach results to.
    """

    assert_param[rank == 2, "ArgMax: rank other than 2 not supported yet"]()

    _argn[type, out_type, axis_type, rank, True](
        input, axis_buf, output, out_chain
    )


# ===----------------------------------------------------------------------===#
# argmin
# ===----------------------------------------------------------------------===#


@export
@always_inline
fn argmin[
    type: DType,
    out_type: DType,
    axis_type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    """
    Finds the indices of the minimum element along the specified axis.

    Parameters:
        type: Type of the input tensor.
        out_type: Type of the output tensor.
        axis_type: Type of the axis tensor.
        rank: The rank of the input / output.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
        out_chain: The chain to attach results to.
    """

    assert_param[rank == 2, "ArgMin: rank other than 2 not supported yet"]()

    _argn[type, out_type, axis_type, rank, True](
        input, axis_buf, output, out_chain
    )


# ===----------------------------------------------------------------------===#
# shape function
# ===----------------------------------------------------------------------===#


@always_inline
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
    var axis = axis_buf[0].to_int()
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
