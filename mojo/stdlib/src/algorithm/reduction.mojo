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
from math import div_ceil, iota
from math import min as _min
from math import none_true as _none_true
from math import align_down
from math.bit import cttz
from math.limit import max_or_inf, min_or_neginf
from sys.info import is_little_endian, simdwidthof, sizeof
from runtime.tracing import TraceLevel

from algorithm import async_parallelize, unroll, vectorize
from algorithm.functional import _get_num_workers
from algorithm.gpu.reduction import reduce_launch
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
        return cttz(Int__(i8_ptr.load()))
    elif width == 16:
        # Cast to int16 and count the number of trailing zeros.
        var local_val = val
        let i16_ptr = Pointer.address_of(local_val).bitcast[Int16]()
        return cttz(Int__(i16_ptr.load()))
    elif width == 32:
        # Cast to int32 and count the number of trailing zeros.
        var local_val = val
        let i32_ptr = Pointer.address_of(local_val).bitcast[Int32]()
        return cttz(Int__(i32_ptr.load()))
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
    let length = len(dst)
    let unrolled_vector_end = align_down(length, unrolled_simd_width)
    let vector_end = align_down(length, simd_width)

    var acc_unrolled_simd = SIMD[acc_type, unrolled_simd_width].splat(init)
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        let val_simd = input_gen_fn[type, unrolled_simd_width](i)
        dst.simd_store(i, val_simd)
        acc_unrolled_simd = reduce_vec_to_vec_fn[
            acc_type, type, unrolled_simd_width
        ](acc_unrolled_simd, val_simd)

    var acc_simd = SIMD[acc_type, simd_width].splat(init)
    for i in range(unrolled_vector_end, vector_end, simd_width):
        let val_simd = input_gen_fn[type, simd_width](i)
        dst.simd_store(i, val_simd)
        acc_simd = reduce_vec_to_vec_fn[acc_type, type, simd_width](
            acc_simd, val_simd
        )

    var acc = reduce_vec_to_scalar_fn[acc_type, unrolled_simd_width](
        acc_unrolled_simd
    )
    acc = reduce_vec_to_vec_fn[acc_type, acc_type, 1](
        acc, reduce_vec_to_scalar_fn[acc_type, simd_width](acc_simd)
    )
    for i in range(vector_end, length):
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
    reduce_fn: fn[acc_type: DType, type: DType, width: Int] (
        SIMD[acc_type, width], SIMD[type, width]
    ) capturing -> SIMD[acc_type, width],
](src: Buffer[size, type], init: SIMD[acc_type, 1]) -> SIMD[acc_type, 1]:
    """Computes a custom reduction of buffer elements.

    Parameters:
        simd_width: The vector width for the computation.
        size: The buffer size.
        type: The buffer elements dtype.
        acc_type: The dtype of the reduction accumulator.
        reduce_fn: The lambda implementing the reduction.

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

    var out: SIMD[type, 1] = 0

    @always_inline
    @parameter
    fn output_fn[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        out = rebind[SIMD[type, 1]](value)

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        type: DType, width: Int
    ](acc: SIMD[type, width], val: SIMD[type, width]) -> SIMD[type, width]:
        return reduce_fn(acc, val)

    let shape = Index(len(src))
    _reduce_generator[
        type,
        1,
        True,
        input_fn,
        output_fn,
        reduce_fn_wrapper,
    ](shape, rebind[SIMD[type, 1]](init), 0, OutputChainPtr())
    return rebind[SIMD[acc_type, 1]](out)


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

    let length = len(src)
    let unrolled_vector_end = align_down(length, unrolled_simd_width)
    let vector_end = align_down(length, simd_width)
    var curr = init
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        curr = reduce_fn[type, unrolled_simd_width](
            src.simd_load[unrolled_simd_width](i)
        )
        if not continue_fn(curr):
            return curr

    for i in range(unrolled_vector_end, vector_end, simd_width):
        curr = reduce_fn[type, simd_width](src.simd_load[simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(vector_end, length):
        curr = reduce_fn[type, 1](src[i])
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
                    simd_width,
                    sz,
                    type,
                    acc_type,
                    map_fn,
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
        constrained[unroll_factor > 0, "unroll_factor must be > 0"]()
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
        src.data, StaticIntTuple[3](h_dynamic, w_dynamic, c_dynamic)
    )
    let output_3d = NDBuffer[2, output_3d_shape, acc_type](
        dst.data, StaticIntTuple[2](h_dynamic, c_dynamic)
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
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple[rank],
    init_value: SIMD[type, 1],
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    constrained[target == "cuda", "only valid on GPUs"]()

    let reduce_dim_normalized = (
        rank + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    if reduce_dim_normalized != rank - 1:
        return out_chain.mark_error(
            "GPU reduction currently limited to inner axis."
        )

    try:
        reduce_launch[
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](shape, reduce_dim_normalized, init_value, out_chain.get_cuda_stream())
    except e:
        out_chain.mark_error(e)


@always_inline
@adaptive
fn _reduce_generator[
    type: DType,
    rank: Int,
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
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple[rank],
    init_value: SIMD[type, 1],
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.
    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        type: The element type we are reducing.
        rank: The rank of the tensor.
        single_thread_blocking_override: if set will run immediately
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        target: The target to run on.

    Args:
        shape: The shape of the tensor we are reducing.
        init_value: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The our chain to attach results to.
    """
    constrained[target == "cpu", "only valid on CPUs"]()
    let reduce_dim_normalized = (
        rank + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    @parameter
    if rank == 1:
        _reduce_along_inner_dimension[
            type,
            rank,
            single_thread_blocking_override,
            input_0_fn,
            output_0_fn,
            reduce_function,
        ](shape, init_value, reduce_dim_normalized, out_chain)
    else:
        if rank - 1 == reduce_dim_normalized:
            _reduce_along_inner_dimension[
                type,
                rank,
                single_thread_blocking_override,
                input_0_fn,
                output_0_fn,
                reduce_function,
            ](shape, init_value, reduce_dim_normalized, out_chain)
        else:
            _reduce_along_outer_dimension[
                type,
                rank,
                single_thread_blocking_override,
                input_0_fn,
                output_0_fn,
                reduce_function,
            ](shape, init_value, reduce_dim_normalized, out_chain)


@always_inline
fn _reduce_along_inner_dimension[
    type: DType,
    rank: Int,
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
    shape: StaticIntTuple[rank],
    init_value: SIMD[type, 1],
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.
    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        type: The element type we are reducing.
        rank: The rank of the tensor.
        single_thread_blocking_override: if set will run immediately
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
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

    alias unroll_factor = 8
    alias simd_width = simdwidthof[type]()
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

        # Iterate over the non reduced dimensions.
        for flat_index in range(start_row, end_row):
            # In normal elementwise get_nd_indices skips the last dimension as it is the dimension being iterated over. In our case we don't know this yet so we do have to calculate the extra one.
            var indices = _get_nd_indices_from_flat_index[rank](
                flat_index, shape, reduce_dim
            )

            var acc_unrolled_simd = SIMD[type, unrolled_simd_width].splat(
                init_value
            )

            @always_inline
            @parameter
            fn reduce_helper_fn[
                width: Int
            ](start: Int, finish: Int, init: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                var acc = init
                for idx in range(start, finish, width):
                    indices[reduce_dim] = idx
                    let load_value = input_0_fn[type, width, rank](indices)
                    acc = reduce_function[type, width](load_value, acc)
                return acc

            # Loop over unroll_factor*simd_width chunks.
            acc_unrolled_simd = reduce_helper_fn(
                0, unrolled_simd_compatible_size, acc_unrolled_simd
            )

            # Loop over remaining simd_width chunks.
            var acc_simd = acc_unrolled_simd.reduce[
                reduce_function, simd_width
            ]()
            acc_simd = reduce_helper_fn(
                unrolled_simd_compatible_size, simd_compatible_size, acc_simd
            )

            var acc_scalar = acc_simd.reduce[reduce_function]()
            # Loop over remaining scalar values.
            acc_scalar = reduce_helper_fn(
                simd_compatible_size, reduce_dim_size, acc_scalar
            )

            # Store the result back to the output.
            indices[reduce_dim] = 0
            output_0_fn[type, 1, rank](indices, acc_scalar)

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
        async_parallelize[reduce_rows](out_chain, num_workers)


@always_inline
fn _reduce_along_outer_dimension[
    type: DType,
    rank: Int,
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
    shape: StaticIntTuple[rank],
    init_value: SIMD[type, 1],
    reduce_dim: Int,
    out_chain: OutputChainPtr,
):
    """Reduce the given tensor using the given reduction function.
    All free vars in func must be "async safe", see async_parallelize.

    Parameters:
        type: The element type we are reducing.
        rank: The rank of the tensor.
        single_thread_blocking_override: if set will run immediately
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
    Args:
        shape: The shape of the tensor we are reducing
        init_value: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        out_chain: The chain to attach results to.
    """
    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.

    alias simd_width = simdwidthof[type]()

    let total_size: Int = shape.flattened_length()
    if total_size == 0:

        @parameter
        if not single_thread_blocking_override:
            out_chain.mark_ready()
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

    @always_inline
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
                var acc_simd = SIMD[type, simd_width].splat(init_value)
                let reduce_vector_idx = slice_idx * inner_dim + inner_dim_idx
                var indices = _get_nd_indices_from_flat_index[rank](
                    reduce_vector_idx, shape, reduce_dim
                )
                for reduce_dim_idx in range(0, reduce_dim_size):
                    indices[reduce_dim] = reduce_dim_idx
                    let load_value = input_0_fn[type, simd_width, rank](indices)
                    acc_simd = reduce_function[type, simd_width](
                        load_value, acc_simd
                    )
                # Store the result back to the output.
                indices[reduce_dim] = 0
                output_0_fn[type, simd_width, rank](indices, acc_simd)

            vectorize[simd_width, reduce_chunk](inner_dim)

    @parameter
    if single_thread_blocking_override:
        reduce_slices(0)
    else:
        async_parallelize[reduce_slices](out_chain, num_workers)


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
    return reduce[simdwidthof[type](), size, type, type, _simd_max_elementwise](
        src, src[0]
    )


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
    return reduce[simdwidthof[type](), size, type, type, _simd_min_elementwise](
        src, src[0]
    )


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
    return reduce[simdwidthof[type](), size, type, type, _simd_sum_elementwise](
        src, 0
    )


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


@adaptive
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

    debug_assert(len(src) != 0, "input must not be empty")

    let total = sum(src)
    let buffer_len = len(src)

    @parameter
    if type.is_integral():
        return total // buffer_len
    else:
        return total / buffer_len


@adaptive
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

        vectorize[simd_width, normalize_integral](len(dst_1d))
    else:
        let n_recip = SIMD[type, 1](1) / n

        @always_inline
        @parameter
        fn normalize_floating[simd_width: Int](idx: Int):
            let elem = dst_1d.simd_load[simd_width](idx)
            let to_store = elem * n_recip
            dst_1d.simd_store[simd_width](idx, to_store)

        vectorize[simd_width, normalize_floating](len(dst_1d))


@always_inline
@adaptive
fn mean[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    /,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    reduce_dim: Int,
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    """Computes the mean across the input and output shape.

    This performs the mean computation on the domain specified by `input_shape`,
    storing the results using the`input_0_fn`. The results' domain is
    `output_shape` which are stored using the `output_0_fn`.

    Parameters:
        type: The type of the input and output.
        rank: The rank of the domain.
        single_thread_blocking_override: Whether the operation is performed async.
        input_fn: The function to load the input.
        output_fn: The function to store the output.
        target: The target architecture.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the mean on.
        output_shape: The output shape.
        out_chain: The output chain to use.
    """
    out_chain.trace[TraceLevel.OP]("mogg.mean")

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
            type,
            rank,
            single_thread_blocking_override,
            input_fn_wrapper,
            wrapped_output_mul,
            reduce_impl,
            target,
        ](input_shape, 0, reduce_dim, out_chain)

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
            type,
            rank,
            single_thread_blocking_override,
            input_fn_wrapper,
            wrapped_output_div,
            reduce_impl,
        ](input_shape, 0, reduce_dim, out_chain)


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
    debug_assert(len(src) > 1, "input length must be greater than 1")

    @always_inline
    @parameter
    fn input_fn[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        let mean_simd = SIMD[type, width].splat(mean_value).cast[_type]()
        let x = src.simd_load[width](idx[0])
        let diff = x.cast[_type]() - mean_simd
        return rebind[SIMD[_type, width]](diff * diff)

    var out: SIMD[type, 1] = 0

    @always_inline
    @parameter
    fn output_fn[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        out = rebind[SIMD[type, 1]](value)

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        type: DType, width: Int
    ](acc: SIMD[type, width], val: SIMD[type, width]) -> SIMD[type, width]:
        return acc + val

    let shape = StaticIntTuple[1](len(src))
    let init = SIMD[type, 1](0)
    _reduce_generator[
        type,
        1,
        True,
        input_fn,
        output_fn,
        reduce_fn_wrapper,
    ](shape, rebind[SIMD[type, 1]](init), 0, OutputChainPtr())
    return out / (len(src) - correction)


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
    rank: Int,
    is_max: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    """
    Finds the indices of the maximum/minimum element along the specified axis.

    Parameters:
        type: Type of the input tensor.
        out_type: Type of the output tensor.
        rank: The rank of the input / output.
        is_max: If True compute then compute argmax, otherwise compute the
                argmin.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        out_chain: The chain to attach results to.
    """
    alias simd_width = simdwidthof[type]()

    var canonical_axis = axis
    if canonical_axis < 0:
        canonical_axis += rank
    if not 0 <= canonical_axis < rank:
        out_chain.mark_error("axis must be between [0, <input rank>)")
        return

    # TODO: Generalize to mid axis.
    if canonical_axis != rank - 1:
        out_chain.mark_error("axis other than innermost not supported yet")
        return

    @unroll
    for subaxis in range(rank):
        if subaxis == canonical_axis:
            if output.dim(subaxis) != 1:
                out_chain.mark_error("expected axis to have size 1 in output")
                return
        elif input.dim(subaxis) != output.dim(subaxis):
            out_chain.mark_error(
                "input and output dims must match aside from 'axis'"
            )
            return

    let axis_size = input.dim(canonical_axis)
    let input_start_ptr = input.data
    let output_start_ptr = output.data
    let input_stride: Int
    let output_stride: Int

    @parameter
    if rank == 1:
        input_stride = input.num_elements()
        output_stride = output.num_elements()
    else:
        input_stride = input.dynamic_stride[canonical_axis - 1]
        output_stride = output.dynamic_stride[canonical_axis - 1]

    @always_inline
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

        var curr_input_offset = 0
        var curr_output_offset = 0
        while curr_input_offset < input.num_elements():
            let input_dim_ptr = input_start_ptr.offset(curr_input_offset)
            let output_dim_ptr = output_start_ptr.offset(curr_output_offset)
            var global_val: SIMD[type, 1]

            @parameter
            if is_max:
                global_val = min_or_neginf[type]()
            else:
                global_val = max_or_inf[type]()

            var global_values: SIMD[type, simd_width]
            if axis_size < simd_width:
                global_values = global_val
            else:
                global_values = input_dim_ptr.simd_load[simd_width]()

            var indices = iota[out_type, simd_width]()
            var global_indices = indices

            let last_simd_index = (axis_size // simd_width) * simd_width
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

            var matching = global_values == global_val
            var idx = Int__(global_indices[_index_of_first_one(matching)])

            # Check trailing indices.
            for j in range(last_simd_index, axis_size, 1):
                let elem = input_dim_ptr.load(j)
                if cmp(global_val, elem):
                    global_val = elem
                    idx = j

            output_dim_ptr.store(idx)
            curr_output_offset += output_stride
            curr_input_offset += input_stride

    # TODO: Shard by dim 0.
    async_parallelize[task_func](out_chain, 1)
    # TODO: Remove after #26325
    out_chain.wait()


# ===----------------------------------------------------------------------===#
# argmax
# ===----------------------------------------------------------------------===#


@always_inline
fn argmax[
    type: DType,
    out_type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    """
    Finds the indices of the maximum element along the specified axis.

    Parameters:
        type: Type of the input tensor.
        out_type: Type of the output tensor.
        rank: The rank of the input / output.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        out_chain: The chain to attach results to.
    """

    _argn[type, out_type, rank, True](input, axis, output, out_chain)


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

    argmax(input, Int__(axis_buf[0]), output, out_chain)


# ===----------------------------------------------------------------------===#
# argmin
# ===----------------------------------------------------------------------===#


@always_inline
fn argmin[
    type: DType,
    out_type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    """
    Finds the indices of the maximum element along the specified axis.

    Parameters:
        type: Type of the input tensor.
        out_type: Type of the output tensor.
        rank: The rank of the input / output.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        out_chain: The chain to attach results to.
    """

    _argn[type, out_type, rank, False](input, axis, output, out_chain)


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

    argmin(input, Int__(axis_buf[0]), output, out_chain)


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
    var axis = Int__(axis_buf[0])
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
fn _cumsum[
    size: Int, type: DType
](dst: Buffer[size, type], src: Buffer[size, type]):
    dst[0] = src[0]
    for i in range(1, size):
        dst[i] = src[i] + dst[i - 1]


@always_inline
fn cumsum[
    size: Int, type: DType
](dst: Buffer[size, type], src: Buffer[size, type]):
    """Computes the cumulative sum of all elements in a buffer.
       dst[i] = src[i] + src[i-1] + ... + src[0].

    Parameters:
        size: The size of the input and output buffers.
        type: The type of the elements of the input and output buffers.

    Args:
        dst: The buffer that stores the result of cumulative sum operation.
        src: The buffer of elements for which the cumulative sum is computed.
    """

    debug_assert(len(src) != 0, "Input must not be empty")
    debug_assert(len(dst) != 0, "Output must not be empty")

    alias simd_width = simdwidthof[type]()

    # For length less than simd_width do serial cumulative sum.
    # Similarly, for the case when simd_width == 2 serial should be faster.
    if size < simd_width or simd_width == 2:
        return _cumsum[size, type](dst, src)

    # Stores the offset (i.e., last value of previous simd_width-elements chunk,
    # replicated across all simd lanes, to be added to all elements of next
    # chunk.
    var offset = SIMD[type, simd_width]()

    # Divide the buffer size to div_size chunks of simd_width elements,
    # to calculate using SIMD and do remaining (tail) serially.
    let div_size = (size // simd_width) * simd_width

    # Number of inner-loop iterations (for shift previous result and add).
    alias rep = _static_log2[simd_width]()

    for i in range(0, div_size, simd_width):
        var x_simd = src.simd_load[simd_width](i)
        var y_simd = SIMD[type, simd_width]()

        @parameter
        fn loop_body[idx: Int]():
            alias a = 2**idx
            y_simd = x_simd.shift_right[a]()
            x_simd = x_simd + y_simd

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
        var x_simd = dst.simd_load[simd_width](i)
        x_simd += offset
        dst.simd_store(i, x_simd)
        offset = offset.splat(x_simd[simd_width - 1])

    # Handles the tail, i.e., num of elements at the end that don't
    # fit within a simd_width-elements vector.
    for i in range(div_size, size):
        dst[i] = dst[i - 1] + src[i]
