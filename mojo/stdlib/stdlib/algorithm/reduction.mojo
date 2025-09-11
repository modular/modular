# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from algorithm import map_reduce
```
"""

from collections import OptionalReg
from math import align_down, ceildiv
from sys.info import simd_width_of, size_of, align_of

from algorithm import sync_parallelize, vectorize
from algorithm.functional import _get_num_workers
from bit import log2_floor
from buffer import NDBuffer
from buffer.buffer import prod_dims
from buffer.dimlist import Dim, DimList
from builtin.math import max as _max
from builtin.math import min as _min
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_valid_target
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg, get_safe_task_id

from utils.index import Index, IndexList, StaticTuple

from ._gpu.reduction import reduce_launch


# ===-----------------------------------------------------------------------===#
# ND indexing helper
# ===-----------------------------------------------------------------------===#


@always_inline
fn _get_nd_indices_from_flat_index(
    flat_index: Int, shape: IndexList, skip_dim: Int, out res: __type_of(shape)
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
    @parameter
    if shape.size == 2:
        if skip_dim == 1:
            return {flat_index, 0}
        else:
            return {0, flat_index}

    res = {}
    var curr_index = flat_index

    @parameter
    for i in reversed(range(shape.size)):
        # There is one dimension we skip, this represents the inner loop that
        # is being traversed.
        if i == skip_dim:
            res[i] = 0
        else:
            res[i] = curr_index._positive_rem(shape[i])
            curr_index = curr_index._positive_div(shape[i])


# ===-----------------------------------------------------------------------===#
# reduce
# ===-----------------------------------------------------------------------===#


@always_inline
@parameter
fn map_reduce[
    simd_width: Int,
    size: Dim,
    dtype: DType,
    acc_type: DType,
    origins_gen: OriginSet,
    input_gen_fn: fn[dtype: DType, width: Int] (Int) capturing [
        origins_gen
    ] -> SIMD[dtype, width],
    origins_vec: OriginSet,
    reduce_vec_to_vec_fn: fn[acc_type: DType, dtype: DType, width: Int] (
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing [origins_vec] -> SIMD[acc_type, width],
    reduce_vec_to_scalar_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width]
    ) -> Scalar[dtype],
](dst: NDBuffer[mut=True, dtype, 1, _, size], init: Scalar[acc_type]) -> Scalar[
    acc_type
]:
    """Stores the result of calling input_gen_fn in dst and simultaneously
    reduce the result using a custom reduction function.

    Parameters:
        simd_width: The vector width for the computation.
        size: The buffer size.
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
    fn output_fn[
        _dtype: DType, width: Int, rank: Int
    ](idx: Int, val: SIMD[_dtype, width]):
        dst.store(idx, rebind[SIMD[dtype, width]](val))

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
fn map_reduce[
    simd_width: Int,
    dtype: DType,
    acc_type: DType,
    origins_gen: OriginSet,
    input_gen_fn: fn[dtype: DType, width: Int] (Int) capturing [
        origins_gen
    ] -> SIMD[dtype, width],
    origins_vec: OriginSet,
    reduce_vec_to_vec_fn: fn[acc_type: DType, dtype: DType, width: Int] (
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing [origins_vec] -> SIMD[acc_type, width],
    reduce_vec_to_scalar_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width]
    ) -> Scalar[dtype],
    output_fn: fn[dtype_: DType, width: Int, alignment: Int] (
        idx: Int, val: SIMD[dtype_, width]
    ) capturing -> None,
](length: Int, init: Scalar[acc_type]) -> Scalar[acc_type]:
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor
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


@always_inline
@parameter
fn reduce[
    reduce_fn: fn[acc_type: DType, dtype: DType, width: Int] (
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[acc_type, width]
](src: NDBuffer[rank=1], init: Scalar) raises -> Scalar[init.dtype]:
    """Computes a custom reduction of buffer elements.

    Parameters:
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
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return src.load[width=width](idx[0])._refine[_dtype]()

    var out: Scalar[init.dtype] = 0

    @always_inline
    @parameter
    fn output_fn[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        out = value._refine[init.dtype, 1]()

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        dtype: DType, width: Int
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return reduce_fn(acc, val)

    var shape = Index(len(src))

    _reduce_generator[
        input_fn,
        output_fn,
        reduce_fn_wrapper,
        single_thread_blocking_override=True,
    ](shape, init=init, reduce_dim=0)

    return out


@always_inline
@parameter
fn reduce_boolean[
    reduce_fn: fn[dtype: DType, width: Int] (SIMD[dtype, width]) capturing [
        _
    ] -> Bool,
    continue_fn: fn (Bool) capturing [_] -> Bool,
](src: NDBuffer[rank=1], init: Bool) -> Bool:
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

    Args:
        src: The input buffer.
        init: The initial value to use.

    Returns:
        The computed reduction value.
    """
    alias simd_width = simd_width_of[src.type]()
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor

    var length = len(src)
    var unrolled_vector_end = align_down(length, unrolled_simd_width)
    var vector_end = align_down(length, simd_width)
    var curr = init
    for i in range(0, unrolled_vector_end, unrolled_simd_width):
        curr = reduce_fn(src.load[width=unrolled_simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(unrolled_vector_end, vector_end, simd_width):
        curr = reduce_fn(src.load[width=simd_width](i))
        if not continue_fn(curr):
            return curr

    for i in range(vector_end, length):
        curr = reduce_fn(src[i])
        if not continue_fn(curr):
            return curr
    return curr


@parameter
fn _reduce_3D[
    map_fn: fn[acc_type: DType, dtype: DType, width: Int] (
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[acc_type, width],
    reduce_fn: fn[dtype: DType, width: Int] (SIMD[dtype, width]) -> Scalar[
        dtype
    ],
](src: NDBuffer, dst: NDBuffer[mut=True, **_], init: Scalar[dst.type]) raises:
    """Performs a reduction across axis 1 of a 3D input buffer."""

    alias simd_width = simd_width_of[dst.type]()

    var h = src.dim[0]()
    var w = src.dim[1]()
    var c = src.dim[2]()

    # If c is 1, we are reducing across the innermost axis, and we can launch H
    # reductions that each reduce W elements of a contiguous buffer.
    if c == 1:

        @__copy_capture(h, w)
        @parameter
        fn reduce_inner_axis() raises:
            alias sz = src.shape.at[1]()
            # TODO: parallelize
            for i in range(h):
                var offset = src._offset(IndexList[src.rank](i, 0, 0))
                var input = NDBuffer[
                    src.type,
                    1,
                    offset.origin,
                    sz,
                    address_space = src.address_space,
                ](offset, w)
                var val = reduce[map_fn](input, init)
                dst[IndexList[dst.rank](i, 0)] = val

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

    alias unroll_factor = get_unroll_factor[simd_width, size_of[dst.type]()]()
    alias usimd_width = unroll_factor * simd_width
    for i in range(h):

        @always_inline
        @__copy_capture(w)
        @parameter
        fn reduce_w_chunked[simd_width: Int](idx: Int):
            var accum = SIMD[init.dtype, simd_width](init)
            for j in range(w):
                var chunk = src.load[width=simd_width](
                    IndexList[src.rank](i, j, idx)
                )
                accum = map_fn(accum, chunk)
            dst.store(IndexList[dst.rank](i, idx), accum)

        vectorize[reduce_w_chunked, usimd_width](c)


@parameter
fn reduce[
    map_fn: fn[acc_type: DType, dtype: DType, width: Int] (
        SIMD[acc_type, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[acc_type, width],
    reduce_fn: fn[dtype: DType, width: Int] (SIMD[dtype, width]) -> Scalar[
        dtype
    ],
    reduce_axis: Int,
](src: NDBuffer, dst: NDBuffer[mut=True, **_], init: Scalar[dst.type]) raises:
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

    Args:
        src: The input buffer.
        dst: The output buffer.
        init: The initial value to use in accumulator.
    """

    var h_dynamic = prod_dims[0, reduce_axis](src)
    var w_dynamic = src.dim[reduce_axis]()
    var c_dynamic = prod_dims[reduce_axis + 1, src.rank](src)

    alias h_static = src.shape.product[reduce_axis]()
    alias w_static = src.shape.at[reduce_axis]()
    alias c_static = src.shape.product[reduce_axis + 1, src.rank]()

    alias input_3d_shape = DimList(h_static, w_static, c_static)
    alias output_2d_shape = DimList(h_static, c_static)

    var input_3d = NDBuffer[
        src.type,
        3,
        shape=input_3d_shape,
        address_space = src.address_space,
    ](src.data, Index(h_dynamic, w_dynamic, c_dynamic))
    var output_2d = NDBuffer[
        dst.type, 2, _, output_2d_shape, address_space = dst.address_space
    ](
        dst.data,
        Index(h_dynamic, c_dynamic),
    )

    _reduce_3D[map_fn, reduce_fn](input_3d, output_2d, init)


# ===-----------------------------------------------------------------------===#
# MOGG reduce functions.
# These take lambdas and don't assume contiguous inputs so can compose
# with mogg kernels / fusion.
# ===-----------------------------------------------------------------------===#


@always_inline
fn _reduce_generator[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_function: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    shape: IndexList[_, element_type = DType.int64],
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        context: The pointer to DeviceContext.
    """
    constrained[is_valid_target[target](), "unsupported target"]()

    @parameter
    if is_cpu[target]():
        _reduce_generator_cpu[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
            single_thread_blocking_override,
        ](shape, init, reduce_dim)
    else:
        _reduce_generator_gpu[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
            single_thread_blocking_override,
        ](shape, init, reduce_dim, context.get_device_context())


@always_inline
fn _reduce_generator_gpu[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_function: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
](
    shape: IndexList[_, element_type = DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
    ctx: DeviceContext,
) raises:
    """Reduce the given tensor using the given reduction function on GPU. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        ctx: The pointer to DeviceContext.
    """

    var reduce_dim_normalized = (
        len(shape) + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    reduce_launch[
        num_reductions,
        input_0_fn,
        output_0_fn,
        reduce_function,
        shape.size,
        init_type,
    ](shape, reduce_dim_normalized, init, ctx)


@always_inline
fn _reduce_generator_cpu[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_function: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
](
    shape: IndexList[_, element_type = DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
) raises:
    """Reduce the given tensor using the given reduction function on CPU. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
    """

    alias rank = shape.size

    var reduce_dim_normalized = (
        rank + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    @parameter
    if shape.size == 1:
        _reduce_along_inner_dimension[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
            single_thread_blocking_override=single_thread_blocking_override,
        ](shape, init, reduce_dim_normalized)
    else:
        if rank - 1 == reduce_dim_normalized:
            _reduce_along_inner_dimension[
                num_reductions,
                init_type,
                input_0_fn,
                output_0_fn,
                reduce_function,
                single_thread_blocking_override=single_thread_blocking_override,
            ](shape, init, reduce_dim_normalized)
        else:
            _reduce_along_outer_dimension[
                num_reductions,
                init_type,
                input_0_fn,
                output_0_fn,
                reduce_function,
                single_thread_blocking_override=single_thread_blocking_override,
            ](shape, init, reduce_dim_normalized)


@always_inline
fn _reduce_generator_wrapper[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    reduce_function: fn[width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    shape: IndexList[_, element_type = DType.int64],
    init: Scalar,
    reduce_dim: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    @always_inline
    @parameter
    fn input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        _dtype: DType,
        width: Int,
        rank: Int,
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    fn reduce_fn[
        ty: DType, width: Int
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
        single_thread_blocking_override=single_thread_blocking_override,
    ](shape, init, reduce_dim, context)


@always_inline
fn _reduce_generator[
    input_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    reduce_function: fn[ty: DType, width: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    shape: IndexList[_, element_type = DType.int64],
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        context: The pointer to DeviceContext.
    """

    alias num_reductions = 1

    @always_inline
    @parameter
    fn output_fn_wrapper[
        dtype: DType, width: Int, rank: Int
    ](
        indices: IndexList[rank],
        val: StaticTuple[SIMD[dtype, width], num_reductions],
    ):
        output_0_fn[dtype, width, rank](indices, val[0])

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](val: SIMD[dtype, width], acc: SIMD[dtype, width]) -> SIMD[dtype, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction index"]()
        return reduce_function[dtype, width](val, acc)

    var init_wrapped = StaticTuple[Scalar[init.dtype], num_reductions](init)
    return _reduce_generator[
        num_reductions,
        init.dtype,
        input_0_fn,
        output_fn_wrapper,
        reduce_fn_wrapper,
        single_thread_blocking_override,
        target,
    ](shape, init_wrapped, reduce_dim, context)


fn _reduce_along_inner_dimension[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_function: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
](
    shape: IndexList[_, element_type = DType.int64],
    init_value: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
):
    var total_size: Int = shape.flattened_length()
    if total_size == 0:
        return

    var reduce_dim_size = shape[reduce_dim]

    var parallelism_size: Int = total_size // reduce_dim_size

    var num_workers: Int

    @parameter
    if single_thread_blocking_override:
        num_workers = 1
    else:
        num_workers = _get_num_workers(total_size)

    var chunk_size = ceildiv(parallelism_size, num_workers)

    alias unroll_factor = 8
    alias simd_width = simd_width_of[init_type]()
    alias unrolled_simd_width = simd_width * unroll_factor

    var unrolled_simd_compatible_size = align_down(
        reduce_dim_size, unrolled_simd_width
    )
    var simd_compatible_size = align_down(reduce_dim_size, simd_width)

    @always_inline
    @parameter
    fn simd_reduce_helper_fn[
        in_width: Int,
        out_width: Int,
    ](
        in_acc_tup: StaticTuple[SIMD[init_type, in_width], num_reductions]
    ) -> StaticTuple[SIMD[init_type, out_width], num_reductions]:
        var out_acc_tup = StaticTuple[
            SIMD[init_type, out_width], num_reductions
        ]()

        @parameter
        for i in range(num_reductions):
            out_acc_tup[i] = in_acc_tup[i].reduce[
                reduce_function[init_type, reduction_idx=i], out_width
            ]()

        return out_acc_tup

    @always_inline
    @parameter
    fn reduce_rows_unrolled(start_row: Int, end_row: Int):
        # Iterate over the non reduced dimensions.
        for flat_index in range(start_row, end_row):
            # In normal elementwise get_nd_indices skips the last dimension as
            # it is the dimension being iterated over. In our case we don't know
            # this yet so we do have to calculate the extra one.
            var indices = _get_nd_indices_from_flat_index(
                flat_index, shape, reduce_dim
            )

            @always_inline
            @parameter
            fn unrolled_reduce_helper_fn[
                width: Int,
            ](
                start: Int,
                finish: Int,
                init: StaticTuple[SIMD[init_type, width], num_reductions],
            ) -> StaticTuple[SIMD[init_type, width], num_reductions]:
                var acc = init
                for idx in range(start, finish, width):
                    indices[reduce_dim] = idx
                    var load_value = input_0_fn[init_type, width](indices)

                    @parameter
                    for i in range(num_reductions):
                        acc[i] = reduce_function[init_type, width, i](
                            load_value, acc[i]
                        )

                return acc

            # initialize our accumulator
            var acc_unrolled_simd_tup = StaticTuple[
                SIMD[
                    init_type,
                    unrolled_simd_width,
                ],
                num_reductions,
            ]()

            @parameter
            for i in range(num_reductions):
                acc_unrolled_simd_tup[i] = SIMD[
                    init_type,
                    unrolled_simd_width,
                ](init_value[i])

            # Loop over unroll_factor*simd_width chunks.
            acc_unrolled_simd_tup = unrolled_reduce_helper_fn[
                unrolled_simd_width
            ](0, unrolled_simd_compatible_size, acc_unrolled_simd_tup)

            # Reduce to simd_width
            var acc_simd_tup = simd_reduce_helper_fn[
                unrolled_simd_width,
                simd_width,
            ](acc_unrolled_simd_tup)

            # Loop over tail simd_width chunks
            acc_simd_tup = unrolled_reduce_helper_fn[simd_width](
                unrolled_simd_compatible_size,
                simd_compatible_size,
                acc_simd_tup,
            )

            # Reduce to scalars
            var acc_scalar_tup = simd_reduce_helper_fn[
                simd_width,
                1,
            ](acc_simd_tup)

            # Loop over tail scalars
            acc_scalar_tup = unrolled_reduce_helper_fn[1](
                simd_compatible_size, reduce_dim_size, acc_scalar_tup
            )

            # Store the result back to the output.
            indices[reduce_dim] = 0
            output_0_fn(indices, acc_scalar_tup)

    @always_inline
    @parameter
    fn reduce_rows(i: Int):
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        var length = end_parallel_offset - start_parallel_offset
        if length <= 0:
            return

        reduce_rows_unrolled(start_parallel_offset, end_parallel_offset)

    @parameter
    if single_thread_blocking_override:
        reduce_rows_unrolled(0, parallelism_size)
    else:
        sync_parallelize[reduce_rows](num_workers)
    _ = reduce_dim_size
    _ = parallelism_size
    _ = chunk_size
    _ = unrolled_simd_compatible_size
    _ = simd_compatible_size


fn _reduce_along_outer_dimension[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_0_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_function: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
](
    shape: IndexList[_, element_type = DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
):
    """Reduce the given tensor using the given reduction function. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to execute in parallel.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        shape: The shape of the tensor we are reducing
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
    """
    alias rank = shape.size
    alias dtype = init.element_type

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    alias simd_width = simd_width_of[dtype]()

    var total_size: Int = shape.flattened_length()
    if total_size == 0:
        return

    var reduce_dim_size = shape[reduce_dim]
    var inner_dim = shape[rank - 1]

    # parallelize across slices of the input, where a slice is [reduce_dim, inner_dim]
    # the slice is composed of [reduce_dim, simd_width] chunks
    # these chunks are reduced simultaneously across the reduce_dim using simd instructions
    # and accumulation
    var parallelism_size: Int = total_size // (reduce_dim_size * inner_dim)

    var num_workers: Int

    @parameter
    if single_thread_blocking_override:
        num_workers = 1
    else:
        num_workers = _get_num_workers(total_size)

    var chunk_size = ceildiv(parallelism_size, num_workers)

    @parameter
    fn reduce_slices(i: Int):
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        var length = end_parallel_offset - start_parallel_offset

        if length <= 0:
            return

        for slice_idx in range(start_parallel_offset, end_parallel_offset):

            @always_inline
            @parameter
            fn reduce_chunk[simd_width: Int](inner_dim_idx: Int):
                var acc_simd_tup = StaticTuple[
                    SIMD[init_type, simd_width], num_reductions
                ]()

                @parameter
                for i in range(num_reductions):
                    acc_simd_tup[i] = SIMD[init_type, simd_width](init[i])

                var reduce_vector_idx = slice_idx * inner_dim + inner_dim_idx
                var indices = _get_nd_indices_from_flat_index(
                    reduce_vector_idx, shape, reduce_dim
                )
                for reduce_dim_idx in range(reduce_dim_size):
                    indices[reduce_dim] = reduce_dim_idx
                    var load_value = input_0_fn[
                        init_type, simd_width, shape.size
                    ](indices)

                    @parameter
                    for i in range(num_reductions):
                        acc_simd_tup[i] = reduce_function[
                            init_type, simd_width, i
                        ](load_value, acc_simd_tup[i])

                indices[reduce_dim] = 0
                output_0_fn[init_type, simd_width, indices.size](
                    indices, acc_simd_tup
                )

            vectorize[reduce_chunk, simd_width](inner_dim)

    @parameter
    if single_thread_blocking_override:
        reduce_slices(0)
    else:
        sync_parallelize[reduce_slices](num_workers)


# ===-----------------------------------------------------------------------===#
# max
# ===-----------------------------------------------------------------------===#


@always_inline
fn _simd_max[
    dtype: DType,
    simd_width: Int,
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


@always_inline
@parameter
fn _simd_max_elementwise[
    acc_type: DType,
    dtype: DType,
    simd_width: Int,
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise max of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return _max(x, y.cast[acc_type]())


fn max(src: NDBuffer[rank=1]) raises -> Scalar[src.type]:
    """Computes the max element in a buffer.

    Args:
        src: The buffer.

    Returns:
        The maximum of the buffer elements.
    """
    return reduce[_simd_max_elementwise](src, Scalar[src.type].MIN)


fn max[
    reduce_axis: Int
](src: NDBuffer, dst: NDBuffer[mut=True, src.type, src.rank, _, _]) raises:
    """Computes the max across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_max_elementwise, _simd_max, reduce_axis](
        src, dst, Scalar[src.type].MIN
    )


@always_inline
fn max[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the max on.
        context: The pointer to DeviceContext.
    """

    @always_inline
    @parameter
    fn input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return _max(v1, v2)

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](input_shape, Scalar[dtype].MIN, reduce_dim, context=context)


# ===-----------------------------------------------------------------------===#
# min
# ===-----------------------------------------------------------------------===#


@always_inline
fn _simd_min[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the min element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_min()


@always_inline
@parameter
fn _simd_min_elementwise[
    acc_type: DType, dtype: DType, simd_width: Int
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise min of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return _min(x, y.cast[acc_type]())


fn min(src: NDBuffer[rank=1]) raises -> Scalar[src.type]:
    """Computes the min element in a buffer.

    Args:
        src: The buffer.

    Returns:
        The minimum of the buffer elements.
    """
    return reduce[_simd_min_elementwise](src, Scalar[src.type].MAX)


fn min[
    reduce_axis: Int
](src: NDBuffer, dst: NDBuffer[mut=True, src.type, src.rank, _, _]) raises:
    """Computes the min across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_min_elementwise, _simd_min, reduce_axis](
        src, dst, Scalar[src.type].MAX
    )


@always_inline
fn min[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the min on.
        context: The pointer to DeviceContext.
    """

    @always_inline
    @parameter
    fn input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return _min(v1, v2)

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](input_shape, Scalar[dtype].MAX, reduce_dim, context=context)


# ===-----------------------------------------------------------------------===#
# sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn _simd_sum[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


@always_inline
@parameter
fn _simd_sum_elementwise[
    acc_type: DType, dtype: DType, simd_width: Int
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise sum of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x + y.cast[acc_type]()


fn sum(src: NDBuffer[rank=1]) raises -> Scalar[src.type]:
    """Computes the sum of buffer elements.

    Args:
        src: The buffer.

    Returns:
        The sum of the buffer elements.
    """

    @parameter
    @always_inline
    fn input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](src.load[width=width](idx))

    return sum[src.type, input_fn_1d](len(src))


fn sum[
    reduce_axis: Int
](src: NDBuffer, dst: NDBuffer[mut=True, src.type, src.rank, _, _]) raises:
    """Computes the sum across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_sum_elementwise, _simd_sum, reduce_axis=reduce_axis](
        src, dst, Scalar[src.type](0)
    )


@always_inline
fn sum[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the sum on.
        context: The pointer to DeviceContext.
    """

    @always_inline
    @parameter
    fn input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](input_shape, Scalar[dtype](0), reduce_dim, context=context)


fn sum[
    dtype: DType,
    input_fn_1d: fn[dtype_: DType, width: Int] (idx: Int) capturing -> SIMD[
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
    fn input_fn_nd[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn_1d[dtype, width](idx[0])._refine[_dtype]()

    var out: Scalar[dtype] = 0

    @always_inline
    @parameter
    fn output_fn[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        out = value._refine[dtype, 1]()

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        dtype: DType, width: Int
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return acc + val

    var shape = IndexList[1](length)

    _reduce_generator[
        input_fn_nd,
        output_fn,
        reduce_fn_wrapper,
        single_thread_blocking_override=True,
    ](
        shape,
        init=Scalar[dtype](0),
        reduce_dim=0,
    )

    return out


# ===-----------------------------------------------------------------------===#
# product
# ===-----------------------------------------------------------------------===#


@always_inline
fn _simd_product[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> Scalar[dtype]:
    """Computes the product of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_mul()


@always_inline
@parameter
fn _simd_product_elementwise[
    acc_type: DType, dtype: DType, simd_width: Int
](x: SIMD[acc_type, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Computes the elementwise product of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x * y.cast[acc_type]()


fn product(src: NDBuffer[rank=1]) raises -> Scalar[src.type]:
    """Computes the product of the buffer elements.

    Args:
        src: The buffer.

    Returns:
        The product of the buffer elements.
    """
    return reduce[_simd_product_elementwise](src, Scalar[src.type](1))


fn product[
    reduce_axis: Int
](src: NDBuffer, dst: NDBuffer[mut=True, src.type, src.rank, _, _]) raises:
    """Computes the product across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    return reduce[_simd_product_elementwise, _simd_product, reduce_axis](
        src, dst, Scalar[src.type](1)
    )


@always_inline
fn product[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the product on.
        context: The pointer to DeviceContext.
    """

    @always_inline
    @parameter
    fn input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return input_fn[width, rank](idx)._refine[_dtype]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        output_fn[width, rank](indices, value._refine[dtype]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 * v2

    _reduce_generator[
        input_fn_wrapper,
        output_fn_wrapper,
        reduce_impl,
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](input_shape, Scalar[dtype](1), reduce_dim, context=context)


# ===-----------------------------------------------------------------------===#
# mean
# ===-----------------------------------------------------------------------===#


fn mean(src: NDBuffer[rank=1]) raises -> Scalar[src.type]:
    """Computes the mean value of the elements in a buffer.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.
    """

    debug_assert(len(src) != 0, "input must not be empty")

    @parameter
    @always_inline
    fn input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](src.load[width=width](idx))

    return mean[src.type, input_fn_1d](len(src))


fn mean[
    reduce_axis: Int
](src: NDBuffer, dst: NDBuffer[mut=True, src.dtype, src.rank, _, _]) raises:
    """Computes the mean across reduce_axis of an NDBuffer.

    Parameters:
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.
    """
    alias simd_width = simd_width_of[dst.dtype]()
    sum[reduce_axis](src, dst)

    var n = src.dim[reduce_axis]()
    var dst_1d = dst.flatten()

    @parameter
    if dst.type.is_integral():

        @always_inline
        @__copy_capture(dst_1d, n)
        @parameter
        fn normalize_integral[simd_width: Int](idx: Int):
            var elem = dst_1d.load[width=simd_width](idx)
            var to_store = elem // n
            dst_1d.store(idx, to_store)

        vectorize[normalize_integral, simd_width](len(dst_1d))
    else:
        var n_recip = Scalar[dst.type](1) / n

        @always_inline
        @__copy_capture(dst_1d, n, n_recip)
        @parameter
        fn normalize_floating[simd_width: Int](idx: Int):
            var elem = dst_1d.load[width=simd_width](idx)
            var to_store = elem * n_recip
            dst_1d.store(idx, to_store)

        vectorize[normalize_floating, simd_width](len(dst_1d))


@always_inline
fn mean[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing [_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    output_shape: __type_of(input_shape),
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
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: The target to run on.

    Args:
        input_shape: The input shape.
        reduce_dim: The axis to perform the mean on.
        output_shape: The output shape.
        context: The pointer to DeviceContext.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return ";".join(
            trace_arg("input", input_shape, dtype),
            trace_arg("output", output_shape, dtype),
        )

    with Trace[TraceLevel.OP, target=target](
        "mean",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):

        @always_inline
        @parameter
        fn reduce_impl[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        @always_inline
        @parameter
        fn input_fn_wrapper[
            _dtype: DType, width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
            return input_fn[width, rank](idx)._refine[_dtype, width]()

        # For floats apply the reciprocal as a multiply.
        @parameter
        if dtype.is_floating_point():
            # Apply mean division before storing to the output lambda.
            var reciprocal = 1.0 / input_shape[reduce_dim]

            @always_inline
            @__copy_capture(reciprocal)
            @parameter
            fn wrapped_output_mul[
                _dtype: DType, width: Int, rank: Int
            ](indices: IndexList[rank], value: SIMD[_dtype, width]):
                var mean_val = value * reciprocal.cast[_dtype]()
                output_fn[width, rank](
                    indices, mean_val._refine[dtype, width]()
                )

            _reduce_generator[
                input_fn_wrapper,
                wrapped_output_mul,
                reduce_impl,
                single_thread_blocking_override=single_thread_blocking_override,
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
            fn wrapped_output_div[
                _dtype: DType, width: Int, rank: Int
            ](indices: IndexList[rank], value: SIMD[_dtype, width]):
                var mean_val = value / dim_size
                output_fn[width, rank](
                    indices, mean_val._refine[dtype, width]()
                )

            _reduce_generator[
                input_fn_wrapper,
                wrapped_output_div,
                reduce_impl,
                single_thread_blocking_override=single_thread_blocking_override,
                target=target,
            ](
                input_shape,
                init=Scalar[dtype](0),
                reduce_dim=reduce_dim,
                context=context,
            )


fn mean[
    dtype: DType,
    input_fn_1d: fn[dtype_: DType, width: Int] (idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int) raises -> Scalar[dtype]:
    # TODO docstring.

    var total = sum[dtype, input_fn_1d](length)

    @parameter
    if dtype.is_integral():
        return total // length
    else:
        return total / length


# ===-----------------------------------------------------------------------===#
# variance
# ===-----------------------------------------------------------------------===#


fn variance(
    src: NDBuffer[rank=1], mean_value: Scalar[src.type], correction: Int = 1
) raises -> Scalar[src.type]:
    """Given a mean, computes the variance of elements in a buffer.

    The mean value is used to avoid a second pass over the data:

    ```
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Args:
        src: The buffer.
        mean_value: The mean value of the buffer.
        correction: Normalize variance by size - correction.

    Returns:
        The variance value of the elements in a buffer.
    """

    debug_assert(len(src) > 1, "input length must be greater than 1")

    @parameter
    @always_inline
    @__copy_capture(src)
    fn input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](src.load[width=width](idx))

    return variance[src.type, input_fn_1d](len(src), mean_value, correction)


fn variance[
    dtype: DType,
    input_fn_1d: fn[dtype_: DType, width: Int] (idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int, mean_value: Scalar[dtype], correction: Int = 1) raises -> Scalar[
    dtype
]:
    # TODO docstring.

    @always_inline
    @parameter
    fn input_fn_nd[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        var mean_simd = SIMD[mean_value.dtype, width](mean_value).cast[_dtype]()
        var x = input_fn_1d[_dtype, width](idx[0])
        var diff = x.cast[_dtype]() - mean_simd
        return (diff * diff)._refine[_dtype]()

    var out: Scalar[dtype] = 0

    @always_inline
    @parameter
    fn output_fn[
        _dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank], value: SIMD[_dtype, width]):
        out = value._refine[dtype, 1]()

    @always_inline
    @parameter
    fn reduce_fn_wrapper[
        dtype: DType, width: Int
    ](acc: SIMD[dtype, width], val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return acc + val

    var shape = IndexList[1](length)

    _reduce_generator[
        input_fn_nd,
        output_fn,
        reduce_fn_wrapper,
        single_thread_blocking_override=True,
    ](
        shape,
        init=Scalar[mean_value.dtype](0),
        reduce_dim=0,
    )

    return out / (length - correction)


fn variance(
    src: NDBuffer[rank=1], correction: Int = 1
) raises -> Scalar[src.type]:
    """Computes the variance value of the elements in a buffer.

    ```
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Args:
        src: The buffer.
        correction: Normalize variance by size - correction (Default=1).

    Returns:
        The variance value of the elements in a buffer.
    """

    @always_inline
    @parameter
    fn input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](src.load[width=width](idx))

    return variance[src.type, input_fn_1d](len(src), correction)


fn variance[
    dtype: DType,
    input_fn_1d: fn[dtype_: DType, width: Int] (idx: Int) capturing -> SIMD[
        dtype_, width
    ],
](length: Int, correction: Int = 1) raises -> Scalar[dtype]:
    var mean_value = mean[dtype, input_fn_1d](length)
    return variance[dtype, input_fn_1d](length, mean_value, correction)


# ===-----------------------------------------------------------------------===#
# all_true
# ===-----------------------------------------------------------------------===#


fn all_true(src: NDBuffer[rank=1]) -> Bool:
    """Returns True if all the elements in a buffer are True and False otherwise.

    Args:
        src: The buffer.

    Returns:
        True if all of the elements of the buffer are True and False otherwise.
    """

    @always_inline
    @parameter
    fn _reduce_fn[
        dtype: DType, simd_width: Int
    ](val: SIMD[dtype, simd_width]) -> Bool:
        @parameter
        if dtype is DType.bool:
            return val.cast[DType.bool]().reduce_and()
        return val.ne(0).reduce_and()

    @always_inline
    @parameter
    fn _continue_fn(val: Bool) -> Bool:
        return val

    return reduce_boolean[_reduce_fn, _continue_fn](src, False)


# ===-----------------------------------------------------------------------===#
# any_true
# ===-----------------------------------------------------------------------===#


fn any_true(src: NDBuffer[rank=1]) -> Bool:
    """Returns True if any the elements in a buffer are True and False otherwise.

    Args:
        src: The buffer.

    Returns:
        True if any of the elements of the buffer are True and False otherwise.
    """

    @always_inline
    @parameter
    fn _reduce_fn[
        dtype: DType, simd_width: Int
    ](val: SIMD[dtype, simd_width]) -> Bool:
        @parameter
        if dtype is DType.bool:
            return val.cast[DType.bool]().reduce_or()
        return val != 0

    @always_inline
    @parameter
    fn _continue_fn(val: Bool) -> Bool:
        return not val

    return reduce_boolean[_reduce_fn, _continue_fn](src, False)


# ===-----------------------------------------------------------------------===#
# none_true
# ===-----------------------------------------------------------------------===#


fn none_true(src: NDBuffer[rank=1]) -> Bool:
    """Returns True if none of the elements in a buffer are True and False
    otherwise.

    Args:
        src: The buffer.

    Returns:
        True if none of the elements of the buffer are True and False otherwise.
    """

    @always_inline
    @parameter
    fn _reduce_fn[
        dtype: DType, simd_width: Int
    ](val: SIMD[dtype, simd_width]) -> Bool:
        @parameter
        if dtype is DType.bool:
            return not val.cast[DType.bool]().reduce_or()
        # TODO: simplify this implementation?
        return val == 0

    @always_inline
    @parameter
    fn _continue_fn(val: Bool) -> Bool:
        return val

    return reduce_boolean[_reduce_fn, _continue_fn](src, True)


# ===-----------------------------------------------------------------------===#
# cumsum function
# ===-----------------------------------------------------------------------===#


@always_inline
fn _cumsum_small(
    dst: NDBuffer[mut=True, rank=1], src: NDBuffer[dst.type, 1, *_]
):
    dst[0] = src[0]
    for i in range(1, len(dst)):
        dst[i] = src[i] + dst[i - 1]


fn cumsum(dst: NDBuffer[mut=True, rank=1], src: NDBuffer[dst.type, 1, *_]):
    """Computes the cumulative sum of all elements in a buffer.
       dst[i] = src[i] + src[i-1] + ... + src[0].

    Args:
        dst: The buffer that stores the result of cumulative sum operation.
        src: The buffer of elements for which the cumulative sum is computed.
    """

    debug_assert(len(src) != 0, "Input must not be empty")
    debug_assert(len(dst) != 0, "Output must not be empty")

    alias simd_width = simd_width_of[dst.type]()

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
    var div_size = align_down(len(dst), simd_width)

    # Number of inner-loop iterations (for shift previous result and add).
    alias rep = log2_floor(simd_width)

    for i in range(0, div_size, simd_width):
        var x_simd = src.load[width=simd_width](i)

        @parameter
        for i in range(rep):
            x_simd += x_simd.shift_right[2**i]()

        dst.store(i, x_simd)

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
        var x_simd = dst.load[width=simd_width](i) + offset
        dst.store(i, x_simd)
        offset = SIMD[dst.type, simd_width](x_simd[simd_width - 1])

    # Handles the tail, i.e., num of elements at the end that don't
    # fit within a simd_width-elements vector.
    for i in range(div_size, len(dst)):
        dst[i] = dst[i - 1] + src[i]
