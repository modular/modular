# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param_msg, debug_assert
from Buffer import Buffer, NDBuffer, prod_dims
from DType import DType
from Functional import vectorize
from Index import StaticIntTuple
from Numerics import inf, neginf
from List import DimList, Dim, create_dim_list
from Range import range
from SIMD import SIMD
from TargetInfo import dtype_sizeof

# ===----------------------------------------------------------------------===#
# reduce
# ===----------------------------------------------------------------------===#
# Implements a simd reduction.
# Consists of 3 steps:
#   1. Iterate over simd_width size chunks of src and use map_fn to update a simd accumulator.
#   2. Reduce the simd accumulator into a single scalar using reduce_fn.
#   3. Iterate over the remainder of src and apply the map_fn on simd elements
#      with simd_width=1.
@always_inline
fn reduce[
    simd_width: Int,
    size: Dim,
    type: DType,
    acc_type: DType,
    map_fn: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow,`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 2> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    reduce_fn: __mlir_type[
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
        SIMD[__mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType], 1],
        `>`,
    ],
](src: Buffer[size, type], init: SIMD[acc_type, 1]) -> SIMD[acc_type, 1]:
    alias unroll_factor = 8  # TODO: search
    # TODO: explicitly unroll like vectorize_unroll does.
    alias unrolled_simd_width = simd_width * unroll_factor
    var acc_simd = SIMD[acc_type, unrolled_simd_width].splat(init)
    let len = src.__len__()
    let vector_end = (len // unrolled_simd_width) * unrolled_simd_width
    for i in range(0, vector_end, unrolled_simd_width):
        acc_simd = map_fn[unrolled_simd_width, acc_type, type](
            acc_simd, src.simd_load[unrolled_simd_width](i)
        )

    var acc = reduce_fn[unrolled_simd_width, acc_type](acc_simd)
    for ii in range(vector_end, len):  # TODO(#8365) use `i`
        acc = map_fn[1, acc_type, type](acc, src[ii])
    return acc[0].value


@always_inline
fn _reduce_3D[
    simd_width: Int,
    input_shape: DimList[3],
    output_shape: DimList[2],
    type: DType,
    acc_type: DType,
    map_fn: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow,`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 2> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    reduce_fn: __mlir_type[
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
        SIMD[__mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType], 1],
        `>`,
    ],
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
        fn reduce_inner_axis():
            alias sz = input_shape.at[1]()
            # TODO: parallelize
            for i in range(h):
                let offset = src._offset(StaticIntTuple[3](i, 0, 0))
                let input = Buffer[sz, type](offset.address, w)
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
    fn get_unroll_factor[simd_width: Int, dtype_size: Int]() -> Int:
        alias cache_line_size = 64
        alias unroll_factor = cache_line_size // (simd_width * dtype_size)
        assert_param_msg[unroll_factor > 0, "unroll_factor must be > 0"]()
        return unroll_factor

    alias unroll_factor = get_unroll_factor[simd_width, dtype_sizeof[type]()]()
    alias usimd_width = unroll_factor * simd_width
    for i in range(h):

        @always_inline
        fn reduce_w_chunked[simd_width: Int](idx: Int):
            var accum = SIMD[acc_type, simd_width].splat(init)
            for j in range(w):
                let chunk = src.simd_load[simd_width](
                    StaticIntTuple[3](i, j, idx)
                )
                accum = map_fn[simd_width, acc_type, type](accum, chunk)
            dst.simd_store[simd_width](StaticIntTuple[2](i, idx), accum)

        vectorize[usimd_width, reduce_w_chunked](c)


@always_inline
fn reduce[
    simd_width: Int,
    rank: Int,
    input_shape: DimList[rank],
    output_shape: DimList[rank],
    type: DType,
    acc_type: DType,
    map_fn: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow,`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 2> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        ` borrow) -> `,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
        ],
        `>`,
    ],
    reduce_fn: __mlir_type[
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
        SIMD[__mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType], 1],
        `>`,
    ],
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
    """

    let h_dynamic = prod_dims[0, reduce_axis](src)
    let w_dynamic = src.dim[reduce_axis]()
    let c_dynamic = prod_dims[reduce_axis + 1, rank](src)

    alias h_static = input_shape.product_range[0, reduce_axis]()
    alias w_static = input_shape.at[reduce_axis]()
    alias c_static = input_shape.product_range[reduce_axis + 1, rank]()

    alias input_3d_shape = create_dim_list(h_static, w_static, c_static)
    alias output_3d_shape = create_dim_list(h_static, c_static)

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
# max
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_max[
    simd_width: Int,
    type: DType,
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    """Helper function that computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


@always_inline
fn _simd_max_elementwise[
    simd_width: Int,
    acc_type: DType,
    type: DType,
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Helper function that computes the elementwise max of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x.max(y.cast[acc_type]())


fn max[
    simd_width: Int,
    size: Dim,
    type: DType,
](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the max element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_max_elementwise, _simd_max
    ](src, src[0])


fn max[
    simd_width: Int,
    rank: Int,
    input_shape: DimList[rank],
    output_shape: DimList[rank],
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the max across reduce_axis of an NDBuffer."""
    return reduce[
        simd_width,
        rank,
        input_shape,
        output_shape,
        type,
        type,
        _simd_max_elementwise,
        _simd_max,
        reduce_axis,
    ](src, dst, neginf[type]())


# ===----------------------------------------------------------------------===#
# min
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_min[
    simd_width: Int,
    type: DType,
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    """Helper function that computes the min element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_min()


@always_inline
fn _simd_min_elementwise[
    simd_width: Int,
    acc_type: DType,
    type: DType,
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Helper function that computes the elementwise min of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x.min(y.cast[acc_type]())


fn min[
    simd_width: Int,
    size: Dim,
    type: DType,
](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the min element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_min_elementwise, _simd_min
    ](src, src[0])


fn min[
    simd_width: Int,
    rank: Int,
    input_shape: DimList[rank],
    output_shape: DimList[rank],
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the min across reduce_axis of an NDBuffer."""
    return reduce[
        simd_width,
        rank,
        input_shape,
        output_shape,
        type,
        type,
        _simd_min_elementwise,
        _simd_min,
        reduce_axis,
    ](src, dst, inf[type]())


# ===----------------------------------------------------------------------===#
# sum
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_sum[
    simd_width: Int,
    type: DType,
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    """Helper function that computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


@always_inline
fn _simd_sum_elementwise[
    simd_width: Int,
    acc_type: DType,
    type: DType,
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Helper function that computes the elementwise sum of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x + y.cast[acc_type]()


fn sum[
    simd_width: Int,
    size: Dim,
    type: DType,
](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the sum element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_sum_elementwise, _simd_sum
    ](src, 0)


fn sum[
    simd_width: Int,
    rank: Int,
    input_shape: DimList[rank],
    output_shape: DimList[rank],
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the sum across reduce_axis of an NDBuffer."""
    return reduce[
        simd_width,
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
    simd_width: Int,
    type: DType,
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    """Helper function that computes the product of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_mul()


@always_inline
fn _simd_product_elementwise[
    simd_width: Int,
    acc_type: DType,
    type: DType,
](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    acc_type, simd_width
]:
    """Helper function that computes the elementwise product of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    return x * y.cast[acc_type]()


fn product[
    simd_width: Int,
    size: Dim,
    type: DType,
](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the product element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_product_elementwise, _simd_product
    ](src, 1)


fn product[
    simd_width: Int,
    rank: Int,
    input_shape: DimList[rank],
    output_shape: DimList[rank],
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the product across reduce_axis of an NDBuffer."""
    return reduce[
        simd_width,
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


fn mean[
    simd_width: Int,
    size: Dim,
    type: DType,
](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the mean value of the elements in a buffer."""

    debug_assert(src.__len__() != 0, "input must not be empty")

    return (SIMD[type, 1](sum[simd_width, size, type](src)) / src.__len__())[
        0
    ].value


fn mean[
    simd_width: Int,
    rank: Int,
    input_shape: DimList[rank],
    output_shape: DimList[rank],
    type: DType,
    reduce_axis: Int,
](
    src: NDBuffer[rank, input_shape, type],
    dst: NDBuffer[rank, output_shape, type],
):
    """Computes the mean across reduce_axis of an NDBuffer."""

    sum[
        simd_width,
        rank,
        input_shape,
        output_shape,
        type,
        reduce_axis,
    ](src, dst)

    let n = src.dim[reduce_axis]()
    let n_recip = SIMD[type, 1](1) / n
    let dst_1d = dst.flatten()

    @always_inline
    fn div[simd_width: Int](idx: Int):
        let elem = dst_1d.simd_load[simd_width](idx)
        let to_store = elem * n_recip
        dst_1d.simd_store[simd_width](idx, to_store)

    vectorize[simd_width, div](dst_1d.__len__())


# ===----------------------------------------------------------------------===#
# variance
# ===----------------------------------------------------------------------===#


fn variance[
    simd_width: Int,
    size: Dim,
    type: DType,
](src: Buffer[size, type]) -> SIMD[type, 1]:
    """Computes the variance value of the elements in a buffer."""

    debug_assert(src.__len__() > 1, "input length must be greater than 1")

    let mean_value = mean[simd_width, size, type](src)

    @always_inline
    fn _simd_variance_elementwise[
        simd_width: Int,
        acc_type: DType,
        type: DType,
    ](x: SIMD[acc_type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
        acc_type, simd_width
    ]:
        """Helper function that computes the equation $sum (x_i - u)^2 + y$"""
        let mean_simd = SIMD[type, simd_width](mean_value).cast[acc_type]()
        let diff = y.cast[acc_type]() - mean_simd
        return x + diff * diff

    let numerator: SIMD[type, 1] = reduce[
        simd_width, size, type, type, _simd_variance_elementwise, _simd_sum
    ](src, 0)
    return (numerator / (src.__len__() - 1))[0].value
