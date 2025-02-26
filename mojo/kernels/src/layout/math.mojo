# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math

from builtin.math import max as b_max
from layout import LayoutTensor

from utils.numerics import min_or_neg_inf


# Updates res with the outer product of lhs, rhs vectors, res += outer(lhs, rhs).
#
@always_inline
fn outer_product_acc(
    res: LayoutTensor,
    lhs: LayoutTensor,
    rhs: LayoutTensor,
):
    constrained[
        res.layout.known_shape()
        and lhs.layout.known_shape()
        and rhs.layout.known_shape(),
        "outer_product_acc expects inputs with statically know shapes",
    ]()
    constrained[res.rank == 2, "Only rank 2 res is allowed."]()
    constrained[lhs.rank == 1, "Only rank 1 lhs is allowed."]()
    constrained[rhs.rank == 1, "Only rank 1 rhs is allowed."]()

    alias dtype = res.dtype

    alias M = res.shape[0]()
    alias N = res.shape[1]()

    constrained[lhs.shape[0]() == M, "lhs shape mismatch"]()
    constrained[rhs.shape[0]() == N, "rhs shape mismatch"]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            res[i, j] += rebind[res.element_type](
                lhs[i].cast[dtype]()
            ) * rebind[res.element_type](rhs[j].cast[dtype]())


@always_inline
fn _reduce[
    axis: Int,
    init_func: fn[dtype: DType, width: Int] () -> SIMD[dtype, width],
    func: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> (SIMD[dtype, width]),
](inp: LayoutTensor, out: LayoutTensor):
    constrained[
        inp.layout.known_shape() and out.layout.known_shape(),
        "_reduce expects inputs with statically know shapes",
    ]()
    constrained[
        inp.rank - 1 == out.rank,
        "_reduce expects output of rank = inp.rank - 1",
    ]()

    @parameter
    for dim in range(axis):

        @parameter
        if dim != axis:
            constrained[
                inp.shape[dim]() == out.shape[dim](),
                "_reduce expects none reduction dims to be the same",
            ]()

    @parameter
    for dim in range(axis + 1, inp.rank):

        @parameter
        if dim != axis:
            constrained[
                inp.shape[dim]() == out.shape[dim - 1](),
                "_reduce expects none reduction dims to be the same",
            ]()

    # TODO(KERN-777): We need to relax this constraine.
    constrained[inp.rank == 2, "Only rank-2 _reduce is supported"]()

    @parameter
    if inp.rank == 2 and axis == 1:

        @parameter
        for i in range(inp.shape[0]()):
            var reduce_val = init_func[out.dtype, out.element_size]()

            @parameter
            for j in range(inp.shape[1]()):
                reduce_val = func(
                    reduce_val,
                    rebind[out.element_type](inp[i, j].cast[out.dtype]()),
                )

            out[i] = reduce_val

    elif inp.rank == 2 and axis == 0:

        @parameter
        for j in range(inp.shape[1]()):
            var reduce_val = init_func[out.dtype, out.element_size]()

            @parameter
            for i in range(inp.shape[0]()):
                reduce_val = func(
                    reduce_val,
                    rebind[out.element_type](inp[i, j].cast[out.dtype]()),
                )

            out[j] = reduce_val


@always_inline
fn sum[axis: Int](inp: LayoutTensor, out: LayoutTensor):
    fn sum_init[dtype: DType, width: Int]() -> SIMD[dtype, width]:
        return 0

    fn sum_func[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a + b

    _reduce[axis, sum_init, sum_func](inp, out)


@always_inline
fn max[axis: Int](inp: LayoutTensor, out: LayoutTensor):
    fn max_init[dtype: DType, width: Int]() -> SIMD[dtype, width]:
        return SIMD[dtype, width].MIN

    fn max_func[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return b_max(a, b)

    _reduce[axis, max_init, max_func](inp, out)


fn _reduce_res_row_major_shape(axis: Int, in_layout: Layout) -> Layout:
    var res_shape = IntTuple()
    for dim in range(0, axis):
        res_shape.append(Int(in_layout.shape[dim]))
    for dim in range(axis + 1, in_layout.rank()):
        res_shape.append(Int(in_layout.shape[dim]))
    return Layout.row_major(res_shape)


@always_inline
fn max[
    axis: Int
](
    inp: LayoutTensor,
    out res: LayoutTensor[
        inp.dtype,
        _reduce_res_row_major_shape(axis, inp.layout),
        address_space = inp.address_space,
        element_layout = inp.element_layout,
        layout_bitwidth = inp.layout_bitwidth,
    ],
):
    var res_tensor = __type_of(res).stack_allocation()
    max[axis](inp, res_tensor)
    return res_tensor


@always_inline
fn max(
    x: LayoutTensor, y: __type_of(x)
) -> __type_of(x.origin_cast[True, MutableAnyOrigin]()):
    constrained[
        x.layout.all_dims_known(), "max expects tensor of statically know shape"
    ]()
    var res_tensor = __type_of(x).stack_allocation()

    @parameter
    for i in range(res_tensor.layout.size()):
        alias idx = x.layout(i)
        res_tensor.ptr[idx] = b_max(x.ptr[idx], y.ptr[idx])
    return res_tensor


@always_inline
fn sum[
    axis: Int,
](
    inp: LayoutTensor,
    out res: LayoutTensor[
        inp.dtype,
        _reduce_res_row_major_shape(axis, inp.layout),
        address_space = inp.address_space,
        element_layout = inp.element_layout,
        layout_bitwidth = inp.layout_bitwidth,
    ],
):
    var res_tensor = __type_of(res).stack_allocation()
    sum[axis](inp, res_tensor)
    return res_tensor


@always_inline
fn exp(
    inp: LayoutTensor,
) -> __type_of(inp.origin_cast[True, MutableAnyOrigin]()):
    @parameter
    fn exp_func(val: inp.element_type) -> inp.element_type:
        return math.exp(val)

    return inp._stack_copy()._elementwise_unary[exp_func]()
