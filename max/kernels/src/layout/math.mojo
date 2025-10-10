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
"""Implements math methods that work on layout tensors."""

import math
from sys.info import simd_width_of

import algorithm.reduction
from algorithm import vectorize
from builtin.math import max as b_max
from layout import LayoutTensor

from utils.index import IndexList


@always_inline
fn outer_product_acc(
    res: LayoutTensor,
    lhs: LayoutTensor,
    rhs: LayoutTensor,
):
    """Updates result tensor with the outer product of two vectors.

    Computes `res += outer(lhs, rhs)` where `lhs` and `rhs` are vectors and
    `res` is a matrix.

    Args:
        res: The result matrix to accumulate into, shape (M, N).
        lhs: The left-hand side vector, shape (M,).
        rhs: The right-hand side vector, shape (N,).

    Constraints:

        All tensors must have statically known shapes.
        `res` must be rank 2.
        `lhs` and `rhs` must be rank 1.
        `res.shape[0]` `==` `lhs.shape[0]` and `res.shape[1]` `==` `rhs.shape[0]`.
    """

    constrained[
        res.layout.known_shape()
        and lhs.layout.known_shape()
        and rhs.layout.known_shape(),
        "outer_product_acc expects inputs with statically known shapes",
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
](inp: LayoutTensor, outp: LayoutTensor):
    constrained[
        inp.layout.known_shape() and outp.layout.known_shape(),
        "_reduce expects inputs with statically know shapes",
    ]()
    constrained[
        inp.rank - 1 == outp.rank,
        "_reduce expects output of rank = inp.rank - 1",
    ]()

    @parameter
    for dim in range(axis):

        @parameter
        if dim != axis:
            constrained[
                inp.shape[dim]() == outp.shape[dim](),
                "_reduce expects none reduction dims to be the same",
            ]()

    @parameter
    for dim in range(axis + 1, inp.rank):

        @parameter
        if dim != axis:
            constrained[
                inp.shape[dim]() == outp.shape[dim - 1](),
                "_reduce expects none reduction dims to be the same",
            ]()

    # TODO(KERN-777): We need to relax this constraine.
    constrained[inp.rank == 2, "Only rank-2 _reduce is supported"]()

    @parameter
    if inp.rank == 2 and axis == 1:

        @parameter
        for i in range(inp.shape[0]()):
            var reduce_val = init_func[outp.dtype, outp.element_size]()

            @parameter
            for j in range(inp.shape[1]()):
                reduce_val = func(
                    reduce_val,
                    rebind[outp.element_type](inp[i, j].cast[outp.dtype]()),
                )

            outp[i] = reduce_val

    elif inp.rank == 2 and axis == 0:

        @parameter
        for j in range(inp.shape[1]()):
            var reduce_val = init_func[outp.dtype, outp.element_size]()

            @parameter
            for i in range(inp.shape[0]()):
                reduce_val = func(
                    reduce_val,
                    rebind[outp.element_type](inp[i, j].cast[outp.dtype]()),
                )

            outp[j] = reduce_val


@always_inline
fn sum[axis: Int](inp: LayoutTensor, outp: LayoutTensor):
    """Computes sum reduction along specified axis.

    Reduces the input tensor by summing elements along the specified axis
    and stores the result in the output tensor.

    Parameters:
        axis: The axis to sum along.

    Args:
        inp: The input tensor to sum.
        outp: The output tensor to store sum results.

    Constraints:
        All tensors must have statically known shapes.
        `outp.rank` must equal `inp.rank - 1`.
        Non-reduction dimensions must match between inp and outp.
        Currently only supports rank-2 inputs.

    Example:

    ```mojo
    from layout import LayoutTensor, Layout
    from layout.math import sum

    data = InlineArray[Int32, 6](0, 1, 2, 3, 4, 5)
    tensor = LayoutTensor[DType.int32, Layout.row_major(2, 3)](data)
    print(tensor)
    print("-----")
    print(sum[0](tensor))
    ```

    Output:

    ```plaintext
    0 1 2
    3 4 5
    -----
    3 5 7
    ```
    """

    fn sum_init[dtype: DType, width: Int]() -> SIMD[dtype, width]:
        return 0

    fn sum_func[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a + b

    _reduce[axis, sum_init, sum_func](inp, outp)


@always_inline
fn max[axis: Int](inp: LayoutTensor, outp: LayoutTensor):
    """Computes maximum reduction along specified axis.

    Reduces the input tensor by taking maximum elements along the specified
    axis and stores the result in the output tensor.

    Parameters:
        axis: The axis to take maximum along.

    Args:
        inp: The input tensor to reduce.
        outp: The output tensor to store maximum results.

    Constraints:
        All tensors must have statically known shapes.
        `outp.rank` must equal `inp.rank - 1`.
        Non-reduction dimensions must match between `inp` and `outp`.
        Currently only supports rank-2 inputs.
    """

    fn max_init[dtype: DType, width: Int]() -> SIMD[dtype, width]:
        return SIMD[dtype, width].MIN

    fn max_func[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return b_max(a, b)

    _reduce[axis, max_init, max_func](inp, outp)


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
        MutableAnyOrigin,
        address_space = inp.address_space,
        element_layout = inp.element_layout,
        layout_int_type = inp.layout_int_type,
        linear_idx_type = inp.linear_idx_type,
    ],
):
    """Computes maximum reduction along specified axis, returning a new tensor.

    Reduces the input tensor by taking maximum elements along the specified
    axis and returns a new tensor with the results.

    Parameters:
        axis: The axis to take maximum along.

    Args:
        inp: The input tensor to reduce.

    Returns:
        A new tensor containing the maximum values along the specified axis.

    Constraints:
        All tensors must have statically known shapes.
        Result will have rank equal to `inp.rank` - 1.
        Non-reduction dimensions in the result match the input.
        Currently only supports rank-2 inputs.
    """

    var res_tensor = __type_of(res).stack_allocation()
    max[axis](inp, res_tensor)
    return res_tensor


@always_inline
fn max[
    dtype: DType, layout: Layout
](
    x: LayoutTensor[dtype, layout, **_], y: LayoutTensor[dtype, layout, **_]
) -> __type_of(x).MutableAnyType:
    """Computes element-wise maximum of two tensors.

    Returns a new tensor containing the element-wise maximum between the
    input tensors.

    Parameters:
        dtype: The data type of the input tensors.
        layout: The layout of the input tensors.

    Args:
        x: First input tensor.
        y: Second input tensor.

    Returns:
        A new tensor containing the element-wise maximum.

    Constraints:
        Input tensors must have statically known shapes and matching layouts.
    """

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
        MutableAnyOrigin,
        address_space = inp.address_space,
        element_layout = inp.element_layout,
        layout_int_type = inp.layout_int_type,
        linear_idx_type = inp.linear_idx_type,
    ],
):
    """Computes sum reduction along specified axis, returning a new tensor.

    Reduces the input tensor by summing elements along the specified axis
    and returns a new tensor with the results.

    Parameters:
        axis: The axis to sum along.

    Args:
        inp: The input tensor to sum.

    Returns:
        A new tensor containing the sum values along the specified axis.

    Constraints:
        All tensors must have statically known shapes.
        Result will have rank equal to `inp.rank` - 1.
        Non-reduction dimensions in the result match the input.
        Currently only supports rank-2 inputs.
    """

    var res_tensor = __type_of(res).stack_allocation()
    sum[axis](inp, res_tensor)
    return res_tensor


fn mean(src: LayoutTensor[**_]) raises -> Scalar[src.dtype]:
    """Computes the mean value of the elements in a buffer.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.
    """
    constrained[src.rank == 1, "src must be of rank 1"]()

    debug_assert(src.size() != 0, "input must not be empty")

    @parameter
    @always_inline
    fn input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        var src_idx = src.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
        )
        return rebind[SIMD[dtype_, width]](src.ptr.load[width=width](src_idx))

    return reduction.mean[src.dtype, input_fn_1d](src.size())


fn mean[
    reduce_axis: Int
](src: LayoutTensor[**_], dst: LayoutTensor[mut=True, src.dtype, **_]) raises:
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
    var dst_1d = LayoutTensor[
        dst.dtype,
        Layout.row_major(UNKNOWN_VALUE),
        address_space = dst.address_space,
    ](
        dst.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](dst.size())
        ),
    )

    @parameter
    if dst.dtype.is_integral():

        @always_inline
        @__copy_capture(dst_1d, n)
        @parameter
        fn normalize_integral[simd_width: Int](idx: Int):
            var idx_1d = dst_1d.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
            )
            var elem = dst_1d.ptr.load[width=simd_width](idx_1d)
            var to_store = elem // n
            dst_1d.ptr.store(idx_1d, to_store)

        vectorize[normalize_integral, simd_width](dst_1d.size())
    else:
        var n_recip = Scalar[dst.dtype](1) / n

        @always_inline
        @__copy_capture(dst_1d, n, n_recip)
        @parameter
        fn normalize_floating[simd_width: Int](idx: Int):
            var idx_1d = dst_1d.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
            )
            var elem = dst_1d.ptr.load[width=simd_width](idx_1d)
            var to_store = elem * n_recip
            dst_1d.ptr.store(idx_1d, to_store)

        vectorize[normalize_floating, simd_width](dst_1d.size())


fn variance(
    src: LayoutTensor[**_], correction: Int = 1
) raises -> Scalar[src.dtype]:
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
        var src_idx = src.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
        )
        return rebind[SIMD[dtype_, width]](src.ptr.load[width=width](src_idx))

    return reduction.variance[src.dtype, input_fn_1d](src.size(), correction)
