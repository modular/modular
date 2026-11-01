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
"""Implements tensor math operations."""

import max.algorithm.reduction as reduction
from std.math.math import max as b_max
from std.sys import align_of
from layout import (
    Coord,
    DefaultEngine,
    Idx,
    TileTensor,
    row_major,
    stack_allocation,
)


@inline(.always)
def outer_product_acc(
    res: TileTensor[mut=True, ...],
    lhs: TileTensor,
    rhs: TileTensor,
):
    """Updates result tensor with the outer product of two vectors.

    Computes `res += outer(lhs, rhs)` where `lhs` and `rhs` are vectors and
    `res` is a matrix. Vectorized tensors multiply element-wise, so the three
    tensors must share one `element_size`.

    Args:
        res: The result matrix to accumulate into, shape (M, N).
        lhs: The left-hand side vector, shape (M,).
        rhs: The right-hand side vector, shape (N,).

    Constraints:

        All tensors must have statically known shapes.
        `res` must be rank 2.
        `lhs` and `rhs` must be rank 1.
        `res.dim[0]()` must equal `lhs.dim[0]()`, and `res.dim[1]()` must
        equal `rhs.dim[0]()`.
    """
    comptime assert (
        res.all_dims_known and lhs.all_dims_known and rhs.all_dims_known
    ), "outer_product_acc expects inputs with statically known shapes"
    comptime assert res.flat_rank == 2, "Only rank 2 res is allowed."
    comptime assert lhs.flat_rank == 1, "Only rank 1 lhs is allowed."
    comptime assert rhs.flat_rank == 1, "Only rank 1 rhs is allowed."
    comptime assert (
        lhs.element_size == res.element_size
        and rhs.element_size == res.element_size
    ), "outer_product_acc expects inputs with the same element size"

    comptime dtype = res.dtype
    comptime M = res.static_shape[0]
    comptime N = res.static_shape[1]

    comptime assert lhs.static_shape[0] == M, "lhs shape mismatch"
    comptime assert rhs.static_shape[0] == N, "rhs shape mismatch"

    comptime for i in range(M):
        comptime for j in range(N):
            res[i, j] += rebind[res.ElementType](
                lhs[Idx[i]].cast[dtype]()
            ) * rebind[res.ElementType](rhs[Idx[j]].cast[dtype]())


@inline(.always)
def max(
    x: TileTensor,
    y: TileTensor[x.dtype, x.LayoutType, ...],
    out result: TileTensor[
        x.dtype,
        x.LayoutType,
        MutUntrackedOrigin,
        Engine=DefaultEngine[element_width=x.element_size],
        address_space=x.address_space,
    ],
):
    """Computes element-wise maxima into a new tensor with the input layout.

    Args:
        x: First input tensor.
        y: Second input tensor with the same dtype, layout, and element width.

    Returns:
        An independently allocated tensor with the input layout and SIMD width.

    Constraints:
        Shapes and strides must be static, and strides must be nonnegative.
    """
    comptime assert x.all_dims_known, "max expects a statically known layout"
    comptime assert (
        x.element_size == y.element_size
    ), "max expects inputs with the same element size"
    comptime for axis in range(x.flat_rank):
        comptime assert (
            x.LayoutType.static_stride[axis] >= 0
        ), "max expects nonnegative strides"

    # Cosize includes each cell's first scalar, not its remaining SIMD lanes.
    comptime storage_size = b_max(
        1, x.LayoutType.static_cosize + x.element_size - 1
    )
    var storage = stack_allocation[
        x.dtype,
        address_space=x.address_space,
        alignment=align_of[x.ElementType](),
    ](row_major[storage_size]())
    var res = type_of(result)(
        storage.unsafe_ptr().unsafe_bitcast[x.ElementType](), x.layout
    )
    comptime for i in range(x.LayoutType.static_product):
        # Strided SIMD cells may have only scalar alignment.
        var a = x.load[alignment=align_of[x.dtype]()](Coord(Idx[i]))
        var b = y.load[width=x.element_size, alignment=align_of[y.dtype]()](
            Coord(Idx[i])
        )
        storage.store[alignment=align_of[x.dtype]()](
            Coord(x.layout(Idx[i])), b_max(a, b)
        )
    return res


@inline(.always)
def _reduce_tile_tensor[
    axis: Int, is_max: Bool
](inp: TileTensor, outp: TileTensor[mut=True, ...]):
    comptime assert inp.rank == 2, "Only rank-2 reduction is supported"
    comptime assert outp.rank == 1, "Reduction output must have rank 1"
    comptime assert 0 <= axis < 2, "Reduction axis must be 0 or 1"
    comptime assert (
        inp.shape_known and outp.shape_known
    ), "Reduction expects statically known shapes"
    comptime assert (
        inp.element_size == outp.element_size
    ), "Reduction input and output must have the same element size"

    comptime output_size = Coord[
        inp.LayoutType._shape_types[1 - axis]
    ].static_product
    comptime reduction_size = Coord[
        inp.LayoutType._shape_types[axis]
    ].static_product
    comptime assert (
        output_size == Coord[outp.LayoutType._shape_types[0]].static_product
    ), "Non-reduction dimensions must match"

    comptime for i in range(output_size):
        var value = SIMD[outp.dtype, outp.element_size].MIN if is_max else SIMD[
            outp.dtype, outp.element_size
        ](0)
        comptime for j in range(reduction_size):
            var element: SIMD[outp.dtype, outp.element_size]
            comptime if axis == 0:
                element = inp.load[
                    width=outp.element_size, alignment=align_of[inp.dtype]()
                ]((j, i)).cast[outp.dtype]()
            else:
                element = inp.load[
                    width=outp.element_size, alignment=align_of[inp.dtype]()
                ]((i, j)).cast[outp.dtype]()
            comptime if is_max:
                value = b_max(value, element)
            else:
                value += element
        outp.store[alignment=align_of[outp.dtype]()]((i,), value)


comptime _TileReductionResult[
    dtype: DType, size: Int, width: Int, address_space: AddressSpace
] = TileTensor[
    dtype,
    type_of(row_major[size * width]()),
    MutUntrackedOrigin,
    address_space=address_space,
].VectorizedType[
    width
]


@inline(.always)
def sum[axis: Int](inp: TileTensor, outp: TileTensor[mut=True, ...]):
    """Computes lane-wise sums along one outer dimension.

    Parameters:
        axis: The outer dimension to reduce, either 0 or 1.

    Args:
        inp: The rank-2 input tensor, including nested outer dimensions.
        outp: The rank-1 output tensor. Values are cast to its dtype before
            accumulation.

    Constraints:
        Shapes must be static. The surviving dimension and SIMD element width
        must match between input and output.
    """
    _reduce_tile_tensor[axis, is_max=False](inp, outp)


@inline(.always)
def max[axis: Int](inp: TileTensor, outp: TileTensor[mut=True, ...]):
    """Computes lane-wise maxima along one outer dimension.

    Parameters:
        axis: The outer dimension to reduce, either 0 or 1.

    Args:
        inp: The rank-2 input tensor, including nested outer dimensions.
        outp: The rank-1 output tensor. Values are cast to its dtype before
            comparison.

    Constraints:
        Shapes must be static. The surviving dimension and SIMD element width
        must match between input and output.
    """
    _reduce_tile_tensor[axis, is_max=True](inp, outp)


@inline(.always)
def sum[
    axis: Int
](inp: TileTensor) -> _TileReductionResult[
    inp.dtype,
    Coord[inp.LayoutType._shape_types[1 - axis]].static_product,
    inp.element_size,
    inp.address_space,
]:
    """Computes lane-wise sums into a new contiguous tensor.

    Parameters:
        axis: The outer dimension to reduce, either 0 or 1.

    Args:
        inp: A statically shaped rank-2 tensor.

    Returns:
        A rank-1 tensor with the input dtype and SIMD element width. Adjacent
        elements have disjoint storage.

    Examples:

    ```mojo
    from std.collections import Array
    from layout import TileTensor, row_major
    from layout.math import sum

    var data: Array[Int32, 6] = [0, 1, 2, 3, 4, 5]
    var tensor = TileTensor(data, row_major[2, 3]())
    var result = sum[0](tensor)
    print(result[0], result[1], result[2])  # 3 5 7
    ```
    """
    comptime size = Coord[inp.LayoutType._shape_types[1 - axis]].static_product
    var result = stack_allocation[
        inp.dtype,
        address_space=inp.address_space,
        alignment=align_of[inp.ElementType](),
    ](row_major[size * inp.element_size]()).vectorize[inp.element_size]()
    sum[axis](inp, result)
    return result


@inline(.always)
def max[
    axis: Int
](inp: TileTensor) -> _TileReductionResult[
    inp.dtype,
    Coord[inp.LayoutType._shape_types[1 - axis]].static_product,
    inp.element_size,
    inp.address_space,
]:
    """Computes lane-wise maxima into a new contiguous tensor.

    Parameters:
        axis: The outer dimension to reduce, either 0 or 1.

    Args:
        inp: A statically shaped rank-2 tensor.

    Returns:
        A rank-1 tensor with the input dtype and SIMD element width. Adjacent
        elements have disjoint storage.
    """
    comptime size = Coord[inp.LayoutType._shape_types[1 - axis]].static_product
    var result = stack_allocation[
        inp.dtype,
        address_space=inp.address_space,
        alignment=align_of[inp.ElementType](),
    ](row_major[size * inp.element_size]()).vectorize[inp.element_size]()
    max[axis](inp, result)
    return result


def variance(src: TileTensor, correction: Int = 1) raises -> Scalar[src.dtype]:
    """Computes the variance value of the elements in a buffer.

    ```
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Args:
        src: The buffer.
        correction: Normalize variance by size - correction (Default=1).

    Returns:
        The variance value of the elements in a buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """

    @inline(.always)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) {imm} -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.load[width=width, alignment=align_of[src.dtype]()](Coord(idx))
        )

    return reduction.variance[src.dtype](
        src.num_elements(), input_fn_1d, correction
    )


def mean(src: TileTensor) raises -> Scalar[src.dtype]:
    """Computes the mean value of the elements in a buffer.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """
    comptime assert src.rank == 1, "src must be of rank 1"

    assert src.num_elements() != 0, "input must not be empty"

    @inline(.always)
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) {imm} -> SIMD[dtype_, width]:
        return rebind[SIMD[dtype_, width]](
            src.load[width=width, alignment=align_of[src.dtype]()](Coord(idx))
        )

    return reduction.mean[src.dtype](src.num_elements(), input_fn_1d)
