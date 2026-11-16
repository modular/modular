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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers gather, scatter, and indexed-access graph ops backed by the `nn.gather_scatter` kernels."""

from std.collections import Array
from std.math import ceildiv, gcd
from std.sys import align_of
from std.sys.info import simd_width_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceContextArray
from max.gpu.host.info import is_cpu
from layout import (
    Coord,
    CoordLike,
    TileTensor,
    coord_to_index_list,
)
from layout.coord import DynamicCoord
from layout.layout import Layout
from nn.concat import fused_concat
from nn.gather_scatter import (
    Axis,
    _atomic_add,
    _atomic_reduce,
    ScatterOobIndexStrategy,
    apply_packed_bitmask,
    gather,
    gather_nd,
    gather_nd_shape,
    gather_reduce,
    gather_sum,
    gather_shape,
    _unsafe_normalize_neg_index as normalize_neg_index,
    scatter_elements,
    scatter_elements_shape,
    scatter_nd,
    scatter_nd_generator,
    scatter_nd_shape,
    scatter_set_constant,
)
from nn.index_tensor import index_tensor
from nn.reshape import reshape, reshape_shape
from nn.slice import (
    copy_to_slice,
    slice_shape,
    sliced_add,
)
from nn.shard_and_stack import (
    ShardStackInput,
    ShardStackOutput,
    shard_and_stack,
)
from nn.split import split
from nn.tile import tile, tile_shape
from extensibility import (
    ElementwiseUnaryOp,
    Fused,
    Input,
    LoadAccess,
    Loadable,
    MutableInput,
    Output,
    Storable,
    StoreAccess,
    Tensor,
    TensorTuple,
    TileTensorable,
    UnsafeFusedView,
    unsafe_fused_view,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import _check_tensor_tuple


@inline(.always)
def check_axis_in_range[idx: Int, dim_size: Int]() raises:
    """Indices passed to gather and scatter ops may be negative. This performs
    a check to see if the axis is valid.

    Raises:
        If the index is out of range [-dim_size, dim_size).
    """
    comptime if -dim_size <= idx < dim_size:
        return

    raise Error("indices must be in range [-dim_size, dim_size)")


@inline(.always)
def check_axis_in_range[dim_size: Int](idx: Int) raises:
    """Indices passed to gather and scatter ops may be negative. This performs
    a check to see if the axis is valid.

    Raises:
        If the index is out of range [-dim_size, dim_size).
    """
    if -dim_size <= idx < dim_size:
        return

    raise Error("indices must be in range [-dim_size, dim_size)")


@inline(.always)
def _is_shape_index_listed[
    indices_type: DType
](
    indices: ImmPointer[Scalar[indices_type], _],
    indices_stride: Int,
    num_indices: Int,
    rank: Int,
    position: Int,
) -> Bool:
    for j in range(num_indices):
        if (
            Int(
                normalize_neg_index(
                    indices[unsafe_offset=j * indices_stride], rank
                )
            )
            == position
        ):
            return True
    return False


@inline(.always)
def _check_shape_index[
    indices_type: DType
](
    indices: ImmPointer[Scalar[indices_type], _],
    indices_stride: Int,
    j: Int,
    rank: Int,
):
    var index = Int(
        normalize_neg_index(indices[unsafe_offset=j * indices_stride], rank)
    )
    assert (
        0 <= index < rank
    ), "Shape indices must be between [-r, r-1] where r is the rank they index."
    for k in range(j):
        assert index != Int(
            normalize_neg_index(indices[unsafe_offset=k * indices_stride], rank)
        ), "Duplicate shape indices point to the same dimension."


@inline(.always)
def _check_shape_indices[
    indices_type: DType
](
    indices: ImmPointer[Scalar[indices_type], _],
    indices_stride: Int,
    num_indices: Int,
    rank: Int,
):
    for j in range(num_indices):
        _check_shape_index(indices, indices_stride, j, rank)


@inline(.always)
def _count_listed_at_or_below[
    indices_type: DType
](
    indices: ImmPointer[Scalar[indices_type], _],
    indices_stride: Int,
    num_indices: Int,
    rank: Int,
    position: Int,
) -> Tuple[Int, Int]:
    """Returns how many distinct valid listed positions are below `position`,
    and whether `position` is listed. Out-of-range indices count toward
    neither, and a duplicated index counts once, so the results match the
    sequential CPU loops even when asserts are compiled out."""
    var below = 0
    var equal = 0
    for j in range(num_indices):
        var index = Int(
            normalize_neg_index(indices[unsafe_offset=j * indices_stride], rank)
        )
        if index < 0 or index > position:
            continue
        var first = True
        for k in range(j):
            if (
                Int(
                    normalize_neg_index(
                        indices[unsafe_offset=k * indices_stride], rank
                    )
                )
                == index
            ):
                first = False
                break
        if not first:
            continue
        below += Int(index < position)
        equal += Int(index == position)
    return (below, equal)


def _squeeze_shape[
    dtype: DType, indices_type: DType
](
    output: MutPointer[Scalar[dtype], _],
    output_stride: Int,
    input: ImmPointer[Scalar[dtype], _],
    input_stride: Int,
    remove_indices: ImmPointer[Scalar[indices_type], _],
    remove_indices_stride: Int,
    num_input_dims: Int,
    num_remove_indices: Int,
):
    _check_shape_indices(
        remove_indices,
        remove_indices_stride,
        num_remove_indices,
        num_input_dims,
    )
    var final_rank = num_input_dims - num_remove_indices
    var output_index = 0
    for input_index in range(num_input_dims):
        if _is_shape_index_listed(
            remove_indices,
            remove_indices_stride,
            num_remove_indices,
            num_input_dims,
            input_index,
        ):
            continue
        # Duplicate indices with asserts compiled out would overrun the output.
        if output_index >= final_rank:
            return
        output[unsafe_offset=output_index * output_stride] = input[
            unsafe_offset=input_index * input_stride
        ]
        output_index += 1


def _unsqueeze_shape[
    dtype: DType, indices_type: DType
](
    output: MutPointer[Scalar[dtype], _],
    output_stride: Int,
    input: ImmPointer[Scalar[dtype], _],
    input_stride: Int,
    padding_indices: ImmPointer[Scalar[indices_type], _],
    padding_indices_stride: Int,
    num_input_dims: Int,
    num_padding_indices: Int,
):
    var final_rank = num_input_dims + num_padding_indices
    _check_shape_indices(
        padding_indices, padding_indices_stride, num_padding_indices, final_rank
    )
    var input_index = 0
    for output_index in range(final_rank):
        var dim: Scalar[dtype] = 1
        if not _is_shape_index_listed(
            padding_indices,
            padding_indices_stride,
            num_padding_indices,
            final_rank,
            output_index,
        ):
            # Invalid or duplicate indices with asserts compiled out would
            # read past the input. Skip rather than stop so the listed
            # positions still get their 1, as in the GPU kernel.
            if input_index >= num_input_dims:
                continue
            dim = input[unsafe_offset=input_index * input_stride]
            input_index += 1
        output[unsafe_offset=output_index * output_stride] = dim


# Shape vectors hold a handful of entries, so each thread handles one position
# and scans every index itself: no shared memory, barriers, or warp ops, which
# keeps the kernel correct for any rank and warp size. `Int` is not
# `DevicePassable`, hence the `Int32` arguments.
comptime _SHAPE_BLOCK = 64


@__name(t"mo_squeeze_shape_{dtype}_{indices_type}")
def _squeeze_shape_kernel[
    dtype: DType, indices_type: DType
](
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    output_stride: Int32,
    input: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    input_stride: Int32,
    indices: ImmPointer[Scalar[indices_type], ImmUntrackedOrigin],
    indices_stride: Int32,
    num_input_dims: Int32,
    num_indices: Int32,
):
    var rank = Int(num_input_dims)
    var count = Int(num_indices)
    var i = global_idx.x
    if i >= rank:
        return
    if i < count:
        _check_shape_index(indices, Int(indices_stride), i, rank)
    var counts = _count_listed_at_or_below(
        indices, Int(indices_stride), count, rank, i
    )
    if counts[1] != 0:
        return
    # Duplicate or out-of-range indices with asserts compiled out would
    # misplace the output position.
    var out_pos = i - counts[0]
    if out_pos < 0 or out_pos >= rank - count:
        return
    output[unsafe_offset=out_pos * Int(output_stride)] = input[
        unsafe_offset=i * Int(input_stride)
    ]


@__name(t"mo_unsqueeze_shape_{dtype}_{indices_type}")
def _unsqueeze_shape_kernel[
    dtype: DType, indices_type: DType
](
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    output_stride: Int32,
    input: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    input_stride: Int32,
    indices: ImmPointer[Scalar[indices_type], ImmUntrackedOrigin],
    indices_stride: Int32,
    num_input_dims: Int32,
    num_indices: Int32,
):
    var count = Int(num_indices)
    var rank = Int(num_input_dims) + count
    var o = global_idx.x
    if o >= rank:
        return
    if o < count:
        _check_shape_index(indices, Int(indices_stride), o, rank)
    var counts = _count_listed_at_or_below(
        indices, Int(indices_stride), count, rank, o
    )
    var dim: Scalar[dtype] = 1
    if counts[1] == 0:
        # Duplicate or out-of-range indices with asserts compiled out would
        # read outside the input.
        var in_pos = o - counts[0]
        if in_pos < 0 or in_pos >= Int(num_input_dims):
            return
        dim = input[unsafe_offset=in_pos * Int(input_stride)]
    output[unsafe_offset=o * Int(output_stride)] = dim


def squeeze_shape_gpu[
    dtype: DType, indices_type: DType
](
    output: MutPointer[Scalar[dtype], _],
    output_stride: Int,
    input: ImmPointer[Scalar[dtype], _],
    input_stride: Int,
    remove_indices: ImmPointer[Scalar[indices_type], _],
    remove_indices_stride: Int,
    num_input_dims: Int,
    num_remove_indices: Int,
    ctx: DeviceContext,
) raises:
    """Removes the dimensions at `remove_indices` from a GPU shape vector.

    Parameters:
        dtype: Element type of the input and output shape vectors.
        indices_type: Element type of the indices.

    Args:
        output: The output shape vector, `num_input_dims - num_remove_indices`
            entries long.
        output_stride: Stride between output entries, in elements.
        input: The input shape vector.
        input_stride: Stride between input entries, in elements.
        remove_indices: The dimensions to remove, possibly negative.
        remove_indices_stride: Stride between indices, in elements.
        num_input_dims: Length of the input shape vector.
        num_remove_indices: Number of indices.
        ctx: Device context to launch on.

    Raises:
        If the kernel launch fails.
    """
    ctx.enqueue_function[_squeeze_shape_kernel[dtype, indices_type]](
        output,
        Int32(output_stride),
        input,
        Int32(input_stride),
        remove_indices,
        Int32(remove_indices_stride),
        Int32(num_input_dims),
        Int32(num_remove_indices),
        grid_dim=max(ceildiv(num_input_dims, _SHAPE_BLOCK), 1),
        block_dim=_SHAPE_BLOCK,
    )


def unsqueeze_shape_gpu[
    dtype: DType, indices_type: DType
](
    output: MutPointer[Scalar[dtype], _],
    output_stride: Int,
    input: ImmPointer[Scalar[dtype], _],
    input_stride: Int,
    padding_indices: ImmPointer[Scalar[indices_type], _],
    padding_indices_stride: Int,
    num_input_dims: Int,
    num_padding_indices: Int,
    ctx: DeviceContext,
) raises:
    """Inserts size-1 dimensions at `padding_indices` into a GPU shape vector.

    Parameters:
        dtype: Element type of the input and output shape vectors.
        indices_type: Element type of the indices.

    Args:
        output: The output shape vector, `num_input_dims + num_padding_indices`
            entries long.
        output_stride: Stride between output entries, in elements.
        input: The input shape vector.
        input_stride: Stride between input entries, in elements.
        padding_indices: The output positions of the new dimensions, possibly
            negative.
        padding_indices_stride: Stride between indices, in elements.
        num_input_dims: Length of the input shape vector.
        num_padding_indices: Number of indices.
        ctx: Device context to launch on.

    Raises:
        If the kernel launch fails.
    """
    ctx.enqueue_function[_unsqueeze_shape_kernel[dtype, indices_type]](
        output,
        Int32(output_stride),
        input,
        Int32(input_stride),
        padding_indices,
        Int32(padding_indices_stride),
        Int32(num_input_dims),
        Int32(num_padding_indices),
        grid_dim=max(
            ceildiv(num_input_dims + num_padding_indices, _SHAPE_BLOCK), 1
        ),
        block_dim=_SHAPE_BLOCK,
    )


@extensibility.register("mo.squeeze_shape")
struct SqueezeShape:
    """Registers the `mo.squeeze_shape` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output_shape: Some[Output & Tensor],
        input_shape: Some[Input & Tensor],
        remove_indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Removes the dimensions at `remove_indices` from `input_shape`.

        Raises:
            If the GPU kernel launch fails.
        """
        comptime assert conforms_to(type_of(output_shape), TileTensorable)
        comptime assert conforms_to(type_of(input_shape), TileTensorable)
        comptime assert conforms_to(type_of(remove_indices), TileTensorable)
        comptime dtype = type_of(output_shape).dtype
        comptime indices_type = type_of(remove_indices).dtype
        comptime assert type_of(output_shape).rank == 1
        comptime assert type_of(input_shape).rank == 1
        comptime assert type_of(remove_indices).rank == 1

        var output_tt = output_shape.to_tile_tensor()
        var input_tt = input_shape.to_tile_tensor[dtype]()
        var remove_tt = remove_indices.to_tile_tensor()
        var num_input_dims = Int(input_tt.dim[0]())
        var num_remove_indices = Int(remove_tt.dim[0]())
        assert num_input_dims - num_remove_indices == Int(
            output_tt.dim[0]()
        ), "Incorrect output shape."

        var output_ptr = output_tt.unsafe_ptr()
        var output_stride = Int(output_tt.dynamic_stride(0))
        var input_ptr = input_tt.unsafe_ptr().as_imm()
        var input_stride = Int(input_tt.dynamic_stride(0))
        var indices_ptr = remove_tt.unsafe_ptr().as_imm()
        var indices_stride = Int(remove_tt.dynamic_stride(0))

        comptime if ctx.target.is_cpu():
            _squeeze_shape(
                output_ptr,
                output_stride,
                input_ptr,
                input_stride,
                indices_ptr,
                indices_stride,
                num_input_dims,
                num_remove_indices,
            )
        else:
            squeeze_shape_gpu(
                output_ptr,
                output_stride,
                input_ptr,
                input_stride,
                indices_ptr,
                indices_stride,
                num_input_dims,
                num_remove_indices,
                ctx,
            )


@extensibility.register_shape_function("mo.squeeze_shape")
def squeeze_shape_fn(
    input_shape: Some[Tensor], remove_indices: Some[Tensor]
) raises -> IndexList[1]:
    """Computes the output shape for the `mo.squeeze_shape` graph op."""
    comptime assert type_of(input_shape).rank == 1, "input_shape must be rank 1"
    comptime assert (
        type_of(remove_indices).rank == 1
    ), "remove_indices must be rank 1"
    var out_dim = (
        coord_to_index_list(input_shape.shape())[0]
        - coord_to_index_list(remove_indices.shape())[0]
    )

    if out_dim < 0:
        raise Error(
            "[squeeze_shape] cannot remove more dimensions than there exists"
        )

    return IndexList[1](out_dim)


@extensibility.register("mo.unsqueeze_shape")
struct UnsqueezeShape:
    """Registers the `mo.unsqueeze_shape` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output_shape: Some[Output & Tensor],
        input_shape: Some[Input & Tensor],
        padding_indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Inserts size-1 dimensions into `input_shape` at `padding_indices`.

        Raises:
            If the GPU kernel launch fails.
        """
        comptime assert conforms_to(type_of(output_shape), TileTensorable)
        comptime assert conforms_to(type_of(input_shape), TileTensorable)
        comptime assert conforms_to(type_of(padding_indices), TileTensorable)
        comptime dtype = type_of(output_shape).dtype
        comptime indices_type = type_of(padding_indices).dtype
        comptime assert type_of(output_shape).rank == 1
        comptime assert type_of(input_shape).rank == 1
        comptime assert type_of(padding_indices).rank == 1

        var output_tt = output_shape.to_tile_tensor()
        var input_tt = input_shape.to_tile_tensor[dtype]()
        var padding_tt = padding_indices.to_tile_tensor()
        var num_input_dims = Int(input_tt.dim[0]())
        var num_padding_indices = Int(padding_tt.dim[0]())
        assert num_input_dims + num_padding_indices == Int(
            output_tt.dim[0]()
        ), "Incorrect output shape."

        var output_ptr = output_tt.unsafe_ptr()
        var output_stride = Int(output_tt.dynamic_stride(0))
        var input_ptr = input_tt.unsafe_ptr().as_imm()
        var input_stride = Int(input_tt.dynamic_stride(0))
        var indices_ptr = padding_tt.unsafe_ptr().as_imm()
        var indices_stride = Int(padding_tt.dynamic_stride(0))

        comptime if ctx.target.is_cpu():
            _unsqueeze_shape(
                output_ptr,
                output_stride,
                input_ptr,
                input_stride,
                indices_ptr,
                indices_stride,
                num_input_dims,
                num_padding_indices,
            )
        else:
            unsqueeze_shape_gpu(
                output_ptr,
                output_stride,
                input_ptr,
                input_stride,
                indices_ptr,
                indices_stride,
                num_input_dims,
                num_padding_indices,
                ctx,
            )


@extensibility.register_shape_function("mo.unsqueeze_shape")
def unsqueeze_shape_fn(
    input_shape: Some[Tensor], remove_indices: Some[Tensor]
) -> IndexList[1]:
    """Computes the output shape for the `mo.unsqueeze_shape` graph op."""
    comptime assert type_of(input_shape).rank == 1, "input_shape must be rank 1"
    comptime assert (
        type_of(remove_indices).rank == 1
    ), "remove_indices must be rank 1"
    var out_dim = (
        coord_to_index_list(input_shape.shape())[0]
        + coord_to_index_list(remove_indices.shape())[0]
    )
    return IndexList[1](out_dim)


@extensibility.register("mo.scatter_nd")
struct ScatterND:
    """Registers the `mo.scatter_nd` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        # Existing implementations do not require static shape information
        scatter_nd(
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            context=ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_nd_shape` takes shapes.
@extensibility.register_shape_function("mo.scatter_nd")
def scatter_nd_shape_fn(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter_nd` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    var shape = scatter_nd_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("gather_scatter.scatter_nd.skip_oob_indices")
struct ScatterNDSkipOobIndices:
    """Registers the `gather_scatter.scatter_nd.skip_oob_indices` graph op with the graph compiler.
    """

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        # Unlike mo.scatter_nd, which requires every index to lie in
        # [-dim_size, dim_size), this skips the update for an index outside
        # that range. Negative indices inside it still wrap, so -1 writes the
        # last row; a caller marking rows to skip needs a value below
        # -dim_size, such as the dtype's min.
        scatter_nd_generator[
            oob_index_strategy=ScatterOobIndexStrategy.SKIP,
            atomic_update_fn=None,
            _trace_description="scatter_nd.skip_oob_indices",
        ](
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            context=ctx,
        )


@extensibility.register("mo.scatter_nd.add")
struct ScatterNDAdd:
    """Registers the `mo.scatter_nd.add` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        scatter_nd_generator[
            atomic_update_fn=_atomic_add,
            _trace_description="scatter_nd.add",
        ](
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            context=ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_nd_shape` takes shapes.
@extensibility.register_shape_function("mo.scatter_nd.add")
def scatter_nd_add_shape(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter_nd.add` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    var shape = scatter_nd_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    """Registers the `mo.scatter_nd.mul` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank

        @inline(.always)
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_nd_generator[
            atomic_update_fn=_atomic_reduce[reduce_fn],
            _trace_description="scatter_nd.mul",
        ](
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            context=ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_nd_shape` takes shapes.
@extensibility.register_shape_function("mo.scatter_nd.mul")
def scatter_nd_mul_shape(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter_nd.mul` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    var shape = scatter_nd_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter_nd.min")
struct ScatterNDMin:
    """Registers the `mo.scatter_nd.min` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank

        @inline(.always)
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_nd_generator[
            atomic_update_fn=_atomic_reduce[reduce_fn],
            _trace_description="scatter_nd.min",
        ](
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            context=ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_nd_shape` takes shapes.
@extensibility.register_shape_function("mo.scatter_nd.min")
def scatter_nd_min_shape(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter_nd.min` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    var shape = scatter_nd_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter_nd.max")
struct ScatterNDMax:
    """Registers the `mo.scatter_nd.max` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank

        @inline(.always)
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_nd_generator[
            atomic_update_fn=_atomic_reduce[reduce_fn],
            _trace_description="scatter_nd.max",
        ](
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            context=ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_nd_shape` takes shapes.
@extensibility.register_shape_function("mo.scatter_nd.max")
def scatter_nd_max_shape(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter_nd.max` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    var shape = scatter_nd_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("sampler.apply_packed_bitmask")
struct ApplyPackedBitmask:
    """Registers the `sampler.apply_packed_bitmask` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        //,
    ](
        output: Some[Output & Tensor],
        logits: Some[Input & Tensor],
        packed: Some[Input & Tensor],
        fill_value: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(logits), TileTensorable)
        comptime assert conforms_to(type_of(packed), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(logits).rank == 2
        comptime assert type_of(packed).rank == 2
        apply_packed_bitmask(
            output.to_tile_tensor[dtype](),
            logits.to_tile_tensor[dtype](),
            packed.to_tile_tensor[.int32](),
            fill_value,
            ctx,
        )


@extensibility.register("gather_scatter.scatter_set_constant")
struct ScatterSetConstant:
    """Registers the `gather_scatter.scatter_set_constant` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        data_type: DType,
        //,
    ](
        data: Some[MutableInput & Tensor],
        indices: Some[Input & Tensor],
        fill_value: Scalar[data_type],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(data), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime assert type_of(data).rank == 2
        comptime assert type_of(indices).rank == 2
        scatter_set_constant(
            data.to_tile_tensor[data_type](),
            indices.to_tile_tensor(),
            fill_value,
            ctx,
        )


@extensibility.register("mo.scatter")
struct Scatter:
    """Registers the `mo.scatter` graph op with the graph compiler."""

    @staticmethod
    def execute[
        axis: Int,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        comptime assert type_of(updates).rank == type_of(output).rank
        comptime assert type_of(indices).rank == type_of(output).rank
        check_axis_in_range[type_of(output).rank](axis)

        scatter_elements(
            input.to_tile_tensor[type_of(output).dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[type_of(output).dtype](),
            normalize_neg_index(axis, type_of(output).rank),
            output.to_tile_tensor(),
            ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_elements_shape` takes
# shapes.
@extensibility.register_shape_function("mo.scatter")
def scatter_shape_fn[
    axis: Int,
](
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    comptime assert (
        type_of(updates).rank == type_of(input).rank
    ), "updates rank must match input rank"
    comptime assert (
        type_of(indices).rank == type_of(input).rank
    ), "indices rank must match input rank"
    var shape = scatter_elements_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
        axis,
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter.add")
struct ScatterAdd:
    """Registers the `mo.scatter.add` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        comptime assert type_of(updates).rank == type_of(output).rank
        comptime assert type_of(indices).rank == type_of(output).rank
        check_axis_in_range[type_of(output).rank](Int(axis))

        scatter_elements[atomic_update_fn=_atomic_add](
            input.to_tile_tensor[type_of(output).dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[type_of(output).dtype](),
            normalize_neg_index(Int(axis), type_of(output).rank),
            output.to_tile_tensor(),
            ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_elements_shape` takes
# shapes.
@extensibility.register_shape_function("mo.scatter.add")
def scatter_add_shape_fn(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
    axis: Scalar,
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter.add` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    comptime assert (
        type_of(updates).rank == type_of(input).rank
    ), "updates rank must match input rank"
    comptime assert (
        type_of(indices).rank == type_of(input).rank
    ), "indices rank must match input rank"
    var shape = scatter_elements_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
        Int(axis),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter.max")
struct ScatterMax:
    """Registers the `mo.scatter.max` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        comptime assert type_of(updates).rank == type_of(output).rank
        comptime assert type_of(indices).rank == type_of(output).rank
        check_axis_in_range[type_of(output).rank](Int(axis))

        @inline(.always)
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_elements[atomic_update_fn=_atomic_reduce[reduce_func]](
            input.to_tile_tensor[type_of(output).dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[type_of(output).dtype](),
            normalize_neg_index(Int(axis), type_of(output).rank),
            output.to_tile_tensor(),
            ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_elements_shape` takes
# shapes.
@extensibility.register_shape_function("mo.scatter.max")
def scatter_max_shape_fn(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
    axis: Scalar,
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter.max` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    comptime assert (
        type_of(updates).rank == type_of(input).rank
    ), "updates rank must match input rank"
    comptime assert (
        type_of(indices).rank == type_of(input).rank
    ), "indices rank must match input rank"
    var shape = scatter_elements_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
        Int(axis),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter.min")
struct ScatterMin:
    """Registers the `mo.scatter.min` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        comptime assert type_of(updates).rank == type_of(output).rank
        comptime assert type_of(indices).rank == type_of(output).rank
        check_axis_in_range[type_of(output).rank](Int(axis))

        @inline(.always)
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_elements[atomic_update_fn=_atomic_reduce[reduce_func]](
            input.to_tile_tensor[type_of(output).dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[type_of(output).dtype](),
            normalize_neg_index(Int(axis), type_of(output).rank),
            output.to_tile_tensor(),
            ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_elements_shape` takes
# shapes.
@extensibility.register_shape_function("mo.scatter.min")
def scatter_min_shape_fn(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
    axis: Scalar,
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter.min` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    comptime assert (
        type_of(updates).rank == type_of(input).rank
    ), "updates rank must match input rank"
    comptime assert (
        type_of(indices).rank == type_of(input).rank
    ), "indices rank must match input rank"
    var shape = scatter_elements_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
        Int(axis),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.scatter.mul")
struct ScatterMul:
    """Registers the `mo.scatter.mul` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        updates: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(updates), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        comptime assert type_of(updates).rank == type_of(output).rank
        comptime assert type_of(indices).rank == type_of(output).rank
        check_axis_in_range[type_of(output).rank](Int(axis))

        @inline(.always)
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_elements[atomic_update_fn=_atomic_reduce[reduce_func]](
            input.to_tile_tensor[type_of(output).dtype](),
            indices.to_tile_tensor(),
            updates.to_tile_tensor[type_of(output).dtype](),
            normalize_neg_index(Int(axis), type_of(output).rank),
            output.to_tile_tensor(),
            ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `scatter_elements_shape` takes
# shapes.
@extensibility.register_shape_function("mo.scatter.mul")
def scatter_mul_shape_fn(
    input: Some[TileTensorable],
    updates: Some[TileTensorable],
    indices: Some[TileTensorable],
    axis: Scalar,
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.scatter.mul` graph op."""
    comptime assert (
        type_of(updates).dtype == type_of(input).dtype
    ), "updates dtype must match input dtype"
    comptime assert (
        type_of(updates).rank == type_of(input).rank
    ), "updates rank must match input rank"
    comptime assert (
        type_of(indices).rank == type_of(input).rank
    ), "indices rank must match input rank"
    var shape = scatter_elements_shape(
        input.to_tile_tensor(),
        updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
        indices.to_tile_tensor(),
        Int(axis),
    )
    return rebind[IndexList[type_of(input).rank]](coord_to_index_list(shape))


@extensibility.register("mo.broadcast_to")
struct BroadcastTo:
    """Registers the `mo.broadcast_to` graph op with the graph compiler."""

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(
        input: Some[Input & Tensor], shape: Some[Input & Tensor]
    ) raises:
        raise Error("Should never be called!")

    @staticmethod
    def shape_impl[
        output_rank: Int
    ](input: Some[Tensor], shape: Some[TileTensorable]) raises -> IndexList[
        output_rank
    ]:
        comptime input_rank = type_of(input).rank
        comptime assert type_of(shape).rank == 1, "shape must be rank 1"
        var input_shape = coord_to_index_list(input.shape())
        var shape_tt = shape.to_tile_tensor()
        if output_rank != Int(shape_tt.dim(0)):
            raise Error(
                "[broadcast_to] requires (len(target_shape) == output_rank)"
            )
        if input_rank > output_rank:
            raise Error("[broadcast_to] requires (input_rank <= output_rank)")

        # move the output shape from buffer into a static int tuple
        var output_shape = IndexList[output_rank]()

        for axis in range(output_rank):
            output_shape[axis] = Int(shape_tt[axis])

        # Validate the compatibility between input and output shapes
        # NOTE we don't need to check the padded dims
        for i in range(input_rank):
            var input_axis = input_rank - i - 1
            var output_axis = output_rank - i - 1
            var input_dim = input_shape[input_axis]
            var output_dim = output_shape[output_axis]
            if input_dim != 1 and input_dim != output_dim:
                raise Error(
                    "[broadcast_to] input dimension at index ",
                    input_axis,
                    " (",
                    input_dim,
                    ") must be either 1 or equal to output dimension at index ",
                    output_axis,
                    " (",
                    output_dim,
                    ")",
                )
        return output_shape


@extensibility.register_shape_function("mo.broadcast_to")
def broadcast_to_shape_fn[
    output_rank: Int
](input: Some[Tensor], shape: Some[TileTensorable]) raises -> IndexList[
    output_rank
]:
    """Computes the output shape for the `mo.broadcast_to` graph op."""
    return BroadcastTo.shape_impl[output_rank=output_rank](input, shape)


@extensibility.register("mo.broadcast_shape")
struct BroadcastShape:
    """Registers the `mo.broadcast_shape` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def broadcast_shape_impl(
        out_buf: TileTensor[mut=True, ...],
        lhs_buf: TileTensor[mut=False, ...],
        rhs_buf: TileTensor[mut=False, ...],
    ):
        # Ensure lhs is always the smaller shape
        var lhs_rank = Int(lhs_buf.dim(0))
        var rhs_rank = Int(rhs_buf.dim(0))
        assert lhs_rank <= rhs_rank, "lhs shape must be the smaller one"

        # lhs_buf =      [l0, l1, ...]
        # rhs_buf = [..., r0, r1, ...]
        # out_buf = [..., o0, o1, ...]
        var size_diff = rhs_rank - lhs_rank
        for i in range(size_diff):
            out_buf.store[width=1](
                Coord(i), rhs_buf.load[width=1](Coord(i)).cast[out_buf.dtype]()
            )

        for lhs_idx in range(lhs_rank):
            var rhs_idx = lhs_idx + size_diff
            var lhs_val = lhs_buf.load[width=1](Coord(lhs_idx))
            var rhs_val = rhs_buf.load[width=1](Coord(rhs_idx))
            var lhs_dim = Int(lhs_val)
            var rhs_dim = Int(rhs_val)
            if lhs_dim == rhs_dim:
                out_buf.store[width=1](
                    Coord(rhs_idx), rhs_val.cast[out_buf.dtype]()
                )

            elif lhs_dim != 1 and rhs_dim != 1:
                assert rhs_dim == 1, "one of the differing dimensions must be 1"

            elif lhs_dim != 1:
                out_buf.store[width=1](
                    Coord(rhs_idx), lhs_val.cast[out_buf.dtype]()
                )

            elif rhs_dim != 1:
                out_buf.store[width=1](
                    Coord(rhs_idx), rhs_val.cast[out_buf.dtype]()
                )

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(
        out_buf: Some[Output & Tensor],
        lhs_buf: Some[Input & Tensor],
        rhs_buf: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(out_buf), TileTensorable)
        comptime assert conforms_to(type_of(lhs_buf), TileTensorable)
        comptime assert conforms_to(type_of(rhs_buf), TileTensorable)
        comptime assert type_of(out_buf).rank == 1
        comptime assert type_of(lhs_buf).rank == 1
        comptime assert type_of(rhs_buf).rank == 1
        var out_tt = out_buf.to_tile_tensor()
        var lhs_tt = lhs_buf.to_tile_tensor()
        var rhs_tt = rhs_buf.to_tile_tensor()
        var lhs_size = Int(lhs_tt.dim(0))
        var rhs_size = Int(rhs_tt.dim(0))
        if lhs_size > rhs_size:
            return BroadcastShape.broadcast_shape_impl(out_tt, rhs_tt, lhs_tt)
        return BroadcastShape.broadcast_shape_impl(out_tt, lhs_tt, rhs_tt)


@extensibility.register_shape_function("mo.broadcast_shape")
def broadcast_shape_fn(
    lhs_buf: Some[Tensor], rhs_buf: Some[Tensor]
) raises -> IndexList[1]:
    """Computes the output shape for the `mo.broadcast_shape` graph op."""
    comptime assert type_of(lhs_buf).rank == 1, "lhs_buf must be rank 1"
    comptime assert type_of(rhs_buf).rank == 1, "rhs_buf must be rank 1"
    var lhs_dim = coord_to_index_list(lhs_buf.shape())[0]
    var rhs_dim = coord_to_index_list(rhs_buf.shape())[0]
    return IndexList[1](max(lhs_dim, rhs_dim))


@extensibility.register("mo.reshape")
struct Reshape:
    """Registers the `mo.reshape` graph op with the graph compiler."""

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.reshape` to always simplify to `mo.static.reshape`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(
        input: Some[Input & Tensor], shape: Some[Input & Tensor]
    ) raises:
        raise Error("Should never be called!")


# TODO(GEX-4377): Bind `input` as `Some[Tensor]` once `reshape_shape` takes
# its element count.
@extensibility.register_shape_function("mo.reshape")
def reshape_shape_fn[
    output_rank: Int
](input: Some[TileTensorable], shape: Some[TileTensorable]) raises -> IndexList[
    output_rank
]:
    """Computes the output shape for the `mo.reshape` graph op."""
    comptime assert type_of(shape).rank == 1, "shape must be rank 1"
    return reshape_shape[output_rank=output_rank](
        input.to_tile_tensor(),
        shape.to_tile_tensor(),
    )


# TODO(GEX-1033) Make it possible to have multiple raises.
@extensibility.register_shape_function("mo.transpose")
def transpose_shape_fn(
    input: Some[Tensor],
    permutations: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.transpose` graph op."""
    comptime rank = type_of(input).rank
    comptime assert (
        type_of(permutations).rank == 1
    ), "permutations must be rank 1"
    var perms = permutations.to_tile_tensor()
    if Int(perms.dim(0)) != rank:
        raise Error("[transpose] permutation size must match input rank")

    comptime for i in range(rank):
        var perm = Int(perms[i])
        if perm < 0 or rank <= perm:
            raise Error(
                "[transpose] each permutation must be within range [0, rank)"
            )

    var input_shape = coord_to_index_list(input.shape())
    var out = IndexList[rank]()

    comptime for i in range(rank):
        out[i] = input_shape[Int(perms[i])]

    return out


# TODO(GEX-4377): Bind `input` as `Some[Tensor]` once `slice_shape` takes its
# shape.
@extensibility.register_shape_function("mo.slice")
def slice_shape_fn(
    input: Some[TileTensorable],
    starts: Some[TileTensorable],
    stops: Some[TileTensorable],
    steps: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.slice` graph op."""
    comptime assert type_of(starts).rank == 1, "starts must be rank 1"
    comptime assert type_of(stops).rank == 1, "stops must be rank 1"
    comptime assert type_of(steps).rank == 1, "steps must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        slice_shape(
            input.to_tile_tensor(),
            starts.to_tile_tensor(),
            stops.to_tile_tensor(),
            steps.to_tile_tensor(),
        )
    )


@extensibility.register("mo.mutable.store")
struct MutableStore(ElementwiseUnaryOp):
    """Registers the `mo.mutable.store` graph op with the graph compiler."""

    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return val

    @staticmethod
    def execute[
        _trace_name: StaticString,
    ](
        buffer: Some[MutableInput & Tensor],
        tensor: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        # TODO: Remove the execute method (GEX-2453).
        raise Error("exec should never be called !")


@extensibility.register("mo.mutable.store.slice")
struct MutableStoreSlice:
    """Registers the `mo.mutable.store.slice` graph op with the graph compiler.
    """

    @staticmethod
    def execute(
        to_buffer: Some[MutableInput & Tensor],
        in_slice: Some[Input & Tensor],
        starts: Some[Input & Tensor],
        stops: Some[Input & Tensor],
        steps: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(to_buffer), TileTensorable)
        comptime assert conforms_to(type_of(in_slice), TileTensorable)
        comptime assert conforms_to(type_of(starts), TileTensorable)
        comptime assert conforms_to(type_of(stops), TileTensorable)
        comptime assert conforms_to(type_of(steps), TileTensorable)
        comptime dtype = type_of(to_buffer).dtype
        comptime assert type_of(in_slice).rank == type_of(to_buffer).rank
        comptime assert type_of(starts).rank == 1
        comptime assert type_of(stops).rank == 1
        comptime assert type_of(steps).rank == 1
        copy_to_slice(
            to_buffer.to_tile_tensor[dtype](),
            in_slice.to_tile_tensor[dtype](),
            starts.to_tile_tensor(),
            stops.to_tile_tensor(),
            steps.to_tile_tensor(),
            ctx,
        )


@extensibility.register("mo.gather_nd")
struct GatherND:
    """Registers the `mo.gather_nd` graph op with the graph compiler."""

    @staticmethod
    def execute[
        batchDims: Int,
        _trace_name: StaticString,
    ](
        output: Some[Output & Tensor],
        data: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(data), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        gather_nd[batch_dims=batchDims](
            data.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            output.to_tile_tensor(),
            ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `gather_nd_shape` takes shapes.
@extensibility.register_shape_function("mo.gather_nd")
def gather_nd_shape_fn[
    batch_dims: Int, output_rank: Int
](
    data: Some[TileTensorable], indices: Some[TileTensorable]
) raises -> IndexList[output_rank]:
    """Computes the output shape for the `mo.gather_nd` graph op."""
    var shape = gather_nd_shape[
        batch_dims=batch_dims,
        output_rank=output_rank,
    ](
        data.to_tile_tensor(),
        indices.to_tile_tensor(),
    )
    return rebind[IndexList[output_rank]](coord_to_index_list(shape))


@extensibility.register("mo.gather")
struct Gather:
    """Registers the `mo.gather` graph op with the graph compiler."""

    @staticmethod
    def execute[
        _trace_name: StaticString,
        axis: Int,
    ](
        output: Some[Output & Fused & StoreAccess],
        input: Some[Input & Fused & LoadAccess],
        indices: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime dtype = type_of(output).dtype
        comptime indices_type = type_of(indices).dtype
        comptime input_rank = type_of(input).rank
        comptime indices_rank = type_of(indices).rank
        comptime output_rank = type_of(output).rank
        comptime assert type_of(input).dtype == dtype

        # Project each host proxy to its access-enabled view so the closures
        # can load and store on the GPU lane.
        comptime assert conforms_to(type_of(input), UnsafeFusedView)
        var in_view = unsafe_fused_view(input)
        comptime assert conforms_to(type_of(in_view), Loadable)
        comptime assert conforms_to(type_of(indices), UnsafeFusedView)
        var indices_view = unsafe_fused_view(indices)
        comptime assert conforms_to(type_of(indices_view), Loadable)
        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def input_fn[
            width: Int, element_alignment: Int
        ](coords: Coord) {var in_view} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](
                in_view.load[width=width, element_alignment=element_alignment](
                    coords
                )
            )

        @inline(.always)
        def indices_fn[
            width: Int
        ](coords: Coord) {var indices_view} -> SIMD[indices_type, width]:
            return rebind[SIMD[indices_type, width]](
                indices_view.load[width=width](coords)
            )

        @inline(.always)
        def output_fn[
            width: SIMDLength, element_alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=element_alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](val),
            )

        gather[
            input_rank=input_rank,
            indices_rank=indices_rank,
            output_rank=output_rank,
            dtype=dtype,
            indices_type=indices_type,
        ](
            Axis(axis, input_rank),
            rebind[DynamicCoord[.int64, input_rank]](
                input.shape().make_dynamic[.int64]()
            ),
            rebind[DynamicCoord[.int64, indices_rank]](
                indices.shape().make_dynamic[.int64]()
            ),
            rebind[DynamicCoord[.int64, output_rank]](
                output.shape().make_dynamic[.int64]()
            ),
            input_fn=input_fn,
            indices_fn=indices_fn,
            output_fn=output_fn,
            context=ctx,
        )


# TODO(GEX-4377): Bind `Some[Tensor]` once `gather_shape` takes shapes.
@extensibility.register_shape_function("mo.gather")
def gather_shape_fn[
    output_rank: Int,
    axis: Int,
](
    input: Some[TileTensorable], indices: Some[TileTensorable]
) raises -> IndexList[output_rank]:
    """Computes the output shape for the `mo.gather` graph op."""
    var shape = gather_shape[output_rank=output_rank](
        input.to_tile_tensor(),
        indices.to_tile_tensor(),
        axis,
    )
    return rebind[IndexList[output_rank]](coord_to_index_list(shape))


@extensibility.register("mo.gather_sum")
struct GatherSum:
    """Registers the `mo.gather_sum` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Fused gather (axis 0) + sum reduction (axis 1).

        Dispatches to the CPU `gather_reduce` kernel or the GPU
        `gather_sum` kernel based on `target`.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        gather_sum(
            output.to_tile_tensor(),
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor[.int32](),
            ctx,
        )


@extensibility.register("mo.tile")
struct Tile:
    """Registers the `mo.tile` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        repeats: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(repeats), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        tile(
            input.to_tile_tensor[dtype](),
            repeats.to_tile_tensor(),
            output.to_tile_tensor[dtype](),
            ctx,
        )


# TODO(GEX-4377): Bind `input` as `Some[Tensor]` once `tile_shape` takes its
# shape.
@extensibility.register_shape_function("mo.tile")
def tile_shape_fn(
    input: Some[TileTensorable],
    repeats: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.tile` graph op."""
    comptime assert type_of(repeats).rank == 1, "repeats must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        tile_shape(
            input.to_tile_tensor(),
            repeats.to_tile_tensor(),
        )
    )


@extensibility.register("gather_scatter.shard_and_stack")
struct ShardWeights:
    """Registers the `gather_scatter.shard_and_stack` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        //,
        axis: Int,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        comptime num_outputs = len(Outs)
        comptime num_inputs = len(Ins)
        comptime assert num_inputs > 0, "must have one or more inputs"
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        _check_tensor_tuple[num_inputs, dtype, rank, *Ins]()
        _check_tensor_tuple[num_outputs, dtype, rank + 1, *Outs]()

        def input_tile[i: Int]() {imm inputs} -> ShardStackInput[dtype, rank]:
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            return rebind[ShardStackInput[dtype, rank]](
                inputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )

        def output_tile[
            i: Int
        ]() {imm outputs} -> ShardStackOutput[dtype, rank + 1]:
            comptime assert conforms_to(type_of(outputs[i]), TileTensorable)
            return rebind[ShardStackOutput[dtype, rank + 1]](
                outputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )

        shard_and_stack[dtype, rank, axis](
            Array[_, num_outputs](fill_with_unrolled=output_tile),
            Array[_, num_inputs](fill_with_unrolled=input_tile),
            dev_ctxs_input,
        )


@extensibility.register("mo.concat")
struct Concat:
    """Registers the `mo.concat` graph op with the graph compiler."""

    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        axis: Int,
    ](
        output: Some[Output & Fused & StoreAccess],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime dtype = type_of(output).dtype
        comptime rank = type_of(output).rank
        var input_shapes = StaticTuple[DynamicCoord[.int, rank], len(Ins)]()

        check_axis_in_range[axis, rank]()

        comptime for i in range(len(Ins)):
            comptime assert Ins[i].dtype == dtype, "inputs must share a dtype"
            comptime assert Ins[i].rank == rank, "inputs must share a rank"
            # `fused_concat` threads the input shapes as dynamic `Coord`s. One
            # is built explicitly per input because the tensors' shape types
            # can differ in their static-vs-dynamic split, while the tuple's
            # element type must be uniform.
            var input_shape = DynamicCoord[.int, rank]()
            comptime for d in range(rank):
                input_shape.replace_dynamic[d](Int(inputs[i].shape()[d]))
            input_shapes[i] = input_shape

        # Copy-capture the views into the device-kernel closures. A
        # by-reference capture leaves the GPU kernel holding a host-side
        # pointer to the capture on Metal, so it reads garbage (concat with >=3
        # inputs silently returned all-zeros). The view tuple stores its views
        # by value, so the copy brings their device pointers into the kernel.
        var input_views = TensorTuple._fused_view(inputs)
        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def inputs_lambda[
            input_index: Int, width: Int, alignment: Int = 1
        ](coords: Coord) {var input_views} -> SIMD[dtype, width]:
            comptime assert input_index < len(Ins), "tensor index out of bounds"
            comptime assert conforms_to(
                type_of(input_views[input_index]), Loadable
            )
            return rebind[SIMD[dtype, width]](
                input_views[input_index].load[
                    width=width, element_alignment=alignment
                ](coords)
            )

        @inline(.always)
        def epilogue_wrapper[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](coords: Coord, value: SIMD[_dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](value),
            )

        fused_concat[
            dtype,
            rank,
            axis=normalize_neg_index(axis, rank),
        ](
            input_shapes,
            row_major(output.shape()),
            inputs_lambda,
            epilogue_wrapper,
            ctx,
        )


@extensibility.register_shape_function("mo.concat")
def concat_shape_fn[
    Ins: TypeList[Trait=Tensor, ...],
    //,
    axis: Int,
](inputs: TensorTuple[*Ins]) raises -> IndexList[Ins[0].rank]:
    """Computes the output shape for the `mo.concat` graph op."""
    comptime for i in range(len(Ins)):
        comptime assert (
            Ins[i].dtype == Ins[0].dtype
        ), "inputs must share a dtype"
    return concat_shape_impl(axis, inputs)


@extensibility.register("mo.split")
struct Split:
    """Registers the `mo.split` graph op with the graph compiler."""

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        //,
        _trace_name: StaticString,
        axis: Int,
    ](
        output: TensorTuple[*Outs],
        input: Some[Input & Tensor],
        split_sizes: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime dtype = type_of(input).dtype
        comptime rank = type_of(input).rank
        comptime assert type_of(split_sizes).rank == 1
        comptime for i in range(len(Outs)):
            comptime assert Outs[i].dtype == dtype, "unexpected output dtype"
            comptime assert Outs[i].rank == rank, "unexpected output rank"

        comptime shape_types = DynamicCoord[.int64, rank].element_types
        # Use Scalar for strides as well since make_dynamic produces all
        # runtime strides.
        comptime stride_types = DynamicCoord[.int64, rank].element_types

        check_axis_in_range[rank](axis)

        comptime normalized_axis = axis + rank if axis < 0 else axis

        var output_bufs = StaticTuple[
            TileTensor[
                dtype,
                Layout[shape_types=shape_types, stride_types=stride_types],
                MutUntrackedOrigin,
            ],
            len(Outs),
        ]()

        comptime for i in range(len(Outs)):
            comptime assert conforms_to(type_of(output[i]), TileTensorable)
            var output_tt = output[i].to_tile_tensor[dtype]()
            output_bufs[i] = rebind[output_bufs.element_type](
                TileTensor(
                    output_tt.unsafe_ptr(),
                    output_tt.layout.make_dynamic[.int64](),
                ),
            )

        split[
            dtype,
            trace_description=_trace_name,
            axis=normalized_axis,
        ](
            input.to_tile_tensor[dtype](),
            output_bufs,
            ctx,
        )


@extensibility.register("split_ith_output_shape")
struct SplitOutputShapeHelper:
    """Registers the `split_ith_output_shape` graph op with the graph compiler.
    """

    @staticmethod
    def execute(
        input_buf: Some[Input & Tensor],
        split_sizes_buf: Some[Input & Tensor],
        split_axis: Scalar,
        output_idx: Scalar,
    ) raises:
        raise Error("Should not be called directly.")


@extensibility.register_shape_function("split_ith_output_shape")
def split_ith_output_shape_fn(
    input_buf: Some[Tensor],
    split_sizes_buf: Some[TileTensorable],
    split_axis: Scalar,
    output_idx: Scalar,
) raises -> IndexList[type_of(input_buf).rank]:
    """Computes the output shape for the `split_ith_output_shape` graph op."""
    comptime rank = type_of(input_buf).rank
    comptime assert (
        type_of(split_sizes_buf).rank == 1
    ), "split_sizes_buf must be rank 1"
    var split_sizes = split_sizes_buf.to_tile_tensor()
    var num_splits = Int(split_sizes.dim(0))
    if not (0 <= Int(output_idx) < num_splits):
        raise Error(
            "[split] output index must be within range [0, len(split_sizes))"
        )

    check_axis_in_range[rank](Int(split_axis))

    var output_split_size = Int(split_sizes[Int(output_idx)])

    var normalized_split_axis = normalize_neg_index(Int(split_axis), rank)

    var split_sizes_sum = 0

    for i in range(num_splits):
        split_sizes_sum += Int(split_sizes[i])
    var output_shape = rebind[IndexList[rank]](
        coord_to_index_list(input_buf.shape())
    )
    if split_sizes_sum != output_shape[normalized_split_axis]:
        raise Error(
            "[split] sum of split sizes must match input dimension at split"
            " axis"
        )

    output_shape[normalized_split_axis] = output_split_size
    return output_shape


@extensibility.register("index_tensor")
struct IndexTensor:
    """Registers the `index_tensor` graph op with the graph compiler."""

    @staticmethod
    def execute[
        batch_dims: Int,
    ](
        output: Some[Output & Tensor],
        data: Some[Input & Tensor],
        indices: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(data), TileTensorable)
        comptime assert conforms_to(type_of(indices), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime indices_type = type_of(indices).dtype
        index_tensor[dtype, indices_type, batch_dims](
            data.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("ragged.sliced.add")
struct Struct_sliced_add_ragged:
    """Registers the `ragged.sliced.add` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute(
        c: Some[Output & Tensor],
        a: Some[Input & Tensor],
        b: Some[Input & Tensor],
        lora_end_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(c), TileTensorable)
        comptime assert conforms_to(type_of(a), TileTensorable)
        comptime assert conforms_to(type_of(b), TileTensorable)
        comptime assert conforms_to(type_of(lora_end_idx), TileTensorable)
        comptime dtype = type_of(c).dtype
        comptime assert type_of(c).rank == 2
        comptime assert type_of(a).rank == 2
        comptime assert type_of(b).rank == 2
        comptime assert type_of(lora_end_idx).rank == 1
        var c_tile_tensor = c.to_tile_tensor()
        var a_tile_tensor = a.to_tile_tensor[dtype]()
        var b_tile_tensor = b.to_tile_tensor[dtype]()

        sliced_add(
            c_tile_tensor,
            a_tile_tensor,
            b_tile_tensor,
            lora_end_idx.to_tile_tensor[.int64](),
            context,
        )
