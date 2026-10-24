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
    IntTuple,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
)
from layout.int_tuple import _IntTupleToCoordLike
from layout.coord import DynamicCoord
from layout.tile_layout import Layout as TileLayout
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
from nn.index_tensor import (
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
    index_tensor,
)
from nn.reshape import reshape, reshape_shape
from nn.slice import (
    copy_to_slice,
    slice_as_view,
    slice_shape,
    sliced_add,
)
from nn.shard_and_stack import shard_and_stack
from nn.split import split
from nn.tile import tile, tile_shape
from extensibility import (
    ElementwiseUnaryOp,
    Input,
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    MutableInput,
    Output,
    OutputTensor,
    OutputVariadicTensors,
    Tensor,
    TensorTuple,
    TileTensorable,
)
from builtin_primitives.primitives import (
    _dense_row_major_anchor_stride,
    view_copy_impl,
)
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _SliceStrideTypes,
    _TransposeStrideTypes,
)


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
    indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_stride: Int,
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    input_stride: Int,
    remove_indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_stride: Int,
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    input_stride: Int,
    padding_indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_stride: Int32,
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    input_stride: Int32,
    indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_stride: Int32,
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    input_stride: Int32,
    indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_stride: Int,
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    input_stride: Int,
    remove_indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_stride: Int,
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    input_stride: Int,
    padding_indices: Pointer[Scalar[indices_type], MutAnyOrigin],
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
    def execute[
        target: StaticString,
    ](
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

        var output_ptr = output_tt.ptr.unsafe_origin_cast[MutAnyOrigin]()
        var output_stride = Int(output_tt.dynamic_stride(0))
        # The helpers take mutable pointers but only read the inputs.
        var input_ptr = input_tt.ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        var input_stride = Int(input_tt.dynamic_stride(0))
        var indices_ptr = remove_tt.ptr.unsafe_mut_cast[
            True
        ]().unsafe_origin_cast[MutAnyOrigin]()
        var indices_stride = Int(remove_tt.dynamic_stride(0))

        comptime if is_cpu[target]():
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
    def execute[
        target: StaticString,
    ](
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

        var output_ptr = output_tt.ptr.unsafe_origin_cast[MutAnyOrigin]()
        var output_stride = Int(output_tt.dynamic_stride(0))
        # The helpers take mutable pointers but only read the inputs.
        var input_ptr = input_tt.ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        var input_stride = Int(input_tt.dynamic_stride(0))
        var indices_ptr = padding_tt.ptr.unsafe_mut_cast[
            True
        ]().unsafe_origin_cast[MutAnyOrigin]()
        var indices_stride = Int(padding_tt.dynamic_stride(0))

        comptime if is_cpu[target]():
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
    def execute[
        target: StaticString,
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
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        # Existing implementations do not require static shape information
        scatter_nd[target=target](
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_nd_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
        )
    )


@extensibility.register("mo.scatter_nd.skip_oob_indices")
struct ScatterNDSkipOobIndices:
    """Registers the `mo.scatter_nd.skip_oob_indices` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
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
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        # Unlike mo.scatter_nd, which requires every index to lie in
        # [-dim_size, dim_size), this skips the update for an index outside
        # that range. Negative indices inside it still wrap, so -1 writes the
        # last row; a caller marking rows to skip needs a value below
        # -dim_size, such as the dtype's min.
        scatter_nd_generator[
            oob_index_strategy=ScatterOobIndexStrategy.SKIP,
            target=target,
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
    def execute[
        target: StaticString,
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
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        scatter_nd_generator[
            target=target,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_nd_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
        )
    )


@extensibility.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    """Registers the `mo.scatter_nd.mul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
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
            target=target,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_nd_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
        )
    )


@extensibility.register("mo.scatter_nd.min")
struct ScatterNDMin:
    """Registers the `mo.scatter_nd.min` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
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
            target=target,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_nd_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
        )
    )


@extensibility.register("mo.scatter_nd.max")
struct ScatterNDMax:
    """Registers the `mo.scatter_nd.max` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
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
            target=target,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_nd_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
        )
    )


@extensibility.register("mo.apply_packed_bitmask")
struct ApplyPackedBitmask:
    """Registers the `mo.apply_packed_bitmask` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
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
        apply_packed_bitmask[target](
            output.to_tile_tensor[dtype](),
            logits.to_tile_tensor[dtype](),
            packed.to_tile_tensor[.int32](),
            fill_value,
            ctx,
        )


@extensibility.register("mo.scatter_set_constant")
struct ScatterSetConstant:
    """Registers the `mo.scatter_set_constant` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        data_type: DType,
        //,
        target: StaticString,
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
        scatter_set_constant[target](
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
        target: StaticString,
        axis: Int,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](axis)

        scatter_elements[target=target](
            input,
            indices,
            updates,
            normalize_neg_index(axis, output.rank),
            output,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_elements_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
            axis,
        )
    )


@extensibility.register("mo.scatter.add")
struct ScatterAdd:
    """Registers the `mo.scatter.add` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        scatter_elements[target=target, atomic_update_fn=_atomic_add](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_elements_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
            Int(axis),
        )
    )


@extensibility.register("mo.scatter.max")
struct ScatterMax:
    """Registers the `mo.scatter.max` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @inline(.always)
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_elements[
            target=target, atomic_update_fn=_atomic_reduce[reduce_func]
        ](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_elements_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
            Int(axis),
        )
    )


@extensibility.register("mo.scatter.min")
struct ScatterMin:
    """Registers the `mo.scatter.min` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @inline(.always)
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_elements[
            target=target, atomic_update_fn=_atomic_reduce[reduce_func]
        ](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_elements_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
            Int(axis),
        )
    )


@extensibility.register("mo.scatter.mul")
struct ScatterMul:
    """Registers the `mo.scatter.mul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @inline(.always)
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_elements[
            target=target, atomic_update_fn=_atomic_reduce[reduce_func]
        ](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
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
    return rebind[IndexList[type_of(input).rank]](
        scatter_elements_shape(
            input.to_tile_tensor(),
            updates.to_tile_tensor().bitcast[type_of(input).dtype]().as_imm(),
            indices.to_tile_tensor(),
            Int(axis),
        )
    )


@extensibility.register("mo.broadcast_to")
struct BroadcastTo:
    """Registers the `mo.broadcast_to` graph op with the graph compiler."""

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(input: InputTensor, shape: InputTensor) raises:
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
        out_buf: ManagedTensorSlice[rank=1, ...],
        lhs_buf: ManagedTensorSlice[rank=1, ...],
        rhs_buf: ManagedTensorSlice[rank=1, ...],
    ):
        # Ensure lhs is always the smaller shape
        var lhs_rank = lhs_buf.size()
        var rhs_rank = rhs_buf.size()
        assert lhs_rank <= rhs_rank, "lhs shape must be the smaller one"

        # lhs_buf =      [l0, l1, ...]
        # rhs_buf = [..., r0, r1, ...]
        # out_buf = [..., o0, o1, ...]
        var size_diff = rhs_rank - lhs_rank
        for i in range(size_diff):
            out_buf[i] = rhs_buf[i].cast[out_buf.dtype]()

        for lhs_idx in range(lhs_rank):
            var rhs_idx = lhs_idx + size_diff
            var lhs_dim = Int(lhs_buf[lhs_idx])
            var rhs_dim = Int(rhs_buf[rhs_idx])
            if lhs_dim == rhs_dim:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.dtype]()

            elif lhs_dim != 1 and rhs_dim != 1:
                assert rhs_dim == 1, "one of the differing dimensions must be 1"

            elif lhs_dim != 1:
                out_buf[rhs_idx] = lhs_buf[lhs_idx].cast[out_buf.dtype]()

            elif rhs_dim != 1:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.dtype]()

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(
        out_buf: OutputTensor[rank=1, ...],
        lhs_buf: InputTensor[rank=1, ...],
        rhs_buf: InputTensor[rank=1, ...],
    ):
        var lhs_size = lhs_buf.size()
        var rhs_size = rhs_buf.size()
        if lhs_size > rhs_size:
            return BroadcastShape.broadcast_shape_impl(
                out_buf, rhs_buf, lhs_buf
            )
        return BroadcastShape.broadcast_shape_impl(out_buf, lhs_buf, rhs_buf)


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


@extensibility.register("mo.static.broadcast_to")
struct StaticBroadcastTo:
    """Registers the `mo.static.broadcast_to` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def build_view[
        out_rank: Int,
    ](x: InputTensor) -> IndexList[out_rank]:
        var new_strides = IndexList[out_rank]()
        comptime delta = out_rank - x.rank

        comptime for i in range(out_rank):
            comptime if i < delta:
                new_strides[i] = 0
            else:
                if x.dim_size[i - delta]() <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = x.stride_length[i - delta]()

        return new_strides

    @staticmethod
    def get_view_strides_list[
        out_rank: Int,
        in_rank: Int,
        input_shape: IntTuple,
        input_strides: IntTuple,
    ]() -> IndexList[out_rank]:
        var new_strides = IndexList[out_rank]()
        comptime delta = out_rank - in_rank

        comptime for i in range(out_rank):
            comptime if i < delta:
                new_strides[i] = 0
            else:
                if Int(input_shape[i - delta]) == UNKNOWN_VALUE:
                    new_strides[i] = -1
                elif Int(input_shape[i - delta]) <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = Int(input_strides[i - delta])
        return new_strides

    @staticmethod
    def input_view[
        dtype: DType,
        in_rank: Int,
        out_rank: Int,
        //,
        output_static_shape: IntTuple,
    ](
        x: InputTensor[dtype=dtype, rank=in_rank, ...],
        output_shape: IndexList[out_rank],
        out result: InputTensor[
            static_spec=x.static_spec.with_int_tuple_layout[
                out_rank,
                output_static_shape,
                Self.get_view_strides_list[
                    out_rank,
                    x.rank,
                    x._static_shape_tuple,
                    x._static_strides_tuple,
                ](),
            ]()
        ],
    ):
        var x_runtime_strides = Self.build_view[out_rank](x)
        return {x.unsafe_ptr(), output_shape, x_runtime_strides}

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        in_rank: Int,
        out_rank: Int,
        _trace_name: StaticString,
    ](
        z: OutputTensor[dtype=dtype, rank=out_rank, ...],
        x: InputTensor[dtype=dtype, rank=in_rank, ...],
        output_shape: IndexList[out_rank],
        ctx: DeviceContext,
    ) raises:
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.

        var x_view = Self.input_view[z._static_shape_tuple](x, output_shape)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](z, x_view, ctx)


@extensibility.register("mo.static.reshape")
struct StaticReshape:
    """Registers the `mo.static.reshape` graph op with the graph compiler."""

    @staticmethod
    def input_view[
        dtype: DType,
        output_rank: Int,
        //,
        output_static_shape: IntTuple,
    ](
        input: InputTensor[dtype=dtype, ...],
        shape: IndexList[output_rank],
        out result: InputTensor[
            static_spec=input.static_spec.with_row_major_int_tuple_layout[
                output_rank, output_static_shape
            ]()
        ],
    ):
        var view_buffer = reshape(
            input.to_tile_tensor[.int64](),
            shape,
        )

        return {
            view_buffer._storage,
            rebind[IndexList[output_rank]](
                coord_to_index_list(view_buffer.layout.shape_coord())
            ),
            rebind[IndexList[output_rank]](
                coord_to_index_list(view_buffer.layout.stride_coord())
            ),
        }

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        output_rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=output_rank, ...],
        input: InputTensor[dtype=dtype, ...],
        shape: IndexList[output_rank],
        ctx: DeviceContext,
    ) raises:
        var view_tensor = Self.input_view[output._static_shape_tuple](
            input, shape
        )

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)


@extensibility.register("mo.reshape")
struct Reshape:
    """Registers the `mo.reshape` graph op with the graph compiler."""

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.reshape` to always simplify to `mo.static.reshape`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(input: InputTensor, shape: InputTensor) raises:
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


@extensibility.register("mo.transpose")
struct Transpose:
    """Registers the `mo.transpose` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def transpose_in_place(
        input: InputTensor,
        permutations: InputTensor[rank=1, ...],
        out result: Tuple[IndexList[input.rank], IndexList[input.rank]],
    ):
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        comptime for i in range(input.rank):
            var dim = Int(permutations[i])
            new_shape[i] = input.dim_size(dim)
            new_stride[i] = input.stride_length(dim)

        return {new_shape, new_stride}

    @staticmethod
    def input_view[
        dtype: DType,
        rank: Int,
        //,
        output_static_shape: IntTuple,
        static_permutations: IntTuple,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        permutations: InputTensor[rank=1, ...],
        out result: InputTensor[
            static_spec=input.static_spec.with_tile_layout[
                rank,
                TileLayout[
                    shape_types=_IntTupleToCoordLike[
                        DType.int, output_static_shape
                    ],
                    stride_types=_TransposeStrideTypes[
                        static_permutations,
                        rank,
                        input.static_spec.static_layout._stride_types,
                    ],
                ],
            ]()
        ],
    ):
        var shape, strides = Self.transpose_in_place(input, permutations)
        return {input.unsafe_ptr(), shape, strides}

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        static_permutations: IntTuple,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        permutations: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var view = Self.input_view[
            output._static_shape_tuple, static_permutations
        ](input, permutations)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view, ctx)

    # TODO(GEX-1033) Make it possible to have multiple raises.
    @inline(.never)
    @staticmethod
    def shape_impl(
        input: Some[Tensor],
        permutations: Some[TileTensorable],
    ) raises -> IndexList[type_of(input).rank]:
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
                    "[transpose] each permutation must be within range [0,"
                    " rank)"
                )

        var input_shape = coord_to_index_list(input.shape())
        var out = IndexList[rank]()

        comptime for i in range(rank):
            out[i] = input_shape[Int(perms[i])]

        return out


@extensibility.register_shape_function("mo.transpose")
def transpose_shape_fn(
    input: Some[Tensor],
    permutations: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.transpose` graph op."""
    return Transpose.shape_impl(input, permutations)


@extensibility.register("mo.slice")
struct Slice:
    """Registers the `mo.slice` graph op with the graph compiler."""

    @staticmethod
    def get_view_alignment[
        rank: Int,
        dtype: DType,
        input_shape: IntTuple,
        input_strides: IntTuple,
        static_starts: IntTuple,
        static_steps: IntTuple,
    ](input_alignment: Int) -> Int:
        # Convert IntTuples to CoordLike types at the MLIR level, then
        # use type-level access (no interpreter heap allocation).
        comptime stride_types = _IntTupleToCoordLike[.int, input_strides]
        comptime start_types = _IntTupleToCoordLike[.int, static_starts]
        comptime step_types = _IntTupleToCoordLike[.int, static_steps]

        # A stride that is only known at runtime still moves the pointer by a
        # whole multiple of this one, so it bounds the alignment the same way a
        # static stride does. 0 when nothing can be established.
        comptime anchor = _dense_row_major_anchor_stride(
            rank, input_shape, input_strides
        )

        var alignment = input_alignment
        comptime for i in range(rank):
            # Bail if step is unknown or negative.
            comptime if not step_types[i].is_static_value or step_types[
                i
            ].static_value < 0:
                return 1

            # The offset for dimension `i` is `start[i] * strides[i]`. If the
            # start is not statically known the offset is unknown.
            comptime if not start_types[i].is_static_value:
                return 1

            comptime if start_types[i].static_value != 0:
                comptime if stride_types[i].is_static_value:
                    alignment = gcd(
                        alignment,
                        start_types[i].static_value
                        * stride_types[i].static_value
                        * align_of[dtype](),
                    )
                else:
                    comptime if anchor == 0:
                        return 1
                    alignment = gcd(
                        alignment,
                        start_types[i].static_value
                        * anchor
                        * align_of[dtype](),
                    )

            # Stepping along a non-innermost dimension moves the pointer by
            # `step[i] * strides[i]` elements, so that stride bounds the
            # alignment.
            comptime if i != rank - 1:
                comptime if stride_types[i].is_static_value:
                    alignment = gcd(
                        alignment,
                        step_types[i].static_value
                        * stride_types[i].static_value
                        * align_of[dtype](),
                    )
                else:
                    comptime if anchor == 0:
                        return 1
                    alignment = gcd(
                        alignment,
                        step_types[i].static_value * anchor * align_of[dtype](),
                    )

        return alignment

    @staticmethod
    def input_view[
        dtype: DType,
        rank: Int,
        //,
        output_static_shape: IntTuple,
        static_starts: IntTuple,
        static_steps: IntTuple,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        starts: InputTensor[rank=1, ...],
        stops: InputTensor[rank=1, ...],
        steps: InputTensor[rank=1, ...],
        out result: InputTensor[
            static_spec=input.static_spec.with_tile_layout_and_alignment[
                rank,
                TileLayout[
                    shape_types=_IntTupleToCoordLike[
                        DType.int, output_static_shape
                    ],
                    stride_types=_SliceStrideTypes[
                        rank,
                        input.static_spec.static_layout._stride_types,
                        _IntTupleToCoordLike[.int, static_steps],
                    ],
                ],
            ](
                Self.get_view_alignment[
                    rank,
                    dtype,
                    input._static_shape_tuple,
                    input._static_strides_tuple,
                    static_starts,
                    static_steps,
                ](input.alignment),
            )
        ],
    ):
        var view_buffer = slice_as_view(
            input.to_tile_tensor[.int64](),
            starts.to_tile_tensor[.int64](),
            stops.to_tile_tensor[.int64](),
            steps.to_tile_tensor[.int64](),
        )

        result = {
            view_buffer._storage,
            rebind[IndexList[rank]](
                coord_to_index_list(view_buffer.layout.shape_coord())
            ),
            rebind[IndexList[rank]](
                coord_to_index_list(view_buffer.layout.stride_coord())
            ),
        }

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        static_starts: IntTuple,
        static_steps: IntTuple,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        starts: InputTensor[rank=1, ...],
        stops: InputTensor[rank=1, ...],
        steps: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var view_tensor = Self.input_view[
            output._static_shape_tuple, static_starts, static_steps
        ](input, starts, stops, steps)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)


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
        target: StaticString,
        _trace_name: StaticString,
    ](
        buffer: MutableInputTensor,
        tensor: FusedInputTensor,
        ctx: DeviceContext,
    ) raises:
        # TODO: Remove the execute method (GEX-2453).
        raise Error("exec should never be called !")


@extensibility.register("mo.mutable.store.slice")
struct MutableStoreSlice:
    """Registers the `mo.mutable.store.slice` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
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
        copy_to_slice[target=target](
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
        target: StaticString,
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
        gather_nd[batch_dims=batchDims, target=target](
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
    return gather_nd_shape[
        batch_dims=batch_dims,
        output_rank=output_rank,
    ](
        data.to_tile_tensor(),
        indices.to_tile_tensor(),
    )


@extensibility.register("mo.gather")
struct Gather:
    """Registers the `mo.gather` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, ...],
        indices: FusedInputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def input_fn[
            width: Int, element_alignment: Int
        ](coords: Coord) {var input} -> SIMD[output.dtype, width]:
            return input._lambda_load[
                width=width, element_alignment=element_alignment
            ](coords)

        @inline(.always)
        def indices_fn[
            width: Int
        ](coords: Coord) {var indices} -> SIMD[indices.dtype, width]:
            return indices._fused_load[width=width](coords)

        @inline(.always)
        def output_fn[
            width: SIMDLength, element_alignment: Int
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            output._lambda_store[
                width=width, element_alignment=element_alignment
            ](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        gather[
            output.dtype,
            indices.dtype,
            target=target,
        ](
            Axis(axis, input.rank),
            input.shape(),
            indices.shape(),
            output.shape(),
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
    return gather_shape[output_rank=output_rank](
        input.to_tile_tensor(),
        indices.to_tile_tensor(),
        axis,
    )


@extensibility.register("mo.gather_sum")
struct GatherSum:
    """Registers the `mo.gather_sum` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
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
        gather_sum[target=target](
            output.to_tile_tensor(),
            input.to_tile_tensor[dtype](),
            indices.to_tile_tensor[.int32](),
            ctx,
        )


@extensibility.register("mo.tile")
struct Tile:
    """Registers the `mo.tile` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
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
        tile[target=target](
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


@extensibility.register("mo.shard_and_stack")
struct ShardWeights:
    """Registers the `mo.shard_and_stack` graph op with the graph compiler."""

    @staticmethod
    def execute[
        axis: Int,
    ](
        outputs: OutputVariadicTensors,
        inputs: InputVariadicTensors[
            dtype=outputs.dtype,
            rank=outputs.rank - 1,
            ...,
        ],
        dev_ctxs_input: DeviceContextArray,
    ) raises:
        shard_and_stack[axis](outputs, inputs, dev_ctxs_input)


@extensibility.register("mo.concat")
struct Concat:
    """Registers the `mo.concat` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        var input_shapes = StaticTuple[IndexList[rank], inputs.size]()

        check_axis_in_range[axis, output.rank]()

        comptime for i in range(inputs.size):
            input_shapes[i] = inputs[i].shape()

        # Copy-capture `inputs` into the device-kernel closure. A by-reference
        # capture leaves the GPU kernel holding a host-side pointer to the
        # capture on Metal, so it reads garbage (concat with >=3 inputs
        # silently returned all-zeros). `inputs` (`FusedInputVariadicTensors`)
        # stores its tensors by value, so the copy brings their device pointers
        # into the kernel.
        @inline(.always)
        def inputs_lambda[
            input_index: Int, width: Int, alignment: Int = 1
        ](coords: Coord) {var inputs} -> SIMD[dtype, width]:
            comptime assert (
                input_index < inputs.size
            ), "tensor index out of bounds"
            return inputs[input_index]._lambda_load[
                width=width, element_alignment=alignment
            ](coords)

        # Copy-capture `output` for the same reason as `inputs` above: a
        # by-reference capture leaves the device kernel holding a host pointer
        # to the output tensor on Metal.
        @inline(.always)
        def epilogue_wrapper[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](coords: Coord, value: SIMD[_dtype, width]) {var output}:
            output._lambda_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](value),
            )

        fused_concat[
            dtype,
            rank,
            axis=normalize_neg_index(axis, rank),
            target=target,
        ](
            input_shapes,
            output.runtime_layout(),
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
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        axis: Int,
    ](
        output: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        split_sizes: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime shape_types = DynamicCoord[.int64, rank].element_types
        # Use Scalar for strides as well since make_dynamic produces all
        # runtime strides.
        comptime stride_types = DynamicCoord[.int64, rank].element_types

        check_axis_in_range[output.rank](axis)

        comptime normalized_axis = axis + rank if axis < 0 else axis

        var output_bufs = StaticTuple[
            TileTensor[
                output.dtype,
                TileLayout[shape_types=shape_types, stride_types=stride_types],
                MutAnyOrigin,
            ],
            output.size,
        ]()

        comptime for i in range(output.size):
            output_bufs[i] = rebind[output_bufs.element_type](
                TileTensor(
                    output[i].unsafe_ptr().as_unsafe_any_origin(),
                    output[i]
                    .to_tile_tensor[.int64]()
                    .layout.make_dynamic[.int64](),
                ),
            )

        split[
            dtype,
            target=target,
            trace_description=_trace_name,
            axis=normalized_axis,
        ](
            input.to_tile_tensor[.int64](),
            output_bufs,
            ctx,
        )


@extensibility.register("split_ith_output_shape")
struct SplitOutputShapeHelper:
    """Registers the `split_ith_output_shape` graph op with the graph compiler.
    """

    @staticmethod
    def execute(
        input_buf: InputTensor,
        split_sizes_buf: InputTensor,
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
        target: StaticString,
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
        index_tensor[dtype, indices_type, batch_dims, target=target](
            data.to_tile_tensor[dtype](),
            indices.to_tile_tensor(),
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("advanced_indexing_getitem")
struct AdvancedIndexingGetItem:
    """Registers the `advanced_indexing_getitem` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        Indices: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tensor: Some[Output & Tensor],
        input_tensor: Some[Input & Fused & LoadAccess],
        indices: TensorTuple[*Indices],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_tensor), TileTensorable)
        comptime input_rank = type_of(input_tensor).rank
        comptime index_rank = Indices[0].rank
        comptime num_index_tensors = len(Indices)
        _check_index_tensors[*Indices]()
        comptime assert type_of(input_tensor).dtype == type_of(out_tensor).dtype
        comptime assert (
            type_of(out_tensor).rank
            == input_rank + index_rank - num_index_tensors
        )
        # Do not support boolean masks at this time.
        comptime assert Indices[0].dtype != .bool

        comptime assert conforms_to(type_of(input_tensor), UnsafeFusedView)
        var in_view = unsafe_fused_view(input_tensor)
        comptime assert conforms_to(type_of(in_view), Loadable)

        @inline(.always)
        def input_tensor_fn[
            dtype: DType, width: Int
        ](idx: Coord) {var in_view} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](in_view.load[width](idx))

        var index_views = TensorTuple._fused_view(indices)

        @inline(.always)
        def indices_fn[
            indices_index: Int,
        ](coordinates: Coord) {var index_views} -> Int:
            comptime assert (
                indices_index < num_index_tensors
            ), "tensor index out of bounds"
            comptime assert conforms_to(
                type_of(index_views[indices_index]), Loadable
            )
            return Int(index_views[indices_index].load[width=1](coordinates))

        advanced_indexing_getitem[
            input_rank=input_rank,
            index_rank=index_rank,
            start_axis=start_axis,
            num_index_tensors=num_index_tensors,
            target=target,
            trace_description=_trace_name,
        ](
            out_tensor.to_tile_tensor(),
            _fused_row_major_strides(input_tensor),
            ctx,
            input_tensor_fn,
            indices_fn,
        )


@extensibility.register_shape_function("advanced_indexing_getitem")
def advanced_indexing_getitem_shape_fn[
    Indices: TypeList[Trait=Tensor, ...],
    //,
    start_axis: Int,
](input_tensor: Some[Tensor], indices: TensorTuple[*Indices]) -> IndexList[
    type_of(input_tensor).rank + Indices[0].rank - Indices.length
]:
    """Computes the output shape for the `advanced_indexing_getitem` graph op.
    """
    comptime input_rank = type_of(input_tensor).rank
    comptime index_rank = Indices[0].rank
    comptime for i in range(Indices.length):
        comptime assert (
            Indices[i].dtype == Indices[0].dtype
        ), "indices must share a dtype"
        comptime assert (
            Indices[i].rank == index_rank
        ), "indices must share a rank"
    return advanced_indexing_getitem_shape[
        start_axis=start_axis, num_index_tensors=Indices.length
    ](
        rebind[IndexList[input_rank]](
            coord_to_index_list(input_tensor.shape())
        ),
        rebind[IndexList[index_rank]](coord_to_index_list(indices[0].shape())),
    )


def _check_index_tensors[*Indices: Tensor]():
    """Checks that an advanced-indexing op's index tensors share a dtype and
    rank."""
    comptime for i in range(len(Indices)):
        comptime assert (
            Indices[i].dtype == Indices[0].dtype
        ), "indices must share a dtype"
        comptime assert (
            Indices[i].rank == Indices[0].rank
        ), "indices must share a rank"


def _fused_row_major_strides(
    tensor: Some[Tensor],
) -> IndexList[type_of(tensor).rank]:
    """Returns the strides a fused tensor argument reads or writes with.

    A fused argument has no buffer of its own. MOGG lays its producer or
    consumer out row-major over its shape, as it does for the legacy fused
    slices.
    """
    return rebind[IndexList[type_of(tensor).rank]](
        coord_to_index_list(row_major(tensor.shape()).stride_coord())
    )


@inline(.always)
def _advanced_indexing_setitem_inplace[
    Indices: TypeList[Trait=Input & Fused & LoadAccess, ...],
    //,
    start_axis: Int,
    target: StaticString,
    trace_description: StaticString,
](
    input_tile: TileTensor[mut=True, ...],
    updates: Some[Input & Fused & LoadAccess],
    indices: TensorTuple[*Indices],
    ctx: DeviceContext,
) raises:
    comptime index_rank = Indices[0].rank
    comptime num_index_tensors = len(Indices)
    _check_index_tensors[*Indices]()
    comptime assert type_of(updates).dtype == input_tile.dtype

    comptime assert conforms_to(type_of(updates), UnsafeFusedView)
    var updates_view = unsafe_fused_view(updates)
    comptime assert conforms_to(type_of(updates_view), Loadable)

    @inline(.always)
    def updates_tensor_fn[
        dtype: DType, width: Int
    ](idx: Coord) {var updates_view} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](updates_view.load[width](idx))

    var index_views = TensorTuple._fused_view(indices)

    @inline(.always)
    def indices_fn[
        indices_index: Int,
    ](coordinates: Coord) {var index_views} -> Int:
        comptime assert (
            indices_index < num_index_tensors
        ), "tensor index out of bounds"
        comptime assert conforms_to(
            type_of(index_views[indices_index]), Loadable
        )
        return Int(index_views[indices_index].load[width=1](coordinates))

    advanced_indexing_setitem_inplace[
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target=target,
        trace_description=trace_description,
    ](
        input_tile,
        rebind[IndexList[index_rank]](coord_to_index_list(indices[0].shape())),
        _fused_row_major_strides(updates),
        ctx,
        updates_tensor_fn,
        indices_fn,
    )


@extensibility.register("advanced_indexing_setitem_inplace")
struct AdvancedIndexingSetItemInplace:
    """Registers the `advanced_indexing_setitem_inplace` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        Indices: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        input_tensor: Some[MutableInput & Tensor],
        updates: Some[Input & Fused & LoadAccess],
        indices: TensorTuple[*Indices],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(input_tensor), TileTensorable)
        _advanced_indexing_setitem_inplace[
            start_axis=start_axis,
            target=target,
            trace_description=_trace_name,
        ](input_tensor.to_tile_tensor(), updates, indices, ctx)


@extensibility.register("advanced_indexing_setitem")
struct AdvancedIndexingSetItem:
    """Registers the `advanced_indexing_setitem` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        Indices: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tensor: Some[Output & Tensor],
        input_tensor: Some[Input & Fused & LoadAccess],
        updates: Some[Input & Fused & LoadAccess],
        indices: TensorTuple[*Indices],
        ctx: DeviceContext,
    ) raises:
        """Implement basic numpy-style advanced indexing with assignment but returns a copy.
        """
        comptime assert conforms_to(type_of(output_tensor), TileTensorable)
        comptime dtype = type_of(output_tensor).dtype
        comptime assert type_of(input_tensor).dtype == dtype
        comptime assert (
            type_of(input_tensor).rank == type_of(output_tensor).rank
        )

        var output_tile = output_tensor.to_tile_tensor()
        comptime assert conforms_to(type_of(input_tensor), UnsafeFusedView)
        var in_view = unsafe_fused_view(input_tensor)
        comptime assert conforms_to(type_of(in_view), Loadable)

        # First copy over input tensor into the output
        @inline(.always)
        def copy_input[
            width: Int, element_alignment: Int = 1
        ](index: Coord) {var in_view, var output_tile}:
            output_tile.store[
                width=width,
                alignment=min(
                    type_of(output_tensor).alignment,
                    element_alignment * align_of[dtype](),
                ),
            ](
                index,
                rebind[SIMD[dtype, width]](
                    in_view.load[
                        width=width, element_alignment=element_alignment
                    ](index)
                ),
            )

        elementwise[
            get_kernel_simd_width[dtype, target](),
            target=target,
            _trace_description=_trace_name + "_p1/2_copy",
        ](copy_input, Coord(output_tensor.shape()).flatten(), ctx)

        # Then run the updates in-place.
        _advanced_indexing_setitem_inplace[
            start_axis=start_axis,
            target=target,
            trace_description=_trace_name + "_p2/2_update",
        ](output_tile, updates, indices, ctx)


@extensibility.register("mo.sliced.add.ragged")
struct Struct_sliced_add_ragged:
    """Registers the `mo.sliced.add.ragged` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=dtype, rank=2, ...],
        a: InputTensor[dtype=dtype, rank=2, ...],
        b: InputTensor[dtype=dtype, rank=2, ...],
        lora_end_idx: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var c_tile_tensor = c.to_tile_tensor[.int64]()
        var a_tile_tensor = a.to_tile_tensor[.int64]()
        var b_tile_tensor = b.to_tile_tensor[.int64]()

        sliced_add[target=target](
            c_tile_tensor,
            a_tile_tensor,
            b_tile_tensor,
            lora_end_idx.to_tile_tensor[.int64](),
            context,
        )
