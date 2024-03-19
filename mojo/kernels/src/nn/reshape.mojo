# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from register import mogg_register, mogg_view_op

from utils.index import StaticIntTuple
from buffer.list import DimList
from utils.loop import unroll


# Reshape assumes inputs are contiguous. It should always be fused last and
# a non-contiguous tensor cannot be fused *into* this as input.
@mogg_register("mo.static.reshape")
@mogg_view_op
@always_inline
fn reshape[
    rank: Int,
    output_rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    new_shape: StaticIntTuple[output_rank],
) -> NDBuffer[type, output_rank]:
    var stride_tuple = StaticIntTuple[output_rank]()
    var stride: Int = 1

    # Create contiguous strides.
    @always_inline
    @parameter
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        var i = output_rank - 1 - idx
        stride_tuple[i] = stride
        stride *= new_shape[i]

    unroll[body, output_rank]()

    # Return the a view with the new shape.
    return NDBuffer[type, output_rank](input.data, new_shape, stride_tuple)


@mogg_register("ndbuffer_reshape")
@mogg_view_op
@always_inline
fn ndbuffer_reshape[
    rank: Int,
    output_rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    new_shape: StaticIntTuple[output_rank],
) -> NDBuffer[type, output_rank]:
    return reshape[rank, output_rank, type, single_thread_blocking_override](
        input, new_shape
    )


@always_inline
fn reshape_shape[
    input_rank: Int,
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    target_shape_buf: NDBuffer[target_shape_type, 1],
) raises -> StaticIntTuple[output_rank]:
    if output_rank != target_shape_buf.dim(0):
        raise Error("[reshape] requires (len(target_shape) == output_rank)")

    # move the target shape from buffer into a static int tuple; also check and
    # record if there's any to-be-inferred dimension (-1).
    var target_shape = StaticIntTuple[output_rank]()
    var to_be_inferred_axis = -1
    var non_negative_dim_prodcut = 1
    for axis in range(output_rank):
        var target_dim = int(target_shape_buf[axis])
        target_shape[axis] = target_dim
        if target_dim < 0:
            if target_dim != -1:
                raise Error(
                    "[reshape] only -1 is allowed as a negative value in target"
                    " shape"
                )
            if to_be_inferred_axis != -1:
                raise Error("[reshape] only one -1 is allowed in target shape")
            to_be_inferred_axis = axis
        else:
            non_negative_dim_prodcut *= target_dim

    var input_num_elems = input_buf.num_elements()
    var output_num_elems = non_negative_dim_prodcut
    # Infer a dimension as the remaining elements, if needed.
    if to_be_inferred_axis != -1:
        target_shape[to_be_inferred_axis] = (
            input_num_elems // non_negative_dim_prodcut
        )
        output_num_elems *= target_shape[to_be_inferred_axis]

    if output_num_elems != input_num_elems:
        raise Error("[reshape] input and output number of elements must match")

    return target_shape
