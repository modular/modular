# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from algorithm import unroll
from utils.list import DimList
from utils.index import StaticIntTuple
from utils._annotations import mogg_register, mogg_view_op


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
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    new_shape: StaticIntTuple[output_rank],
) -> NDBuffer[output_rank, DimList.create_unknown[output_rank](), type]:
    var stride_tuple = StaticIntTuple[output_rank]()
    var stride: Int = 1

    # Create contiguous strides.
    @always_inline
    @parameter
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        let i = output_rank - 1 - idx
        stride_tuple[i] = stride
        stride *= new_shape[i]

    unroll[output_rank, body]()

    # Return the a view with the new shape.
    return NDBuffer[output_rank, DimList.create_unknown[output_rank](), type](
        input.data, new_shape, stride_tuple
    )


@always_inline
fn reshape_shape[
    input_rank: Int,
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    target_shape_buf: NDBuffer[
        1, DimList.create_unknown[1](), target_shape_type
    ],
) -> StaticIntTuple[output_rank]:
    # TODO(#17512)
    debug_assert(
        output_rank == target_shape_buf.dim(0),
        "output rank must match target shape",
    )

    # move the target shape from buffer into a static int tuple; also check and
    # record if there's any to-be-inferred dimension (-1).
    var target_shape = StaticIntTuple[output_rank]()
    var to_be_inferred_axis = -1
    var non_negative_dim_prodcut = 1
    for axis in range(output_rank):
        let target_dim = int(target_shape_buf[axis])
        target_shape[axis] = target_dim
        if target_dim == -1:
            # TODO(#17512)
            debug_assert(
                to_be_inferred_axis == -1,
                "only one -1 is allowed in target shape",
            )
            to_be_inferred_axis = axis
        else:
            # TODO(#17512)
            debug_assert(
                target_dim >= 0,
                "only -1 is allowed as a negative value in target shape",
            )
            non_negative_dim_prodcut *= target_dim

    let input_num_elems = input_buf.num_elements()
    var output_num_elems = non_negative_dim_prodcut
    # Infer a dimension as the remaining elements, if needed.
    if to_be_inferred_axis != -1:
        # TODO(#17512)
        debug_assert(
            non_negative_dim_prodcut != 0,
            (
                "concrete dimensions must not contain 0 if there's a"
                " to-be-inferred dimension"
            ),
        )
        debug_assert(
            input_num_elems % non_negative_dim_prodcut == 0,
            "to-be-inferred dimension must be an integer",
        )
        target_shape[to_be_inferred_axis] = (
            input_num_elems // non_negative_dim_prodcut
        )
        output_num_elems = input_num_elems

    # TODO(#17512)
    debug_assert(
        output_num_elems == input_num_elems,
        "output and input number of elements must match",
    )

    return target_shape
