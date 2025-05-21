# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for scatter."""

from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import DeviceRef, DimLike, TensorType
from ..value import TensorValue, TensorValueLike
from .nonzero import nonzero


def scatter(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: TensorValueLike = -1,
) -> TensorValue:
    """
    Creates a new symbolic tensor where the updates are written to input according to indices.

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A new symbolic tensor representing the result of the scatter operation.
    """

    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)
    axis = dtype_promotion._promote_to_strong(
        axis, DType.int64, DeviceRef.CPU()
    )

    # TODO(GEX-2197): Support scatter on GPU
    old_device = input.device
    input = input.to(DeviceRef.CPU())
    updates = updates.to(DeviceRef.CPU())
    indices = indices.to(DeviceRef.CPU())
    axis = axis.to(DeviceRef.CPU())

    return Graph.current._add_op(
        rmo.mo_scatter,
        input.type.to_mlir(),
        input,
        updates,
        indices,
        axis,
    )[0].tensor.to(old_device)


def masked_scatter(
    input: TensorValueLike,
    mask: TensorValueLike,
    updates: TensorValueLike,
    out_dim: DimLike,
) -> TensorValue:
    """
    Creates a new symbolic tensor where the updates are written to input where mask is true.

    Args:
        input: The input symbolic tensor to write elements to.
        mask: A symbolic tensor of boolean values to update.
        updates: A symbolic tensor of elements to write to input.
        out_dim: The new data-dependent dimension.

    Returns:
        A new symbolic tensor representing the result of the masked_scatter operation.
    """
    input, updates = TensorValue(input), TensorValue(updates)
    mask = dtype_promotion._promote_to_strong(
        mask, DType.bool, input.type.device or DeviceRef.CPU()
    )

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    # input_size = reduce(mul, input.shape, 1)
    # updates_size = reduce(mul, updates.shape, 1)
    # TODO: This is a bug. They don't have to match.
    # Assuming it will throw a run-time error if updates_size != non-zeros in mask
    # if input_size != updates_size and updates_size != 1:
    #    raise ValueError(
    #        f"The number of elements in the input ({input_size}) and the number"
    #        f" of elements in updates ({updates_size}) must match"
    #    )

    mask = mask.broadcast_to(input.shape)
    indices = nonzero(mask, out_dim)

    updates = updates.flatten()

    return Graph.current._add_op(
        rmo.mo_scatter_nd,
        TensorType(input.dtype, input.shape, input.device).to_mlir(),
        input,
        updates,
        indices,
    )[0].tensor
