# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for allgather."""

from __future__ import annotations

from collections.abc import Iterable

from max.mlir.dialects import mo

from ..graph import Graph  # noqa
from ..type import Shape, TensorType
from ..value import TensorValue


def allgather(inputs: Iterable[TensorValue]) -> list[TensorValue]:
    """Collective allgather operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and concatenates them along the 0th dimension.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to gather.

    Returns:
        An iterable outputs which all hold the gathered output. Each output
        is a rank-1 array.
    """
    # Convert `inputs` to list since we'll iterate over it multiple times.
    inputs = list(inputs)
    if len(inputs) < 2:
        return inputs

    shape = inputs[0].shape
    dtype = inputs[0].dtype
    if not all(input.shape == shape for input in inputs[1:]):
        msg = (
            "allgather operation must have the same shape across all"
            " input tensors."
        )
        raise ValueError(msg)
    if not all(input.dtype == dtype for input in inputs[1:]):
        all_dtypes = ", ".join([str(x.dtype) for x in inputs])
        msg = (
            "allgather operation must have the same dtype across all"
            f" input tensors. Got: {all_dtypes}"
        )
        raise ValueError(msg)
    if not all(input.device for input in inputs[1:]):
        msg = (
            "allgather operation inputs must have an explicit device. "
            f"Got: {inputs}"
        )
        raise ValueError(msg)

    devices = []
    for input in inputs:
        if input.device in devices:
            all_devices = ", ".join([str(x.device) for x in inputs])
            msg = (
                "allgather operation must have unique devices across its input "
                f"tensors. Got: {all_devices}"
            )
            raise ValueError(msg)
        devices.append(input.device)

    output_shape = Shape(shape)
    output_shape[0] = inputs[0].shape[0] * len(inputs)
    output_types = [
        TensorType(dtype, output_shape, device=x.device).to_mlir()
        for x in inputs
    ]
    results = Graph.current._add_op(
        mo.distributed_allgather,
        output_types,
        inputs,
    )
    return [res.tensor for res in results]
