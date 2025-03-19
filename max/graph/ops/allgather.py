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
from .concat import concat


def allgather(inputs: Iterable[TensorValue], dim: int = 0) -> list[TensorValue]:
    """Collective allgather operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and concatenates them along the 0th dimension.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to gather.
        dim: Dimension to concatenate the input tensors. Defaults to 0.

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

    if not -shape.rank <= dim < shape.rank:
        raise IndexError(f"Dimension out of range {shape.rank}, {dim=}")
    if dim < 0:
        dim += shape.rank

    output_shape = Shape(shape)
    num_devices = len(inputs)
    output_shape[0] = shape[0] * num_devices
    output_types = [
        TensorType(dtype, output_shape, device=x.device).to_mlir()
        for x in inputs
    ]
    results = Graph.current._add_op(
        mo.distributed_allgather,
        output_types,
        inputs,
    )
    outputs = [res.tensor for res in results]

    if dim == 0:
        return outputs

    # Slice the output tensors and re-concatenate along the desired dim.
    reconcatenated_outputs = []
    for output in outputs:
        # Can't use ops.chunk because the 0th dimension might be dynamic.
        chunked_outputs = [
            output[i * shape[0] : (i + 1) * shape[0], ...]
            for i in range(num_devices)
        ]
        reconcatenated_outputs.append(concat(chunked_outputs, axis=dim))
    return reconcatenated_outputs
