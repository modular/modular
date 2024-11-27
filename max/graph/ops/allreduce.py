# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for allreduce."""

from __future__ import annotations

from collections.abc import Iterable

from ..graph import Graph
from ..value import TensorValue
from .elementwise import add
from .transfer_to import transfer_to


def sum(inputs: Iterable[TensorValue]) -> Iterable[TensorValue]:
    """Collective allreduce summation operation.

    This op is a collective op which takes in tensors from different devices and outputs
    tensors on different devices. In particular, this operation will gather the inputs
    across different devices and reduce them via a summation operation. The result is
    then broadcasted back to the same devices that the inputs came from.

    Args:
        inputs: The input tensors reduce.

    Returns:
        An iterable outputs which all hold the reduction output.
    """
    shape = None
    devices = []

    for input in inputs:
        if not shape:
            shape = input.shape
        if input.shape != shape:
            raise ValueError(
                "allreduce.sum operation must have the same shape across all"
                " input tensors."
            )
        if not input.device:
            raise ValueError(
                f"allreduce.sum operation input = {input} needs to have an"
                " explicit device."
            )
        if input.device in devices:
            raise ValueError(
                "allreduce.sum operation must have unique devices across it's"
                " input tensors."
            )
        devices.append(input.device)

    # Naive Impl
    # Explicit transfer all to 1 device
    reduction_device = devices[0]
    post_transfer_inputs = []
    for input in inputs:
        post_transfer_inputs.append(transfer_to(input, reduction_device))
    # Sum all the inputs
    cumsum_output = post_transfer_inputs[0]
    for input in post_transfer_inputs[1:]:
        cumsum_output = add(cumsum_output, input)
    # Explicit transfer back to all device
    transferred_outputs = []
    for device in devices:
        transferred_outputs.append(transfer_to(cumsum_output, device))

    return transferred_outputs
