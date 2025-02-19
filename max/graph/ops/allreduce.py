# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for allreduce."""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph  # noqa
from ..type import DeviceRef
from ..value import BufferType, BufferValue, TensorValue


class Signals:
    """Signal buffers used for peer-to-peer communication in allreduce.

    Device code uses these buffers by enabling peer-to-peer access.
    Then thread blocks use the buffers to implement barriers for
    synchronization, and to hold intermediate communication results.
    """

    NUM_BYTES = (1 + 128) * 1024 * 1024
    """The size of the signal buffers used for communication in allreduce."""
    # NOTE: ``NUM_BYTES`` must stay in sync with the size of the ``Signal``
    # Mojo struct + the size of the intermediate buffer for communication.

    devices: list[DeviceRef]
    """List of devices that these signals communicate between."""

    def __init__(self, devices: Iterable[DeviceRef]) -> None:
        """
        Args:
            devices: Devices between which these signals communicate.
        """
        self.devices = list(devices)

    def input_types(self) -> list[BufferType]:
        """Gets graph input types corresponding to these signal buffers."""
        return [
            BufferType(
                dtype=DType.uint8, shape=(Signals.NUM_BYTES,), device=dev
            )
            for dev in self.devices
        ]


def sum(
    inputs: Iterable[TensorValue], signal_buffers: Iterable[BufferValue]
) -> list[TensorValue]:
    """Collective allreduce summation operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and reduce them via a summation operation.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to reduce.
        signal_buffers: Device buffer values used for synchronization.

    Returns:
        An iterable outputs which all hold the reduction output.
    """
    # Convert `inputs` to list since we'll iterate over it twice.
    inputs = list(inputs)

    shape = None
    devices = []

    for input in inputs:
        if not shape:
            shape = input.shape
        if input.shape != shape:
            msg = (
                "allreduce.sum operation must have the same shape across all"
                " input tensors."
            )
            raise ValueError(msg)
        if not input.device:
            msg = (
                f"allreduce.sum operation input = {input} needs to have an"
                " explicit device."
            )
            raise ValueError(msg)
        if input.device in devices:
            msg = (
                "allreduce.sum operation must have unique devices across its"
                " input tensors."
            )
            raise ValueError(msg)
        devices.append(input.device)

    if len(devices) == 1:
        # Nothing to do.
        return inputs

    if len(devices) not in {2, 4}:
        msg = f"allreduce sum only supports 2 or 4 devices, but got {len(devices)}"
        raise ValueError(msg)

    # Map from the number of devices to a fixed-num-devices allreduce kernel.
    allreduce_op = {
        2: mo.distributed_allreduce_2gpu_sum,
        4: mo.distributed_allreduce_4gpu_sum,
    }[len(devices)]

    results = Graph.current._add_op(
        allreduce_op,
        [x.type.to_mlir() for x in inputs],
        *signal_buffers,
        inputs,
    )
    return [res.tensor for res in results]
