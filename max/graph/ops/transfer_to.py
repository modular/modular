# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for transfer_to."""

from __future__ import annotations

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue


def transfer_to(x: TensorValue, device: DeviceRef) -> TensorValue:
    """Device-to-Device transfer operation.

    This op transfers the input tensor from its current device over to another. A device represents a
    computation unit, like CPU, GPU, etc. This op is useful for instance when working with
    accelerators, like GPU, where for instance one may need to move data from GPU to GPU, or
    from one GPU to CPU.

    Args:
        x: The input tensor to transfer.
        device: The device to transfer to.

    Returns:
        A tensor transferred to device specified.
    """
    if device == x.type.device:
        return x

    return Graph.current._add_op(
        rmo.mo_transfer,
        TensorType(dtype=x.dtype, shape=x.shape, device=device).to_mlir(),
        x,
    )[0].tensor
