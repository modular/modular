# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.driver import Tensor
from max.engine import MojoValue
from max.graph import BufferValue, TensorValue, Value
from nvitop import Device as NVITOPDevice
from typing_extensions import TypeGuard


def is_h100_h200() -> bool:
    """Checks if this is an H100 or H200 GPU."""
    devices = NVITOPDevice.all()
    return bool(devices) and (
        "H100" in devices[0].name() or "H200" in devices[0].name()
    )


def is_b200() -> bool:
    """Checks if this is an B200 GPU."""
    devices = NVITOPDevice.all()
    return bool(devices) and ("B200" in devices[0].name())


def are_all_tensors_iterable(
    it: Iterable[Tensor | MojoValue],
) -> TypeGuard[Iterable[Tensor]]:
    return all(isinstance(value, Tensor) for value in it)


def are_all_tensors_sequence(
    it: Sequence[Tensor | MojoValue],
) -> TypeGuard[Sequence[Tensor]]:
    return all(isinstance(value, Tensor) for value in it)


def are_all_buffer_values_sequence(
    it: Sequence[Value],
) -> TypeGuard[Sequence[BufferValue]]:
    return all(isinstance(value, BufferValue) for value in it)


def are_all_tensor_values_iterable(
    it: Iterable[Value],
) -> TypeGuard[Iterable[TensorValue]]:
    return all(isinstance(value, TensorValue) for value in it)


def are_all_tensor_values_sequence(
    it: Sequence[Value],
) -> TypeGuard[Sequence[TensorValue]]:
    return all(isinstance(value, TensorValue) for value in it)
