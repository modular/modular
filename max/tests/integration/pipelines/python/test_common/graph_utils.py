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
from typing_extensions import TypeGuard


def are_all_tensors_iterable(
    it: Iterable[Tensor | MojoValue],
) -> TypeGuard[Iterable[Tensor]]:
    for value in it:
        if not isinstance(value, Tensor):
            return False
    return True


def are_all_tensors_sequence(
    it: Sequence[Tensor | MojoValue],
) -> TypeGuard[Sequence[Tensor]]:
    for value in it:
        if not isinstance(value, Tensor):
            return False
    return True


def are_all_buffer_values_sequence(
    it: Sequence[Value],
) -> TypeGuard[Sequence[BufferValue]]:
    for value in it:
        if not isinstance(value, BufferValue):
            return False
    return True


def are_all_tensor_values_iterable(
    it: Iterable[Value],
) -> TypeGuard[Iterable[TensorValue]]:
    for value in it:
        if not isinstance(value, TensorValue):
            return False
    return True


def are_all_tensor_values_sequence(
    it: Sequence[Value],
) -> TypeGuard[Sequence[TensorValue]]:
    for value in it:
        if not isinstance(value, TensorValue):
            return False
    return True
