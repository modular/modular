# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from enum import Enum
from typing import Any, Tuple, Union

IndexType = Union[int, Tuple[Union[int, slice], ...]]

class DType(Enum): ...

class Device:
    def __str__(self) -> str: ...

class Tensor:
    dtype: DType
    shape: Tuple[int, ...]
    rank: int
    is_contiguous: bool
    is_host: bool

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: DType,
        device: Device,
    ) -> None: ...
    def set(self, index: IndexType, value: Any) -> None: ...
    def get(self, index: IndexType) -> Tensor: ...
    def item(self) -> Any: ...
    def copy_to(self, device: Device) -> Tensor: ...

def cpu_device(device_id: int) -> Device: ...
