# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""MAX Driver APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The Device class is not used in this file, but is used by others that import
# this file. Ruff will try to remove it, so ignore that Ruff check on this line.
from max._core.driver import (  # noqa: F401
    CPU,
    Accelerator,
    Device,
    accelerator_count,
)


def accelerator_api() -> str:
    """Returns the API used to program the accelerator."""
    if accelerator_count() > 0:
        return Accelerator().api
    return CPU().api


@dataclass(frozen=True)
class DeviceSpec:
    """Specification for a device, containing its ID and type.

    This class provides a way to specify device parameters like ID and type (CPU/GPU)
    for creating Device instances.
    """

    id: int
    """Provided id for this device."""

    device_type: Literal["cpu", "gpu"] = "cpu"
    """Type of specified device."""

    @staticmethod
    def cpu(id: int = -1):
        """Creates a CPU device specification."""
        return DeviceSpec(id, "cpu")

    @staticmethod
    def accelerator(id: int = -1):
        """Creates an accelerator (GPU) device specification."""
        return DeviceSpec(id, "gpu")
