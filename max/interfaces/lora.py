# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Shared types for LoRA queue operations."""

from __future__ import annotations

from enum import Enum

import msgspec


class LoRAType(Enum):
    """
    Enumeration for LoRA Types.
    """

    A = "lora_A"
    """Represents the LoRA A matrix (high rank tensor to low rank tensor)."""

    B = "lora_B"
    """Represents the LoRA B matrix. (low rank tensor to high rank tensor)"""

    BIAS = "lora.bias"
    """Represents the LoRA bias matrix. (added to matrix B)"""


class LoRAOperation(Enum):
    """Enum for different LoRA operations."""

    LOAD = "load"
    UNLOAD = "unload"
    LIST = "list"


class LoRAStatus(Enum):
    """Enum for LoRA operation status."""

    SUCCESS = "success"
    LOAD_NAME_EXISTS = "load_name_exists"
    UNLOAD_NAME_NONEXISTENT = "unload_name_nonexistent"
    LOAD_ERROR = "load_error"
    UNLOAD_ERROR = "unload_error"
    LOAD_SLOTS_FULL = "load_slots_full"
    LOAD_INVALID_PATH = "load_invalid_path"
    LOAD_INVALID_ADAPTER = "load_invalid_adapter"


class LoRARequest(msgspec.Struct, omit_defaults=True):
    """Container for LoRA adapter requests."""

    operation: LoRAOperation
    lora_name: str | None = None
    lora_path: str | None = None


class LoRAResponse(msgspec.Struct):
    """Response from LoRA operations."""

    status: LoRAStatus
    message: str | list[str]
