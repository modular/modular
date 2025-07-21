# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces and result/status classes for engine operations in the MAX API.

This module defines the core status and result structures used by engine components
to communicate operation outcomes, including success and cancellation states.
"""

from enum import Enum
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

import msgspec
from typing_extensions import TypeAlias


class EngineStatus(str, Enum):
    """
    Represents the status of an engine operation.

    Status values:
        ACTIVE: Indicates that the engine executed the operation successfully and request remains active.
        CANCELLED: Indicates that the request was cancelled before completion; no further data will be provided.
        COMPLETE: Indicates that the engine executed the operation successfully and the request is completed.
    """

    ACTIVE = "active"
    """Indicates that the engine executed the operation successfully and request remains active."""
    CANCELLED = "cancelled"
    """Indicates that the request was cancelled before completion; no further data will be provided."""
    COMPLETE = "complete"
    """Indicates that the request was previously finished and no further data should be streamed."""


@runtime_checkable
class EngineOutput(Protocol):
    """
    Abstract base class representing the output of an engine operation.

    Subclasses must implement the `is_done` property to indicate whether
    the engine operation has completed.
    """

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the engine operation has completed.

        Returns:
            bool: True if the operation is done, False otherwise.
        """
        ...


O = TypeVar("O", bound=EngineOutput)

EngineOutputs: TypeAlias = dict[str, O]
"""
Type alias for a dictionary mapping string keys to EngineOutput instances.

This is used to represent a collection of engine outputs, where each key
identifies a specific output.
"""

# TODO: Make EngineOutputs bound to EngineOutput, and remove redundancy
T = TypeVar("T")


class EngineResult(msgspec.Struct, Generic[T], tag=True, omit_defaults=True):
    """Structure representing the result of an engine operation."""

    status: EngineStatus
    """The status of the operation."""
    result: Optional[T]
    """The result data of the operation."""

    @classmethod
    def cancelled(cls) -> "EngineResult":
        """
        Create an EngineResult representing a cancelled operation.

        Returns:
            EngineResult: An EngineResult with CANCELLED status and no result.
        """
        return EngineResult(status=EngineStatus.CANCELLED, result=None)

    @classmethod
    def complete(cls, result: T) -> "EngineResult":
        """
        Create an ``EngineResult`` representing a completed operation.

        Returns:
            EngineResult: An ``EngineResult`` with ``COMPLETE`` status and no result.
        """
        return EngineResult(status=EngineStatus.COMPLETE, result=result)

    @classmethod
    def active(cls, result: T) -> "EngineResult":
        """
        Create an ``EngineResult`` representing an active operation.

        Args:
            result: The result data of the operation.

        Returns:
            EngineResult: An ``EngineResult`` with ``ACTIVE`` status and the provided result.
        """
        return EngineResult(status=EngineStatus.ACTIVE, result=result)

    @property
    def stop_stream(self) -> bool:
        """
        Determine if the stream should continue based on the current status.

        Returns:
            bool: True if the stream should stop, False otherwise.
        """
        return self.status != EngineStatus.ACTIVE
