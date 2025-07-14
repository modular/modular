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
from typing import Generic, Optional, TypeVar

import msgspec


class EngineStatus(str, Enum):
    """
    Represents the status of an engine operation.

    Status values:
        SUCCESSFUL: Indicates that the engine executed the operation successfully and data is returned.
        CANCELLED: Indicates that the request was cancelled before completion; no further data will be provided.
        COMPLETE: Indicates that the request was previously finished and no further data should be streamed.
    """

    SUCCESSFUL = "successful"
    CANCELLED = "cancelled"
    COMPLETE = "complete"


T = TypeVar("T")


class EngineResult(msgspec.Struct, Generic[T], tag=True, omit_defaults=True):
    """
    Structure representing the result of an engine operation.

    Attributes:
        status (EngineStatus): The status of the operation.
        result (Any): The result data of the operation.
    """

    status: EngineStatus
    result: Optional[T]

    @classmethod
    def cancelled(cls) -> "EngineResult":
        """
        Create an EngineResult representing a cancelled operation.

        Returns:
            EngineResult: An EngineResult with CANCELLED status and no result.
        """
        return EngineResult(status=EngineStatus.CANCELLED, result=None)

    @classmethod
    def complete(cls) -> "EngineResult":
        """
        Create an EngineResult representing a completed operation.

        Returns:
            EngineResult: An EngineResult with COMPLETE status and no result.
        """
        return EngineResult(status=EngineStatus.COMPLETE, result=None)

    @classmethod
    def successful(cls, result: T) -> "EngineResult":
        """
        Create an EngineResult representing a successful operation.

        Args:
            result (T): The result data of the operation.

        Returns:
            EngineResult: An EngineResult with SUCCESSFUL status and the provided result.
        """
        return EngineResult(status=EngineStatus.SUCCESSFUL, result=result)

    def continue_stream(self) -> bool:
        """
        Determine if the stream should continue based on the current status.

        Returns:
            bool: True if the stream should continue, False otherwise.
        """
        return self.status not in [
            EngineStatus.CANCELLED,
            EngineStatus.COMPLETE,
        ]
