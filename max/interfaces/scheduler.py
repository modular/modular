# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from enum import Enum
from typing import Generic, Optional

import msgspec

from .pipeline import PipelineOutputType


class SchedulerStatus(str, Enum):
    """
    Represents the status of a scheduler operation for a specific pipeline execution.

    The scheduler manages the execution of pipeline operations and returns status updates
    to indicate the current state of the pipeline execution. This enum defines the possible
    states that a pipeline operation can be in from the scheduler's perspective.

    """

    ACTIVE = "active"
    """Indicates that the scheduler executed the pipeline operation successfully and request remains active."""
    CANCELLED = "cancelled"
    """Indicates that the pipeline operation was cancelled before completion; no further data will be provided."""
    COMPLETE = "complete"
    """Indicates that the pipeline operation was previously finished and no further data should be streamed."""


class SchedulerResult(msgspec.Struct, Generic[PipelineOutputType]):
    """
    Structure representing the result of a scheduler operation for a specific pipeline execution.

    This class encapsulates the outcome of a pipeline operation as managed by the scheduler,
    including both the execution status and any resulting data from the pipeline. The scheduler
    uses this structure to communicate the state of pipeline operations back to clients,
    whether the operation is still running, has completed successfully, or was cancelled.

    The generic type parameter allows this result to work with different types of pipeline
    outputs while maintaining type safety.

    """

    status: SchedulerStatus
    """The current status of the pipeline operation from the scheduler's perspective."""

    result: Optional[PipelineOutputType]
    """The pipeline output data, if any. May be None for cancelled operations or during intermediate states of streaming operations."""

    @classmethod
    def cancelled(cls) -> "SchedulerResult":
        """
        Create a SchedulerResult representing a cancelled pipeline operation.

        Returns:
            SchedulerResult: A SchedulerResult with CANCELLED status and no result.
        """
        return SchedulerResult(status=SchedulerStatus.CANCELLED, result=None)

    @classmethod
    def complete(cls, result: PipelineOutputType) -> "SchedulerResult":
        """
        Create a SchedulerResult representing a completed pipeline operation.

        Args:
            result: The final pipeline output data.

        Returns:
            SchedulerResult: A SchedulerResult with COMPLETE status and the final result.
        """
        return SchedulerResult(status=SchedulerStatus.COMPLETE, result=result)

    @classmethod
    def active(cls, result: PipelineOutputType) -> "SchedulerResult":
        """
        Create a SchedulerResult representing an active pipeline operation.

        Args:
            result: The current pipeline output data (may be partial for streaming operations).

        Returns:
            SchedulerResult: A SchedulerResult with ACTIVE status and the provided result.
        """
        return SchedulerResult(status=SchedulerStatus.ACTIVE, result=result)

    @property
    def stop_stream(self) -> bool:
        """
        Determine if the pipeline operation stream should continue based on the current status.

        Returns:
            bool: True if the pipeline operation stream should stop (CANCELLED or COMPLETE),
                  False if it should continue (ACTIVE).
        """
        return self.status != SchedulerStatus.ACTIVE
