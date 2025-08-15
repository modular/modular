# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import Generic, Optional

import msgspec

from .pipeline import PipelineOutputType


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

    is_done: bool
    """The current status of the pipeline operation from the scheduler's perspective."""

    result: Optional[PipelineOutputType]
    """The pipeline output data, if any. May be None for cancelled operations or during intermediate states of streaming operations."""

    @classmethod
    def cancelled(cls) -> "SchedulerResult":
        """
        Create a SchedulerResult representing a cancelled pipeline operation.

        Returns:
            SchedulerResult: A SchedulerResult that is done.
        """
        return SchedulerResult(is_done=True, result=None)

    @classmethod
    def create(cls, result: PipelineOutputType) -> "SchedulerResult":
        """
        Create a SchedulerResult representing a pipeline operation with some result.

        Args:
            result: The pipeline output data.

        Returns:
            SchedulerResult: A SchedulerResult with a result.
        """
        return SchedulerResult(is_done=result.is_done, result=result)
