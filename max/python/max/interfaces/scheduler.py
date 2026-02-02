# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from typing import Generic

import msgspec

from .pipeline import PipelineOutputType


class SchedulerError(msgspec.Struct):
    """Structure representing an error that occurred during scheduling.

    This class captures exception details for communication across process
    boundaries, allowing clients to receive meaningful error information
    instead of hanging indefinitely when a pipeline execution fails.
    """

    error_type: str
    """The exception class name (e.g., 'RuntimeError', 'CUDAOutOfMemoryError')."""

    error_message: str
    """The exception message string."""

    traceback_str: str
    """The full traceback string for debugging."""

    @classmethod
    def from_exception(cls, exc: BaseException) -> SchedulerError:
        """Create a SchedulerError from an exception.

        Args:
            exc: The exception to capture.

        Returns:
            SchedulerError: A SchedulerError with the exception details.
        """
        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_str="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )


class Scheduler(ABC):
    """Abstract base class defining the interface for schedulers."""

    @abstractmethod
    def run_iteration(self):  # noqa: ANN201
        """The core scheduler routine that creates and executes batches.

        This method should implement the core scheduling logic including:
        - Batch creation and management
        - Request scheduling
        """
        ...


class SchedulerResult(msgspec.Struct, Generic[PipelineOutputType]):
    """
    Structure representing the result of a scheduler operation for a specific pipeline execution.

    This class encapsulates the outcome of a pipeline operation as managed by the scheduler,
    including both the execution status and any resulting data from the pipeline. The scheduler
    uses this structure to communicate the state of pipeline operations back to clients,
    whether the operation is still running, has completed successfully, was cancelled, or
    encountered an error.

    The generic type parameter allows this result to work with different types of pipeline
    outputs while maintaining type safety.

    """

    is_done: bool
    """The current status of the pipeline operation from the scheduler's perspective."""

    result: PipelineOutputType | None
    """The pipeline output data, if any. May be None for cancelled operations or during intermediate states of streaming operations."""

    error: SchedulerError | None = None
    """Error details if the pipeline execution failed. None for successful operations."""

    @classmethod
    def cancelled(cls) -> SchedulerResult[PipelineOutputType]:
        """
        Create a SchedulerResult representing a cancelled pipeline operation.

        Returns:
            SchedulerResult: A SchedulerResult that is done.
        """
        return SchedulerResult(is_done=True, result=None)

    @classmethod
    def create(
        cls, result: PipelineOutputType
    ) -> SchedulerResult[PipelineOutputType]:
        """
        Create a SchedulerResult representing a pipeline operation with some result.

        Args:
            result: The pipeline output data.

        Returns:
            SchedulerResult: A SchedulerResult with a result.
        """
        return SchedulerResult(is_done=result.is_done, result=result)

    @classmethod
    def from_error(
        cls, exc: BaseException
    ) -> SchedulerResult[PipelineOutputType]:
        """
        Create a SchedulerResult representing a failed pipeline operation.

        Args:
            exc: The exception that caused the failure.

        Returns:
            SchedulerResult: A SchedulerResult with error details.
        """
        return SchedulerResult(
            is_done=True, result=None, error=SchedulerError.from_exception(exc)
        )
