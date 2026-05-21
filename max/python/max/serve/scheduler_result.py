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

"""Defines the :class:`SchedulerResult` data structure for MAX serve."""

from __future__ import annotations

from typing import Generic

import msgspec
from max.interfaces.pipeline import PipelineOutputType

__all__ = ["SchedulerResult"]


class SchedulerResult(msgspec.Struct, Generic[PipelineOutputType]):
    """Structure representing the result of a scheduler operation.

    Encapsulates the outcome of a pipeline operation as managed by the
    scheduler, including both the execution status and any resulting data.
    The scheduler uses this structure to communicate the state of pipeline
    operations back to clients, whether the operation is still running, has
    completed successfully, or was cancelled.
    """

    is_done: bool
    """Whether the pipeline operation is complete."""

    result: PipelineOutputType | None
    """The pipeline output data, if any. ``None`` for cancelled operations or
    intermediate streaming states."""

    @classmethod
    def cancelled(cls) -> SchedulerResult[PipelineOutputType]:
        """Creates a ``SchedulerResult`` representing a cancelled operation.

        Returns:
            A :class:`SchedulerResult` with ``is_done=True`` and no result.
        """
        return SchedulerResult(is_done=True, result=None)

    @classmethod
    def create(
        cls, result: PipelineOutputType
    ) -> SchedulerResult[PipelineOutputType]:
        """Creates a ``SchedulerResult`` wrapping a pipeline output.

        Args:
            result: The pipeline output data.

        Returns:
            A :class:`SchedulerResult` reflecting the output's completion state.
        """
        return SchedulerResult(is_done=result.is_done, result=result)
