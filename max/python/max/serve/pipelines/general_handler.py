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
"""GeneralPipelineHandler for handling OpenResponses API requests."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from max.interfaces import BaseContext, PipelineOutput
from max.interfaces.request import OpenResponsesRequest
from max.serve.pipelines.llm import BasePipeline
from max.serve.telemetry.stopwatch import StopWatch


class GeneralPipelineHandler(
    BasePipeline[BaseContext, OpenResponsesRequest, PipelineOutput]
):
    """General pipeline handler for OpenResponses API requests.

    This is a minimal implementation that can be extended for specific
    modalities (image, video, etc.).
    """

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[PipelineOutput, None]:
        """Generates and streams responses for the provided request.

        Args:
            request: The OpenResponses request to process.

        Yields:
            PipelineOutput chunks as generation progresses.
        """
        total_sw = StopWatch()
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        try:
            # Create context from request
            context = await self.tokenizer.new_context(request)

            # Stream responses from the engine
            async for response in self.model_worker.stream(
                request.request_id, context
            ):
                yield response
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    total_sw.elapsed_ms,
                )
