# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import logging
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    GenerationStatus,
    PixelGenerationContext,
    PixelGenerationOutput,
    PixelGenerationRequest,
)
from max.serve.pipelines.llm import BasePipeline
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms

logger = logging.getLogger("max.serve")


class PixelGeneratorPipeline(
    BasePipeline[
        PixelGenerationContext, PixelGenerationRequest, PixelGenerationOutput
    ]
):
    """Base class for diffusion-based image and video generation pipelines."""

    async def next_chunk(
        self, request: PixelGenerationRequest
    ) -> AsyncGenerator[PixelGenerationOutput, None]:
        """Generates and streams images or videos for the provided request."""

        total_sw = StopWatch()
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    context.request_id, context
                ):
                    assert isinstance(response, PixelGenerationOutput)

                    # Postprocess visual data: normalize and transpose
                    if (
                        response.pixel_data is not None
                        and response.pixel_data.size > 0
                    ):
                        pixel_data = await self.tokenizer.postprocess(
                            response.pixel_data
                        )
                        # Create new output with processed visual data
                        response = PixelGenerationOutput(
                            request_id=response.request_id,
                            final_status=response.final_status,
                            pixel_data=pixel_data,
                        )

                    yield response
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    total_sw.elapsed_ms,
                )

    async def generate_full_image(
        self, request: PixelGenerationRequest
    ) -> PixelGenerationOutput:
        """Generates complete image for the provided request."""
        image_chunks: list[PixelGenerationOutput] = []
        np_chunks: list[npt.NDArray[np.floating[Any]]] = []
        async for chunk in self.next_chunk(request):
            if chunk.pixel_data.size == 0:
                continue
            np_chunks.append(chunk.pixel_data)
            image_chunks.append(chunk)

        if len(image_chunks) == 0:
            return PixelGenerationOutput(
                request_id=request.request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
            )

        combined_image = np.concatenate(np_chunks, axis=-1)

        # We should only return from the next_chunk loop when the last chunk
        # is done.
        last_chunk = image_chunks[-1]
        assert last_chunk.is_done

        return PixelGenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            pixel_data=combined_image,
        )

    async def generate_full_video(
        self, request: PixelGenerationRequest
    ) -> PixelGenerationOutput:
        raise NotImplementedError("Not implemented yet!")
