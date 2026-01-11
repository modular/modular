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
import sys
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    BaseContext,
    GenerationStatus,
    PipelineOutput,
    PixelGenerationOutput,
    PixelGenerationRequest,
    Request,
)
from max.pipelines.core import (
    PixelContext,
)
from max.serve.pipelines.llm import BasePipeline
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms

if sys.version_info < (3, 11):
    pass
else:
    pass

logger = logging.getLogger("max.serve")

ContextType = TypeVar("ContextType", bound=BaseContext)
RequestType = TypeVar("RequestType", bound=Request)
OutputType = TypeVar("OutputType", bound=PipelineOutput)


class PixelGeneratorPipeline(
    BasePipeline[PixelContext, PixelGenerationRequest, PixelGenerationOutput]
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
                # context = await self.tokenizer.new_context(request)
                # For image generation, create context directly from request
                # since Qwen-3-4B haven't worked.
                context = PixelContext(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    messages=request.messages,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    negative_prompt=request.negative_prompt,
                    num_images_per_prompt=request.num_images_per_prompt,
                )

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    context.request_id, context
                ):
                    assert isinstance(response, PixelGenerationOutput)

                    # Postprocess image: normalize [-1,1] → [0,1] and transpose NCHW → NHWC
                    # This is analogous to tokenizer.decode() in text generation
                    if (
                        response.pixel_data is not None
                        and response.pixel_data.size > 0
                    ):
                        image_np = response.pixel_data
                        image_np = (image_np * 0.5 + 0.5).clip(min=0.0, max=1.0)
                        image_np = image_np.transpose(0, 2, 3, 1)  # NCHW → NHWC
                        # Create new output with processed image
                        response = PixelGenerationOutput(
                            pixel_data=image_np,
                            metadata=response.metadata,
                            steps_executed=response.steps_executed,
                            final_status=response.final_status,
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
            if chunk.pixel_data.size == 0 or chunk.pixel_data.size == 0:
                continue
            np_chunks.append(chunk.pixel_data)
            image_chunks.append(chunk)

        if len(image_chunks) == 0:
            return PixelGenerationOutput(
                steps_executed=sum(
                    chunk.steps_executed for chunk in image_chunks
                ),
                final_status=GenerationStatus.END_OF_SEQUENCE,
            )

        combined_image = np.concatenate(np_chunks, axis=-1)

        # We should only return from the next_chunk loop when the last chunk
        # is done.
        last_chunk = image_chunks[-1]
        assert last_chunk.is_done

        return PixelGenerationOutput(
            pixel_data=combined_image,
            metadata=last_chunk.metadata,
            steps_executed=sum(chunk.steps_executed for chunk in image_chunks),
            final_status=GenerationStatus.END_OF_SEQUENCE,
        )

    async def generate_full_video(
        self, request: PixelGenerationRequest
    ) -> PixelGenerationOutput:
        raise NotImplementedError("Not implemented yet!")
