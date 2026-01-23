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

"""High-level interface for pixel generation using diffusion models."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event, Thread

from max.interfaces import (
    PixelContext,
    PixelGenerationOutput,
    PixelGenerationPipeline,
    PixelGenerationRequest,
    PixelGenerationTokenizer,
    RequestID,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig

# from max.serve.config import Settings
# from max.serve.scheduler.queues import SchedulerZmqConfigs


@dataclass
class _Request:
    """Internal request for batching multiple pixel generation requests."""

    id: RequestID
    prompts: Sequence[str]
    height: list[int]
    width: list[int]
    num_inference_steps: list[int]
    guidance_scale: list[float]
    num_images_per_prompt: list[int]
    negative_prompts: Sequence[str | None] | None = None


@dataclass
class _Response:
    """Internal response containing generated images."""

    outputs: list[PixelGenerationOutput]


@dataclass
class _ThreadControl:
    """Thread synchronization primitives."""

    ready: Event = field(default_factory=Event)
    cancel: Event = field(default_factory=Event)


class PixelGenerator:
    """High-level interface for generating pixels using diffusion models."""

    # Thread control and communication
    _pc: _ThreadControl
    _async_runner: Thread
    _request_queue: queue.Queue[_Request]
    _pending_requests: dict[RequestID, queue.Queue[_Response]]

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        """Initialize the image generator.

        Args:
            pipeline_config: Configuration specifying the model and parameters.
        """
        # TODO: Add support for offline inference
        # settings = Settings(MAX_SERVE_OFFLINE_INFERENCE=True)

        # Initialize thread control and queues
        self._pc = _ThreadControl()
        self._request_queue: queue.Queue[_Request] = queue.Queue()
        self._pending_requests: dict[RequestID, queue.Queue[_Response]] = {}

        # Start async runner
        self._async_runner = Thread(
            target=_run_async_worker,
            args=(
                self._pc,
                pipeline_config,
                self._request_queue,
                self._pending_requests,
            ),
        )
        self._async_runner.start()

        # Wait for worker to be ready
        self._pc.ready.wait()

    def __del__(self) -> None:
        """Clean up resources."""
        self._pc.cancel.set()
        if self._async_runner.is_alive():
            self._async_runner.join(timeout=5.0)

    def generate(
        self,
        prompts: str | Sequence[str],
        *,
        height: list[int] | None = None,
        width: list[int] | None = None,
        num_inference_steps: list[int] | None = None,
        guidance_scale: list[float] | None = None,
        num_images_per_prompt: list[int] | None = None,
    ) -> _Response:
        """Generate images from text prompts.

        This method is thread-safe and can be called from multiple threads.

        Args:
            prompts: Single prompt string or sequence of prompts.
            height: Image height in pixels.
            width: Image width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images per prompt.

        Returns:
            _Response containing the generated PIL Images.

        """
        # Normalize prompts to sequence
        if isinstance(prompts, str):
            prompts = [prompts]

        if negative_prompts is None:
            negative_prompts = [None] * len(prompts)
        if num_images_per_prompt is None:
            num_images_per_prompt = [1] * len(prompts)
        if height is None:
            height = [1024] * len(prompts)
        if width is None:
            width = [1024] * len(prompts)
        if num_inference_steps is None:
            num_inference_steps = [50] * len(prompts)
        if guidance_scale is None:
            guidance_scale = [3.5] * len(prompts)

        # Create internal request
        request = _Request(
            id=RequestID(),
            prompts=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )

        # Submit request and wait for response
        return self._submit_and_wait(request)

    def _submit_and_wait(self, request: _Request) -> _Response:
        """Submit a request to the queue and wait for response."""
        response_queue: queue.Queue[_Response] = queue.Queue()
        self._pending_requests[request.id] = response_queue

        try:
            self._request_queue.put_nowait(request)
            response = response_queue.get()
            return response
        finally:
            self._pending_requests.pop(request.id, None)


def _run_async_worker(
    pc: _ThreadControl,
    pipeline_config: PipelineConfig,
    request_queue: queue.Queue[_Request],
    pending_requests: Mapping[RequestID, queue.Queue[_Response]],
) -> None:
    pipeline_task = PIPELINE_REGISTRY.retrieve_pipeline_task(pipeline_config)
    tokenizer, model_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
        task=pipeline_task,
    )
    pipeline = model_factory()

    asyncio.run(
        _async_worker(pipeline, tokenizer, pc, request_queue, pending_requests)
    )


async def _async_worker(
    pipeline: PixelGenerationPipeline,
    tokenizer: PixelGenerationTokenizer,
    pc: _ThreadControl,
    request_queue: queue.Queue[_Request],
    pending_requests: Mapping[RequestID, queue.Queue[_Response]],
) -> None:
    """Background worker that processes image generation requests.

    This function runs in a separate thread and continuously processes
    requests from the queue until cancellation is signaled.
    """

    pc.ready.set()

    # TODO: After adding a Scheduler for PixelGenerationPipeline, need to update this to use the Scheduler.
    while True:
        if pc.cancel.is_set():
            break
        try:
            request: _Request = request_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        if request.negative_prompts is not None:
            assert len(request.negative_prompts) == len(request.prompts), (
                "Number of negative prompts must match number of prompts"
            )

        outputs: list[PixelGenerationOutput] = []
        for (
            prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            num_images_per_prompt,
            negative_prompt,
        ) in zip(
            request.prompts,
            request.height,
            request.width,
            request.num_inference_steps,
            request.guidance_scale,
            request.num_images_per_prompt,
            request.negative_prompts,
            strict=False,
        ):
            context: PixelContext = await tokenizer.new_context(
                PixelGenerationRequest(
                    request_id=RequestID(),
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                )
            )
            output = await pipeline.aexecute(context)
            outputs.append(output)

        if response_queue := pending_requests.get(request.id):
            response_queue.put(_Response(outputs=outputs))
