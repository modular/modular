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
from typing import Any

from max.interfaces import (
    PixelGenerationInputs,
    PixelGenerationOutput,
    RequestID,
)
from max.interfaces.request import OpenResponsesRequest
from max.pipelines import PixelContext, PixelGenerationPipeline
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.lib.tokenizer import PixelGenerationTokenizer


def normalize_sequence(value: Any, length: int) -> Sequence[Any]:
    if not isinstance(value, Sequence):
        value = [value] * length
    return value


@dataclass
class _Request:
    """Internal request for batching multiple pixel generation requests."""

    id: RequestID
    model_name: str
    prompts: str | list[str]
    height: list[int] | int = 1024
    width: list[int] | int = 1024
    num_inference_steps: list[int] | int = 50
    guidance_scale: list[float] | float = 3.5
    num_visuals_per_prompt: list[int] | int = 1
    negative_prompts: list[str | None] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.prompts, str):
            self.prompts = [self.prompts]
        length = len(self.prompts)
        self.height = normalize_sequence(self.height, length)
        self.width = normalize_sequence(self.width, length)
        self.num_inference_steps = normalize_sequence(
            self.num_inference_steps, length
        )
        self.guidance_scale = normalize_sequence(self.guidance_scale, length)
        self.num_visuals_per_prompt = normalize_sequence(
            self.num_visuals_per_prompt, length
        )
        self.negative_prompts = normalize_sequence(
            self.negative_prompts, length
        )


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
        model_name: str,
        prompts: str | Sequence[str],
        *,
        negative_prompts: Sequence[str | None] | None = None,
        height: list[int] | None = None,
        width: list[int] | None = None,
        num_inference_steps: list[int] | None = None,
        guidance_scale: list[float] | None = None,
        num_visuals_per_prompt: list[int] | None = None,
    ) -> _Response:
        """Generate images from text prompts.

        This method is thread-safe and can be called from multiple threads.

        Args:
            prompts: Single prompt string or sequence of prompts.
            negative_prompts: Single negative prompt string or sequence of negative prompts.
            height: Image height in pixels.
            width: Image width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_visuals_per_prompt: Number of images per prompt.

        Returns:
            _Response containing the generated PIL Images.

        """
        request_dict = {
            "id": RequestID(),
            "model_name": model_name,
            "prompts": prompts,
            "negative_prompts": negative_prompts,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_visuals_per_prompt": num_visuals_per_prompt,
        }
        request_dict = {k: v for k, v in request_dict.items() if v is not None}  # type: ignore

        # Create internal request
        request = _Request(**request_dict)

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

        batch: {RequestID: PixelContext} = {}
        for (
            prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            num_visuals_per_prompt,
            negative_prompt,
        ) in zip(
            request.prompts,
            request.height,
            request.width,
            request.num_inference_steps,
            request.guidance_scale,
            request.num_visuals_per_prompt,
            request.negative_prompts,
            strict=False,
        ):
            request_id = RequestID()
            request = OpenResponsesRequest(
                request_id=request_id,
                model_name=request.model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_visuals_per_prompt=num_visuals_per_prompt,
            )
            context: PixelContext = await tokenizer.new_context(request)
            batch[request_id] = context
        inputs = PixelGenerationInputs(batch=batch)
        outputs = await asyncio.to_thread(pipeline.execute, inputs)

        if response_queue := pending_requests.get(request.id):
            response_queue.put(_Response(outputs=outputs))
