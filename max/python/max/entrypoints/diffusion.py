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

import queue
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event, Thread

import tqdm
from max.interfaces import (
    GenerationStatus,
    PipelineTask,
    PixelGenerationInputs,
    PixelGenerationOutput,
    PixelGenerationPipeline,
    PixelGenerationRequest,
    RequestID,
)
from max.interfaces.pipeline_variants.text_generation import (
    TextGenerationRequestMessage,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig


@dataclass
class PixelContext:
    """Concrete implementation of PixelGenerationContext protocol."""

    request_id: RequestID
    prompt: str | None = None
    negative_prompt: str | None = None
    messages: list[TextGenerationRequestMessage] | None = None
    max_text_encoder_length: int = 512
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    num_images_per_prompt: int = 1
    seed: int | None = None
    model_name: str = ""
    true_cfg_scale: float = 1.0
    status: GenerationStatus = field(default=GenerationStatus.ACTIVE)

    @property
    def is_done(self) -> bool:
        return self.status.is_done


@dataclass
class _PixelBatchRequest:
    """Internal request for batching multiple pixel generation requests."""

    id: RequestID
    prompts: Sequence[str]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int
    use_tqdm: bool = True
    negative_prompts: Sequence[str | None] | None = None


@dataclass
class _PixelBatchResponse:
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
    _thread_control: _ThreadControl
    _worker_thread: Thread
    _request_queue: queue.Queue[_PixelBatchRequest]
    _pending_requests: dict[RequestID, queue.Queue[_PixelBatchResponse]]

    # Configuration
    pipeline_config: PipelineConfig
    model_name: str

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        """Initialize the image generator.

        Args:
            pipeline_config: Configuration specifying the model and parameters.
        """
        self.pipeline_config = pipeline_config
        self.model_name = pipeline_config.model.model_path

        # Initialize thread control and queues
        self._thread_control = _ThreadControl()
        self._request_queue = queue.Queue()
        self._pending_requests = {}

        # Start background worker
        self._worker_thread = Thread(
            target=_run_worker,
            args=(
                self._thread_control,
                self.pipeline_config,
                self._request_queue,
                self._pending_requests,
            ),
            daemon=True,
        )
        self._worker_thread.start()

        # Wait for worker to be ready
        self._thread_control.ready.wait()

    def __del__(self) -> None:
        """Clean up resources."""
        self._thread_control.cancel.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def generate(
        self,
        prompts: str | Sequence[str],
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        use_tqdm: bool = True,
    ) -> _PixelBatchResponse:
        """Generate images from text prompts.

        This method is thread-safe and can be called from multiple threads.

        Args:
            prompts: Single prompt string or sequence of prompts.
            height: Image height in pixels.
            width: Image width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images per prompt.
            use_tqdm: Show progress bar.

        Returns:
            _PixelBatchResponse containing the generated PIL Images.

        Example:
            ```python
            result = generator.generate(
                "A cat sitting on a couch",
                height=1024,
                width=1024,
                num_inference_steps=30,
            )
            result.images[0].save("cat.png")
            ```
        """
        # Normalize prompts to sequence
        if isinstance(prompts, str):
            prompts = [prompts]

        # Create internal request
        request = _PixelBatchRequest(
            id=RequestID(),
            prompts=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            use_tqdm=use_tqdm,
        )

        # Submit request and wait for response
        return self._submit_and_wait(request)

    def create(
        self,
        request: PixelGenerationRequest,
    ) -> PixelGenerationOutput:
        """Generate images using OpenAI-compatible request format.

        Args:
            request: OpenAI-compatible image generation request.

        Returns:
            OpenAI-compatible response with base64-encoded images.
        """
        raise NotImplementedError("Not implemented.")

    def _submit_and_wait(
        self, request: _PixelBatchRequest
    ) -> _PixelBatchResponse:
        """Submit a request to the queue and wait for response."""
        response_queue: queue.Queue[_PixelBatchResponse] = queue.Queue()
        self._pending_requests[request.id] = response_queue

        try:
            self._request_queue.put_nowait(request)
            response = response_queue.get()
            return response
        finally:
            self._pending_requests.pop(request.id, None)

    @classmethod
    def from_model(cls, model: str, **kwargs) -> PixelGenerator:
        """Create a PixelGenerator from a model identifier.

        Args:
            model: Model identifier (e.g., "black-forest-labs/FLUX.1-schnell").
            **kwargs: Additional PipelineConfig arguments.

        Returns:
            Configured PixelGenerator instance.

        Example:
            ```python
            pipeline = PixelGenerator.from_model(
                "black-forest-labs/FLUX.1-schnell"
            )
            ```
        """
        config = PipelineConfig(model=model, **kwargs)
        return cls(config)


def _run_worker(
    thread_control: _ThreadControl,
    pipeline_config: PipelineConfig,
    request_queue: queue.Queue[_PixelBatchRequest],
    pending_requests: Mapping[RequestID, queue.Queue[_PixelBatchResponse]],
) -> None:
    """Background worker that processes image generation requests.

    This function runs in a separate thread and continuously processes
    requests from the queue until cancellation is signaled.
    """
    # Load the pipeline
    _, model_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
        task=PipelineTask.PIXEL_GENERATION,
    )
    pipeline = model_factory()

    # Signal that we're ready
    thread_control.ready.set()

    # Main processing loop
    while not thread_control.cancel.is_set():
        try:
            request = request_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        # Process the request
        outputs = _process_request(pipeline, request)

        # Send response
        if response_queue := pending_requests.get(request.id):
            response_queue.put(_PixelBatchResponse(outputs=outputs))


def _process_request(
    pipeline: PixelGenerationPipeline,
    request: _PixelBatchRequest,
) -> list[PixelGenerationOutput]:
    """Process a single pixel generation request.

    Args:
        pipeline: The pixel generation pipeline (e.g., PixelGenerationPipeline).
        request: The request to process.

    Returns:
        List of PixelGenerationOutput.
    """
    # Handle negative prompts
    negative_prompts = request.negative_prompts
    if negative_prompts is None or len(request.prompts) != len(
        negative_prompts
    ):
        if negative_prompts is None or len(negative_prompts) == 0:
            negative_prompts = [None] * len(request.prompts)
        else:
            raise ValueError(
                "Number of prompts and negative prompts must be the same."
            )

    # TODO: temp hard coding for true cfg scale. Need to be removed.
    if all(neg is not None for neg in negative_prompts):
        true_cfg_scale = 1.0
    else:
        true_cfg_scale = 4.0

    prompt_list = list(zip(request.prompts, negative_prompts, strict=False))
    if request.use_tqdm:
        prompt_list = list(tqdm.tqdm(prompt_list, desc="Generating images"))

    # Generate images for each prompt using the internal diffusion pipeline
    batch: dict[RequestID, PixelContext] = {}
    for prompt, negative_prompt in prompt_list:
        request_id = RequestID()
        batch[request_id] = PixelContext(
            request_id=request_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images_per_prompt,
            true_cfg_scale=true_cfg_scale,
        )

    inputs = PixelGenerationInputs(batch=batch)

    outputs: list[PixelGenerationOutput] = pipeline.execute(inputs)

    return outputs
