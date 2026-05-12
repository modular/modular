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
"""MAX pipeline for pixel generation using diffusion models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from max.driver import load_devices
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineOutputsDict,
    PixelGenerationContextType,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.generation import GenerationOutput
from max.interfaces.request.open_responses import (
    OutputImageContent,
    OutputVideoContent,
)

from ..interfaces.cache_mixin import DenoisingCacheConfig
from ..interfaces.diffusion_pipeline import DiffusionPipeline

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..pipeline_executor import PipelineExecutor

logger = logging.getLogger("max.pipelines")


class PixelGenerationPipeline(
    Pipeline[
        PixelGenerationInputs[PixelGenerationContextType], GenerationOutput
    ],
    Generic[PixelGenerationContextType],
):
    """Pixel generation pipeline for diffusion models.

    Args:
        pipeline_config: Configuration for the pipeline and runtime behavior.
        pipeline_model: The diffusion pipeline model class to instantiate.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[DiffusionPipeline]
        | type[PipelineExecutor[Any, Any, Any]],
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles
        from max.pipelines.lib.pipeline_executor import PipelineExecutor

        self._pipeline_config = pipeline_config
        # Use the first component's device_specs for session initialization.
        first_config = next(iter(pipeline_config.models.values()))
        self._devices = load_devices(first_config.device_specs)

        # Initialize Session.
        session = InferenceSession(devices=[*self._devices])

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        if issubclass(pipeline_model, PipelineExecutor):
            # Merge CLI-supplied cache_config into runtime so the executor
            # receives TaylorSeer / FBCache settings.
            if cache_config is not None:
                pipeline_config.runtime.denoising_cache = cache_config
            self._use_executor = True
            self._executor: PipelineExecutor[Any, Any, Any] | None = (
                pipeline_model(
                    manifest=pipeline_config.models,
                    session=session,
                    runtime_config=pipeline_config.runtime,
                )
            )
            self._pipeline_model: DiffusionPipeline | None = None
        else:
            self._use_executor = False
            self._executor = None
            # Weight paths are resolved per-component inside
            # _load_sub_models.
            self._pipeline_model = pipeline_model(
                pipeline_config=self._pipeline_config,
                session=session,
                devices=self._devices,
                weight_paths=[],
                cache_config=cache_config
                or pipeline_config.runtime.denoising_cache,
            )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    def execute(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> PipelineOutputsDict[GenerationOutput]:
        """Runs the pixel generation pipeline for the given inputs."""
        model_inputs, flat_batch = self.prepare_batch(inputs.batch)
        if not flat_batch or model_inputs is None:
            return {}

        if self._use_executor:
            assert self._executor is not None
            try:
                executor_outputs = self._executor.execute(model_inputs)
            except Exception:
                logger.error(
                    "Encountered an exception while executing pixel "
                    "batch (executor path): batch_size=%d",
                    len(flat_batch),
                )
                raise
            images = np.from_dlpack(executor_outputs.images)
            num_images_per_prompt = np.from_dlpack(
                model_inputs.num_images_per_prompt
            ).item()
            assert isinstance(num_images_per_prompt, int)
        else:
            assert self._pipeline_model is not None
            try:
                model_outputs = self._pipeline_model.execute(
                    model_inputs=model_inputs
                )
            except Exception:
                logger.error(
                    "Encountered an exception while executing pixel "
                    "batch: batch_size=%d, num_images_per_prompt=%s, "
                    "height=%s, width=%s, num_inference_steps=%s",
                    len(flat_batch),
                    model_inputs.num_images_per_prompt,
                    model_inputs.height,
                    model_inputs.width,
                    model_inputs.num_inference_steps,
                )
                raise
            images = model_outputs.images
            num_images_per_prompt = model_inputs.num_images_per_prompt

        expected_images = len(flat_batch) * num_images_per_prompt

        # Video output: shape [B, C, T, H, W]
        if isinstance(images, np.ndarray) and images.ndim == 5:
            video_responses: dict[RequestID, GenerationOutput] = {}
            for index, (request_id, _context) in enumerate(flat_batch):
                # video_clip shape: [C, T, H, W]
                video_clip = images[index]
                # Video decoders are expected to return uint8 pixel values.
                # Reorder to [T, H, W, C] and keep raw frames until the serving
                # layer decides how to encode them for the final response.
                frames = np.transpose(video_clip, (1, 2, 3, 0))
                video_responses[request_id] = GenerationOutput(
                    request_id=request_id,
                    final_status=GenerationStatus.END_OF_SEQUENCE,
                    output=[OutputVideoContent.from_numpy_frames(frames)],
                )
            return video_responses

        if images.shape[0] != expected_images:
            raise ValueError(
                "Unexpected number of images returned from pipeline: "
                f"expected {expected_images}, got {images.shape[0]}."
            )

        responses: dict[RequestID, GenerationOutput] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_images_per_prompt
            pixel_data = images[offset : offset + num_images_per_prompt]

            output_format = getattr(_context, "output_format", "jpeg")
            responses[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=[
                    OutputImageContent.from_numpy(img, format=output_format)
                    for img in pixel_data
                ],
            )

        return responses

    def prepare_batch(
        self,
        batch: dict[RequestID, PixelGenerationContextType],
    ) -> tuple[
        Any,
        list[tuple[RequestID, PixelGenerationContextType]],
    ]:
        """Prepare model inputs for pixel generation execution.

        Delegates to the pipeline model for model-specific input preparation.

        Args:
            batch: Dictionary mapping request IDs to their PixelContext objects.

        Returns:
            A tuple of:
                - Model inputs ready for execution, or None if batch is empty.
                - list: Flattened batch as (request_id, context) tuples for
                  response mapping.

        Raises:
            ValueError: If batch size is larger than 1 (not yet supported).
        """
        if not batch:
            return None, []

        # Flatten batch to list of (request_id, context) tuples
        flat_batch = list(batch.items())

        if len(flat_batch) > 1:
            raise ValueError(
                "Batching of different requests is not supported yet."
            )

        if self._use_executor:
            assert self._executor is not None
            contexts = [ctx for _rid, ctx in flat_batch]
            model_inputs = self._executor.prepare_inputs(contexts)
        else:
            assert self._pipeline_model is not None
            model_inputs = self._pipeline_model.prepare_inputs(flat_batch[0][1])
        return model_inputs, flat_batch

    def release(self, request_id: RequestID) -> None:
        """Release resources associated with a request.

        Args:
            request_id: The request ID to release resources for.
        """
        pass
