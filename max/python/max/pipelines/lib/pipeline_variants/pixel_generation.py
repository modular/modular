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
from pathlib import Path
from typing import TYPE_CHECKING, Generic

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
from max.interfaces.request.open_responses import OutputImageContent

from ..interfaces.diffusion_pipeline import (
    DiffusionPipeline,
    PixelModelInputs,
)
from .utils import get_weight_paths

if TYPE_CHECKING:
    from ..config import PipelineConfig

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
        pipeline_model: type[DiffusionPipeline],
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles

        self._pipeline_config = pipeline_config
        model_config = pipeline_config.model
        self._devices = load_devices(pipeline_config.model.device_specs)

        # Initialize Session.
        session = InferenceSession(devices=self._devices)

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Download weights if required and get absolute weight paths.
        weight_paths: list[Path] = get_weight_paths(model_config)

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            weight_paths=weight_paths,
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

        try:
            model_outputs = self._pipeline_model.execute(
                model_inputs=model_inputs
            )
        except Exception:
            batch_size = len(flat_batch)
            logger.error(
                "Encountered an exception while executing pixel batch: "
                "batch_size=%d, num_images_per_prompt=%s, height=%s, width=%s, "
                "num_inference_steps=%s",
                batch_size,
                model_inputs.num_images_per_prompt,
                model_inputs.height,
                model_inputs.width,
                model_inputs.num_inference_steps,
            )
            raise

        images = model_outputs.images
        num_images_per_prompt = model_inputs.num_images_per_prompt
        expected_images = len(flat_batch) * num_images_per_prompt

        # Handle both numpy array (NHWC) and list of images
        if isinstance(images, np.ndarray):
            # images shape: (batch_size, H, W, C) or (batch_size, C, H, W)
            # Convert NCHW to NHWC if needed
            if images.ndim == 4 and images.shape[1] in (1, 3, 4):
                # Likely NCHW format, convert to NHWC
                images = np.transpose(images, (0, 2, 3, 1))
            # Denormalize from [-1, 1] to [0, 1] range
            images = (images * 0.5 + 0.5).clip(min=0.0, max=1.0)
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            # Denormalize each image from [-1, 1] to [0, 1] range
            image_list = [
                (np.asarray(img, dtype=np.float32) * 0.5 + 0.5).clip(
                    min=0.0, max=1.0
                )
                for img in images
            ]

        if len(image_list) != expected_images:
            raise ValueError(
                "Unexpected number of images returned from pipeline: "
                f"expected {expected_images}, got {len(image_list)}."
            )

        responses: dict[RequestID, GenerationOutput] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_images_per_prompt
            # Select images for this request (already in NHWC format)
            pixel_data = np.stack(
                image_list[offset : offset + num_images_per_prompt], axis=0
            ).astype(np.float32, copy=False)

            responses[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=[
                    OutputImageContent.from_numpy(img, format="png")
                    for img in pixel_data
                ],
            )

        return responses

    def prepare_batch(
        self,
        batch: dict[RequestID, PixelGenerationContextType],
    ) -> tuple[
        PixelModelInputs | None,
        list[tuple[RequestID, PixelGenerationContextType]],
    ]:
        """Prepare model inputs for pixel generation execution.

        Delegates to the pipeline model for model-specific input preparation.

        Args:
            batch: Dictionary mapping request IDs to their PixelContext objects.

        Returns:
            A tuple of:
                - PixelModelInputs | None: Inputs ready for model execution,
                  or None if batch is empty.
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

        model_inputs = self._pipeline_model.prepare_inputs(flat_batch[0][1])
        return model_inputs, flat_batch

    def release(self, request_id: RequestID) -> None:
        """Release resources associated with a request.

        Args:
            request_id: The request ID to release resources for.
        """
        pass
