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
from max.interfaces.request.open_responses import (
    OutputAudioContent,
    OutputImageContent,
    OutputVideoContent,
)

from ..interfaces.diffusion_pipeline import (  # type: ignore[import-not-found]
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
        session = InferenceSession(devices=[*self._devices])

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
                "batch_size=%d, num_visuals_per_prompt=%s, height=%s, width=%s, "
                "num_inference_steps=%s",
                batch_size,
                model_inputs.num_visuals_per_prompt,
                model_inputs.height,
                model_inputs.width,
                model_inputs.num_inference_steps,
            )
            raise

        image_list = getattr(model_outputs, "images", None)
        num_visuals_per_prompt = model_inputs.num_visuals_per_prompt

        # Handle image outputs (image-generation models) - gate on None to
        # avoid a NameError from the old `images` reference.
        if image_list is not None:
            if isinstance(image_list, np.ndarray):
                # images shape: (batch_size, H, W, C) or (batch_size, C, H, W)
                # Convert NCHW to NHWC if needed
                if image_list.ndim == 4 and image_list.shape[1] in (1, 3, 4):
                    image_list = np.transpose(image_list, (0, 2, 3, 1))
                # Denormalize from [-1, 1] to [0, 1] range
                image_list = (image_list * 0.5 + 0.5).clip(min=0.0, max=1.0)
                image_list = [image_list[i] for i in range(image_list.shape[0])]
            else:
                image_list = [
                    (np.asarray(img, dtype=np.float32) * 0.5 + 0.5).clip(
                        min=0.0, max=1.0
                    )
                    for img in image_list
                ]

            expected_images = len(flat_batch) * num_visuals_per_prompt
            if len(image_list) != expected_images:
                raise ValueError(
                    "Unexpected number of images returned from pipeline: "
                    f"expected {expected_images}, got {len(image_list)}."
                )

        video_output = getattr(model_outputs, "frames", None)

        # Convert MAX Tensor → numpy if the pipeline returned a Tensor.
        def _tensor_to_numpy(t: object) -> np.ndarray | None:
            if t is None or isinstance(t, np.ndarray):
                return t  # type: ignore[return-value]
            try:
                # MAX Tensor supports __dlpack__ / np.from_dlpack.
                from max.driver import CPU
                from max.dtype import DType

                return np.from_dlpack(t.cast(DType.float32).to(CPU()))  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                return t.to_numpy()  # type: ignore[union-attr]
            except Exception:
                logger.warning(
                    "Could not convert pipeline tensor output to numpy; skipping."
                )
                return None

        video_output = _tensor_to_numpy(video_output)
        audio_output = _tensor_to_numpy(getattr(model_outputs, "audio", None))

        frame_rate = int(getattr(model_inputs, "frame_rate", 24) or 24)
        audio_sample_rate = int(
            getattr(model_inputs, "audio_sampling_rate", 16000) or 16000
        )

        responses: dict[RequestID, GenerationOutput] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_visuals_per_prompt
            # Select images for this request (already in NHWC format)
            output_content = []
            if image_list is not None and len(image_list) > 0:
                pixel_data = image_list[
                    offset : offset + num_visuals_per_prompt
                ]
                pixel_data = pixel_data.astype(np.float32, copy=False)
                output_content.extend(
                    [
                        OutputImageContent.from_numpy(img, format="png")
                        for img in pixel_data
                    ]
                )

            # Video output - [batch, frames, height, width, channels] in [0,1].
            # Store as lossless numpy so the consumer can call encode_video
            # directly (mirrors the diffusers encode_video pattern).
            if video_output is not None:
                for i in range(num_visuals_per_prompt):
                    idx = offset + i
                    if idx < len(video_output):
                        output_content.append(
                            OutputVideoContent.from_numpy(
                                video_output[idx],
                                fps=frame_rate,
                                format="numpy",
                            )
                        )

            # Audio output - [batch, channels, samples] or [batch, samples].
            if audio_output is not None:
                for i in range(num_visuals_per_prompt):
                    idx = offset + i
                    if idx < len(audio_output):
                        output_content.append(
                            OutputAudioContent.from_numpy(
                                audio_output[idx],
                                sample_rate=audio_sample_rate,
                                format="wav",
                            )
                        )

            responses[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=output_content,
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
