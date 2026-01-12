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
"""MAX pipeline for model inference and generation (Pixel Generation variant)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import load_devices
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    Pipeline,
    PipelineOutputsDict,
    PipelineTokenizer,
    PixelGenerationContextType,
    PixelGenerationInputs,
    PixelGenerationOutput,
    PixelGenerationRequest,
    RequestID,
)

if TYPE_CHECKING:
    from ..config import PipelineConfig

from ..config_enums import RepoType
from ..hf_utils import download_weight_files
from ..interfaces import PipelineModel

logger = logging.getLogger("max.pipelines")


class PixelGenerationPipeline(
    Pipeline[
        PixelGenerationInputs[PixelGenerationContextType], PixelGenerationOutput
    ],
):
    """Pipeline for diffusion-based image generation models."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[PixelGenerationContextType]],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            PixelGenerationContextType,
            npt.NDArray[np.integer[Any]],
            PixelGenerationRequest,
        ],
    ) -> None:
        """Initialize an image generation pipeline instance.

        Args:
            pipeline_config: Configuration for the pipeline and runtime behavior.
            pipeline_model: Concrete model implementation to use for execution.
            eos_token_id: Not used for image generation, kept for interface compatibility.
            weight_adapters: Mapping from weights format to adapter implementation.
            tokenizer: Tokenizer implementation used to build contexts.
        """
        self._pipeline_config = pipeline_config
        self._devices = load_devices(pipeline_config.model_config.device_specs)
        self._weight_adapters = weight_adapters
        self._tokenizer = tokenizer

        # Initialize Session.
        from max.engine import InferenceSession

        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        if not self._pipeline_config.model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        weight_model_id = (
            self._pipeline_config.model_config._weights_repo_id
            if self._pipeline_config.model_config._weights_repo_id
            else self._pipeline_config.model_config.model_path
        )

        weight_paths: list[Path] = []
        if (
            self._pipeline_config.model_config.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            weight_paths = download_weight_files(
                huggingface_model_id=weight_model_id,
                filenames=[
                    str(x)
                    for x in self._pipeline_config.model_config.weight_path
                ],
                revision=self._pipeline_config.model_config.huggingface_weight_revision,
                force_download=self._pipeline_config.model_config.force_download,
            )
        else:
            weight_paths = [
                self._pipeline_config.model_config.model_path / x
                for x in self._pipeline_config.model_config.weight_path
            ]

        from max.graph.weights import load_weights as _load_weights
        from max.graph.weights import weights_format as _weights_format

        # For image generation, we still need kv_cache_config for the text encoder
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=self._pipeline_config.model_config.huggingface_config,
            encoding=self._pipeline_config.model_config.quantization_encoding,
            devices=self._devices,
            kv_cache_config=self._pipeline_config.model_config.kv_cache_config,
            weights=_load_weights(weight_paths),
            adapter=self._weight_adapters.get(
                _weights_format(weight_paths), None
            ),
            return_logits=None,  # No logits for image generation
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        PixelGenerationContextType,
        npt.NDArray[np.integer[Any]],
        PixelGenerationRequest,
    ]:
        """Return the tokenizer used for building contexts."""
        return self._tokenizer

    @property
    def kv_managers(self) -> list[Any]:
        """Return empty list - image generation doesn't use KV cache."""
        return []

    def execute(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> PipelineOutputsDict[PixelGenerationOutput]:
        """Execute the image generation pipeline.

        For diffusion models, this runs the full denoising loop and returns
        the generated images.

        Args:
            inputs: The batch of image generation contexts to process.

        Returns:
            Dictionary mapping request IDs to their generated images.
        """
        from max.interfaces import GenerationStatus
        from max.pipelines.architectures.z_image_module_v3.model import (
            ZImageInputs,
        )

        results: dict[RequestID, PixelGenerationOutput] = {}

        # Flatten batches
        for batch in inputs.batches:
            for request_id, context in batch.items():
                # Construct model inputs from context
                model_inputs = ZImageInputs(
                    prompt=context.prompt,
                    height=context.height,
                    width=context.width,
                    num_inference_steps=context.num_inference_steps,
                    guidance_scale=context.guidance_scale,
                    negative_prompt=context.negative_prompt,
                    num_images_per_prompt=context.num_images_per_prompt,
                )

                # Execute the diffusion pipeline
                model_outputs = self._pipeline_model.execute(model_inputs)

                # Mark context as done
                context.status = GenerationStatus.END_OF_SEQUENCE

                # Convert driver tensor to numpy array
                # Raw output is in [-1, 1] range with NCHW format
                # Postprocessing (normalization + transpose) happens in serve pipeline
                image_np = model_outputs.hidden_states.to_numpy()

                results[request_id] = PixelGenerationOutput(
                    final_status=GenerationStatus.END_OF_SEQUENCE,
                    steps_executed=context.num_inference_steps,
                    pixel_data=image_np,
                )

        return results

    def release(self, request_id: RequestID) -> None:
        """Release resources for a completed request.

        For image generation, there's no KV cache to release.
        """
        pass  # No KV cache to release for image generation
