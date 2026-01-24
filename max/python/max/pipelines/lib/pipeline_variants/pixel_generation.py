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
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import numpy.typing as npt
from max.driver import load_devices
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineOutputsDict,
    PipelineTokenizer,
    PixelGenerationContextType,
    PixelGenerationInputs,
    PixelGenerationOutput,
    PixelGenerationRequest,
    RequestID,
)

from ..interfaces import PipelineModel
from ..interfaces.generate import GenerateMixin
from .utils import get_weight_paths

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class PixelGenerationPipeline(
    Pipeline[
        PixelGenerationInputs[PixelGenerationContextType], PixelGenerationOutput
    ],
    GenerateMixin[PixelGenerationContextType, PixelGenerationRequest],
    Generic[PixelGenerationContextType],
):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[PixelGenerationContextType]],
        eos_token_id: int,
        tokenizer: PipelineTokenizer[
            PixelGenerationContextType,
            npt.NDArray[np.integer[Any]],
            PixelGenerationRequest,
        ],
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles
        from max.graph.weights import load_weights as _load_weights

        self._pipeline_config = pipeline_config
        model_config = pipeline_config.model
        self._devices = load_devices(pipeline_config.model.device_specs)
        self._tokenizer = tokenizer
        # Diffusion pipelines do not rely on HuggingFace text configs for EOS.
        self._eos_token_id = set([eos_token_id])

        # Initialize Session.
        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Retrieve the weights repo id (falls back to model_path when unset).
        weight_paths: list[Path] = get_weight_paths(model_config)

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=self._pipeline_config.model.huggingface_config,
            encoding=self._pipeline_config.model.quantization_encoding,
            devices=self._devices,
            kv_cache_config=self._pipeline_config.model.kv_cache,
            weights=_load_weights(weight_paths),
            adapter=None,
            return_logits=None,
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
        """Return the tokenizer used for building contexts and decoding."""
        return self._tokenizer

    def execute(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> PipelineOutputsDict[PixelGenerationOutput]:
        model_inputs, flat_batch = self.prepare_batch(inputs.batch)
        if not flat_batch:
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
                model_inputs.get("num_images_per_prompt"),
                model_inputs.get("height"),
                model_inputs.get("width"),
                model_inputs.get("num_inference_steps"),
            )
            raise

        image_list = model_outputs.images
        num_images_per_prompt = int(model_inputs["num_images_per_prompt"])
        expected_images = len(flat_batch) * num_images_per_prompt
        if len(image_list) != expected_images:
            raise ValueError(
                "Unexpected number of images returned from pipeline: "
                f"expected {expected_images}, got {len(image_list)}."
            )

        responses: dict[RequestID, PixelGenerationOutput] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_images_per_prompt
            if num_images_per_prompt == 1:
                pixel_data = image_list[offset]
            else:
                pixel_data = np.stack(
                    image_list[offset : offset + num_images_per_prompt],
                    axis=0,
                )
            pixel_data = pixel_data.astype(np.float32, copy=False)
            responses[request_id] = PixelGenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                pixel_data=pixel_data,
            )

        return responses

    def prepare_batch(
        self,
        batch: dict[RequestID, PixelGenerationContextType],
    ) -> tuple[
        dict[str, Any], list[tuple[RequestID, PixelGenerationContextType]]
    ]:
        """Prepare batched model inputs for pixel generation execution."""
        # TODO: Implement batching method with PixelContext.
        raise NotImplementedError(
            "prepare_batch is not implemented for PixelGenerationPipeline yet."
        )
