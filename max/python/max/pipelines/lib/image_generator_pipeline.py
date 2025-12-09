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
"""Image generator pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from max.interfaces import (
    ImageGenerationInputs,
    ImageGenerationOutput,
    Pipeline,
    RequestID,
)
from max.nn import ReturnLogits
from max.pipelines.core import TextAndVisionContext

if TYPE_CHECKING:
    from .config import PipelineConfig

from max.serve.telemetry.metrics import METRICS

from .interfaces import PipelineModel

ImageGeneratorPipelineType = Pipeline[
    ImageGenerationInputs[TextAndVisionContext], ImageGenerationOutput
]


class ImageGeneratorPipeline(ImageGeneratorPipelineType):
    """Converts text to image.

    This pipeline passes all of the work through to the PipelineModel.
    """

    @no_type_check
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel],
        **unused_kwargs,
    ) -> None:
        """Initializes the image generation pipeline.

        Args:
            pipeline_config: The configuration for the pipeline.
            pipeline_model: The pipeline model to use.
        """
        # Create the pipeline model.
        # None of the arguments are used except for the config.
        self.pipeline_model = pipeline_model(
            pipeline_config=pipeline_config,
            session=None,
            huggingface_config=None,
            encoding=None,
            devices=None,
            kv_cache_config=None,
            weights=None,
            adapter=None,
            return_logits=ReturnLogits.ALL,
        )
        assert hasattr(self.pipeline_model, "image_pipeline")
        self.image_pipeline = self.pipeline_model.image_pipeline

    def execute(
        self, inputs: ImageGenerationInputs[TextAndVisionContext]
    ) -> dict[RequestID, ImageGenerationOutput]:
        METRICS.input_tokens(
            sum(ctx.active_length for ctx in inputs.batch.values())
        )

        next_chunk = getattr(self.pipeline_model, "next_chunk")  # type: ignore[has-type]  # noqa: B009
        outputs = next_chunk(inputs.batch)
        METRICS.output_tokens(
            sum(output.steps_executed for output in outputs.values())
        )

        return outputs

    def release(self, request_id: RequestID) -> None:
        release = getattr(self.pipeline_model, "release")  # type: ignore[has-type]  # noqa: B009
        release(request_id)
