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

from max.interfaces import (
    ImageGenerationInputs,
    ImageGenerationOutput,
    PipelineTask,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig


class DiffusionPipeline:
    """Entrypoint for image-generation diffusion pipelines."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        # NOTE: Currently, this entrypoint is implemented minimally
        # for offline image generation.
        # It will be developed further to support serving as well.
        self.pipeline_config = pipeline_config
        _, model_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=PipelineTask.IMAGE_GENERATION,
        )
        self.pipeline = model_factory()

    def __call__(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        true_cfg_scale: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
    ) -> ImageGenerationOutput:
        """Generate images from a prompt with the configured pipeline."""
        # TODO: consider all possible diffusion tasks,
        # e.g. T2I, I2I, T2V, I2V, V2V.
        inputs = ImageGenerationInputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )
        pipeline_output: ImageGenerationOutput = self.pipeline.execute(inputs)
        return pipeline_output
