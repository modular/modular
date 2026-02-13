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
"""Image generation modality provider options."""

from pydantic import BaseModel, ConfigDict, Field


class ImageProviderOptions(BaseModel):
    """Options specific to image generation pipelines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    negative_prompt: str | None = Field(
        None,
        description=(
            "A text description of what to exclude from the generated image. "
            "Used to guide the generation away from unwanted elements."
        ),
    )

    secondary_prompt: str | None = Field(
        None,
        description="The second text prompt to generate images for.",
    )

    secondary_negative_prompt: str | None = Field(
        None,
        description="The second negative prompt to guide what NOT to generate.",
    )

    guidance_scale: float = Field(
        3.5,
        description=(
            "Guidance scale for classifier-free guidance. "
            "Higher values make the generation follow the prompt more closely. "
            "Set to 1.0 to disable CFG. Defaults to 3.5."
        ),
        gt=0.0,
    )

    true_cfg_scale: float = Field(
        1.0,
        description=(
            "True classifier-free guidance scale. "
            "True CFG is enabled when true_cfg_scale > 1.0 and negative_prompt is provided. "
            "Defaults to 1.0."
        ),
        gt=0.0,
    )

    width: int | None = Field(
        None,
        description="The width of the generated image in pixels.",
        gt=0,
    )

    height: int | None = Field(
        None,
        description="The height of the generated image in pixels.",
        gt=0,
    )

    steps: int = Field(
        50,
        description=(
            "The number of denoising steps. More steps generally produce higher quality "
            "results but take longer to generate. Defaults to 50."
        ),
        gt=0,
    )

    num_images: int = Field(
        1,
        description="The number of images to generate. Defaults to 1.",
        ge=1,
    )
