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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
            "Set to 1.0 to disable CFG. Some distilled or guidance-light models "
            "may prefer lower values. Defaults to 3.5."
        ),
        ge=0.0,
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
        description="The width of the generated image in pixels. Must be at least 128 and a multiple of 16.",
        ge=128,
    )

    height: int | None = Field(
        None,
        description="The height of the generated image in pixels. Must be at least 128 and a multiple of 16.",
        ge=128,
    )

    # Maximum total pixel area (e.g. 1024x1024). Requests exceeding this
    # would allocate huge latent tensors and risk OOM on the GPU.
    _MAX_PIXEL_AREA: int = 1024 * 1024

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "ImageProviderOptions":
        for name, value in [("width", self.width), ("height", self.height)]:
            if value is not None and value % 16 != 0:
                raise ValueError(
                    f"{name} must be a multiple of 16, got {value}"
                )

        if (
            self.width is not None
            and self.height is not None
            and self.width * self.height > self._MAX_PIXEL_AREA
        ):
            raise ValueError(
                f"width * height ({self.width} * {self.height} ="
                f" {self.width * self.height}) exceeds the maximum"
                f" pixel area of {self._MAX_PIXEL_AREA}"
            )

        return self

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

    residual_threshold: float | None = Field(
        None,
        description=(
            "Relative difference threshold for first-block cache (FBCache) "
            "reuse during denoising. Lower values skip fewer steps (higher "
            "quality, slower). None uses the model-specific default."
        ),
        gt=0.0,
    )

    strength: float = Field(
        0.6,
        description=(
            "Image-to-image strength. Must be in (0, 1]. "
            "Higher values add more noise and preserve less from input image. "
            "Ignored for text-to-image requests."
        ),
        gt=0.0,
        le=1.0,
    )

    cfg_normalization: bool = Field(
        False,
        description=(
            "Enable CFG output renormalization when supported by the selected model. "
            "When enabled, the guided prediction norm is clipped to the positive "
            "prediction norm."
        ),
    )

    cfg_truncation: float = Field(
        1.0,
        description=(
            "CFG truncation threshold in normalized time when supported by the selected model. "
            "CFG is disabled for steps where t_norm > cfg_truncation."
        ),
        gt=0.0,
    )

    output_format: str = Field(
        "jpeg",
        description=(
            "The image format to use for encoding the output (e.g., 'jpeg', "
            "'png', 'webp'). Defaults to 'jpeg'."
        ),
    )
