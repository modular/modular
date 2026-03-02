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
"""Provider-specific options for MAX platform and modalities."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .max import MaxProviderOptions
from .modality import ImageProviderOptions, VideoProviderOptions


class ProviderOptions(BaseModel):
    """Container for all provider-specific options.

    Includes both universal MAX options and modality-specific options.
    All options are validated at the API layer.

    Example:
        {
            "max": {"target_endpoint": "instance-123"},
            "image": {"width": 1024, "height": 768}
        }
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max: MaxProviderOptions | None = Field(
        None,
        description="Universal MAX platform options.",
    )

    image: ImageProviderOptions | None = Field(
        None,
        description="Image generation modality options.",
    )

    video: VideoProviderOptions | None = Field(
        None,
        description="Video generation modality options.",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_video_into_image(cls, data: Any) -> Any:
        """When only video options are provided, backfill image options from them.

        The pixel-generation tokenizer reads from ``provider_options.image`` for
        parameters that are shared between image and video pipelines (guidance_scale,
        height, width, steps, negative_prompt).  When a caller supplies only
        ``video`` options we transparently propagate those shared fields so the
        tokenizer can find them, while preserving any explicitly-set ``image``
        options.
        """
        if not isinstance(data, dict):
            return data
        video = data.get("video")
        if video is not None and data.get("image") is None:
            if isinstance(video, dict):
                negative_prompt = video.get("negative_prompt")
                height = video.get("height")
                width = video.get("width")
                steps = video.get("steps") or 50
                guidance_scale = video.get("guidance_scale") or 3.5
            else:
                negative_prompt = video.negative_prompt
                height = video.height
                width = video.width
                steps = video.steps if video.steps is not None else 50
                guidance_scale = (
                    video.guidance_scale
                    if video.guidance_scale is not None
                    else 3.5
                )
            data["image"] = ImageProviderOptions(
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
            )
        return data

    # Add more modality fields here as needed:
    # text: TextModalityOptions | None = None
    # tts: TTSModalityOptions | None = None


__all__ = [
    "ImageProviderOptions",
    "MaxProviderOptions",
    "ProviderOptions",
    "VideoProviderOptions",
]
