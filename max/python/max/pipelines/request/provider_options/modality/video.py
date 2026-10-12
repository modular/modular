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
"""Video generation modality provider options."""

from pydantic import Field

from .common import GeneratedMediaResponseFormat, PixelProviderOptionsBase


class VideoProviderOptions(PixelProviderOptionsBase):
    """Options specific to video generation pipelines.

    Inherits all generation options shared with other pixel modalities from
    :class:`PixelProviderOptionsBase`; adds video-specific fields
    (``frames_per_second``, ``num_frames``, ``guidance_scale_2``) and
    overrides the default response format.
    """

    frames_per_second: int | None = Field(
        None,
        description=(
            "The frame rate for video generation in frames per second (fps). "
            "Common values are 24, 30, or 60 fps."
        ),
        gt=0,
    )

    num_frames: int | None = Field(
        None,
        description=(
            "The number of frames to generate for video output. "
            "Total video duration equals num_frames / frames_per_second."
        ),
        gt=0,
    )

    guidance_scale_2: float | None = Field(
        None,
        description="Secondary guidance scale for boundary timestep switching.",
        gt=0.0,
    )

    response_format: GeneratedMediaResponseFormat = Field(
        GeneratedMediaResponseFormat.url,
        description=(
            "How generated videos are returned. Use 'url' for file-backed "
            "downloads or 'b64_json' for inline base64-encoded mp4 data."
        ),
    )
