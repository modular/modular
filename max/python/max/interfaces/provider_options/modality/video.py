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

from pydantic import BaseModel, ConfigDict, Field


class VideoProviderOptions(BaseModel):
    """Options specific to video generation pipelines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    negative_prompt: str | None = Field(
        None,
        description=(
            "A text description of what to exclude from the generated video. "
            "Used to guide the generation away from unwanted elements."
        ),
    )

    width: int | None = Field(
        None,
        description="The width of the generated video in pixels.",
        gt=0,
    )

    height: int | None = Field(
        None,
        description="The height of the generated video in pixels.",
        gt=0,
    )

    steps: int | None = Field(
        None,
        description=(
            "The number of denoising steps. More steps generally produce higher quality "
            "results but take longer to generate."
        ),
        gt=0,
    )

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
