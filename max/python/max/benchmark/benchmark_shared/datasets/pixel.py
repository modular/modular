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

"""Base class for pixel generation datasets."""

from __future__ import annotations

import dataclasses

from .local import LocalBenchmarkDataset
from .types import (
    PixelGenerationImageOptions,
    PixelGenerationSampledRequest,
)


class PixelBenchmarkDataset(LocalBenchmarkDataset):
    """Base class for pixel generation benchmark datasets."""

    def _build_image_options(
        self,
        *,
        image_width: int | None = None,
        image_height: int | None = None,
        image_steps: int | None = None,
        image_guidance_scale: float | None = None,
        image_negative_prompt: str | None = None,
        image_seed: int | None = None,
        num_frames: int | None = None,
        video_request_schema: str | None = None,
        video_duration_seconds: int | None = None,
        video_aspect_ratio: str | None = None,
        video_short_edge: int | None = None,
        keyframes: str | None = None,
    ) -> PixelGenerationImageOptions | None:
        options = PixelGenerationImageOptions(
            width=image_width,
            height=image_height,
            steps=image_steps,
            guidance_scale=image_guidance_scale,
            negative_prompt=image_negative_prompt,
            seed=image_seed,
            num_frames=num_frames,
            # ``default`` is the body every backend already sends, so it stays
            # unset here and leaves the option object None when nothing else is
            # set (the payload builders treat None as the default body).
            request_schema=(
                None
                if video_request_schema in (None, "default")
                else video_request_schema
            ),
            duration_seconds=video_duration_seconds,
            aspect_ratio=video_aspect_ratio,
            short_edge=video_short_edge,
            keyframes=keyframes,
        )
        if all(value is None for value in dataclasses.asdict(options).values()):
            return None
        return options

    def _build_request(
        self,
        prompt: str,
        image_options: PixelGenerationImageOptions | None,
        input_image_paths: list[str] | None = None,
    ) -> PixelGenerationSampledRequest:
        return PixelGenerationSampledRequest(
            prompt_formatted=prompt,
            prompt_len=0,
            output_len=None,
            encoded_images=[],
            ignore_eos=True,
            input_image_paths=input_image_paths or [],
            image_options=image_options,
        )
