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
"""Smoke tests for the text-to-video benchmark task plumbing."""

from __future__ import annotations

from max.benchmark.benchmark_shared.config import (
    PIXEL_GENERATION_TASKS,
)
from max.benchmark.benchmark_shared.datasets.pixel_synthetic import (
    SyntheticPixelBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.types import (
    PixelGenerationImageOptions,
    PixelGenerationSampledRequest,
)
from max.benchmark.benchmark_shared.request import (
    PixelGenerationRequestFuncInput,
    _build_pixel_generation_payload,
)


def test_text_to_video_is_a_pixel_generation_task() -> None:
    assert "text-to-video" in PIXEL_GENERATION_TASKS


def test_synthetic_pixel_dataset_propagates_num_frames() -> None:
    dataset = SyntheticPixelBenchmarkDataset()
    dataset.dataset_name = "synthetic-pixel"
    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=None,
        benchmark_task="text-to-video",
        image_width=480,
        image_height=480,
        image_steps=8,
        num_frames=17,
    )
    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, PixelGenerationSampledRequest)
        # text-to-video has no input image
        assert request.input_image_paths == []
        assert request.image_options is not None
        assert request.image_options.num_frames == 17
        assert request.image_options.width == 480


def test_pixel_generation_payload_routes_video_to_video_provider_options() -> (
    None
):
    image_options = PixelGenerationImageOptions(
        width=480, height=480, steps=8, num_frames=17
    )
    func_input = PixelGenerationRequestFuncInput(
        model="some/model",
        session_id=None,
        prompt="A cat dancing",
        input_image_paths=None,
        api_url="http://localhost:8000/v1/responses",
        image_options=image_options,
    )
    payload = _build_pixel_generation_payload(func_input)
    assert "provider_options" in payload
    assert "video" in payload["provider_options"]
    assert "image" not in payload["provider_options"]
    video = payload["provider_options"]["video"]
    assert video["num_frames"] == 17
    assert video["width"] == 480
    assert video["height"] == 480
    assert video["steps"] == 8


def test_pixel_generation_payload_routes_image_when_no_num_frames() -> None:
    image_options = PixelGenerationImageOptions(
        width=1024, height=1024, steps=28
    )
    func_input = PixelGenerationRequestFuncInput(
        model="some/model",
        session_id=None,
        prompt="A cat",
        input_image_paths=None,
        api_url="http://localhost:8000/v1/responses",
        image_options=image_options,
    )
    payload = _build_pixel_generation_payload(func_input)
    assert "provider_options" in payload
    assert "image" in payload["provider_options"]
    assert "video" not in payload["provider_options"]
    image = payload["provider_options"]["image"]
    assert "num_frames" not in image
    assert image["width"] == 1024
