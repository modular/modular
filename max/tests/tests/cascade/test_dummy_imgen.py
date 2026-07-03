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
"""Functional tests for the dummy image-generation pipeline."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import numpy as np
import pytest
from max.experimental.cascade import ImageGenRequest, LocalRuntime
from max.tests.tests.cascade.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from PIL import Image


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


@pytest.mark.asyncio
async def test_imgen_pipeline(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_imgen_pipeline()
    await pipeline.deploy(runtime)

    req = ImageGenRequest(
        width=64,
        height=64,
        num_steps=3,
        output_format="JPEG",
    )
    image_data = await pipeline.generate_image(req, "a beautiful sunset")

    raw = (
        image_data.tobytes()
        if isinstance(image_data, np.ndarray)
        else image_data
    )
    image = Image.open(io.BytesIO(raw))
    assert image.size[0] > 0
    assert image.size[1] > 0


@pytest.mark.asyncio
async def test_imgen_pipeline_streaming(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_imgen_pipeline()
    await pipeline.deploy(runtime)

    num_steps = 3
    req = ImageGenRequest(
        width=64,
        height=64,
        num_steps=num_steps,
        output_format="JPEG",
    )

    frames = [
        frame async for frame in pipeline.generate_image_streaming(req, "a cat")
    ]
    # The denoiser emits one frame per denoising step and the downstream
    # streaming workers forward the stream verbatim.
    assert len(frames) == num_steps
    for frame in frames:
        assert isinstance(frame, bytes)
        image = Image.open(io.BytesIO(frame))
        assert image.size == (64, 64)
