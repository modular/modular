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
"""Runs AutoModel.generate on FLUX.2 Klein, an image model, on a GPU."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Iterator

import huggingface_hub
import numpy as np
import pytest
from max.pipelines import AutoModel
from max.pipelines.modeling.types import PixelGenerationInputs, RequestID
from max.pipelines.request import (
    OpenResponsesRequest,
    OpenResponsesRequestBody,
    OutputImageContent,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from PIL import Image

MODEL = "black-forest-labs/FLUX.2-klein-4B"
PROMPTS = ["a red fox in snow", "a lighthouse at dusk"]
SIZE = 512
# The distilled Klein checkpoint runs exactly 4 steps.
STEPS = 4


@pytest.fixture(scope="module")
def model() -> Iterator[AutoModel]:
    if huggingface_hub.constants.HF_HUB_OFFLINE:
        # Not every GPU runner's Hugging Face cache holds every model yet.
        try:
            huggingface_hub.snapshot_download(MODEL, local_files_only=True)
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pytest.skip(f"{MODEL} is not in this runner's cache.")
    with AutoModel(MODEL) as model:
        yield model


def _generate(
    model: AutoModel, prompts: list[str], seed: int
) -> list[Image.Image]:
    """Returns each prompt's one image."""
    result = model.generate(
        prompt=prompts,
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        seed=seed,
    )
    assert [len(row) for row in result.images] == [1] * len(prompts)
    return [row[0] for row in result.images]


def _direct(model: AutoModel, prompt: str, seed: int) -> Image.Image:
    """Runs one request through the pipeline itself, as serve does."""
    pipeline = model._image_pipeline
    tokenizer = model._image_tokenizer
    assert pipeline is not None and tokenizer is not None
    request = OpenResponsesRequest(
        request_id=RequestID(),
        body=OpenResponsesRequestBody(
            model=MODEL,
            input=prompt,
            seed=seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    height=SIZE, width=SIZE, steps=STEPS, output_format="png"
                )
            ),
        ),
    )
    context = asyncio.run(tokenizer.new_context(request))
    output = pipeline.execute(
        PixelGenerationInputs(batch={context.request_id: context})
    )[context.request_id]
    [content] = output.output
    assert isinstance(content, OutputImageContent)
    assert content.image_data is not None
    return Image.open(io.BytesIO(base64.b64decode(content.image_data)))


def test_generates_seeded_images_like_the_pipeline(model: AutoModel) -> None:
    images = _generate(model, PROMPTS, seed=0)
    assert len(images) == len(PROMPTS)
    for image in images:
        assert image.size == (SIZE, SIZE)
        assert image.mode == "RGB"
    pixels = [np.asarray(image) for image in images]
    assert not np.array_equal(pixels[0], pixels[1])

    again = _generate(model, PROMPTS, seed=0)
    for expected, image in zip(pixels, again, strict=True):
        np.testing.assert_array_equal(np.asarray(image), expected)

    [other] = _generate(model, PROMPTS[:1], seed=1)
    assert not np.array_equal(np.asarray(other), pixels[0])

    np.testing.assert_array_equal(
        np.asarray(_direct(model, PROMPTS[0], seed=0)), pixels[0]
    )
