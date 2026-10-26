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
    InputImageContent,
    InputMessage,
    InputTextContent,
    OpenResponsesRequest,
    OpenResponsesRequestBody,
    OutputImageContent,
    UserMessage,
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
    model: AutoModel,
    prompts: list[str],
    seed: int,
    images: list[list[Image.Image]] | None = None,
) -> list[Image.Image]:
    """Returns each prompt's one image."""
    result = model.generate(
        prompt=prompts,
        images=images,
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        seed=seed,
    )
    assert [len(row) for row in result.images] == [1] * len(prompts)
    return [row[0] for row in result.images]


def _direct(
    model: AutoModel,
    prompt: str,
    seed: int,
    input_image: Image.Image | None = None,
) -> Image.Image:
    """Runs the request serve builds through the pipeline itself.

    An input image travels as a PNG data URI in the message, as in a
    ``/v1/responses`` request, rather than as ``new_context``'s argument.
    """
    pipeline = model._image_pipeline
    tokenizer = model._image_tokenizer
    assert pipeline is not None and tokenizer is not None
    message_input: str | list[InputMessage] = prompt
    if input_image is not None:
        buffer = io.BytesIO()
        input_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        message_input = [
            UserMessage(
                content=[
                    InputImageContent(
                        image_url=f"data:image/png;base64,{data}"
                    ),
                    InputTextContent(text=prompt),
                ]
            )
        ]
    request = OpenResponsesRequest(
        request_id=RequestID(),
        body=OpenResponsesRequestBody(
            model=MODEL,
            input=message_input,
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


def test_edits_an_input_image_like_the_pipeline(model: AutoModel) -> None:
    [source] = _generate(model, PROMPTS[:1], seed=0)
    prompt = "the same fox at night"
    [edited] = _generate(model, [prompt], seed=0, images=[[source]])
    assert edited.size == (SIZE, SIZE)
    assert edited.mode == "RGB"
    assert not np.array_equal(np.asarray(edited), np.asarray(source))
    np.testing.assert_array_equal(
        np.asarray(edited),
        np.asarray(_direct(model, prompt, seed=0, input_image=source)),
    )
