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
"""Runs AutoModel and AutoProcessor on text and image rows with Gemma 3 4B."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from _auto_model_vision_gpu import (
    check_image_rows,
    check_pil_image_rows,
    check_text_rows,
    load,
)
from max.pipelines import AutoModel, AutoProcessor
from PIL import Image

# Gemma 3 expands each image into this many prompt tokens.
IMAGE_TOKENS = 256


@pytest.fixture(scope="module")
def model() -> Iterator[AutoModel]:
    yield from load("google/gemma-3-4b-it")


def test_text_prompts_match_the_tokenizer_built_rows(model: AutoModel) -> None:
    check_text_rows(model)


def test_pil_image_rows_match_the_bytes_rows(model: AutoModel) -> None:
    check_pil_image_rows(model)


def test_image_rows_match_the_tokenizer_built_rows(model: AutoModel) -> None:
    check_image_rows(model)


def test_auto_processor_inputs_decode_to_the_new_text(
    model: AutoModel,
) -> None:
    processor = AutoProcessor(model.pipeline_config.model.model_path)
    conversations: list[list[dict[str, object]]] = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (64, 64), "red"),
                    },
                    {
                        "type": "text",
                        "text": "What color is this image? Answer in one word.",
                    },
                ],
            }
        ],
        [{"role": "user", "content": "What is 2 + 2? Answer with a number."}],
    ]
    outputs = model.generate(
        **processor(conversations), max_new_tokens=16, top_k=1
    )
    image_length, text_length = outputs.prompt_lengths
    assert image_length > IMAGE_TOKENS > text_length
    new_tokens = [
        sequence[length:]
        for sequence, length in zip(
            outputs.sequences, outputs.prompt_lengths, strict=True
        )
    ]
    image_text, text_text = processor.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    assert "red" in image_text.lower()
    assert "4" in text_text
