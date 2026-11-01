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
"""Runs AutoPipeline on conversations with and without an image on a GPU."""

from __future__ import annotations

import huggingface_hub
import pytest
from max.pipelines import AutoPipeline, AutoProcessor
from PIL import Image

MODEL = "google/gemma-3-4b-it"


@pytest.fixture(autouse=True)
def _skip_if_not_cached() -> None:
    if huggingface_hub.constants.HF_HUB_OFFLINE:
        # Not every GPU runner's Hugging Face cache holds every model yet.
        try:
            huggingface_hub.snapshot_download(MODEL, local_files_only=True)
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pytest.skip(f"{MODEL} is not in this runner's cache.")


def test_a_conversation_with_an_image_and_one_without() -> None:
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
    processor = AutoProcessor(MODEL)
    with AutoPipeline(MODEL, max_batch_size=2, max_length=1024) as pipe:
        texts = pipe(conversations, max_new_tokens=16, top_k=1)
        outputs = pipe.model.generate(
            **processor(conversations), max_new_tokens=16, top_k=1
        )
    assert texts == processor.batch_decode(
        [
            sequence[length:]
            for sequence, length in zip(
                outputs.sequences, outputs.prompt_lengths, strict=True
            )
        ],
        skip_special_tokens=True,
    )
    image_text, text_text = texts
    assert "red" in image_text.lower()
    assert "4" in text_text
