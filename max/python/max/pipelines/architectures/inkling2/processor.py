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
"""Inkling 2 processor: token ids plus image pixel blocks.

The output has the shape :class:`~max.pipelines.lib.TextAndVisionTokenizer`
reads. The chat template emits one ``<|unused_200054|>`` per image, between
``<|content_image|>`` and ``<|end_message|>``, and this grows it into one token
per image block.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import huggingface_hub
from PIL import Image
from transformers import PreTrainedTokenizerBase

from .image_processor import Inkling2ImageProcessor

PROCESSOR_CONFIG_FILE = "processor_config.json"


class Inkling2Processor:
    """Stands in for transformers' ``Inkling2Processor``."""

    def __init__(
        self,
        delegate: PreTrainedTokenizerBase,
        processor_config: Mapping[str, Any],
    ) -> None:
        """Builds the image processor from ``processor_config.json``.

        Raises:
            ValueError: If the file names an image token the tokenizer lacks.
        """
        self.delegate = delegate
        self.image_processor = Inkling2ImageProcessor(
            processor_config["image_processor"]["size"]["height"]
        )
        image_token = processor_config["image_token"]
        vocab = delegate.get_vocab()
        if image_token not in vocab:
            raise ValueError(
                f"{PROCESSOR_CONFIG_FILE} names {image_token!r} as the image "
                "token, but the tokenizer vocabulary has no such entry"
            )
        self.image_token_id = int(vocab[image_token])

    def apply_chat_template(self, messages: Any, **options: Any) -> str:
        """Renders the conversation with the tokenizer's chat template."""
        templated = self.delegate.apply_chat_template(messages, **options)
        assert isinstance(templated, str)
        return templated

    def __call__(
        self,
        *,
        text: str | Sequence[int],
        images: Sequence[Image.Image] | None = None,
        add_special_tokens: bool = True,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        """Encodes ``text`` and expands every image placeholder.

        Returns:
            ``input_ids`` as ``[ids]`` and ``pixel_values`` as ``[[per-image
            [tokens, side, side, 3] arrays]]``, batched like a HuggingFace
            processor's output.
        """
        del unused_kwargs
        token_ids = (
            self.delegate.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(text, str)
            else list(text)
        )
        images = images or []
        blocks = [self.image_processor(image) for image in images]
        return {
            "input_ids": [
                expand_image_placeholders(
                    token_ids,
                    self.image_token_id,
                    [len(image_blocks) for image_blocks in blocks],
                )
            ],
            "pixel_values": [blocks],
        }


def load_processor_config(
    model_path: str, revision: str | None = None
) -> dict[str, Any]:
    """Reads ``processor_config.json`` out of the checkpoint."""
    local = Path(model_path) / PROCESSOR_CONFIG_FILE
    if not local.is_file():
        local = Path(
            huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=PROCESSOR_CONFIG_FILE,
                revision=revision,
            )
        )
    return json.loads(local.read_text())


def expand_image_placeholders(
    token_ids: Sequence[int], placeholder_id: int, patch_counts: Sequence[int]
) -> list[int]:
    """Grows each single placeholder token into that image's run of patches.

    The chat template emits one placeholder per image; the run length is only
    known once the image has been through the processor.
    """
    markers = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == placeholder_id
    ]
    if len(markers) != len(patch_counts):
        raise ValueError(
            f"prompt carries {len(markers)} image placeholder(s) but "
            f"{len(patch_counts)} image input(s) were provided"
        )
    # Downstream recovers each image's run as a maximal contiguous stretch of
    # placeholder tokens, so back-to-back runs would fuse into one image.
    if any(second - first == 1 for first, second in pairwise(markers)):
        raise ValueError(
            "prompt places two image placeholders back to back; separate "
            "consecutive images with at least one text token"
        )
    expanded = list(token_ids)
    # Back to front, so the untouched markers keep their indices.
    for index, count in reversed(list(zip(markers, patch_counts, strict=True))):
        expanded[index : index + 1] = [placeholder_id] * count
    return expanded
