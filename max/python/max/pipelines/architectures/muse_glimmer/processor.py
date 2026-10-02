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

"""Muse Glimmer processor: token ids plus image patches, in the shape
:class:`~max.pipelines.lib.TextAndVisionTokenizer` reads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerBase

from .image_processor import MuseGlimmerImageProcessor


class MuseGlimmerProcessor:
    """Stands in for transformers' ``MuseGlimmerProcessor`` on images."""

    def __init__(
        self,
        delegate: PreTrainedTokenizerBase,
        processor_config: Mapping[str, Any],
    ) -> None:
        """Builds the image processor from ``processor_config.json``."""
        self.delegate = delegate
        self.image_processor = MuseGlimmerImageProcessor(
            processor_config["image_processor"]
        )
        ids = delegate.convert_tokens_to_ids(
            ["<|patch|>", "<|image_start|>", "<|image_end|>"]
        )
        assert isinstance(ids, list)
        self.image_token_id, self.image_start_id, self.image_end_id = ids

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
        """Encodes ``text`` and grows each ``<|patch|>`` into
        ``<|image_start|>``, ``h * w / 4`` patch tokens and ``<|image_end|>``,
        as transformers' ``replace_image_token`` does.

        Returns:
            ``input_ids`` as ``[ids]``, ``pixel_values`` as ``[[per-image
            [h * w, 1176] arrays]]`` and ``image_grid_thw`` as ``[[n, 3]]``,
            batched like a HuggingFace processor's output.
        """
        del unused_kwargs
        token_ids = (
            self.delegate.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(text, str)
            else list(text)
        )
        processed = [self.image_processor(image) for image in images or []]
        markers = [
            i for i, t in enumerate(token_ids) if t == self.image_token_id
        ]
        if len(markers) != len(processed):
            raise ValueError(
                f"prompt carries {len(markers)} <|patch|> placeholder(s) but "
                f"{len(processed)} image(s) were provided"
            )
        merge = self.image_processor.merge_size**2
        expanded = list(token_ids)
        # Back to front, so the untouched markers keep their indices.
        for index, (_, grid) in reversed(
            list(zip(markers, processed, strict=True))
        ):
            expanded[index : index + 1] = [
                self.image_start_id,
                *[self.image_token_id] * (int(grid.prod()) // merge),
                self.image_end_id,
            ]
        return {
            "input_ids": [expanded],
            "pixel_values": [[pixels for pixels, _ in processed]],
            "image_grid_thw": [
                np.stack([grid for _, grid in processed])
                if processed
                else np.zeros((0, 3), dtype=np.int64)
            ],
        }
