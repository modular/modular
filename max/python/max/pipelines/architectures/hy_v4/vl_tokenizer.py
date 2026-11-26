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
"""Hy4 VL tokenizer for image prompts."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.tokenizer import (
    max_tokens_to_generate,
    open_image,
    run_with_default_executor,
)
from max.pipelines.modeling.types import (
    TextGenerationRequest,
)
from max.support.image import hash_image

from .image_processor import HYV4VLImageProcessor
from .tokenizer import HYV4VLTokenizer


class HYV4VLImageTokenizer(HYV4VLTokenizer):
    """Renders Hy4 VL chats and preprocesses their images.

    The rendered prompt holds one image pad token per image. Each pad grows
    into the image's ``rows x (columns + 1)`` run, one newline per merged
    row, and the image's patches and patch grid go into the context.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path, pipeline_config, revision=revision, **kwargs
        )
        self._processor = HYV4VLImageProcessor.from_pretrained(
            model_path, revision
        )
        hf_config = pipeline_config.model.huggingface_config
        self._image_token_id: int = hf_config.image_token_id

    # The checkpoint serves one image per prompt, as vLLM's recipe limits it.
    max_images_per_request = 1

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Builds the context, with each image's pad run and patches."""
        text = await super().new_context(request)
        # The numpy resize is CPU-bound; keep it off the event loop.
        images = [
            await run_with_default_executor(self._processor, open_image(image))
            for image in request.images_for_processing()
        ]
        ids = text.tokens.all
        pads = np.flatnonzero(ids == self._image_token_id)
        if len(pads) != len(images):
            raise ValueError(
                f"The prompt has {len(pads)} image pad tokens, but the request "
                f"has {len(images)} images."
            )
        merge = self._processor.merge_size
        counts = [(h // merge) * (w // merge + 1) for _, (_, h, w) in images]
        repeats = np.ones(len(ids), np.int64)
        repeats[pads] = counts
        ids = np.repeat(ids, repeats)
        if self.max_length and len(ids) > self.max_length:
            raise PromptTooLongError(len(ids), self.max_length)
        max_new = max_tokens_to_generate(
            len(ids), self.max_length, request.sampling_params.max_new_tokens
        )
        starts = pads + np.cumsum([0, *counts[:-1]]) - np.arange(len(pads))
        return TextAndVisionContext(
            **{f.name: getattr(text, f.name) for f in dataclasses.fields(text)}
            | {
                "tokens": TokenBuffer(array=ids),
                "max_length": (
                    len(ids) + max_new
                    if max_new is not None
                    else self.max_length
                ),
                "vision_token_ids": [self._image_token_id],
                "images": [
                    ImageMetadata(
                        start_idx=int(start),
                        end_idx=int(start) + count,
                        pixel_values=patches,
                        image_hash=hash_image(patches),
                    )
                    for start, count, (patches, _) in zip(
                        starts, counts, images, strict=True
                    )
                ],
                "extra_model_args": {
                    "image_grids": np.array(
                        [grid[1:] for _, grid in images], np.int64
                    ).reshape(-1, 2)
                },
            }
        )
