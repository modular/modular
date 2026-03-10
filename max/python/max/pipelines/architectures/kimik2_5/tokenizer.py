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

"""Kimi K2.5 tokenizer for multimodal (text + vision) inputs.

Follows the same pattern as ``Qwen2_5VLTokenizer``: extends
``TextAndVisionTokenizer`` and delegates vision preprocessing to the
custom ``KimiK2_5VisionProcessor`` (PIL + NumPy only, no torch).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    ImageMetadata,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TokenBuffer,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import TextAndVisionTokenizer, max_tokens_to_generate
from max.support.image import find_contiguous_ranges, hash_image
from transformers import AutoTokenizer

from .context import KimiK2_5TextAndVisionContext
from .vision_processor import (
    KimiK2_5VisionProcessor,
    _to_pil,
)

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

# Kimi K2.5 special token for image placeholder padding.
_MEDIA_PAD_TOKEN = "<|media_pad|>"


class KimiK2_5VLTokenizer(TextAndVisionTokenizer):
    """Kimi K2.5 tokenizer for multimodal (text + vision) inputs.

    Extends ``TextAndVisionTokenizer`` with a custom vision processor
    (``KimiK2_5VisionProcessor``) instead of ``AutoProcessor``, following
    the same architecture as ``Qwen2_5VLTokenizer``.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        context_validators: list[Callable[[TextAndVisionContext], None]]
        | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        config = pipeline_config.model.huggingface_config

        self._default_eos_token_ids = set([self.eos])
        if eos_token_id := getattr(config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._default_eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._default_eos_token_ids.update(eos_token_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        self._context_validators = (
            context_validators if context_validators else []
        )

        # Resolve the media pad token ID used as the vision placeholder.
        media_pad_id = self.delegate.convert_tokens_to_ids(_MEDIA_PAD_TOKEN)
        if isinstance(media_pad_id, list):
            media_pad_id = media_pad_id[0]
        if media_pad_id == self.delegate.unk_token_id:
            raise ValueError(
                f"Token {_MEDIA_PAD_TOKEN!r} not found in tokenizer vocabulary"
            )
        self.media_pad_token_id: int = media_pad_id
        self.vision_token_ids = [self.media_pad_token_id]

        # Build the custom vision processor from HF config.
        media_proc_cfg = getattr(config, "media_proc_cfg", None)
        self.vision_processor = KimiK2_5VisionProcessor(
            media_proc_cfg=media_proc_cfg
        )

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        """Applies the tokenizer's chat template to messages."""
        templated = self.delegate.apply_chat_template(
            [msg.model_dump() for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(templated, str)
        return templated

    def _process_images(
        self, request: TextGenerationRequest
    ) -> dict[str, npt.NDArray[Any]]:
        """Converts raw image bytes from the request into preprocessed arrays.

        Args:
            request: The text generation request containing raw image bytes
                in ``request.images``.

        Returns:
            Dictionary with ``pixel_values`` and ``grid_thws`` arrays,
            or empty dict when no images are provided.
        """
        if not request.images:
            return {}

        pil_images = [_to_pil(img_bytes) for img_bytes in request.images]
        medias = [{"type": "image", "image": img} for img in pil_images]
        return self.vision_processor.preprocess(medias)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> KimiK2_5TextAndVisionContext:
        """Creates a ``KimiK2_5TextAndVisionContext`` for the Kimi K2.5 model.

        Processes text through the delegate tokenizer and images through
        ``KimiK2_5VisionProcessor``, then assembles the context with
        ``ImageMetadata`` entries for each image.
        """
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
            prompt = self.apply_chat_template(request.messages)
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        encoded_prompt = await self.encode(prompt, add_special_tokens)

        # Process images through the custom vision processor.
        vision_outputs = self._process_images(request)

        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        grid_thws = np.empty((0, 3), dtype=np.int64)
        pixel_values: list[npt.NDArray[Any]] = []

        if vision_outputs:
            grid_thws = vision_outputs["grid_thws"]
            all_pixels = vision_outputs["pixel_values"]
            # Split the concatenated pixel array into per-image chunks
            # using the (t, h, w) grid dimensions to compute patch counts.
            offsets = np.prod(grid_thws, axis=1).cumsum()
            pixel_values = np.split(all_pixels, offsets[:-1].tolist())

        json_schema = (
            json.dumps(request.response_format.get("json_schema"))
            if request.response_format
            else None
        )

        if request.sampling_params.ignore_eos:
            eos_token_ids: set[int] = set()
        else:
            eos_token_ids = self._default_eos_token_ids

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise ValueError(
                "encoded_prompt is greater than the max_length of the tokenizer"
            )

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, self.vision_token_ids
        )

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )

        context = KimiK2_5TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            tokens=token_buffer,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
            grid_thws=grid_thws,
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
                    image_hash=hash_image(pixels)
                    if self.enable_prefix_caching
                    else None,
                )
                for (start_idx, end_idx), pixels in zip(
                    start_and_end_idxs, pixel_values, strict=True
                )
            ],
            vision_token_ids=self.vision_token_ids,
        )

        for validator in self._context_validators:
            validator(context)

        return context
