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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    ImageMetadata,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TokenBuffer,
)
from max.pipelines.lib import TextAndVisionTokenizer, max_tokens_to_generate
from max.pipelines.lib.tokenizer import run_with_default_executor
from max.support.image import find_contiguous_ranges, hash_image
from transformers import AutoTokenizer

from .context import KimiK2_5TextAndVisionContext
from .layers.vision.data_processing import compute_position_ids
from .vision_processor import (
    KimiK2_5VisionProcessor,
    _to_pil,
)

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

# Kimi K2.5 special token for image placeholder padding.
_MEDIA_PAD_TOKEN = "<|media_pad|>"

# Chat turn terminator. The HF tokenizer lists [EOS] as eos_token, but the
# chat format ends assistant turns with <|im_end|>.  We need both in the
# EOS set so generation stops.
_IM_END_TOKEN = "<|im_end|>"


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

        im_end_id = self.delegate.convert_tokens_to_ids(_IM_END_TOKEN)
        if isinstance(im_end_id, int):
            self._default_eos_token_ids.add(im_end_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.enable_vision_caching = (
            pipeline_config.runtime.max_vision_cache_entries > 0
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

        # rope_max_width is needed to compute per-image position IDs.
        vision_cfg = getattr(config, "vision_config", None)
        self.rope_max_width: int = int(
            getattr(vision_cfg, "rope_max_width", 512)
        )

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transforms the provided prompt into a token array.

        Kimi's ``TikTokenTokenizer.encode()`` accepts
        ``allow_special_tokens`` instead of the HuggingFace-standard
        ``add_special_tokens``.  Passing the unrecognised kwarg causes
        HF to silently fall back to the slow
        ``PreTrainedTokenizer.encode()`` path (actually slow in
        transformers v5+).  This override calls the delegate with
        ``allow_special_tokens=True`` so the fast tiktoken path is
        always used.

        The ``add_special_tokens`` parameter is accepted for interface
        compatibility but is effectively a no-op: the tiktoken fast
        path never prepends/appends BOS/EOS tokens regardless, and
        ``allow_special_tokens`` (whether tiktoken *recognises* special
        token strings in the input) must stay ``True`` so that
        chat-templated text is tokenised correctly.
        """
        encoded_prompt: npt.NDArray[np.integer[Any]]
        if isinstance(prompt, str):

            def _encode_fn(
                prompt: str,
            ) -> npt.NDArray[np.integer[Any]]:
                return np.array(
                    self.delegate.encode(prompt, allow_special_tokens=True)
                )

            encoded_prompt = await run_with_default_executor(_encode_fn, prompt)

            max_length = self.max_length or self.delegate.model_max_length
            if max_length and len(encoded_prompt) > max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length"
                    f" ({len(encoded_prompt)} > {max_length})."
                )
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
    ) -> str:
        """Applies the tokenizer's chat template to messages."""
        templated = self.delegate.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tokenize=False,
            tools=tools,
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
            prompt = self.apply_chat_template(request.messages, request.tools)
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        encoded_prompt = np.array(await self.encode(prompt, add_special_tokens))

        placeholder_positions = np.where(
            encoded_prompt == self.media_pad_token_id
        )[0]
        num_placeholders = len(placeholder_positions)

        merge_len = self.vision_processor.cfg.merge_kernel_size**2
        if request.images:
            if num_placeholders != len(request.images):
                raise ValueError(
                    f"Number of <|media_pad|> placeholders ({num_placeholders}) "
                    f"must match number of images ({len(request.images)})"
                )

            # Process images through the custom vision processor.
            vision_outputs = self._process_images(request)

            grid_thws = np.empty((0, 3), dtype=np.int64)
            pixel_values: list[npt.NDArray[Any]] = []
            position_ids = np.empty(0, dtype=np.int64)

            if not vision_outputs:
                raise ValueError(
                    "Images provided but vision processor returned empty"
                )

            grid_thws = vision_outputs["grid_thws"].copy()
            all_pixels = vision_outputs["pixel_values"]
            # Split the concatenated pixel array into per-image chunks
            # using the (t, h, w) grid dimensions to compute patch counts.
            offsets = np.prod(grid_thws, axis=1).cumsum()
            pixel_values = np.split(all_pixels, offsets[:-1].tolist())

            grid_thws_list = [(int(t), int(h), int(w)) for t, h, w in grid_thws]
            position_ids = compute_position_ids(
                grid_thws_list, self.rope_max_width
            )
            max_h = int(grid_thws[:, 1].max())
            max_w = int(grid_thws[:, 2].max())

            # Expand each media placeholder to match the number of merged
            # vision tokens for its corresponding image.
            for i in range(len(grid_thws)):
                idx = int(placeholder_positions[-(i + 1)])
                t, h, w = grid_thws[-(i + 1)]
                num_img_tokens = int((t * h * w) // merge_len)
                encoded_prompt = np.insert(
                    encoded_prompt,
                    idx,
                    [self.media_pad_token_id] * (num_img_tokens - 1),
                )
        else:
            # No images: initialize empty arrays for text-only requests
            max_h = 0
            max_w = 0
            grid_thws = np.empty((0, 3), dtype=np.int64)
            pixel_values = []
            position_ids = np.empty(0, dtype=np.int64)

        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        # Positions of the image-placeholder token within this context's
        # token buffer (relative to the start of encoded_prompt).
        image_token_indices = np.where(
            encoded_prompt == self.media_pad_token_id
        )[0].astype(np.int32)

        json_schema = (
            json.dumps(request.response_format.get("json_schema"))
            if request.response_format
            and request.response_format.get("json_schema")
            else None
        )

        grammar = (
            request.response_format.get("grammar")
            if request.response_format
            else None
        )

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise ValueError(
                f"encoded_prompt length {encoded_prompt.shape[0]} is greater than the max_length of the tokenizer {self.max_length}"
            )

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, self.vision_token_ids
        )

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )

        context = KimiK2_5TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            tokens=token_buffer,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            grammar=grammar,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            grid_thws=grid_thws,
            position_ids=position_ids,
            image_token_indices=image_token_indices,
            max_h=max_h,
            max_w=max_w,
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
                    image_hash=hash_image(pixels)
                    if self.enable_prefix_caching or self.enable_vision_caching
                    else None,
                )
                for (start_idx, end_idx), pixels in zip(
                    start_and_end_idxs, pixel_values, strict=True
                )
            ],
            vision_token_ids=self.vision_token_ids,
        )

        return context
