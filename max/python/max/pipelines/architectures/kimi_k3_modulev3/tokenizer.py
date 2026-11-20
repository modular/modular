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

"""Tokenizes Kimi K3 chat segments and expands image embedding spans."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    TextTokenizer,
    max_tokens_to_generate,
    resolve_eos_token_ids,
)
from max.pipelines.lib.tokenizer import (
    checked_token_ids,
    encode_dkv_cache_hint,
    open_image,
    resolve_token_id_bound,
    run_with_default_executor,
)
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import hash_image
from transformers import AutoTokenizer

from .context import KimiK3TextAndVisionContext
from .vision_config import has_vision_tower
from .vision_processor import (
    KimiK3VisionProcessor,
    VisionOutput,
    compute_position_ids,
    load_media_config,
)

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


def expand_image_tokens(
    token_ids: npt.NDArray[np.int64], counts: Sequence[int], image_token_id: int
) -> tuple[npt.NDArray[np.int64], list[tuple[int, int]]]:
    """Expands each raw placeholder without joining adjacent images' spans."""
    positions = np.flatnonzero(token_ids == image_token_id)
    if len(positions) != len(counts):
        raise ValueError(
            "Image placeholder count does not match the media count"
        )
    parts: list[npt.NDArray[np.int64]] = []
    spans: list[tuple[int, int]] = []
    previous = 0
    offset = 0
    for position, count in zip(positions, counts, strict=True):
        if count <= 0:
            raise ValueError("Image embedding counts must be positive")
        parts.append(token_ids[previous:position])
        start = int(position) + offset
        spans.append((start, start + count))
        parts.append(np.full(count, image_token_id, dtype=np.int64))
        previous = int(position) + 1
        offset += count - 1
    parts.append(token_ids[previous:])
    return np.concatenate(parts), spans


class KimiK3VLTokenizer(TextAndVisionTokenizer):
    """Preserves K3's segment tokenization and original-size media headers."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: object,
    ) -> None:
        if chat_template is not None:
            raise ValueError(
                "Kimi K3 image chats render with the checkpoint's own encoder;"
                " --chat-template is not supported"
            )
        self.model_path = model_path
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length
        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )
        self._token_id_bound = resolve_token_id_bound(
            self.delegate, pipeline_config
        )
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.enable_vision_caching = (
            pipeline_config.runtime.vision_cache_utilization != 0
        )
        token_id = self.delegate.convert_tokens_to_ids("<|media_pad|>")
        if (
            not isinstance(token_id, int)
            or token_id == self.delegate.unk_token_id
        ):
            raise ValueError("Kimi K3 tokenizer has no media-pad token")
        self.media_pad_token_id = token_id
        self.vision_token_ids = [token_id]
        self.vision_processor = KimiK3VisionProcessor(
            load_media_config(model_path, revision)
        )

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.int64]:
        """Encodes an explicit prompt or validates already-tokenized input."""
        if isinstance(prompt, str):

            def encode_prompt() -> npt.NDArray[np.int64]:
                # Any keyword argument sends the checkpoint's encode to the
                # transformers path instead of its tiktoken table.
                return np.asarray(self.delegate.encode(prompt), dtype=np.int64)

            encoded = await run_with_default_executor(encode_prompt)
        else:
            encoded = checked_token_ids(prompt, self._token_id_bound).astype(
                np.int64
            )
        if self.max_length and len(encoded) > self.max_length:
            raise PromptTooLongError(len(encoded), self.max_length)
        return encoded

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Renders a chat for callers that request its text form."""
        options = {"add_generation_prompt": True, **chat_template_options}
        rendered = self.delegate.apply_chat_template(
            [message.model_dump(exclude_none=True) for message in messages],
            tools=tools,
            tokenize=False,
            **options,
        )
        assert isinstance(rendered, str)
        return rendered

    def _prepare_images(
        self, request: TextGenerationRequest
    ) -> tuple[list[str], VisionOutput]:
        images = [
            open_image(image) for image in request.images_for_processing()
        ]
        prompts = [
            self.vision_processor.make_image_prompt(*image.size)
            for image in images
        ]
        vision = self.vision_processor.preprocess(
            [{"type": "image", "image": image} for image in images]
        )
        return prompts, vision

    def _encode_messages(
        self, request: TextGenerationRequest, image_prompts: list[str]
    ) -> npt.NDArray[np.int64]:
        # The checkpoint encodes ordinary text and structural tokens as
        # separate segments; rendering and re-encoding loses that boundary.
        options: dict[str, Any] = {
            "add_generation_prompt": True,
            **(request.chat_template_options or {}),
        }
        # With image_prompts set, the reference expands every literal
        # placeholder in user text, so a chat without images omits them.
        if image_prompts or request.number_of_images:
            options["image_prompts"] = image_prompts
        return np.asarray(
            self.delegate.apply_chat_template(
                [
                    message.model_dump(exclude_none=True)
                    for message in request.messages
                ],
                tools=request.tools,
                tokenize=True,
                **options,
            ),
            dtype=np.int64,
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> KimiK3TextAndVisionContext:
        """Builds token state and one image span per uploaded image."""
        if request.videos:
            raise ValueError("Kimi K3 supports still images only")
        if request.prompt is None and not request.messages:
            raise ValueError("A Kimi K3 request requires a prompt or messages")
        prompts, vision = await run_with_default_executor(
            self._prepare_images, request
        )
        if request.prompt is not None:
            encoded = await self.encode(request.prompt)
        else:
            encoded = await run_with_default_executor(
                self._encode_messages, request, prompts
            )
        grids = vision["grid_thws"]
        counts = (
            np.prod(grids, axis=1)
            // self.vision_processor.cfg.merge_kernel_size**2
        ).tolist()
        encoded, spans = expand_image_tokens(
            encoded, counts, self.media_pad_token_id
        )
        if self.max_length and len(encoded) > self.max_length:
            raise PromptTooLongError(len(encoded), self.max_length)
        patch_offsets = np.prod(grids, axis=1).cumsum()
        pixels = (
            np.split(vision["pixel_values"], patch_offsets[:-1].tolist())
            if prompts
            else []
        )
        max_new_tokens = request.sampling_params.max_new_tokens
        max_gen_tokens = max_tokens_to_generate(
            len(encoded), self.max_length, max_new_tokens
        )
        response_format = request.response_format
        cached = self.enable_prefix_caching or self.enable_vision_caching
        return KimiK3TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            tokens=TokenBuffer(array=encoded),
            vocab_size=self.tokenizer_vocab_size,
            max_length=len(encoded) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json.dumps(response_format.json_schema)
            if response_format and response_format.json_schema
            else None,
            grammar=response_format.grammar if response_format else None,
            grammar_state=GrammarEnforcementState.from_response_format(
                response_format
            ),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            grid_thws=grids,
            position_ids=compute_position_ids(grids),
            image_token_indices=np.flatnonzero(
                encoded == self.media_pad_token_id
            ).astype(np.int32),
            images=[
                ImageMetadata(
                    start_idx=start,
                    end_idx=end,
                    pixel_values=image_pixels,
                    image_hash=hash_image(raw, 0) if cached else None,
                )
                for (start, end), image_pixels, raw in zip(
                    spans, pixels, request.images, strict=True
                )
            ],
            vision_token_ids=self.vision_token_ids,
            cache_salt=request.cache_salt,
        )


def kimi_k3_tokenizer(
    model_path: str,
    pipeline_config: PipelineConfig,
    *,
    revision: str | None = None,
    max_length: int | None = None,
    trust_remote_code: bool = False,
    enable_llama_whitespace_fix: bool = False,
    chat_template: str | None = None,
    **kwargs: object,
) -> TextTokenizer | KimiK3VLTokenizer:
    """Selects text-only tokenization when the checkpoint has no image tower."""
    tokenizer_type = (
        KimiK3VLTokenizer
        if has_vision_tower(pipeline_config.model.huggingface_config)
        else TextTokenizer
    )
    return tokenizer_type(
        model_path,
        pipeline_config,
        revision=revision,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        enable_llama_whitespace_fix=enable_llama_whitespace_fix,
        chat_template=chat_template,
        **kwargs,
    )
