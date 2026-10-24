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

"""DeepSeek-V4.1 tokenizer rendering prompts with the checkpoint's encoder."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TextAndVisionContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import InputError, PromptTooLongError
from max.pipelines.lib.tokenizer import (
    ReasoningDelimitersMixin,
    TextAndVisionTokenizer,
    _render_custom_chat_template,
    encode_dkv_cache_hint,
    max_tokens_to_generate,
    open_image,
    resolve_eos_token_ids,
    resolve_token_id_bound,
    run_with_default_executor,
)
from max.pipelines.modeling.types import (
    ImageContentPart,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import hash_image
from PIL import Image
from transformers import AutoTokenizer

from .batch_vision_inputs import IMAGE_GRIDS
from .encoding_dsv41 import encode_messages
from .image_processor import (
    DeepseekV41ImageProcessor,
    ImageGrid,
    expand_image_spans,
)
from .model_config import MAX_IMAGES_PER_REQUEST, max_images_per_request
from .reasoning_effort import resolve_reasoning_effort

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


# The encoder only needs an image block to have a source; the pixels come
# from the request's images, in the same order.
_IMAGE_BLOCK = {"type": "image", "data": "request.images"}


def _flatten_message(message: TextGenerationRequestMessage) -> dict[str, Any]:
    """Flattens a message to a role/content dict for the V4.1 encoder.

    Joins multi-part text with ``\\n\\n``, the separator the V4.1 prompt
    format uses between text parts. A message with images keeps its parts as
    content blocks, which the encoder renders as image placeholders joined by
    the same separator.
    """
    content: str | list[dict[str, Any]]
    if isinstance(message.content, str):
        content = message.content
    else:
        blocks: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextContentPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContentPart):
                blocks.append(dict(_IMAGE_BLOCK))
            else:
                raise InputError("DeepSeek-V4.1 accepts text and images only.")
        if any(block["type"] == "image" for block in blocks):
            content = blocks
        else:
            content = "\n\n".join(block["text"] for block in blocks)

    flattened: dict[str, Any] = {
        "role": str(message.role),
        "content": content,
    }
    if message.tool_calls is not None:
        flattened["tool_calls"] = message.tool_calls
    if message.tool_call_id is not None:
        flattened["tool_call_id"] = message.tool_call_id
    if message.reasoning_content is not None:
        flattened["reasoning_content"] = message.reasoning_content
    return flattened


def render_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[TextGenerationRequestTool] | None,
    chat_template_options: Mapping[str, Any],
) -> str:
    """Renders an OpenAI-style conversation into a DeepSeek-V4.1 prompt.

    Thinking is on by default, with the default reasoning-effort budget.

    Args:
        messages: Flattened messages (``role``, ``content`` and the optional
            ``tool_calls``, ``tool_call_id``, ``reasoning_content`` keys).
        tools: OpenAI-format tool definitions, attached to the leading system
            message (one is inserted when the conversation has none).
        chat_template_options: Request chat-template kwargs. Reads
            ``thinking``, ``enable_thinking``, ``reasoning_effort`` and
            ``drop_thinking``.

    Returns:
        The prompt string, BOS included.

    Raises:
        InputError: If ``reasoning_effort`` is not ``None``, ``"none"``, an
            int in [1, 100], or a named effort, or if the encoder rejects
            the conversation.
    """
    thinking_mode, reasoning_effort = resolve_reasoning_effort(
        chat_template_options
    )

    conversation = [dict(m) for m in messages]
    if tools:
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, {"role": "system", "content": ""})
        conversation[0]["tools"] = list(tools)

    try:
        return encode_messages(
            conversation,
            thinking_mode=thinking_mode,
            drop_thinking=bool(
                chat_template_options.get("drop_thinking", True)
            ),
            reasoning_effort=reasoning_effort,
        )
    except (AssertionError, NotImplementedError, ValueError) as e:
        # The encoder signals unsupported input with these; InputError
        # carries the message to the client as an HTTP 400.
        raise InputError(f"DeepSeek-V4.1 prompt encoding failed: {e}") from e


class DeepseekV41Tokenizer(ReasoningDelimitersMixin, TextAndVisionTokenizer):
    """Tokenizer for DeepSeek-V4.1 that renders chat with the V4.1 encoder.

    A custom ``--chat-template``, when given, replaces the encoder. Each image
    placeholder in the encoded prompt grows into the image's span.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
        **unused_kwargs,
    ) -> None:
        # Not the base initializer: the checkpoint ships no AutoProcessor.
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
        # The Engram token map has one entry per tokenizer id, so an id past
        # the tokenizer has no entry even where the embedding has a row.
        self._token_id_bound = min(
            len(self.delegate),
            resolve_token_id_bound(self.delegate, pipeline_config),
        )
        self._chat_template = chat_template
        self._resolve_reasoning_delimiters(self.delegate)
        config = pipeline_config.model.huggingface_config
        self.image_token_id = int(config.image_token_id)
        self.image_processor = DeepseekV41ImageProcessor(config.vision_config)
        # The router reads this to reject requests over the cap.
        self.max_images_per_request = max_images_per_request(pipeline_config)
        overrides = pipeline_config.model.vision_config_overrides
        if (
            overrides.get(MAX_IMAGES_PER_REQUEST) == 0
            and pipeline_config.runtime.vision_cache_utilization > 0
        ):
            raise ValueError(
                f"{MAX_IMAGES_PER_REQUEST} 0 serves text only; also pass "
                "--vision-cache-utilization 0 so no KV memory is reserved "
                "for a vision encoder cache that is never filled."
            )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Renders messages with the DeepSeek-V4.1 encoder.

        A custom ``--chat-template`` replaces the encoder. It sees each image
        as an ``{"type": "image"}`` content block, and text-only messages as
        the shared text path flattens them.

        Raises:
            InputError: If the encoder rejects the conversation.
            ValueError: If a custom template fails to render.
        """
        if self._chat_template is None:
            return render_prompt(
                [_flatten_message(message) for message in messages],
                tools,
                chat_template_options,
            )
        return _render_custom_chat_template(
            self.delegate,
            self._chat_template,
            [
                _flatten_message(message)
                if message.number_of_images
                else message.flatten_content()
                for message in messages
            ],
            tools,
            **{"add_generation_prompt": True, **chat_template_options},
        )

    def _expand_images(
        self,
        token_ids: npt.NDArray[np.integer[Any]],
        images: Sequence[bytes | Image.Image],
    ) -> tuple[npt.NDArray[np.int64], list[ImageMetadata], list[ImageGrid]]:
        opened = [open_image(image) for image in images]
        # Checked before patchifying: images below min_pixels are upscaled,
        # so many tiny ones would take gigabytes before a later check.
        length = len(token_ids) + sum(
            self.image_processor.plan(image.width, image.height)[0].num_tokens
            - 1
            for image in opened
        )
        if self.max_length and length > self.max_length:
            raise PromptTooLongError(length, self.max_length)
        processed = [self.image_processor(image) for image in opened]
        expanded, starts = expand_image_spans(
            token_ids.tolist(),
            [grid for _, grid in processed],
            self.image_token_id,
        )
        metadata = [
            ImageMetadata(
                start_idx=start,
                end_idx=start + grid.num_tokens,
                pixel_values=patches,
                # Prefix caching and the vision encoder cache both key on
                # it; equal patch arrays can come from different grids.
                image_hash=hash(
                    (hash_image(patches), grid.n_vit_h, grid.n_vit_w)
                ),
            )
            for start, (patches, grid) in zip(starts, processed, strict=True)
        ]
        return expanded, metadata, [grid for _, grid in processed]

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Builds the context, expanding every image into its span.

        Raises:
            InputError: If the request carries a video, an image cannot be
                decoded, or the prompt's placeholder count differs from the
                image count.
        """
        if request.videos:
            raise InputError("DeepSeek-V4.1 does not support video inputs.")
        if request.prompt is not None:
            text_ids = await self.encode(request.prompt)
        elif request.messages:
            # Off the event loop: the render is pure Python over the whole
            # conversation.
            prompt = await run_with_default_executor(
                self.apply_chat_template,
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            text_ids = await self.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")
        token_ids, images, grids = await run_with_default_executor(
            self._expand_images, text_ids, request.images_for_processing()
        )

        max_gen_tokens = max_tokens_to_generate(
            len(token_ids),
            self.max_length,
            request.sampling_params.max_new_tokens,
        )
        extra_model_args: dict[str, npt.NDArray[Any]] = {}
        if images:
            extra_model_args["image_token_indices"] = np.flatnonzero(
                token_ids == self.image_token_id
            )
            extra_model_args[IMAGE_GRIDS] = np.array(
                [(grid.n_vit_h, grid.n_vit_w) for grid in grids], np.int64
            )
        response_format = request.response_format
        return TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=TokenBuffer(array=token_ids),
            vocab_size=self.tokenizer_vocab_size,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json.dumps(response_format.json_schema)
            if response_format and response_format.json_schema is not None
            else None,
            grammar=response_format.grammar if response_format else None,
            grammar_state=GrammarEnforcementState.from_response_format(
                response_format
            ),
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            cache_salt=request.cache_salt,
            images=images,
            vision_token_ids=[self.image_token_id],
            extra_model_args=extra_model_args,
        )
