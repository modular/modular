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
"""Text and image tokenization with the DeepSeek-V4.1 prompt encoder."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TextAndVisionContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import InputError, PromptTooLongError
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.tokenizer import (
    ReasoningDelimitersMixin,
    TextTokenizer,
    _render_custom_chat_template,
    encode_dkv_cache_hint,
    max_tokens_to_generate,
    open_image,
    replace_unpaired_surrogates,
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
from tokenizers import Encoding
from transformers import PreTrainedTokenizerFast

from ..deepseekV41.encoding_dsv41 import (
    IMAGE_PLACEHOLDER,
    ImageSlot,
    encode_messages,
)
from ..deepseekV41.image_processor import (
    DeepseekV41ImageProcessor,
    ImageGrid,
    expand_image_spans,
)
from ..deepseekV41.reasoning_effort import resolve_reasoning_effort
from .interface import has_vision_tower
from .model_config import (
    MAX_IMAGES_PER_REQUEST,
    max_images_per_request,
    vision_config,
)
from .vision_inputs import IMAGE_GRIDS

# The encoder only needs an image block to have a source; the pixels come
# from the request's images, in the same order.
_IMAGE_BLOCK = {"type": "image", "data": "request.images"}


def encode_literal_text(
    delegate: PreTrainedTokenizerFast, text: str, add_special_tokens: bool
) -> list[int]:
    """Encodes text in which every image placeholder is text.

    The tokenizer cannot leave out a single added token, so each
    placeholder's final character is encoded on its own, which keeps the
    placeholder from matching. The pieces go to the tokenizer as one batch,
    so a placeholder costs no tokenizer call of its own.
    """
    *placeholder_ends, rest = replace_unpaired_surrogates(text).split(
        IMAGE_PLACEHOLDER
    )
    if not placeholder_ends:
        return delegate.encode(rest, add_special_tokens=add_special_tokens)
    backend = delegate.backend_tokenizer
    *heads, tail, close = backend.encode_batch(
        [piece + IMAGE_PLACEHOLDER[:-1] for piece in placeholder_ends]
        + [rest, IMAGE_PLACEHOLDER[-1]],
        add_special_tokens=False,
    )
    merged = Encoding.merge(
        [encoding for head in heads for encoding in (head, close)] + [tail]
    )
    return backend.post_process(
        merged, add_special_tokens=add_special_tokens
    ).ids


def _flatten_message(message: TextGenerationRequestMessage) -> dict[str, Any]:
    r"""Flattens a message to a role/content dict for the V4.1 encoder.

    Joins multi-part text with ``\n\n``, the separator the V4.1 prompt
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


def _encode_conversation(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[TextGenerationRequestTool] | None,
    chat_template_options: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
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
            return_multi_modal_data=True,
        )
    except (AssertionError, NotImplementedError, ValueError) as e:
        # The encoder signals unsupported input with these; InputError
        # carries the message to the client as an HTTP 400.
        raise InputError(f"DeepSeek-V4.1 prompt encoding failed: {e}") from e


def render_prompt_with_image_slots(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[TextGenerationRequestTool] | None,
    chat_template_options: Mapping[str, Any],
) -> tuple[str, list[ImageSlot]]:
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
        The prompt, BOS included, and the slot of each placeholder the
        encoder inserted for an image block, in prompt order. A slot's
        ``image_index`` counts the image blocks in message order, as the
        request lists its images.

    Raises:
        InputError: If ``reasoning_effort`` is not ``None``, ``"none"``, an
            int in [1, 100], or a named effort, or if the encoder rejects
            the conversation.
    """
    prompt, media = _encode_conversation(messages, tools, chat_template_options)
    return prompt, media["image_slots"]


class DeepseekV41Tokenizer(ReasoningDelimitersMixin, TextTokenizer):
    """Renders chat with the V4.1 encoder and grows each image into its span.

    A custom ``--chat-template``, when given, replaces the encoder for
    text-only requests. Each image placeholder the encoder inserts grows into
    the image's span; a placeholder from caller text stays text. A
    ``--tokenizer-impl`` chat encoder encodes the text-only chats.
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
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
        )
        # The Engram token map has one entry per tokenizer id, so an id past
        # the tokenizer has no entry even where the embedding has a row.
        self._token_id_bound = min(len(self.delegate), self._token_id_bound)
        self._resolve_reasoning_delimiters(self.delegate)
        config = pipeline_config.model.huggingface_config
        token = getattr(config, "image_token_id", None)
        self.image_token_id = None if token is None else int(token)
        self.vision_token_ids = (
            [] if self.image_token_id is None else [self.image_token_id]
        )
        self.image_processor: DeepseekV41ImageProcessor | None = None
        # The router reads this to reject requests over the cap.
        self.max_images_per_request: int | None = 0
        if not has_vision_tower(config):
            return
        self.max_images_per_request = max_images_per_request(pipeline_config)
        if (
            self.max_images_per_request == 0
            and pipeline_config.runtime.vision_cache_utilization > 0
        ):
            raise ValueError(
                f"{MAX_IMAGES_PER_REQUEST} 0 serves text only; also pass "
                "--vision-cache-utilization 0 so no KV memory is reserved "
                "for a vision encoder cache that is never filled."
            )
        if vision_config(pipeline_config) is None:
            self.max_images_per_request = 0
        else:
            self.image_processor = DeepseekV41ImageProcessor(
                config.vision_config
            )

    @property
    def expects_content_wrapping(self) -> bool:
        """Whether request messages arrive as typed content parts.

        They do: the encoder renders each image part as an image block.
        """
        return True

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Renders messages with the DeepSeek-V4.1 encoder.

        A custom ``--chat-template`` replaces the encoder for text-only
        requests. Template code cannot say which placeholder stands for which
        image, so a request with images needs the encoder.

        Raises:
            InputError: If the encoder rejects the conversation, or a custom
                template is set and the messages carry images or the template
                fails to render.
        """
        prompt, _ = self._render_chat(messages, tools, **chat_template_options)
        return prompt

    def _render_chat(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        /,
        **chat_template_options: Any,
    ) -> tuple[str, list[ImageSlot]]:
        if self._chat_template is None:
            return render_prompt_with_image_slots(
                [_flatten_message(message) for message in messages],
                tools,
                chat_template_options,
            )
        if any(message.number_of_images for message in messages):
            raise InputError(
                "DeepSeek-V4.1 does not support image inputs with a custom"
                " chat template; use the built-in template."
            )
        try:
            prompt = _render_custom_chat_template(
                self.delegate,
                self._chat_template,
                [message.flatten_content() for message in messages],
                tools,
                **{"add_generation_prompt": True, **chat_template_options},
            )
        except ValueError as e:
            # InputError carries the message to the client as an HTTP 400.
            raise InputError(str(e)) from e
        return prompt, []

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Encodes a prompt in which every image placeholder is text.

        Only the image slots the chat renderer places become the image
        token, so a prompt given as text or ids never holds one.

        Raises:
            InputError: If a token-id prompt holds the image token.
            PromptTooLongError: If the prompt is longer than the context.
        """
        if not isinstance(prompt, str):
            ids = await super().encode(prompt, add_special_tokens)
            if self.image_token_id is not None and np.any(
                ids == self.image_token_id
            ):
                raise InputError(
                    f"The prompt holds the image token {self.image_token_id};"
                    " images can only be given in chat messages."
                )
            return ids
        encoded = await run_with_default_executor(
            encode_literal_text, self.delegate, prompt, add_special_tokens
        )
        max_length = self.max_length or self.delegate.model_max_length
        if max_length and len(encoded) > max_length:
            raise PromptTooLongError(len(encoded), max_length)
        return np.array(encoded)

    def _expand_images(
        self,
        token_ids: npt.NDArray[np.integer[Any]],
        images: Sequence[bytes | Image.Image],
        image_positions: Sequence[int],
    ) -> tuple[npt.NDArray[np.int64], list[ImageMetadata], list[ImageGrid]]:
        opened = [open_image(image) for image in images]
        processor = self.image_processor
        assert not opened or processor is not None
        # Checked before patchifying: images below min_pixels are upscaled,
        # so many tiny ones would take gigabytes before a later check.
        length = len(token_ids) + sum(
            processor.plan(image.width, image.height)[0].num_tokens - 1
            for image in opened
            if processor is not None
        )
        if self.max_length and length > self.max_length:
            raise PromptTooLongError(length, self.max_length)
        if processor is None or not opened:
            return np.asarray(token_ids, dtype=np.int64), [], []
        assert self.image_token_id is not None
        processed = [processor(image) for image in opened]
        expanded, starts = expand_image_spans(
            token_ids.tolist(),
            [grid for _, grid in processed],
            self.image_token_id,
            image_positions,
        )
        metadata = [
            ImageMetadata(
                start_idx=start,
                end_idx=start + grid.num_tokens,
                pixel_values=patches,
                # The vision encoder cache keys on it; equal patch arrays
                # can come from different grids.
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
            InputError: If the request carries a video, carries an image this
                deployment does not serve, an image cannot be decoded, the
                image count differs from the messages' image parts, or images
                come with a custom chat template.
        """
        if request.videos:
            raise InputError("DeepSeek-V4.1 does not support video inputs.")
        images = request.images_for_processing()
        if images and self.image_processor is None:
            raise InputError(
                "This DeepSeek-V4.1 deployment serves text only: it has no "
                "vision encoder, or caps images at 0."
            )
        image_positions: list[int] = []
        if request.prompt is not None:
            text_ids = await self.encode(request.prompt)
        elif request.messages:
            # Off the event loop: the render is pure Python over the whole
            # conversation.
            prompt, slots = await run_with_default_executor(
                self._render_chat,
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            if len(slots) != len(images):
                raise InputError(
                    f"The messages have {len(slots)} image part(s) but"
                    f" {len(images)} image(s) were given."
                )
            images = [images[slot.image_index] for slot in slots]
            if slots:
                text_ids, image_positions = await run_with_default_executor(
                    self._encode_image_prompt,
                    prompt,
                    [slot.offset for slot in slots],
                )
            elif IMAGE_PLACEHOLDER in prompt:
                # A chat encoder makes the placeholder in caller text the
                # image token; HuggingFace keeps it text.
                text_ids = await self.encode(prompt, add_special_tokens=False)
            else:
                text_ids = await self._encode_chat_prompt(prompt)
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")
        token_ids, image_metadata, grids = await run_with_default_executor(
            self._expand_images, text_ids, images, image_positions
        )

        max_gen_tokens = max_tokens_to_generate(
            len(token_ids),
            self.max_length,
            request.sampling_params.max_new_tokens,
        )
        extra_model_args: dict[str, npt.NDArray[Any]] = {}
        if grids:
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
            images=image_metadata,
            vision_token_ids=self.vision_token_ids,
            extra_model_args=extra_model_args,
        )

    def _encode_image_prompt(
        self, prompt: str, image_offsets: Sequence[int]
    ) -> tuple[npt.NDArray[np.int64], list[int]]:
        """Encodes a prompt with the image token only at ``image_offsets``.

        Returns:
            The ids and the token position of each image token.
        """
        ids: list[int] = []
        positions: list[int] = []
        cursor = 0
        for offset in image_offsets:
            assert self.image_token_id is not None
            ids += encode_literal_text(
                self.delegate, prompt[cursor:offset], add_special_tokens=False
            )
            positions.append(len(ids))
            ids.append(self.image_token_id)
            cursor = offset + len(IMAGE_PLACEHOLDER)
        ids += encode_literal_text(
            self.delegate, prompt[cursor:], add_special_tokens=False
        )
        return np.asarray(ids, dtype=np.int64), positions
