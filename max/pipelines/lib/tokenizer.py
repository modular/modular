# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# mypy: disable-error-code="import-not-found"
"""Implementations of provided tokenizers."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
from collections.abc import Sequence
from typing import Any, Optional, Union, cast

import numpy as np
import torch
from max.pipelines.core import (
    PipelineTokenizer,
    TextAndVisionContext,
    TextContext,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
)
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger("max.pipelines")


class IdentityPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, str, TokenGeneratorRequest],
):
    @property
    def eos(self) -> int:
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> str:
        return prompt

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: str,
        **kwargs,
    ) -> str:
        if isinstance(encoded, str):
            return encoded
        return ""


class PreTrainedPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, np.ndarray, TokenGeneratorRequest],
):
    def __init__(
        self,
        delegate: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        assert isinstance(
            delegate, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        )
        self.delegate = delegate

    def apply_chat_template(
        self, messages: list[TokenGeneratorRequestMessage]
    ) -> str:
        try:
            templated_message = self.delegate.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception:
            msg = (
                "apply_chat_template failed for"
                " PreTrainedTokenGeneratorTokenizer"
            )
            logger.warning(msg)
            return "\n".join([str(message["content"]) for message in messages])

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        return np.array(self.delegate.encode(prompt))

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: np.ndarray,
        **kwargs,
    ) -> str:
        return self.delegate.decode(encoded, **kwargs)


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int | None,
    max_new_tokens: int | None = None,
) -> int | None:
    """Returns the max number of new tokens to generate."""
    if max_length is None:
        return max_new_tokens
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens is None:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


class TextTokenizer(
    PipelineTokenizer[TextContext, np.ndarray, TokenGeneratorRequest]
):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        try:
            self.delegate = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                # If `max_length` is None, the max length will be taken
                # from the HuggingFace tokenizer_config.
                model_max_length=max_length,
            )
        except Exception as e:
            msg = (
                f"Failed to load tokenizer from {model_path}. "
                "This can happen if:\n"
                "- The model is not fully supported by the transformers python package\n"
                "- Required configuration files are missing\n"
                "- The model path is incorrect\n"
                "- '--trust_remote_code=True' is needed but not set\n"
            )
            raise ValueError(msg) from e

        # As we are adding special tokens during chat templating prior to tokenization,
        # when add_special_tokens=True, we duplicate BOS tokens specifically.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )

        # configure Llama whitespace fix if needed
        self._enable_llama_whitespace_fix = (
            enable_llama_whitespace_fix and self._is_llama_tokenizer
        )
        (
            self._llama_whitespace_fix_dummy_token_id,
            self._llama_whitespace_fix_dummy_token_len,
        ) = self._llama_whitespace_fix_dummy_token

    def apply_chat_template(
        self,
        messages: list[TokenGeneratorRequestMessage],
        tools: Optional[list[TokenGeneratorRequestTool]],
        chat_template_options: Optional[dict[str, Any]] = None,
    ) -> str:
        chat_template_options = chat_template_options or {
            "add_generation_prompt": True
        }
        try:
            templated_message = self.delegate.apply_chat_template(
                messages,
                tokenize=False,
                tools=tools,
                **chat_template_options,
            )
            return cast(str, templated_message)
        except Exception:
            msg = (
                "apply_chat_template failed for"
                f" TextTokenizer({self.model_path})"
            )
            logger.warning(msg)
            return "\n".join([str(message["content"]) for message in messages])

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: Union[str, Sequence[int]], add_special_tokens: bool = True
    ) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        encoded_prompt: np.ndarray
        if isinstance(prompt, str):
            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            if add_special_tokens:
                encoded_prompt = await run_with_default_executor(
                    self._encode_with_special_tokens, prompt
                )
            else:
                encoded_prompt = await run_with_default_executor(
                    self._encode_without_special_tokens, prompt
                )

            max_length = self.max_length or self.delegate.model_max_length
            if max_length and len(encoded_prompt) > max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length ({len(encoded_prompt)} > {max_length})."
                )
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(
        self, context: TextContext, encoded: np.ndarray, **kwargs
    ) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        # Sometimes, encoded comes in as an int so, make it np array
        if isinstance(encoded, int):
            encoded = np.array(encoded)

        # There is an issue where Llama tokenizer strips leading spaces
        # if a single token is decoded at a time. This is a temporary
        # fix until the issue resolved on the Tokenizers side.
        # More information:
        # https://github.com/huggingface/transformers/issues/31643
        # https://github.com/Lightning-AI/litgpt/pull/1559
        if self._enable_llama_whitespace_fix and encoded.size == 1:
            return self._decode_with_llama_whitespace_fix(encoded, **kwargs)

        return self.delegate.decode(encoded, **kwargs)

    async def new_context(self, request: TokenGeneratorRequest) -> TextContext:
        """Create a new TextContext object, leveraging necessary information like
        cache_seq_id and prompt from TokenGeneratorRequest."""

        prompt: Union[str, list[int]]
        add_special_tokens = True
        if request.prompt is not None:
            if isinstance(request.prompt, str):
                prompt = str(request.prompt)
            else:
                prompt = [int(t) for t in request.prompt]
        elif request.messages is not None:
            prompt = self.apply_chat_template(
                request.messages, request.tools, request.chat_template_options
            )
            # Chat templating already adds special tokens, therefore we step around this here.
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        encoded_prompt = await self.encode(
            prompt, add_special_tokens=add_special_tokens
        )

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.max_new_tokens is not None:
            max_new_tokens = request.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            len(encoded_prompt),
            self.max_length,
            max_new_tokens,
        )

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )
        context = TextContext(
            prompt=prompt,
            cache_seq_id=request.index,
            max_length=len(encoded_prompt) + max_gen_tokens
            if max_gen_tokens is not None
            else None,
            tokens=np.array(encoded_prompt),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            ignore_eos=request.ignore_eos,
        )
        return context

    @property
    def _is_llama_tokenizer(self) -> bool:
        tokenizers = (
            LlamaTokenizer,
            LlamaTokenizerFast,
            CodeLlamaTokenizer,
            CodeLlamaTokenizerFast,
        )
        return isinstance(self.delegate, tokenizers)

    @property
    def _llama_whitespace_fix_dummy_token(self) -> tuple[int, int]:
        dummy_token_id = 33  # \x1e
        dummy_token_decoded = self.delegate.decode([dummy_token_id])
        return dummy_token_id, len(dummy_token_decoded)

    def _decode_with_llama_whitespace_fix(
        self, encoded: np.ndarray, **kwargs
    ) -> str:
        if encoded.shape == ():
            # The np.insert below will replace the token instead of prepend it
            # if the array is actually a scalar.  Reshape to a 1-length rank-1
            # array in this case.  See MODELS-467 for symptom.
            encoded = encoded.reshape((1,))
        decoded = self.delegate.decode(
            np.insert(encoded, 0, self._llama_whitespace_fix_dummy_token_id),
            **kwargs,
        )
        return decoded[self._llama_whitespace_fix_dummy_token_len :]


class TextAndVisionTokenizer(
    PipelineTokenizer[TextAndVisionContext, np.ndarray, TokenGeneratorRequest],
):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # If `max_length` is None, the max length will be taken
            # from the HuggingFace tokenizer_config.
            model_max_length=max_length,
        )
        # As we are adding special tokens during chat templating prior to tokenization,
        # when add_special_tokens=True, we duplicate BOS tokens specifically.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _wrap_str_message_content(
        self, messages: list[TokenGeneratorRequestMessage]
    ) -> list[TokenGeneratorRequestMessage]:
        # Wrap string type values of "content" key with "type": "text" and its
        # value. For example, if the message is {"content": "Hello, world!"},
        # it will be wrapped with {"type": "text", "text": "Hello, world!"}.
        # This is a workaround for LlamaVision's chat template:
        # https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/chat_template.json
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [
                    {"type": "text", "text": message["content"]}
                ]
            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if "content" in content and content["type"] == "text":
                        content["text"] = content.pop("content")
        return messages

    def apply_chat_template(
        self, messages: list[TokenGeneratorRequestMessage]
    ) -> str:
        # TODO: Refactor this.
        if self.model_path == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            messages = self._wrap_str_message_content(messages)
        try:
            templated_message = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception as e:
            msg = "apply_chat_template failed for TextAndVisionTokenizer"
            logger.warning(msg)
            logger.warning(str(e))
            prompt = []
            for message in messages:
                if isinstance(message["content"], str):
                    prompt.append(message["content"])
                elif isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            if "text" in content:
                                prompt.append(content["text"])
                            else:
                                prompt.append(content["content"])
            return "\n".join(prompt)

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return True

    async def encode(
        self, prompt: Union[str, Sequence[int]], add_special_tokens: bool = True
    ) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        encoded_prompt: np.ndarray
        if isinstance(prompt, str):
            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            if add_special_tokens:
                encoded_prompt = await run_with_default_executor(
                    self._encode_with_special_tokens, prompt
                )
            else:
                encoded_prompt = await run_with_default_executor(
                    self._encode_without_special_tokens, prompt
                )

            max_length = self.max_length or self.delegate.model_max_length
            if max_length and len(encoded_prompt) > max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length ({len(encoded_prompt)} > {max_length})."
                )
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(
        self, context: TextAndVisionContext, encoded: np.ndarray, **kwargs
    ) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        return self.delegate.decode(encoded, **kwargs)

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information like
        cache_seq_id and prompt from TokenGeneratorRequest."""
        prompt: Union[str, Sequence[int]]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
            add_special_tokens = False
        else:
            msg = f"{request} does not provide messages or prompt."
            raise ValueError(msg)

        # Load images.
        images = (
            [
                Image.open(io.BytesIO(image_data))
                for image_data in request.images
            ]
            if request.images
            else None
        )
        # PixtralProcessor returns a torch tensor or a list of torch tensors.
        # LlamaVision returns a np Array.
        processed_inputs = self.processor(
            text=prompt,
            images=images,
            add_special_tokens=add_special_tokens,
        )

        if "input_ids" not in processed_inputs:
            msg = "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            raise ValueError(msg)
        encoded_prompt = np.array(processed_inputs["input_ids"][0])

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.max_new_tokens is not None:
            max_new_tokens = request.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0],
            self.max_length,
            max_new_tokens,
        )

        extra_model_args = dict()

        if images is not None:
            if "pixel_values" not in processed_inputs:
                msg = "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                raise ValueError(msg)
            pixel_values = processed_inputs["pixel_values"][0]
            if isinstance(pixel_values, list):
                pixel_values = tuple(
                    tensor.numpy() if torch.is_tensor(tensor) else tensor
                    for tensor in pixel_values
                )
            elif torch.is_tensor(pixel_values):
                pixel_values = (pixel_values.numpy(),)
            if "aspect_ratio_ids" in processed_inputs:
                extra_model_args["aspect_ratio_ids"] = (
                    processed_inputs.aspect_ratio_ids
                )
            if "aspect_ratio_mask" in processed_inputs:
                extra_model_args["aspect_ratio_mask"] = (
                    processed_inputs.aspect_ratio_mask
                )
        else:
            pixel_values = ()

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        context = TextAndVisionContext(
            prompt=prompt,
            pixel_values=pixel_values,
            extra_model_args=extra_model_args,
            cache_seq_id=request.index,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else None,
            json_schema=json_schema,
            ignore_eos=request.ignore_eos,
        )
        return context
