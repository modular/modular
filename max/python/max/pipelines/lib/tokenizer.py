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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    ImageMetadata,
    PipelineTokenizer,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.support.image import find_contiguous_ranges
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

TokenGeneratorContext = TypeVar("TokenGeneratorContext")

_P = ParamSpec("_P")
_R = TypeVar("_R")


class IdentityPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, str, TextGenerationRequest],
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
        encoded: str,
        **kwargs,
    ) -> str:
        if isinstance(encoded, str):
            return encoded
        return ""


class PreTrainedPipelineTokenizer(
    PipelineTokenizer[
        TokenGeneratorContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ],
):
    def __init__(
        self, delegate: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> None:
        assert isinstance(
            delegate, PreTrainedTokenizer | PreTrainedTokenizerFast
        )
        self.delegate = delegate

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        templated_message = self.delegate.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(templated_message, str)
        return templated_message

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> npt.NDArray[np.integer[Any]]:
        return np.array(self.delegate.encode(prompt))

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
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


async def run_with_default_executor(
    fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)


class TextTokenizer(
    PipelineTokenizer[
        TextContext, npt.NDArray[np.integer[Any]], TextGenerationRequest
    ]
):
    """Encapsulates creation of TextContext and specific token encode/decode logic.

    Args:
        model_path: Path to the model/tokenizer
        revision: Git revision/branch to use
        max_length: Maximum sequence length
        trust_remote_code: Whether to trust remote code from the model
        enable_llama_whitespace_fix: Enable whitespace fix for Llama tokenizers
        pipeline_config: Optional pipeline configuration
        chat_template: Optional custom chat template string to override the one
                        shipped with the HuggingFace model config. This allows
                        customizing the prompt formatting for different use cases.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        pipeline_config: PipelineConfig | None = None,
        chat_template: str | None = None,
        context_validators: list[Callable[[TextContext], None]] | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

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
            raise ValueError(
                f"Failed to load tokenizer from {model_path}. "
                "This can happen if:\n"
                "- The model is not fully supported by the transformers python package\n"
                "- Required configuration files are missing\n"
                "- The model path is incorrect\n"
                "- '--trust-remote-code' is needed but not set\n"
            ) from e

        # Override chat template if provided
        # This will be used by the delegate's apply_chat_template method automatically
        self._custom_template_provided = chat_template is not None
        if chat_template is not None:
            self.delegate.chat_template = chat_template
            logger.info(
                f"Set custom chat template on tokenizer for {model_path}"
            )

        # As we are adding special tokens during chat templating prior to tokenization,
        # when add_special_tokens=True, we duplicate BOS tokens specifically.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )
        self.max_length = max_length or self.delegate.model_max_length

        # configure Llama whitespace fix if needed
        self._enable_llama_whitespace_fix = (
            enable_llama_whitespace_fix and self._is_llama_tokenizer
        )
        (
            self._llama_whitespace_fix_dummy_token_id,
            self._llama_whitespace_fix_dummy_token_len,
        ) = self._llama_whitespace_fix_dummy_token

        # cache tokenizer eos token ids
        self._default_eos_token_ids = set([self.eos])

        self._context_validators = (
            context_validators if context_validators else []
        )

        if pipeline_config:
            huggingface_config = pipeline_config.model_config.huggingface_config
            if eos_token_id := getattr(
                huggingface_config, "eos_token_id", None
            ):
                if isinstance(eos_token_id, int):
                    self._default_eos_token_ids.add(eos_token_id)
                elif isinstance(eos_token_id, list):
                    self._default_eos_token_ids.update(eos_token_id)

    @staticmethod
    def _flatten_text_generation_request_message(
        messages: list[TextGenerationRequestMessage],
    ) -> list[dict[str, str]]:
        flattened_messages: list[dict[str, str]] = []
        for message in messages:
            flattened_message = {
                "role": message["role"],
                "content": "",
            }
            if isinstance(message["content"], str):
                flattened_message["content"] = message["content"]
            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if "type" not in content:
                        raise ValueError(
                            "Malformed message content, missing 'type' field"
                        )
                    if content["type"] != "text":
                        raise ValueError(
                            f"Unsupported content type: {content['type']}"
                        )

                    if flattened_message["content"] != "":
                        flattened_message["content"] += "\n"

                    flattened_message["content"] += content["text"]

                if "content" not in flattened_message:
                    raise ValueError(
                        "Malformed message content, missing 'content' field with type 'text'"
                    )
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}"
                )

            flattened_messages.append(flattened_message)

        return flattened_messages

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        chat_template_options: dict[str, Any] | None = None,
    ) -> str:
        chat_template_options = chat_template_options or {
            "add_generation_prompt": True
        }

        flattened_messages = self._flatten_text_generation_request_message(
            messages
        )

        try:
            templated_message = self.delegate.apply_chat_template(
                flattened_messages,
                tokenize=False,
                tools=tools,
                **chat_template_options,
            )
        except Exception as e:
            if self._custom_template_provided:
                # Provide additional context when a custom template is used
                error_msg = (
                    f"Failed to apply custom chat template. This may indicate an issue "
                    f"with your custom prompt template. Please check your template syntax "
                    f"and ensure it properly handles the provided messages and tools.\n\n"
                    f"Template variables available:\n"
                    f"- messages: List of conversation messages with 'role' and 'content' fields\n"
                    f"- tools: List of available tools (if provided)\n"
                    f"- add_generation_prompt: Boolean for adding generation prompt\n\n"
                    f"Original error: {type(e).__name__}: {str(e)}"
                )
                raise ValueError(error_msg) from e
            else:
                # Re-raise the original error for default templates
                raise

        assert isinstance(templated_message, str)
        return templated_message

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transform the provided prompt into a token array."""

        encoded_prompt: npt.NDArray[np.integer[Any]]
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

            if self.max_length and len(encoded_prompt) > self.max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length ({len(encoded_prompt)} > {self.max_length})."
                )

            encoded_prompt = np.array(encoded_prompt)
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
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

    async def _generate_prompt_and_token_ids(
        self,
        prompt: Sequence[int] | str | None,
        messages: list[TextGenerationRequestMessage] | None,
        tools: list[TextGenerationRequestTool] | None = None,
        chat_template_options: dict[str, Any] | None = None,
    ) -> tuple[str | list[int], npt.NDArray[np.integer[Any]]]:
        if prompt and messages:
            raise ValueError("both prompt and messages cannot be provided.")

        if isinstance(prompt, str):
            return prompt, await self.encode(prompt, add_special_tokens=True)
        elif isinstance(prompt, list):
            return prompt, await self.encode(prompt, add_special_tokens=True)
        elif isinstance(messages, list):
            prompt = self.apply_chat_template(
                messages, tools, chat_template_options
            )
            return prompt, await self.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError(
                "either prompt must be provided as a list[int] or str, or messages must be provided as a list[TextGenerationRequestMessage]"
            )

    async def _get_eos_variables(
        self,
        ignore_eos: bool,
        stop_token_ids: list[int] | None,
        stop: list[str] | None,
    ) -> tuple[set[int], list[list[int]]]:
        eos_token_ids = self._default_eos_token_ids
        eos_sequences = list()

        if ignore_eos:
            eos_token_ids = set()
        elif stop_token_ids:
            eos_token_ids.update(stop_token_ids)
        elif stop:
            eos_sequences = await self._encode_stop_criteria(stop)

        return eos_token_ids, eos_sequences

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        """Create a new TextContext object, leveraging necessary information from TextGenerationRequest."""
        # Encode Prompt / Messages
        prompt, token_ids = await self._generate_prompt_and_token_ids(
            prompt=request.prompt,
            messages=request.messages,
            tools=request.tools,
            chat_template_options=request.chat_template_options,
        )

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        eos_token_ids, eos_sequences = await self._get_eos_variables(
            request.sampling_params.ignore_eos,
            request.sampling_params.stop_token_ids,
            request.sampling_params.stop,
        )

        # Calculate Max Length
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            len(token_ids), self.max_length, max_new_tokens
        )

        context = TextContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            eos_sequences=eos_sequences,
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=np.array(token_ids),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            target_endpoint=request.target_endpoint,
        )

        for validator in self._context_validators:
            validator(context)

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
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
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

    async def _encode_stop_criteria(self, stop: list[str]) -> list[list[int]]:
        """Encodes `stop` to be used as stop criteria during generation."""
        stop_tokenized: list[list[int]] = []
        for stop_crit in stop:
            tokenized: list[int] = (
                await self.encode(stop_crit, False)
            ).tolist()
            stop_tokenized.append(tokenized)

        return stop_tokenized


class TextAndVisionTokenizer(
    PipelineTokenizer[
        TextAndVisionContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ],
):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        pipeline_config: PipelineConfig | None = None,
        context_validators: list[Callable[[TextAndVisionContext], None]]
        | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # If `max_length` is None, the max length will be taken
            # from the HuggingFace tokenizer_config.
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        config = AutoConfig.from_pretrained(
            model_path, revision=revision, trust_remote_code=trust_remote_code
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
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )
        self._default_eos_token_ids = set([self.eos])

        self._context_validators = (
            context_validators if context_validators else []
        )

        # Llama-3.2-11B-Vision uses image_token_index
        # Qwen2.5VL uses image_token_id
        # Pixtral uses image_token_index
        # ...
        vision_token_ids: list[int] = []
        for vision_token_id_name in [
            "image_token_id",
            "image_token_index",
        ]:
            if vision_token_id := getattr(config, vision_token_id_name, None):
                vision_token_ids.append(vision_token_id)
        if not vision_token_ids:
            raise ValueError("vision_token_id not found in model_config config")
        self.vision_token_ids = vision_token_ids

        # This is pixtral specific hack as it also has a image_break_token_id
        if image_break_token_id := getattr(
            self.processor, "image_break_token_id", None
        ):
            self.vision_token_ids.append(image_break_token_id)

    def _wrap_str_message_content(
        self, messages: list[TextGenerationRequestMessage]
    ) -> list[TextGenerationRequestMessage]:
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
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        # TODO: Refactor this.
        if self.model_path == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            messages = self._wrap_str_message_content(messages)
        templated_message = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(templated_message, str)
        return templated_message

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return True

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transform the provided prompt into a token array."""

        encoded_prompt: npt.NDArray[np.integer[Any]]
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
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        return self.delegate.decode(encoded, **kwargs)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information from TextGenerationRequest."""
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Load images.
        images = (
            [
                _convert_image_mode(Image.open(io.BytesIO(image_data)), "RGB")
                for image_data in request.images
            ]
            if request.images
            else None
        )

        # LlamaVision & InternVL returns a python list
        processed_inputs = self.processor(
            text=prompt,
            images=images,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )

        if "input_ids" not in processed_inputs:
            raise ValueError(
                "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            )

        # TODO: This is a hack to support both LlamaVision, Pixtral and InternVL.
        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(processed_inputs["input_ids"])
        else:
            encoded_prompt = np.array(processed_inputs["input_ids"][0])

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        extra_model_args = dict()

        if images is not None:
            if "pixel_values" not in processed_inputs:
                raise ValueError(
                    "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                )
            pixel_values = processed_inputs["pixel_values"][0]
            if isinstance(pixel_values, list):
                pixel_values = tuple(pixel_values)
            elif isinstance(pixel_values, np.ndarray):
                pixel_values = (pixel_values,)
            else:
                raise ValueError(
                    f"pixel_values is not a numpy array but it is {type(pixel_values)}"
                )

            if "aspect_ratio_ids" in processed_inputs:
                extra_model_args["aspect_ratio_ids"] = (
                    processed_inputs.aspect_ratio_ids
                )
            if "aspect_ratio_mask" in processed_inputs:
                extra_model_args["aspect_ratio_mask"] = (
                    processed_inputs.aspect_ratio_mask
                )
        else:
            pixel_values = tuple()

        # Pass through image token indices if present
        if "image_token_indices" in processed_inputs:
            extra_model_args["image_token_indices"] = processed_inputs[
                "image_token_indices"
            ]

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        if request.sampling_params.ignore_eos:
            eos_token_ids = set()
        else:
            eos_token_ids = self._default_eos_token_ids

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise ValueError(
                "encoded_prompt is greater than the max_length of the tokenizer"
            )

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, self.vision_token_ids
        )

        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            extra_model_args=extra_model_args,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
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


def _rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def _convert_image_mode(image: Image.Image, to_mode: str):  # noqa: ANN202
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return _rgba_to_rgb(image)
    else:
        return image.convert(to_mode)
