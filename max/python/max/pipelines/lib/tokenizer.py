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
import io
import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Device
from max.dtype import DType
from max.interfaces import (
    ImageMetadata,
    PipelineTokenizer,
    PixelGenerationRequest,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TokenBuffer,
)
from max.pipelines.core import PixelContext, TextAndVisionContext, TextContext
from max.support.image import find_contiguous_ranges, hash_image
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
from typing_extensions import ParamSpec

from .diffusion_schedulers import SchedulerFactory

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

TokenGeneratorContext = TypeVar("TokenGeneratorContext")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _handle_decode_overflow(
    encoded: npt.NDArray[np.integer[Any]],
    vocab_size: int,
) -> str:
    """Diagnose and raise a helpful OverflowError for token decoding issues.

    Args:
        encoded: The token array that caused the overflow.
        vocab_size: The tokenizer's vocabulary size.
        original_error: The original OverflowError that was caught.

    """
    issues = []

    if (encoded >= vocab_size).any():
        invalid_mask = encoded >= vocab_size
        invalid_indices = np.where(invalid_mask)[0]
        invalid_values = encoded[invalid_mask]
        issues.append(
            f"Token IDs exceeding vocab size ({vocab_size}) at indices "
            f"{invalid_indices.tolist()}: {invalid_values.tolist()}"
        )

    if (encoded < 0).any():
        negative_mask = encoded < 0
        negative_indices = np.where(negative_mask)[0]
        negative_values = encoded[negative_mask]
        issues.append(
            f"Negative token IDs at indices {negative_indices.tolist()}: "
            f"{negative_values.tolist()}"
        )

    if issues:
        error_msg = (
            f"OverflowError during token decoding. Invalid token IDs detected:\n"
            f"  {'; '.join(issues)}\n"
            f"  Vocab size: {vocab_size}, Array shape: {encoded.shape}, "
            f"dtype: {encoded.dtype}"
        )
    else:
        error_msg = (
            f"OverflowError during token decoding (no obvious invalid values). "
            f"Vocab size: {vocab_size}, Array shape: {encoded.shape}, "
            f"dtype: {encoded.dtype}, Token IDs: {encoded.tolist()}"
        )

    logger.error(error_msg)
    return error_msg


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
            [msg.model_dump() for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
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
        try:
            return self.delegate.decode(encoded, **kwargs)
        except OverflowError as e:
            error_msg = _handle_decode_overflow(encoded, len(self.delegate))
            raise OverflowError(error_msg) from e


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
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
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
            huggingface_config = pipeline_config.model.huggingface_config
            if eos_token_id := getattr(
                huggingface_config, "eos_token_id", None
            ):
                if isinstance(eos_token_id, int):
                    self._default_eos_token_ids.add(eos_token_id)
                elif isinstance(eos_token_id, list):
                    self._default_eos_token_ids.update(eos_token_id)

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        chat_template_options: dict[str, Any] | None = None,
    ) -> str:
        chat_template_options = chat_template_options or {
            "add_generation_prompt": True
        }

        try:
            templated_message = self.delegate.apply_chat_template(
                [message.flatten_content() for message in messages],
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

            def _encode_fn(
                prompt: str, add_special_tokens: bool
            ) -> npt.NDArray[np.integer[Any]]:
                return self.delegate.encode(
                    prompt, add_special_tokens=add_special_tokens
                )

            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            encoded_prompt = await run_with_default_executor(
                _encode_fn,
                prompt,
                add_special_tokens,
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

        try:
            return self.delegate.decode(encoded, **kwargs)
        except OverflowError as e:
            error_msg = _handle_decode_overflow(encoded, len(self.delegate))
            raise OverflowError(error_msg) from e

    async def _generate_prompt_and_token_ids(
        self,
        prompt: Sequence[int] | str | None,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        chat_template_options: dict[str, Any] | None = None,
    ) -> tuple[str | list[int], npt.NDArray[np.integer[Any]]]:
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
        _prompt, token_ids = await self._generate_prompt_and_token_ids(
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

        token_buffer = TokenBuffer(
            array=token_ids.astype(np.int64, copy=False),
        )

        context = TextContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            eos_sequences=eos_sequences,
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=token_buffer,
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
    """Encapsulates creation of TextAndVisionContext and specific token encode/decode logic."""

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
            # If `max_length` is None, the max length will be taken
            # from the HuggingFace tokenizer_config.
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # Use the pre-loaded HuggingFace config from pipeline_config
        config = pipeline_config.model.huggingface_config

        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )
        self._default_eos_token_ids = set([self.eos])

        huggingface_config = pipeline_config.model.huggingface_config
        if eos_token_id := getattr(huggingface_config, "eos_token_id", None):
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

        # Qwen2.5VL uses image_token_id
        # Pixtral uses image_token_index
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

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        # This converts between the Pydantic TextGenerationRequestMessage
        # to a dict for the HF delegate
        templated_message = self.processor.apply_chat_template(
            [msg.model_dump() for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
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

            def _encode_fn(
                prompt: str, add_special_tokens: bool
            ) -> npt.NDArray[np.integer[Any]]:
                return self.delegate.encode(
                    prompt, add_special_tokens=add_special_tokens
                )

            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            encoded_prompt = await run_with_default_executor(
                _encode_fn,
                prompt,
                add_special_tokens,
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
        try:
            return self.delegate.decode(encoded, **kwargs)
        except OverflowError as e:
            error_msg = _handle_decode_overflow(encoded, len(self.delegate))
            raise OverflowError(error_msg) from e

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information from TextGenerationRequest."""
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
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

        # InternVL returns a python list
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

        # TODO: This is a hack to support both Pixtral and InternVL.
        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(
                processed_inputs["input_ids"], dtype=np.int64
            )
        else:
            encoded_prompt = np.array(
                processed_inputs["input_ids"][0], dtype=np.int64
            )

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

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )

        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            extra_model_args=extra_model_args,
            tokens=token_buffer,
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


class PixelGenerationTokenizer(
    PipelineTokenizer[
        PixelContext, npt.NDArray[np.integer[Any]], PixelGenerationRequest
    ]
):
    """Encapsulates creation of PixelContext and specific token encode/decode logic.

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
        pipeline_config: PipelineConfig,
        subfolder: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        subfolder_2: str | None = None,
        max_length_2: int | None = None,
        chat_template: str | None = None,
        context_validators: list[Callable[[PixelContext], None]] | None = None,
        wrap_prompt_as_chat: bool = False,
        default_chat_template_options: dict[str, Any] | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.delegate_2 = None
        self.max_length_2 = None

        try:
            self.delegate = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                model_max_length=max_length,
                subfolder=subfolder,
            )
            self.max_length = max_length or self.delegate.model_max_length
            if subfolder_2 is not None:
                self.delegate_2 = AutoTokenizer.from_pretrained(
                    model_path,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    model_max_length=max_length_2,
                    subfolder=subfolder_2,
                )
                self.max_length_2 = (
                    max_length_2 or self.delegate_2.model_max_length
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
        # This will be used by the delegate(s)'s apply_chat_template method automatically
        self._custom_template_provided = chat_template is not None
        if chat_template is not None:
            self.delegate.chat_template = chat_template
            logger.info(
                f"Set custom chat template on tokenizer for {model_path}"
            )

        self._context_validators = (
            context_validators if context_validators else []
        )

        # Prompt wrapping and default chat template options
        self._wrap_prompt_as_chat = wrap_prompt_as_chat
        self._default_chat_template_options = default_chat_template_options

        # Extract diffusers_config
        if not pipeline_config or not hasattr(
            pipeline_config.model, "diffusers_config"
        ):
            raise ValueError(
                "pipeline_config.model.diffusers_config is required for PixelGenerationTokenizer. "
                "Please provide a pipeline_config with a valid diffusers_config."
            )
        if pipeline_config.model.diffusers_config is None:
            raise ValueError(
                "pipeline_config.model.diffusers_config cannot be None. "
                "Please provide a valid diffusers_config."
            )
        self.diffusers_config = pipeline_config.model.diffusers_config

    def _calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def _prepare_latent_image_ids(
        height: int, width: int
    ) -> npt.NDArray[np.float32]:
        latent_image_ids = np.zeros((height, width, 3), dtype=np.float32)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + np.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + np.arange(width)[None, :]
        )

        (
            latent_image_id_height,
            latent_image_id_width,
            latent_image_id_channels,
        ) = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels,
        )

        return latent_image_ids

    def _randn_tensor(
        self,
        shape: tuple,
        seed: int | None,
    ) -> npt.NDArray[np.float32]:
        rng = np.random.RandomState(seed)
        return rng.standard_normal(shape).astype(np.float32)

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        seed: int | None,
        vae_scale_factor: int,
    ) -> npt.NDArray[np.float32]:
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        latents = self._randn_tensor(shape, seed)
        latent_image_ids = self._prepare_latent_image_ids(
            height // 2, width // 2
        )

        return latents, latent_image_ids

    def _retrieve_timesteps(
        self,
        scheduler: Any,
        num_inference_steps: int | None = None,
        device: Device | None = None,
        sigmas: list[float] | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray[np.float32], int]:
        r"""
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`Any`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`Device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` must be `None`.

        Returns:
            `Tuple[npt.NDArray[np.float32], int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if sigmas is not None:
            try:
                scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            except TypeError as e:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                ) from e
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(
                num_inference_steps, device=device, **kwargs
            )
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    async def _generate_tokens_ids(
        self,
        prompt: str | None,
        prompt_2: str | None = None,
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,
        messages: list[TextGenerationRequestMessage] | None = None,
        chat_template_options: dict[str, Any] | None = None,
        do_true_cfg: bool = False,
    ) -> tuple[
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.bool],
        npt.NDArray[np.integer[Any]] | None,
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]] | None,
    ]:
        """Tokenize prompt(s) with encoder model(s).

        Returns:
            Tuple of (token_ids, token_ids_2, negative_token_ids, negative_token_ids_2).
            token_ids_2 and negative_token_ids_2 are None if no secondary tokenizer is configured.
        """
        if prompt is not None:
            token_ids, attn_mask = await self.encode(prompt)
        elif messages is not None:
            templated = self.apply_chat_template(
                messages, chat_template_options
            )
            token_ids, attn_mask = await self.encode(templated)

        token_ids_2: npt.NDArray[np.integer[Any]] | None = None
        if self.delegate_2 is not None:
            token_ids_2, _attn_mask_2 = await self.encode(
                prompt_2 or prompt,
                use_secondary=True,
            )

        negative_token_ids: npt.NDArray[np.integer[Any]] | None = None
        negative_token_ids_2: npt.NDArray[np.integer[Any]] | None = None
        if do_true_cfg:
            negative_token_ids, _attn_mask_neg = await self.encode(
                negative_prompt
            )
            if self.delegate_2 is not None:
                negative_token_ids_2, _attn_mask_neg_2 = await self.encode(
                    negative_prompt_2 or negative_prompt,
                    use_secondary=True,
                )

        return (
            token_ids,
            attn_mask,
            token_ids_2,
            negative_token_ids,
            negative_token_ids_2,
        )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        chat_template_options: dict[str, Any] | None = None,
    ) -> str:
        templated_message = self.delegate.apply_chat_template(
            [message.flatten_content() for message in messages],
            tokenize=False,
            **chat_template_options,
        )
        if not isinstance(templated_message, str):
            raise ValueError("Chat template did not return a string")
        return templated_message

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self,
        prompt: str,
        *,
        use_secondary: bool = False,
    ) -> tuple[npt.NDArray[np.integer[Any]], npt.NDArray[np.bool_]]:
        """Transform the provided prompt into a token array."""

        delegate = self.delegate_2 if use_secondary else self.delegate
        max_sequence_length = (
            self.max_length_2 if use_secondary else self.max_length
        )

        tokenizer_output: npt.NDArray[np.integer[Any]]

        def _encode_fn(prompt: str) -> npt.NDArray[np.integer[Any]]:
            return delegate(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
            )

        # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
        # Add a standard (non-async) lock in the executor thread if needed.
        tokenizer_output = await run_with_default_executor(_encode_fn, prompt)

        if max_sequence_length and len(tokenizer_output) > max_sequence_length:
            raise ValueError(
                f"Input string is larger than tokenizer's max length ({len(tokenizer_output)} > {max_sequence_length})."
            )

        encoded_prompt = np.array(tokenizer_output.input_ids)
        attention_mask = np.array(tokenizer_output.attention_mask).astype(
            np.bool_
        )

        return encoded_prompt, attention_mask

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        raise NotImplementedError(
            "Decoding is not implemented for this tokenizer."
        )

    async def new_context(
        self, request: PixelGenerationRequest
    ) -> PixelContext:
        """Create a new PixelContext object, leveraging necessary information from PixelGenerationRequest."""
        prompt: str | None = request.prompt
        messages: list[TextGenerationRequestMessage] | None = None

        do_true_cfg = (
            request.true_cfg_scale > 1.0 and request.negative_prompt is not None
        )

        if self._wrap_prompt_as_chat:
            messages = [
                TextGenerationRequestMessage(role="user", content=prompt)
            ]
            prompt = None
            if self._default_chat_template_options:
                chat_template_options = self._default_chat_template_options

        # 1. Tokenize prompts
        (
            token_ids,
            attn_mask,
            token_ids_2,
            negative_token_ids,
            negative_token_ids_2,
        ) = await self._generate_tokens_ids(
            prompt,
            request.prompt_2,
            request.negative_prompt,
            request.negative_prompt_2,
            messages,
            chat_template_options,
            do_true_cfg,
        )

        token_buffer = TokenBuffer(
            array=token_ids.astype(np.int64, copy=False),
        )
        mask_buffer = TokenBuffer(
            array=attn_mask.astype(np.bool_, copy=False),
        )
        token_buffer_2 = None
        if token_ids_2 is not None:
            token_buffer_2 = TokenBuffer(
                array=token_ids_2.astype(np.int64, copy=False),
            )
        negative_token_buffer = TokenBuffer(
            array=negative_token_ids.astype(np.int64, copy=False),
        )
        negative_token_buffer_2 = None
        if negative_token_ids_2 is not None:
            negative_token_buffer_2 = TokenBuffer(
                array=negative_token_ids_2.astype(np.int64, copy=False),
            )

        # 3. Resolve image dimensions
        # Get defaults from diffusers_config
        vae_config = self.diffusers_config.components["vae"].config_dict
        transformer_config = self.diffusers_config.components[
            "transformer"
        ].config_dict
        scheduler_config = self.diffusers_config.components[
            "scheduler"
        ].config_dict

        # Compute vae_scale_factor from block_out_channels
        block_out_channels = vae_config.get("block_out_channels", None)
        vae_scale_factor = (
            2 ** (len(block_out_channels) - 1) if block_out_channels else 8
        )

        default_sample_size = 128
        height = request.height or default_sample_size * vae_scale_factor
        width = request.width or default_sample_size * vae_scale_factor
        num_channels_latents = transformer_config.in_channels // 4

        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        mu = self._calculate_shift(
            image_seq_len,
            scheduler_config.get("base_image_seq_len", 256),
            scheduler_config.get("max_image_seq_len", 4096),
            scheduler_config.get("base_shift", 0.5),
            scheduler_config.get("max_shift", 1.15),
        )

        # Create scheduler from config
        scheduler_component = self.diffusers_config.components["scheduler"]
        scheduler = SchedulerFactory.create(
            scheduler_component.class_name, scheduler_config
        )

        sigmas = np.linspace(
            1.0,
            1 / request.num_inference_steps,
            request.num_inference_steps,
        )
        if (
            hasattr(scheduler.config, "use_flow_sigmas")
            and scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        timesteps, num_inference_steps = self._retrieve_timesteps(
            scheduler,
            request.num_inference_steps,
            CPU(),
            sigmas=sigmas,
            mu=mu,
        )
        if request.model_name == "Tongyi-MAI/Z-Image-Turbo":
            timesteps = (1000 - timesteps) / 1000
        else:
            timesteps = timesteps / 1000
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        latents, latent_image_ids = self._prepare_latents(
            request.num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            request.seed,
            vae_scale_factor,
        )

        if transformer_config.guidance_embeds:
            guidance = Tensor.constant(
                [request.guidance_scale], device=CPU(), dtype=DType.float32
            )
        else:
            guidance = None

        # 5. Build the context
        context = PixelContext(
            request_id=request.request_id,
            max_text_encoder_length=self.max_length,
            tokens=token_buffer,
            mask=mask_buffer,
            tokens_2=token_buffer_2,
            negative_token_ids=negative_token_buffer,
            negative_token_ids_2=negative_token_buffer_2,
            timesteps=timesteps,
            sigmas=scheduler.sigmas,
            latents=latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images_per_prompt,
            model_name=request.model_name,
        )

        for validator in self._context_validators:
            validator(context)

        return context
