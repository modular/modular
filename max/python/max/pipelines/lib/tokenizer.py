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
from max.interfaces import (
    ImageMetadata,
    PipelineTokenizer,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TokenBuffer,
)
from max.pipelines.core import TextAndVisionContext, TextContext
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
        """Returns the end-of-sequence token ID (0 for identity)."""
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> str:
        """Returns the prompt unchanged (identity encoding)."""
        return prompt

    async def decode(
        self,
        encoded: str,
        **kwargs,
    ) -> str:
        """Returns the encoded string unchanged (identity decoding)."""
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
        """Applies the delegate's chat template to the messages."""
        templated_message = self.delegate.apply_chat_template(
            [msg.model_dump() for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(templated_message, str)
        return templated_message

    @property
    def eos(self) -> int:
        """Returns the end-of-sequence token ID from the delegate."""
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> npt.NDArray[np.integer[Any]]:
        """Encodes the prompt to token ids via the delegate."""
        return np.array(self.delegate.encode(prompt))

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        """Decodes token ids to text via the delegate."""
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
    """Runs a callable in the default thread pool executor.

    Args:
        fn: Callable to run.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        The result of ``fn(*args, **kwargs)``.
    """
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
                        shipped with the Hugging Face model config. This allows
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
        """Applies the delegate chat template to messages (and optional tools)."""
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
        """Returns the end-of-sequence token ID from the delegate."""
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transforms the provided prompt into a token array."""
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
        """Applies the processor's chat template to the messages."""
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
        """Returns the end-of-sequence token ID from the delegate."""
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return True

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transforms the provided prompt into a token array."""
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
