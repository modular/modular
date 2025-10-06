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

"""Idefics3-specific tokenizer implementation."""

from __future__ import annotations

import functools
import io
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.interfaces import TextGenerationRequest, TextGenerationRequestMessage
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import TextAndVisionTokenizer
from PIL import Image
from PIL.Image import Image as ImageType
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig


class Idefics3Tokenizer(TextAndVisionTokenizer):
    """Idefics3-specific tokenizer. This class has new_context modified to use the transformers tokenizer and processor.

    This class only overrides __init__ to create a custom processor, while inheriting
    all other methods (including new_context) from TextAndVisionTokenizer.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        pipeline_config: PipelineConfig | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )

        # Set max_length after delegate is created (like parent class)
        self.max_length = max_length or self.delegate.model_max_length

        # Set up encode methods (copied from TextAndVisionTokenizer)
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )

        # Load config for image processing
        config = AutoConfig.from_pretrained(
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )

        # Get vision config overrides from pipeline config.
        vision_overrides = (
            pipeline_config.model_config.vision_config_overrides
            if pipeline_config
            else {}
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision
        )

        # Initialize default EOS token IDs (required by parent class new_context method)
        self._default_eos_token_ids = set([self.eos])

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        """Decode token array back into readable text, filtering out special tokens."""
        # Force skip_special_tokens=True to filter out tokens like <end_of_utterance>
        kwargs_with_special_filter = kwargs.copy()
        kwargs_with_special_filter["skip_special_tokens"] = True
        return self.delegate.decode(encoded, **kwargs_with_special_filter)

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        """Apply the chat template to the messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Content can be a string or list of multimodal content parts.

        Returns:
            The formatted prompt string with chat template applied.

        Raises:
            ValueError: If template application fails.
        """

        # Convert to text-only messages first
        text_messages: list[dict[str, Any]] = []
        for message in messages:
            text_message: dict[str, Any] = {"role": message.get("role")}
            content = message.get("content")

            if isinstance(content, str):
                text_message["content"] = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        # Handle both "content" and "text" keys
                        text_content = item.get("content") or item.get(
                            "text", ""
                        )
                        if text_content:
                            text_parts.append(text_content)
                    elif isinstance(item, dict) and item.get("type") == "image":
                        # Add image placeholder
                        text_parts.append("<image>")
                text_message["content"] = " ".join(text_parts)
            else:
                text_message["content"] = ""

            text_messages.append(text_message)

        templated_prompt = self.delegate.apply_chat_template(
            text_messages, tokenize=False, add_generation_prompt=True
        )

        return templated_prompt

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

        # Convert image bytes to PIL Image objects.
        if request.images is not None and len(request.images) > 0:
            images = []
            for image_bytes in request.images:
                try:
                    img: ImageType = Image.open(io.BytesIO(image_bytes))
                    # Ensure image is in RGB format to avoid channel format issues
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    # Validate image has reasonable dimensions
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError(
                            f"Invalid image dimensions: {img.size}"
                        )
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to process image: {e}") from e
        else:
            images = None

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

        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(processed_inputs["input_ids"])
        else:
            encoded_prompt = np.array(processed_inputs["input_ids"][0])

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        extra_model_args = dict()

        if images is not None:
            if "pixel_values" not in processed_inputs:
                raise ValueError(
                    "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                )
            pixel_values = processed_inputs["pixel_values"]

            if isinstance(pixel_values, list):
                pixel_values = tuple(pixel_values)
            elif isinstance(pixel_values, np.ndarray):
                # Handle the extra batch dimension that return_tensors="np" adds
                if pixel_values.ndim == 5:
                    # Remove the extra batch dimension (1, N, C, H, W) -> (N, C, H, W)
                    pixel_values = pixel_values.squeeze(0)
                pixel_values = (pixel_values,)
            else:
                raise ValueError(
                    f"pixel_values is not a numpy array but it is {type(pixel_values)}"
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

        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            pixel_values=pixel_values,
            extra_model_args=extra_model_args,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
        )
        return context


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
