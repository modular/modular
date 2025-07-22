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
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from max.pipelines.core import (
    TextAndVisionContext,
    TokenGeneratorRequest,
)
from max.pipelines.lib import TextAndVisionTokenizer
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)

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

        # Convert image bytes to PIL Image objects.
        if request.images is not None and len(request.images) > 0:
            images = [
                Image.open(io.BytesIO(image_bytes))
                for image_bytes in request.images
            ]
        else:
            images = None

        processed_inputs = self.processor(
            text=prompt, images=images, add_special_tokens=add_special_tokens
        )

        if "input_ids" not in processed_inputs:
            msg = "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            raise ValueError(msg)

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
                msg = "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                raise ValueError(msg)
            pixel_values = processed_inputs["pixel_values"]
            if isinstance(pixel_values, list):
                pixel_values = tuple(
                    tensor.numpy() if torch.is_tensor(tensor) else tensor
                    for tensor in pixel_values
                )
            elif torch.is_tensor(pixel_values):
                pixel_values = (pixel_values.numpy(),)
            elif isinstance(pixel_values, np.ndarray):
                pixel_values = (pixel_values,)
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

        context = TextAndVisionContext(
            request_id=request.id,
            prompt=prompt,
            eos_token_ids=eos_token_ids,
            pixel_values=pixel_values,
            extra_model_args=extra_model_args,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
        )
        context.assign_to_cache(request.index)
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
