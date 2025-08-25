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

from __future__ import annotations

import functools
import io
import json
import logging
from collections.abc import Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from max.interfaces import TextGenerationRequest, TextGenerationRequestMessage
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    process_vision_info,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    max_tokens_to_generate,
)
from max.pipelines.lib.config import PipelineConfig
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

logger = logging.getLogger("max.pipelines")


def _convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image:
    """Convert image to the specified mode."""
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return _rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


def _rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


class Qwen2_5VLTokenizer(TextAndVisionTokenizer):
    """Qwen2.5VL-specific tokenizer that handles vision and text processing.

    This tokenizer uses separate AutoTokenizer and AutoImageProcessor
    to handle multimodal inputs for the Qwen2.5VL model, avoiding
    torchvision dependencies that come with AutoProcessor.
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
    ):
        """Initialize the tokenizer with separate image processor instead of AutoProcessor."""
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # Create encoding functions. Used by encode method in parent class.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )

        # Skip loading AutoProcessor and create AutoImageProcessor instead
        self.img_processor = AutoImageProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # Initialize EOS token IDs
        self._default_eos_token_ids = set([self.eos])
        self._default_eos_token_ids = set([self.eos])

        if pipeline_config:
            huggingface_config = pipeline_config.model_config.huggingface_config
            if eos_token_id := getattr(
                huggingface_config, "eos_token_id", None
            ):
                if isinstance(eos_token_id, int):
                    self._default_eos_token_ids.add(eos_token_id)
                elif isinstance(eos_token_id, list):
                    self._default_eos_token_ids.update(eos_token_id)

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        """Apply chat template using tokenizer directly (not processor)."""
        # Use tokenizer directly instead of processor to avoid AutoProcessor dependency
        templated_message = self.delegate.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(templated_message, str)
        return templated_message

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext for Qwen2.5VL processing.

        This method processes both text and vision inputs using the Qwen2.5VL
        processor and extracts the necessary components for model execution.
        """
        # Determine prompt
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

        # Load and process images
        image_inputs = None
        if request.messages:
            # process_vision_info returns (image_inputs, video_inputs, placeholder_text)
            # Convert messages to the format expected by qwen_vl_utils
            # TextGenerationRequestMessage is a TypedDict, so it's already dict-like
            messages_data = [dict(msg) for msg in request.messages]
            image_inputs, _, _ = process_vision_info(
                messages_data
            )  # We ignore video_inputs for image-only use case
        else:
            # Fall back to using the loaded images
            if request.images:
                image_inputs = [
                    _convert_image_mode(
                        Image.open(io.BytesIO(image_data)), "RGB"
                    )
                    for image_data in request.images
                ]

        # Step 1: Build chat text with tokenizer (not image processor)
        if isinstance(prompt, str):
            text = prompt
        else:
            # prompt is already processed tokens, convert back to text for processing
            text = self.delegate.decode(prompt, skip_special_tokens=True)

        # Step 2: Process images with image processor (if any)
        processed_images = {}
        if image_inputs:
            processed_images = self.img_processor(
                images=image_inputs, return_tensors="np"
            )

            # Step 3: Expand <|image_pad|> placeholders using image_grid_thw and merge_size**2
            if "image_grid_thw" in processed_images:
                grid = processed_images[
                    "image_grid_thw"
                ]  # List of (t, h, w) tuples
                merge_len = self.img_processor.merge_size**2

                # Expand placeholders for each image individually
                if "<|image_pad|>" in text:
                    for t, h, w in grid:
                        num_img_tokens = (t * h * w) // merge_len
                        # Replace first occurrence of <|image_pad|> with multiple <|image_pad|> tokens
                        # Use placeholder approach from example to avoid recursive replacement
                        placeholder_tokens = "<|placeholder|>" * num_img_tokens
                        text = text.replace(
                            "<|image_pad|>", placeholder_tokens, 1
                        )

                    # Convert all placeholders back to <|image_pad|> tokens
                    text = text.replace("<|placeholder|>", "<|image_pad|>")

        # Step 4: Tokenize the expanded text
        tokenizer_inputs = self.delegate(
            [text], padding=True, return_tensors=None
        )

        # Combine tokenizer and image processor outputs
        processed_inputs = {
            "input_ids": tokenizer_inputs["input_ids"],
            "attention_mask": tokenizer_inputs["attention_mask"],
        }

        # Add image processing results
        if processed_images:
            if "pixel_values" in processed_images:
                processed_inputs["pixel_values"] = processed_images[
                    "pixel_values"
                ]
            if "image_grid_thw" in processed_images:
                processed_inputs["image_grid_thw"] = processed_images[
                    "image_grid_thw"
                ]

        if "input_ids" not in processed_inputs:
            msg = "input_ids not generated by tokenizer"
            raise ValueError(msg)

        # Extract input_ids
        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(processed_inputs["input_ids"])
        else:
            encoded_prompt = np.array(processed_inputs["input_ids"][0])

        # Calculate max generation tokens
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        # Prepare extra model arguments
        extra_model_args = {}

        # Process vision inputs for Qwen2.5VL (image-only)
        pixel_values: tuple[npt.NDArray[Any], ...] = tuple()
        if image_inputs is not None:
            if "pixel_values" in processed_inputs:
                pixel_values_raw = processed_inputs["pixel_values"]

                # Handle numpy array from image processor
                if isinstance(pixel_values_raw, np.ndarray):
                    pixel_values = (pixel_values_raw,)
                elif hasattr(
                    pixel_values_raw, "numpy"
                ):  # fallback for torch tensor
                    pixel_values_np = pixel_values_raw.numpy()
                    pixel_values = (pixel_values_np,)
                else:
                    raise ValueError(
                        f"pixel_values is not a numpy array but {type(pixel_values_raw)}"
                    )

            # Extract image_grid_thw if present (Qwen2.5VL specific)
            if "image_grid_thw" in processed_inputs:
                image_grid_thw = processed_inputs["image_grid_thw"]
                # Handle numpy array from image processor
                if isinstance(image_grid_thw, np.ndarray):
                    extra_model_args["image_grid_thw"] = image_grid_thw
                elif hasattr(
                    image_grid_thw, "numpy"
                ):  # fallback for torch tensor
                    extra_model_args["image_grid_thw"] = image_grid_thw.numpy()
                else:
                    extra_model_args["image_grid_thw"] = np.array(
                        image_grid_thw
                    )

            # Extract attention_mask
            if "attention_mask" in processed_inputs:
                attention_mask = processed_inputs["attention_mask"]
                # Handle various formats from tokenizer
                if isinstance(attention_mask, list):
                    extra_model_args["attention_mask"] = np.array(
                        attention_mask
                    )
                elif isinstance(attention_mask, np.ndarray):
                    extra_model_args["attention_mask"] = attention_mask
                elif hasattr(
                    attention_mask, "numpy"
                ):  # fallback for torch tensor
                    extra_model_args["attention_mask"] = attention_mask.numpy()
                else:
                    extra_model_args["attention_mask"] = np.array(
                        attention_mask
                    )

        # Handle JSON schema if provided
        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        # Determine EOS token IDs
        if request.sampling_params.ignore_eos:
            eos_token_ids = set()
        else:
            eos_token_ids = self._default_eos_token_ids

        # Create and return context
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
        )
        return context
