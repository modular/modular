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

import io
import json
import logging
from collections.abc import Sequence
from typing import Union

import numpy as np
from max.interfaces import TextGenerationRequest
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    process_vision_info,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    max_tokens_to_generate,
)
from PIL import Image

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
    background_color=(255, 255, 255),  # noqa: ANN001
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


class Qwen2_5VLTokenizer(TextAndVisionTokenizer):
    """Qwen2.5VL-specific tokenizer that handles vision and text processing.

    This tokenizer integrates with HuggingFace's AutoProcessor
    to handle multimodal inputs for the Qwen2.5VL model.
    """

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
        images = None
        if request.images:
            images = [
                _convert_image_mode(Image.open(io.BytesIO(image_data)), "RGB")
                for image_data in request.images
            ]

        # Process vision info using qwen_vl_utils if using messages
        image_inputs = None
        video_inputs = None
        if request.messages:
            # process_vision_info returns (image_inputs, video_inputs, placeholder_text)
            # Convert messages to the format expected by qwen_vl_utils
            # TextGenerationRequestMessage is a TypedDict, so it's already dict-like
            messages_data = [dict(msg) for msg in request.messages]
            image_inputs, video_inputs, _ = process_vision_info(messages_data)
        else:
            # Fall back to using the loaded images
            image_inputs = images
            video_inputs = None

        # Use processor to handle text and vision inputs
        processed_inputs = self.processor(
            text=[prompt] if isinstance(prompt, str) else prompt,
            images=image_inputs or images,
            videos=video_inputs,
            padding=True,
            return_tensors="np",
        )

        if "input_ids" not in processed_inputs:
            msg = "input_ids not provided in AutoProcessor output"
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

        # Process vision inputs for Qwen2.5VL
        pixel_values: tuple[np.ndarray, ...] = tuple()
        if images is not None or image_inputs is not None:
            if "pixel_values" in processed_inputs:
                pixel_values_raw = processed_inputs["pixel_values"]

                # Convert torch tensor to numpy array
                if hasattr(pixel_values_raw, "numpy"):  # torch tensor
                    pixel_values_np = pixel_values_raw.numpy()
                    pixel_values = (pixel_values_np,)
                elif isinstance(pixel_values_raw, np.ndarray):
                    pixel_values = (pixel_values_raw,)
                else:
                    raise ValueError(
                        f"pixel_values is not a torch tensor or numpy array but {type(pixel_values_raw)}"
                    )

            # Extract image_grid_thw if present (Qwen2.5VL specific)
            if "image_grid_thw" in processed_inputs:
                image_grid_thw = processed_inputs["image_grid_thw"]
                # Convert torch tensor to numpy array if needed
                if hasattr(image_grid_thw, "numpy"):
                    extra_model_args["image_grid_thw"] = image_grid_thw.numpy()
                else:
                    extra_model_args["image_grid_thw"] = image_grid_thw

            # Extract other vision-related inputs
            if "attention_mask" in processed_inputs:
                attention_mask = processed_inputs["attention_mask"]
                # Convert torch tensor to numpy array if needed
                if hasattr(attention_mask, "numpy"):
                    extra_model_args["attention_mask"] = attention_mask.numpy()
                else:
                    extra_model_args["attention_mask"] = attention_mask

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
