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

"""InternVL-specific tokenizer implementation."""

from __future__ import annotations

import functools

import numpy as np
from max.pipelines.lib import TextAndVisionTokenizer
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def calculate_num_patches_for_image(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> int:
    """Calculate the number of patches for an image without actual preprocessing."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio.
    target_ratios = list(
        set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target.
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the number of blocks
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks


class InternVLProcessor:
    """Custom processor for InternVL that handles text tokenization and image token insertion.

    This mimics the interface expected by TextAndVisionTokenizer while handling
    InternVL's tokenizer-only approach (no AutoProcessor available).

    NOTE: This processor does NOT do actual image preprocessing (no torch/torchvision).
    It only handles text formatting and token insertion. Image preprocessing happens
    later in the model layer.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, config
    ):
        self.tokenizer = tokenizer
        self.config = config

        # InternVL configuration
        self.image_size = getattr(config.vision_config, "image_size", 448)
        self.max_dynamic_patch = getattr(config, "max_dynamic_patch", 12)

        # Image token configuration
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        # Calculate number of image tokens per patch
        patch_size = getattr(config.vision_config, "patch_size", 14)
        downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.num_image_token = int(
            (self.image_size // patch_size) ** 2 * (downsample_ratio**2)
        )

    def __call__(
        self,
        text: str,
        images: list[Image.Image] | None = None,
        add_special_tokens: bool = True,
    ) -> dict:
        """Process text and images for InternVL.

        This method needs to match the interface expected by TextAndVisionTokenizer.new_context().
        """
        if images is None or len(images) == 0:
            # Text-only case - just tokenize normally
            return self.tokenizer(text, add_special_tokens=add_special_tokens)

        # Process images and format prompt (without actual image preprocessing)
        processed_text = text
        raw_pixel_values = []

        for image in images:
            # Calculate number of patches based on image dimensions
            num_patches = calculate_num_patches_for_image(
                image,
                min_num=1,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=True,
            )

            # Format text with image tokens
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches)
                + self.IMG_END_TOKEN
            )

            # Replace <image> with InternVL's format
            if "<image>" in processed_text:
                processed_text = processed_text.replace(
                    "<image>", image_tokens, 1
                )
            else:
                # If no <image> placeholder, prepend to text
                processed_text = image_tokens + "\n" + processed_text

            # Convert PIL image to basic numpy array (no torch/torchvision preprocessing)
            # Just convert to RGB and numpy array - full preprocessing happens later
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert PIL image to numpy array (H, W, C format)
            image_array = np.array(image, dtype=np.float32)
            raw_pixel_values.append(image_array)

        # Tokenize the processed text
        text_inputs = self.tokenizer(
            processed_text, add_special_tokens=add_special_tokens
        )

        # Add basic pixel values - TextAndVisionTokenizer expects this
        # These are raw image arrays that will be preprocessed later in the model layer
        if raw_pixel_values:
            text_inputs["pixel_values"] = [raw_pixel_values]

        return text_inputs


class InternVLTokenizer(TextAndVisionTokenizer):
    """InternVL-specific tokenizer that handles tokenizer-only models with custom image processing.

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
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )

        # Set up encode methods (copied from TextAndVisionTokenizer)
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )

        # Load config for image processing
        config = AutoConfig.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # Create custom processor instead of AutoProcessor (which doesn't exist for InternVL)
        self.processor = InternVLProcessor(self.delegate, config)
