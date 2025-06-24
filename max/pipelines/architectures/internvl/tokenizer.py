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
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# The token ID for "<IMG_CONTEXT>" in the InternVL tokenizer.
# This is used to identify where to insert image embeddings in the text.
IMAGE_CONTEXT_TOKEN_ID = 151667

# ImageNet normalization constants.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
            if min_num <= i * j <= max_num
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


def imagenet_normalize(img: Image.Image, input_size: int) -> np.ndarray:
    """Normalize image using ImageNet normalization.

    This converts PIL image to normalized numpy array with proper preprocessing.
    """
    # Convert to RGB if needed.
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize with BICUBIC interpolation.
    img = img.resize((input_size, input_size), Image.Resampling.BICUBIC)

    # Convert to numpy array and normalize to [0, 1].
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Apply ImageNet normalization.
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    return img_array


def crop_into_patches(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """Dynamically preprocess image with adaptive tiling."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio.
    target_ratios = list(
        set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target..
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height.
    # Note: Each individual patch will be square (image_size x image_size),
    # but the overall image is resized to maintain aspect ratio by using
    # different numbers of patches horizontally and vertically.
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split the image into patches.
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for y_block in range(target_aspect_ratio[1]):
        for x_block in range(target_aspect_ratio[0]):
            box = (
                x_block * image_size,
                y_block * image_size,
                (x_block + 1) * image_size,
                (y_block + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def preprocess_image_to_tensor(
    pil_image: Image.Image,
    *,
    input_size: int = 448,
    max_num: int = 12,
) -> np.ndarray:
    """Preprocess image to tensor with dynamic patching - must match InternVLProcessor."""
    images = crop_into_patches(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [imagenet_normalize(image, input_size) for image in images]
    return np.stack(pixel_values)


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

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding:
        """Converts a list of dictionaries with `"role"` and `"content"` keys to a formatted string.

        This method handles multimodal messages by extracting text content before
        forwarding to the HF tokenizer.
        """
        # Convert multimodal messages to text-only for the tokenizer
        text_messages = []
        for message in messages:
            text_message = {"role": message.get("role")}
            content = message.get("content")

            if isinstance(content, str):
                text_message["content"] = content
            elif isinstance(content, list):
                # Extract text from multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        # Handle both "content" and "text" keys
                        text_content = item.get("content") or item.get(
                            "text", ""
                        )
                        if text_content:
                            text_parts.append(text_content)
                text_message["content"] = " ".join(text_parts)
            else:
                text_message["content"] = ""

            text_messages.append(text_message)

        # Forward to the HF tokenizer with text-only messages
        return self.tokenizer.apply_chat_template(
            text_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
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

            # Replace <image> or <|image|> with InternVL's format.
            if "<image>" in processed_text:
                processed_text = processed_text.replace(
                    "<image>", image_tokens, 1
                )
            elif "<|image|>" in processed_text:
                # Handle the test prompt format with pipes.
                processed_text = processed_text.replace(
                    "<|image|>", image_tokens, 1
                )
            else:
                # If no <image> placeholder, prepend to text
                processed_text = image_tokens + "\n" + processed_text

            image_array = preprocess_image_to_tensor(
                image,
                input_size=self.image_size,
                max_num=self.max_dynamic_patch,
            )
            raw_pixel_values.append(image_array)

        # Tokenize the processed text
        text_inputs = self.tokenizer(
            processed_text, add_special_tokens=add_special_tokens
        )

        # Add basic pixel values - TextAndVisionTokenizer expects this
        # These are raw image arrays that will be preprocessed later in the model layer
        if raw_pixel_values:
            text_inputs["pixel_values"] = [raw_pixel_values]

            # Compute image token indices for optimization
            input_ids = text_inputs.get("input_ids", [])
            # Handle various input formats (list, nested list, numpy array)
            # by converting to numpy and flattening.
            seq = np.asarray(input_ids).ravel()

            image_token_indices = (
                (seq == IMAGE_CONTEXT_TOKEN_ID).nonzero()[0].astype(np.int32)
            )
            text_inputs["image_token_indices"] = image_token_indices

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

        # Create custom processor instead of AutoProcessor (which doesn't exist for InternVL)
        self.processor = InternVLProcessor(self.delegate, config)

        # Initialize default EOS token IDs (required by parent class new_context method)
        self._default_eos_token_ids = set([self.eos])
