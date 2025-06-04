# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# SPDX-FileCopyrightText: 2025 Modular
# SPDX-FileCopyrightText: Copyright (c) 2024 OpenGVLab
# SPDX-License-Identifier: MIT

"""InternVL-specific utilities for image preprocessing and model inference."""

from __future__ import annotations

from collections.abc import Iterable

import requests
import torch
import torchvision.transforms as T
from max.pipelines.architectures.internvl.tokenizer import (
    find_closest_aspect_ratio,
)
from PIL import Image
from test_common.torch_utils import (
    TextGenerationRequest,
    run_text_generation_with_custom_image_processing,
)
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """Build transform pipeline for InternVL image preprocessing."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def dynamic_preprocess(
    image: Image.Image,
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
            if i * j <= max_num and i * j >= min_num
        )
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target.
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(
    image_file: str | Image.Image,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    """Load and preprocess image for InternVL."""
    if isinstance(image_file, str) and image_file.startswith("http"):
        # Download image from URL
        image = Image.open(requests.get(image_file, stream=True).raw).convert(
            "RGB"
        )
    elif isinstance(image_file, str):
        # Load from file path
        image = Image.open(image_file).convert("RGB")
    else:
        # Assume it's already a PIL Image
        image = image_file.convert("RGB")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def _setup_internvl_model(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> None:
    """Set up InternVL model with required special tokens."""
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = data_processor.convert_tokens_to_ids(
        IMG_CONTEXT_TOKEN
    )
    model.img_context_token_id = img_context_token_id


def _format_internvl_prompt(
    prompt: str,
    image: str,
    pixel_values: torch.Tensor,
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> tuple[str, torch.Tensor]:
    """Format prompt for InternVL with proper image tokens."""
    # Get actual patch count from preprocessed image
    num_patches = pixel_values.shape[0]

    # InternVL-specific tokens
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    # Build the proper prompt with image tokens
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    # Replace <image> with InternVL's format, accounting for actual number of patches
    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * (model.num_image_token * num_patches)
        + IMG_END_TOKEN
    )
    formatted_prompt = prompt.replace("<image>", image_tokens, 1)

    return formatted_prompt, pixel_values


def run_text_generation(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    requests: Iterable[TextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
) -> list[dict]:
    """Run text generation for InternVL with custom image preprocessing."""
    return run_text_generation_with_custom_image_processing(
        model=model,
        data_processor=data_processor,
        device=device,
        requests=requests,
        num_steps=num_steps,
        print_outputs=print_outputs,
        image_loader_fn=load_image,
        prompt_formatter_fn=_format_internvl_prompt,
        model_setup_fn=_setup_internvl_model,
    )
