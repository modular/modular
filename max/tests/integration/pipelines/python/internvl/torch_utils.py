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
    InternVLProcessor,
    find_closest_aspect_ratio,
)
from PIL import Image
from test_common.torch_utils import (
    TextGenerationRequest,
    run_text_generation_with_custom_image_processing,
)
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_pil_image_from_url(image_file: str) -> Image.Image:
    """Load PIL image from URL, file path, or existing PIL image."""
    if isinstance(image_file, str) and image_file.startswith("http"):
        return Image.open(requests.get(image_file, stream=True).raw).convert(
            "RGB"
        )
    else:
        # Handle local files.
        return Image.open(image_file).convert("RGB")


def build_transform(input_size: int) -> T.Compose:
    """Build transform pipeline for InternVL image preprocessing."""
    return T.Compose(
        [
            T.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def crop_into_patches(
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


def preprocess_image_to_tensor(
    pil_image: Image.Image, input_size: int = 448, max_num: int = 12
) -> torch.Tensor:
    """Preprocess image to tensor with dynamic patching - must match InternVLProcessor."""
    transform = build_transform(input_size=input_size)
    images = crop_into_patches(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


def run_text_generation(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    textgen_requests: Iterable[TextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
) -> list[dict]:
    """Run text generation for InternVL using InternVLProcessor for text formatting."""
    # Set up model tokens.
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = data_processor.convert_tokens_to_ids(
        IMG_CONTEXT_TOKEN
    )
    model.img_context_token_id = img_context_token_id

    # Create multimodal processor for text formatting.
    config = AutoConfig.from_pretrained(
        "OpenGVLab/InternVL3-8B-Instruct", trust_remote_code=True
    )
    processor = InternVLProcessor(data_processor, config)

    def internvl_request_processor(
        request: TextGenerationRequest,
    ) -> dict[str, torch.Tensor]:
        if request.is_multimodal:
            assert len(request.images) == 1
            pil_image = load_pil_image_from_url(request.images[0])

            # Use InternVLProcessor for text formatting.
            result = processor(text=request.prompt, images=[pil_image])

            # Resize and split the image into patches.
            pixel_values = preprocess_image_to_tensor(pil_image)

            return {
                "input_ids": torch.tensor(result["input_ids"])
                .unsqueeze(0)
                .to(device),
                "attention_mask": torch.tensor(result["attention_mask"])
                .unsqueeze(0)
                .to(device),
                "pixel_values": pixel_values.to(device).to(model.dtype),
            }
        else:
            encoded_prompt = data_processor.encode(
                request.prompt, return_tensors="pt"
            ).to(device)
            return {
                "input_ids": encoded_prompt,
                "attention_mask": torch.ones_like(encoded_prompt),
            }

    return run_text_generation_with_custom_image_processing(
        model=model,
        data_processor=data_processor,
        device=device,
        textgen_requests=textgen_requests,
        num_steps=num_steps,
        print_outputs=print_outputs,
        request_processor_fn=internvl_request_processor,
    )
