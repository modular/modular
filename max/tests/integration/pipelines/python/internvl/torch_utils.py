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
from max.pipelines.architectures.internvl.tokenizer import (
    InternVLProcessor,
    preprocess_image_to_tensor,
)
from PIL import Image
from test_common.torch_utils import (
    TextGenerationRequest,
    run_text_generation_with_custom_image_processing,
)
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def load_pil_image_from_url(image_file: str) -> Image.Image:
    """Load PIL image from URL, file path, or existing PIL image."""
    if isinstance(image_file, str) and image_file.startswith("http"):
        return Image.open(requests.get(image_file, stream=True).raw).convert(
            "RGB"
        )
    else:
        # Handle local files.
        return Image.open(image_file).convert("RGB")


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
