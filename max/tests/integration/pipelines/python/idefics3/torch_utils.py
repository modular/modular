# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for running torch models for testing."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from test_common.storage import load_image
from test_common.test_data import MockTextGenerationRequest
from test_common.torch_utils import _create_logits_store
from transformers import (
    LogitsProcessorList,
    MllamaProcessor,
    PixtralProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def run_text_generation(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer
    | PreTrainedTokenizerFast
    | MllamaProcessor
    | PixtralProcessor,
    device: torch.device,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
    use_cache: bool | None = None,
):
    """Run text generation using standard data processor for both text and images."""

    def idefics_request_processor(
        request: MockTextGenerationRequest,
    ) -> dict[str, torch.Tensor]:
        if request.is_multimodal:
            processed_images = [load_image(image) for image in request.images]
            return data_processor(
                images=processed_images,
                text=request.prompt,
                return_tensors="pt",
            ).to(device)
        else:
            return data_processor(
                text=request.prompt,
                return_tensors="pt",
            ).to(device)

    return run_text_generation_with_custom_image_processing(
        model=model,
        data_processor=data_processor,
        device=device,
        textgen_requests=textgen_requests,
        num_steps=num_steps,
        print_outputs=print_outputs,
        use_cache=use_cache,
        request_processor_fn=idefics_request_processor,
    )


def run_text_generation_with_custom_image_processing(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer
    | PreTrainedTokenizerFast
    | MllamaProcessor
    | PixtralProcessor,
    device: torch.device,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int,
    print_outputs: bool,
    request_processor_fn: Callable[
        [MockTextGenerationRequest], dict[str, torch.Tensor]
    ],
    use_cache: bool | None = None,
):
    """Run text generation with custom request processing for specialized models."""
    saved_logits, store_logits = _create_logits_store()
    results = []

    for request in textgen_requests:
        inputs = request_processor_fn(request)

        outputs = model.generate(
            **inputs,
            max_new_tokens=num_steps,
            do_sample=False,
            logits_processor=LogitsProcessorList([store_logits]),
            num_return_sequences=1,
            pad_token_id=getattr(data_processor, "eos_token_id", None),
        )

        if print_outputs:
            print(
                "Prompt:",
                f"{request.prompt[:100]}...{request.prompt[-100:]}"
                if len(request.prompt) > 200
                else request.prompt,
            )
            print(
                "Output:",
                data_processor.batch_decode(outputs, skip_special_tokens=True)[
                    0
                ],
            )

        results.append({"prompt": request.prompt, "values": saved_logits[:]})
        saved_logits.clear()

    return results
