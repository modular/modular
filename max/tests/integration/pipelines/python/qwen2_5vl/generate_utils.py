# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities and testdataa for running qwen2.5vl in generate_llm_logits."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from qwen_vl_utils.vision_process import process_vision_info
from test_common.test_data import MockTextGenerationRequest
from test_common.torch_utils import _create_logits_store
from transformers import (
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Qwen2_5_VLProcessor,
)

INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_messages(
        prompt="Describe this image.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "s3://modular-bazel-artifacts-public/artifacts/model_testdata/qwen2_5vl_instruct_image.jpg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
        is_multimodal=True,
    ),
    MockTextGenerationRequest.with_messages(
        prompt="Compare these two images. What is the difference between them?",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "s3://modular-bazel-artifacts-public/artifacts/model_testdata/qwen2_5vl_instruct_image_a.jpg",
                    },
                    {
                        "type": "image",
                        "image": "s3://modular-bazel-artifacts-public/artifacts/model_testdata/qwen2_5vl_instruct_image_b.jpg",
                    },
                    {
                        "type": "text",
                        "text": "Compare these two images. What is the difference between them?",
                    },
                ],
            },
        ],
        is_multimodal=True,
    ),
]


def default_image_text_processor(
    data_processor: PreTrainedTokenizer
    | PreTrainedTokenizerFast
    | Qwen2_5_VLProcessor,
    image: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Default image+text processing for most vision-language models."""
    return data_processor(images=image, text=prompt, return_tensors="pt").to(
        device
    )


def run_text_generation(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer
    | PreTrainedTokenizerFast
    | Qwen2_5_VLProcessor,
    device: torch.device,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
    use_cache: bool | None = None,
) -> list[dict[str, Any]]:
    """Run text generation using standard data processor for both text and images."""

    def request_processor(
        request: MockTextGenerationRequest,
    ) -> dict[str, torch.Tensor]:
        if request.is_multimodal:
            texts = data_processor.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(request.messages)  # type: ignore
            return data_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
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
        request_processor_fn=request_processor,
    )


def run_text_generation_with_custom_image_processing(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer
    | PreTrainedTokenizerFast
    | Qwen2_5_VLProcessor,
    device: torch.device,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int,
    print_outputs: bool,
    request_processor_fn: Callable[
        [MockTextGenerationRequest], dict[str, torch.Tensor]
    ],
    use_cache: bool | None = None,
) -> list[dict[str, Any]]:
    """Run text generation with custom request processing for specialized models."""
    del device, use_cache  # Unused.
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
            # Trim outputs
            outputs = outputs[:, len(inputs["input_ids"][0]) :]
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
