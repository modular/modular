# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for running torch models for testing."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import requests
import torch
from PIL import Image
from transformers import (
    LogitsProcessorList,
    MllamaProcessor,
    PixtralProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from test_common.test_data import MockTextGenerationRequest


def _create_logits_store() -> tuple[list[dict], Callable]:
    """Create a logits storage function and container.

    The `saved_logits` is captured into the `store_logits` closure, which is
    injected into `model.generate` as a logits processor.
    This allows saving the logits.
    """
    saved_logits = []

    def store_logits(input_ids: torch.LongTensor, scores: torch.FloatTensor):
        _ = input_ids  # Unused.
        # Currently always passing in one batch at a time.
        scores_np = scores[0].cpu().detach().numpy()
        next_token = scores_np.argmax(axis=-1)
        saved_logits.append(
            {
                "next_token": next_token,
                "next_token_logits": scores_np[next_token],
                "logits": scores_np,
            }
        )
        return scores

    return saved_logits, store_logits


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

    def standard_request_processor(
        request: MockTextGenerationRequest,
    ) -> dict[str, torch.Tensor]:
        if len(request.images) > 0:
            processed_images = [
                Image.open(requests.get(image, stream=True).raw)
                for image in request.images
            ]
            assert len(processed_images) == 1
            return data_processor(
                images=processed_images[0],
                text=request.prompt,
                return_tensors="pt",
            ).to(device)
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
        use_cache=use_cache,
        request_processor_fn=standard_request_processor,
    )


def run_text_generation_with_custom_image_processing(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
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
        generate_kwargs = request_processor_fn(request)
        outputs = model.generate(
            **generate_kwargs,
            max_new_tokens=num_steps,
            do_sample=False,
            logits_processor=LogitsProcessorList([store_logits]),
            num_return_sequences=1,
            pad_token_id=getattr(data_processor, "eos_token_id", None),
            # Only pass use_cache if it's not None to avoid conflicts with
            # models such as InternVL that hardcode use_cache in their
            # generate_kwargs.
            **({"use_cache": use_cache} if use_cache is not None else {}),
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


def run_embeddings_generation(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    prompts: Iterable[str],
):
    """Generates embeddings for the input prompts."""
    results = []
    for prompt in prompts:
        encoded_input = data_processor(
            [prompt], padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        output = model(**encoded_input)
        embeddings = (
            output.last_hidden_state.cpu().detach().to(torch.float32).numpy()
        )
        embeddings = embeddings[0]
        results.append({"prompt": prompt, "embeddings": embeddings})
    return results
