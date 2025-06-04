# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for running torch models for testing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class TextGenerationRequest:
    """Request for text generation testing, supporting both text-only and multimodal inputs."""

    prompt: str
    """The text prompt to be processed by the model."""

    images: list[str]
    """List of image URLs or file paths. None for text-only requests."""

    @property
    def is_multimodal(self) -> bool:
        """Returns True if this request includes images."""
        return len(self.images) > 0

    @classmethod
    def text_only(cls, prompt: str) -> TextGenerationRequest:
        """Creates a text-only generation request."""
        return cls(prompt=prompt, images=[])

    @classmethod
    def with_images(
        cls, prompt: str, images: list[str]
    ) -> TextGenerationRequest:
        """Creates a multimodal generation request."""
        return cls(prompt=prompt, images=images)


def _process_images(images: Iterable[str]) -> list[Image.Image]:
    return [
        Image.open(requests.get(image, stream=True).raw) for image in images
    ]


def default_image_text_processor(
    data_processor, image, prompt: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """Default image+text processing for most vision-language models."""
    return data_processor(images=image, text=prompt, return_tensors="pt").to(
        device
    )


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
    requests: list[TextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
    use_cache: bool | None = None,
):
    saved_logits, store_logits = _create_logits_store()
    results = []

    for request in requests:
        if request.is_multimodal:
            processed_images = _process_images(request.images)
            # Assume one image per prompt for now.
            assert len(processed_images) == 1

            inputs = data_processor(
                images=processed_images[0],
                text=request.prompt,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=num_steps,
                do_sample=False,
                logits_processor=LogitsProcessorList([store_logits]),
                num_return_sequences=1,
                pad_token_id=getattr(data_processor, "eos_token_id", None),
                use_cache=use_cache,
            )
        else:
            # Process text-only inputs.
            encoded_prompt = data_processor.encode(
                request.prompt, return_tensors="pt"
            ).to(device)
            mask = torch.ones_like(encoded_prompt)

            outputs = model.generate(
                input_ids=encoded_prompt,
                attention_mask=mask,
                max_new_tokens=num_steps,
                do_sample=False,
                logits_processor=LogitsProcessorList([store_logits]),
                num_return_sequences=1,
                pad_token_id=getattr(data_processor, "eos_token_id", None),
                use_cache=use_cache,
            )

        if print_outputs:
            print(
                "Prompt:",
                f"{request.prompt[:100]}..."
                if len(request.prompt) > 100
                else request.prompt,
            )
            print(
                "Output:",
                data_processor.batch_decode(outputs, skip_special_tokens=True)[
                    0
                ],
            )

        # TODO: We likely want to track input image here too for multimodal requests.
        results.append({"prompt": request.prompt, "values": saved_logits[:]})
        saved_logits.clear()

    return results


def run_text_generation_with_custom_image_processing(
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    requests: list[TextGenerationRequest],
    num_steps: int,
    print_outputs: bool,
    image_loader_fn: Callable[[str], torch.Tensor],
    prompt_formatter_fn: Callable[
        [
            str,
            str,
            torch.Tensor,
            PreTrainedModel,
            PreTrainedTokenizer | PreTrainedTokenizerFast,
        ],
        tuple[str, torch.Tensor],
    ],
    model_setup_fn: Callable[
        [PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast],
        None,
    ],
):
    """Run text generation with custom image processing for specialized models like InternVL."""
    saved_logits, store_logits = _create_logits_store()

    # Call model setup function (e.g., for setting special tokens)
    model_setup_fn(model, data_processor)

    results = []

    # Process each multimodal request.
    for request in requests:
        # Use custom image loader on first image (assume single image for now).
        assert len(request.images) == 1
        pixel_values = image_loader_fn(request.images[0])

        # Use custom prompt formatter
        formatted_prompt, pixel_values = prompt_formatter_fn(
            request.prompt,
            request.images[0],
            pixel_values,
            model,
            data_processor,
        )

        # Process the formatted prompt
        inputs = data_processor(formatted_prompt, return_tensors="pt")
        processed_inputs = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        # Move to device
        pixel_values = (
            processed_inputs["pixel_values"].to(device).to(model.dtype)
        )
        input_ids = processed_inputs["input_ids"].to(device)
        attention_mask = processed_inputs["attention_mask"].to(device)

        # Generate with model
        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=num_steps,
            do_sample=False,
            logits_processor=LogitsProcessorList([store_logits]),
            num_return_sequences=1,
            pad_token_id=getattr(data_processor, "eos_token_id", None),
        )

        if print_outputs:
            print(
                "Prompt:",
                f"{request.prompt[:100]}..."
                if len(request.prompt) > 100
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
