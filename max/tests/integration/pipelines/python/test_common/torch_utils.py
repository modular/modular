# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for running torch models for testing."""

from typing import Iterable, List, Optional, Union

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


def _process_images(images: Iterable[str]) -> List[Image.Image]:
    return [
        Image.open(requests.get(image, stream=True).raw) for image in images
    ]


def run_text_generation(
    model: PreTrainedModel,
    data_processor: Union[
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        MllamaProcessor,
        PixtralProcessor,
    ],
    device: torch.device,
    prompts: Iterable[str],
    images: Optional[Iterable[str]] = None,
    num_steps=10,
):
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

    results = []
    if images:
        processed_images = _process_images(images)

        for image, prompt in zip(processed_images, prompts):
            inputs = data_processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(device)
            model.generate(
                **inputs,
                max_new_tokens=num_steps,
                do_sample=False,
                logits_processor=LogitsProcessorList([store_logits]),
                num_return_sequences=1,
            )

            # TODO: We likely want to track input image here too.
            results.append({"prompt": prompt, "values": saved_logits[:]})
            saved_logits.clear()
    else:
        for prompt in prompts:
            encoded_prompt = data_processor.encode(
                prompt, return_tensors="pt"
            ).to(device)
            mask = torch.ones_like(encoded_prompt)
            model.generate(
                input_ids=encoded_prompt,
                attention_mask=mask,
                max_new_tokens=num_steps,
                do_sample=False,
                logits_processor=LogitsProcessorList([store_logits]),
                num_return_sequences=1,
            )
            results.append({"prompt": prompt, "values": saved_logits[:]})
            saved_logits.clear()

    return results


def run_embeddings_generation(
    model: PreTrainedModel,
    data_processor: Union[
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    ],
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
        embeddings = output.last_hidden_state.cpu().detach().numpy()
        results.append({"prompt": prompt, "embeddings": embeddings})
    return results
