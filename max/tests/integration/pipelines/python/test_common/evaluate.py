# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for evaluating models and comparing the logits."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import numpy as np
import requests
from max.nn.kv_cache import KVCacheInputsSequence
from max.pipelines import PipelineModel
from max.pipelines.core import PipelineTokenizer, TokenGeneratorRequest
from typing_extensions import NotRequired


class TokenInfo(TypedDict):
    """Information about a token in the output."""

    next_token: int
    """The next token in the output."""
    next_token_logits: float
    """The logits for the next token."""
    logits: np.ndarray
    """The logits for the token."""


class ModelOutput(TypedDict):
    """The prompt and the output of a model run."""

    prompt: str
    """The prompt that was used to generate the output."""
    values: NotRequired[list[TokenInfo]]
    """Outputs from a text generation model."""
    embeddings: NotRequired[np.ndarray]
    """Outputs from a text embedding model."""


@dataclass(frozen=True)
class TextGenerationRequest:
    """Request for text generation testing, supporting both text-only and multimodal inputs."""

    prompt: str
    """The text prompt to be processed by the model."""

    images: list[str]
    """List of image URLs or file paths. Empty for text-only requests."""

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


NUM_STEPS = 10


def resolve_image_from_url(image_ref: str) -> bytes:
    return requests.get(image_ref).content


def run_model(
    model: PipelineModel,
    tokenizer: PipelineTokenizer,
    requests: Iterable[TextGenerationRequest],
    num_steps: int = NUM_STEPS,
    print_outputs: bool = False,
    batch_size: int = 1,
    reference: list[ModelOutput] | None = None,
) -> list[dict[str, Any]]:
    """Runs the model for N steps on each request provided."""
    return asyncio.run(
        run_model_async(
            model,
            tokenizer,
            requests=requests,
            num_steps=num_steps,
            print_outputs=print_outputs,
            batch_size=batch_size,
            reference=reference,
        )
    )


async def run_model_async(
    model: PipelineModel,
    tokenizer: PipelineTokenizer,
    requests: Iterable[TextGenerationRequest],
    num_steps: int = NUM_STEPS,
    print_outputs: bool = False,
    batch_size: int = 1,
    reference: list[ModelOutput] | None = None,
) -> list[dict[str, Any]]:
    """Runs the model for N steps on each request provided."""
    assert batch_size >= 1

    results = []

    async def _evaluate_batch(
        batch_prompts: dict[str, str],
        batch_contexts: dict[str, Any],
        batch_reference: dict[str, ModelOutput],
    ) -> None:
        values: dict[str, list[Any]] = {req_id: [] for req_id in batch_contexts}
        for _ in range(num_steps):
            is_eos = next_token_with_logits(
                model, batch_contexts, values, tokenizer.eos, batch_reference
            )
            if is_eos:
                break
        for req_id, prompt in batch_prompts.items():
            context = batch_contexts[req_id]
            results.append({"prompt": prompt, "values": values[req_id]})
            if print_outputs:
                print(
                    "Prompt:",
                    f"{prompt[:100]}...{prompt[-100:]}"
                    if len(prompt) > 200
                    else prompt,
                )
                print(
                    "Output:",
                    await tokenizer.decode(
                        context,
                        np.array(
                            [v["next_token"] for v in values[req_id]],
                            dtype=np.int64,
                        ),
                    ),
                )
            model.kv_manager.release(context.cache_seq_id)

    # Evaluate requests.
    batch_contexts: dict[str, Any] = {}
    batch_prompts: dict[str, str] = {}
    batch_reference: dict[str, ModelOutput] = {}

    for i, request in enumerate(requests):
        curr_req_id = str(uuid.uuid4())

        context = await tokenizer.new_context(
            TokenGeneratorRequest(
                id="",
                index=len(batch_contexts),
                prompt=request.prompt,
                model_name="llama3",
                # Download images for this specific request.
                images=[
                    resolve_image_from_url(image_url)
                    for image_url in request.images
                ],
            )
        )
        batch_prompts[curr_req_id] = request.prompt
        batch_contexts[curr_req_id] = context
        if reference:
            batch_reference[curr_req_id] = reference[i]
        if len(batch_contexts) == batch_size:
            await _evaluate_batch(
                batch_prompts, batch_contexts, batch_reference
            )
            batch_prompts.clear()
            batch_contexts.clear()
            batch_reference.clear()
    if batch_contexts:
        await _evaluate_batch(batch_prompts, batch_contexts, batch_reference)

    return results


def next_token_with_logits(
    model: PipelineModel,
    req_to_context_dict: dict[str, Any],
    update_values: dict[str, list[Any]],
    eos_token: Optional[int] = None,
    req_to_reference_dict: dict[str, ModelOutput] = {},
) -> bool:
    """Generates the next token and stores the logits.

    This method runs llama3.execute, stores the logits, and updates the context
    with the next token.

    Args:
        model: Model to execute.
        req_to_context_dict: Dictionary of request ids to Llama3Context.
        update_values: Dictionary of request ids to lists of next_token &
            logits. These lists are updated in this method.
        eos_token: Encoded end-of-sequence token used to signal the early
            stopping of token generation. If not provided, generation may
            continue past EOS token.
        req_to_reference_dict: Dictionary of request ids to ModelOutput.
            If there is a reference for a request, next token will select the
            same tokens as the reference.

    Returns:
        bool: True if the token is an end-of-sentence token, otherwise False.
    """
    # Flatten our batch for consistent indexing.
    context_batch = list(req_to_context_dict.values())

    # Claim cache rows for our batch.
    for context in context_batch:
        if not model.kv_manager.contains(context.cache_seq_id):
            model.kv_manager.external_claim([context.cache_seq_id])

    # Fetch kv inputs.
    kv_cache_inputs = model.kv_manager.fetch(context_batch)

    # Get Model inputs
    model_inputs = model.prepare_initial_token_inputs(
        context_batch=context_batch,
        # Flatten the KV cache inputs as expected by PipelineModel.execute.
        kv_cache_inputs=KVCacheInputsSequence(kv_cache_inputs=kv_cache_inputs),
    )

    model_outputs = model.execute(model_inputs)

    assert model_outputs.next_token_logits
    logits = model_outputs.next_token_logits.to_numpy()
    next_tokens = [req_logits.argmax(axis=-1) for req_logits in logits]

    has_eos = False
    for req_id, req_logits, next_token in zip(
        req_to_context_dict, logits, next_tokens
    ):
        update_values[req_id].append(
            {
                # We record the base next_token here.
                # If it deviates from the reference, we want to see that.
                "next_token": next_token,
                "next_token_logits": req_logits[next_token],
                "logits": req_logits,
            }
        )
        # Update the context for the next input.
        # If we have a reference, always select the reference's next token.
        if req_id in req_to_reference_dict:
            ref = req_to_reference_dict[req_id]["values"]
            ref_next_token = ref[0]["next_token"]
            next_token = ref_next_token

            # Drop token now that it is generated.
            req_to_reference_dict[req_id]["values"] = ref[1:]

        req_to_context_dict[req_id].update(int(next_token))
        if next_token == eos_token:
            has_eos = True
            break

    model.kv_manager.step(context_batch)
    return has_eos


def compare_values(
    actual,  # noqa: ANN001
    expected,  # noqa: ANN001
    *,
    rtol=1e-2,  # noqa: ANN001
    atol=1e-5,  # noqa: ANN001
    compare_fn=None,  # noqa: ANN001
) -> None:
    """Compares two dictionaries of values."""
    keys = expected[0].keys()
    if keys == {"prompt", "values"}:
        compare_text_generation(
            actual, expected, rtol=rtol, atol=atol, compare_fn=compare_fn
        )
    elif keys == {"prompt", "embeddings"}:
        compare_embeddings(
            actual, expected, rtol=rtol, atol=atol, compare_fn=compare_fn
        )
    else:
        raise ValueError(
            f"Unable to compare dictionaries with keys {keys}, does not match "
            "the expected keys of a text generation or embedding model."
        )


def compare_text_generation(
    actual,  # noqa: ANN001
    expected,  # noqa: ANN001
    *,
    rtol=1e-2,  # noqa: ANN001
    atol=1e-5,  # noqa: ANN001
    compare_fn=None,  # noqa: ANN001
) -> None:
    """Compares the values between two computed logits.

    The data structure of the actual/expected logits should be:
    [
        {"prompt": "prompt 1", "values": [{"key": value, ...}],
        {"prompt": "prompt 2", "values": [{"key": value, ...}],
        ...
    ]

    The "values" list contains the logits at each step for the prompt.

    The `actual` logits structure must be a subset of the expected logits.
    E.g. if the `expected` values contains logits for 10 steps, the `actual`
    values can contain any number of steps between 1-10.

    Args:
        actual: Data structure containing computed values.
        expected: Data structure containing reference values.
        rtol: The relative tolerance (used if `compare_fn` is not provided).
        atol: The absolute tolerance (used if `compare_fn` is not provided).
        compare_fn: A callable that takes the arguments
            (actual, expected, description) and raises an assertion error
            if the check fails.
    """
    expected_prompts = {x["prompt"]: x["values"] for x in expected}
    actual_prompts = {x["prompt"]: x["values"] for x in actual}

    diff = actual_prompts.keys() - expected_prompts.keys()
    if diff:
        raise ValueError(
            f"Golden values for prompts {diff} not found. Please re-run"
            " `gen_golden_values`."
        )

    for prompt, values in actual_prompts.items():
        expected_values = expected_prompts[prompt]
        actual_steps = len(values)
        expected_steps = len(expected_values)

        assert actual_steps <= expected_steps

        for step in range(actual_steps):
            inference_results = values[step]
            expected_results = expected_values[step]

            for key, value in inference_results.items():
                expected_value = expected_results[key]
                short = f"{prompt[:15]}..." if len(prompt) > 15 else prompt
                description = f"'{key}' on step={step} for the prompt='{short}'"
                if compare_fn:
                    compare_fn(value, expected_value, description)
                else:
                    np.testing.assert_allclose(
                        value,
                        expected_value,
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Got different values for {description}.",
                        verbose=True,
                    )


def compare_embeddings(
    actual,  # noqa: ANN001
    expected,  # noqa: ANN001
    *,
    rtol=1e-2,  # noqa: ANN001
    atol=1e-5,  # noqa: ANN001
    compare_fn=None,  # noqa: ANN001
) -> None:
    """Compares the values between two computed embeddings.

    The data structure of the actual/expected dictionaries should be:
    [
        {"prompt": "prompt 1", "embeddings": embeddings,
        {"prompt": "prompt 2", "embeddings": embeddings,
        ...
    ]

    Args:
        actual: Data structure containing computed values.
        expected: Data structure containing reference values.
        rtol: The relative tolerance (used if `compare_fn` is not provided).
        atol: The absolute tolerance (used if `compare_fn` is not provided).
        compare_fn: A callable that takes the arguments
            (actual, expected, description) and raises an assertion error
            if the check fails.
    """
    expected_prompts = {x["prompt"]: x["embeddings"] for x in expected}
    actual_prompts = {x["prompt"]: x["embeddings"] for x in actual}

    if expected_prompts.keys() < actual_prompts.keys():
        diff = actual_prompts.keys() - expected_prompts.keys()
        raise ValueError(
            f"Golden values for prompts {diff} not found. Please re-run"
            " `gen_golden_values`."
        )

    for prompt, embeddings in actual_prompts.items():
        expected_embeddings = expected_prompts[prompt]
        short = f"{prompt[:15]}..." if len(prompt) > 15 else prompt
        description = f"embeddings for prompt '{short}'"
        if compare_fn:
            compare_fn(embeddings, expected_embeddings, description)
        else:
            np.testing.assert_allclose(
                embeddings,
                expected_embeddings,
                rtol=rtol,
                atol=atol,
                err_msg=f"Got different {description}.",
                verbose=True,
            )
