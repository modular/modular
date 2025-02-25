# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for evaluating models and comparing the logits."""

import asyncio
import uuid
from typing import Any, Iterable, Optional

import numpy as np
import requests
from max.pipelines import PipelineModel
from max.pipelines.interfaces import PipelineTokenizer, TokenGeneratorRequest
from max.pipelines.kv_cache import KVCacheInputsSequence

NUM_STEPS = 10
PROMPTS = (
    """One of the most important aspects of performance benchmarking when it pertains to comparison of different implementations is making sure comparisons are fair. This is a place where most discussions occur, as deviation from best practices can make one’s performance claims easy to dismiss. For faster results of a given implementation (the Mojo implementation in our case) to be meaningful, the comparison needs to be apples-to-apples.
    * Make sure you use equivalent optimization flags across implementations; even though flags (like -O3 in C) that enable multiple optimizations at once cannot always be equivalent to another language’s -O3, make sure you don’t compare something like a debug build with an implementation that uses the fast optimization flag.
    * Make sure that if one implementation has auto-vectorization or automatic multithreading enabled the same applies to all implementations to be compared (unless for a given language one of these performs worse when turned-on, in which case one could keep the fastest implementation for comparison purposes).
    * Use the latest (or best) combination of compilers, libraries, etc. — an older compiler version (for example) may perform better for whatever reason; however it should be considered sufficient to test with the latest stable version. One can test with older or experimental versions if they are so inclined.
    * Use the same input file (if applicable) or same input data. Avoid random data generation that may stress different code paths.
    * Use the same algorithm (if applicable) across all your implementations.
    * Use equivalent error testing as it applies to different domains’ best practices (e.g., input sanitizing, corner case testing).
    * Remove any unnecessary I/O (e.g., writing to file/screen for debug purposes) and keep only what is practically necessary — make sure you do so in a manner that code is not optimized out (see #6)!
    * Try to apply the same level of manual optimization (within reason) — if you write multi-threaded/vectorized code in Mojo, you should try to compare it to an equivalent implementation of the other language. There is a case to be made here, however, if the other language does not have such capabilities or they are so difficult to use that implementing them is beyond what one can reasonably do. This can highlight the programmability aspect of Mojo (or one language against another more generally), but this fact should be listed so that people can take the performance claims under this light.""",
    "def is_prime(x):\n",
    "The meaning of life is ",
    """Translate the English text to Italian.
    Text: Sometimes, I've believed as many as six impossible things before breakfast.
    Translation:""",
)

# TODO: Improve the set of verification inputs for multi-model models.
PROMPTS_MULTI_MODAL = (
    "<|image|><|begin_of_text|>If I had to write a haiku for this one"
)

IMAGES_MULTI_MODAL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

PIXTRAL_PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"
PIXTRAL_IMG_URL = "https://picsum.photos/id/237/400/300"


def resolve_image_from_url(image_ref: str) -> bytes:
    return requests.get(image_ref).content


def run_model(
    model: PipelineModel,
    tokenizer: PipelineTokenizer,
    prompts: Iterable[str] = PROMPTS,
    images: Optional[Iterable[str]] = None,
    num_steps: int = NUM_STEPS,
) -> list[dict[str, Any]]:
    """Runs the model for N steps on each prompt provide."""
    return asyncio.run(
        run_model_async(
            model,
            tokenizer,
            prompts=prompts,
            images=images,
            num_steps=num_steps,
        )
    )


async def run_model_async(
    model: PipelineModel,
    tokenizer: PipelineTokenizer,
    prompts: Iterable[str] = PROMPTS,
    images: Optional[Iterable[str]] = None,
    num_steps: int = NUM_STEPS,
) -> list[dict[str, Any]]:
    """Runs the model for N steps on each prompt provide."""

    # Download images.
    downloaded_images: Optional[list[bytes]] = None
    if images:
        downloaded_images = []
        for url in images:
            downloaded_images.append(resolve_image_from_url(url))

    results = []
    # Evaluate prompts individually (not batched).
    # TODO: add batched version of run_model.
    for prompt in prompts:
        curr_req_id = str(uuid.uuid4())
        context = await tokenizer.new_context(
            TokenGeneratorRequest(
                id="",
                index=0,
                prompt=prompt,
                model_name="llama3",
                images=downloaded_images,
            )
        )
        values: dict[str, list[Any]] = {curr_req_id: []}
        for _ in range(num_steps):
            is_eos = next_token_with_logits(
                model, {curr_req_id: context}, values, tokenizer.eos
            )
            if is_eos:
                break
        results.append({"prompt": prompt, "values": values[curr_req_id]})

        model.kv_manager.release(context.cache_seq_id)

    return results


def next_token_with_logits(
    model: PipelineModel,
    req_to_context_dict: dict[str, Any],
    update_values: dict[str, list[Any]],
    eos_token: Optional[int] = None,
) -> bool:
    """Generates the next token and stores the logits.

    This method runs llama3.execute, stores the logits, and updates the context
    with the next token.

    Args:
        model: Model to execute.
        req_to_context_dict: Dictionary of request ids to Llama3Context.
        update_values: Dictionary of request ids to lists of next_token &
            logits. These lists are updated in this method.
        eos_token: Encoded end-of-sequence token used to signal the early stopping of token generation. If not provided, generation may continue past EOS token.

    Returns:
        bool: True if the token is an end-of-sentence token, otherwise False.
    """
    # Flatten our batch for consistent indexing.
    context_batch = list(req_to_context_dict.values())

    # Claim cache rows for our batch.
    for context in context_batch:
        if not model.kv_manager.contains(context.cache_seq_id):
            model.kv_manager.external_claim([context.cache_seq_id])

    # Get prompts for each seq_id in batch.
    seq_ids_and_prompts = {}
    for ctx in context_batch:
        prompt = ctx.next_tokens
        assert len(prompt) == ctx.active_length
        seq_ids_and_prompts[ctx.cache_seq_id] = prompt

    # Fetch kv inputs.
    kv_cache_inputs = model.kv_manager.fetch(seq_ids_and_prompts)

    # Get Model inputs
    model_inputs = model.prepare_initial_token_inputs(context_batch)

    model_outputs = model.execute(
        model_inputs,
        # Flatten the KV cache inputs as expected by PipelineModel.execute.
        kv_cache_inputs=KVCacheInputsSequence(kv_cache_inputs=kv_cache_inputs),
    )
    assert model_outputs.next_token_logits
    logits = model_outputs.next_token_logits.to_numpy()
    next_tokens = [req_logits.argmax(axis=-1) for req_logits in logits]

    seq_ids_and_new_tokens = {
        ctx.cache_seq_id: np.array([next_tokens[i]])
        for i, ctx in enumerate(context_batch)
    }
    model.kv_manager.step(seq_ids_and_new_tokens)

    for req_id, req_logits, next_token in zip(
        req_to_context_dict, logits, next_tokens
    ):
        update_values[req_id].append(
            {
                "next_token": next_token,
                "next_token_logits": req_logits[next_token],
                "logits": req_logits,
            }
        )
        # Update the context for the next input.
        req_to_context_dict[req_id].update(int(next_token))
        if next_token == eos_token:
            return True
    return False


def compare_values(actual, expected, *, rtol=1e-2, atol=1e-5, compare_fn=None):
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
    actual, expected, *, rtol=1e-2, atol=1e-5, compare_fn=None
):
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
    actual, expected, *, rtol=1e-2, atol=1e-5, compare_fn=None
):
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
