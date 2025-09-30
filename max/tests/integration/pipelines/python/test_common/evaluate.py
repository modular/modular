# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for evaluating models and comparing the logits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypedDict

import numpy as np
from max import pipelines
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.interfaces import (
    LogitsProcessor,
    PipelineTokenizer,
    ProcessorInputs,
    RequestID,
    SamplingParams,
)
from transformers import PreTrainedTokenizerBase
from typing_extensions import NotRequired

from .test_data import MockTextGenerationRequest


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


NUM_STEPS = 10


def run_model(
    pipeline: pipelines.TextGenerationPipeline,
    tokenizer: PipelineTokenizer,
    requests: Sequence[MockTextGenerationRequest],
    num_steps: int = NUM_STEPS,
    print_outputs: bool = False,
    batch_size: int = 1,
    reference: list[ModelOutput] | None = None,
) -> list[dict[str, Any]]:
    """Runs the pipeline for N steps on each request provided."""
    assert batch_size >= 1
    assert hasattr(tokenizer, "delegate")
    hf_tokenizer = tokenizer.delegate
    assert isinstance(hf_tokenizer, PreTrainedTokenizerBase)

    ids = [RequestID() for _ in requests]
    prompts_by_id = {id: request.prompt for id, request in zip(ids, requests)}
    stored_logits = StoreLogits(ids, tokenizer)

    logits_processors: list[LogitsProcessor]
    if reference:
        reference_by_id = {
            id: reference for id, reference in zip(ids, reference)
        }
        replace_logits = ReplaceLogitsWithReference(
            pipeline._devices, reference_by_id
        )
        logits_processors = [stored_logits, replace_logits]
    else:
        logits_processors = [stored_logits]

    sampling_params = SamplingParams(
        top_k=1, max_new_tokens=num_steps, logits_processors=logits_processors
    )

    text_gen_requests = []
    for i, request in enumerate(requests):
        text_gen_requests.append(
            request.to_text_generation_request(ids[i], sampling_params)
        )

    for i in range(0, len(requests), batch_size):
        batch = text_gen_requests[i : i + batch_size]
        outputs = pipeline.generate(batch)
        if print_outputs:
            for j in range(len(batch)):
                request = requests[i + j]
                prompt = request.prompt
                print(
                    "Prompt:",
                    f"{prompt[:100]}...{prompt[-100:]}"
                    if len(prompt) > 200
                    else prompt,
                )
                print(
                    "Output:",
                    tokenizer.delegate.decode(outputs[j].tokens),
                )

    results: list[dict[str, Any]] = []
    for req_id, values in stored_logits.values.items():
        results.append({"prompt": prompts_by_id[req_id], "values": values})
    return results


class StoreLogits:
    def __init__(
        self, ids: Sequence[RequestID], tokenizer: PipelineTokenizer
    ) -> None:
        self.values: dict[RequestID, list[TokenInfo]] = {id: [] for id in ids}
        self.reached_eos = {id: False for id in ids}
        self.tokenizer = tokenizer

    def __call__(self, inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        context = inputs.context
        if self.reached_eos[context.request_id]:
            return
        next_token_logits = logits[-1, :].to_numpy().copy()
        next_token = next_token_logits.argmax(axis=-1)
        self.values[context.request_id].append(
            {
                # We record the base next_token here.
                # If it deviates from the reference, we want to see that.
                "next_token": next_token,
                "next_token_logits": next_token_logits[next_token],
                "logits": next_token_logits,
            }
        )
        if next_token == self.tokenizer.eos:
            self.reached_eos[context.request_id] = True


class ReplaceLogitsWithReference:
    def __init__(
        self,
        devices: Sequence[Device],
        reference_by_id: dict[RequestID, ModelOutput],
    ) -> None:
        self.reference_by_id = reference_by_id
        self.step_by_id = {id: 0 for id in reference_by_id}
        device_ref = DeviceRef.from_device(devices[0])

        def _replace_logits(
            logits: BufferValue, next_token: TensorValue
        ) -> None:
            logits[-1, next_token] = ops.constant(
                1e5, dtype=DType.float32, device=device_ref
            )

        replace_logits_graph = Graph(
            "replace_logits",
            _replace_logits,
            input_types=[
                BufferType(
                    DType.float32, ("seq_len", "vocab_size"), device_ref
                ),
                TensorType(DType.int64, (), DeviceRef.CPU()),
            ],
        )
        session = InferenceSession(devices=devices)
        self.replace_logits = session.load(replace_logits_graph)

    def __call__(self, inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        context = inputs.context
        # Assign the argmax of the reference to the logits.
        reference = self.reference_by_id[context.request_id]
        step = self.step_by_id[context.request_id]
        next_token = reference["values"][step]["next_token"]
        self.replace_logits(logits, next_token)
        self.step_by_id[context.request_id] += 1


def compare_values(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    compare_fn: Callable[[Any, Any, str], None] | None = None,
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
            "the expected keys of a text generation or embedding pipeline."
        )


def compare_text_generation(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    compare_fn: Callable[[Any, Any, str], None] | None = None,
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
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    compare_fn: Callable[[Any, Any, str], None] | None = None,
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
