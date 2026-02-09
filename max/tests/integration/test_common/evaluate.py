# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Utilities for evaluating models and comparing the logits."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict, TypeVar

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
    TextGenerationRequest,
)
from transformers import PreTrainedTokenizerBase
from typing_extensions import NotRequired

from .numerics import log_softmax
from .test_data import MockTextGenerationRequest


class TokenInfo(TypedDict, total=False):
    """Information about a token in the output."""

    next_token: int
    """The next token in the output."""
    next_token_logits: float
    """The logits for the next token (always present)."""
    logits: np.ndarray
    """The logits for the token (always present)."""
    next_token_logprobs: float
    """The logprobs for the next token (when generate_logprobs=True)."""
    logprobs: np.ndarray
    """The logprobs for the token (when generate_logprobs=True)."""


class ModelOutput(TypedDict):
    """The prompt and the output of a model run."""

    prompt: str
    """The prompt that was used to generate the output."""
    values: NotRequired[list[TokenInfo]]
    """Outputs from a text generation model."""
    embeddings: NotRequired[np.ndarray]
    """Outputs from a text embedding model."""


class ModelOutputView:
    """Convenience accessors for ModelOutput values."""

    def __init__(self, output: ModelOutput) -> None:
        self._output = output

    @property
    def prompt(self) -> str:
        return self._output["prompt"]

    @property
    def values(self) -> list[TokenInfo]:
        values = self._output.get("values")
        if values is None:
            raise ValueError("ModelOutput has no token values")
        return values

    @property
    def mode(self) -> str:
        if not self.values:
            return "logits"
        token_info = self.values[0]
        if "logits" in token_info:
            return "logits"
        if "logprobs" in token_info:
            return "logprobs"
        raise ValueError("ModelOutput values missing logits/logprobs")

    @property
    def logits(self) -> list[np.ndarray]:
        if self.mode == "logprobs":
            raise ValueError(
                "ModelOutput stores logprobs; logits are unavailable"
            )
        return [token["logits"] for token in self.values]

    @property
    def logprobs(self) -> list[np.ndarray]:
        if not self.values:
            return []
        if "logprobs" in self.values[0]:
            return [token["logprobs"] for token in self.values]
        if "logits" not in self.values[0]:
            raise ValueError("ModelOutput has no logits or logprobs")
        return [log_softmax(token["logits"]) for token in self.values]


NUM_STEPS = 10

T = TypeVar("T")


def _create_batches(
    requests: Sequence[T], batch_sizes: int | list[int] = 1
) -> list[Sequence[T]]:
    """Group requests into batches."""
    if isinstance(batch_sizes, list):
        if sum(batch_sizes) != len(requests):
            raise ValueError(
                "The sum of the batch sizes must be equal to the number of requests."
            )
    else:
        batch_sizes = [batch_sizes] * len(requests)
    batches = []

    start = 0
    for size in batch_sizes:
        batches.append(requests[start : start + size])
        start += size
    return batches


def _create_requests(
    ids: Sequence[RequestID],
    requests: Sequence[MockTextGenerationRequest],
    num_steps: int,
    reference_by_id: dict[RequestID, ModelOutput] | None = None,
    logits_processors: list[LogitsProcessor] | None = None,
) -> list[TextGenerationRequest]:
    """Create text generation requests.

    Will correctly set `max_new_tokens` if reference outputs are provided.
    """
    text_generation_requests = []
    for id, request in zip(ids, requests, strict=True):
        if reference_by_id:
            assert reference_by_id[id].values is not None
            max_new_tokens = len(reference_by_id[id]["values"])
        else:
            max_new_tokens = num_steps
        text_generation_requests.append(
            request.to_text_generation_request(
                id,
                SamplingParams(
                    ignore_eos=True,
                    top_k=1,
                    max_new_tokens=max_new_tokens,
                    logits_processors=logits_processors,
                ),
            )
        )
    return text_generation_requests


def run_model(
    pipeline: pipelines.TextGenerationPipelineInterface,
    tokenizer: PipelineTokenizer,
    requests: Sequence[MockTextGenerationRequest],
    num_steps: int = NUM_STEPS,
    print_outputs: bool = False,
    batch_size: int | list[int] = 1,
    reference: list[ModelOutput] | None = None,
    generate_logprobs: bool = False,
) -> list[dict[str, Any]]:
    """Runs the pipeline for N steps on each request provided."""
    assert hasattr(tokenizer, "delegate")
    hf_tokenizer = tokenizer.delegate
    assert isinstance(hf_tokenizer, PreTrainedTokenizerBase)

    ids = [RequestID() for _ in requests]
    prompts_by_id = {
        id: request.prompt for id, request in zip(ids, requests, strict=True)
    }
    stored_logits = StoreLogits(
        ids, tokenizer, generate_logprobs=generate_logprobs
    )

    logits_processors: list[LogitsProcessor]
    if reference:
        reference_by_id = {
            id: reference for id, reference in zip(ids, reference, strict=True)
        }
        replace_logits = ReplaceLogitsWithReference(
            pipeline._devices, reference_by_id
        )
        logits_processors = [stored_logits, replace_logits]
    else:
        logits_processors = [stored_logits]
        reference_by_id = None

    batched_requests = _create_batches(requests, batch_size)
    batched_ids = _create_batches(ids, batch_size)

    for ids_in_batch, requests_in_batch in zip(
        batched_ids, batched_requests, strict=True
    ):
        batch = _create_requests(
            ids_in_batch,
            requests_in_batch,
            num_steps,
            reference_by_id,
            logits_processors,
        )
        outputs = pipeline.generate(batch)
        if print_outputs:
            for j in range(len(batch)):
                request = requests_in_batch[j]
                prompt = request.prompt
                print(
                    "Prompt:",
                    f"{prompt[:100]}...{prompt[-100:]}"
                    if len(prompt) > 200
                    else prompt,
                )
                print(
                    "Output:",
                    tokenizer.delegate.decode(
                        outputs[j].tokens, skip_special_tokens=True
                    ),
                )

    results: list[dict[str, Any]] = []
    for req_id, values in stored_logits.values.items():
        results.append({"prompt": prompts_by_id[req_id], "values": values})
    return results


class StoreLogits:
    def __init__(
        self,
        ids: Sequence[RequestID],
        tokenizer: PipelineTokenizer,
        generate_logprobs: bool = False,
    ) -> None:
        self.values: dict[RequestID, list[TokenInfo]] = {id: [] for id in ids}
        self.tokenizer = tokenizer
        self.generate_logprobs = generate_logprobs

    def __call__(self, inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        context = inputs.context
        # Don't do anything if the context is already finished.
        # This code path is possible due to overlap scheduling possibly executing
        # one more forward pass than needed.
        if context.is_done:
            return
        next_token_logits = logits[-1, :].to_numpy().copy()
        next_token = next_token_logits.argmax(axis=-1)

        entry: TokenInfo = {
            # We record the base next_token here.
            # If it deviates from the reference, we want to see that.
            "next_token": next_token,
            "next_token_logits": next_token_logits[next_token],
            "logits": next_token_logits,
        }

        if self.generate_logprobs:
            next_token_logprobs_array = log_softmax(next_token_logits)
            entry["next_token_logprobs"] = float(
                next_token_logprobs_array[next_token]
            )
            entry["logprobs"] = next_token_logprobs_array

        self.values[context.request_id].append(entry)


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
        # Don't do anything if the context is already finished.
        # This code path is possible due to overlap scheduling possibly executing
        # one more forward pass than needed.
        if context.is_done:
            return
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
        short = f"{prompt[:15]}..." if len(prompt) > 15 else prompt

        for step in range(actual_steps):
            inference_results = values[step]
            expected_results = expected_values[step]

            inference_next_token = inference_results["next_token"]
            expected_next_token = expected_results["next_token"]
            if inference_next_token != expected_next_token:
                # Always use logits for comparison (logits are always present)
                inference_logits = inference_results["logits"]
                expected_logits = expected_results["logits"]
                print(
                    f"⚠️ Got mismatching next_token: {inference_next_token} !="
                    f" {expected_next_token} on step={step} for the prompt='{short}'"
                )
                print(
                    f"Logits for generated token {inference_next_token}: {inference_logits[inference_next_token]} (inference) vs {expected_logits[inference_next_token]} (reference)"
                )
                print(
                    f"Logits for expected token {expected_next_token}: {inference_logits[expected_next_token]} (inference) vs {expected_logits[expected_next_token]} (reference)"
                )

            for key, value in inference_results.items():
                expected_value = expected_results[key]
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
