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
"""Lightweight model output types and comparison utilities.

This module contains type definitions and comparison functions that do not
depend on heavy packages (transformers, torch, MAX runtime). It is safe to
import in contexts where fast startup is required, such as simple verification
tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
from typing_extensions import NotRequired


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


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log(softmax(x)) with numerical stability.

    Uses the identity log(softmax(x)) = x - log(sum(exp(x)))
    with max-subtraction to prevent overflow.
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    shifted_x = x - x_max
    exp_shifted = np.exp(shifted_x)
    with np.errstate(divide="ignore"):
        sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
        log_sum_exp = np.log(sum_exp)
    return shifted_x - log_sum_exp


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
        return [_log_softmax(token["logits"]) for token in self.values]


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
