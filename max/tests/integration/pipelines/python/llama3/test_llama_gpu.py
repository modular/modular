# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 with full weights on GPU and compares it to previously generated
golden values.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from evaluate_llama import SupportedTestModels
from llama3.llama3 import load_llama3_and_kv_manager
from llama3.llama3_token_gen import Llama3Tokenizer
from max.driver import CUDA
from max.engine import InferenceSession
from max.pipelines.interfaces import TokenGeneratorRequest
from test_common.evaluate import (
    PROMPTS,
    compare_values,
    next_token_with_logits,
    run_model,
)
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path

if TYPE_CHECKING:
    from dataprocessing import TextContext


def kl_divergence_verifier(
    predicted: np.ndarray, expected: np.ndarray, description: str
) -> None:
    """Verifies predicted vs. expected vectors using KL divergence.

    This is preferable to elementwise comparison for vector distances, because
    elementwise comparison doesn't take into account the magnitude of the
    vectors.
    Furthermore since we are comparing logits, use the KL divergence as a
    distance between expected and predicted probability distributions (over
    next token classification).
    """
    if isinstance(predicted, np.int64):
        # Assert that token ids generated match PyTorch exactly.
        assert predicted == expected, description
        return

    eps_bf16 = torch.finfo(torch.bfloat16).eps
    if isinstance(predicted, np.float32):
        # Treat the "next logits" element specially until it is removed.
        assert np.isclose(
            predicted, expected, rtol=2 * eps_bf16, atol=2 * eps_bf16
        )
        return

    logits_predicted = torch.from_numpy(predicted)
    logits_expected = torch.from_numpy(expected)

    # Convert logits to log-probabilities.
    log_probs_expected = F.log_softmax(logits_expected, dim=-1)
    log_probs_predicted = F.log_softmax(logits_predicted, dim=-1)

    # Compute the KL divergence.
    kl_divergence = F.kl_div(
        log_probs_predicted,
        log_probs_expected,
        reduction="sum",
        log_target=True,
    )

    # Note that this threshold was set experimentally so that the test passes
    # with the existing Llama 3.1 implementation.
    # Assert that the KL divergence between predicted and expected log
    # probabilities is below 0.01.
    threshold = 0.01
    assert kl_divergence < threshold, description


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
def test_llama(model: str, encoding: str, testdata_directory: Path) -> None:
    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config(max_length=512)
    tokenizer = Llama3Tokenizer(config)
    session = InferenceSession(device=config.device)
    model, _ = load_llama3_and_kv_manager(config, session)
    actual = run_model(
        model,
        tokenizer,
        prompts=PROMPTS[:1],
    )

    golden_data_path = find_runtime_path(
        "torch_llama3_1_bfloat16_golden.json",
        testdata_directory,
        bazel_dir=Path("torch_llama_golden"),
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    compare_values(
        actual,
        expected_results,
        compare_fn=kl_divergence_verifier,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
async def test_llama_ragged(model: str, encoding: str) -> None:
    prompt_a = PROMPTS[0]
    prompt_b = PROMPTS[1]

    stored_logits: dict[str, list[TextContext]] = {"A": [], "B": [], "C": []}

    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config()
    tokenizer = Llama3Tokenizer(config)
    session = InferenceSession(device=config.device)
    llama, _ = load_llama3_and_kv_manager(config, session)

    def request(prompt: str, idx: int) -> TokenGeneratorRequest:
        return TokenGeneratorRequest(
            id=str(idx), index=idx, prompt=prompt, model_name=model
        )

    # Send in A and B for context encoding.
    context_a = await tokenizer.new_context(request(prompt_a, idx=0))
    context_b = await tokenizer.new_context(request(prompt_b, idx=1))
    next_token_with_logits(
        llama, {"A": context_a, "B": context_b}, stored_logits
    )

    # Send in both B and C for token generation.
    next_token_with_logits(
        llama, {"A": context_a, "B": context_b}, stored_logits
    )
