# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 with full weights on GPU and compares it to previously generated
golden values.
"""

from functools import partial
from pathlib import Path

import pytest
from evaluate_llama import SupportedTestModels
from llama3.llama3 import load_llama3_and_kv_manager
from max.engine import InferenceSession
from max.pipelines import TextContext, TextTokenizer
from max.pipelines.interfaces import TokenGeneratorRequest
from test_common.distance_metrics import kl_divergence_verifier
from test_common.evaluate import (
    PROMPTS,
    compare_values,
    next_token_with_logits,
    run_model,
)
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path


@pytest.mark.parametrize(
    "model_name,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
def test_llama(
    model_name: str, encoding: str, testdata_directory: Path
) -> None:
    test_model = SupportedTestModels.get(model_name, encoding)
    config = test_model.build_config(max_length=512)
    tokenizer = TextTokenizer(config)
    session = InferenceSession(devices=[config.device])
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

    # Note that this threshold was set experimentally so that the test passes
    # with the existing Llama 3.1 implementation.
    compare_values(
        actual,
        expected_results,
        compare_fn=partial(kl_divergence_verifier, threshold=0.1),
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
    config = test_model.build_config(max_cache_batch_size=4)
    tokenizer = TextTokenizer(config)
    session = InferenceSession(devices=[config.device])
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
